"""
Mesure 03 — Combien d'images par seconde le simulateur peut-il produire ?

C'est la question qui décide pour Dreamer. La mesure 02 a montré qu'à `train_ratio` 64,
l'apprentissage consomme 54 pas d'environnement par seconde. Le simulateur doit suivre,
avec 2 caméras par drone.

Isaac Sim ne supporte pas de reconstruire une scène dans le même processus : on mesure
donc UNE configuration par exécution, et `run_all.sh` boucle dessus.

Usage :
    PYTHONUNBUFFERED=1 ~/isaac5_env/bin/python experiments/03_render_bench/render_bench.py \
        --envs 8 --res 64 --kit_args="--/rtx/verifyDriverVersion/enabled=false"
    bash experiments/03_render_bench/run_all.sh       # le balayage complet

Le drapeau `verifyDriverVersion` évite l'avertissement de pilote : il est connu sur cette
machine et sans effet sur le rendu (même réglage que `calibrate_gate.py`).
"""

import argparse
import csv
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--envs", type=int, default=8)
parser.add_argument("--res", type=int, default=64)
parser.add_argument("--cams", type=int, default=2)
parser.add_argument("--steps", type=int, default=60)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import TiledCamera, TiledCameraCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

OUT = Path(__file__).parent
CSV = OUT / "resultats.csv"

CAM_COMMON = dict(
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=12.0, focus_distance=2.0, horizontal_aperture=13.86,
        clipping_range=(0.1, 30.0),
    ),
    width=args.res, height=args.res,
)


def _rack(name, x, color):
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 4.0, 3.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(x, 0.0, 1.5)),
    )


@configclass
class BenchSceneCfg(InteractiveSceneCfg):
    """Sol, lumière, deux rangées de racks à regarder, caméras latérales."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    dome = AssetBaseCfg(prim_path="/World/Light",
                        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9)))
    rack_left = _rack("RackL", -1.5, (0.70, 0.55, 0.35))
    rack_right = _rack("RackR", 1.5, (0.65, 0.50, 0.30))
    cam_left = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/CamL",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 1.5), rot=(0.5, -0.5, 0.5, 0.5),
                                        convention="ros"),
        **CAM_COMMON,
    )


def main():
    cfg = BenchSceneCfg(num_envs=args.envs, env_spacing=8.0, replicate_physics=True)
    if args.cams >= 2:
        cfg.cam_right = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/CamR",
            offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 1.5), rot=(0.5, 0.5, -0.5, 0.5),
                                            convention="ros"),
            **CAM_COMMON,
        )

    def mark(msg):
        print(f"[ÉTAPE] {time.strftime('%H:%M:%S')} {msg}", flush=True)

    mark("création du contexte de simulation")
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1 / 120, device="cuda:0", render_interval=1)
    )
    mark("construction de la scène")
    scene = InteractiveScene(cfg)
    mark("sim.reset()")
    sim.reset()

    cams = [s for s in scene.sensors.values() if isinstance(s, TiledCamera)]
    dt = sim.get_physics_dt()

    mark(f"chauffe : {len(cams)} caméra(s) — le 1er rendu compile les shaders, ça peut être long")
    for i in range(20):
        sim.step()
        scene.update(dt)
        if i in (0, 1, 4, 9):
            mark(f"  pas de chauffe {i + 1}/20 fait")

    # vérification que le rendu produit vraiment quelque chose
    rgb = cams[0].data.output["rgb"]
    non_noir = float((rgb > 8).float().mean())
    print(f"\n[VÉRIF] image {tuple(rgb.shape)} | pixels non noirs : {non_noir:.1%} "
          f"| valeur moyenne : {float(rgb.float().mean()):.1f}")
    if non_noir < 0.02:
        print("[VÉRIF] ATTENTION : l'image est quasiment noire — le rendu ne marche pas.")

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        sim.step()
        scene.update(dt)
        for c in cams:
            _ = c.data.output["rgb"]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    env_sps = args.envs * args.steps / elapsed
    row = dict(envs=args.envs, res=args.res, cams=len(cams),
               env_steps_per_s=round(env_sps, 1),
               images_per_s=round(env_sps * len(cams), 1),
               vram_gb=round(peak, 2), render_ok=round(non_noir, 4))

    print(f"[RESULTAT] {args.envs} envs | {args.res}px | {len(cams)} cams "
          f"| {env_sps:.1f} pas env/s | {env_sps * len(cams):.1f} images/s | {peak:.2f} Go")

    new = not CSV.exists()
    with open(CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)

    simulation_app.close()


if __name__ == "__main__":
    main()
