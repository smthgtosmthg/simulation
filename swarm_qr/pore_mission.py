"""La mission de Pore et al. sur notre simulateur : chef d'orchestre séparé du nôtre.

Ce programme ne partage avec `mission.py` que le socle physique — la scène, les pilotes
ArduPilot, le contrôleur de vol — pour que la comparaison porte sur la méthode et non sur la
qualité du pilotage. Tout le reste vient de `pore.py`, donc de l'article.

    pore_mission.py --drones 3 --budget 600 --seed 9033 --sortie experiments/13_pore/nominal
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description="la methode de Pore et al. (Symmetry 2026)")
parser.add_argument("--drones", type=int, default=3)
parser.add_argument("--seed", type=int, default=9033)
parser.add_argument("--budget", type=float, default=600.0, help="secondes de vol simulees")
parser.add_argument("--sortie", required=True)
parser.add_argument("--panne", default="", help="i:t — le drone i s'arrete a l'instant t")
parser.add_argument("--obstacle", default="", help="x,y,t — un bloc de 1x1x2 m apparait la-bas")
parser.add_argument("--codes-attendus", type=int, default=0, dest="codes_attendus")
parser.add_argument("--part-arret", type=float, default=0.95, dest="part_arret")
parser.add_argument("--grace", type=float, default=60.0)
parser.add_argument("--sans-progres", type=float, default=120.0, dest="sans_progres")
parser.add_argument("--video", action="store_true")
parser.add_argument("--cameras", type=int, default=2, choices=[2, 3, 5])
parser.add_argument("--headless", action="store_true", default=True)
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True, "extra_args": ["--/rtx/verifyDriverVersion/enabled=false"]})

import traceback  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from swarm_qr import control, pore  # noqa: E402
from swarm_qr.env import scene as scene_mod  # noqa: E402
from swarm_qr.env.config import CAMERAS  # noqa: E402
from swarm_qr.env.layout import make_layout  # noqa: E402
from swarm_qr.env.pilot import PHYS_DT, Clock, Pilot  # noqa: E402
from swarm_qr.experiments import _img  # noqa: E402

PHYS_PAR_CYCLE = 160          # 0,2 s simulée, comme notre mission : cinq observations par seconde
FLY_ALT = 1.6
V_TRANSIT = 1.0               # m/s, borne de vitesse de l'article (équation 4)
V_APPROCHE = 0.6
GAIN = 0.5
TOL = 0.35
TENUE_S = 2.0                 # s de « hover-and-scan » à chaque arrêt
PASSES_MAX = 3                # un arrêt injoignable est repris, mais pas indéfiniment
SORTIE = Path(args.sortie)


def _json_sur(o):
    if isinstance(o, np.ndarray):
        return [round(float(x), 3) for x in o.ravel()]
    if isinstance(o, (np.floating, np.integer)):
        return round(float(o), 4)
    raise TypeError(type(o))


def camera_frontale(drone_prim: str) -> Camera:
    """« A forward-facing fisheye camera » (§2.4). Notre frontale de série fait 160 × 120, trop
    peu pour décoder ; on en monte une à la définition de nos caméras de lecture, pour que la
    comparaison porte sur la méthode et non sur la qualité du capteur."""
    cam = Camera(prim_path=f"{drone_prim}/body/Cam_pore",
                 translation=np.array([CAMERAS.side_offset, 0.0, -CAMERAS.below]),
                 orientation=scene_mod._cam_orientation(0.0),
                 resolution=(CAMERAS.side_width, CAMERAS.side_height))
    cam.initialize()
    cam.set_focal_length(CAMERAS.focal_length)
    cam.set_horizontal_aperture(CAMERAS.horizontal_aperture)
    cam.set_clipping_range(CAMERAS.near, CAMERAS.far)
    return cam


class Drone:
    """Un drone de leur flotte : pilote, contrôleur de vol, caméra frontale, plan figé."""

    def __init__(self, i: int, scene, clock, plan: list[pore.Arret]):
        self.i = i
        self.pilot = Pilot(scene.world, i, clock)
        if not self.pilot.ready(FLY_ALT + 0.3 * i, lambda: float(scene.position(i)[2])):
            raise RuntimeError(f"le drone {i} n'a pas decolle")
        self.ctrl = control.Controleur(self.pilot, lambda: scene.position(i), lambda: scene.yaw(i),
                                       lambda: scene.velocity(i), v_approche=V_APPROCHE, tol=TOL,
                                       gain=GAIN, v_transit=V_TRANSIT)
        self.cam = camera_frontale(scene_mod.DRONE_PRIM.format(i))
        self.plan = plan
        self.k = 0
        self.arret: pore.Arret | None = None
        self.t_arrivee: float | None = None
        self.vivant = True
        self.attentes = 0
        self.confiance = None
        self.trajectoire: list[list[float]] = []
        self.inclinaisons: list[list[float]] = []
        self.decisions: list[dict] = []
        self.relectures: list[pore.Arret] = []
        self.passe = 1
        self.attente = None


def main() -> None:
    SORTIE.mkdir(parents=True, exist_ok=True)
    layout = make_layout(args.seed)
    scene = scene_mod.build(layout, with_sitl=True, n_drones=args.drones)
    scene.world.reset()
    scene.finalize()
    omni.timeline.get_timeline_interface().play()

    clock = Clock(scene.world)
    plans = pore.plan_zigzag(layout, args.drones)
    plans = pore.attribue(plans, [scene.position(i) for i in range(args.drones)])
    drones = [Drone(i, scene, clock, plans[i]) for i in range(args.drones)]
    K = np.asarray(drones[0].cam.get_intrinsics_matrix(), float)
    attendu = CAMERAS.side_width / (2.0 * math.tan(math.radians(CAMERAS.fov_deg) / 2.0))
    if abs(K[0, 0] - attendu) > 1.0:
        raise RuntimeError(f"camera frontale mal reglee : fx={K[0, 0]:.1f}, attendu {attendu:.1f}")
    print("plan de couverture : " + ", ".join(f"drone {d.i} {len(d.plan)} arrets" for d in drones))

    champ = pore.ChampDeRisque(layout)
    routeur = pore.Routeur(layout)
    inventaire = pore.Inventaire()
    tags = scene.tags
    codes_attendus = args.codes_attendus or len({t.tag_id for t in tags})

    panne = None
    if args.panne:
        d, t = args.panne.split(":")
        panne = (int(d), float(t))
    obstacle_prevu = None
    if args.obstacle:
        ox, oy, ot = (float(v) for v in args.obstacle.split(","))
        obstacle_prevu = {"x": ox, "y": oy, "t": ot, "pose": False}
    cams_video = {}
    index_video: list[dict] = []
    if args.video:
        jeux = {2: scene_mod.CAMERAS_VIDEO_2, 3: scene_mod.CAMERAS_VIDEO_3, 5: scene_mod.CAMERAS_VIDEO}
        cams_video = scene_mod.cameras_fixes(jeux[args.cameras])
        for nom in list(cams_video) + ["lecteur"]:
            (SORTIE / "video" / nom).mkdir(parents=True, exist_ok=True)

    journal = {"methode": "pore2026", "seed": args.seed, "drones": args.drones, "budget_s": args.budget,
               "arret": {"codes_attendus": codes_attendus, "part": args.part_arret,
                         "grace_s": args.grace, "sans_progres_s": args.sans_progres},
               "panne": args.panne, "evenements": [], "codes_par_t": [], "instantanes": []}
    mur0 = time.monotonic()
    fin = None
    t_dernier_code, t_part_atteinte = 0.0, None
    n_cycle = 0
    cycles_ms: list[float] = []

    def evenement(genre, **kw):
        journal["evenements"].append({"t": round(clock.t, 1), "genre": genre, **kw})

    def tient_sur_place(d: Drone) -> None:
        """À vitesse nulle l'autopilote dérive : on tient la position activement."""
        d.ctrl.assigne(control.Consigne(scene.position(d.i).copy(), scene.yaw(d.i)))

    # ---------------------------------------------------------------- un arrêt à la fois

    def prochain_arret(d: Drone) -> pore.Arret | None:
        """Le plan est figé : on prend l'arrêt suivant. S'il est dans le risque — un obstacle
        apparu, un coéquipier — on le déplace au minimum (§2.3 point 5) ; s'il n'y a pas
        d'issue, on le garde pour un second passage et on avance."""
        while d.k < len(d.plan) or d.relectures:
            if d.k >= len(d.plan):
                if d.passe >= PASSES_MAX:
                    return None
                d.plan, d.relectures, d.k = d.relectures, [], 0
                d.passe += 1
                evenement("second_passage", drone=d.i, passe=d.passe, arrets=len(d.plan))
                continue
            a = d.plan[d.k]
            d.k += 1
            if champ.sur(a.position):
                return a
            q = pore.deflexion(a.position, champ)
            if q is not None:
                evenement("deflexion", drone=d.i, de=a.position, vers=q)
                return pore.Arret(q, a.cap, a.face, a.etage, a.vise)
            d.relectures.append(a)
        return None

    def envoie(d: Drone, a: pore.Arret) -> None:
        points = routeur.chemin(scene.position(d.i), a.position)
        d.ctrl.assigne(control.Consigne(a.position.copy(), a.cap), waypoints=points)
        d.arret, d.t_arrivee = a, None
        d.decisions.append({"t": round(clock.t, 1), "face": a.face, "etage": a.etage,
                            "position": a.position, "points": len(points)})

    def lit(d: Drone) -> None:
        """Le flux de la caméra frontale est décodé « au sol » : gris, seuillage adaptatif,
        cv2.QRCodeDetector, puis dédoublonnage et score de confiance."""
        img = d.cam.get_rgb()
        if img is None or getattr(img, "ndim", 0) != 3 or not img.size:
            return
        bgr = _img.to_bgr(img)
        lus = pore.decode(bgr)
        d.confiance = max((c for _, c in lus), default=None)
        for code, conf in lus:
            if code not in {t.tag_id for t in tags}:
                continue                                   # le décor de l'entrepôt porte d'autres codes
            neuf = inventaire.recoit(code, d.i, scene.position(d.i), clock.t, conf, arret=d.arret)
            if neuf:
                evenement("code", drone=d.i, code=code, confiance=round(conf, 3))

    def enregistre_video() -> None:
        n = len(index_video)
        for nom, cam in cams_video.items():
            img = cam.get_rgb()
            if img is not None and getattr(img, "ndim", 0) == 3 and img.size:
                cv2.imwrite(str(SORTIE / "video" / nom / f"{n:05d}.jpg"), _img.to_bgr(img),
                            [cv2.IMWRITE_JPEG_QUALITY, 80])
        lecteur = next((d for d in drones if d.vivant and d.arret is not None), None)
        if lecteur is not None:
            img = lecteur.cam.get_rgb()
            if img is not None and getattr(img, "ndim", 0) == 3 and img.size:
                cv2.imwrite(str(SORTIE / "video" / "lecteur" / f"{n:05d}.jpg"),
                            cv2.resize(_img.to_bgr(img), (480, 360)), [cv2.IMWRITE_JPEG_QUALITY, 80])
        index_video.append({"n": n, "t": round(clock.t, 1), "codes": len(inventaire.codes),
                            "lecteur": lecteur.i if lecteur else None,
                            "drones": [{"i": d.i, "vivant": d.vivant} for d in drones]})

    # ---------------------------------------------------------------- la boucle
    try:
        while clock.t < args.budget and fin is None:
            t0 = time.perf_counter()
            n_cycle += 1
            for _ in range(PHYS_PAR_CYCLE - 1):
                scene.world.step(render=False)
            scene.world.step(render=True)
            clock.t += PHYS_PAR_CYCLE * PHYS_DT

            if panne and panne[1] <= clock.t:
                d = drones[panne[0]]
                if d.vivant:
                    d.vivant = False
                    d.arret = None
                    tient_sur_place(d)
                    evenement("panne", drone=d.i)
                    print(f"  t={clock.t:6.1f} s  PANNE du drone {d.i}")
            if obstacle_prevu and not obstacle_prevu["pose"] and clock.t >= obstacle_prevu["t"]:
                emprise = scene_mod.ajoute_obstacle("bloc_1", (obstacle_prevu["x"], obstacle_prevu["y"]))
                obstacle_prevu["pose"] = True
                journal["obstacle"] = {**emprise, "t": round(clock.t, 1)}
                evenement("obstacle", t_apparition=round(clock.t, 1), x=obstacle_prevu["x"], y=obstacle_prevu["y"])
                print(f"  t={clock.t:6.1f} s  OBSTACLE pose en ({obstacle_prevu['x']}, {obstacle_prevu['y']})")

            vivants = [d for d in drones if d.vivant]
            for d in vivants:
                pos = scene.position(d.i)
                d.trajectoire.append([round(clock.t, 2), *[round(float(x), 3) for x in pos]])
                haut = Rotation.from_quat(scene.drones[d.i].state.attitude).apply([0.0, 0.0, 1.0])
                d.inclinaisons.append([round(clock.t, 2),
                                       round(float(np.degrees(np.arccos(np.clip(haut[2], -1.0, 1.0)))), 1)])
                inclinaison = d.inclinaisons[-1][1]
                if clock.t > 120.0 and (pos[2] < 0.3 or inclinaison > 70.0):
                    d.vivant = False
                    d.arret = None
                    evenement("chute", drone=d.i, position=[round(float(v), 2) for v in pos])
                    print(f"  t={clock.t:6.1f} s  CHUTE du drone {d.i}")
                    continue
                dirs, portees = scene.lidar(d.i)
                champ.integre_lidar(pos, dirs, portees, t=clock.t)
                lit(d)

            for d in vivants:
                if not d.vivant:
                    continue
                autres = [(o.i, scene.position(o.i), scene.velocity(o.i)) for o in vivants if o.i != d.i]
                champ.annonce_coequipiers([p for _, p, _ in autres])
                if pore.cede_le_passage(d.i, scene.position(d.i), scene.velocity(d.i), autres):
                    if d.attente is None and d.arret is not None:
                        d.attente = (d.ctrl.consigne, list(d.ctrl.points[d.ctrl.i_point:]))
                        tient_sur_place(d)
                        d.attentes += 1
                    d.ctrl.tick()
                    continue
                if d.attente is not None:
                    consigne, points = d.attente
                    d.attente = None
                    d.ctrl.assigne(consigne, waypoints=points)
                d.ctrl.v_transit = pore.vitesse_selon_confiance(V_TRANSIT, d.confiance)
                phase = d.ctrl.tick()
                if d.arret is None:
                    a = prochain_arret(d)
                    if a is not None:
                        envoie(d, a)
                    continue
                if phase is control.Phase.ATTEINT:
                    if d.t_arrivee is None:
                        d.t_arrivee = clock.t
                    if clock.t - d.t_arrivee < TENUE_S:
                        continue
                    d.decisions[-1].update({"fin": round(clock.t, 1), "phase": "atteint"})
                    d.arret = None
                elif phase is control.Phase.ABANDON:
                    if d.arret is not None:
                        d.relectures.append(d.arret)
                    d.decisions[-1].update({"fin": round(clock.t, 1), "phase": "abandon",
                                            "raison": d.ctrl.bilan.raison if d.ctrl.bilan else ""})
                    d.arret = None

            lus = len(inventaire.codes)
            if journal["codes_par_t"] and lus > journal["codes_par_t"][-1][1]:
                t_dernier_code = clock.t
            journal["codes_par_t"].append([round(clock.t, 1), lus])
            if args.video:
                enregistre_video()
            if args.part_arret > 0 and lus >= args.part_arret * codes_attendus:
                if t_part_atteinte is None:
                    t_part_atteinte = clock.t
                    evenement("part_atteinte", codes=lus, attendus=codes_attendus)
                elif clock.t - t_part_atteinte >= args.grace:
                    fin = f"inventaire a {lus / codes_attendus:.0%} et grace ecoulee"
            if args.sans_progres > 0 and lus > 0 and clock.t - t_dernier_code >= args.sans_progres:
                fin = f"sans code nouveau depuis {args.sans_progres:.0f} s"
            if not [d for d in drones if d.vivant]:
                fin = "aucun drone vivant"
            elif all(d.arret is None and d.k >= len(d.plan) and not d.relectures for d in drones if d.vivant):
                fin = "plan termine"
            cycles_ms.append((time.perf_counter() - t0) * 1000)
            if n_cycle % 150 == 0:
                etat = " ".join(f"d{d.i}:{'panne' if not d.vivant else (d.arret.face.split(':')[-1] if d.arret else 'libre')}"
                                for d in drones)
                print(f"  t={clock.t:6.1f} s  {lus:3d} codes  {etat}  ({(time.monotonic() - mur0) / 60:.0f} min)",
                      flush=True)
    except Exception:
        traceback.print_exc()
        fin = fin or "erreur"
    finally:
        fin = fin or "budget epuise"
        cm = np.array(cycles_ms) if cycles_ms else np.zeros(1)
        journal.update({
            "fin": fin, "t_sim_s": round(clock.t, 1), "mur_min": round((time.monotonic() - mur0) / 60, 1),
            "inventaire": inventaire.bilan(),
            "cycles": {"n": int(len(cm)), "mediane_ms": round(float(np.median(cm)), 1),
                       "max_ms": round(float(cm.max()), 1)},
            "agents": [{"i": d.i, "vivant": d.vivant, "decisions": d.decisions, "attentes": d.attentes,
                        "arrets_servis": sum(1 for x in d.decisions if x.get("phase") == "atteint"),
                        "arrets_prevus": len(plans[d.i]), "passes": d.passe,
                        "arrets_non_servis": len(d.plan) - d.k + len(d.relectures),
                        "trajectoire": d.trajectoire, "inclinaisons": d.inclinaisons} for d in drones],
            "verite": [{"code": t.tag_id, "position": list(map(float, t.position))} for t in tags],
            "codes": sorted(inventaire.codes),
            "lectures": [{"code": e.code, "drone": e.drone, "t": round(e.t, 1),
                          "confiance": round(e.confiance, 3)} for e in inventaire.evenements],
        })
        (SORTIE / "mission.json").write_text(json.dumps(journal, default=_json_sur))
        if args.video:
            (SORTIE / "video" / "index.json").write_text(json.dumps(
                {"pas_s": 0.2, "codes_attendus": codes_attendus, "images": index_video}))
        b = inventaire.bilan()
        print(f"fin : {fin} a t={clock.t:.0f} s ; {len(inventaire.codes)}/{codes_attendus} codes, "
              f"{b['doublons']} doublons, confiance moyenne {b['confiance_moyenne']} ; "
              f"{(time.monotonic() - mur0) / 60:.0f} min de calcul", flush=True)
        for d in drones:
            print(f"  drone {d.i} : {sum(1 for x in d.decisions if x.get('phase') == 'atteint')} arrets servis "
                  f"sur {len(plans[d.i])} prevus, {d.attentes} attentes de priorite")
        print("PORE MISSION FINIE")
        simulation_app.close()


if __name__ == "__main__":
    main()
