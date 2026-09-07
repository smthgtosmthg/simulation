"""Banc du contrôleur — étape 3.

  banc.py --mode freinage           trois lois d'approche sur le même trajet, trajectoires enregistrées
  banc.py --mode poses --poses 100  poses tirées devant les racks, transit + approche + lecture du QR
  banc.py --mode essaim             trois drones décollent et volent en même temps

Commande : DISPLAY=:1 PYTHONUNBUFFERED=1 timeout -s KILL 4h ~/isaac5_env/bin/python banc.py --mode ...

Le banc vérifie sa chaîne avant de mesurer (calibration de la caméra), enregistre la position
vraie du drone à chaque pas, et s'arrête si un drone ne décolle pas.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("freinage", "poses", "essaim"), default="freinage")
parser.add_argument("--seed", type=int, default=9033)
parser.add_argument("--poses", type=int, default=100)
parser.add_argument("--graine-tirage", type=int, default=20260906)
parser.add_argument("--v-transit", type=float, default=None)
parser.add_argument("--nom", default=None)
args, _ = parser.parse_known_args()

NOM = args.nom or args.mode

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp(
    {"headless": True, "extra_args": ["--/rtx/verifyDriverVersion/enabled=false"]}
)

import traceback  # noqa: E402

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402

from swarm_qr import control  # noqa: E402
from swarm_qr import perception as P  # noqa: E402
from swarm_qr.env import scene as scene_mod  # noqa: E402
from swarm_qr.env.config import CAMERAS, INTERIOR, RACKS  # noqa: E402
from swarm_qr.env.layout import make_layout  # noqa: E402
from swarm_qr.env.pilot import Clock, Pilot  # noqa: E402
from swarm_qr.experiments import _img  # noqa: E402
from swarm_qr.experiments._vol import ALT_TRANSIT_MIN, allee, chemin  # noqa: E402

FLY_ALT = 1.6
DEGAGEMENT_MIN = 0.50          # hélices 0,26 m + oscillation mesurée 0,21 m + marge
MARGE_MUR = 0.8
Z_MIN, Z_MAX = 1.2, 5.5
D_POSE = (1.5, 3.2)            # distance caméra-panneau tirée pour les poses
ALPHA_POSE = 40.0              # degrés de biais au plus
D_APPARENTE_MAX = 3.8          # reste dans la zone de lecture mesurée à l'étape 2
FENETRE_FREINAGE = 20.0        # s simulées enregistrées par essai
REPETITIONS = 3
RECUL_FREINAGE = 4.0           # m entre le départ et la cible, le long de l'allée
COUCHE_ESSAIM = 0.7            # m d'écart d'altitude entre drones en transit


# ---------------------------------------------------------------- géométrie

def cap_vers_panneau(cible_cam, tag) -> float:
    """Le cap qui met le panneau dans l'axe de la caméra gauche."""
    v = np.array(tag.position, float) - np.array(cible_cam, float)
    return math.atan2(v[1], v[0]) - math.pi / 2.0


def consigne_drone(cible_cam, psi) -> np.ndarray:
    """La caméra gauche est décalée du corps : viser avec elle, pas avec le corps."""
    lateral = np.array([-math.sin(psi), math.cos(psi), 0.0]) * CAMERAS.side_offset
    return np.array(cible_cam, float) - lateral + np.array([0.0, 0.0, CAMERAS.below])


def degagement(p, layout) -> float:
    d = 99.0
    for r in layout.racks:
        y0, y1 = r.y_bounds
        if y0 < p[1] < y1:
            d = min(d, abs(p[0] - r.x) - RACKS.depth / 2.0)
    return d


def dans_l_entrepot(p) -> bool:
    return (INTERIOR.x_min + MARGE_MUR < p[0] < INTERIOR.x_max - MARGE_MUR
            and INTERIOR.y_min + MARGE_MUR < p[1] < INTERIOR.y_max - MARGE_MUR
            and Z_MIN < p[2] < Z_MAX)


def pose_devant(rng, tag, layout):
    """Une pose de lecture tirée devant un panneau ; None si elle n'est pas volable."""
    D = rng.uniform(*D_POSE)
    a = math.radians(rng.uniform(-ALPHA_POSE, ALPHA_POSE))
    if D / math.cos(a) > D_APPARENTE_MAX:
        return None
    n = np.array(tag.normal, float)
    t = np.cross([0.0, 0.0, 1.0], n)
    t /= max(np.linalg.norm(t), 1e-9)
    cible_cam = np.array(tag.position, float) + D * (math.cos(a) * n + math.sin(a) * t)
    psi = cap_vers_panneau(cible_cam, tag)
    cible = consigne_drone(cible_cam, psi)
    if degagement(cible, layout) < DEGAGEMENT_MIN or not dans_l_entrepot(cible):
        return None
    return cible, psi, D, math.degrees(a)


def panneaux_lisibles(scene):
    return [t for t in scene.tags if abs(t.normal[0]) > 0.9
            and Z_MIN < t.position[2] + CAMERAS.below < Z_MAX]


def verifie_calibration(cam) -> np.ndarray:
    K = np.asarray(cam.get_intrinsics_matrix(), dtype=float)
    fx_attendu = CAMERAS.side_width / (2.0 * math.tan(math.radians(CAMERAS.fov_deg) / 2.0))
    if abs(K[0, 0] - fx_attendu) > 1.0 or abs(K[0, 0] - K[1, 1]) > 0.1:
        raise RuntimeError(f"calibration incoherente : fx={K[0,0]:.1f}, attendu {fx_attendu:.1f}")
    print(f"[VERIF] calibration : fx = fy = {K[0,0]:.1f} (attendu {fx_attendu:.1f})")
    return K


# ---------------------------------------------------------------- drones

def accesseurs(scene, i):
    return (lambda: scene.position(i)), (lambda: scene.yaw(i)), (lambda: scene.velocity(i))


def demarre(scene, clock, i) -> Pilot:
    pilot = Pilot(scene.world, i, clock)
    if not pilot.ready(FLY_ALT, lambda: float(scene.position(i)[2])):
        raise RuntimeError(f"le drone {i} n'a pas decolle")
    return pilot


def controleur(scene, pilot, i, **kw) -> control.Controleur:
    get_pos, get_yaw, get_vel = accesseurs(scene, i)
    if args.v_transit is not None:
        kw.setdefault("v_transit", args.v_transit)
    return control.Controleur(pilot, get_pos, get_yaw, get_vel, **kw)


def lecture(scene, K, tag, i=0) -> dict:
    img = _img.to_bgr(scene.capture("left", i))
    L = P.lire(img, K, tag.size, cible=tag.tag_id, decodeur="zxing")
    cam_pos = np.asarray(scene.cameras[i]["left"].get_world_pose()[0], float)
    D_vraie = float(np.linalg.norm(cam_pos - np.array(tag.position, float)))
    return {"lu": L.etat is P.Etat.LU, "D_vraie": round(D_vraie, 3),
            "D_vue": round(float(L.distance), 3) if L.position_fiable else None}


def monte(n_drones: int):
    layout = make_layout(args.seed)
    scene = scene_mod.build(layout, with_sitl=True, n_drones=n_drones)
    scene.world.reset()
    scene.finalize()
    omni.timeline.get_timeline_interface().play()
    K = verifie_calibration(scene.cameras[0]["left"])
    return layout, scene, K


class Trace:
    """Position vraie, vitesse et phase à chaque pas."""

    def __init__(self):
        self.lignes = []

    def __call__(self, ctrl):
        p = ctrl.get_pos()
        v = ctrl.get_vel()
        self.lignes.append([round(ctrl.pilot.sim_clock, 3), *[round(float(x), 4) for x in p],
                            *[round(float(x), 4) for x in v], ctrl.phase.value,
                            round(float(ctrl.get_yaw()), 4)])


# ---------------------------------------------------------------- freinage

def mode_freinage() -> None:
    layout, scene, K = monte(1)
    clock = Clock(scene.world)
    pilot = demarre(scene, clock, 0)
    tag = sorted([t for t in panneaux_lisibles(scene) if t.normal[0] > 0.9],
                 key=lambda t: abs(t.position[2] - FLY_ALT))[0]
    n = np.array(tag.normal, float)
    cible_cam = np.array(tag.position, float) + 2.0 * n
    psi = cap_vers_panneau(cible_cam, tag)
    cible = consigne_drone(cible_cam, psi)
    sens = 1.0 if cible[1] - RECUL_FREINAGE > INTERIOR.y_min + MARGE_MUR else -1.0
    depart = cible + np.array([0.0, -sens * RECUL_FREINAGE, 0.0])
    if degagement(depart, layout) < DEGAGEMENT_MIN or not dans_l_entrepot(depart):
        raise RuntimeError("pas de place pour le trajet de freinage")
    print(f"cible {np.round(cible, 2)}  depart {np.round(depart, 2)}  cap {math.degrees(psi):.0f} deg")

    mise_en_place = controleur(scene, pilot, 0)
    mise_en_place.assigne(control.Consigne(depart, psi), chemin(layout, scene.position(0), depart))
    b = control.rejoindre(mise_en_place)
    if b.phase is not control.Phase.ATTEINT:
        raise RuntimeError(f"depart non atteint : {b.raison}")

    vitesse_defaut = pilot.get_param("WPNAV_SPEED")
    print(f"WPNAV_SPEED par defaut : {vitesse_defaut:g} cm/s")
    runs = []
    for loi in control.LOIS:
        # même plafond pour toutes les lois : l'autopilote ne doit pas gagner en allant plus vite
        pilot.set_param("WPNAV_SPEED", 100.0 * control.V_APPROCHE if loi == "autopilote"
                        else vitesse_defaut)
        for rep in range(REPETITIONS):
            mise_en_place.assigne(control.Consigne(depart, psi))
            control.rejoindre(mise_en_place)
            pilot.pump(2.0)
            ctrl = controleur(scene, pilot, 0, loi=loi)
            ctrl.assigne(control.Consigne(cible, psi), budget_s=FENETRE_FREINAGE + 5.0)
            trace = Trace()
            t0 = clock.t
            while clock.t - t0 < FENETRE_FREINAGE:
                ctrl.tick()
                trace(ctrl)
                pilot.pump(control.DT_TICK)
            bilan = ctrl.bilan.as_dict() if ctrl.bilan else {"phase": ctrl.phase.value}
            runs.append({"loi": loi, "rep": rep, "traj": trace.lignes, "bilan": bilan})
            print(f"  {loi:14s} essai {rep}: phase finale {ctrl.phase.value:9s}  "
                  f"err {float(np.linalg.norm(cible - scene.position(0))):.2f} m")
    pilot.set_param("WPNAV_SPEED", vitesse_defaut)
    pilot.pump(0.5)
    (HERE / f"{NOM}.json").write_text(json.dumps({
        "seed": args.seed, "tol": control.TOL, "v_approche": control.V_APPROCHE,
        "wpnav_speed_defaut_cms": vitesse_defaut,
        "gain": control.GAIN, "dt": control.DT_TICK, "cible": cible.tolist(),
        "depart": depart.tolist(), "psi": psi, "runs": runs}))
    print(f"{len(runs)} essais dans {NOM}.json")


# ---------------------------------------------------------------- poses

def mode_poses() -> None:
    layout, scene, K = monte(1)
    clock = Clock(scene.world)
    pilot = demarre(scene, clock, 0)
    ctrl = controleur(scene, pilot, 0)
    rng = random.Random(args.graine_tirage)
    panneaux = panneaux_lisibles(scene)
    sortie = HERE / f"{NOM}.jsonl"
    sortie.write_text("")
    bilans = []
    k = 0
    while k < args.poses:
        tag = rng.choice(panneaux)
        tirage = pose_devant(rng, tag, layout)
        if tirage is None:
            continue
        cible, psi, D, alpha = tirage
        depart = scene.position(0)
        points = chemin(layout, depart, cible)
        ctrl.assigne(control.Consigne(cible, psi), points)
        trace = Trace()
        mur0 = time.monotonic()
        bilan = control.rejoindre(ctrl, on_tick=trace)
        mur = time.monotonic() - mur0
        lu = lecture(scene, K, tag) if bilan.phase is control.Phase.ATTEINT else None
        ligne = {"indice": k, "tag": tag.tag_id, "D": round(D, 3), "alpha": round(alpha, 1),
                 "cible": [round(float(x), 3) for x in cible], "psi": round(psi, 4),
                 "depart": [round(float(x), 3) for x in depart],
                 "points": [[round(float(x), 3) for x in w] for w in points],
                 "allee_depart": allee(layout, depart[0]), "allee_cible": allee(layout, cible[0]),
                 "bilan": bilan.as_dict(), "lecture": lu, "t_mur": round(mur, 1),
                 "debit": round(bilan.t_total / max(mur, 1e-6), 3), "traj": trace.lignes}
        with sortie.open("a") as f:
            f.write(json.dumps(ligne) + "\n")
        bilans.append(bilan)
        etat = bilan.phase.value if bilan.phase is control.Phase.ATTEINT else f"{bilan.phase.value}:{bilan.raison}"
        print(f"  pose {k:3d}  {len(points)} pts  {bilan.longueur:5.1f} m  {etat:15s} "
              f"{bilan.t_total:5.1f} s  err {bilan.err_finale:.2f} m  "
              f"{'lu' if lu and lu['lu'] else 'non lu' if lu else '-'}")
        k += 1
    pilot.hold(0.0)
    pilot.pump(0.5)
    (HERE / f"meta_{NOM}.json").write_text(json.dumps({
        "seed": args.seed, "poses": args.poses, "v_transit": ctrl.v_transit,
        "v_approche": ctrl.v_approche, "gain": ctrl.gain, "tol": ctrl.tol,
        "tol_cap": ctrl.tol_cap, "tenue_s": ctrl.tenue_s, "patience_s": ctrl.patience_s,
        "K": K.tolist()}))
    atteints = sum(b.phase is control.Phase.ATTEINT for b in bilans)
    print(f"{atteints}/{len(bilans)} poses atteintes")


# ---------------------------------------------------------------- essaim

def mode_essaim() -> None:
    n = 3
    layout, scene, K = monte(n)
    clock = Clock(scene.world)
    pilots = [demarre(scene, clock, i) for i in range(n)]
    ctrls = [controleur(scene, pilots[i], i) for i in range(n)]
    rng = random.Random(args.graine_tirage)
    panneaux = panneaux_lisibles(scene)

    def cible_dans_allee(exclues):
        for _ in range(2000):
            tag = rng.choice(panneaux)
            tirage = pose_devant(rng, tag, layout)
            if tirage is None:
                continue
            a = allee(layout, tirage[0][0])
            if a not in exclues:
                return tag, tirage, a
        raise RuntimeError("pas trois allees distinctes")

    allees = []
    cibles = []
    for i in range(n):
        tag, tirage, a = cible_dans_allee(allees)
        allees.append(a)
        cibles.append((tag, tirage))
    rapport = {"seed": args.seed, "drones": n, "manches": []}

    def manche(nom, affectations):
        traces = [Trace() for _ in range(n)]
        for i, (tag, (cible, psi, D, alpha)) in enumerate(affectations):
            points = chemin(layout, scene.position(i), cible)
            for w in points:
                w[2] = ALT_TRANSIT_MIN + COUCHE_ESSAIM * i
            ctrls[i].assigne(control.Consigne(cible, psi), points)
            print(f"  drone {i} -> allee {allee(layout, cible[0])}, {len(points)} pts, "
                  f"{ctrls[i].longueur:.1f} m")
        mur0 = time.monotonic()
        t0 = clock.t
        pas = 0
        while not all(c.phase in control.TERMINALES for c in ctrls):
            control.pas(ctrls, clock)
            for c, tr in zip(ctrls, traces):
                tr(c)
            pas += 1
        mur = time.monotonic() - mur0
        lignes = []
        for i, (tag, (cible, psi, D, alpha)) in enumerate(affectations):
            b = ctrls[i].bilan
            ecart = float(np.linalg.norm(cible - scene.position(i)))
            lu = lecture(scene, K, tag, i) if b.phase is control.Phase.ATTEINT else None
            print(f"  drone {i}: {b.phase.value:8s} {b.raison:7s} {b.t_total:5.1f} s  "
                  f"err {ecart:.2f} m  {'lu' if lu and lu['lu'] else 'non lu' if lu else '-'}")
            lignes.append({"drone": i, "tag": tag.tag_id, "cible": cible.tolist(), "psi": psi,
                           "bilan": b.as_dict(), "ecart_final": round(ecart, 3), "lecture": lu,
                           "traj": traces[i].lignes})
        sim = clock.t - t0
        print(f"  manche {nom}: {sim:.0f} s simulees en {mur:.0f} s reels "
              f"({sim / mur:.2f}x), {pas / mur:.1f} pas de mission/s")
        rapport["manches"].append({"nom": nom, "t_sim": round(sim, 1), "t_mur": round(mur, 1),
                                   "drones": lignes})

    manche("vers trois allees", cibles)
    secondes = []
    for i, a in enumerate(allees):
        for _ in range(2000):
            tag = rng.choice(panneaux)
            tirage = pose_devant(rng, tag, layout)
            if tirage is not None and allee(layout, tirage[0][0]) == a:
                secondes.append((tag, tirage))
                break
    if len(secondes) == n:
        manche("seconde pose dans la meme allee", secondes)
    for c in ctrls:
        c.pilot.hold(0.0)
    clock.pump(0.5)
    (HERE / f"{NOM}.json").write_text(json.dumps(rapport))
    print(f"rapport dans {NOM}.json")


MODES = {"freinage": mode_freinage, "poses": mode_poses, "essaim": mode_essaim}

try:
    MODES[args.mode]()
    print(f"BANC {args.mode.upper()} FINI")
except Exception:
    traceback.print_exc()
finally:
    simulation_app.close()
