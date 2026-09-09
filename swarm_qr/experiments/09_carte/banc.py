"""Banc de la carte partagée — étape 4.

  banc.py --mode verifie                  contrôles de la chaîne lidar → carte, sans vol
  banc.py --mode vol [--etages 0,1,2]     patrouille : la carte se remplit et guide le drone

Commande : DISPLAY=:1 PYTHONUNBUFFERED=1 timeout -s KILL 3h ~/isaac5_env/bin/python banc.py --mode ...

Le mode `verifie` s'arrête si le lidar ne dit pas la vérité — chaque rayon est refait par le
moteur physique — ou si sa conversion vers le repère du monde dépend du cap du drone. Le mode
`vol` ne connaît pas le plan de l'entrepôt : chaque déplacement est calculé sur la carte que
le drone a découverte lui-même. Il enregistre la carte, la trajectoire vraie et la vérité des
panneaux ; le jugement se fait ensuite par `analyse.py`, sans simulateur.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("verifie", "vol"), default="verifie")
parser.add_argument("--seed", type=int, default=9033)
parser.add_argument("--etages", default="0,1,2", help="étagères patrouillées, ex. 0 ou 0,1,2")
parser.add_argument("--allees", type=int, default=0, help="nombre d'allées, 0 = toutes")
parser.add_argument("--detecteur", default="", help="'auto' = poids retenus à l'étape 7, ou un .pt ; vide = repérage classique")
parser.add_argument("--sortie", default="", help="dossier des sorties (défaut : ce dossier)")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp(
    {"headless": True, "extra_args": ["--/rtx/verifyDriverVersion/enabled=false"]}
)

import traceback  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from swarm_qr import control, mapping  # noqa: E402
from swarm_qr.env import scene as scene_mod  # noqa: E402
from swarm_qr.env.config import CAMERAS, INTERIOR, LIDAR, RACKS  # noqa: E402
from swarm_qr.env.layout import make_layout  # noqa: E402
from swarm_qr.env.pilot import PHYS_DT, Clock, Pilot  # noqa: E402
from swarm_qr.experiments import _img  # noqa: E402

FLY_ALT = 1.6
RECUL = 2.2                 # distance de patrouille au bord d'un rack ; les cartons sont 20 cm en retrait
RECUL_MIN = 0.9             # dans une allée étroite, on se rapproche plutôt que d'y renoncer
MARGE_OPPOSEE = 0.85        # distance minimale gardée avec le rack ou le mur d'en face
HAUTEUR_PANNEAU = 0.225     # le QR est au milieu de la face d'un carton de 50 cm posé sur l'étagère
V_PATROUILLE = 0.6
PHYS_PAR_IMAGE = 160        # cadence de mission : 5 images par seconde
INCIDENCE_MAX = 60.0        # au-delà, la position déduite des quatre coins n'est plus fiable
DEPORT_RAYON = 0.45         # au-delà de la coque : un rayon parti du centre touche le drone
TOL_LIDAR_CM = 10.0
TOL_ROTATION_CM = 15.0
PANNEAU_M = 0.40


# ---------------------------------------------------------------- vérité terrain

def dans_un_rack(points, layout, marge: float = 0.0) -> np.ndarray:
    p = np.atleast_2d(np.asarray(points, float))
    out = np.zeros(len(p), dtype=bool)
    for r in layout.racks:
        (x0, x1), (y0, y1) = r.x_bounds, r.y_bounds
        out |= ((p[:, 0] > x0 - marge) & (p[:, 0] < x1 + marge)
                & (p[:, 1] > y0 - marge) & (p[:, 1] < y1 + marge))
    return out


def hors_des_murs(points, marge: float = 0.0) -> np.ndarray:
    p = np.atleast_2d(np.asarray(points, float))
    return ~((p[:, 0] > INTERIOR.x_min - marge) & (p[:, 0] < INTERIOR.x_max + marge)
             & (p[:, 1] > INTERIOR.y_min - marge) & (p[:, 1] < INTERIOR.y_max + marge))


# ---------------------------------------------------------------- contrôles

def pose_drone(scene, pos, cap_rad: float) -> None:
    scene.drones[0].set_world_pose(
        position=np.asarray(pos, float),
        orientation=Rotation.from_euler("z", cap_rad).as_quat()[[3, 0, 1, 2]])
    for _ in range(3):
        scene.world.step(render=True)             # le lidar ne se rafraîchit qu'au rendu


def controle_contre_la_physique(scene, pos, cap_rad: float, echantillon: int = 5) -> dict:
    """Chaque rayon du lidar est refait par le moteur physique ; les deux distances doivent
    coïncider. Le rayon de contrôle part au-delà de la coque, sinon la physique touche le
    drone lui-même."""
    from omni.physx import get_physx_scene_query_interface

    pose_drone(scene, pos, cap_rad)
    dirs, portees = scene.lidar(0)
    q = get_physx_scene_query_interface()
    ecarts = []
    for i in range(0, len(dirs), echantillon):
        if not np.isfinite(portees[i]) or portees[i] < DEPORT_RAYON + 0.1:
            continue
        h = q.raycast_closest((pos + dirs[i] * DEPORT_RAYON).tolist(), dirs[i].tolist(),
                              float(LIDAR.max_range))
        if h["hit"]:
            ecarts.append(abs(portees[i] - (h["distance"] + DEPORT_RAYON)))
    e = np.array(ecarts)
    med, faux = float(np.median(e)) * 100, float((e > 0.5).mean())
    print(f"[VERIF] lidar contre la physique, cap {math.degrees(cap_rad):.0f} deg : "
          f"{len(e)} rayons, ecart median {med:.1f} cm, {faux:.1%} de rayons faux")
    if med > TOL_LIDAR_CM or faux > 0.05:
        raise RuntimeError(f"le lidar ne dit pas la verite : {med:.0f} cm, {faux:.0%} de faux")
    return {"cap_deg": round(math.degrees(cap_rad)), "rayons": len(e),
            "ecart_median_cm": round(med, 2), "part_faux": round(faux, 4)}


def nuage(scene, pos, cap_rad: float) -> np.ndarray:
    pose_drone(scene, pos, cap_rad)
    dirs, portees = scene.lidar(0)
    ok = np.isfinite(portees) & (portees > 0)
    return np.asarray(pos, float) + dirs[ok] * portees[ok, None]


def controle_rotation(scene, pos) -> float:
    """Le monde ne bouge pas quand le drone tourne : les nuages pris à deux caps doivent se
    superposer. C'est ce qui prouve que la conversion vers le repère du monde est juste."""
    pres = lambda p: p[np.linalg.norm(p[:, :2] - np.asarray(pos)[:2], axis=1) < 6.0]
    a, b = pres(nuage(scene, pos, 0.0)), pres(nuage(scene, pos, math.pi / 2))
    d = np.linalg.norm(a[:, None, :2] - b[None, :, :2], axis=2).min(axis=1)
    ecart = float(np.median(d)) * 100
    print(f"[VERIF] rotation : cap 0 et cap 90 donnent le meme nuage a {ecart:.1f} cm")
    if ecart > TOL_ROTATION_CM:
        raise RuntimeError(f"les deux nuages different de {ecart:.0f} cm : repere lidar faux")
    return round(ecart, 2)


def controle_allees(scene, layout, pos) -> float:
    """À hauteur de vol, un point touché doit tomber sur un rack ou un mur, jamais dans une
    allée libre. L'entrepôt contient d'autres objets à d'autres hauteurs ; on ne juge que la
    bande où le drone vole."""
    pts = nuage(scene, pos, 0.0)
    pres = pts[np.linalg.norm(pts[:, :2] - np.asarray(pos)[:2], axis=1) < mapping.PORTEE_CARTE]
    bande = pres[(pres[:, 2] > 0.6) & (pres[:, 2] < 5.5)]
    solide = dans_un_rack(bande, layout, marge=0.35) | hors_des_murs(bande, marge=-0.35)
    part = float(solide.mean()) if len(bande) else 0.0
    print(f"[VERIF] points a hauteur de vol : {len(bande)}, {part:.1%} sur un rack ou un mur")
    if part < 0.95:
        raise RuntimeError(f"{1 - part:.1%} des points tombent dans une allee libre")
    return round(part, 4)


def point_de_controle(layout):
    r0, r1 = layout.racks[0], layout.racks[1]
    x = 0.5 * (r0.x_bounds[1] + r1.x_bounds[0])
    return np.array([x, 0.5 * (r0.y_bounds[0] + r0.y_bounds[1]), FLY_ALT])


def mode_verifie() -> None:
    layout = make_layout(args.seed)
    scene = scene_mod.build(layout, with_sitl=False, n_drones=1)
    scene.world.reset()
    scene.finalize()
    omni.timeline.get_timeline_interface().play()
    pos = point_de_controle(layout)
    print(f"point de controle : {np.round(pos, 2)}\n")

    physique = [controle_contre_la_physique(scene, pos, c) for c in (0.0, math.pi / 2, math.pi)]
    rotation = controle_rotation(scene, pos)
    allees = controle_allees(scene, layout, pos)

    carte = mapping.Carte()
    pose_drone(scene, pos, 0.0)
    dirs, portees = scene.lidar(0)
    t0 = time.perf_counter()
    for _ in range(10):
        carte.integre_lidar(pos, dirs, portees, t=0.0)
    ms_lidar = (time.perf_counter() - t0) * 100
    t0 = time.perf_counter()
    for _ in range(10):
        carte.integre_couverture(pos, [-1.0, 0.0, 0.0], t=0.0)
    ms_couv = (time.perf_counter() - t0) * 100
    r = carte.resume()
    print(f"\n[VERIF] cout : {ms_lidar:.1f} ms par tour de lidar, {ms_couv:.1f} ms par couverture")
    print(f"[VERIF] memoire : {r['octets'] / 1e6:.1f} Mo pour {r['cases']:,} cases")
    if ms_lidar + ms_couv > 200.0:
        raise RuntimeError("une observation coute trop cher devant le pas de decision")

    (HERE / "verification.json").write_text(json.dumps({
        "seed": args.seed, "point": pos.tolist(), "rayons_par_tour": int(len(dirs)),
        "physique": physique, "rotation_cm": rotation, "part_sur_matiere": allees,
        "ms_lidar": round(ms_lidar, 2), "ms_couverture": round(ms_couv, 2),
        "resume": r}, indent=1))
    cv2.imwrite(str(HERE / "vue_un_tour.png"), mapping.vue_de_dessus(carte))
    print(f"\nun seul tour de lidar : {r['part_connue']:.1%} de l'entrepot deja connu")


# ---------------------------------------------------------------- patrouille

def allees(layout):
    """Les bandes libres entre murs et racks, avec les faces à lire dans chacune."""
    racks = sorted(layout.racks, key=lambda r: r.x)
    bornes = [INTERIOR.x_min] + [x for r in racks for x in r.x_bounds] + [INTERIOR.x_max]
    out = []
    for k in range(len(racks) + 1):
        gauche, droite = bornes[2 * k], bornes[2 * k + 1]
        faces = []
        if k > 0:
            faces.append((racks[k - 1], +1))          # face est du rack de gauche
        if k < len(racks):
            faces.append((racks[k], -1))              # face ouest du rack de droite
        out.append({"k": k, "x0": gauche, "x1": droite, "faces": faces})
    return out


def trajets(layout, etages, spawn_x: float, n_allees: int):
    """La liste des allers simples : chaque face de rack longée à chaque hauteur. L'ordre suit
    l'espace découvert : l'allée du départ d'abord, puis les voisines de proche en proche, et
    chaque nouvelle allée est rejointe par le couloir que le passage précédent a révélé."""
    liste = allees(layout)
    k0 = next((a["k"] for a in liste if a["x0"] <= spawn_x < a["x1"]), 0)
    ordre = [a for a in liste if a["k"] >= k0] + [a for a in reversed(liste) if a["k"] < k0]
    if n_allees:
        ordre = ordre[:n_allees]
    out = []
    for a in ordre:
        largeur = a["x1"] - a["x0"]
        recul = min(RECUL, largeur - MARGE_OPPOSEE)
        if recul < RECUL_MIN:
            continue
        sens = 1
        for niveau in etages:
            z = RACKS.shelf_levels[niveau] + HAUTEUR_PANNEAU + CAMERAS.below
            for rack, cote in a["faces"]:
                x = rack.x + cote * (RACKS.depth / 2.0 + recul)
                y0, y1 = rack.y_bounds[0] + 0.3, rack.y_bounds[1] - 0.3
                debut, fin = (y0, y1) if sens > 0 else (y1, y0)
                # la caméra gauche regarde à 90 degrés du cap : cap = normale de la face + 90
                cap = math.atan2(0.0, cote) + math.pi / 2.0
                out.append({"allee": a["k"], "rack": rack.prim, "cote": cote, "etage": niveau,
                            "debut": [x, debut, z], "fin": [x, fin, z], "cap": cap})
                sens = -sens
    return out


from swarm_qr.observation import Observateur  # noqa: E402


class Patrouille:
    """La patrouille d'un drone : ses observations versées dans une carte neuve. La mécanique
    d'observation est celle du système (`swarm_qr.observation`), commune à la mission."""

    def __init__(self, scene, layout, K, detecteur=None):
        self.scene, self.layout, self.K = scene, layout, K
        self.carte = mapping.Carte()
        self.obs = Observateur(scene, self.carte, K, drone=0, detecteur=detecteur)
        self.compte = self.obs.compte
        self.compte["replanifications"] = 0
        self.couts = self.obs.couts

    @property
    def trajectoire(self):
        return self.obs.trajectoire

    def observe(self, t: float) -> None:
        self.obs.observe(t)


SORTIE = Path(args.sortie) if args.sortie else HERE


def mode_vol() -> None:
    SORTIE.mkdir(parents=True, exist_ok=True)
    etages = [int(e) for e in args.etages.split(",")]
    layout = make_layout(args.seed)
    scene = scene_mod.build(layout, with_sitl=True, n_drones=1)
    scene.world.reset()
    scene.finalize()
    omni.timeline.get_timeline_interface().play()
    K = np.asarray(scene.cameras[0]["left"].get_intrinsics_matrix(), float)
    fx_attendu = CAMERAS.side_width / (2.0 * math.tan(math.radians(CAMERAS.fov_deg) / 2.0))
    if abs(K[0, 0] - fx_attendu) > 1.0:
        raise RuntimeError(f"calibration incoherente : fx={K[0,0]:.1f}, attendu {fx_attendu:.1f}")
    print(f"[VERIF] calibration : fx = {K[0,0]:.1f} (attendu {fx_attendu:.1f})")

    clock = Clock(scene.world)
    pilot = Pilot(scene.world, 0, clock)
    if not pilot.ready(FLY_ALT, lambda: float(scene.position(0)[2])):
        raise RuntimeError("le drone n'a pas decolle")
    ctrl = control.Controleur(pilot, lambda: scene.position(0), lambda: scene.yaw(0),
                              lambda: scene.velocity(0), v_approche=V_PATROUILLE)
    detecteur = None
    if args.detecteur:
        from swarm_qr.detecteur import Detecteur

        detecteur = Detecteur() if args.detecteur == "auto" else Detecteur(args.detecteur)
        print(f"oeil appris : {detecteur.imgsz} px, seuil {detecteur.conf}")
    pat = Patrouille(scene, layout, K, detecteur)

    def pas(t_max: float | None = None) -> None:
        """Un pas de mission : commande, physique, rendu, observation."""
        ctrl.tick()
        for _ in range(PHYS_PAR_IMAGE - 1):
            scene.world.step(render=False)
        scene.world.step(render=True)
        clock.t += PHYS_PAR_IMAGE * PHYS_DT
        pat.observe(clock.t)

    for _ in range(3):                              # premier regard autour de soi
        pas()
    legs = trajets(layout, etages, float(scene.position(0)[0]), args.allees)
    print(f"{len(legs)} allers simples, etages {etages}\n")
    bilans, vues = [], []
    for k, leg in enumerate(legs):
        debut, fin = np.array(leg["debut"]), np.array(leg["fin"])
        # le transit est calculé sur la carte découverte, jamais sur le plan de l'entrepôt
        depart = scene.position(0)
        points = pat.carte.chemin(depart, debut, altitude=float(debut[2]))
        if points is None:
            print(f"  aller {k:2d} : debut inaccessible sur la carte, ecarte")
            pat.carte.ecarte(debut)
            bilans.append({**leg, "transit": None, "traversee": None})
            continue
        pat.carte.reserve(0, debut)
        ctrl.tol = 0.35
        ctrl.assigne(control.Consigne(debut, leg["cap"]), points)
        replans = 0
        n = 0
        t_depart = clock.t
        while ctrl.phase not in control.TERMINALES:
            pas()
            n += 1
            # un obstacle découvert en route peut couper le chemin déjà prévu : on le
            # revérifie toutes les 0,4 s — à 1,5 m/s, une seconde vaut déjà 1,5 m — et on
            # recalcule s'il le faut
            if n % 2 == 0 and ctrl.phase in (control.Phase.TRANSIT, control.Phase.APPROCHE):
                ici = scene.position(0)
                reste = [ici] + list(ctrl.points[ctrl.i_point:]) + [debut]
                if not all(pat.carte.segment_libre(a, b, altitude=float(debut[2]))
                           for a, b in zip(reste[:-1], reste[1:])):
                    nouveau = pat.carte.chemin(ici, debut, altitude=float(debut[2]))
                    if nouveau is None:
                        break
                    points = nouveau
                    ctrl.assigne(control.Consigne(debut, leg["cap"]), points)
                    replans += 1
                    pat.compte["replanifications"] += 1
        if ctrl.phase not in control.TERMINALES:
            print(f"  aller {k:2d} : chemin coupe et aucun autre, ecarte")
            pat.carte.ecarte(debut)
            pat.carte.libere(0)
            bilans.append({**leg, "transit": None, "traversee": None})
            continue
        transit = ctrl.bilan.as_dict()
        transit["points"] = len(points)
        transit["replanifications"] = replans
        # chaque recalcul relance le chronomètre du contrôleur : le temps du transit entier
        # se mesure depuis le départ
        transit["t_total"] = round(clock.t - t_depart, 2)
        traversee = None
        if ctrl.phase is control.Phase.ATTEINT:
            pat.carte.reserve(0, fin)
            ctrl.assigne(control.Consigne(fin, leg["cap"]))
            while ctrl.phase not in control.TERMINALES:
                pas()
            traversee = ctrl.bilan.as_dict()
        pat.carte.libere(0)
        r = pat.carte.resume()
        etat = (traversee or transit)["phase"]
        print(f"  aller {k:2d} allee {leg['allee']} {leg['rack']} cote {leg['cote']:+d} "
              f"etage {leg['etage']} : transit {len(points)} pts {transit['t_total']:5.1f} s"
              f"{' (' + str(replans) + ' recalculs)' if replans else ''}, "
              f"traversee {etat:8s} -> {r['part_connue']:.1%} connu, {r['codes_lus']:3d} codes, "
              f"{r['pistes']:3d} pistes")
        bilans.append({**leg, "transit": transit, "traversee": traversee})
        vues.append(mapping.vue_de_dessus(pat.carte, trajectoire=[p[1:] for p in pat.trajectoire]))
        cv2.imwrite(str(SORTIE / f"vue_{k:02d}.png"), vues[-1])

    pilot.hold(0.0)
    clock.pump(0.5)
    pat.carte.sauve(SORTIE / "carte")
    (SORTIE / "vol.json").write_text(json.dumps({
        "seed": args.seed, "etages": etages, "t_sim_s": round(clock.t, 1),
        "detecteur": args.detecteur,
        "compte": pat.compte,
        "ms": {k: round(float(np.median(v)), 1) for k, v in pat.couts.items() if v},
        "allers": bilans, "trajectoire": pat.trajectoire,
        "verite": [{"code": t.tag_id, "position": list(map(float, t.position)),
                    "normale": list(map(float, t.normal)), "taille": round(float(t.size), 3)}
                   for t in scene.tags],
        "racks": [{"prim": r.prim, "x": r.x_bounds, "y": r.y_bounds} for r in layout.racks],
    }, indent=1))
    cv2.imwrite(str(SORTIE / "carte_finale.png"),
                mapping.vue_de_dessus(pat.carte, trajectoire=[p[1:] for p in pat.trajectoire]))
    if len(vues) > 1:
        _img.video(vues, SORTIE / "carte_qui_se_remplit.mp4", fps=2)
    print("\ncarte, trajectoire et verite enregistrees ; le jugement se fait par analyse.py")


MODES = {"verifie": mode_verifie, "vol": mode_vol}

try:
    MODES[args.mode]()
    print(f"BANC {args.mode.upper()} FINI")
except Exception:
    traceback.print_exc()
finally:
    simulation_app.close()
