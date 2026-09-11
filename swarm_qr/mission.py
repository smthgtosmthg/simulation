"""Le chef d'orchestre : une mission complète à plusieurs drones (étape 5).

  mission.py --seed 9033 --drones 3 --budget 600 --detecteur auto --sortie experiments/11_mission/nominale
  mission.py ... --panne 1:180            le drone 1 tombe en panne à 180 s de temps simulé
  mission.py ... --guide smolvlm --lam 1  le guide de l'étape 8 conseille une zone et un côté

Trois rythmes : le contrôleur à chaque pas de physique, l'observation cinq fois par seconde, la
décision quand un drone n'a plus de cible. Une seule carte, partagée par tous. Les drones
s'évitent par leurs réservations, et par une règle de priorité : à moins de 1,5 m d'un drone de
plus petit numéro, on s'arrête et on le laisse passer. La mission finit quand plus personne n'a
de cible, ou quand le budget de temps est épuisé.
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
sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=9033)
parser.add_argument("--drones", type=int, default=3)
parser.add_argument("--budget", type=float, default=600.0, help="secondes de temps simulé")
parser.add_argument("--detecteur", default="auto", help="'auto', un .pt, ou '' pour le repérage classique")
parser.add_argument("--guide", default="", help="'' = sans guide ; 'entraine' = le modèle 3B + adaptateur LoRA ; sinon un nom de modèle (étape 8)")
parser.add_argument("--modele-guide", default=str(Path.home() / "Documents" / "qwen2.5-vl-3b"), dest="modele_guide")
parser.add_argument("--adaptateur", default=str(Path(__file__).resolve().parent / "experiments" / "12_guide" / "adaptateur_lora"))
parser.add_argument("--lam", type=float, default=0.0, help="poids de l'avis du guide dans la note")
parser.add_argument("--panne", default="", help="drone:temps — ce drone cesse d'agir à cet instant simulé")
parser.add_argument("--instantanes", type=float, default=10.0, help="secondes simulées entre deux instantanés (0 = aucun)")
parser.add_argument("--codes-attendus", type=int, default=0, dest="codes_attendus",
                    help="taille connue de l'inventaire (0 = le nombre de cartons de la scène)")
parser.add_argument("--part-arret", type=float, default=0.95, dest="part_arret",
                    help="part des codes attendus à partir de laquelle on accorde la grâce puis on s'arrête (0 = jamais)")
parser.add_argument("--grace", type=float, default=60.0, help="secondes de vol accordées après la part atteinte")
parser.add_argument("--sans-progres", type=float, default=120.0, dest="sans_progres",
                    help="secondes sans code nouveau après lesquelles on s'arrête (0 = jamais)")
parser.add_argument("--video", action="store_true", help="enregistre les caméras fixes à chaque rendu (vidéo à vitesse réelle)")
parser.add_argument("--cameras", type=int, default=2, choices=[2, 3, 5],
                    help="2 = couloir central + grande zone (choix de l'utilisatrice) ; 3 = + vue d'ensemble ; 5 = tous les couloirs")
parser.add_argument("--obstacle", default="", help="x,y,t — un bloc de 1x1x2 m apparaît à cet endroit à cet instant simulé")
parser.add_argument("--sortie", required=True)
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp(
    {"headless": True, "extra_args": ["--/rtx/verifyDriverVersion/enabled=false"]}
)

import traceback  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402

from swarm_qr import control, mapping, planning  # noqa: E402
from swarm_qr.env import scene as scene_mod  # noqa: E402
from swarm_qr.env.config import CAMERAS  # noqa: E402
from swarm_qr.env.layout import make_layout  # noqa: E402
from swarm_qr.env.pilot import PHYS_DT, Clock, Pilot  # noqa: E402
from swarm_qr.experiments import _img  # noqa: E402
from swarm_qr.observation import Observateur  # noqa: E402

PHYS_PAR_CYCLE = 160          # 0,2 s simulée : cinq observations par seconde
FLY_ALT = 1.6
SEPARATION = 2.5              # m ; en dessous, le drone de plus grand numéro s'arrête (fumée 7 : 55 cm à 1,5)
SEPARATION_Z = 1.2
URGENCE = 1.5                 # m ; en dessous, même le prioritaire s'arrête
BLOCAGE_S = 15.0              # s sans bouger de 30 cm avec une cible : on abandonne la cible
RECALCULS_MAX = 30
DECISION_REPOS_S = 2.0        # s entre deux décisions d'un drone sans cible
V_APPROCHE = 0.6              # m/s ; l'approche d'une pose tenue, comme la patrouille de l'étape 4
RAYON_COEQUIPIER = 2.0        # m ; un coéquipier en vol est un obstacle de ce rayon pour les chemins
MARGE_ALTITUDE = 1.5          # m ; on change d'altitude à au moins ça de tout obstacle connu
TOL_ARRIVEE = 0.35            # m ; à trois drones, une pose tenue oscille de 30 cm : 15 cm ne s'atteint jamais
GAIN_MISSION = 0.5            # vitesse commandée par mètre d'écart ; 0,9 (étape 3, un drone) oscille à trois drones
V_TRANSIT_MISSION = 1.0       # m/s ; un transit plus lent dépasse moins près des racks
LECTURE_S = 2.0               # s de tenue devant une cible de lecture : dix images
FIN_S = 30.0                  # s sans aucune cible pour personne : la mission est finie
AVIS_S = 5.0                  # s entre deux demandes au guide
SORTIE = Path(args.sortie)


def _json_sur(o):
    """Les entiers et flottants numpy qui se glissent dans le journal."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"non serialisable : {type(o)}")


class Agent:
    """Un drone dans la mission : son pilote, son contrôleur, ses capteurs, son cerveau."""

    def __init__(self, i, scene, clock, carte, K, detecteur, lam):
        self.i = i
        self.pilot = Pilot(scene.world, i, clock)
        if not self.pilot.ready(FLY_ALT + 0.3 * i, lambda: float(scene.position(i)[2])):
            raise RuntimeError(f"le drone {i} n'a pas decolle")
        self.ctrl = control.Controleur(self.pilot, lambda: scene.position(i), lambda: scene.yaw(i),
                                       lambda: scene.velocity(i), v_approche=V_APPROCHE, tol=TOL_ARRIVEE,
                                       gain=GAIN_MISSION, v_transit=V_TRANSIT_MISSION)
        self.obs = Observateur(scene, carte, K, drone=i, detecteur=detecteur)
        self.cerveau = planning.Cerveau(carte, i, lam=lam)
        self.cible: planning.Cible | None = None
        self.t_arrivee: float | None = None
        self.t_sans_cible: float | None = None
        self.vivant = True
        self.decisions: list[dict] = []
        self.attentes = 0
        self.avis: planning.Avis | None = None
        self.t_avis = -1e9
        self.codes_avant = 0
        self.t_cible = 0.0
        self.t_decision = -1e9
        self.attente = None                              # (consigne, points) mis de côté en cédant le passage


def zones_candidates(cibles, n_max: int = 6, taille: float = 3.0) -> list[dict]:
    """Les zones proposées au guide : les groupes de cibles les plus utiles, sur une grille de
    trois mètres, numérotées de 1 à n."""
    groupes: dict[tuple, list] = {}
    for c in cibles:
        cle = (math.floor(c.origine[0] / taille), math.floor(c.origine[1] / taille))
        groupes.setdefault(cle, []).append(c)
    zones = []
    for cle, cs in groupes.items():
        centre = np.mean([c.origine[:2] for c in cs], axis=0)
        cotes = [c.cote for c in cs if c.genre != "explorer"]
        zones.append({"centre": [round(float(v), 2) for v in centre], "rayon": taille * 0.75,
                      "utilite": round(float(sum(c.utilite for c in cs)), 1), "cibles": len(cs),
                      "genres": sorted({c.genre for c in cs}),
                      "n_lire": sum(1 for c in cs if c.genre == "lire"),
                      "n_couvrir": sum(1 for c in cs if c.genre == "couvrir"),
                      "n_couvrir_cartons": sum(1 for c in cs if c.genre == "couvrir" and c.utilite > planning.UTILITE_SURFACE * planning.SURFACE_MIN * 2),
                      "n_explorer": sum(1 for c in cs if c.genre == "explorer"),
                      "cote": planning.NOMS_COTES[max(set(cotes), key=cotes.count)] if cotes else None})
    zones.sort(key=lambda z: -z["utilite"])
    for k, z in enumerate(zones[:n_max]):
        z["numero"] = k + 1
    return zones[:n_max]


def vue_annotee(carte, trajectoires, zones) -> np.ndarray:
    img = mapping.vue_de_dessus(carte, trajectoire=trajectoires[0] if trajectoires else None)
    ech = img.shape[1] / carte.forme[0]
    ny = carte.forme[1]

    def px(p):
        i, j = carte.indice(p)[0][:2]
        return int((i + 0.5) * ech), int((ny - j - 0.5) * ech)

    for tr, couleur in zip(trajectoires[1:], mapping.COULEURS_DRONES[1:]):
        pts = [px(p) for p in tr if carte.sur_la_carte(p)[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(img, a, b, couleur, 1, cv2.LINE_AA)
    for z in zones:
        c = px([*z["centre"], 0.0])
        cv2.circle(img, c, int(z["rayon"] * ech / carte.g.cell), (0, 0, 230), 2)
        cv2.putText(img, str(z["numero"]), (c[0] - 8, c[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 230), 2, cv2.LINE_AA)
    return img


def verite_des_zones(zones, tags, carte) -> dict:
    """Ce que le guide devrait répondre, connu après coup : la zone qui contient le plus de
    panneaux non lus, et le côté d'où ils se lisent."""
    lus = carte.codes
    restants = [t for t in tags if t.tag_id not in lus]
    meilleur, n_meilleur, cote = None, -1, None
    for z in zones:
        c = np.array(z["centre"])
        dans = [t for t in restants if np.linalg.norm(np.array(t.position[:2]) - c[:2]) <= z["rayon"]]
        if len(dans) > n_meilleur:
            meilleur, n_meilleur = z["numero"], len(dans)
            if dans:
                normale = np.mean([t.normal for t in dans], axis=0)
                cote = planning.NOMS_COTES[carte.cardinal(normale)]
    return {"zone": meilleur, "panneaux_restants": n_meilleur, "cote": cote,
            "restants_total": len(restants)}


def main() -> None:
    SORTIE.mkdir(parents=True, exist_ok=True)
    (SORTIE / "instantanes").mkdir(exist_ok=True)
    layout = make_layout(args.seed)
    scene = scene_mod.build(layout, with_sitl=True, n_drones=args.drones)
    scene.world.reset()
    scene.finalize()
    omni.timeline.get_timeline_interface().play()
    K = np.asarray(scene.cameras[0]["left"].get_intrinsics_matrix(), float)
    fx_attendu = CAMERAS.side_width / (2.0 * math.tan(math.radians(CAMERAS.fov_deg) / 2.0))
    if abs(K[0, 0] - fx_attendu) > 1.0:
        raise RuntimeError(f"calibration incoherente : fx={K[0, 0]:.1f}, attendu {fx_attendu:.1f}")

    detecteur = None
    if args.detecteur:
        from swarm_qr.detecteur import Detecteur

        detecteur = Detecteur() if args.detecteur == "auto" else Detecteur(args.detecteur)
        print(f"oeil appris : {detecteur.imgsz} px, seuil {detecteur.conf}")
    guide = None
    if args.guide:
        from swarm_qr.guide import Guide, GuideEntraine

        guide = (GuideEntraine(args.modele_guide, args.adaptateur) if args.guide == "entraine"
                 else Guide(args.guide))
        print(f"guide : {args.guide}, lambda {args.lam}")

    clock = Clock(scene.world)
    carte = mapping.Carte()
    agents = [Agent(i, scene, clock, carte, K, detecteur, args.lam) for i in range(args.drones)]
    panne = None
    if args.panne:
        d, t = args.panne.split(":")
        panne = (int(d), float(t))
    tags = scene.tags
    jeux = {2: scene_mod.CAMERAS_VIDEO_2, 3: scene_mod.CAMERAS_VIDEO_3, 5: scene_mod.CAMERAS_VIDEO}
    cams_video = scene_mod.cameras_fixes(jeux[args.cameras]) if args.video else {}
    if args.video:
        for nom in list(cams_video) + ["lecteur"]:
            (SORTIE / "video" / nom).mkdir(parents=True, exist_ok=True)
        index_video: list[dict] = []
    obstacle_prevu = None
    if args.obstacle:
        ox, oy, ot = (float(v) for v in args.obstacle.split(","))
        obstacle_prevu = {"x": ox, "y": oy, "t": ot, "pose": False}
    racks_connus = [{"prim": r.prim, "x": list(r.x_bounds), "y": list(r.y_bounds)} for r in layout.racks]
    journal = {"seed": args.seed, "drones": args.drones, "budget_s": args.budget,
               "arret": {"codes_attendus": args.codes_attendus or len({t.tag_id for t in tags}), "part": args.part_arret,
                         "grace_s": args.grace, "sans_progres_s": args.sans_progres},
               "detecteur": args.detecteur, "guide": args.guide, "lam": args.lam,
               "panne": args.panne, "evenements": [], "codes_par_t": [], "instantanes": []}
    mur0 = time.monotonic()
    n_cycle = 0
    t_fin_candidats = None
    fin = None
    # un carton porte le même code sur ses deux faces : l'inventaire se compte en codes distincts
    codes_attendus = args.codes_attendus or len({t.tag_id for t in tags})
    t_dernier_code, t_part_atteinte = 0.0, None
    prochain_instantane = 0.0
    avis_en_cours: dict[int, object] = {}

    def evenement(genre, **kw):
        journal["evenements"].append({"t": round(clock.t, 1), "genre": genre, **kw})

    def libre_de_passage(a: Agent) -> bool:
        """Le drone de plus grand numéro cède le passage à 1,5 m ; à 80 cm, tout le monde
        s'arrête, prioritaire ou non."""
        p = scene.position(a.i)
        for b in agents:
            if b.i == a.i or not b.vivant:
                continue
            q = scene.position(b.i)
            proche = np.linalg.norm(p[:2] - q[:2])
            if proche < URGENCE and abs(p[2] - q[2]) < SEPARATION_Z:
                return False
            if b.i < a.i and proche < SEPARATION and abs(p[2] - q[2]) < SEPARATION_Z:
                return False
        return True

    def bloque(a: Agent) -> bool:
        """Un drone qui a une cible depuis quinze secondes et n'a pas bougé de 30 cm depuis
        est coincé : contre un coéquipier, contre une structure que la carte ignore, ou dans
        une boucle de recalculs. La cible est abandonnée et écartée. On compte depuis
        l'affectation : un drone qui tenait sa place vient forcément d'être immobile."""
        if a.cible is None or clock.t - a.t_cible < BLOCAGE_S:
            return False
        recents = [p for p in a.obs.trajectoire if p[0] >= max(a.t_cible, clock.t - BLOCAGE_S)]
        if len(recents) < 5:
            return False
        pts = np.array([p[1:] for p in recents])
        return bool(np.ptp(pts, axis=0).max() < 0.3)

    def tient_sur_place(a: Agent) -> None:
        """Un drone sans cible tient sa position activement : à vitesse nulle, ArduPilot
        dérive de plusieurs centimètres par seconde (étape 3), jusque dans un rack."""
        ici = scene.position(a.i)
        a.ctrl.assigne(control.Consigne(ici.copy(), scene.yaw(a.i)))

    def avec_coequipiers(a: Agent):
        """Pendant que ce drone planifie, les autres sont des obstacles dans la carte."""
        carte.obstacles_mobiles = [(scene.position(b.i).copy(), RAYON_COEQUIPIER)
                                   for b in agents if b.i != a.i and b.vivant]

    def decide(a: Agent) -> None:
        if a.cible is not None:
            lu = a.cible.genre == "lire" and len(carte.codes) > a.codes_avant
            if lu:
                carte.oublie_pistes(a.cible.origine, 1.0)
            a.cerveau.constate(a.cible, lu)
            if a.ctrl.phase is control.Phase.ABANDON:
                carte.ecarte(a.cible.position)
            a.decisions[-1].update({"fin": round(clock.t, 1), "phase": a.ctrl.phase.value,
                                    "lu": bool(lu), "raison": a.ctrl.bilan.raison if a.ctrl.bilan else ""})
            carte.libere(a.i)
            a.cible = None
        avec_coequipiers(a)
        choix = a.cerveau.choisit(scene.position(a.i), avis=a.avis, t=clock.t)
        carte.obstacles_mobiles = []
        if choix is None:
            if a.t_sans_cible is None:
                a.t_sans_cible = clock.t
                evenement("sans_cible", drone=a.i)
                tient_sur_place(a)
            return
        a.t_sans_cible = None
        cap = choix.cap if choix.cap is not None else scene.yaw(a.i)
        ici = scene.position(a.i)
        points = list(choix.points)
        if abs(choix.position[2] - ici[2]) > 0.5:
            # l'altitude d'abord, puis le trajet : un drone qui monte en avançant entre dans
            # une structure avant d'être au-dessus. Et il change d'altitude loin de tout
            # obstacle : à trois drones, un drone tenu dérive jusqu'à un mètre.
            degage = a.cerveau.point_degage(ici, MARGE_ALTITUDE)
            # le point dégagé le plus proche peut être de l'autre côté d'un montant : on n'y va
            # que si le segment est libre, sinon on change d'altitude sur place
            if degage is None or not carte.segment_libre(ici, np.array([degage[0], degage[1], ici[2]])):
                degage = ici
            prefixe = [] if np.linalg.norm(degage[:2] - ici[:2]) < 0.3 else [np.array([degage[0], degage[1], ici[2]])]
            points = prefixe + [np.array([degage[0], degage[1], choix.position[2]])] + points
        a.ctrl.assigne(control.Consigne(choix.position, cap), points)
        carte.reserve(a.i, choix.position)
        a.cible = choix
        a.attente = None
        a.t_cible = clock.t
        a.t_arrivee = None
        a.codes_avant = len(carte.codes)
        a.decisions.append({"t": round(clock.t, 1), "genre": choix.genre,
                            "position": [round(float(v), 2) for v in choix.position],
                            "origine": [round(float(v), 2) for v in choix.origine],
                            "cote": choix.cote, "note": round(choix.note, 1),
                            "distance": round(choix.distance, 1), "points": len(choix.points),
                            "avis": None if a.avis is None else a.avis.phrase})

    def replanifie(a: Agent) -> None:
        c = a.cible
        ici = scene.position(a.i)
        reste = [ici] + list(a.ctrl.points[a.ctrl.i_point:]) + [c.position]
        avec_coequipiers(a)
        try:
            if all(carte.segment_libre(u, v, altitude=float(c.position[2])) for u, v in zip(reste[:-1], reste[1:])):
                return
            nouveau = carte.chemin(ici, c.position, altitude=float(c.position[2]))
        finally:
            carte.obstacles_mobiles = []
        if nouveau is None:
            abandonne(a, "chemin coupe")
            return
        cap = c.cap if c.cap is not None else scene.yaw(a.i)
        a.decisions[-1]["replanifications"] = a.decisions[-1].get("replanifications", 0) + 1
        if a.decisions[-1]["replanifications"] > RECALCULS_MAX:
            abandonne(a, "trop de recalculs")
            return
        a.ctrl.assigne(control.Consigne(c.position, cap), nouveau)

    def abandonne(a: Agent, raison: str) -> None:
        a.ctrl.phase = control.Phase.ABANDON
        a.ctrl.bilan = control.Bilan(control.Phase.ABANDON, raison, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        evenement("abandon", drone=a.i, raison=raison)

    def sauve_journaux_ardupilot() -> None:
        """Les journaux de bord des autopilotes (dataflash) vivent dans un dossier temporaire
        effacé à la sortie : on les copie avant, ils disent de l'intérieur pourquoi un drone tombe."""
        import shutil
        for i, d in enumerate(scene.drones):
            try:
                outil = d._backends[0].ardupilot_tool
                src = Path(outil.root_fs.name) / "logs"
                dst = SORTIE / "ardupilot_logs" / f"drone_{i}"
                dst.mkdir(parents=True, exist_ok=True)
                for f in src.glob("*.BIN"):
                    shutil.copy2(f, dst / f.name)
            except Exception as e:                              # noqa: BLE001
                print(f"  journaux ArduPilot du drone {i} non copies ({type(e).__name__}: {e})", flush=True)

    def enregistre_video() -> None:
        """Une image par caméra fixe à chaque rendu, plus la caméra de lecture du drone qui lit ;
        la vidéo est assemblée après le vol (experiments/11_mission/video.py)."""
        n = len(index_video)
        lecteur = next((a.i for a in agents if a.vivant and a.cible is not None and a.cible.genre == "lire"
                        and a.ctrl.phase is control.Phase.ATTEINT), None)
        if lecteur is None:
            lecteur = next((a.i for a in agents if a.vivant), None)
        for nom, cam in cams_video.items():
            img = cam.get_rgb()
            if img is not None and getattr(img, "ndim", 0) == 3 and img.size:
                cv2.imwrite(str(SORTIE / "video" / nom / f"{n:05d}.jpg"), _img.to_bgr(img), [cv2.IMWRITE_JPEG_QUALITY, 85])
        if lecteur is not None:
            img = scene.cameras[lecteur]["left"].get_rgb()
            if img is not None and getattr(img, "ndim", 0) == 3 and img.size:
                cv2.imwrite(str(SORTIE / "video" / "lecteur" / f"{n:05d}.jpg"),
                            cv2.resize(_img.to_bgr(img), (480, 360)), [cv2.IMWRITE_JPEG_QUALITY, 85])
        index_video.append({"n": n, "t": round(clock.t, 1), "codes": len(carte.codes), "lecteur": lecteur,
                            "drones": [{"i": a.i, "vivant": a.vivant,
                                        "position": [round(float(v), 2) for v in scene.position(a.i)],
                                        "genre": a.cible.genre if a.cible else None} for a in agents]})
        if n % 100 == 0:
            (SORTIE / "video" / "index.json").write_text(json.dumps(
                {"pas_s": 0.2, "codes_attendus": codes_attendus, "images": index_video}))

    def instantane() -> None:
        cibles = agents[0].cerveau.candidats(clock.t)
        zones = zones_candidates(cibles)
        if not zones:
            return
        k = len(journal["instantanes"])
        trajs = [[p[1:] for p in a.obs.trajectoire] for a in agents]
        vue = vue_annotee(carte, trajs, zones)
        cv2.imwrite(str(SORTIE / "instantanes" / f"{k:03d}_vue.png"), vue)
        for a in agents:
            if a.vivant:
                for nom, suffixe in (("left", ""), ("right", "d")):
                    img = scene.cameras[a.i][nom].get_rgb()
                    if img is not None and getattr(img, "ndim", 0) == 3 and img.size:
                        cv2.imwrite(str(SORTIE / "instantanes" / f"{k:03d}_cam{a.i}{suffixe}.jpg"),
                                    cv2.resize(_img.to_bgr(img), (512, 384)), [cv2.IMWRITE_JPEG_QUALITY, 85])
        journal["instantanes"].append({
            "k": k, "t": round(clock.t, 1), "zones": zones,
            "drones": [{"i": a.i, "position": [round(float(v), 2) for v in scene.position(a.i)],
                        "cap": round(scene.yaw(a.i), 3), "vivant": a.vivant,
                        "phase": a.ctrl.phase.name if a.ctrl.phase is not None else None,
                        "cible": None if a.cible is None else {
                            "genre": a.cible.genre, "position": _liste(a.cible.position),
                            "origine": _liste(a.cible.origine), "cote": a.cible.cote,
                            "utilite": round(float(a.cible.utilite), 1), "note": round(float(a.cible.note), 1),
                            "depuis": round(clock.t - a.t_cible, 1)}} for a in agents],
            "verite": verite_des_zones(zones, tags, carte),
            **dossier_de_la_carte(k, cibles)})

    def _liste(v):
        return [round(float(x), 2) for x in np.asarray(v).ravel()]

    def dossier_de_la_carte(k: int, cibles) -> dict:
        """Tout ce que la carte sait à cet instant, pour les bancs hors ligne : la grille
        elle-même, les panneaux lus, les pistes, toutes les cibles candidates, les frontières et
        les réservations. Un échec ici ne doit jamais arrêter un vol."""
        try:
            carte.sauve(SORTIE / "instantanes" / f"{k:03d}_carte")
            front = carte.frontieres(planning.Z_MIN, planning.Z_MAX)
            return {
                "resume": carte.resume(),
                "panneaux": [{"code": q.code, "position": _liste(q.position), "normale": _liste(q.normale),
                              "lectures": q.lectures, "vu_le": round(q.vu_le, 1)} for q in carte.panneaux],
                "pistes": [{"position": _liste(q.position),
                            "normale": None if q.normale is None else _liste(q.normale),
                            "vues": q.vues, "vu_le": round(q.vu_le, 1)} for q in carte.pistes],
                "cibles": [{"genre": c.genre, "position": _liste(c.position), "origine": _liste(c.origine),
                            "cote": c.cote, "utilite": round(float(c.utilite), 1)} for c in cibles],
                "frontieres": {"cases": int(len(front)),
                               "exemples": [_liste(f) for f in np.asarray(front)[:200]]},
                "reservations": [{"drone": r.drone, "cible": _liste(r.cible), "jusqu_a": round(r.jusqu_a, 1)}
                                 for r in carte.reservations.values()],
            }
        except Exception as e:                                  # noqa: BLE001
            print(f"  instantane {k} : dossier incomplet ({type(e).__name__}: {e})", flush=True)
            return {}

    def demande_avis(a: Agent) -> None:
        """Le guide travaille en arrière-plan ; l'avis sert à la prochaine décision."""
        if guide is None or clock.t - a.t_avis < AVIS_S:
            return
        from swarm_qr.guide import GuideEntraine
        if a.i in avis_en_cours:
            if not avis_en_cours[a.i].done():
                return
            avis = avis_en_cours.pop(a.i).result()
            if avis is not None:
                a.avis = avis
                evenement("avis", drone=a.i, zone=avis.phrase)
        cibles = a.cerveau.candidats(clock.t)
        zones = zones_candidates(cibles)
        if not zones:
            return
        if isinstance(guide, GuideEntraine):
            # le guide entraîné lit le dossier de la carte, sans image
            cas = {"t": round(clock.t, 1), "codes_lus": len(carte.codes), "drone": a.i,
                   "position": [float(v) for v in scene.position(a.i)],
                   "cap_deg": round(float(np.degrees(scene.yaw(a.i))), 1),
                   "coequipiers": [{"i": b.i, "position": [float(v) for v in scene.position(b.i)],
                                    "cap": float(scene.yaw(b.i)), "vivant": b.vivant} for b in agents if b.vivant],
                   "racks": racks_connus}
            avis_en_cours[a.i] = guide.demande(cas, zones)
            a.t_avis = clock.t
            return
        img = scene.cameras[a.i]["left"].get_rgb()
        if img is None or getattr(img, "ndim", 0) != 3 or not img.size:
            return
        vue = vue_annotee(carte, [[p[1:] for p in a.obs.trajectoire]], zones)
        avis_en_cours[a.i] = guide.demande(_img.to_bgr(img), vue, zones, scene.position(a.i))
        a.t_avis = clock.t

    print(f"mission : {args.drones} drones, entrepot {args.seed}, budget {args.budget:.0f} s\n")
    cycles_ms: list[float] = []
    try:
        while clock.t < args.budget:
            mur_cycle = time.monotonic()
            # --- commande ---
            for a in agents:
                if not a.vivant:
                    a.ctrl.tick()                       # en panne : il tient sa place, sans rien décider
                    continue
                if libre_de_passage(a):
                    if a.attente is not None:           # le passage est libre : on reprend la cible
                        consigne, points = a.attente
                        a.attente = None
                        a.ctrl.assigne(consigne, points)
                    a.ctrl.tick()
                else:
                    if a.attente is None:               # céder le passage : tenir sa place, activement
                        a.attente = (a.ctrl.consigne, list(a.ctrl.points[a.ctrl.i_point:]))
                        tient_sur_place(a)
                    a.ctrl.tick()
                    a.attentes += 1
            # --- physique et rendu ---
            for _ in range(PHYS_PAR_CYCLE - 1):
                scene.world.step(render=False)
            scene.world.step(render=True)
            if obstacle_prevu and not obstacle_prevu["pose"] and clock.t >= obstacle_prevu["t"]:
                emprise = scene_mod.ajoute_obstacle("bloc_1", (obstacle_prevu["x"], obstacle_prevu["y"]))
                obstacle_prevu["pose"] = True
                journal["obstacle"] = {**emprise, "t": round(clock.t, 1)}
                evenement("obstacle", t_apparition=round(clock.t, 1), x=obstacle_prevu["x"], y=obstacle_prevu["y"])
                print(f"  t={clock.t:6.1f} s  OBSTACLE pose en ({obstacle_prevu['x']}, {obstacle_prevu['y']})")
            if args.video:
                enregistre_video()
            clock.t += PHYS_PAR_CYCLE * PHYS_DT
            n_cycle += 1
            # --- observation ---
            for a in agents:
                if a.vivant:
                    a.obs.observe(clock.t, oeil=(n_cycle % len(agents) == a.i))
            carte.vieillit(clock.t)
            journal["codes_par_t"].append([round(clock.t, 1), len(carte.codes)])
            # --- panne ---
            if panne and clock.t >= panne[1] and agents[panne[0]].vivant:
                a = agents[panne[0]]
                a.vivant = False
                carte.libere(a.i)
                a.cible = None
                tient_sur_place(a)
                evenement("panne", drone=a.i)
                print(f"  t={clock.t:6.1f} s  PANNE du drone {a.i}")
            # --- chute : un drone au sol ne vole plus ; il libère ses cibles comme une panne ---
            for a in agents:
                inclinaison = a.obs.inclinaisons[-1][1] if a.obs.inclinaisons else 0.0
                if a.vivant and clock.t > 120.0 and (scene.position(a.i)[2] < 0.3 or inclinaison > 70.0):
                    a.vivant = False
                    carte.libere(a.i)
                    a.cible = None
                    evenement("chute", drone=a.i, position=[round(float(v), 2) for v in scene.position(a.i)])
                    print(f"  t={clock.t:6.1f} s  CHUTE du drone {a.i}")
            # --- décision : une seule par cycle, pour ne jamais immobiliser la physique ---
            decide_fait = False
            for a in agents:
                if not a.vivant:
                    continue
                demande_avis(a)
                if a.cible is not None and a.ctrl.phase not in control.TERMINALES and bloque(a):
                    abandonne(a, "immobile")
                phase = a.ctrl.phase
                if a.cible is not None and phase not in control.TERMINALES:
                    if n_cycle % 2 == 0 and phase in (control.Phase.TRANSIT, control.Phase.APPROCHE):
                        replanifie(a)
                    continue
                if a.cible is not None and phase is control.Phase.ATTEINT:
                    if a.t_arrivee is None:
                        a.t_arrivee = clock.t
                    if a.cible.genre != "explorer" and clock.t - a.t_arrivee < LECTURE_S:
                        continue
                if a.cible is None and clock.t - a.t_decision < DECISION_REPOS_S:
                    continue                            # sans cible : une décision toutes les deux secondes
                if decide_fait:
                    continue                            # au cycle suivant
                a.t_decision = clock.t
                decide(a)
                decide_fait = True
            # --- fin de mission ---
            vivants = [a for a in agents if a.vivant]
            if vivants and all(a.t_sans_cible is not None for a in vivants):
                if t_fin_candidats is None:
                    t_fin_candidats = clock.t
                elif clock.t - t_fin_candidats > FIN_S:
                    fin = "plus aucune cible"
                    break
            else:
                t_fin_candidats = None
            if not vivants:
                fin = "aucun drone vivant"
                break
            # --- arrêt sur l'inventaire : la taille de l'inventaire est connue, on ne vole pas
            #     600 s pour un dernier code introuvable ---
            lus = len(carte.codes)
            if journal["codes_par_t"] and len(journal["codes_par_t"]) > 1 and lus > journal["codes_par_t"][-2][1]:
                t_dernier_code = clock.t
            if args.part_arret > 0 and lus >= args.part_arret * codes_attendus:
                if t_part_atteinte is None:
                    t_part_atteinte = clock.t
                    evenement("part_atteinte", codes=lus, attendus=codes_attendus)
                elif clock.t - t_part_atteinte >= args.grace:
                    fin = f"inventaire a {lus / codes_attendus:.0%} et grace ecoulee"
                    break
            if args.sans_progres > 0 and lus > 0 and clock.t - t_dernier_code >= args.sans_progres:
                fin = f"sans code nouveau depuis {args.sans_progres:.0f} s"
                break
            # --- instantanés et journal ---
            if args.instantanes and clock.t >= prochain_instantane:
                instantane()
                prochain_instantane = clock.t + args.instantanes
            cycles_ms.append((time.monotonic() - mur_cycle) * 1000)
            journal.setdefault("cycles_t", []).append([round(clock.t, 1), round(cycles_ms[-1])])
            if n_cycle % 300 == 0:
                brouillon = {"t_sim_s": round(clock.t, 1), "evenements": journal["evenements"],
                             "agents": [{"i": a.i, "vivant": a.vivant, "decisions": a.decisions,
                                         "trajectoire": a.obs.trajectoire, "inclinaisons": a.obs.inclinaisons}
                                        for a in agents]}
                (SORTIE / "brouillon.json").write_text(json.dumps(brouillon, default=_json_sur))
            if n_cycle % 150 == 0:
                r = carte.resume()
                etat = " ".join(f"d{a.i}:{'panne' if not a.vivant else (a.cible.genre if a.cible else 'libre')}"
                                for a in agents)
                print(f"  t={clock.t:6.1f} s  {r['part_connue']:.0%} connu  {r['codes_lus']:3d} codes  "
                      f"{r['pistes']:3d} pistes  {etat}  ({(time.monotonic() - mur0) / 60:.0f} min)")
        if fin is None:
            fin = "budget epuise"
    finally:
        if args.video:
            (SORTIE / "video" / "index.json").write_text(json.dumps(
                {"pas_s": 0.2, "codes_attendus": codes_attendus, "images": index_video}))
        sauve_journaux_ardupilot()
        for a in agents:
            a.pilot.hold()
        clock.pump(0.5)
        # --- rapport ---
        carte.sauve(SORTIE / "carte")
        cm = np.array(cycles_ms) if cycles_ms else np.zeros(1)
        journal.update({
            "fin": fin, "t_sim_s": round(clock.t, 1), "mur_min": round((time.monotonic() - mur0) / 60, 1),
            "cycles": {"n": int(len(cm)), "mediane_ms": round(float(np.median(cm)), 1),
                       "max_ms": round(float(cm.max()), 1), "lents_plus_de_500_ms": int((cm > 500).sum()),
                       "lents_plus_de_1_s": int((cm > 1000).sum())},
            "agents": [{"i": a.i, "vivant": a.vivant, "decisions": a.decisions, "attentes": a.attentes,
                        "compte": a.obs.compte,
                        "ms": {k: round(float(np.median(v)), 1) for k, v in a.obs.couts.items() if v},
                        "trajectoire": a.obs.trajectoire, "inclinaisons": a.obs.inclinaisons} for a in agents],
            "resume": carte.resume(),
            "verite": [{"code": t.tag_id, "position": list(map(float, t.position)),
                        "normale": list(map(float, t.normal)), "taille": round(float(t.size), 3)}
                       for t in tags],
            "racks": [{"prim": r.prim, "x": r.x_bounds, "y": r.y_bounds} for r in layout.racks],
        })
        (SORTIE / "mission.json").write_text(json.dumps(journal, indent=1, default=_json_sur))
        trajs = [[p[1:] for p in a.obs.trajectoire] for a in agents]
        cv2.imwrite(str(SORTIE / "carte_finale.png"), vue_annotee(carte, trajs, []))
        r = carte.resume()
        print(f"\nfin : {fin} a t={clock.t:.0f} s ; {r['codes_lus']} codes lus, {r['part_connue']:.0%} connu, "
              f"{r['pistes']} pistes restantes ; {(time.monotonic() - mur0) / 60:.0f} min de calcul")


try:
    main()
    print("MISSION FINIE")
except Exception:
    traceback.print_exc()
finally:
    simulation_app.close()
