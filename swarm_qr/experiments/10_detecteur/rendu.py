"""Étape 7 — les images et leurs cadres, fabriqués sans un seul clic.

  rendu.py --seed 3 --images 500                      caméra libre, poses au hasard dans l'entrepôt
  rendu.py --seed 9033 --images 400 --relabel optique,sans_qr,vol,traversee
                                                      + cadres des images de l'étape 2 (poses relues)

Un cadre n'est écrit que si l'objet est réellement visible : sa surface est échantillonnée,
chaque échantillon est projeté dans l'image, et un rayon du moteur physique confirme que rien
ne s'interpose. Le cadre entoure les échantillons visibles, pas l'objet entier : un carton à
moitié caché reçoit un cadre sur sa moitié visible.
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
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--images", type=int, default=500)
parser.add_argument("--graine", type=int, default=None, help="graine du tirage des poses")
parser.add_argument("--relabel", default="", help="jeux de l'étape 2 à annoter, séparés par des virgules")
parser.add_argument("--sortie", default=str(HERE / "jeu"))
parser.add_argument("--relabel-rendu", action="store_true",
                    help="recalcule les cadres de jeu/rendu_<seed> depuis ses poses, sans nouveau rendu")
parser.add_argument("--debug", default="", help="jeu:image — diagnostic des panneaux proches pour cette pose")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp(
    {"headless": True, "extra_args": ["--/rtx/verifyDriverVersion/enabled=false"]}
)

import traceback  # noqa: E402

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from swarm_qr import perception as P  # noqa: E402
from swarm_qr.env import scene as scene_mod  # noqa: E402
from swarm_qr.env.config import CAMERAS, INTERIOR, OBSTACLES, RACKS  # noqa: E402
from swarm_qr.env.layout import make_layout  # noqa: E402
from swarm_qr.experiments import _img  # noqa: E402

CLASSES = ("qr", "carton")
ETAPE2 = HERE.parent / "07_enveloppe"
SORTIE = Path(args.sortie)

# --- tirage des poses ---
Z_MIN, Z_MAX = 0.9, 5.3
MARGE_MUR = 0.7
MARGE_RACK = 0.5
PART_VISE_RACK = 0.7          # le reste regarde n'importe où : allées, murs, bouts de rack
LACET_SIGMA_DEG = 25.0
TANGAGE_SIGMA_DEG, TANGAGE_MAX_DEG = 6.0, 15.0
ROULIS_SIGMA_DEG, ROULIS_MAX_DEG = 3.0, 8.0
RENDUS_PAR_IMAGE = 7          # retard mesuré à l'étape 2 : 5 rendus

# --- annotation ---
GRILLE = 5                    # échantillons par côté de surface
SEUIL_VISIBLE = {"qr": 0.30, "carton": 0.20}
COTE_MIN_PX = {"qr": 8.0, "carton": 14.0}
TOL_RAYON = 0.03              # + 1 % de la distance : le collider et la maille diffèrent d'autant
PORTEE_MAX = 40.0            # tout l'entrepôt : un QR non annoté serait appris comme « rien »


# ---------------------------------------------------------------- géométrie

class Projecteur:
    """La même projection que l'analyse de l'étape 2, vérifiée là-bas à quelques pixels près :
    caméra en convention monde, avant = +X, gauche = +Y, haut = +Z."""

    def __init__(self, K, largeur: int, hauteur: int):
        self.K, self.l, self.h = np.asarray(K, float), largeur, hauteur

    def pixels(self, cam_pos, R, points):
        c = (np.atleast_2d(points) - cam_pos) @ R
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.K[0, 2] - self.K[0, 0] * c[:, 1] / c[:, 0]
            v = self.K[1, 2] - self.K[1, 1] * c[:, 2] / c[:, 0]
        return np.stack([u, v], axis=1), c[:, 0]

    def dedans(self, uv, marge: float = 0.0) -> np.ndarray:
        return ((uv[:, 0] >= -marge) & (uv[:, 0] < self.l + marge)
                & (uv[:, 1] >= -marge) & (uv[:, 1] < self.h + marge))


def rotation(cam_quat_wxyz) -> np.ndarray:
    w, x, y, z = cam_quat_wxyz
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def quat_wxyz(lacet_rad: float, tangage_rad: float, roulis_rad: float) -> np.ndarray:
    R = Rotation.from_euler("ZYX", [lacet_rad, -tangage_rad, roulis_rad]).as_matrix()
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def boites_cartons(stage, chemins) -> list[dict]:
    """Boîte de chaque carton gardé, en repère monde, par ses huit coins transformés."""
    from pxr import Gf, UsdGeom

    from swarm_qr.env.qr_tags import _local_bounds

    xcache = UsdGeom.XformCache()
    out = []
    for chemin in chemins:
        prim = stage.GetPrimAtPath(chemin)
        lo, hi = _local_bounds(prim)
        m = xcache.GetLocalToWorldTransform(prim)
        coins = np.array([list(m.Transform(Gf.Vec3d(x, y, z)))
                          for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        out.append({"chemin": chemin, "lo": coins.min(axis=0), "hi": coins.max(axis=0)})
    return out


def panneaux(tags) -> list[dict]:
    """Le code lui-même (sans la marge blanche du panneau), comme les décodeurs le rendent."""
    out = []
    for t in tags:
        n = np.array(t.normal, float)
        centre = np.array(t.position, float)
        u = np.cross([0.0, 0.0, 1.0], n)
        u /= max(np.linalg.norm(u), 1e-9)
        demi = P.taille_code(t.size) / 2.0
        out.append({"code": t.tag_id, "carton": t.box_path, "centre": centre, "normale": n,
                    "u": u, "demi": demi, "cote": 2 * demi})
    return out


def grille(centre, axe_a, axe_b, demi_a, demi_b, n: int = GRILLE) -> np.ndarray:
    s = np.linspace(-1.0, 1.0, n)
    return np.array([centre + a * demi_a * axe_a + b * demi_b * axe_b for a in s for b in s])


def faces_visibles(boite, cam_pos) -> list[np.ndarray]:
    """Échantillons sur les faces du carton tournées vers la caméra, un peu en retrait des
    arêtes pour ne pas tomber à côté du collider."""
    lo, hi = boite["lo"], boite["hi"]
    centre, demi = (lo + hi) / 2.0, (hi - lo) / 2.0
    out = []
    for axe in range(3):
        for signe in (-1.0, 1.0):
            c = centre.copy()
            c[axe] += signe * demi[axe]
            if (cam_pos[axe] - c[axe]) * signe <= 0.0:
                continue
            a, b = [k for k in range(3) if k != axe]
            ea, eb = np.eye(3)[a], np.eye(3)[b]
            out.append(grille(c, ea, eb, 0.94 * demi[a], 0.94 * demi[b]))
    return out


# ---------------------------------------------------------------- physique

class Physique:
    def __init__(self):
        from omni.physx import get_physx_scene_query_interface

        self.q = get_physx_scene_query_interface()

    def visible(self, origine, cible, proprietaire: str = "") -> bool:
        """Rien ne s'interpose si le premier objet touché est le propriétaire du point visé,
        ou si la distance touchée est celle du point. Le collider d'un carton est plus petit
        que sa forme visible (mesuré : 10 cm de moins sur un rayon oblique), donc la distance
        seule déclarait cachés des panneaux parfaitement visibles."""
        v = np.asarray(cible, float) - origine
        d = float(np.linalg.norm(v))
        if d < 0.05:
            return False
        h = self.q.raycast_closest(origine.tolist(), (v / d).tolist(), d + 0.5)
        if not h["hit"]:
            return False
        if proprietaire and str(h.get("collision", "")).startswith(proprietaire):
            return True
        return abs(float(h["distance"]) - d) <= TOL_RAYON + 0.01 * d

    def touche(self, origine, direction, portee: float):
        h = self.q.raycast_closest(np.asarray(origine, float).tolist(),
                                   np.asarray(direction, float).tolist(), float(portee))
        return (float(h["distance"]), str(h.get("collision", ""))) if h["hit"] else (None, "")


def controle_colliders(stage, physique, gardes, cartons) -> None:
    """Deux vérités physiques. Les cartons non retenus doivent avoir disparu de la scène,
    sinon les rayons s'arrêteraient sur des cartons que la caméra ne voit pas. Et les cartons
    gardés doivent, eux, arrêter les rayons, sinon tout serait déclaré visible."""
    restants = set(scene_mod._find_boxes(stage))
    if restants != set(gardes):
        raise RuntimeError(f"{len(restants - set(gardes))} cartons non retenus sont encore dans la scene")
    touches, testes = 0, cartons[:40]
    for b in testes:
        haut = np.array([(b["lo"][0] + b["hi"][0]) / 2, (b["lo"][1] + b["hi"][1]) / 2, b["hi"][2]])
        d, qui = physique.touche(haut + [0.0, 0.0, 0.10], [0.0, 0.0, -1.0], 0.30)
        touches += int(d is not None and abs(d - 0.10) < 0.03)
    print(f"[VERIF] cartons non retenus dans la scene : 0 ; cartons gardes solides : "
          f"{touches}/{len(testes)}")
    if touches < 0.9 * len(testes):
        raise RuntimeError("des cartons gardes n'arretent pas les rayons")


# ---------------------------------------------------------------- annotation

class Annoteur:
    def __init__(self, projecteur: Projecteur, physique: Physique, cartons, panneaux):
        self.pr, self.ph, self.cartons, self.panneaux = projecteur, physique, cartons, panneaux

    def _objet(self, classe, cam_pos, R, echantillons, centre, extra, proprietaire: str):
        """Cadre des échantillons visibles ; None si l'objet est hors champ ou derrière."""
        pts = np.concatenate(echantillons)
        uv, prof = self.pr.pixels(cam_pos, R, pts)
        devant = prof > 0.05
        if not devant.any():
            return None
        dans = devant & self.pr.dedans(uv, marge=2.0)
        if not dans.any():
            return None
        vis = np.zeros(len(pts), dtype=bool)
        for i in np.flatnonzero(dans):
            vis[i] = self.ph.visible(cam_pos, pts[i], proprietaire)
        part = float(vis.sum() / max(len(pts), 1))
        if not vis.any():
            return None
        # Objet entier : les échantillons extrêmes sont les bords (le carton est échantillonné
        # 3 % en retrait, on le rend). Objet partiel : le vrai bord visible est quelque part
        # entre deux échantillons, on élargit d'un demi-pas.
        etendue = uv[devant].max(axis=0) - uv[devant].min(axis=0)
        if part >= 0.98:
            marge = 0.03 * etendue if classe == "carton" else np.zeros(2)
        else:
            marge = np.minimum(etendue / (GRILLE - 1) / 2.0, 12.0)
        x0, y0 = np.clip(uv[vis].min(axis=0) - marge, 0, [self.pr.l, self.pr.h])
        x1, y1 = np.clip(uv[vis].max(axis=0) + marge, 0, [self.pr.l, self.pr.h])
        cote = min(x1 - x0, y1 - y0)
        garde = part >= SEUIL_VISIBLE[classe] and cote >= COTE_MIN_PX[classe]
        return {"classe": classe, "bbox": [round(float(v), 1) for v in (x0, y0, x1, y1)],
                "distance": round(float(np.linalg.norm(centre - cam_pos)), 3),
                "visible": round(part, 3), "ignore": not garde, **extra}

    def annote(self, cam_pos, cam_quat) -> list[dict]:
        cam_pos = np.asarray(cam_pos, float)
        R = rotation(cam_quat)
        avant = R[:, 0]
        objets = []
        for p in self.panneaux:
            v = p["centre"] - cam_pos
            d = float(np.linalg.norm(v))
            if d > PORTEE_MAX or np.dot(v, avant) <= 0.0 or np.dot(p["normale"], v) >= 0.0:
                continue                       # loin, derrière la caméra, ou vu de dos
            inc = math.degrees(math.acos(min(1.0, -float(np.dot(p["normale"], v)) / d)))
            ech = [grille(p["centre"], p["u"], np.array([0.0, 0.0, 1.0]), p["demi"], p["demi"])]
            o = self._objet("qr", cam_pos, R, ech, p["centre"],
                            {"code": p["code"], "incidence": round(inc, 1), "cote_m": p["cote"]},
                            p["carton"])
            if o is not None:
                objets.append(o)
        for b in self.cartons:
            centre = (b["lo"] + b["hi"]) / 2.0
            v = centre - cam_pos
            if np.linalg.norm(v) > PORTEE_MAX or np.dot(v, avant) <= 0.0:
                continue
            ech = faces_visibles(b, cam_pos)
            if not ech:
                continue
            o = self._objet("carton", cam_pos, R, ech, centre, {"carton": b["chemin"]}, b["chemin"])
            if o is not None:
                objets.append(o)
        return objets


# ---------------------------------------------------------------- tirage des poses

def dans_un_rack(x, y, layout, marge) -> bool:
    for r in layout.racks:
        (x0, x1), (y0, y1) = r.x_bounds, r.y_bounds
        if x0 - marge < x < x1 + marge and y0 - marge < y < y1 + marge:
            return True
    for ox0, ox1, oy0, oy1, _ in OBSTACLES:
        if ox0 - marge < x < ox1 + marge and oy0 - marge < y < oy1 + marge:
            return True
    return False


def tire_pose(rng, layout):
    for _ in range(1000):
        x = rng.uniform(INTERIOR.x_min + MARGE_MUR, INTERIOR.x_max - MARGE_MUR)
        y = rng.uniform(INTERIOR.y_min + MARGE_MUR, INTERIOR.y_max - MARGE_MUR)
        z = rng.uniform(Z_MIN, Z_MAX)
        if dans_un_rack(x, y, layout, MARGE_RACK):
            continue
        if rng.random() < PART_VISE_RACK:
            rack = min(layout.racks, key=lambda r: abs(r.x - x))
            lacet = 0.0 if rack.x > x else math.pi
            lacet += math.radians(rng.normal(0.0, LACET_SIGMA_DEG))
        else:
            lacet = rng.uniform(-math.pi, math.pi)
        tangage = math.radians(float(np.clip(rng.normal(0, TANGAGE_SIGMA_DEG), -TANGAGE_MAX_DEG, TANGAGE_MAX_DEG)))
        roulis = math.radians(float(np.clip(rng.normal(0, ROULIS_SIGMA_DEG), -ROULIS_MAX_DEG, ROULIS_MAX_DEG)))
        return np.array([x, y, z]), quat_wxyz(lacet, tangage, roulis)
    raise RuntimeError("aucune pose libre trouvee")


# ---------------------------------------------------------------- écriture

class Jeu:
    def __init__(self, nom: str, seed: int):
        self.dossier = SORTIE / nom
        (self.dossier / "images").mkdir(parents=True, exist_ok=True)
        (self.dossier / "labels").mkdir(parents=True, exist_ok=True)
        self.manifeste = (self.dossier / "manifeste.jsonl").open("w", buffering=1)
        self.seed, self.nom, self.n = seed, nom, 0

    def ecrit(self, image_bgr, source: Path | None, cam_pos, cam_quat, objets, largeur, hauteur, extra=None):
        base = f"{self.seed}_{self.n:05d}"
        cible = self.dossier / "images" / f"{base}.jpg"
        if image_bgr is not None:
            _img.save(image_bgr, cible)
        elif source is not None and source.resolve() != cible.resolve():
            if cible.is_symlink() or cible.exists():
                cible.unlink()
            cible.symlink_to(source.resolve())
        lignes = []
        for o in objets:
            if o["ignore"]:
                continue
            x0, y0, x1, y1 = o["bbox"]
            lignes.append(f"{CLASSES.index(o['classe'])} {(x0 + x1) / 2 / largeur:.6f} "
                          f"{(y0 + y1) / 2 / hauteur:.6f} {(x1 - x0) / largeur:.6f} {(y1 - y0) / hauteur:.6f}")
        (self.dossier / "labels" / f"{base}.txt").write_text("\n".join(lignes) + ("\n" if lignes else ""))
        self.manifeste.write(json.dumps({
            "image": f"{base}.jpg", "jeu": self.nom, "seed": self.seed,
            "cam_pos": [round(float(v), 4) for v in cam_pos],
            "cam_quat": [round(float(v), 5) for v in cam_quat],
            "objets": objets, **(extra or {})}) + "\n")
        self.n += 1

    def clot(self, K, largeur, hauteur):
        self.manifeste.close()
        (self.dossier / "meta.json").write_text(json.dumps({
            "seed": self.seed, "images": self.n, "classes": list(CLASSES),
            "largeur": largeur, "hauteur": hauteur,
            "K": [[round(float(v), 4) for v in r] for r in K]}, indent=2))
        print(f"  {self.nom} : {self.n} images annotees")


# ---------------------------------------------------------------- programme

def main() -> None:
    layout = make_layout(args.seed)
    scene = scene_mod.build(layout, with_sitl=False, n_drones=0)
    stage = omni.usd.get_context().get_stage()

    pos0 = np.array([0.0, 0.0, 2.0])
    cam = Camera(prim_path="/World/BancCam", translation=pos0, orientation=quat_wxyz(0, 0, 0),
                 resolution=(CAMERAS.side_width, CAMERAS.side_height))
    scene.world.reset()
    cam.initialize()
    cam.set_focal_length(CAMERAS.focal_length)
    cam.set_horizontal_aperture(CAMERAS.horizontal_aperture)
    cam.set_clipping_range(CAMERAS.near, CAMERAS.far)
    omni.timeline.get_timeline_interface().play()
    for _ in range(5):
        scene.world.step(render=True)

    K = np.asarray(cam.get_intrinsics_matrix(), dtype=float)
    fx_attendu = CAMERAS.side_width / (2.0 * math.tan(math.radians(CAMERAS.fov_deg) / 2.0))
    if abs(K[0, 0] - fx_attendu) > 1.0:
        raise RuntimeError(f"calibration incoherente : fx={K[0, 0]:.1f}, attendu {fx_attendu:.1f}")
    print(f"[VERIF] calibration fx = {K[0, 0]:.1f} (attendu {fx_attendu:.1f})")

    physique = Physique()
    cartons = boites_cartons(stage, scene.kept_boxes)
    pans = panneaux(scene.tags)
    controle_colliders(stage, physique, scene.kept_boxes, cartons)

    # contrôle d'aplomb : un panneau vu de face, à 2 m, doit se projeter au centre de l'image
    # avec le bon côté en pixels. On cherche un panneau dégagé (la caméra à 2 m ne doit pas
    # être dans un rack, et rien ne doit le cacher) : un panneau caché ne prouverait rien.
    pr = Projecteur(K, CAMERAS.side_width, CAMERAS.side_height)
    ann = Annoteur(pr, physique, cartons, pans)
    aplomb = None
    for p0 in sorted(pans, key=lambda p: abs(p["centre"][2] - 1.6))[:40]:
        cam_pos = p0["centre"] + p0["normale"] * 2.0
        if dans_un_rack(cam_pos[0], cam_pos[1], layout, 0.3):
            continue
        q = quat_wxyz(math.atan2(-p0["normale"][1], -p0["normale"][0]), 0.0, 0.0)
        objs = [o for o in ann.annote(cam_pos, q) if o.get("code") == p0["code"]]
        if objs and objs[0]["visible"] >= 0.95:
            aplomb = (p0, cam_pos, q, objs[0])
            break
    if aplomb is None:
        raise RuntimeError("aplomb : aucun panneau degage trouve parmi 40 candidats")
    p0, cam_pos, q, o = aplomb
    uv, _ = pr.pixels(cam_pos, rotation(q), p0["centre"][None, :])
    cote_px = K[0, 0] * p0["cote"] / 2.0
    x0, y0, x1, y1 = o["bbox"]
    if abs(uv[0, 0] - K[0, 2]) > 2 or abs(uv[0, 1] - K[1, 2]) > 2:
        raise RuntimeError(f"aplomb : projection {uv[0]} pour le centre de l'image")
    if abs((x1 - x0) - cote_px) > 0.08 * cote_px:
        raise RuntimeError(f"aplomb : cadre {x1 - x0:.0f} px pour {cote_px:.0f} attendus")
    print(f"[VERIF] aplomb : panneau {p0['code']} a 2 m, cadre {x1 - x0:.0f} px "
          f"(attendu {cote_px:.0f}), visible {o['visible']:.2f}")

    # --- diagnostic d'une pose : pourquoi tel panneau a ou n'a pas de cadre ---
    if args.debug:
        nom_jeu, nom_image = args.debug.split(":")
        m = next(json.loads(l) for l in (SORTIE / nom_jeu / "manifeste.jsonl").read_text().splitlines()
                 if json.loads(l)["image"] == nom_image)
        cam_pos, R = np.array(m["cam_pos"]), rotation(m["cam_quat"])
        print(f"\n[DEBUG] {args.debug} : camera {cam_pos.round(2)}, avant {R[:, 0].round(2)}")
        for p in sorted(pans, key=lambda p: np.linalg.norm(p["centre"] - cam_pos)):
            v = p["centre"] - cam_pos
            d = float(np.linalg.norm(v))
            if d > 4.0 and p["code"] != m.get("cible"):
                continue
            pts = grille(p["centre"], p["u"], np.array([0.0, 0.0, 1.0]), p["demi"], p["demi"])
            uv, prof = pr.pixels(cam_pos, R, pts)
            dans = (prof > 0.05) & pr.dedans(uv, marge=2.0)
            detail = ""
            n_vis = 0
            for i in np.flatnonzero(dans):
                if physique.visible(cam_pos, pts[i], p["carton"]):
                    n_vis += 1
                elif not detail:
                    w = pts[i] - cam_pos
                    dd = float(np.linalg.norm(w))
                    hd, qui = physique.touche(cam_pos, w / dd, dd + 0.5)
                    detail = f" | 1er echantillon cache : attendu {dd:.3f} m, touche {hd} m sur {qui[-60:]}"
            print(f"  {p['code']} d={d:.2f} m face={'oui' if np.dot(p['normale'], v) < 0 else 'NON (dos)'} "
                  f"devant={'oui' if np.dot(v, R[:, 0]) > 0 else 'NON'} dans_image={int(dans.sum())}/25 "
                  f"visibles={n_vis} uv_centre={uv[12].round(0)}{detail}")

    # --- ré-annotation d'un jeu déjà rendu : mêmes images, cadres recalculés ---
    if args.relabel_rendu:
        nom = f"rendu_{args.seed}"
        anciens = [json.loads(l) for l in (SORTIE / nom / "manifeste.jsonl").read_text().splitlines() if l.strip()]
        jeu = Jeu(nom, args.seed)
        for m in anciens:
            objets = ann.annote(m["cam_pos"], m["cam_quat"])
            jeu.ecrit(None, SORTIE / nom / "images" / m["image"], m["cam_pos"], m["cam_quat"], objets,
                      CAMERAS.side_width, CAMERAS.side_height)
        jeu.clot(K, CAMERAS.side_width, CAMERAS.side_height)

    # --- images neuves ---
    if args.images > 0:
        rng = np.random.default_rng(args.graine if args.graine is not None else args.seed)
        jeu = Jeu(f"rendu_{args.seed}", args.seed)
        t0 = time.perf_counter()
        for i in range(args.images):
            cam_pos, q = tire_pose(rng, layout)
            cam.set_world_pose(position=cam_pos, orientation=q, camera_axes="world")
            img = scene_mod.capture_camera(scene.world, cam, settle=RENDUS_PAR_IMAGE,
                                           physique_entre_rendus=0)
            p_reel, q_reel = cam.get_world_pose()
            objets = ann.annote(np.asarray(p_reel, float), np.asarray(q_reel, float))
            jeu.ecrit(_img.to_bgr(img), None, p_reel, q_reel, objets,
                      CAMERAS.side_width, CAMERAS.side_height)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{args.images} images, {(time.perf_counter() - t0) / (i + 1):.2f} s/image")
        jeu.clot(K, CAMERAS.side_width, CAMERAS.side_height)

    # --- images de l'étape 2, cadres calculés depuis les poses relues ---
    for nom in [s for s in args.relabel.split(",") if s]:
        meta = json.loads((ETAPE2 / f"meta_{nom}.json").read_text())
        if meta["seed_entrepot"] != args.seed:
            raise RuntimeError(f"{nom} vient de l'entrepot {meta['seed_entrepot']}, pas {args.seed}")
        lignes = [json.loads(l) for l in (ETAPE2 / f"poses_{nom}.jsonl").read_text().splitlines() if l.strip()]
        K2 = np.array(meta["K"], float)
        if np.abs(K2 - K).max() > 1.0:
            raise RuntimeError(f"{nom} : calibration differente de celle du banc")
        par_run = {}
        for m in lignes:
            par_run.setdefault(m["run"], {})[m["indice"]] = m
        retard = 1 if nom == "traversee" else 0
        jeu = Jeu(f"etape2_{nom}", args.seed)
        sautees = 0
        for m in lignes:
            if not m.get("image"):
                continue
            ref = par_run[m["run"]].get(m["indice"] - retard)
            if ref is None:
                sautees += 1
                continue
            objets = ann.annote(ref["cam_pos"], ref["cam_quat"])
            if meta.get("sans_qr"):
                objets = [o for o in objets if o["classe"] != "qr"]
            jeu.ecrit(None, ETAPE2 / f"images_{nom}" / m["image"], ref["cam_pos"], ref["cam_quat"],
                      objets, CAMERAS.side_width, CAMERAS.side_height,
                      {"cible": m["tag"], "D_m": m["D_m"], "alpha_deg": m["alpha_deg"]})
        jeu.clot(K, CAMERAS.side_width, CAMERAS.side_height)
        if sautees:
            print(f"  {nom} : {sautees} images sans pose de reference (retard), sautees")


try:
    main()
    print("RENDU FINI")
except Exception:
    traceback.print_exc()
finally:
    simulation_app.close()
