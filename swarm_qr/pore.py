"""La méthode de Pore, Patle et Thorat (Symmetry 2026, 18(4):548), portée sur notre simulateur.

Ce module est **autonome** : il n'importe ni `mapping`, ni `planning`, ni `observation`. Il ne
reçoit que ce que les auteurs donnent à leur système, et rien de plus :

1. le plan de l'entrepôt, connu avant le décollage — chez eux les arrêts sont donnés « en
   coordonnées de l'entrepôt », et le champ de risque est bâti « à partir de la carte 3D
   disponible » (§2.1 et §2.3) ;
2. un lidar 3D embarqué, qui rafraîchit le champ de risque ;
3. une caméra frontale, dont le flux est décodé au sol par `cv2.QRCodeDetector`, précédé d'un
   passage en niveaux de gris et d'un seuillage adaptatif (§2.4) ;
4. un score de confiance par lecture, et une relecture programmée quand il est bas (§2.4).

Ce qu'il n'a pas, parce que les auteurs ne l'ont pas : notre carte construite en vol, notre
détecteur appris de cartons, nos pistes, nos frontières, notre formule de décision.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .env.config import INTERIOR, RACKS

# --- valeurs de l'article -------------------------------------------------------------------
R_SAFE = 0.5             # m ; §2.3 point 4 : « we used r_safe = 0.5 m »
BETA = 10.0              # §2.3 point 4 : « β ∈ [8, 12] m^-1 »
TAU = 0.4                # §2.3 point 4 : « τ = 0.4 »
SEPARATION_MIN = 1.2     # m ; cas II : « s(t) ≥ s_min = 1.2 m »
CONFLIT_S = 3.0          # s ; cas II : le suiveur s'arrête si le conflit prévu arrive avant
DEDUP_S = 1.5            # s ; §3.1 : « dedup dwell ∆t_dedup (e.g., 1.5 s) »
DEFLEXION_MAX = 1.2      # m ; §2.3 point 5 : la correction reste « proche du chemin nominal »

# --- notre transposition, expliquée une par une ---------------------------------------------
RECUL = 1.65             # m de la face. L'article dit seulement que le drone « garde une distance
                         # constante des étagères », sans chiffre. On prend celle qui sert LEUR
                         # décodeur : OpenCV lit 92 % entre 1,5 et 1,8 m, puis 79 % à 2,2 m et 40 %
                         # à 3 m (banc de l'étape 2). Le nôtre, zbar, tient 96 % jusqu'à 4 m ; lui
                         # donner notre recul les pénaliserait pour un choix qui n'est pas le leur.
RECUL_MIN = 0.9          # en dessous, l'allée est déclarée inaccessible
MARGE_OPPOSEE = 0.85     # m gardés avec le rack ou le mur d'en face
HAUTEUR_PANNEAU = 0.225  # le code est au milieu de la face d'un carton posé sur l'étagère
PAS_ARRET = 1.5          # m entre deux arrêts « hover-and-scan » le long d'une face
BORD = 0.3               # m non parcourus aux deux bouts d'un rack
MARGE_COULOIR = 1.2      # m entre le bout des racks et le couloir de traversée
CONFIANCE_MIN = 0.35     # sous ce score, l'arrêt est reprogrammé (l'article le prévoit sans chiffre)


# ============================================================== le plan de couverture

@dataclass
class Arret:
    """Un point « hover-and-scan » : où se tenir, vers où regarder, à quelle face il appartient."""
    position: np.ndarray
    cap: float
    face: str
    etage: int
    vise: np.ndarray

    def cle(self) -> tuple:
        return (self.face, self.etage, round(float(self.position[1]), 2))


def allees(layout) -> list[dict]:
    """Les bandes libres entre les murs et les racks, lues sur le plan connu."""
    racks = sorted(layout.racks, key=lambda r: r.x)
    bornes = [INTERIOR.x_min] + [x for r in racks for x in r.x_bounds] + [INTERIOR.x_max]
    out = []
    for k in range(len(racks) + 1):
        gauche, droite = bornes[2 * k], bornes[2 * k + 1]
        faces = []
        if k > 0:
            faces.append((racks[k - 1], +1))
        if k < len(racks):
            faces.append((racks[k], -1))
        out.append({"k": k, "x0": gauche, "x1": droite, "centre": 0.5 * (gauche + droite), "faces": faces})
    return out


def faces_lisibles(layout) -> list[dict]:
    """Les faces de rack qu'un drone peut longer : l'allée doit être assez large."""
    out = []
    for a in allees(layout):
        recul = min(RECUL, (a["x1"] - a["x0"]) - MARGE_OPPOSEE)
        if recul < RECUL_MIN:
            continue
        for rack, cote in a["faces"]:
            out.append({"nom": f"{rack.prim}:{'est' if cote > 0 else 'ouest'}", "allee": a["k"],
                        "x_drone": rack.x + cote * (RACKS.depth / 2.0 + recul),
                        "x_face": rack.x + cote * RACKS.depth / 2.0,
                        "y0": rack.y_bounds[0] + BORD, "y1": rack.y_bounds[1] - BORD,
                        "rack": rack.prim, "cote": cote})
    return out


def _ys(face: dict) -> np.ndarray:
    return np.arange(face["y0"], face["y1"] + 1e-6, PAS_ARRET)


def _coupe_en_deux(face: dict) -> list[dict]:
    """Couper un rack en deux moitiés confiées à deux drones qui partent des bouts opposés :
    c'est le cas II de l'article, « two UAVs scanning the same rack from opposite ends »."""
    milieu = 0.5 * (face["y0"] + face["y1"])
    return [{**face, "y1": milieu}, {**face, "y0": milieu}]


def secteurs(faces: list[dict], n_drones: int) -> list[list[dict]]:
    """Le répartiteur de secteurs : **un rack par drone** quand il y en a autant, ce qui est le
    cas III de l'article, « a sector allocator assigns each vehicle a disjoint rack ». S'il y a
    plus de racks que de drones, un drone prend des racks voisins ; s'il y en a moins, le rack
    le plus chargé est coupé en deux moitiés, cas II."""
    groupes: list[list[dict]] = []
    for f in faces:
        if groupes and groupes[-1][0]["rack"] == f["rack"]:
            groupes[-1].append(f)
        else:
            groupes.append([f])

    def charge(g):
        return sum(len(_ys(f)) for f in g)

    while len(groupes) > n_drones and len(groupes) > 1:
        k = min(range(len(groupes) - 1), key=lambda k: charge(groupes[k]) + charge(groupes[k + 1]))
        groupes[k:k + 2] = [groupes[k] + groupes[k + 1]]
    while len(groupes) < n_drones and groupes:
        k = max(range(len(groupes)), key=lambda k: charge(groupes[k]))
        lourd = groupes.pop(k)
        moities = [[], []]
        for f in lourd:
            for m, moitie in zip(moities, _coupe_en_deux(f)):
                m.append(moitie)
        groupes[k:k] = moities
    return groupes + [[] for _ in range(n_drones - len(groupes))]


def plan_zigzag(layout, n_drones: int, etages=(0, 1, 2)) -> list[list[Arret]]:
    """« The UAV follows a predefined zig-zag coverage path with hover-and-scan stops at
    waypoints (x, y, z) in the warehouse frame » (§2.1). Une face est parcourue sur toute sa
    longueur à un étage, puis la hauteur monte d'un cran et le sens s'inverse : c'est le
    hover–scan–advance, « the orange staircase is the discrete shelf-level command z(t) ».
    Les drones de rang impair partent du bout opposé — les serpentins en miroir du §2.5."""
    plans: list[list[Arret]] = []
    for d, bloc in enumerate(secteurs(faces_lisibles(layout), n_drones)):
        arrets: list[Arret] = []
        sens = 1 if d % 2 == 0 else -1
        for f in bloc:
            for etage in etages:
                z = RACKS.shelf_levels[etage] + HAUTEUR_PANNEAU
                for y in _ys(f)[::sens]:
                    p = np.array([f["x_drone"], float(y), z])
                    vise = np.array([f["x_face"], float(y), z])
                    arrets.append(Arret(p, math.atan2(0.0, f["x_face"] - f["x_drone"]), f["nom"], etage, vise))
                sens = -sens
        plans.append(arrets)
    return plans


def attribue(plans: list[list[Arret]], departs) -> list[list[Arret]]:
    """Chaque drone prend le secteur dont le premier arrêt est le plus proche de lui. Chez les
    auteurs les drones sont déjà postés à leur rack au départ (figure 8) ; ici ils naissent où
    la scène les place."""
    libres = list(range(len(plans)))
    out: list[list[Arret]] = [[] for _ in plans]
    for i, p in enumerate(departs):
        if not libres:
            break
        k = min(libres, key=lambda k: float(np.linalg.norm(plans[k][0].position - np.asarray(p)))
                if plans[k] else float("inf"))
        out[i] = plans[k]
        libres.remove(k)
    return out


# ============================================================== le champ de risque

def _distance_boite(p, x0: float, x1: float, y0: float, y1: float) -> float:
    dx = max(x0 - p[0], 0.0, p[0] - x1)
    dy = max(y0 - p[1], 0.0, p[1] - y1)
    return math.hypot(dx, dy)


class ChampDeRisque:
    """« A signed-distance-derived, sigmoid Risk Factor provides continuous risk values and
    gradients everywhere in free space » (§2.3 point 4). La distance signée vient de la carte 3D
    connue — murs et racks — et des retours lidar récents, qui sont le seul apport en vol.

    Les retours lidar ne sont gardés que quelques secondes : c'est une couche de sécurité à
    court horizon, pas une carte qui se souvient."""

    MEMOIRE_S = 3.0

    def __init__(self, layout, memoire_s: float = MEMOIRE_S):
        self.racks = [(r.x_bounds[0], r.x_bounds[1], r.y_bounds[0], r.y_bounds[1]) for r in layout.racks]
        self.memoire_s = memoire_s
        self.echos: list[tuple[float, np.ndarray]] = []
        self.mobiles: list[np.ndarray] = []

    def integre_lidar(self, origine, directions, portees, t: float) -> None:
        o = np.asarray(origine, float)
        d = np.asarray(directions, float)
        r = np.asarray(portees, float)
        vus = np.isfinite(r) & (r > 0.4) & (r < 25.0)
        if vus.any():
            self.echos.append((t, o + d[vus] * r[vus, None]))
        self.echos = [(u, p) for u, p in self.echos if t - u <= self.memoire_s]

    def annonce_coequipiers(self, positions) -> None:
        self.mobiles = [np.asarray(p, float) for p in positions]

    def distance(self, p) -> float:
        """Distance libre au plus proche obstacle : murs, racks, échos lidar, coéquipiers."""
        p = np.asarray(p, float)
        d = min(p[0] - INTERIOR.x_min, INTERIOR.x_max - p[0],
                p[1] - INTERIOR.y_min, INTERIOR.y_max - p[1])
        for x0, x1, y0, y1 in self.racks:
            d = min(d, _distance_boite(p, x0, x1, y0, y1))
        for _, pts in self.echos:
            e = pts - p
            loin = np.abs(e[:, 2]) < 1.0                    # un écho au sol ou au plafond ne gêne pas
            if loin.any():
                d = min(d, float(np.linalg.norm(e[loin, :2], axis=1).min()))
        for q in self.mobiles:
            d = min(d, float(np.linalg.norm(q[:2] - p[:2])))
        return float(d)

    def risque(self, p) -> float:
        """R_f(p) = σ(β (r_safe − r(p))), équation 9."""
        return 1.0 / (1.0 + math.exp(-BETA * (R_SAFE - self.distance(p))))

    def sur(self, p) -> bool:
        return self.risque(p) <= TAU

    def gradient(self, p, pas: float = 0.15) -> np.ndarray:
        """Le gradient du risque dans le plan, par différences finies."""
        p = np.asarray(p, float)
        g = np.zeros(3)
        for k in (0, 1):
            e = np.zeros(3)
            e[k] = pas
            g[k] = (self.risque(p + e) - self.risque(p - e)) / (2 * pas)
        return g


def deflexion(p, champ: ChampDeRisque, dans_allee=None, pas: float = 0.25,
              maxi: float = DEFLEXION_MAX) -> np.ndarray | None:
    """« When a predicted segment violates safety, we locally correct frames by minimally
    deflecting the nominal path while penalizing risk » (§2.3 point 5). On descend le gradient
    du risque par petits pas, sans s'éloigner du point nominal de plus de `maxi`, et sans
    sortir de l'allée. Rend None si aucun point sûr n'est trouvé."""
    p0 = np.asarray(p, float)
    q = p0.copy()
    for _ in range(int(maxi / pas) + 1):
        if champ.sur(q) and (dans_allee is None or dans_allee(q)):
            return q
        g = champ.gradient(q)
        n = float(np.linalg.norm(g[:2]))
        if n < 1e-9:
            return None
        q = q - pas * g / n
        if float(np.linalg.norm(q[:2] - p0[:2])) > maxi:
            return None
    return q if champ.sur(q) and (dans_allee is None or dans_allee(q)) else None


# ============================================================== le routage dans les allées

class Routeur:
    """« Linear keep-in constraints that restrict motion to the aisle polytope » (§2.3). Le
    chemin nominal reste dans les allées : on descend l'allée de départ jusqu'à un couloir de
    traversée, on change d'allée, on remonte. Aucune carte découverte n'intervient : toute la
    géométrie vient du plan connu."""

    def __init__(self, layout):
        self.allees = allees(layout)
        self.racks = [(r.x_bounds[0], r.x_bounds[1], r.y_bounds[0], r.y_bounds[1]) for r in layout.racks]
        y0 = min(r[2] for r in self.racks) - MARGE_COULOIR
        y1 = max(r[3] for r in self.racks) + MARGE_COULOIR
        self.couloirs = [max(y0, INTERIOR.y_min + MARGE_COULOIR / 2), min(y1, INTERIOR.y_max - MARGE_COULOIR / 2)]

    def allee_de(self, p) -> dict:
        x = float(p[0])
        for a in self.allees:
            if a["x0"] <= x <= a["x1"]:
                return a
        return min(self.allees, key=lambda a: min(abs(x - a["x0"]), abs(x - a["x1"])))

    def coupe_un_rack(self, a, b) -> bool:
        a, b = np.asarray(a, float)[:2], np.asarray(b, float)[:2]
        n = max(int(np.linalg.norm(b - a) / 0.2), 1)
        for k in range(n + 1):
            q = a + (b - a) * k / n
            for x0, x1, y0, y1 in self.racks:
                if x0 <= q[0] <= x1 and y0 <= q[1] <= y1:
                    return True
        return False

    def chemin(self, depart, arrivee) -> list[np.ndarray]:
        """Les points de passage entre deux arrêts, sans le point d'arrivée."""
        a = np.asarray(depart, float)
        b = np.asarray(arrivee, float)
        if not self.coupe_un_rack(a, b):
            return []
        aa, ab = self.allee_de(a), self.allee_de(b)
        candidats = []
        for y in self.couloirs:
            p1 = np.array([aa["centre"], y, b[2]])
            p2 = np.array([ab["centre"], y, b[2]])
            route = [p1, p2]
            if any(self.coupe_un_rack(u, v) for u, v in zip([a] + route, route + [b])):
                continue
            longueur = sum(float(np.linalg.norm(v - u)) for u, v in zip([a] + route, route + [b]))
            candidats.append((longueur, route))
        if not candidats:
            return []
        return min(candidats, key=lambda c: c[0])[1]


# ============================================================== la lecture au sol

def _gris_seuille(bgr: np.ndarray) -> np.ndarray:
    """« Preprocessing: grayscale conversion and adaptive thresholding to mitigate indoor
    lighting variations » (§2.4)."""
    gris = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)


def _nettete(gris: np.ndarray, coins: np.ndarray) -> float:
    """Le score de confiance de l'article : « based on OpenCV decoding probability or secondary
    heuristic (e.g., sharpness of detected QR region) ». On prend la netteté de la zone du code,
    normalisée entre 0 et 1."""
    x0, y0 = np.clip(coins.min(axis=0).astype(int) - 2, 0, None)
    x1, y1 = coins.max(axis=0).astype(int) + 2
    zone = gris[max(y0, 0):min(y1, gris.shape[0]), max(x0, 0):min(x1, gris.shape[1])]
    if zone.size < 16:
        return 0.0
    return float(min(cv2.Laplacian(zone, cv2.CV_64F).var() / 500.0, 1.0))


def decode(bgr: np.ndarray) -> list[tuple[str, float]]:
    """La chaîne du sol : gris, seuillage adaptatif, `cv2.QRCodeDetector`, correction
    Reed–Solomon (incluse dans OpenCV). Rend les codes lus avec leur confiance."""
    gris = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    out: dict[str, float] = {}
    for image in (gris, _gris_seuille(bgr)):
        det = cv2.QRCodeDetector()
        try:
            ok, textes, points, _ = det.detectAndDecodeMulti(image)
        except cv2.error:
            continue
        if not ok or points is None:
            continue
        for texte, coins in zip(textes, points):
            if not texte:
                continue
            c = _nettete(gris, np.asarray(coins).reshape(-1, 2))
            out[texte] = max(out.get(texte, 0.0), c)
    return sorted(out.items())


# ============================================================== l'inventaire au sol

@dataclass
class Lecture:
    code: str
    drone: int
    position: np.ndarray
    t: float
    confiance: float


class Inventaire:
    """Le serveur d'inventaire de l'article : il reçoit les lectures, écarte les doublons dans
    une fenêtre de `DEDUP_S`, et redemande un passage quand la confiance est basse (§2.4)."""

    def __init__(self, dedup_s: float = DEDUP_S, confiance_min: float = CONFIANCE_MIN):
        self.dedup_s, self.confiance_min = dedup_s, confiance_min
        self.evenements: list[Lecture] = []
        self.uniques: dict[str, Lecture] = {}
        self.doublons = 0
        self.a_relire: set[tuple] = set()

    def recoit(self, code: str, drone: int, position, t: float, confiance: float, arret=None) -> bool:
        """Rend vrai si le code entre à l'inventaire pour la première fois."""
        e = Lecture(code, drone, np.asarray(position, float).copy(), t, confiance)
        ancien = self.uniques.get(code)
        if ancien is not None and t - ancien.t < self.dedup_s:
            self.doublons += 1
            return False
        self.evenements.append(e)
        if confiance < self.confiance_min:
            if arret is not None:
                self.a_relire.add(arret.cle())
            if ancien is not None:
                self.doublons += 1
                return False
        if ancien is None:
            self.uniques[code] = e
            return True
        self.doublons += 1
        if confiance > ancien.confiance:
            self.uniques[code] = e
        return False

    @property
    def codes(self) -> set[str]:
        return set(self.uniques)

    def bilan(self) -> dict:
        conf = [e.confiance for e in self.evenements] or [0.0]
        return {"codes_uniques": len(self.uniques), "lectures": len(self.evenements),
                "doublons": self.doublons, "a_relire": len(self.a_relire),
                "confiance_moyenne": round(float(np.mean(conf)), 3),
                "confiance_min": round(float(np.min(conf)), 3)}


# ============================================================== la coordination

def temps_avant_conflit(pa, va, pb, vb, s_min: float = SEPARATION_MIN) -> float:
    """Le « predicted time-to-conflict τ_c(∆x, ∆y, ∆v) » du cas II : dans combien de temps les
    deux drones seront-ils à moins de s_min, s'ils gardent leur vitesse. Infini s'ils s'éloignent."""
    dp = np.asarray(pb, float)[:2] - np.asarray(pa, float)[:2]
    dv = np.asarray(vb, float)[:2] - np.asarray(va, float)[:2]
    a = float(dv @ dv)
    if a < 1e-9:
        return 0.0 if float(dp @ dp) < s_min ** 2 else float("inf")
    b = 2.0 * float(dp @ dv)
    c = float(dp @ dp) - s_min ** 2
    disc = b * b - 4 * a * c
    if disc < 0:
        return float("inf")
    t = (-b - math.sqrt(disc)) / (2 * a)
    if t < 0:
        t = (-b + math.sqrt(disc)) / (2 * a)
    return t if t >= 0 else float("inf")


def cede_le_passage(moi: int, pa, va, autres) -> bool:
    """« A token-based rule at the GCS pauses the follower whenever the predicted
    time-to-conflict is under 3 s. » Le jeton va au plus petit numéro, comme leur couple
    meneur–suiveur."""
    for j, pb, vb in autres:
        if temps_avant_conflit(pa, va, pb, vb) < CONFLIT_S and j < moi:
            return True
    return False


def vitesse_selon_confiance(v_max: float, confiance: float | None,
                            seuil: float = CONFIANCE_MIN, plancher: float = 0.35) -> float:
    """« Speed is modulated by a confidence-aware controller that throttles forward velocity
    when the per-frame decode confidence dips below threshold » (cas III)."""
    if confiance is None or confiance >= seuil:
        return v_max
    return max(plancher, v_max * max(confiance, 0.0) / seuil)
