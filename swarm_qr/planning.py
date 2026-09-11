"""Le cerveau géométrique : où aller ensuite (étape 5).

Trois sortes de cibles, toutes tirées de la carte, jamais du plan de l'entrepôt :
- LIRE : une piste, c'est-à-dire un QR repéré de loin par l'œil appris et pas encore lu ;
- COUVRIR : une surface connue occupée, à hauteur de vol, jamais regardée depuis le côté libre
  d'assez près pour y lire un code — un carton repéré y vaut trois fois plus ;
- EXPLORER : une frontière entre le connu et l'inconnu.

Chaque cible reçoit une note : son utilité, moins le coût du trajet, moins une pénalité si un
coéquipier a réservé l'endroit. Le guide de l'étape 8 peut ajouter un bonus sur une zone et un
côté. Le meilleur candidat est vérifié par un vrai chemin sur la carte avant d'être retenu.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import mapping
from .env.config import CAMERAS

D_LECTURE = 2.0              # m, au milieu de l'enveloppe de lecture 1,5–4 m (étape 2)
UTILITE_LIRE = 30.0
UTILITE_SURFACE = 1.0        # par cube de surface jamais regardé du bon côté...
BONUS_CARTON = 3.0           # ...trois fois plus si l'œil appris y a vu un carton
PLAFOND_COUVRIR = 30.0
UTILITE_FRONTIERE = 0.15     # par case de frontière du groupe
PLAFOND_EXPLORER = 20.0
COUT_METRE = 1.0
PENALITE_RESERVE = 1000.0    # une cible réservée par un autre est exclue, sauf s'il ne reste rien
RAYON_RESERVE = 4.0
PENALITE_VOISIN = 40.0       # une cible à moins de 4 m d'un coéquipier en vol coûte plus qu'une piste
RAYON_VOISIN = 4.0
BONUS_ZONE = 10.0            # avis du guide : la zone conseillée...
BONUS_COTE = 5.0             # ...et le côté d'abordage conseillé
GROUPE_XY = 1.0              # m, taille des groupes de surfaces
GROUPE_Z = 0.5
GROUPE_FRONTIERE = 2.0
SURFACE_MIN = 3              # cubes ; en dessous, c'est du bruit
Z_MIN, Z_MAX = 0.9, 4.5      # bande de vol ; au-dessus, on survole les racks, et leurs panneaux de signalisation sont à 5 m
PISTE_Z_MAX = 5.0            # une piste plus haute que le dernier étage est un fantôme
ALT_MIN = 1.4                # m, altitude minimale d'une pose
ALT_EXPLORATION = 1.8
DEGAGEMENT_MAX = 1.25        # m, jusqu'où reculer une pose prise dans la marge d'un obstacle
MARGE_POSE = 0.9             # m ; une pose tenue oscille : plus loin des obstacles que le trajet (0,6)
CANDIDATS_VERIFIES = 6       # les meilleurs par la ligne droite reçoivent un vrai chemin

CARDINAUX = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
OPPOSE = {0: 1, 1: 0, 2: 3, 3: 2}
NOMS_COTES = {0: "est", 1: "ouest", 2: "nord", 3: "sud"}


@dataclass
class Cible:
    genre: str                  # lire / couvrir / explorer
    position: np.ndarray        # la pose du drone
    cap: float | None           # None : garder le cap courant
    utilite: float
    origine: np.ndarray         # ce qu'on va regarder
    cote: int                   # côté d'où l'on regarde (cardinal de origine → drone)
    distance: float = math.inf
    note: float = -math.inf
    points: list = field(default_factory=list)

    def cle(self) -> tuple:
        return (self.genre, *np.round(self.origine, 1).tolist())


@dataclass
class Avis:
    """Le conseil du guide : une zone (centre, rayon) et un côté d'abordage."""
    centre: np.ndarray
    rayon: float
    cote: int | None = None
    phrase: str = ""


def cap_pour_regarder(direction) -> float:
    """La caméra gauche regarde à 90 degrés à gauche du cap : pour la pointer dans `direction`,
    le cap vaut cette direction moins 90 degrés."""
    return math.atan2(direction[1], direction[0]) - math.pi / 2.0


def _horizontal(v) -> np.ndarray | None:
    h = np.array([v[0], v[1], 0.0], dtype=float)
    n = np.linalg.norm(h)
    return h / n if n > 1e-6 else None


class Cerveau:
    """La décision d'un drone. Sans état sauf le compte des tentatives : deux visites d'une
    piste sans lecture, et la piste est écartée."""

    def __init__(self, carte: mapping.Carte, drone: int, lam: float = 0.0):
        self.carte, self.drone, self.lam = carte, drone, lam
        self.tentatives: list[list] = []          # [position, nombre de visites sans lecture]
        self._couts: dict[int, np.ndarray] = {}

    # ---------------------------------------------------------------- candidats

    CACHE_S = 2.0

    def candidats(self, t: float | None = None) -> list[Cible]:
        """Les candidats sont les mêmes pour tous les drones ; seules les notes diffèrent. Ils
        sont calculés une fois par deux secondes et partagés sur la carte : un calcul par
        drone et par cycle immobiliserait la physique, et le pilote automatique, qui tourne
        dans un autre processus, perd le contrôle du drone quand la simulation s'arrête."""
        cache = getattr(self.carte, "_cache_cibles", None)
        if t is not None and cache is not None and 0.0 <= t - cache[0] < self.CACHE_S:
            return cache[1]
        self._couts = {}
        cibles = self._lire() + self._couvrir() + self._explorer()
        if t is not None:
            self.carte._cache_cibles = (t, cibles)
        return cibles

    def _couts_a(self, z: float) -> np.ndarray:
        k = int(round(z / 0.25))
        if k not in self._couts:
            self._couts[k] = self.carte.couts(z - mapping.DESSOUS, z + mapping.EPAISSEUR, marge=MARGE_POSE)
        return self._couts[k]

    def _praticable(self, p) -> bool:
        if not self.carte.sur_la_carte(p)[0]:
            return False
        i, j, _ = self.carte.indice(p)[0]
        return bool(np.isfinite(self._couts_a(float(p[2]))[i, j]))

    DISTANCES_LECTURE = (2.0, 1.75, 1.5, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0)
    DISTANCES_PRES = (1.2, 1.0, 1.4, 0.9)       # une étiquette de 12 cm se lit vers 1,2 m (étape 7)

    def _pose_de_lecture(self, origine, n, distances=DISTANCES_LECTURE) -> np.ndarray | None:
        """Une pose devant `origine`, du côté `n`, à distance de lecture. On essaie d'abord
        2 m, puis plus près jusqu'à 1,5 m, puis plus loin jusqu'à 4 m : la pose doit être
        praticable et rien de connu ne doit se trouver entre elle et la surface."""
        for d in distances:
            pose = np.asarray(origine, dtype=float) + n * d + np.array([0.0, 0.0, CAMERAS.below])
            pose[2] = float(np.clip(pose[2], ALT_MIN, Z_MAX))
            if not self._praticable(pose):
                continue
            devant = self.carte.premier_obstacle(pose, -n, portee=d + 0.5)
            if devant is not None and devant < d - 0.4:
                continue
            return pose
        return None

    def point_degage(self, position, marge: float) -> np.ndarray | None:
        """Le point praticable le plus proche de `position`, à `marge` de tout obstacle connu, à
        la même altitude ; None s'il n'y en a pas à moins de 3 m."""
        p = np.asarray(position, dtype=float)
        cout = self.carte.couts(p[2] - mapping.DESSOUS, p[2] + mapping.EPAISSEUR, marge=marge)
        pas = self.carte.g.cell
        n = int(3.0 / pas)
        for dx, dy in sorted(((dx, dy) for dx in range(-n, n + 1) for dy in range(-n, n + 1)),
                             key=lambda o: o[0] * o[0] + o[1] * o[1]):
            q = p + np.array([dx * pas, dy * pas, 0.0])
            if not self.carte.sur_la_carte(q)[0]:
                continue
            i, j, _ = self.carte.indice(q)[0]
            if np.isfinite(cout[i, j]):
                return q
        return None

    def _pose_libre(self, centre) -> np.ndarray | None:
        """La pose praticable la plus proche d'un point, à moins de DEGAGEMENT_MAX."""
        c = np.asarray(centre, dtype=float).copy()
        c[2] = float(np.clip(c[2], ALT_MIN, Z_MAX))
        pas = self.carte.g.cell
        n = int(DEGAGEMENT_MAX / pas)
        offsets = sorted(((dx, dy) for dx in range(-n, n + 1) for dy in range(-n, n + 1)),
                         key=lambda o: o[0] * o[0] + o[1] * o[1])
        for dx, dy in offsets:
            if dx * dx + dy * dy > n * n:
                continue
            q = c + np.array([dx * pas, dy * pas, 0.0])
            if self._praticable(q):
                return q
        return None

    def normale_par_la_carte(self, position, indice=None) -> np.ndarray | None:
        """Le côté d'où l'on lit un panneau. Un panneau est collé sur la longue face d'un rack :
        sa normale est perpendiculaire au grand axe de la structure occupée qui l'entoure. Entre
        les deux côtés perpendiculaires, celui où l'espace libre est le plus proche ; à égalité,
        celui qui regarde vers la direction d'aperçu `indice`. La direction d'aperçu seule ne
        suffit pas : un QR vu à 70 degrés de biais se lit de face, pas de biais."""
        c = self.carte
        p = np.asarray(position, dtype=float)
        k0, k1 = c._tranche(p[2] - 0.75, p[2] + 0.75)
        i0, j0, _ = c.indice(p - 3.0)[0]
        i1, j1, _ = c.indice(p + 3.0)[0]
        bloc = c.occupation[max(i0, 0):i1 + 1, max(j0, 0):j1 + 1, k0:k1] > mapping.SEUIL_OCCUPE
        ij = np.argwhere(bloc.any(axis=2)) + [max(i0, 0), max(j0, 0)]
        if len(ij) < 10:
            return None if indice is None else CARDINAUX[c.cardinal(indice)].copy()
        xy = ij * c.g.cell
        xy = xy - xy.mean(axis=0)
        _, vecteurs = np.linalg.eigh(xy.T @ xy)
        axe = vecteurs[:, -1]                                      # le grand axe, horizontal
        perpendiculaires = [k for k, d in enumerate(CARDINAUX) if abs(np.dot(d[:2], axe)) < 0.5]
        if not perpendiculaires:
            return None if indice is None else CARDINAUX[c.cardinal(indice)].copy()
        pas = np.array([[0.25], [0.5], [0.75], [1.0], [1.25], [1.5]])

        def premier_libre(d):
            etats = c.etat(p + d[None, :] * pas)
            libres = np.flatnonzero(etats == mapping.LIBRE)
            return float(pas[libres[0], 0]) if len(libres) else np.inf

        meilleur = min(perpendiculaires, key=lambda k: (premier_libre(CARDINAUX[k]),
                                                        0.0 if indice is None else -float(np.dot(CARDINAUX[k][:2], indice[:2]))))
        return CARDINAUX[meilleur].copy() if np.isfinite(premier_libre(CARDINAUX[meilleur])) else None

    def _cotes_possibles(self, piste) -> list[np.ndarray]:
        """Les côtés d'où lire une piste, du plus probable au moins probable : d'abord celui que
        la structure porteuse impose, puis, pour une seconde tentative, son opposé."""
        apercu = None if piste.normale is None else _horizontal(piste.normale)
        n = self.normale_par_la_carte(piste.position, apercu)
        if n is None:
            return []
        return [n, -n]

    def _lire(self) -> list[Cible]:
        out = []
        for piste in self.carte.pistes:
            if self.carte.est_ecartee(piste.position) or piste.position[2] > PISTE_Z_MAX:
                continue
            cotes = self._cotes_possibles(piste)
            if not cotes:
                continue
            # trois chances : à 2 m ; puis plus près, à 1,2 m, parce que les petites
            # étiquettes ne se lisent pas de loin (étape 7) ; puis de l'autre côté. Ensuite
            # la piste est écartée.
            visites = self._visites(piste.position)
            essais = ([(cotes[0], self.DISTANCES_LECTURE)] if visites == 0 else
                      [(cotes[0], self.DISTANCES_PRES), (cotes[0], self.DISTANCES_LECTURE)] if visites == 1 else
                      [(c, self.DISTANCES_LECTURE) for c in cotes[::-1]])
            pose, n = None, None
            for n, distances in essais:
                pose = self._pose_de_lecture(piste.position, n, distances)
                if pose is not None:
                    break
            if pose is None:
                continue
            out.append(Cible("lire", pose, cap_pour_regarder(-n), UTILITE_LIRE,
                             piste.position.copy(), self.carte.cardinal(n)))
        return out

    def _couvrir(self) -> list[Cible]:
        """Les surfaces occupées dont la case libre d'en face n'a jamais été regardée depuis ce
        côté, groupées par mètre et par demi-mètre de hauteur."""
        c = self.carte
        k0, k1 = c._tranche(Z_MIN, Z_MAX)
        occ = c.occupation[:, :, k0:k1]
        dur = occ > mapping.SEUIL_OCCUPE
        libre = occ < mapping.SEUIL_LIBRE
        couv = c.couverture[:, :, k0:k1]
        groupes: dict[tuple, list] = {}
        for cote, (dx, dy) in enumerate([(1, 0), (-1, 0), (0, 1), (0, -1)]):
            bit = np.uint8(1 << OPPOSE[cote])          # une caméra regardant vers la surface
            # surface en (i, j), case libre en face en (i+dx, j+dy)
            sx = slice(max(-dx, 0), dur.shape[0] - max(dx, 0))
            sy = slice(max(-dy, 0), dur.shape[1] - max(dy, 0))
            fx = slice(max(dx, 0), dur.shape[0] - max(-dx, 0))
            fy = slice(max(dy, 0), dur.shape[1] - max(-dy, 0))
            a_faire = dur[sx, sy, :] & libre[fx, fy, :] & ((couv[fx, fy, :] & bit) == 0)
            idx = np.argwhere(a_faire)
            if not len(idx):
                continue
            idx[:, 0] += sx.start
            idx[:, 1] += sy.start
            idx[:, 2] += k0
            centres = c.centre(idx)
            carton = c.semantique[tuple(idx.T)] == mapping.SEM_CARTON
            cles = np.column_stack([np.full(len(idx), cote), np.floor(centres[:, 0] / GROUPE_XY),
                                    np.floor(centres[:, 1] / GROUPE_XY), np.floor(centres[:, 2] / GROUPE_Z)])
            for cle, p, est_carton in zip(map(tuple, cles.astype(int)), centres, carton):
                groupes.setdefault(cle, []).append((p, est_carton))
        out = []
        for cle, elems in groupes.items():
            if len(elems) < SURFACE_MIN:
                continue
            cote = int(cle[0])
            d = CARDINAUX[cote]
            pts = np.array([p for p, _ in elems])
            centre = pts.mean(axis=0)
            utilite = min(sum(BONUS_CARTON if k else UTILITE_SURFACE for _, k in elems), PLAFOND_COUVRIR)
            pose = self._pose_de_lecture(centre, d)
            if pose is None or self.carte.est_ecartee(pose):
                continue
            out.append(Cible("couvrir", pose, cap_pour_regarder(-d), utilite, centre, cote))
        return out

    def _explorer(self) -> list[Cible]:
        pts = self.carte.frontieres(Z_MIN, Z_MAX)
        if not len(pts):
            return []
        cles = np.floor(pts[:, :2] / GROUPE_FRONTIERE).astype(int)
        groupes: dict[tuple, list] = {}
        for cle, p in zip(map(tuple, cles), pts):
            groupes.setdefault(cle, []).append(p)
        out = []
        for elems in groupes.values():
            centre = np.mean(elems, axis=0)
            pose = self._pose_libre(np.array([centre[0], centre[1], ALT_EXPLORATION]))
            if pose is None or self.carte.est_ecartee(pose):
                continue
            out.append(Cible("explorer", pose, None, min(UTILITE_FRONTIERE * len(elems), PLAFOND_EXPLORER),
                             pose.copy(), 0))
        return out

    # ---------------------------------------------------------------- décision

    def note(self, cible: Cible, position, distance: float, avis: Avis | None = None) -> float:
        n = cible.utilite - COUT_METRE * distance
        if self.carte.reserve_par_un_autre(cible.position, self.drone, RAYON_RESERVE):
            n -= PENALITE_RESERVE
        for d, c in self.carte.coequipiers.items():
            if d != self.drone and np.linalg.norm(c.position[:2] - cible.position[:2]) < RAYON_VOISIN:
                n -= PENALITE_VOISIN
        if avis is not None and self.lam > 0:
            if np.linalg.norm(cible.origine[:2] - avis.centre[:2]) <= avis.rayon:
                n += self.lam * BONUS_ZONE
                if avis.cote is not None and cible.cote == avis.cote and cible.genre != "explorer":
                    n += self.lam * BONUS_COTE
        return float(n)

    def choisit(self, position, cibles: list[Cible] | None = None,
                avis: Avis | None = None, t: float | None = None) -> Cible | None:
        """La meilleure cible atteignable depuis `position`, avec son chemin sur la carte."""
        p = np.asarray(position, dtype=float)
        if cibles is None:
            cibles = self.candidats(t)
        self._couts = {}
        vivants = [c for c in cibles if self._visites(c.origine) < 3]
        for c in vivants:
            c.distance = float(np.linalg.norm(c.position - p))
            c.note = self.note(c, p, c.distance, avis)
        vivants.sort(key=lambda c: -c.note)
        meilleur = None
        for c in vivants[:CANDIDATS_VERIFIES]:
            points = self.carte.chemin(p, c.position, altitude=float(c.position[2]))
            if points is None:
                c.note = -math.inf
                continue
            trajet = [p] + points + [c.position]
            c.distance = float(sum(np.linalg.norm(b - a) for a, b in zip(trajet[:-1], trajet[1:])))
            c.note = self.note(c, p, c.distance, avis)
            c.points = points
            if meilleur is None or c.note > meilleur.note:
                meilleur = c
        return meilleur

    def _visites(self, origine) -> int:
        for pos, n in self.tentatives:
            if np.linalg.norm(pos - origine) < mapping.FUSION + 0.15:
                return n
        return 0

    def constate(self, cible: Cible, lu: bool) -> None:
        """Après une visite : une piste visitée sans lecture compte une tentative ; à la
        troisième, elle est écartée pour que personne n'y retourne. Les pistes bougent de
        quelques centimètres à chaque nouvelle vue : on compte par proximité."""
        if cible.genre != "lire" or lu:
            return
        for entree in self.tentatives:
            if np.linalg.norm(entree[0] - cible.origine) < mapping.FUSION + 0.15:
                entree[1] += 1
                if entree[1] >= 3:
                    self.carte.ecarte(cible.origine, duree=1e9)      # pour de bon
                return
        self.tentatives.append([np.asarray(cible.origine, dtype=float).copy(), 1])
