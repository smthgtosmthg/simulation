"""La carte partagée : la mémoire commune des trois drones.

Deux structures, parce que l'espace et les panneaux ne se retiennent pas de la même façon.

**La grille** retient l'espace, en cubes de 25 cm : ce qui est occupé, ce qui est libre, et ce
qui a déjà été regardé d'assez près pour qu'un QR y aurait été lisible.

**La table** retient les panneaux. Un panneau lu a une identité — son code — et une position
connue au centimètre ; l'enfermer dans un cube jetterait cette précision. Chaque carton porte
le même code sur ses deux faces : deux lectures distantes de l'épaisseur d'un carton sont deux
panneaux, jamais un seul. Un motif repéré sans être lu n'a pas d'identité : on le suit comme
une piste, fusionnée par proximité, jusqu'à ce qu'une lecture lui donne un nom.

La carte calcule aussi les itinéraires : elle seule connaît les obstacles. L'inconnu y est
traversable mais coûteux — sans cela un drone ne sortirait jamais de sa zone explorée, et sans
le coût il couperait tout droit à travers un rack jamais vu.

Ce module ne dépend pas du simulateur : il reçoit des poses, des distances et des lectures, et
se teste sans rien lancer.
"""

from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .env.config import CAMERAS, MAP

# --- occupation, en log-odds : on additionne des preuves au lieu de les écraser
L_OCCUPE = 0.85
L_LIBRE = -0.40
L_PLAFOND = 5.0
SEUIL_OCCUPE = 1.0
SEUIL_LIBRE = -1.0

# Au-delà, deux rayons voisins du lidar (2 degrés) s'écartent de plus d'un cube et le
# creusement du vide devient troué : seule la partie fiable de la mesure sert à la carte.
PORTEE_CARTE = 8.0

# L'enveloppe de lecture mesurée à l'étape 2, en distance apparente.
LIRE_MIN = 1.5
LIRE_MAX = 4.0

RAYON_DRONE = 0.6          # hélices (0,34 m) plus l'oscillation de tenue mesurée à l'étape 2 (0,21 m)
EPAISSEUR = 0.85           # tranche d'altitude du planificateur : le rayon plus l'oscillation verticale (0,25 m)
DESSOUS = 2.0              # un drone ne survole pas une structure à moins de 2 m sous lui : un rack se contourne
SOL = 0.5                  # le sol et ce qui y traîne ne sont pas une structure à contourner
GARDE_SOL = 0.9            # ... à condition de passer au moins aussi haut au-dessus
FUSION = 0.45              # deux détections à moins de ça sont le même panneau


def tranche_de_vol(z: float, epaisseur: float = EPAISSEUR) -> tuple[float, float]:
    """La tranche d'altitude qui bloque le passage à l'altitude `z`. Un rack se contourne, mais
    le sol ne se contourne pas : le lidar marque occupées les cases de plancher qu'il voit, et
    sans cette règle un drone ne pourrait plus voler à hauteur du premier étage au-dessus d'un
    plancher déjà cartographié."""
    return min(max(z - DESSOUS, SOL), z - GARDE_SOL), z + epaisseur
COUT_INCONNU = 20.0        # traverser une case inconnue coûte vingt fois une case libre : l'inconnu au milieu d'un rack est du rack
RESERVATION_S = 45.0       # une réservation non renouvelée expire
SILENCE_S = 5.0            # un drone muet depuis plus longtemps libère ses cibles
LISTE_NOIRE_S = 120.0
SEM_CARTON = 1               # canal sémantique : un carton repéré par l'œil appris (étape 7)

INCONNU, LIBRE, OCCUPE = 0, 1, 2


@dataclass
class Panneau:
    code: str
    position: np.ndarray
    normale: np.ndarray
    vu_le: float
    lectures: int = 1


@dataclass
class Piste:
    position: np.ndarray
    normale: np.ndarray | None
    vu_le: float
    vues: int = 1


@dataclass
class Reservation:
    drone: int
    cible: np.ndarray
    jusqu_a: float


@dataclass
class Coequipier:
    position: np.ndarray
    vu_le: float


class Carte:
    def __init__(self, grille=MAP):
        self.g = grille
        self.forme = tuple(grille.shape)
        self.origine = np.array([grille.x_min, grille.y_min, grille.z_min], dtype=float)
        self.occupation = np.zeros(self.forme, dtype=np.float32)
        # quatre bits par case : de quel côté la case a été regardée (+x, -x, +y, -y). Un
        # panneau n'est lisible que vu de face ; une case vue à travers un rack depuis
        # l'autre côté ne compte pas pour lui
        self.couverture = np.zeros(self.forme, dtype=np.uint8)
        # ce que l'œil appris dira de chaque case — allée, rack, mur — vide jusqu'à l'étape 7
        self.semantique = np.zeros(self.forme, dtype=np.uint8)
        self.obstacles_mobiles: list = []          # [(position, rayon)] : les coéquipiers, pour les chemins
        self.panneaux: list[Panneau] = []
        self.pistes: list[Piste] = []
        self.reservations: dict[int, Reservation] = {}
        self.coequipiers: dict[int, Coequipier] = {}
        self.liste_noire: list[tuple[np.ndarray, float]] = []
        self.t = 0.0

    # ------------------------------------------------------------ repères

    def indice(self, points) -> np.ndarray:
        """Indice de cube. Borné juste au-delà de la grille avant la conversion en entier :
        une position absurde déborderait l'entier, et `dedans` la rejette de toute façon."""
        p = np.atleast_2d(np.asarray(points, dtype=float))
        brut = np.floor((p - self.origine) / self.g.cell)
        brut = np.nan_to_num(brut, nan=-1.0, posinf=1e6, neginf=-1.0)
        return np.clip(brut, -1.0, np.array(self.forme, dtype=float)).astype(np.int32)

    def centre(self, idx) -> np.ndarray:
        i = np.atleast_2d(np.asarray(idx, dtype=float))
        return self.origine + (i + 0.5) * self.g.cell

    def dedans(self, idx) -> np.ndarray:
        i = np.atleast_2d(np.asarray(idx))
        return np.all((i >= 0) & (i < np.array(self.forme)), axis=1)

    def sur_la_carte(self, points) -> np.ndarray:
        p = np.atleast_2d(np.asarray(points, dtype=float))
        fini = np.isfinite(p).all(axis=1)
        return fini & self.dedans(self.indice(np.where(fini[:, None], p, 0.0)))

    def etat(self, points) -> np.ndarray:
        idx = self.indice(points)
        ok = self.dedans(idx)
        out = np.full(len(idx), INCONNU, dtype=np.int8)
        if ok.any():
            v = self.occupation[tuple(idx[ok].T)]
            e = np.full(len(v), INCONNU, dtype=np.int8)
            e[v > SEUIL_OCCUPE] = OCCUPE
            e[v < SEUIL_LIBRE] = LIBRE
            out[ok] = e
        return out

    # ------------------------------------------------------------ occupation

    def integre_lidar(self, origine, directions, portees, t: float | None = None) -> None:
        """Un tour de lidar : directions unitaires en monde, distance par rayon (infinie si
        rien n'a été touché). Le long de chaque rayon l'espace traversé devient libre et le
        point touché devient occupé. Une case vue libre dix fois puis occupée une fois reste
        libre : c'est ce qui absorbe les mesures aberrantes."""
        if t is not None:
            self.t = t
        o = np.asarray(origine, dtype=float)
        d = np.asarray(directions, dtype=float)
        d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        r = np.asarray(portees, dtype=float)

        touche = np.isfinite(r) & (r > 0.0) & (r <= PORTEE_CARTE)
        vide = np.clip(np.where(np.isfinite(r), r, PORTEE_CARTE), 0.0, PORTEE_CARTE)
        # trois points par case : un rayon qui effleure le coin d'une case doit y déposer un
        # point, sinon la case reste inconnue au milieu du libre
        pas = self.g.cell / 3.0
        ts = np.arange(1, int(PORTEE_CARTE / pas) + 1, dtype=float) * pas
        # une case entière avant l'impact : sinon on efface la surface qu'on vient de mesurer
        libres = ts[None, :] < (vide[:, None] - self.g.cell)
        if libres.any():
            pts = o + d[:, None, :] * ts[None, :, None]
            self._ajoute(pts[libres], L_LIBRE)
        if touche.any():
            self._ajoute(o + d[touche] * r[touche, None], L_OCCUPE)

    def _ajoute(self, points, valeur: float) -> None:
        idx = self.indice(points)
        idx = idx[self.dedans(idx)]
        if not len(idx):
            return
        np.add.at(self.occupation, tuple(idx.T), valeur)
        np.clip(self.occupation, -L_PLAFOND, L_PLAFOND, out=self.occupation)

    # ------------------------------------------------------------ couverture

    @staticmethod
    def cardinal(direction) -> int:
        """Le côté cardinal le plus proche d'une direction horizontale : 0 +x, 1 -x, 2 +y, 3 -y."""
        v = np.asarray(direction, dtype=float)
        if abs(v[0]) >= abs(v[1]):
            return 0 if v[0] > 0 else 1
        return 2 if v[1] > 0 else 3

    def couvert_pour(self, points, normale) -> np.ndarray:
        """Un panneau de normale `normale` posé en ces points aurait-il été lisible ? Oui si la
        case a été regardée depuis le côté vers lequel il fait face."""
        bit = 1 << self.cardinal(-np.asarray(normale, dtype=float))
        idx = self.indice(points)
        ok = self.dedans(idx)
        out = np.zeros(len(idx), dtype=bool)
        if ok.any():
            out[ok] = (self.couverture[tuple(idx[ok].T)] & bit) > 0
        return out

    def integre_couverture(self, position, avant, t: float | None = None,
                           pas_deg: float = 3.0) -> int:
        """Ce qu'une caméra de lecture a vu d'assez près pour y lire un QR.

        Une case n'est couverte que si sa distance apparente — la distance divisée par le
        cosinus de l'angle au bord du champ — tombe dans l'enveloppe de l'étape 2, et si aucun
        obstacle connu ne la cache. On retient aussi de quel côté elle a été regardée. Une case
        couverte ne veut pas dire « vue » mais « un QR posé là, face à la caméra, aurait été
        lisible ». Rend le nombre de cases nouvellement couvertes de ce côté.
        """
        if t is not None:
            self.t = t
        o = np.asarray(position, dtype=float)
        axe = np.asarray(avant, dtype=float)
        axe = axe / max(np.linalg.norm(axe), 1e-9)
        bit = np.uint8(1 << self.cardinal(axe))
        dirs = _cone(axe, CAMERAS.fov_deg, CAMERAS.fov_deg * CAMERAS.side_height / CAMERAS.side_width,
                     pas_deg)
        cosinus = np.clip(dirs @ axe, 1e-3, 1.0)
        pas = self.g.cell * 0.5
        ts = np.arange(pas, LIRE_MAX + pas, pas)
        apparente = ts[None, :] / cosinus[:, None]
        pts = o + dirs[:, None, :] * ts[None, :, None]
        idx = self.indice(pts.reshape(-1, 3)).reshape(len(dirs), len(ts), 3)
        dedans = self.dedans(idx.reshape(-1, 3)).reshape(len(dirs), len(ts))
        plein = np.zeros(dedans.shape, dtype=bool)
        plein[dedans] = self.occupation[tuple(idx[dedans].T)] > SEUIL_OCCUPE
        cache = np.cumsum(plein, axis=1) > 0                  # tout ce qui suit un obstacle
        bon = dedans & ~cache & (apparente >= LIRE_MIN) & (apparente <= LIRE_MAX)
        if not bon.any():
            return 0
        cases = tuple(idx[bon].T)
        nouvelles = int(np.count_nonzero((self.couverture[cases] & bit) == 0))
        self.couverture[cases] |= bit
        return nouvelles

    # ------------------------------------------------------------ panneaux

    def premier_obstacle(self, origine, direction, portee: float = PORTEE_CARTE,
                         pas: float = 0.1) -> float | None:
        """Distance, le long d'un rayon, du premier cube connu occupé ; None s'il n'y en a pas
        avant `portee`. Sert à placer un motif repéré de loin : la carte accumule les tours de
        lidar, là où un rayon isolé, à 8 m, passe à 28 cm de son voisin et peut tomber sur le
        carton d'à côté ou, par un trou, sur le fond du rack."""
        o = np.asarray(origine, dtype=float)
        d = np.asarray(direction, dtype=float)
        d = d / max(np.linalg.norm(d), 1e-9)
        ts = np.arange(pas, portee + pas / 2, pas)
        pts = o + ts[:, None] * d
        idx = self.indice(pts)
        ok = self.dedans(idx)
        occ = np.zeros(len(ts), dtype=bool)
        occ[ok] = self.occupation[tuple(idx[ok].T)] > SEUIL_OCCUPE
        k = np.flatnonzero(occ)
        return float(ts[k[0]]) if len(k) else None

    def marque(self, points, valeur: int = SEM_CARTON) -> int:
        """Étiquette sémantique sur les cubes de `points` ; rend le nombre de cubes marqués."""
        idx = self.indice(points)
        ok = self.dedans(idx)
        if ok.any():
            self.semantique[tuple(idx[ok].T)] = valeur
        return int(ok.sum())

    @property
    def codes(self) -> set[str]:
        """Les cartons dont on connaît le contenu : c'est la mesure de la mission."""
        return {p.code for p in self.panneaux}

    def faces(self, code: str) -> list[Panneau]:
        return [p for p in self.panneaux if p.code == code]

    def integre_lecture(self, code: str, position, normale, t: float | None = None):
        """Un QR décodé. La lecture rejoint la face connue la plus proche portant le même
        code ; trop loin, c'est l'autre face du carton, et elle devient un second panneau.
        L'identité étant certaine, la lecture chasse toute piste au même endroit."""
        if t is not None:
            self.t = t
        p = np.asarray(position, dtype=float)
        if not self.sur_la_carte(p)[0]:
            return None
        n = np.asarray(normale, dtype=float)
        memes = self.faces(code)
        proche = min(memes, key=lambda q: np.linalg.norm(q.position - p), default=None)
        if proche is None or np.linalg.norm(proche.position - p) > FUSION:
            if len(memes) >= 2:
                # un carton n'a que deux faces : une troisième lecture loin des deux
                # connues est une erreur de décodage, pas un panneau
                return None
            proche = Panneau(code, p, n, self.t)
            self.panneaux.append(proche)
        else:
            k = proche.lectures
            proche.position = (proche.position * k + p) / (k + 1)
            proche.normale = n
            proche.lectures = k + 1
            proche.vu_le = self.t
        self.pistes = [q for q in self.pistes if np.linalg.norm(q.position - p) > FUSION]
        return proche

    def integre_reperage(self, position, normale=None, t: float | None = None):
        """Un motif repéré sans être lu. Sa position est une supposition, moyennée avec les
        vues précédentes du même endroit ; un repérage sur un panneau déjà lu est ignoré."""
        if t is not None:
            self.t = t
        p = np.asarray(position, dtype=float)
        if not self.sur_la_carte(p)[0]:
            return None
        for panneau in self.panneaux:
            if np.linalg.norm(panneau.position - p) <= FUSION:
                return None
        for piste in self.pistes:
            if np.linalg.norm(piste.position - p) <= FUSION:
                k = piste.vues
                piste.position = (piste.position * k + p) / (k + 1)
                piste.vues = k + 1
                piste.vu_le = self.t
                if normale is not None:
                    # la direction d'aperçu moyennée sur toutes les vues : un drone qui longe
                    # une allée voit le même code de biais puis de face, la moyenne tend vers
                    # la normale du panneau
                    n = np.asarray(normale, dtype=float)
                    if piste.normale is not None:
                        n = piste.normale * k + n
                    piste.normale = n / max(np.linalg.norm(n), 1e-9)
                return piste
        piste = Piste(p, None if normale is None else np.asarray(normale, dtype=float), self.t)
        self.pistes.append(piste)
        return piste

    def oublie_pistes(self, point, rayon: float) -> int:
        """Efface les pistes autour d'un point : après une lecture réussie devant une piste,
        les pistes voisines sont le même panneau placé à quelques dizaines de centimètres."""
        p = np.asarray(point, dtype=float)
        avant = len(self.pistes)
        self.pistes = [q for q in self.pistes if np.linalg.norm(q.position - p) > rayon]
        return avant - len(self.pistes)

    # ------------------------------------------------------------ équipe

    def annonce(self, drone: int, position, t: float | None = None) -> None:
        if t is not None:
            self.t = t
        self.coequipiers[drone] = Coequipier(np.asarray(position, dtype=float), self.t)

    def reserve(self, drone: int, cible, duree: float = RESERVATION_S) -> None:
        self.reservations[drone] = Reservation(drone, np.asarray(cible, dtype=float),
                                               self.t + duree)

    def libere(self, drone: int) -> None:
        self.reservations.pop(drone, None)

    def reserve_par_un_autre(self, point, drone: int, rayon: float = 2.0) -> bool:
        p = np.asarray(point, dtype=float)
        return any(r.drone != drone and np.linalg.norm(r.cible - p) < rayon
                   for r in self.reservations.values())

    def ecarte(self, cible, duree: float = LISTE_NOIRE_S) -> None:
        self.liste_noire.append((np.asarray(cible, dtype=float), self.t + duree))

    def est_ecartee(self, point, rayon: float = 0.6) -> bool:
        p = np.asarray(point, dtype=float)
        return any(np.linalg.norm(c - p) < rayon for c, _ in self.liste_noire)

    def vieillit(self, t: float) -> list[int]:
        """Fait expirer ce qui n'a pas été renouvelé et rend les drones devenus muets. Un drone
        en panne libère ainsi ses cibles sans qu'aucun code ne le surveille."""
        self.t = t
        self.reservations = {d: r for d, r in self.reservations.items() if r.jusqu_a > t}
        self.liste_noire = [(c, e) for c, e in self.liste_noire if e > t]
        muets = [d for d, c in self.coequipiers.items() if t - c.vu_le > SILENCE_S]
        for d in muets:
            self.reservations.pop(d, None)
        return muets

    # ------------------------------------------------------------ frontières

    def frontieres(self, z_min: float, z_max: float) -> np.ndarray:
        """Les cases libres qui touchent l'inconnu, dans une tranche d'altitude : c'est là qu'il
        faut aller pour découvrir du nouveau. Rend leurs centres."""
        k0, k1 = self._tranche(z_min, z_max)
        occ = self.occupation[:, :, k0:k1]
        libre = (occ < SEUIL_LIBRE).any(axis=2) & ~(occ > SEUIL_OCCUPE).any(axis=2)
        inconnu = ~((occ < SEUIL_LIBRE) | (occ > SEUIL_OCCUPE)).any(axis=2)

        def voisins(m):
            v = np.zeros(m.shape, dtype=np.int8)
            v[1:, :] += m[:-1, :]
            v[:-1, :] += m[1:, :]
            v[:, 1:] += m[:, :-1]
            v[:, :-1] += m[:, 1:]
            return v

        # un trou d'une case au milieu du connu n'est pas une frontière, c'est un défaut
        # d'échantillonnage : l'inconnu qui compte tient à d'autres cases inconnues
        region = inconnu & (voisins(inconnu.astype(np.int8)) >= 2)
        ij = np.argwhere(libre & (voisins(region.astype(np.int8)) > 0))
        if not len(ij):
            return np.zeros((0, 3))
        z = 0.5 * (z_min + z_max)
        return np.column_stack([self.centre(np.column_stack([ij, np.zeros(len(ij))]))[:, :2],
                                np.full(len(ij), z)])

    # ------------------------------------------------------------ chemin

    def _tranche(self, z_min: float, z_max: float) -> tuple[int, int]:
        k0 = max(0, int((z_min - self.g.z_min) / self.g.cell))
        k1 = min(self.forme[2], int((z_max - self.g.z_min) / self.g.cell) + 1)
        return k0, max(k1, k0 + 1)

    def couts(self, z_min: float, z_max: float, marge: float = RAYON_DRONE) -> np.ndarray:
        """Vue de dessus des coûts de passage dans une tranche d'altitude : infini sur un
        obstacle connu élargi du rayon du drone, 1 sur du libre connu, COUT_INCONNU ailleurs."""
        k0, k1 = self._tranche(z_min, z_max)
        occ = self.occupation[:, :, k0:k1]
        dur = (occ > SEUIL_OCCUPE).any(axis=2)
        libre = (occ < SEUIL_LIBRE).any(axis=2)
        # rayon en cases, sans arrondi vers le haut : arrondir 0,6 m à 3 cases ferait 0,75 m
        # et fermerait les allées étroites
        r = marge / self.g.cell
        n = int(math.ceil(r))
        nx, ny = dur.shape
        gros = dur.copy()
        for dx in range(-n, n + 1):
            for dy in range(-n, n + 1):
                if dx * dx + dy * dy > r * r or (dx == 0 and dy == 0):
                    continue
                sx, sy = slice(max(dx, 0), nx + min(dx, 0)), slice(max(dy, 0), ny + min(dy, 0))
                tx, ty = slice(max(-dx, 0), nx - max(dx, 0)), slice(max(-dy, 0), ny - max(dy, 0))
                gros[sx, sy] |= dur[tx, ty]
        cout = np.full(dur.shape, COUT_INCONNU, dtype=np.float32)
        cout[libre] = 1.0
        cout[gros] = np.inf
        # les coéquipiers en vol sont des obstacles qui bougent : un chemin les contourne, une
        # pose ne se prend pas contre eux — c'est en amont, pas à la dernière seconde, que
        # deux drones s'évitent
        for position, rayon in self.obstacles_mobiles:
            i, j, _ = self.indice(position)[0]
            n = int(math.ceil(rayon / self.g.cell))
            i0, i1 = max(i - n, 0), min(i + n + 1, nx)
            j0, j1 = max(j - n, 0), min(j + n + 1, ny)
            if i0 < i1 and j0 < j1:
                ii, jj = np.mgrid[i0:i1, j0:j1]
                cout[i0:i1, j0:j1][((ii - i) ** 2 + (jj - j) ** 2) * self.g.cell ** 2 <= rayon ** 2] = np.inf
        return cout

    def couts_de_vol(self, z: float) -> np.ndarray:
        """Les coûts de passage pour un vol à l'altitude `z` : tout obstacle connu entre 2 m
        sous le drone et 0,85 m au-dessus bloque la colonne. Survoler les cartons du dernier
        étage d'un rack, entre ses montants, est possible dans le simulateur ; ce serait une
        collision dans un vrai entrepôt."""
        return self.couts(*tranche_de_vol(z))

    def chemin(self, depart, arrivee, altitude: float | None = None,
               epaisseur: float = EPAISSEUR) -> list[np.ndarray] | None:
        """Points de passage de `depart` à `arrivee`, à altitude constante. Liste vide si la
        ligne droite passe par du libre connu ; None si aucun chemin n'existe."""
        a = np.asarray(depart, dtype=float)
        b = np.asarray(arrivee, dtype=float)
        z = float(a[2] if altitude is None else altitude)
        cout = self.couts(*tranche_de_vol(z, epaisseur))
        ia, ib = tuple(self.indice(a)[0][:2]), tuple(self.indice(b)[0][:2])
        if not self._praticable(ib, cout):
            return None
        if not self._praticable(ia, cout):
            # le drone est dans la marge élargie d'un obstacle, pas dans l'obstacle : il doit
            # pouvoir en sortir, sinon il resterait bloqué là où il se trouve
            k0, k1 = self._tranche(*tranche_de_vol(z, epaisseur))
            if (self.occupation[ia[0], ia[1], k0:k1] > SEUIL_OCCUPE).any():
                return None
            cout = cout.copy()
            cout[ia] = COUT_INCONNU
        if self._droite_ok(ia, ib, cout, 1.0):
            return []
        brut = _astar(cout, ia, ib)
        if brut is None:
            return None
        lisse = self._elague(brut, cout)
        return [np.array([*self.centre([i, j, 0])[0][:2], z]) for i, j in lisse[1:-1]]

    def segment_libre(self, a, b, altitude: float | None = None, epaisseur: float = EPAISSEUR) -> bool:
        """Un segment déjà planifié passe-t-il encore ? Faux dès qu'un obstacle connu élargi le
        coupe : c'est le signal pour recalculer le chemin pendant le transit."""
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        z = float(a[2] if altitude is None else altitude)
        cout = self.couts(*tranche_de_vol(z, epaisseur))
        ia, ib = tuple(self.indice(a)[0][:2]), tuple(self.indice(b)[0][:2])
        return np.isfinite(self._cout_droite(ia, ib, cout))

    def pose_atteignable(self, point, marge: float = RAYON_DRONE) -> bool:
        """Un drone peut-il tenir cette pose ? Vrai si sa case est libre de tout obstacle connu
        élargi du rayon du drone. Test à une case, sans calcul de chemin."""
        cout = self.couts(*tranche_de_vol(float(point[2])), marge=marge)
        return self._praticable(tuple(self.indice(point)[0][:2]), cout)

    def _praticable(self, ij, cout) -> bool:
        i, j = int(ij[0]), int(ij[1])
        return 0 <= i < cout.shape[0] and 0 <= j < cout.shape[1] and np.isfinite(cout[i, j])

    @staticmethod
    def _cout_droite(a, b, cout) -> float:
        """Coût d'un segment droit : la longueur pondérée par les cases traversées, infini si
        l'une d'elles est un obstacle."""
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) * 2 + 1
        longueur = math.hypot(b[0] - a[0], b[1] - a[1])
        total = 0.0
        for k in range(n + 1):
            i = int(round(a[0] + (b[0] - a[0]) * k / n))
            j = int(round(a[1] + (b[1] - a[1]) * k / n))
            c = float(cout[i, j])
            if not np.isfinite(c):
                return math.inf
            total += c
        return total * longueur / (n + 1)

    def _droite_ok(self, a, b, cout, cout_max: float) -> bool:
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) * 2 + 1
        for k in range(n + 1):
            i = int(round(a[0] + (b[0] - a[0]) * k / n))
            j = int(round(a[1] + (b[1] - a[1]) * k / n))
            if cout[i, j] > cout_max:
                return False
        return True

    def _elague(self, cases, cout) -> list:
        """Ne garde que les points où il faut tourner. Un raccourci n'est accepté que s'il ne
        coûte pas plus cher que le bout de chemin qu'il remplace : couper tout droit par de
        l'inconnu ne doit jamais défaire un détour par du libre connu."""
        cumul = [0.0]
        for a, b in zip(cases[:-1], cases[1:]):
            cumul.append(cumul[-1] + math.hypot(b[0] - a[0], b[1] - a[1]) * float(cout[b]))
        out = [cases[0]]
        i = 0
        while i < len(cases) - 1:
            j = len(cases) - 1
            while j > i + 1 and self._cout_droite(cases[i], cases[j], cout) > (cumul[j] - cumul[i]) * 1.02 + 1e-6:
                j -= 1
            out.append(cases[j])
            i = j
        return out

    # ------------------------------------------------------------ sauvegarde et bilan

    def sauve(self, souche) -> None:
        souche = Path(souche)
        np.savez_compressed(souche.with_suffix(".npz"), occupation=self.occupation,
                            couverture=self.couverture, semantique=self.semantique)
        souche.with_suffix(".json").write_text(json.dumps({
            "t": self.t,
            "panneaux": [{"code": p.code, "position": p.position.tolist(),
                          "normale": p.normale.tolist(), "vu_le": p.vu_le,
                          "lectures": p.lectures} for p in self.panneaux],
            "pistes": [{"position": q.position.tolist(),
                        "normale": None if q.normale is None else q.normale.tolist(),
                        "vu_le": q.vu_le, "vues": q.vues} for q in self.pistes],
        }))

    @classmethod
    def charge(cls, souche) -> "Carte":
        souche = Path(souche)
        c = cls()
        z = np.load(souche.with_suffix(".npz"))
        c.occupation, c.couverture = z["occupation"], z["couverture"]
        if "semantique" in z:
            c.semantique = z["semantique"]
        d = json.loads(souche.with_suffix(".json").read_text())
        c.t = d["t"]
        c.panneaux = [Panneau(p["code"], np.array(p["position"]), np.array(p["normale"]),
                              p["vu_le"], p["lectures"]) for p in d["panneaux"]]
        c.pistes = [Piste(np.array(q["position"]),
                          None if q["normale"] is None else np.array(q["normale"]),
                          q["vu_le"], q["vues"]) for q in d["pistes"]]
        return c

    def resume(self) -> dict:
        occ = self.occupation > SEUIL_OCCUPE
        libre = self.occupation < SEUIL_LIBRE
        total = int(np.prod(self.forme))
        return {
            "cases": total,
            "occupees": int(occ.sum()),
            "libres": int(libre.sum()),
            "inconnues": total - int(occ.sum()) - int(libre.sum()),
            "couvertes": int(np.count_nonzero(self.couverture)),
            "part_connue": round(float((occ | libre).mean()), 4),
            "codes_lus": len(self.codes),
            "faces_lues": len(self.panneaux),
            "pistes": len(self.pistes),
            "reservations": len(self.reservations),
            "octets": int(self.occupation.nbytes + self.couverture.nbytes + self.semantique.nbytes),
        }


# ---------------------------------------------------------------- outils

def _cone(axe, fov_h_deg: float, fov_v_deg: float, pas_deg: float) -> np.ndarray:
    """Directions unitaires régulièrement réparties dans le champ d'une caméra."""
    a = np.asarray(axe, dtype=float)
    a = a / max(np.linalg.norm(a), 1e-9)
    haut = np.array([0.0, 0.0, 1.0])
    droite = np.cross(a, haut)
    droite /= max(np.linalg.norm(droite), 1e-9)
    haut = np.cross(droite, a)
    hs = np.radians(np.arange(-fov_h_deg / 2, fov_h_deg / 2 + 1e-6, pas_deg))
    vs = np.radians(np.arange(-fov_v_deg / 2, fov_v_deg / 2 + 1e-6, pas_deg))
    H, V = np.meshgrid(hs, vs, indexing="ij")
    d = a[None, None, :] + np.tan(H)[..., None] * droite + np.tan(V)[..., None] * haut
    d = d.reshape(-1, 3)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


_VOISINS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]


def _astar(cout: np.ndarray, depart: tuple, arrivee: tuple) -> list | None:
    nx, ny = cout.shape
    h = lambda c: math.hypot(c[0] - arrivee[0], c[1] - arrivee[1])
    ouvert = [(h(depart), 0.0, depart)]
    venu = {depart: None}
    meilleur = {depart: 0.0}
    while ouvert:
        _, g, c = heapq.heappop(ouvert)
        if c == arrivee:
            chemin = []
            while c is not None:
                chemin.append(c)
                c = venu[c]
            return chemin[::-1]
        if g > meilleur.get(c, math.inf):
            continue
        for dx, dy, pas in _VOISINS:
            v = (c[0] + dx, c[1] + dy)
            if not (0 <= v[0] < nx and 0 <= v[1] < ny) or not np.isfinite(cout[v]):
                continue
            g2 = g + pas * float(cout[v])
            if g2 < meilleur.get(v, math.inf):
                meilleur[v] = g2
                venu[v] = c
                heapq.heappush(ouvert, (g2 + h(v), g2, v))
    return None


# ---------------------------------------------------------------- vue de dessus

COULEURS = {
    "inconnu": (110, 110, 110),
    "libre": (185, 185, 185),
    "couvert": (245, 245, 245),
    "occupe": (60, 60, 60),
}
COULEURS_DRONES = [(255, 60, 0), (0, 160, 255), (200, 0, 200)]


def vue_de_dessus(carte: Carte, echelle: int = 6, z_min: float = 0.6, z_max: float = 5.5,
                  trajectoire=None) -> np.ndarray:
    """L'image que l'on doit pouvoir lire à l'œil : gris foncé les obstacles, blanc ce qui a été
    regardé d'assez près pour lire, gris clair le libre, gris moyen l'inconnu, vert les
    panneaux lus, orange les motifs repérés non lus, un rond par drone avec un trait vers sa
    cible réservée."""
    import cv2

    k0, k1 = carte._tranche(z_min, z_max)
    tranche = carte.occupation[:, :, k0:k1]
    occ = (tranche > SEUIL_OCCUPE).any(axis=2)
    libre = (tranche < SEUIL_LIBRE).any(axis=2)
    couvert = (carte.couverture[:, :, k0:k1] > 0).any(axis=2)

    nx, ny = occ.shape
    img = np.full((ny, nx, 3), COULEURS["inconnu"], dtype=np.uint8)
    img[libre.T] = COULEURS["libre"]
    img[couvert.T] = COULEURS["couvert"]
    img[occ.T] = COULEURS["occupe"]
    img = cv2.resize(img, (nx * echelle, ny * echelle), interpolation=cv2.INTER_NEAREST)
    img = cv2.flip(img, 0)

    def px(p):
        if not carte.sur_la_carte(p)[0]:
            return None
        i, j = carte.indice(p)[0][:2]
        return int((i + 0.5) * echelle), int((ny - j - 0.5) * echelle)

    if trajectoire is not None and len(trajectoire) > 1:
        pts = [q for q in (px(p) for p in trajectoire) if q is not None]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(img, a, b, (255, 130, 40), 1, cv2.LINE_AA)      # BGR : bleu
    for piste in carte.pistes:
        if (q := px(piste.position)) is not None:
            cv2.circle(img, q, max(2, echelle // 2), (0, 140, 255), -1)
    for panneau in carte.panneaux:
        if (q := px(panneau.position)) is not None:
            cv2.circle(img, q, max(2, echelle // 2), (0, 170, 0), -1)
    for drone, c in carte.coequipiers.items():
        p = px(c.position)
        if p is None:
            continue
        couleur = COULEURS_DRONES[drone % len(COULEURS_DRONES)]
        r = carte.reservations.get(drone)
        if r is not None and (q := px(r.cible)) is not None:
            cv2.line(img, p, q, couleur, 1, cv2.LINE_AA)
        cv2.circle(img, p, echelle, couleur, -1)
        cv2.putText(img, str(drone), (p[0] + echelle, p[1] - echelle),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, couleur, 1, cv2.LINE_AA)
    return img
