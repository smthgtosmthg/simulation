"""Les deux références du système, avec la même perception et le même contrôleur.

`arrets_zigzag` : la méthode statique de Pore et al. (Symmetry 2026) — un chemin de couverture
en zigzag prévu à l'avance le long de chaque face de rack, avec des arrêts « hover-and-scan »,
la hauteur avançant par paliers d'étagère (hover–scan–advance) et, à plusieurs drones, des
serpentins en miroir partant des extrémités opposées. Aucune carte, aucune décision : la liste
des arrêts est fixée avant le décollage.

`cibles_omniscientes` : le glouton qui connaît la position de tous les codes (triche assumée) et
va toujours au plus proche non lu. C'est le plafond pratique.
"""
from __future__ import annotations

import math

import numpy as np

from .env.config import CAMERAS, INTERIOR, RACKS
from .planning import UTILITE_LIRE, Cible, cap_pour_regarder
from . import mapping

RECUL = 2.2                 # m au bord du rack : au milieu de l'enveloppe de lecture (étape 2)
RECUL_MIN = 0.9             # dans une allée étroite, on se rapproche plutôt que d'y renoncer
MARGE_OPPOSEE = 0.85        # distance gardée avec le rack ou le mur d'en face
HAUTEUR_PANNEAU = 0.225     # le QR est au milieu de la face d'un carton de 50 cm posé sur l'étagère
PAS_ARRET = 1.5             # m entre deux arrêts de lecture le long d'une face
BORD = 0.3                  # m non parcourus aux deux bouts d'un rack


def allees(layout) -> list[dict]:
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


def faces_accessibles(layout) -> list[dict]:
    out = []
    for a in allees(layout):
        recul = min(RECUL, (a["x1"] - a["x0"]) - MARGE_OPPOSEE)
        if recul < RECUL_MIN:
            continue                                    # allée trop étroite : face inaccessible
        for rack, cote in a["faces"]:
            out.append({"rack": rack.prim, "cote": cote, "x_drone": rack.x + cote * (RACKS.depth / 2.0 + recul),
                        "x_face": rack.x + cote * RACKS.depth / 2.0,
                        "y0": rack.y_bounds[0] + BORD, "y1": rack.y_bounds[1] - BORD,
                        "cap": math.atan2(0.0, cote) + math.pi / 2.0,      # la caméra gauche lit à 90° du cap
                        "cardinal": mapping.Carte.cardinal([cote, 0.0])})
    return out


def _ys(face: dict) -> np.ndarray:
    """Les abscisses des arrêts le long d'une face, du sud au nord."""
    return np.arange(face["y0"], face["y1"] + 1e-6, PAS_ARRET)


def secteurs(faces: list[dict], n_drones: int) -> list[list[dict]]:
    """Le répartiteur de secteurs de l'article : des blocs de faces voisines, de charge proche,
    pour que deux drones ne se croisent pas. Les faces sortent rack par rack, donc un bloc est
    un rack entier quand il y a autant de racks que de drones."""
    poids = [len(_ys(f)) for f in faces]
    part = sum(poids) / max(n_drones, 1)
    blocs: list[list[dict]] = []
    bloc: list[dict] = []
    acc = 0.0
    for k, f in enumerate(faces):
        bloc.append(f)
        acc += poids[k]
        reste = len(faces) - k - 1
        if len(blocs) < n_drones - 1 and acc >= part and reste >= n_drones - len(blocs) - 1:
            blocs.append(bloc)
            bloc, acc = [], 0.0
    blocs.append(bloc)
    return blocs + [[] for _ in range(n_drones - len(blocs))]


def arrets_zigzag(layout, n_drones: int, etages=(0, 1, 2)) -> list[list[Cible]]:
    """Par drone, la liste ordonnée des arrêts de lecture, fixée avant le décollage. Chaque
    drone reçoit un secteur, qu'il balaie face par face : une face est parcourue sur toute sa
    longueur à un étage, puis on monte d'un étage et on repart en sens inverse — le
    hover–scan–advance de l'article. Les drones de rang impair commencent par l'extrémité
    opposée, ce qui donne les serpentins en miroir."""
    listes: list[list[Cible]] = []
    for d, bloc in enumerate(secteurs(faces_accessibles(layout), n_drones)):
        arrets: list[Cible] = []
        sens = 1 if d % 2 == 0 else -1
        for f in bloc:
            for niveau in etages:
                z = RACKS.shelf_levels[niveau] + HAUTEUR_PANNEAU + CAMERAS.below
                ys = _ys(f)[::sens]
                arrets += [Cible("lire", np.array([f["x_drone"], float(y), z]), f["cap"], UTILITE_LIRE,
                                 np.array([f["x_face"], float(y), z]), f["cardinal"]) for y in ys]
                sens = -sens
        listes.append(arrets)
    return listes


def attribue(listes: list[list[Cible]], departs) -> list[list[Cible]]:
    """Donne à chaque drone le secteur dont le premier arrêt est le plus proche de lui. Dans
    l'article les drones sont déjà postés à leur rack au départ ; ici ils naissent où la scène
    les place, et sans cette attribution l'un d'eux traverse tout l'entrepôt avant de lire."""
    libres = list(range(len(listes)))
    out: list[list[Cible]] = [[] for _ in listes]
    for i, p in enumerate(departs):
        if not libres:
            break
        k = min(libres, key=lambda k: float(np.linalg.norm(np.asarray(listes[k][0].position) - np.asarray(p)))
                if listes[k] else float("inf"))
        out[i] = listes[k]
        libres.remove(k)
    return out


class PlanFixe:
    """La consommation du plan par un drone. Il prend ses arrêts dans l'ordre ; celui qu'il ne
    peut pas atteindre sur le moment — une palette au sol, un coéquipier, un obstacle apparu —
    est mis de côté et repris au passage suivant. Le plan avance toujours : un arrêt hors
    d'atteinte ne bloque jamais ceux qui le suivent, ce qui figeait la version précédente."""

    def __init__(self, arrets: list[Cible], fenetre: int = 12, chemins: int = 3, passes: int = 3):
        self.arrets = list(arrets)
        self.de_cote: list[Cible] = []
        self.k = 0
        self.passe = 1
        self.fenetre, self.chemins, self.passes = fenetre, chemins, passes

    def prochain(self, carte, choisit, position) -> Cible | None:
        """Le prochain arrêt atteignable. `choisit(position, [cible])` calcule le chemin et rend
        la cible, ou None s'il n'y en a pas ; il coûte cher, donc on le limite par décision."""
        calculs = 0
        for _ in range(self.fenetre):
            if self.k >= len(self.arrets):
                if not self.de_cote or self.passe >= self.passes:
                    return None
                self.arrets, self.de_cote = self.de_cote, []
                self.k, self.passe = 0, self.passe + 1
                continue
            cible = self.arrets[self.k]
            self.k += 1
            if carte.pose_atteignable(cible.position) and calculs < self.chemins:
                calculs += 1
                choix = choisit(position, [cible])
                if choix is not None:
                    return choix
            self.de_cote.append(cible)
        return None

    def remet(self, cible: Cible) -> None:
        """Un arrêt abandonné en route — chemin coupé, drone immobilisé — repasse à la fin de la
        liste : l'article replanifie localement, il ne renonce pas à une étagère."""
        if self.passe < self.passes:
            self.de_cote.append(cible)

    @property
    def fini(self) -> bool:
        return self.k >= len(self.arrets) and (not self.de_cote or self.passe >= self.passes)

    def bilan(self) -> dict:
        return {"restants": len(self.arrets) - self.k, "mis_de_cote": len(self.de_cote),
                "passes": self.passe, "fini": self.fini}


def cibles_omniscientes(tags, codes_lus: set[str]) -> list[Cible]:
    """Une cible par face de carton dont le code n'est pas encore lu, à 2 m devant la face, la
    caméra à la hauteur du panneau. Le choix du plus proche est fait par `Cerveau.choisit`."""
    out = []
    for t in tags:
        if t.tag_id in codes_lus:
            continue
        n = np.array(t.normal, dtype=float)
        n[2] = 0.0
        if np.linalg.norm(n) < 1e-6:
            continue
        n /= np.linalg.norm(n)
        p = np.array(t.position, dtype=float)
        pose = np.array([p[0] + RECUL * n[0], p[1] + RECUL * n[1], p[2] + CAMERAS.below])
        if not (INTERIOR.x_min + MARGE_OPPOSEE < pose[0] < INTERIOR.x_max - MARGE_OPPOSEE):
            continue                                    # la pose serait dans le mur (allée est)
        out.append(Cible("lire", pose, cap_pour_regarder(-n), UTILITE_LIRE, p.copy(), mapping.Carte.cardinal(n)))
    return out
