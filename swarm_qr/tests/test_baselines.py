"""Les deux références, testées sans simulateur : le plan du balayage fixe et les cibles du
glouton omniscient se vérifient sur la seule géométrie de l'entrepôt."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from swarm_qr import baselines as B  # noqa: E402
from swarm_qr.env.config import INTERIOR, RACKS  # noqa: E402
from swarm_qr.env.layout import make_layout  # noqa: E402

LAYOUT = make_layout(9033)


def test_chaque_drone_recoit_un_secteur_d_un_seul_tenant():
    faces = B.faces_accessibles(LAYOUT)
    blocs = B.secteurs(faces, 3)
    assert sum(len(b) for b in blocs) == len(faces)
    plat = [f for b in blocs for f in b]
    assert [f["x_drone"] for f in plat] == [f["x_drone"] for f in faces]   # ordre conservé
    for bloc in blocs:
        racks = {f["rack"] for f in bloc}
        assert len(racks) == 1                     # un rack par drone : le cas III de l'article


def test_les_charges_des_secteurs_sont_proches():
    poids = [sum(len(B._ys(f)) for f in b) for b in B.secteurs(B.faces_accessibles(LAYOUT), 3)]
    assert min(poids) > 0
    assert max(poids) <= 2 * min(poids)


def test_le_plan_monte_en_serpentin():
    """Un étage est parcouru dans un sens, le suivant dans l'autre : hover–scan–advance."""
    for liste in B.arrets_zigzag(LAYOUT, 3):
        par_etage: dict[tuple[float, float], list[float]] = {}
        for c in liste:
            par_etage.setdefault((round(float(c.position[0]), 2), round(float(c.position[2]), 2)),
                                 []).append(float(c.position[1]))
        for (x, _), ys in par_etage.items():
            assert all(a < b for a, b in zip(ys, ys[1:])) or all(a > b for a, b in zip(ys, ys[1:]))
        sens = [ys[-1] > ys[0] for ys in par_etage.values()]
        assert all(a != b for a, b in zip(sens, sens[1:]))          # alterné d'un étage au suivant


def test_les_arrets_sont_a_distance_de_lecture_et_dans_l_entrepot():
    for liste in B.arrets_zigzag(LAYOUT, 3):
        for c in liste:
            d = abs(float(c.position[0]) - float(c.origine[0]))
            assert B.RECUL_MIN <= d <= B.RECUL + 1e-6
            assert INTERIOR.x_min < c.position[0] < INTERIOR.x_max
            assert INTERIOR.y_min < c.position[1] < INTERIOR.y_max
            assert c.genre == "lire"


def test_le_plan_couvre_toutes_les_faces_accessibles_et_les_trois_etages():
    faces = B.faces_accessibles(LAYOUT)
    vus = {(round(c.position[0], 2), round(c.position[2], 2))
           for liste in B.arrets_zigzag(LAYOUT, 3) for c in liste}
    attendus = {(round(f["x_drone"], 2), round(z + B.HAUTEUR_PANNEAU + 0.11, 2))
                for f in faces for z in RACKS.shelf_levels}
    assert vus == attendus


def test_le_glouton_ne_vise_que_les_codes_non_lus():
    class Tag:
        def __init__(self, i, p, n):
            self.tag_id, self.position, self.normal = i, np.array(p), np.array(n)

    tags = [Tag("A", [-3.0, 0.0, 1.5], [-1.0, 0.0, 0.0]), Tag("B", [-3.0, 2.0, 1.5], [-1.0, 0.0, 0.0])]
    assert len(B.cibles_omniscientes(tags, set())) == 2
    restant = B.cibles_omniscientes(tags, {"A"})
    assert len(restant) == 1 and abs(restant[0].origine[1] - 2.0) < 1e-6
    assert abs(restant[0].position[0] - (-3.0 - B.RECUL)) < 1e-6


class CarteFeinte:
    """Une carte qui refuse les poses dont l'abscisse y est dans `bloquees`."""

    def __init__(self, bloquees=()):
        self.bloquees = set(bloquees)

    def pose_atteignable(self, point, marge=None):
        return round(float(point[1]), 2) not in self.bloquees


def _deroule(plan, carte):
    """Fait tourner le plan jusqu'au bout et rend les arrêts servis, dans l'ordre."""
    servis = []
    for _ in range(10_000):
        c = plan.prochain(carte, lambda p, cs: cs[0], np.zeros(3))
        if c is None and plan.fini:
            return servis
        if c is not None:
            servis.append(c)
    raise AssertionError("le plan ne se termine pas")


def test_le_plan_sert_tous_les_arrets_quand_tout_est_atteignable():
    liste = B.arrets_zigzag(LAYOUT, 3)[0]
    servis = _deroule(B.PlanFixe(liste), CarteFeinte())
    assert [id(c) for c in servis] == [id(c) for c in liste]


def test_un_arret_hors_d_atteinte_ne_bloque_pas_les_suivants():
    liste = B.arrets_zigzag(LAYOUT, 3)[0]
    bloquees = {round(float(c.position[1]), 2) for c in liste[:2]}
    plan = B.PlanFixe(liste)
    servis = _deroule(plan, CarteFeinte(bloquees))
    assert len(servis) == len([c for c in liste if round(float(c.position[1]), 2) not in bloquees])
    assert plan.bilan()["passes"] == B.PlanFixe(liste).passes      # les arrêts bloqués ont été repris


def test_le_plan_se_termine_meme_si_rien_n_est_atteignable():
    liste = B.arrets_zigzag(LAYOUT, 3)[2]
    plan = B.PlanFixe(liste)
    assert _deroule(plan, CarteFeinte({round(float(c.position[1]), 2) for c in liste})) == []
    assert plan.fini


def test_un_arret_mis_de_cote_est_repris_plus_tard():
    liste = B.arrets_zigzag(LAYOUT, 3)[2]
    carte = CarteFeinte({round(float(liste[0].position[1]), 2)})
    plan = B.PlanFixe(liste)
    servis = []
    for _ in range(5):
        c = plan.prochain(carte, lambda p, cs: cs[0], np.zeros(3))
        if c is not None:
            servis.append(c)
    assert id(liste[0]) not in [id(c) for c in servis]        # comparer des Cible avec == est ambigu
    carte.bloquees.clear()                                   # l'obstacle a disparu
    assert id(liste[0]) in [id(c) for c in _deroule(plan, carte)]


def test_un_arret_abandonne_en_route_est_repris():
    liste = B.arrets_zigzag(LAYOUT, 3)[2]
    plan = B.PlanFixe(liste)
    carte = CarteFeinte()
    premier = plan.prochain(carte, lambda p, cs: cs[0], np.zeros(3))
    plan.remet(premier)                                       # le chemin a été coupé en route
    assert id(premier) in [id(c) for c in _deroule(plan, carte)]


def test_chaque_drone_recoit_le_secteur_le_plus_proche():
    listes = B.arrets_zigzag(LAYOUT, 3)
    departs = [listes[2][0].position, listes[0][0].position, listes[1][0].position]
    donnes = B.attribue(listes, departs)
    assert [id(l) for l in donnes] == [id(listes[2]), id(listes[0]), id(listes[1])]
    assert sorted(len(l) for l in donnes) == sorted(len(l) for l in listes)
