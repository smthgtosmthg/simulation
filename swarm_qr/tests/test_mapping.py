"""La carte partagée, testée sans simulateur.

Un monde de boîtes alignées et un lidar analytique suffisent : on connaît la vérité, donc on
vérifie la carte reconstruite case par case, en une fraction de seconde.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from swarm_qr import mapping as M  # noqa: E402
from swarm_qr.mapping import INCONNU, LIBRE, OCCUPE, Carte  # noqa: E402


# ---------------------------------------------------------------- monde d'essai

class Monde:
    """Des boîtes alignées sur les axes, et un lidar qui dit la vérité."""

    def __init__(self, boites):
        self.boites = [(np.asarray(lo, float), np.asarray(hi, float)) for lo, hi in boites]

    def occupe(self, p) -> bool:
        p = np.asarray(p, float)
        return any(np.all(p >= lo) and np.all(p <= hi) for lo, hi in self.boites)

    def portee(self, o, d, maxi=25.0) -> float:
        o, d = np.asarray(o, float), np.asarray(d, float)
        best = np.inf
        for lo, hi in self.boites:
            with np.errstate(divide="ignore", invalid="ignore"):
                t1, t2 = (lo - o) / d, (hi - o) / d
            tmin = np.nanmax(np.minimum(t1, t2))
            tmax = np.nanmin(np.maximum(t1, t2))
            if tmax >= max(tmin, 0.0) and 0.0 < tmin < best:
                best = tmin
        return best if best <= maxi else np.inf

    def tour(self, o, n_h=180, n_v=10, fov_v=36.0):
        az = np.radians(np.arange(n_h) * 360.0 / n_h)
        el = np.radians(np.linspace(-fov_v / 2, fov_v / 2, n_v))
        A, E = np.meshgrid(az, el, indexing="ij")
        d = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1).reshape(-1, 3)
        return d, np.array([self.portee(o, u) for u in d])


MUR = Monde([((2.0, -6.0, 0.0), (2.6, 6.0, 4.0))])       # un rack, à x = 2,0..2,6
VIDE = Monde([])


def _carte_avec_mur():
    c = Carte()
    for y in np.arange(-5.0, 5.1, 0.5):
        o = [0.0, float(y), 1.6]
        c.integre_lidar(o, *MUR.tour(o), t=0.0)
    return c


def _idx(c, p):
    return tuple(c.indice([p])[0])


# ---------------------------------------------------------------- mémoire et vitesse

def test_memoire_de_la_grille():
    c = Carte()
    r = c.resume()
    assert r["cases"] == 84 * 128 * 24
    assert r["octets"] / 1e6 < 2.0            # la grille pleine tient largement
    assert c.g.cell <= 0.25                   # deux cartons voisins (0,5 m) tombent séparés


def test_une_observation_reste_rapide():
    c = Carte()
    d, r = MUR.tour([0.0, 0.0, 1.6])
    t0 = time.perf_counter()
    for _ in range(5):
        c.integre_lidar([0.0, 0.0, 1.6], d, r, t=0.0)
        c.integre_couverture([0.0, 0.0, 1.6], [1.0, 0.0, 0.0], t=0.0)
    ms = (time.perf_counter() - t0) * 1000 / 5
    assert ms < 150.0, f"{ms:.0f} ms par observation, trop lent devant le pas de décision"


# ---------------------------------------------------------------- occupation

def test_un_lidar_ne_connait_que_les_surfaces():
    c = Carte()
    for y in (-3.0, 0.0, 3.0):
        o = [0.0, y, 1.6]
        c.integre_lidar(o, *MUR.tour(o), t=0.0)
    assert c.etat([[2.1, 0.0, 1.6]])[0] == OCCUPE        # la face avant, à x = 2,0
    assert c.etat([[1.0, 0.0, 1.6]])[0] == LIBRE
    assert c.etat([[0.5, 2.0, 1.6]])[0] == LIBRE
    assert c.etat([[4.0, 0.0, 1.6]])[0] == INCONNU       # derrière : rien n'est inventé


def test_une_mesure_aberrante_ne_renverse_pas_une_case():
    c = Carte()
    o = [0.0, 0.0, 1.6]
    d, r = MUR.tour(o)
    for _ in range(6):
        c.integre_lidar(o, d, r, t=0.0)
    c.integre_lidar(o, np.array([[1.0, 0.0, 0.0]]), np.array([1.0]), t=0.0)
    assert c.etat([[1.0, 0.0, 1.6]])[0] == LIBRE


def test_la_carte_reconstruite_colle_a_la_verite():
    """Le test d'exactitude, en séparant les trois choses qu'un lidar peut savoir : le vide
    devant, la surface, et l'ombre derrière."""
    c = _carte_avec_mur()
    ys = np.arange(-3.5, 3.6, 0.25)
    devant = c.etat([[x, y, 1.6] for x in np.arange(0.5, 1.95, 0.25) for y in ys])
    assert (devant != INCONNU).mean() > 0.95, f"{(devant != INCONNU).mean():.0%} du vide connu"
    assert (devant[devant != INCONNU] == LIBRE).all(), "du vide pris pour un obstacle"
    face = c.etat([[2.1, y, 1.6] for y in ys])
    assert (face == OCCUPE).mean() > 0.95, f"{(face == OCCUPE).mean():.0%} de la face trouvée"
    derriere = c.etat([[x, y, 1.6] for x in np.arange(2.75, 3.6, 0.25) for y in ys])
    assert (derriere == INCONNU).all(), "la carte invente ce qui est caché"


# ---------------------------------------------------------------- couverture

def test_la_couverture_suit_la_distance_apparente():
    c = Carte()
    assert c.integre_couverture([0.0, 0.0, 1.6], [1.0, 0.0, 0.0], t=0.0) > 0
    vu = lambda p: c.couverture[_idx(c, p)] > 0
    assert not vu([1.2, 0.0, 1.6])                               # trop près
    assert vu([2.5, 0.0, 1.6])                                   # dans l'enveloppe
    assert vu([3.9, 0.0, 1.6])                                   # 3,9 m de face : lisible
    a = math.radians(27.0)
    assert not vu([3.9 * math.cos(a), 3.9 * math.sin(a), 1.6])   # 4,4 m apparents
    assert not vu([5.0, 0.0, 1.6])                               # trop loin
    assert not vu([-2.5, 0.0, 1.6])                              # derrière soi


def test_la_couverture_retient_de_quel_cote_on_a_regarde():
    """Une caméra qui regarde vers +x rend lisible un panneau qui lui fait face (normale -x),
    pas un panneau tourné vers +x, même si sa case est dans le champ — c'est le cas d'un
    panneau vu à travers un rack depuis l'autre côté."""
    c = Carte()
    c.integre_couverture([0.0, 0.0, 1.6], [1.0, 0.0, 0.0], t=0.0)
    assert c.couvert_pour([[2.5, 0.0, 1.6]], [-1, 0, 0])[0]
    assert not c.couvert_pour([[2.5, 0.0, 1.6]], [1, 0, 0])[0]
    assert not c.couvert_pour([[2.5, 0.0, 1.6]], [0, 1, 0])[0]
    c.integre_couverture([5.0, 0.0, 1.6], [-1.0, 0.0, 0.0], t=1.0)
    assert c.couvert_pour([[2.5, 0.0, 1.6]], [1, 0, 0])[0]


def test_la_couverture_s_arrete_au_mur():
    c = Carte()
    o = [0.0, 0.0, 1.6]
    c.integre_lidar(o, *MUR.tour(o), t=0.0)
    c.integre_couverture(o, [1.0, 0.0, 0.0], t=0.0)
    assert not c.couverture[_idx(c, [3.5, 0.0, 1.6])] > 0


# ---------------------------------------------------------------- panneaux

def test_un_code_lu_deux_fois_au_meme_endroit_reste_un_panneau():
    c = Carte()
    c.integre_lecture("BOX_007", [2.0, 1.0, 1.5], [-1, 0, 0], t=1.0)
    c.integre_lecture("BOX_007", [2.1, 1.0, 1.5], [-1, 0, 0], t=2.0)
    assert len(c.panneaux) == 1 and c.codes == {"BOX_007"}
    assert c.panneaux[0].lectures == 2
    assert abs(c.panneaux[0].position[0] - 2.05) < 1e-6


def test_les_deux_faces_d_un_carton_restent_deux_panneaux():
    """Un carton porte le même code devant et derrière. Les confondre placerait le panneau au
    milieu du carton, à mi-chemin des deux faces."""
    c = Carte()
    c.integre_lecture("BOX_007", [2.0, 1.0, 1.5], [-1, 0, 0], t=1.0)
    c.integre_lecture("BOX_007", [2.72, 1.0, 1.5], [1, 0, 0], t=2.0)
    assert len(c.panneaux) == 2 and c.codes == {"BOX_007"}
    for q in c.panneaux:
        assert min(abs(q.position[0] - 2.0), abs(q.position[0] - 2.72)) < 1e-6


def test_un_carton_n_a_que_deux_faces():
    """Une troisième lecture du même code, loin des deux faces connues, est une erreur de
    décodage : la carte la refuse au lieu de créer un panneau fantôme."""
    c = Carte()
    c.integre_lecture("BOX_007", [2.0, 1.0, 1.5], [-1, 0, 0], t=1.0)
    c.integre_lecture("BOX_007", [2.72, 1.0, 1.5], [1, 0, 0], t=2.0)
    assert c.integre_lecture("BOX_007", [-5.0, 6.0, 3.0], [1, 0, 0], t=3.0) is None
    assert len(c.panneaux) == 2


def test_un_segment_planifie_se_revele_coupe():
    """Le drone découvre un mur en route : le segment déjà planifié doit être déclaré coupé."""
    c = Carte()
    a, b = [0.0, 0.0, 1.6], [4.0, 0.0, 1.6]
    assert c.segment_libre(a, b)
    o = [0.0, 0.0, 1.6]
    c.integre_lidar(o, *MUR.tour(o), t=0.0)
    assert not c.segment_libre(a, b)


def test_deux_reperages_au_meme_endroit_font_une_piste():
    c = Carte()
    c.integre_reperage([2.0, 1.0, 1.5], t=1.0)
    c.integre_reperage([2.2, 1.0, 1.5], t=2.0)
    assert len(c.pistes) == 1 and c.pistes[0].vues == 2


def test_deux_cartons_voisins_restent_deux_pistes():
    c = Carte()
    c.integre_reperage([2.0, 1.0, 1.5], t=1.0)
    c.integre_reperage([2.0, 1.5, 1.5], t=1.0)       # le carton d'à côté, 50 cm plus loin
    assert len(c.pistes) == 2


def test_une_lecture_chasse_la_piste_et_un_reperage_sur_un_lu_ne_cree_rien():
    c = Carte()
    c.integre_reperage([2.0, 1.0, 1.5], t=1.0)
    c.integre_lecture("BOX_007", [2.05, 1.0, 1.5], [-1, 0, 0], t=2.0)
    assert c.pistes == [] and len(c.panneaux) == 1
    c.integre_reperage([2.1, 1.0, 1.5], t=3.0)
    assert c.pistes == []


def test_une_position_absurde_est_refusee():
    """Une fausse détection produit une position 3D aberrante : la carte doit la jeter."""
    c = Carte()
    assert c.integre_reperage([1e6, 0.0, 1.5], t=1.0) is None
    assert c.integre_reperage([float("nan"), 0.0, 1.5], t=1.0) is None
    assert c.integre_lecture("BOX_009", [0.0, 0.0, 1e9], [-1, 0, 0], t=1.0) is None
    assert c.pistes == [] and c.panneaux == []
    c.pistes.append(M.Piste(np.array([1e9, 1e9, 1.5]), None, 1.0))
    M.vue_de_dessus(c)                                    # et le dessin n'explose pas


# ---------------------------------------------------------------- équipe

def test_une_reservation_expire_toute_seule():
    c = Carte()
    c.annonce(0, [0, 0, 1.6], t=0.0)
    c.reserve(0, [3.0, 0.0, 1.5], duree=10.0)
    assert c.reserve_par_un_autre([3.0, 0.0, 1.5], drone=1)
    assert not c.reserve_par_un_autre([3.0, 0.0, 1.5], drone=0)
    c.annonce(0, [0, 0, 1.6], t=11.0)
    c.vieillit(11.0)
    assert not c.reserve_par_un_autre([3.0, 0.0, 1.5], drone=1)


def test_un_drone_muet_libere_ses_cibles():
    """La panne d'un drone est gérée sans une ligne de code qui la surveille."""
    c = Carte()
    for d in (0, 1):
        c.annonce(d, [d, 0, 1.6], t=0.0)
        c.reserve(d, [3.0 + d, 0.0, 1.5], duree=999.0)
    c.annonce(0, [0, 0, 1.6], t=30.0)                     # le drone 1 ne parle plus
    assert c.vieillit(30.0) == [1]
    assert not c.reserve_par_un_autre([4.0, 0.0, 1.5], drone=0)
    assert c.reserve_par_un_autre([3.0, 0.0, 1.5], drone=1)


def test_la_liste_noire_oublie_apres_un_temps():
    c = Carte()
    c.ecarte([3.0, 0.0, 1.5], duree=10.0)
    assert c.est_ecartee([3.0, 0.0, 1.5])
    c.vieillit(11.0)
    assert not c.est_ecartee([3.0, 0.0, 1.5])


# ---------------------------------------------------------------- frontières

def test_les_frontieres_bordent_le_connu():
    c = Carte()
    o = [0.0, 0.0, 1.6]
    c.integre_lidar(o, *VIDE.tour(o), t=0.0)
    f = c.frontieres(1.0, 2.2)
    assert len(f) > 0
    cout = c.couts(1.0, 2.2)
    assert all(cout[tuple(c.indice(p)[0][:2])] == 1.0 for p in f), "une frontière hors du libre"
    d = np.linalg.norm(f[:, :2], axis=1)
    assert d.min() > 5.0, "une frontière au milieu du disque libre"
    assert d.max() < M.PORTEE_CARTE + 0.5


# ---------------------------------------------------------------- chemins

def test_ligne_droite_quand_le_libre_est_connu():
    c = _carte_avec_mur()
    assert c.chemin([0.0, -2.0, 1.6], [0.0, 2.0, 1.6]) == []


def test_le_chemin_contourne_le_mur_par_ses_coins():
    c = _carte_avec_mur()
    pts = c.chemin([1.0, 0.0, 1.6], [4.0, 0.0, 1.6])
    assert pts, "aucun détour proposé alors que le mur bloque"
    cout = c.couts(1.0, 2.2)
    for p in pts:
        assert np.isfinite(cout[tuple(c.indice(p)[0][:2])]), "un point de passage sur un obstacle"
        assert abs(p[2] - 1.6) < 1e-6
    assert len(pts) <= 4, f"{len(pts)} points de passage, un drone n'a besoin que des coins"


def test_le_drone_peut_sortir_de_la_marge_d_un_obstacle():
    """Posé à 30 cm d'un mur, le drone est dans la marge élargie, pas dans le mur : il doit
    pouvoir planifier un chemin qui l'en sort, au lieu d'être déclaré bloqué."""
    c = _carte_avec_mur()
    assert c.chemin([1.7, 0.0, 1.6], [0.0, 3.0, 1.6]) is not None


def test_pas_de_chemin_vers_l_interieur_d_un_obstacle():
    c = _carte_avec_mur()
    assert c.chemin([0.0, 0.0, 1.6], [2.3, 0.0, 1.6]) is None


def test_l_inconnu_reste_traversable():
    """Sinon un drone ne sortirait jamais de la zone qu'il a déjà explorée."""
    c = Carte()
    pts = c.chemin([0.0, 0.0, 1.6], [8.0, 8.0, 1.6])
    assert pts is not None


def test_le_chemin_prefere_le_libre_connu_a_l_inconnu():
    """Un couloir connu en U de 8 m contre une ligne droite de 4 m dans l'inconnu : le drone
    doit prendre le couloir. Couper tout droit, c'est peut-être traverser un rack jamais vu."""
    c = Carte()
    for x in np.arange(2.0, 4.01, 0.25):
        c._ajoute([[x, 0.0, z] for z in (1.25, 1.5, 1.75)], -5.0)
        c._ajoute([[x, 4.0, z] for z in (1.25, 1.5, 1.75)], -5.0)
    for y in np.arange(0.0, 4.01, 0.25):
        c._ajoute([[2.0, y, z] for z in (1.25, 1.5, 1.75)], -5.0)
    depart, arrivee = [4.0, 0.0, 1.6], [4.0, 4.0, 1.6]
    pts = c.chemin(depart, arrivee)
    assert pts, "le chemin coupe tout droit par l'inconnu"
    etapes = [np.array(depart)] + list(pts) + [np.array(arrivee)]
    for a, b in zip(etapes[:-1], etapes[1:]):
        for s in np.linspace(0, 1, 40):
            assert c.etat([a + (b - a) * s])[0] == LIBRE, "le chemin sort du libre connu"


# ---------------------------------------------------------------- sauvegarde et vue

def test_sauver_puis_charger_rend_la_meme_carte(tmp_path):
    c = _carte_avec_mur()
    c.integre_couverture([0.0, 0.0, 1.6], [1.0, 0.0, 0.0], t=1.0)
    c.integre_lecture("BOX_007", [2.0, 1.0, 1.5], [-1, 0, 0], t=2.0)
    c.integre_reperage([2.0, -2.0, 1.5], t=3.0)
    c.sauve(tmp_path / "carte")
    d = Carte.charge(tmp_path / "carte")
    assert np.array_equal(c.occupation, d.occupation)
    assert np.array_equal(c.couverture, d.couverture)
    assert d.codes == {"BOX_007"} and len(d.pistes) == 1 and d.t == 3.0


def test_la_vue_de_dessus_est_une_image_lisible():
    c = _carte_avec_mur()
    c.integre_couverture([0.0, 0.0, 1.6], [1.0, 0.0, 0.0], t=0.0)
    c.integre_lecture("BOX_007", [2.0, 1.0, 1.5], [-1, 0, 0], t=1.0)
    c.integre_reperage([2.0, -2.0, 1.5], t=1.0)
    c.annonce(0, [0.0, 0.0, 1.6], t=1.0)
    c.reserve(0, [1.5, 1.0, 1.5])
    img = M.vue_de_dessus(c, trajectoire=[[0, -2, 1.6], [0, 2, 1.6]])
    assert img.ndim == 3 and img.shape[2] == 3
    assert len({tuple(x) for x in img.reshape(-1, 3)[::37]}) >= 4


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))


def test_le_canal_semantique_se_marque_par_position():
    c = Carte()
    p = np.array([[-5.0, 0.0, 1.5], [-5.0, 0.0, 1.5], [999.0, 0.0, 1.5]])
    assert c.marque(p) == 2                       # le point hors carte est ignoré
    assert (c.semantique == M.SEM_CARTON).sum() == 1
    assert c.semantique[tuple(c.indice(p[:1])[0])] == M.SEM_CARTON


def test_le_premier_obstacle_sur_un_rayon_vient_de_la_carte():
    c = _carte_avec_mur()
    origine = np.array([-5.0, 0.0, 1.5])
    d = c.premier_obstacle(origine, np.array([1.0, 0.0, 0.0]), portee=8.0)
    assert d is not None
    mur = c.centre(np.argwhere(c.occupation > M.SEUIL_OCCUPE))[:, 0].min()
    assert abs((origine[0] + d) - mur) < 0.3               # au cube près
    assert c.premier_obstacle(origine, np.array([-1.0, 0.0, 0.0]), portee=3.0) is None


def test_un_coequipier_est_un_obstacle_pour_les_chemins():
    c = _carte_avec_mur()
    depart, arrivee = np.array([0.0, -4.0, 1.6]), np.array([0.0, 4.0, 1.6])
    assert c.chemin(depart, arrivee) == []                     # ligne droite libre
    c.obstacles_mobiles = [(np.array([0.0, 0.0, 1.6]), 1.5)]
    points = c.chemin(depart, arrivee)
    assert points is not None and len(points) >= 1            # il contourne le coéquipier
    assert all(np.linalg.norm(p[:2]) > 1.2 for p in points)
    c.obstacles_mobiles = []
    assert c.chemin(depart, arrivee) == []


def test_le_sol_vu_par_le_lidar_ne_ferme_pas_le_premier_etage():
    """Le lidar marque occupées les cases de plancher qu'il voit. Sans la règle du sol, un
    drone ne pourrait plus voler à hauteur du premier étage au-dessus d'un plancher déjà
    cartographié : c'est ce qui bloquait le balayage fixe."""
    c = Carte()
    depart, arrivee = np.array([0.0, -4.0, 1.66]), np.array([0.0, 4.0, 1.66])
    sol = np.array([[0.0, y, 0.1] for y in np.arange(-4.0, 4.0, 0.2)])
    c.occupation[tuple(c.indice(sol).T)] = 5.0
    assert c.chemin(depart, arrivee) == []                     # le sol n'est pas une structure
    assert c.pose_atteignable(arrivee)
    haut = np.array([[0.0, y, 1.2] for y in np.arange(-1.0, 1.0, 0.2)])
    c.occupation[tuple(c.indice(haut).T)] = 5.0
    assert c.chemin(depart, arrivee) != []                     # un vrai obstacle se contourne


def test_le_sol_reste_un_obstacle_pour_un_vol_rasant():
    c = Carte()
    sol = np.array([[0.0, y, 0.3] for y in np.arange(-4.0, 4.0, 0.2)])
    c.occupation[tuple(c.indice(sol).T)] = 5.0
    assert not c.pose_atteignable(np.array([0.0, 0.0, 0.9]))   # 0,6 m au-dessus : trop bas
    assert c.pose_atteignable(np.array([0.0, 0.0, 1.66]))      # 1,36 m au-dessus : on passe
