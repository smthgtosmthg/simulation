"""Tests SANS Isaac des briques pures de SwarmScan-Map : mapper, layouts, curriculum.

  ~/isaac5_env/bin/python rl_inventory/tests/test_swarmscan_map_pure.py
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from rl_inventory.swarmscan_map.config_map import MAP_CFG, CurriculumConfig, GateConfig, LayoutConfig, MapConfig
from rl_inventory.swarmscan_map.curriculum import GateCurriculum
from rl_inventory.swarmscan_map.layouts import LayoutGenerator
from rl_inventory.config_rl import CFG
from rl_inventory.swarmscan_map.mapping import SwarmMapper

CFG_MAX_LIN_VEL = CFG.action.max_lin_vel_mps


def _mapper(B=2, D=2):
    cfg = MapConfig()
    return SwarmMapper(cfg, B, D, "cpu"), cfg


def _update(m, pos, yaw, hits=None, scan_ok=True, alive=None):
    B, D = m.B, m.D
    R = 8
    if hits is None:
        hits = torch.zeros(B, D, R, 3)
        valid = torch.zeros(B, D, R, dtype=torch.bool)
    else:
        valid = torch.ones(B, D, hits.shape[2], dtype=torch.bool)
    return m.update(
        pos=pos, yaw=yaw, hits=hits, hit_valid=valid,
        scan_ok=torch.full((B, D), scan_ok, dtype=torch.bool),
        scan_range=4.0, fov_deg=60.0,
        alive=alive if alive is not None else torch.ones(B, D, dtype=torch.bool),
    )


def test_mapper_scan_and_occupancy():
    m, cfg = _mapper()
    pos = torch.zeros(2, 2, 3)
    pos[:, 1, 1] = 8.0                      # drone 1 ailleurs
    pos[..., 2] = 1.0                       # bande 0
    yaw = torch.zeros(2, 2)                 # cap +x → caméras vers ±y
    wall = torch.zeros(2, 2, 8, 3)
    wall[..., 0] = torch.linspace(-0.5, 0.5, 8)
    wall[..., 1] = 2.0                      # mur d'impacts à y=2, à GAUCHE du drone 0
    wall[:, 1, :, 1] += 8.0
    c = _update(m, pos, yaw, hits=wall)

    assert m.occ.sum() > 0, "occupation vide"
    gx = int((0.0 - m.x0) / cfg.cell_m)
    gy = int((2.0 - m.y0) / cfg.cell_m)
    assert m.occ[0, gy, gx] == 1.0, "impact lidar non enregistré"
    assert m.scan[0, 0].sum() > 0, "empreinte de scan vide (bande 0)"
    assert m.scan[0, 0, gy, gx] == 1.0, "cellule latérale gauche non scannée"
    row0 = int((0.0 - m.y0) / cfg.cell_m)   # rangée y=0 : devant/derrière le drone
    ahead = m.scan[0, 0, row0, int((1.0 - m.x0) / cfg.cell_m):]
    behind = m.scan[0, 0, row0, : int((-1.0 - m.x0) / cfg.cell_m)]
    assert ahead.sum() == 0 and behind.sum() == 0, "scan devant/derrière (cônes latéraux violés)"
    assert c["new"][0, 0] > 0 and c["facade"][0, 0] > 0, "comptes new/facade nuls"
    assert c["facade"][0, 0] <= c["new"][0, 0]

    c2 = _update(m, pos, yaw, hits=wall)
    assert c2["new"][0, 0] == 0, "re-scan compté comme nouveau"
    assert c2["overlap"][0, 0] == 0, \
        "taxe d'overlap sur SA PROPRE empreinte : c'était une taxe constante d'être vivant"
    print("ok  mapper scan/occupation/overlap (cônes latéraux)")


def test_mapper_overlap_equipe_seulement():
    """La taxe d'overlap ne doit punir que le travail en double AVEC UN COÉQUIPIER."""
    m, _ = _mapper(B=1, D=2)
    pos = torch.zeros(1, 2, 3)
    pos[..., 2] = 1.0                       # les 2 drones au MÊME endroit
    yaw = torch.zeros(1, 2)
    _update(m, pos, yaw)                    # 1er passage : tout est neuf, pas d'overlap
    c = _update(m, pos, yaw)                # 2e passage : les deux re-balayent les mêmes cellules
    assert c["overlap"][0, 0] > 0 and c["overlap"][0, 1] > 0, "overlap d'équipe non détecté"

    m2, _ = _mapper(B=1, D=2)
    pos2 = pos.clone()
    pos2[:, 1, 1] = 10.0                    # drones séparés
    _update(m2, pos2, yaw)
    c2 = _update(m2, pos2, yaw)
    assert c2["overlap"][0, 0] == 0, "overlap facturé alors que les drones sont séparés"
    print("ok  mapper overlap = travail en double d'ÉQUIPE uniquement")


def test_mapper_bilateral_cone():
    m, cfg = _mapper(B=1, D=1)
    pos = torch.zeros(1, 1, 3)
    pos[..., 2] = 1.0
    _update(m, pos, torch.zeros(1, 1))      # cap +x

    def cell(x, y):
        return m.scan[0, 0, int((y - m.y0) / cfg.cell_m), int((x - m.x0) / cfg.cell_m)].item()

    assert cell(0.0, 2.0) == 1.0, "cône gauche absent"
    assert cell(0.0, -2.0) == 1.0, "cône droit absent"
    assert cell(2.0, 0.0) == 0.0, "l'avant est couvert (il ne doit plus l'être)"
    assert cell(-2.0, 0.0) == 0.0, "l'arrière est couvert"
    print("ok  mapper cônes bilatéraux (gauche+droite oui, avant/arrière non)")


def test_mapper_marginal_agglutination():
    m, _ = _mapper()
    pos = torch.zeros(2, 2, 3)
    pos[..., 2] = 1.0                       # 2 drones au MÊME endroit
    yaw = torch.zeros(2, 2)
    c = _update(m, pos, yaw)
    assert c["new"][0, 0] > 0
    assert c["marginal"][0, 0] == 0 and c["marginal"][0, 1] == 0, \
        "contribution marginale non nulle pour des drones agglutinés"
    m2, _ = _mapper()
    pos2 = pos.clone()
    pos2[:, 1, 1] = 10.0                    # drones séparés
    c2 = _update(m2, pos2, yaw)
    assert c2["marginal"][0, 0] > 0 and c2["marginal"][0, 1] > 0
    print("ok  mapper contribution marginale (agglutiné=0, séparés>0)")


def test_mapper_ego_rotation():
    m, cfg = _mapper(B=1, D=1)
    m.qr_read[0, int((0.0 - m.y0) / cfg.cell_m), int((2.0 - m.x0) / cfg.cell_m)] = 1.0
    pos = torch.zeros(1, 1, 3)
    P, C = cfg.crop_px, cfg.n_channels
    qr_ch = 2 + cfg.n_bands + 1             # occ, explored, scan×nb, frontière, QR-LUS

    def crop_qr(yaw_val):
        maps = m.ego_maps(pos, torch.full((1, 1), yaw_val))
        return maps.view(len(cfg.crop_spans_m), C, P, P)[0, qr_ch]

    face = crop_qr(0.0)
    dos = crop_qr(3.14159)
    r_face = int(torch.nonzero(face > 0.1)[:, 0].float().mean())
    r_dos = int(torch.nonzero(dos > 0.1)[:, 0].float().mean())
    assert r_face < P // 2 <= r_dos, f"rotation égocentrique fausse (avant={r_face}, arrière={r_dos})"
    print("ok  mapper crops égocentriques (avant en haut, rotation correcte)")


def test_layouts_split_and_determinism():
    lg = LayoutGenerator(LayoutConfig(), n_tags=123, device="cpu")
    train, val, test = (set(lg.config_ids(s)) for s in ("train", "val", "test"))
    assert not (train & val) and not (train & test) and not (val & test), "splits non disjoints"
    ids = torch.tensor([510, 510, 525])
    a, p = lg.masks(ids)
    assert torch.equal(a[0], a[1]) and torch.equal(p[0], p[1]), "même config id → masques différents"
    assert not torch.equal(a[0], a[2]), "configs distinctes identiques"
    assert (p & ~a).sum() == 0, "tag pré-lu mais inactif"
    frac = a.float().mean(dim=1)
    assert ((frac >= 0.55) & (frac <= 1.0)).all(), f"fraction active hors bornes: {frac}"
    print("ok  layouts split gelé + déterminisme + préread ⊆ actifs")


def test_curriculum_adr():
    g = GateConfig()
    cur = GateCurriculum(g, CurriculumConfig(min_episodes_per_notch=10))
    t0 = cur.thresholds()
    assert t0["read_distance_m"] == g.adr_read_distance_m[0], "gate initial non tolérant"
    assert t0["dwell_steps"] == float(g.dwell_steps), \
        "le dwell est une propriété du capteur (2 images calibrées), jamais un cran de curriculum"
    # La vitesse mesurée par le gate est la norme 3D (speed_uses_vz) : le maximum atteignable
    # est sqrt(v_plan² + v_z²) = sqrt(2)·CFG_MAX_LIN_VEL, pas CFG_MAX_LIN_VEL.
    v_max_3d = math.sqrt(2.0) * CFG_MAX_LIN_VEL
    # INVARIANT D'UN ADR : au cran 0 la contrainte doit être MORTE (on isole la géométrie),
    # au nominal elle doit être VIVANTE. L'ancienne assertion exigeait l'inverse au cran 0 et
    # a verrouillé 34 runs au niveau 0 : seules 22 % des actions y passaient le gate de vitesse.
    assert t0["max_speed_mps"] >= v_max_3d, \
        "le cran 0 contraint déjà la vitesse : géométrie et vitesse doivent s'apprendre ensemble"
    assert g.max_speed_mps < CFG_MAX_LIN_VEL, \
        "le cap NOMINAL doit être une vraie contrainte, sinon la tâche est vide"
    for _ in range(30):
        cur.on_episodes_end([1.0] * 10)
    assert cur.level > 0, "ADR ne monte pas malgré le succès"
    while not cur.nominal:
        cur.on_episodes_end([1.0] * 10)
    tn = cur.thresholds()
    assert tn["read_distance_m"] == g.read_distance_m and tn["view_angle_deg"] == g.view_angle_deg
    assert tn["dwell_steps"] == float(g.dwell_steps)
    assert cur.dropout_prob() >= 0.0
    lvl = cur.level
    for _ in range(200):
        cur.on_episodes_end([0.0] * 10)
    assert cur.level < lvl, "ADR ne redescend pas après effondrement"
    print("ok  curriculum ADR bidirectionnel (tolérant → nominal → recul)")


def test_curriculum_promotion_atteignable():
    """La barre de promotion doit être franchissable par une politique IMPARFAITE.

    Le test qui manquait : pendant 34 runs la barre valait EMA > 0.88 tenue 25 épisodes
    alors que la performance mesurée plafonnait à 0.39 de moyenne. Le curriculum n'a jamais
    quitté le niveau 0, donc progress = 0, donc le gate nominal n'a jamais été entraîné.
    """
    c = CurriculumConfig()
    g = GateConfig()
    hi, _ = GateCurriculum(g, c).thresholds_hi_lo()
    barre = hi + c.promo_margin
    torch.manual_seed(0)
    # politique honnête mais imparfaite : taux de lecture moyen 0.70, très variable
    conc = 8.0
    for moyenne in (0.70, 0.75):
        ech = torch.distributions.Beta(moyenne * conc, (1 - moyenne) * conc).sample((64, 6000))
        ema = torch.zeros(64)
        run = torch.zeros(64)
        promu = torch.zeros(64, dtype=torch.bool)
        for t in range(ech.shape[1]):
            ema = (1 - c.ema_alpha) * ema + c.ema_alpha * ech[:, t]
            run = torch.where(ema > barre, run + 1, torch.zeros_like(run))
            promu |= run >= c.confirm_episodes
        taux = float(promu.float().mean())
        assert taux > 0.9, (
            f"barre de promotion {barre:.2f} infranchissable : une politique à {moyenne:.2f} "
            f"de moyenne ne promeut que dans {taux:.0%} des cas — le curriculum est mort"
        )
    print(f"ok  barre de promotion {barre:.2f} franchissable par une politique imparfaite")


def test_couverture_pese_face_aux_lectures():
    """La couverture est la récompense DENSE de la conception v2 : elle doit peser.

    Mesuré sur le run socle_v2 : couvrir toute l'arène rapportait 124 points contre
    111 tags × 25 = 2775 pour les lectures, parce que scan_norm suivait la taille du cône.
    """
    m = MAP_CFG.map
    r = MAP_CFG.reward
    cellules_bande = ((m.bounds_x_m[1] - m.bounds_x_m[0]) * (m.bounds_y_m[1] - m.bounds_y_m[0])) / m.cell_m**2
    # hypothèse prudente : 30 % de façades, contribution marginale sur tout le neuf
    par_cellule = (0.30 * r.facade_gain + 0.70 * r.area_gain + r.marginal_gain) / r.scan_norm
    couverture_totale = cellules_bande * par_cellule
    lectures_totales = 111 * r.new_qr
    assert couverture_totale > 0.2 * lectures_totales, (
        f"couvrir une bande entière vaut {couverture_totale:.0f} pts contre {lectures_totales:.0f} pts "
        f"de lectures : le signal dense est invisible, la tâche redevient une récompense éparse"
    )
    print(f"ok  couverture d'une bande = {couverture_totale:.0f} pts vs lectures = {lectures_totales:.0f} pts")


def test_gates_de_couverture_non_contraignants():
    """scan_speed/scan_yawrate sont documentés « non contraignants » : le vérifier."""
    v_max_3d = math.sqrt(2.0) * CFG.action.max_lin_vel_mps
    assert MAP_CFG.map.scan_speed_mps >= v_max_3d, (
        f"scan_speed_mps={MAP_CFG.map.scan_speed_mps} < vitesse 3D max {v_max_3d:.3f} : "
        f"la carte de couverture se coupe en vol rapide, la mémoire spatiale devient fausse"
    )
    assert MAP_CFG.map.scan_yawrate_rps >= CFG.action.max_yaw_rate_rps
    print(f"ok  gates de couverture non contraignants (v3D max {v_max_3d:.3f} m/s)")


def test_dims_coherence():
    from rl_inventory.swarmscan_map.config_map import MAP_CFG as m
    n_scales = len(m.map.crop_spans_m)
    assert m.map.n_channels == 6 + m.map.n_bands
    assert m.map.obs_dim == n_scales * m.map.n_channels * 32 * 32
    print(f"ok  dims: {n_scales} échelles × {m.map.n_channels} canaux × 32² = {m.map.obs_dim}")


if __name__ == "__main__":
    test_mapper_scan_and_occupancy()
    test_mapper_bilateral_cone()
    test_mapper_overlap_equipe_seulement()
    test_mapper_marginal_agglutination()
    test_mapper_ego_rotation()
    test_layouts_split_and_determinism()
    test_curriculum_adr()
    test_dims_coherence()
    print("\nTOUS LES TESTS PURS PASSENT")
