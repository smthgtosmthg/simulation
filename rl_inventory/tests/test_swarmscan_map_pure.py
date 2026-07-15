"""Tests SANS Isaac des briques pures de SwarmScan-Map : mapper, layouts, curriculum.

  ~/isaac5_env/bin/python rl_inventory/tests/test_swarmscan_map_pure.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from rl_inventory.swarmscan_map.config_map import MAP_CFG, CurriculumConfig, GateConfig, LayoutConfig, MapConfig
from rl_inventory.swarmscan_map.curriculum import GateCurriculum
from rl_inventory.swarmscan_map.layouts import LayoutGenerator
from rl_inventory.swarmscan_map.mapping import SwarmMapper


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
    yaw = torch.zeros(2, 2)                 # face +x
    wall = torch.zeros(2, 2, 8, 3)
    wall[..., 0] = 2.0                      # mur d'impacts à x=2 devant le drone 0
    wall[..., 1] = torch.linspace(-0.5, 0.5, 8)
    wall[:, 1, :, 1] += 8.0
    c = _update(m, pos, yaw, hits=wall)

    assert m.occ.sum() > 0, "occupation vide"
    gx = int((2.0 - m.x0) / cfg.cell_m)
    gy = int((0.0 - m.y0) / cfg.cell_m)
    assert m.occ[0, gy, gx] == 1.0, "impact lidar non enregistré"
    assert m.scan[0, 0].sum() > 0, "empreinte de scan vide (bande 0)"
    behind = m.scan[0, 0, :, : int((-1.0 - m.x0) / cfg.cell_m)]
    assert behind.sum() == 0, "scan derrière le drone (cône violé)"
    assert c["new"][0, 0] > 0 and c["facade"][0, 0] > 0, "comptes new/facade nuls"
    assert c["facade"][0, 0] <= c["new"][0, 0]

    c2 = _update(m, pos, yaw, hits=wall)
    assert c2["new"][0, 0] == 0, "re-scan compté comme nouveau"
    assert c2["overlap"][0, 0] > 0, "overlap non détecté au re-scan"
    print("ok  mapper scan/occupation/overlap")


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
    assert t0["read_distance_m"] == g.adr_read_distance_m[0] and t0["dwell_steps"] == 1.0, "gate initial non tolérant"
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


def test_dims_coherence():
    from rl_inventory.swarmscan_map.config_map import MAP_CFG as m
    n_scales = len(m.map.crop_spans_m)
    assert m.map.n_channels == 6 + m.map.n_bands
    assert m.map.obs_dim == n_scales * m.map.n_channels * 32 * 32
    print(f"ok  dims: {n_scales} échelles × {m.map.n_channels} canaux × 32² = {m.map.obs_dim}")


if __name__ == "__main__":
    test_mapper_scan_and_occupancy()
    test_mapper_marginal_agglutination()
    test_mapper_ego_rotation()
    test_layouts_split_and_determinism()
    test_curriculum_adr()
    test_dims_coherence()
    print("\nTOUS LES TESTS PURS PASSENT")
