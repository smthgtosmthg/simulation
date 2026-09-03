"""Tests SANS Isaac de la logique pure des baselines (Pore + Active Inference).

  ~/isaac5_env/bin/python rl_inventory/tests/test_baselines_pure.py
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from rl_inventory.swarmscan_map.baselines.pore_planner import allocate_sectors, greedy_tour, waypoint_velocity
from rl_inventory.swarmscan_map.baselines.aif_controller import (
    candidate_positions, collision_risk, epistemic_value, expected_free_energy,
    nearest_work_cell, pragmatic_value, N_HEADINGS,
)


def test_sectors_disjoints_et_complets():
    pts = torch.rand(60, 3) * torch.tensor([30.0, 20.0, 3.0]) - torch.tensor([15.0, 10.0, 0.0])
    lab = allocate_sectors(pts, 3)
    assert lab.shape == (60,) and set(lab.tolist()) <= {0, 1, 2}, "étiquettes invalides"
    assert all((lab == k).sum() > 0 for k in range(3)), "un secteur est vide"
    centers = torch.stack([pts[lab == k, :2].mean(0) for k in range(3)])
    own = torch.norm(pts[:, :2] - centers[lab], dim=1)
    other = torch.cdist(pts[:, :2], centers).min(dim=1).values
    assert torch.allclose(own, other, atol=1e-4), "un point est plus proche du centre d'un AUTRE secteur"
    print("ok  allocation de secteurs (disjoints, complets, cohérents)")


def test_tournee_gloutonne():
    pts = torch.tensor([[5.0, 0, 1], [1.0, 0, 1], [3.0, 0, 1], [9.0, 0, 1]])
    order = greedy_tour(pts, start=torch.tensor([0.0, 0.0, 1.0]))
    assert order.tolist() == [1, 2, 0, 3], f"ordre glouton faux : {order.tolist()}"
    assert sorted(order.tolist()) == [0, 1, 2, 3], "la tournée ne visite pas tout"
    print("ok  tournée gloutonne (plus proche voisin, complète)")


def test_loi_de_commande_converge_et_scan():
    pos = torch.tensor([[0.0, 0.0, 1.0]])
    yaw = torch.zeros(1)
    target = torch.tensor([[4.0, 0.0, 1.5]])
    tyaw = torch.tensor([math.pi])
    lidar = torch.ones(1, 72)                      # rien autour
    act = waypoint_velocity(pos, yaw, target, tyaw, lidar)
    assert act[0, 0] > 0.5 and abs(act[0, 1]) < 0.2, "ne fonce pas vers le waypoint devant lui"
    assert act[0, 2] > 0.3, "ne monte pas vers l'altitude cible"
    near = waypoint_velocity(torch.tensor([[3.8, 0.0, 1.5]]), yaw, target, tyaw, lidar)
    v_cmd = float(torch.norm(near[0, :2])) * 1.5
    assert v_cmd <= 0.6, f"trop rapide près du point de scan ({v_cmd:.2f} m/s > gate 0.6)"
    assert abs(float(near[0, 3])) > 0.5, "ne tourne pas vers le cap de scan"
    wall = torch.ones(1, 72)
    wall[0, 0:4] = 0.05                            # mur droit devant (secteurs égo 0-20°)
    rep = waypoint_velocity(pos, yaw, target, tyaw, wall)
    assert rep[0, 0] < act[0, 0] - 0.2, "le champ de risque ne freine pas face au mur"
    print("ok  loi de commande (P-contrôleur, ralenti de scan ≤0.6, répulsion)")


def _grid(B=1, H=40, W=40, cell=0.25):
    xs = (torch.arange(W) + 0.5) * cell - W * cell / 2
    ys = (torch.arange(H) + 0.5) * cell - H * cell / 2
    cx = xs.view(1, 1, W).expand(1, H, W)
    cy = ys.view(1, H, 1).expand(1, H, W)
    return cx, cy


def test_aif_prefere_le_travail():
    cx, cy = _grid()
    work = torch.zeros(1, 40, 40)
    work[0, :, 32:] = 1.0                          # travail à l'EST (x > +3 m)
    pos = torch.zeros(1, 3)
    cand = candidate_positions(pos)
    assert cand.shape == (1, N_HEADINGS + 1, 2)
    epi = epistemic_value(cand, work, cx, cy)
    near_xy, near_d, has = nearest_work_cell(work, cx, cy, pos)
    assert bool(has[0]) and near_xy[0, 0] > 2.5, "cellule de travail la plus proche mal localisée"
    prag = pragmatic_value(cand, near_xy)
    risk = torch.zeros(1, N_HEADINGS + 1)
    G = expected_free_energy(epi, prag, risk)
    best = int(G.argmin(dim=-1))
    ang = best * 2 * math.pi / N_HEADINGS
    assert math.cos(ang) > 0.7, f"l'AIF ne va pas vers le travail (candidat {best})"
    print("ok  AIF : G minimal dans la direction du travail restant")


def test_aif_evite_les_murs():
    cx, cy = _grid()
    work = torch.zeros(1, 40, 40)
    work[0, :, 32:] = 1.0                          # travail à l'est…
    lidar = torch.ones(1, 72)
    lidar[0, 0:6] = 0.04                           # …mais mur juste devant (est, yaw=0) à ~0.3 m
    risk = collision_risk(lidar, torch.zeros(1))
    assert risk[0, 0] > 0.5, "le mur devant n'est pas perçu comme risqué"
    pos = torch.zeros(1, 3)
    cand = candidate_positions(pos)
    epi = epistemic_value(cand, work, cx, cy)
    near_xy, _, _ = nearest_work_cell(work, cx, cy, pos)
    G = expected_free_energy(epi, pragmatic_value(cand, near_xy), risk)
    best = int(G.argmin(dim=-1))
    assert best != 0, "l'AIF fonce dans le mur malgré le risque"
    print("ok  AIF : la pénalité de collision domine face à un mur")


def test_aif_hover_quand_tout_est_fait():
    cx, cy = _grid()
    work = torch.zeros(1, 40, 40)                  # plus aucun travail
    pos = torch.zeros(1, 3)
    cand = candidate_positions(pos)
    epi = epistemic_value(cand, work, cx, cy)
    assert epi.abs().sum() == 0
    near_xy, _, has = nearest_work_cell(work, cx, cy, pos)
    assert not bool(has[0])
    G = expected_free_energy(epi, torch.zeros(1, N_HEADINGS + 1), torch.zeros(1, N_HEADINGS + 1))
    assert int(G.argmin(dim=-1)) == N_HEADINGS, "sans travail, le hover (coût nul) doit gagner"
    print("ok  AIF : hover quand plus rien à scanner")


if __name__ == "__main__":
    test_sectors_disjoints_et_complets()
    test_tournee_gloutonne()
    test_loi_de_commande_converge_et_scan()
    test_aif_prefere_le_travail()
    test_aif_evite_les_murs()
    test_aif_hover_quand_tout_est_fait()
    print("\nTOUS LES TESTS PURS DES BASELINES PASSENT")
