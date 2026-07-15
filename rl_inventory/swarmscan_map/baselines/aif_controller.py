"""Active Inference : sélection d'action par minimisation de l'énergie libre attendue G.

Réécrit à zéro pour l'arène SwarmScan (l'ancienne implémentation `scripts/12_*` n'est
pas réutilisée). Même principe que la conception AIF du projet (docs/doc.md) :
G(candidat) = − valeur épistémique (cellules de façade encore non scannées que le
candidat révélerait) − valeur pragmatique (se rapprocher de la frontière de travail)
+ coût de mouvement + pénalité de risque de collision ; choix par argmin (éval
déterministe) ou softmax(−G/T).

Information : AUCUNE position de tag. La croyance = les grilles construites en ligne
par l'essaim à partir de ses capteurs (mêmes grilles que la mémoire de la politique
RL : occupation, couverture-scan par bande — même base d'information, comparaison
équitable). La logique est en fonctions torch pures, testables sans Isaac.
"""

from __future__ import annotations

import math

import torch

N_HEADINGS = 8
LOOKAHEAD_S = 1.6


def candidate_positions(pos: torch.Tensor, v_fast: float = 1.2) -> torch.Tensor:
    """(B,3) → (B, K, 2) positions projetées des K = N_HEADINGS + 1 (hover) candidats."""
    B = pos.shape[0]
    ang = torch.arange(N_HEADINGS, device=pos.device) * (2 * math.pi / N_HEADINGS)
    step = v_fast * LOOKAHEAD_S
    off = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1) * step        # (K-1,2)
    off = torch.cat([off, torch.zeros(1, 2, device=pos.device)], dim=0)       # + hover
    return pos[:, :2].unsqueeze(1) + off.unsqueeze(0)


def epistemic_value(
    cand_xy: torch.Tensor,     # (B,K,2)
    work: torch.Tensor,        # (B,H,W) cellules de façade NON scannées (croyance)
    cell_x: torch.Tensor,      # (1,H,W) coordonnées monde des cellules
    cell_y: torch.Tensor,
    radius: float = 2.5,
) -> torch.Tensor:
    """(B,K) : combien de cellules de travail chaque candidat mettrait à portée de scan."""
    dx = cell_x.unsqueeze(1) - cand_xy[..., 0].unsqueeze(-1).unsqueeze(-1)    # (B,K,H,W)
    dy = cell_y.unsqueeze(1) - cand_xy[..., 1].unsqueeze(-1).unsqueeze(-1)
    within = (dx * dx + dy * dy) <= radius * radius
    count = (within & (work.unsqueeze(1) > 0.5)).flatten(2).sum(-1).float()
    return count / 52.0   # normalisé par l'empreinte de scan (~52 cellules) : commensurable avec risque/coûts


def pragmatic_value(cand_xy: torch.Tensor, nearest_work_xy: torch.Tensor) -> torch.Tensor:
    """(B,K) : − distance du candidat à la cellule de travail la plus proche (se rapprocher paye)."""
    return -torch.norm(cand_xy - nearest_work_xy.unsqueeze(1), dim=-1)


def collision_risk(lidar72: torch.Tensor, yaw: torch.Tensor, safe: float = 1.2) -> torch.Tensor:
    """(B,72),(B,) → (B,K) : risque par candidat = proximité d'obstacle dans sa direction (±25°)."""
    B = lidar72.shape[0]
    d_m = lidar72 * 8.0
    sector_w = torch.arange(72, device=lidar72.device) * (2 * math.pi / 72) + yaw.unsqueeze(1)  # (B,72) angles monde
    ang = torch.arange(N_HEADINGS, device=lidar72.device) * (2 * math.pi / N_HEADINGS)
    risks = []
    for a in ang:
        diff = torch.atan2(torch.sin(sector_w - a), torch.cos(sector_w - a)).abs()
        cone = diff < math.radians(25)
        dmin = d_m.masked_fill(~cone, 8.0).min(dim=1).values
        risks.append(((safe - dmin) / safe).clamp(min=0.0) ** 2)
    risks.append(torch.zeros(B, device=lidar72.device))                       # hover : sans risque
    return torch.stack(risks, dim=-1)


def expected_free_energy(
    epi: torch.Tensor, prag: torch.Tensor, risk: torch.Tensor,
    w_epi: float = 1.0, w_prag: float = 0.35, w_risk: float = 30.0, move_cost: float = 0.5,
) -> torch.Tensor:
    """G (B,K) — plus bas = meilleur. Le hover (dernier candidat) paye le coût de mouvement en négatif."""
    K = epi.shape[1]
    move = torch.full_like(epi, move_cost)
    move[:, K - 1] = 0.0
    return -w_epi * epi - w_prag * prag + w_risk * risk + move


def nearest_work_cell(work: torch.Tensor, cell_x, cell_y, pos: torch.Tensor):
    """(B,H,W) → (B,2) coordonnées de la cellule de travail la plus proche, (B,) sa distance,
    (B,) vrai s'il reste du travail."""
    B = work.shape[0]
    d2 = (cell_x - pos[:, 0].view(B, 1, 1)) ** 2 + (cell_y - pos[:, 1].view(B, 1, 1)) ** 2
    d2 = d2.masked_fill(work < 0.5, float("inf"))
    flat = d2.flatten(1)
    j = flat.argmin(dim=1)
    has = torch.isfinite(flat.gather(1, j.unsqueeze(1)).squeeze(1))
    H, W = work.shape[1:]
    xy = torch.stack([cell_x.flatten()[j % (H * W)], cell_y.flatten()[j % (H * W)]], dim=-1)
    dist = flat.gather(1, j.unsqueeze(1)).squeeze(1).sqrt().nan_to_num(posinf=99.0)
    return xy, dist, has


class ActiveInferenceController:
    """Contrôleur AIF complet : croyance (grilles de l'essaim) → G par candidat → commande vitesse."""

    def __init__(self, env, temperature: float = 0.0):
        self.env = env
        self.T = temperature
        m = env._mapper
        self.cell_x, self.cell_y = m.cell_x, m.cell_y

    def _work_grid(self, band: torch.Tensor) -> torch.Tensor:
        """Cellules de façade non scannées dans la bande d'altitude du drone (B,H,W)."""
        m = self.env._mapper
        occ_dil = torch.nn.functional.max_pool2d(m.occ.unsqueeze(1), 3, 1, 1).squeeze(1)
        B = m.B
        scan_band = m.scan[torch.arange(B, device=band.device), band]
        return ((occ_dil > 0.5) & (scan_band < 0.5)).float()

    def act(self) -> dict:
        e = self.env
        actions = {}
        for k, a in enumerate(e.possible_agents):
            pos = e._drones[k].data.root_pos_w - e.scene.env_origins
            yaw = e._yaw[:, k]
            band = (torch.bucketize(pos[:, 2].contiguous(), e._mapper._band_edges) - 1).clamp(0, e._mapper.cfg.n_bands - 1)
            work = self._work_grid(band)
            # le travail restant toutes bandes confondues sert de repli (changer d'étage)
            m = e._mapper
            occ_dil = torch.nn.functional.max_pool2d(m.occ.unsqueeze(1), 3, 1, 1).squeeze(1)
            any_work = ((occ_dil > 0.5) & (m.scan.min(dim=1).values < 0.5)).float()
            use = torch.where(work.flatten(1).sum(-1, keepdim=True) > 0,
                              work.flatten(1), any_work.flatten(1)).view_as(work)

            cand = candidate_positions(pos)
            epi = epistemic_value(cand, use, self.cell_x, self.cell_y)
            near_xy, near_d, _ = nearest_work_cell(use, self.cell_x, self.cell_y, pos)
            prag = pragmatic_value(cand, near_xy)
            risk = collision_risk(e._lidar_cache[k], yaw)
            G = expected_free_energy(epi, prag, risk)
            if self.T > 0:
                probs = torch.softmax(-G / self.T, dim=-1)
                choice = torch.multinomial(probs, 1).squeeze(-1)
            else:
                choice = G.argmin(dim=-1)

            ang = torch.arange(N_HEADINGS + 1, device=e.device) * (2 * math.pi / N_HEADINGS)
            head = ang[choice.clamp(max=N_HEADINGS - 1)]
            is_hover = choice == N_HEADINGS
            near = near_d < 2.0
            speed = torch.where(near, torch.full_like(near_d, 0.35), torch.full_like(near_d, 1.2))
            speed = torch.where(is_hover, torch.zeros_like(speed), speed)
            vx_w, vy_w = torch.cos(head) * speed, torch.sin(head) * speed
            cy, sy = torch.cos(yaw), torch.sin(yaw)
            vx_b = vx_w * cy + vy_w * sy
            vy_b = -vx_w * sy + vy_w * cy

            # altitude : viser le milieu de la bande qui contient encore du travail le plus proche
            band_mid = torch.tensor([0.9, 2.25, 3.7], device=e.device)
            has_here = work.flatten(1).sum(-1) > 0
            tgt_z = torch.where(has_here, band_mid[band], band_mid[(band + 1) % 3])
            vz = (1.0 * (tgt_z - pos[:, 2])).clamp(-1.0, 1.0)

            # cap : face à la cellule de travail visée (près), sinon face au déplacement
            to_work = torch.atan2(near_xy[:, 1] - pos[:, 1], near_xy[:, 0] - pos[:, 0])
            yaw_des = torch.where(near, to_work, head)
            err = torch.atan2(torch.sin(yaw_des - yaw), torch.cos(yaw_des - yaw))
            yaw_rate = (1.5 * err).clamp(-1.0, 1.0)

            act = torch.stack([vx_b / 1.5, vy_b / 1.5, vz, yaw_rate], dim=-1).clamp(-1.0, 1.0)
            n_xy = act[:, :2].norm(dim=1, keepdim=True).clamp(min=1.0)
            act = torch.cat([act[:, :2] / n_xy, act[:, 2:]], dim=-1)
            act[e._dead[:, k]] = 0.0
            actions[a] = act
        return actions

    def on_reset(self, env_ids):
        pass  # la croyance vit dans le mapper de l'env, remis à zéro par l'env lui-même
