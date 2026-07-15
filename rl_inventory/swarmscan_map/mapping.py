"""Mémoire spatiale de l'essaim : grilles globales par env + crops égocentriques multi-échelles.

Pur torch (aucune dépendance Isaac) : tout est testable sans simulateur.
Les grilles ne contiennent QUE ce que les capteurs ont vu — jamais de position de QR non lu.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .config_map import MapConfig


class SwarmMapper:
    """Grilles partagées par l'essaim (comm parfaite) : occupation, exploré, scan par bande,
    QR lus, trajectoires par drone — et leur découpe égocentrique pour l'observation."""

    def __init__(self, cfg: MapConfig, num_envs: int, num_drones: int, device: torch.device | str):
        self.cfg = cfg
        self.B, self.D = num_envs, num_drones
        self.device = torch.device(device)
        self.W, self.H = cfg.grid_wh
        self.x0, self.x1 = cfg.bounds_x_m
        self.y0, self.y1 = cfg.bounds_y_m

        xs = self.x0 + (torch.arange(self.W, device=self.device) + 0.5) * cfg.cell_m
        ys = self.y0 + (torch.arange(self.H, device=self.device) + 0.5) * cfg.cell_m
        self.cell_x = xs.view(1, 1, self.W).expand(1, self.H, self.W)
        self.cell_y = ys.view(1, self.H, 1).expand(1, self.H, self.W)

        nb = cfg.n_bands
        self.occ = torch.zeros(self.B, self.H, self.W, device=self.device)
        self.explored = torch.zeros(self.B, self.H, self.W, device=self.device)
        self.scan = torch.zeros(self.B, nb, self.H, self.W, device=self.device)
        self.qr_read = torch.zeros(self.B, self.H, self.W, device=self.device)
        self.traj = torch.zeros(self.B, self.D, self.H, self.W, device=self.device)
        self._band_edges = torch.tensor(cfg.height_bands_m, device=self.device)

    def reset(self, env_ids: torch.Tensor):
        self.occ[env_ids] = 0.0
        self.explored[env_ids] = 0.0
        self.scan[env_ids] = 0.0
        self.qr_read[env_ids] = 0.0
        self.traj[env_ids] = 0.0

    def _cell_idx(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gx = ((x - self.x0) / self.cfg.cell_m).long().clamp(0, self.W - 1)
        gy = ((y - self.y0) / self.cfg.cell_m).long().clamp(0, self.H - 1)
        inside = (x >= self.x0) & (x < self.x1) & (y >= self.y0) & (y < self.y1)
        return gx, gy, inside

    def update(
        self,
        pos: torch.Tensor,        # (B,D,3) positions locales env
        yaw: torch.Tensor,        # (B,D)
        hits: torch.Tensor,       # (B,D,R,3) impacts lidar locaux env
        hit_valid: torch.Tensor,  # (B,D,R) impact réel (< portée max)
        scan_ok: torch.Tensor,    # (B,D) gate vitesse/lacet satisfait
        scan_range: float,
        fov_deg: float,
        alive: torch.Tensor,      # (B,D)
    ) -> dict[str, torch.Tensor]:
        """Met à jour toutes les grilles ; renvoie les comptes de cellules pour la récompense."""
        B, D = self.B, self.D
        alive_f = alive.float()

        # trajectoires (décroissance exponentielle + tampon de position)
        self.traj *= self.cfg.traj_decay
        gx, gy, _ = self._cell_idx(pos[..., 0], pos[..., 1])
        bd = torch.arange(B, device=self.device).view(B, 1).expand(B, D)
        dd = torch.arange(D, device=self.device).view(1, D).expand(B, D)
        self.traj[bd[alive], dd[alive], gy[alive], gx[alive]] = 1.0

        # occupation : impacts lidar de tous les drones vivants
        h = hits[alive]                                   # (K,R,3)
        v = hit_valid[alive]
        if h.numel() > 0:
            hx, hy = h[..., 0][v], h[..., 1][v]
            ogx, ogy, ins = self._cell_idx(hx, hy)
            envs = bd[alive].view(-1, 1).expand_as(hit_valid[alive])[v]
            flat = envs[ins] * (self.H * self.W) + ogy[ins] * self.W + ogx[ins]
            self.occ.view(-1).index_fill_(0, flat, 1.0)

        # exploré : disque autour de chaque drone vivant
        r2 = self.cfg.explored_radius_m ** 2
        for k in range(D):
            dx = self.cell_x - pos[:, k, 0].view(B, 1, 1)
            dy = self.cell_y - pos[:, k, 1].view(B, 1, 1)
            disk = ((dx * dx + dy * dy) <= r2) & alive[:, k].view(B, 1, 1)
            self.explored = torch.maximum(self.explored, disk.float())

        # empreinte de scan : cône caméra × portée × gate vitesse, dans la bande d'altitude du drone
        occ_dil = F.max_pool2d(self.occ.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
        band = (torch.bucketize(pos[..., 2].contiguous(), self._band_edges) - 1).clamp(0, self.cfg.n_bands - 1)
        cos_half = math.cos(math.radians(fov_deg))
        stamps, news, facades = [], [], []
        for k in range(D):
            dx = self.cell_x - pos[:, k, 0].view(B, 1, 1)
            dy = self.cell_y - pos[:, k, 1].view(B, 1, 1)
            dist = torch.sqrt(dx * dx + dy * dy).clamp_min(1e-6)
            c, s = torch.cos(yaw[:, k]).view(B, 1, 1), torch.sin(yaw[:, k]).view(B, 1, 1)
            in_cone = (dx * c + dy * s) / dist >= cos_half
            ok = (scan_ok[:, k] & alive[:, k]).view(B, 1, 1)
            stamp = (dist <= scan_range) & in_cone & ok
            prev = self.scan[bd[:, 0], band[:, k]]        # (B,H,W) bande courante du drone k
            new = stamp & (prev < 0.5)
            stamps.append(stamp)
            news.append(new)
            facades.append(new & (occ_dil > 0.5))

        union_new = torch.zeros(B, self.H, self.W, dtype=torch.bool, device=self.device)
        counts = {
            "new": torch.zeros(B, D, device=self.device),
            "facade": torch.zeros(B, D, device=self.device),
            "marginal": torch.zeros(B, D, device=self.device),
            "overlap": torch.zeros(B, D, device=self.device),
        }
        for k in range(D):
            others = torch.zeros_like(union_new)
            for j in range(D):
                if j != k:
                    others |= news[j]
            counts["new"][:, k] = news[k].flatten(1).sum(-1).float()
            counts["facade"][:, k] = facades[k].flatten(1).sum(-1).float()
            counts["marginal"][:, k] = (news[k] & ~others).flatten(1).sum(-1).float()
            counts["overlap"][:, k] = (stamps[k] & ~news[k]).flatten(1).sum(-1).float() * alive_f[:, k]
            union_new |= news[k]

        for k in range(D):
            self.scan[bd[:, 0], band[:, k]] = torch.maximum(
                self.scan[bd[:, 0], band[:, k]], stamps[k].float()
            )
        return counts

    def mark_read(self, env_idx: torch.Tensor, xy: torch.Tensor):
        """Mémorise l'emplacement des QR lus (connaissance acquise à la lecture)."""
        if env_idx.numel() == 0:
            return
        gx, gy, ins = self._cell_idx(xy[:, 0], xy[:, 1])
        flat = env_idx[ins] * (self.H * self.W) + gy[ins] * self.W + gx[ins]
        self.qr_read.view(-1).index_fill_(0, flat, 1.0)

    def frontier(self) -> torch.Tensor:
        """Bord connu/inconnu de la zone explorée."""
        unexplored = 1.0 - self.explored
        near_unknown = F.max_pool2d(unexplored.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
        return ((self.explored > 0.5) & (near_unknown > 0.5)).float()

    def ego_maps(self, pos: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        """Crops égocentriques multi-échelles (B, D, S*C*P*P), drone au centre, avant vers le haut."""
        B, D, P = self.B, self.D, self.cfg.crop_px
        common = torch.cat(
            [
                self.occ.unsqueeze(1),
                self.explored.unsqueeze(1),
                self.scan,
                self.frontier().unsqueeze(1),
                self.qr_read.unsqueeze(1),
            ],
            dim=1,
        )                                                     # (B, 3+nb, H, W)
        mates = self.traj.sum(dim=1, keepdim=True) - self.traj  # (B,D,H,W)
        full = torch.cat(
            [
                common.unsqueeze(1).expand(B, D, common.shape[1], self.H, self.W),
                self.traj.unsqueeze(2),
                mates.clamp(0.0, 1.0).unsqueeze(2),
            ],
            dim=2,
        ).reshape(B * D, self.cfg.n_channels, self.H, self.W)

        px = pos[..., 0].reshape(B * D, 1, 1)
        py = pos[..., 1].reshape(B * D, 1, 1)
        c = torch.cos(yaw).reshape(B * D, 1, 1)
        s = torch.sin(yaw).reshape(B * D, 1, 1)
        lin = (torch.arange(P, device=self.device).float() + 0.5) / P - 0.5   # (-0.5, 0.5)
        u = (-lin).view(1, P, 1)                                              # avant en haut du crop
        v = lin.view(1, 1, P)                                                 # droite vers la droite
        crops = []
        for span in self.cfg.crop_spans_m:
            wx = px + span * (u * c + v * s)
            wy = py + span * (u * s - v * c)
            xn = 2.0 * (wx - self.x0) / (self.x1 - self.x0) - 1.0
            yn = 2.0 * (wy - self.y0) / (self.y1 - self.y0) - 1.0
            grid = torch.stack([xn.expand(B * D, P, P), yn.expand(B * D, P, P)], dim=-1)
            crops.append(F.grid_sample(full, grid, mode="bilinear", padding_mode="zeros", align_corners=False))
        return torch.cat(crops, dim=1).reshape(B, D, -1)
