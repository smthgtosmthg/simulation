"""Planner type Pore et al. 2026 : waypoints pré-planifiés + champ de risque + hover-and-scan.

Fidèle à leurs hypothèses : carte et positions des tags connues a priori, allocation
STATIQUE des secteurs entre drones (aucune ré-allocation en cas de panne — c'est le
paradigme qu'on compare), arrêt-scan à chaque point de vue, évitement par répulsion
lidar (leur champ de risque, version réactive).

La logique pure (allocation, tournée, loi de commande) est en fonctions torch sans
dépendance Isaac : testable à sec. `PorePlanner` est l'adaptateur vers l'environnement.
"""

from __future__ import annotations

import math

import torch


# ------------------------------------------------------------------ logique pure

def allocate_sectors(points: torch.Tensor, n_drones: int, iters: int = 12) -> torch.Tensor:
    """K-means 1D-init déterministe → étiquette de secteur par point (N,) ∈ [0, n_drones).

    Secteurs disjoints et équilibrés spatialement : la coordination de Pore
    (Case III : « sector allocator », plages disjointes fixées à l'avance).
    """
    n = points.shape[0]
    if n == 0:
        return torch.zeros(0, dtype=torch.long, device=points.device)
    order = points[:, 0].argsort()
    centers = points[order[torch.linspace(0, n - 1, n_drones, device=points.device).long()], :2].clone()
    for _ in range(iters):
        d = torch.cdist(points[:, :2], centers)
        lab = d.argmin(dim=1)
        for k in range(n_drones):
            m = lab == k
            if m.any():
                centers[k] = points[m, :2].mean(dim=0)
    return lab


def greedy_tour(points: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
    """Tournée gloutonne au plus proche voisin depuis `start` → indices ordonnés (N,).

    Équivalent pratique du serpentin de Pore : un ordre de visite fixe, calculé
    hors-ligne, jamais remis en cause en vol.
    """
    n = points.shape[0]
    if n == 0:
        return torch.zeros(0, dtype=torch.long, device=points.device)
    left = torch.ones(n, dtype=torch.bool, device=points.device)
    order = torch.zeros(n, dtype=torch.long, device=points.device)
    cur = start[:2]
    for i in range(n):
        d = torch.norm(points[:, :2] - cur, dim=-1).masked_fill(~left, float("inf"))
        j = d.argmin()
        order[i] = j
        left[j] = False
        cur = points[j, :2]
    return order


def waypoint_velocity(
    pos: torch.Tensor,        # (B,3) position drone
    yaw: torch.Tensor,        # (B,)
    target: torch.Tensor,     # (B,3) point de vue visé
    target_yaw: torch.Tensor,  # (B,) cap de scan (face au tag)
    lidar72: torch.Tensor,    # (B,72) distances égocentriques normalisées (×8 m)
    v_max: float = 1.5,
    v_scan: float = 0.35,
    slow_radius: float = 1.2,
    risk_radius: float = 1.1,
    k_rep: float = 1.6,
) -> torch.Tensor:
    """Loi de commande (B,4) ∈ [−1,1] : P-contrôleur vers le waypoint + répulsion « champ de risque ».

    Ralentit sous `slow_radius` (hover-and-scan : le gate exige ≤0,6 m/s), vise le
    cap de transit loin du point, le cap de scan près du point.
    """
    B = pos.shape[0]
    delta = target - pos
    dist_xy = torch.norm(delta[:, :2], dim=-1).clamp_min(1e-6)

    speed = torch.where(dist_xy < slow_radius,
                        torch.full_like(dist_xy, v_scan),
                        (0.9 * dist_xy).clamp(max=v_max))
    v_world = delta[:, :2] / dist_xy.unsqueeze(-1) * speed.unsqueeze(-1)

    # répulsion : chaque secteur lidar plus proche que risk_radius pousse en sens inverse
    ang = torch.arange(72, device=pos.device) * (2 * math.pi / 72)          # angles égocentriques
    ang_w = ang.unsqueeze(0) + yaw.unsqueeze(1)                             # angles monde
    d_m = lidar72 * 8.0
    w = ((risk_radius - d_m) / risk_radius).clamp(min=0.0) ** 2             # (B,72)
    rep = -(torch.stack([torch.cos(ang_w), torch.sin(ang_w)], dim=-1) * w.unsqueeze(-1)).sum(dim=1)
    v_world = v_world + k_rep * rep

    cy, sy = torch.cos(yaw), torch.sin(yaw)
    vx_b = v_world[:, 0] * cy + v_world[:, 1] * sy
    vy_b = -v_world[:, 0] * sy + v_world[:, 1] * cy

    vz = (1.2 * delta[:, 2]).clamp(-1.0, 1.0)

    yaw_des = torch.where(dist_xy < slow_radius, target_yaw, torch.atan2(delta[:, 1], delta[:, 0]))
    err = torch.atan2(torch.sin(yaw_des - yaw), torch.cos(yaw_des - yaw))
    yaw_rate = (1.5 * err).clamp(-1.0, 1.0)

    act = torch.stack([vx_b / v_max, vy_b / v_max, vz, yaw_rate], dim=-1).clamp(-1.0, 1.0)
    n_xy = act[:, :2].norm(dim=1, keepdim=True).clamp(min=1.0)
    act = torch.cat([act[:, :2] / n_xy, act[:, 2:]], dim=-1)
    return act


# ------------------------------------------------------------------ adaptateur environnement

class PorePlanner:
    """Contrôleur scripté complet : planifie hors-ligne, exécute, ne s'adapte jamais.

    Par environnement et par drone : liste ordonnée de points de vue (1,0 m devant
    chaque face de tag lisible, cap vers le tag). Un tag lu → ses points de vue
    sont retirés. Timeout par point de vue (anti-blocage). Pas de ré-allocation :
    si un drone meurt, son secteur reste orphelin (fidèle au paradigme).
    """

    def __init__(self, env, scan_dist: float = 1.0, viewpoint_timeout_s: float = 8.0):
        self.env = env
        self.scan_dist = scan_dist
        self.timeout = int(viewpoint_timeout_s * 30)
        B, D = env.num_envs, len(env.possible_agents)
        self.B, self.D = B, D
        self.plans = [[None] * D for _ in range(B)]   # [env][drone] → dict(points, yaws, tags)
        self.idx = torch.zeros(B, D, dtype=torch.long, device=env.device)
        self.hold = torch.zeros(B, D, dtype=torch.long, device=env.device)

    def on_reset(self, env_ids):
        e = self.env
        alt_lo, alt_hi = e.cfg.alt_min + 0.1, e.cfg.alt_max - 0.1
        faces_pos = e._qr_pos_local            # (2n,3)
        faces_nrm = e._qr_normal
        n = e._n_cartons
        for b in env_ids.tolist():
            readable = e._readable[b]           # (n,)
            tags = torch.nonzero(readable).squeeze(-1)
            if tags.numel() == 0:
                for k in range(self.D):
                    self.plans[b][k] = {"pts": torch.zeros(0, 3, device=e.device),
                                        "yaw": torch.zeros(0, device=e.device),
                                        "tag": torch.zeros(0, dtype=torch.long, device=e.device)}
                continue
            fidx = torch.cat([tags * 2, tags * 2 + 1])
            vp = faces_pos[fidx] + faces_nrm[fidx] * self.scan_dist
            vp[:, 2] = vp[:, 2].clamp(alt_lo, alt_hi)
            vyaw = torch.atan2(-faces_nrm[fidx, 1], -faces_nrm[fidx, 0])
            vtag = torch.cat([tags, tags])

            lab = allocate_sectors(vp, self.D)
            for k in range(self.D):
                m = lab == k
                pts, yws, tgs = vp[m], vyaw[m], vtag[m]
                start = e._drones[k].data.root_pos_w[b] - e.scene.env_origins[b]
                order = greedy_tour(pts, start)
                self.plans[b][k] = {"pts": pts[order], "yaw": yws[order], "tag": tgs[order]}
            self.idx[b] = 0
            self.hold[b] = 0

    def _advance(self, b: int, k: int):
        """Saute les points de vue dont le tag est déjà lu ; gère le timeout."""
        plan = self.plans[b][k]
        n = plan["pts"].shape[0]
        i = int(self.idx[b, k])
        while i < n and bool(self.env._read[b, plan["tag"][i]]):
            i += 1
            self.hold[b, k] = 0
        if i < n and int(self.hold[b, k]) > self.timeout:
            i += 1
            self.hold[b, k] = 0
        self.idx[b, k] = min(i, n)

    def act(self) -> dict:
        e = self.env
        actions = {}
        for k, a in enumerate(e.possible_agents):
            pos = e._drones[k].data.root_pos_w - e.scene.env_origins
            yaw = e._yaw[:, k]
            target = pos.clone()
            tyaw = yaw.clone()
            for b in range(self.B):
                self._advance(b, k)
                plan = self.plans[b][k]
                i = int(self.idx[b, k])
                if plan is not None and i < plan["pts"].shape[0]:
                    target[b] = plan["pts"][i]
                    tyaw[b] = plan["yaw"][i]
                    self.hold[b, k] += 1
            act = waypoint_velocity(pos, yaw, target, tyaw, e._lidar_cache[k])
            act[e._dead[:, k]] = 0.0
            actions[a] = act
        return actions
