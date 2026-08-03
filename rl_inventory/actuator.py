"""Modèle d'actionneur : l'action fixe la vitesse VISÉE, la vitesse réelle la rejoint
avec un retard du premier ordre et une accélération bornée.

Pur torch, aucune dépendance Isaac : testable sans simulateur.

Sans ce modèle, write_root_velocity_to_sim(action × v_max) fait passer le drone de 0 à
1,5 m/s en un pas de contrôle = 45 m/s² = 4,59 g. Un Crazyflie 2.1 plafonne à
g·tan(20°) = 3,57 m/s² (PID_VEL_ROLL_MAX du firmware Bitcraze) ; même à poussée maximale
(poussée/poids = 1,82) le plafond absolu est 14,9 m/s². Les 45 m/s² exigeraient une
assiette de 77,7° et un rapport poussée/poids de 4,69.

Conséquence mesurée de son absence : le bruit d'exploration fabriquait des creux de
vitesse d'un seul pas qui satisfaisaient le gate de lecture sans que le drone ralentisse
(87,9 % des lectures d'une politique aléatoire survenaient juste après une survitesse).
"""

from __future__ import annotations

import torch


def integrate(vel, cmd, dt, *, accel_xy, accel_z, yaw_accel, tau_xy, tau_z, tau_yaw):
    """Un pas d'intégration, en repère MONDE. vel, cmd : (N, 4) = (vx, vy, vz, taux de lacet).

    L'accélération planaire est bornée en NORME : la capacité d'accélération horizontale
    d'un quadrirotor ne dépend que de son inclinaison, pas de son azimut. Borner dans le
    monde équivaut donc exactement à borner dans le corps.
    """
    out = vel.clone()
    acc_xy = (cmd[:, :2] - vel[:, :2]) / tau_xy
    scale = (accel_xy / acc_xy.norm(dim=1, keepdim=True).clamp_min(1e-9)).clamp(max=1.0)
    out[:, :2] = vel[:, :2] + acc_xy * scale * dt
    out[:, 2] = vel[:, 2] + ((cmd[:, 2] - vel[:, 2]) / tau_z).clamp(-accel_z, accel_z) * dt
    out[:, 3] = vel[:, 3] + ((cmd[:, 3] - vel[:, 3]) / tau_yaw).clamp(-yaw_accel, yaw_accel) * dt
    return out


def altitude_envelope(vz, z, alt_min, alt_max, accel_z, max_delta=None):
    """Borne vz par ce qu'un freinage à accel_z permet encore d'arrêter avant la butée.

    Appliquée à l'ÉTAT et pas seulement à la commande : couper la commande laisserait le
    retard du filtre dépasser la butée de v·tau.

    max_delta borne la correction à ce qu'une accélération réelle permet : sans lui, la
    barrière coupait la vitesse d'un coup près du plafond (0,71 g mesuré pour un modèle
    limité à 0,37 g). Suivre l'enveloppe exige exactement accel_z, donc borner à
    accel_z·dt reste cohérent avec elle.
    """
    up = torch.sqrt(2.0 * accel_z * (alt_max - z).clamp_min(0.0))
    down = torch.sqrt(2.0 * accel_z * (z - alt_min).clamp_min(0.0))
    out = torch.maximum(torch.minimum(vz, up), -down)
    if max_delta is not None:
        out = vz + (out - vz).clamp(-max_delta, max_delta)
    return out
