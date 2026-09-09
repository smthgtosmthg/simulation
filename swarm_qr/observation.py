"""Une observation complète d'un drone : ce que ses capteurs apportent à la carte partagée.

À chaque cycle de mission, pour un drone : le lidar dessine l'espace, les deux caméras
latérales lisent les codes et l'œil appris repère QR et cartons, la couverture note ce qu'une
caméra a vu d'assez près pour lire. Rien ici ne consulte le plan de l'entrepôt : la position
d'un code vient de la direction de l'image et de la distance du lidar, celle d'un motif repéré
de loin vient de la carte.
"""

from __future__ import annotations

import re
import time
from collections import deque

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from . import mapping
from . import perception as P
from .env.config import QR_CFG
from .experiments import _img

INCIDENCE_MAX = 60.0        # degrés ; au-delà, la position déduite des quatre coins n'est plus fiable
# L'inventaire connaît le format de ses étiquettes : l'entrepôt porte d'autres QR (affiches,
# décor) qu'un lecteur inscrirait sinon comme des cartons.
FORMAT_CODE = re.compile("^" + re.escape(QR_CFG.payload_fmt).replace(re.escape("{:03d}"), r"\d{3}") + "$")
PANNEAU_M = 0.40            # taille supposée d'un code pour l'orientation seulement


def _isaac(v_opencv):
    """OpenCV regarde selon z, x à droite, y en bas ; la caméra d'Isaac regarde selon x, y à
    gauche, z en haut."""
    return np.array([v_opencv[2], -v_opencv[0], -v_opencv[1]])


def rayon_du_pixel(pixel, K, cam_R) -> np.ndarray:
    """Direction unitaire, en repère monde, du rayon qui passe par un pixel."""
    rayon_cam = np.array([(pixel[0] - K[0, 2]) / K[0, 0], (pixel[1] - K[1, 2]) / K[1, 1], 1.0])
    return cam_R @ _isaac(rayon_cam / np.linalg.norm(rayon_cam))


def distance_par_le_lidar(origine, direction, nuage):
    """Distance le long d'un rayon de caméra, lue dans le nuage lidar : le point du nuage le
    plus proche du rayon, s'il en passe un à moins de quelques centimètres. Le lidar donne
    la distance, l'image donne la direction et l'identité — la taille de l'étiquette n'a
    plus à être connue."""
    if nuage is None or not len(nuage):
        return None
    v = nuage - origine
    t = v @ direction
    perp = np.linalg.norm(v - t[:, None] * direction[None, :], axis=1)
    perp[t < 0.3] = np.inf
    k = int(np.argmin(perp))
    if perp[k] > min(0.25, 0.06 + 0.04 * t[k]):
        return None
    return float(t[k])


def position_monde(coins, K, cam_pos, cam_R, nuage):
    """Position et normale en monde d'un code lu, ou None hors de l'enveloppe. La distance vient
    du lidar le long du rayon du centre du code ; l'orientation vient des quatre coins et ne
    dépend pas de la taille de l'étiquette."""
    t3, R3 = P.pose_3d(coins, P.taille_code(PANNEAU_M), K)
    if t3 is None:
        return None
    inc = P.incidence_deg(t3, R3[:, 2])
    if min(inc, 180.0 - inc) > INCIDENCE_MAX:
        return None
    rayon = rayon_du_pixel(coins.mean(axis=0), K, cam_R)
    d = distance_par_le_lidar(cam_pos, rayon, nuage)
    if d is None or not (0.3 < d <= mapping.LIRE_MAX):
        return None
    monde = cam_pos + rayon * d
    n = cam_R @ _isaac(R3[:, 2])
    if np.dot(n, cam_pos - monde) < 0:
        n = -n
    return monde, n


def point_vise(pixel, K, cam_pos, cam_R, nuage, carte, portee: float = mapping.PORTEE_CARTE):
    """Point du monde visé par un pixel. C'est tout ce qu'un cadre de l'œil appris donne : une
    direction, sans les quatre coins d'un code. De près, la distance vient du rayon lidar qui
    passe au plus près de la direction ; de loin, les rayons sont trop espacés (28 cm à 8 m) et
    c'est la carte, qui accumule les tours, qui donne le premier obstacle sur la direction."""
    rayon = rayon_du_pixel(pixel, K, cam_R)
    d = distance_par_le_lidar(cam_pos, rayon, nuage)
    if d is None or d > mapping.LIRE_MAX:
        d = carte.premier_obstacle(cam_pos, rayon, portee)
    if d is None or not (0.3 < d <= portee):
        return None
    return cam_pos + rayon * d, rayon


def poses_cameras(scene, drone: int, origine, dirs, portees) -> dict:
    """Les poses des deux caméras et le nuage lidar du même instant : l'image lue au cycle
    suivant est celle de cet instant-là (mesuré à l'étape 2 : une image de retard)."""
    out = {}
    for nom in ("left", "right"):
        p, q = scene.cameras[drone][nom].get_world_pose()
        out[nom] = (np.asarray(p, float), Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]]).as_matrix())
    ok = np.isfinite(portees) & (portees > 0)
    out["nuage"] = origine + dirs[ok] * portees[ok, None]
    return out


class Observateur:
    """Les capteurs d'un drone versés dans la carte partagée, cycle après cycle."""

    def __init__(self, scene, carte: mapping.Carte, K, drone: int = 0, detecteur=None):
        self.scene, self.carte, self.K, self.drone = scene, carte, np.asarray(K, float), drone
        self.detecteur = detecteur
        self.poses = deque(maxlen=2)
        self.trajectoire: list[list[float]] = []
        self.inclinaisons: list[list[float]] = []      # [t, degrés] : une collision se voit à l'inclinaison
        self.compte = {"images": 0, "lectures": 0, "reperages": 0, "lectures_placees": 0,
                       "cartons_reperes": 0}
        self.couts = {"lidar": [], "couverture": [], "decodage": [], "detecteur": []}
        self.cycles = 0

    def observe(self, t: float, oeil: bool = True) -> None:
        """`oeil=False` : ce cycle, l'œil appris ne regarde pas les images de ce drone (les
        drones se le partagent : il coûte 95 ms par image quand la carte graphique rend aussi)."""
        scene, carte, i = self.scene, self.carte, self.drone
        pos = scene.position(i)
        self.trajectoire.append([round(t, 2), *[round(float(x), 3) for x in pos]])
        haut = Rotation.from_quat(scene.drones[i].state.attitude).apply([0.0, 0.0, 1.0])
        self.inclinaisons.append([round(t, 2), round(float(np.degrees(np.arccos(np.clip(haut[2], -1.0, 1.0)))), 1)])
        dirs, portees = scene.lidar(i)
        t0 = time.perf_counter()
        carte.integre_lidar(pos, dirs, portees, t=t)
        self.couts["lidar"].append((time.perf_counter() - t0) * 1000)
        carte.annonce(i, pos, t=t)
        self.poses.append(poses_cameras(scene, i, pos, dirs, portees))
        if len(self.poses) < 2:
            return
        anciennes = self.poses[0]
        nuage = anciennes["nuage"]
        self.cycles += 1
        # l'œil appris coûte 35 ms par image : une caméra par cycle, en alternance, suffit à
        # cinq images par seconde ; la lecture, elle, regarde toujours les deux
        cam_apprise = ("left", "right")[self.cycles % 2]
        for nom in ("left", "right"):
            img = scene.cameras[i][nom].get_rgb()
            if img is None or getattr(img, "ndim", 0) != 3 or not img.size:
                continue
            cam_pos, cam_R = anciennes[nom]
            bgr = _img.to_bgr(img)
            self._lit(bgr, cam_pos, cam_R, nuage, t)
            if self.detecteur is None:
                self._repere_classique(bgr, cam_pos, cam_R, nuage, t)
            elif oeil and nom == cam_apprise:
                self._repere_appris(bgr, cam_pos, cam_R, nuage, t)
            t0 = time.perf_counter()
            carte.integre_couverture(cam_pos, cam_R @ np.array([1.0, 0.0, 0.0]), t=t)
            self.couts["couverture"].append((time.perf_counter() - t0) * 1000)
            self.compte["images"] += 1

    def _lit(self, bgr, cam_pos, cam_R, nuage, t) -> None:
        gris = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        t0 = time.perf_counter()
        for code, coins in P.DECODEURS["zxing"](gris):
            if not FORMAT_CODE.match(code):
                self.compte["codes_etrangers"] = self.compte.get("codes_etrangers", 0) + 1
                continue
            self.compte["lectures"] += 1
            pn = position_monde(coins, self.K, cam_pos, cam_R, nuage)
            if pn is not None:
                self.carte.integre_lecture(code, pn[0], pn[1], t=t)
                self.compte["lectures_placees"] += 1
        self.couts["decodage"].append((time.perf_counter() - t0) * 1000)

    def _repere_classique(self, bgr, cam_pos, cam_R, nuage, t) -> None:
        for quad in P.repere_motifs(bgr):
            self.compte["reperages"] += 1
            pn = position_monde(quad, self.K, cam_pos, cam_R, nuage)
            if pn is not None:
                self.carte.integre_reperage(pn[0], pn[1], t=t)

    def _repere_appris(self, bgr, cam_pos, cam_R, nuage, t) -> None:
        """Un QR repéré devient une piste tournée vers la caméra qui l'a vu : c'est de ce
        côté-là qu'il faudra revenir le lire. Un carton repéré marque le canal sémantique."""
        t0 = time.perf_counter()
        for d in self.detecteur.detecte(bgr):
            pv = point_vise(d.centre, self.K, cam_pos, cam_R, nuage, self.carte)
            if pv is None:
                continue
            pos, rayon = pv
            if d.classe == "qr":
                # la taille du cadre donne une distance grossière (les étiquettes font 0,12 à
                # 0,40 m) ; un point placé deux fois trop loin est passé par un trou du rack
                d_cadre = self.K[0, 0] * P.taille_code(PANNEAU_M) / max(d.cote_px, 1.0)
                dist = float(np.linalg.norm(pos - cam_pos))
                if not (0.45 * d_cadre <= dist <= 1.8 * d_cadre):
                    self.compte["reperages_incoherents"] = self.compte.get("reperages_incoherents", 0) + 1
                    continue
                self.compte["reperages"] += 1
                vers_camera = -np.array([rayon[0], rayon[1], 0.0])
                norme = np.linalg.norm(vers_camera)
                self.carte.integre_reperage(pos, vers_camera / norme if norme > 1e-6 else None, t=t)
            else:
                self.compte["cartons_reperes"] += self.carte.marque(pos)
        self.couts["detecteur"].append((time.perf_counter() - t0) * 1000)
