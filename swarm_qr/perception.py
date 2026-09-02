"""Perception : voir, lire et situer un QR code dans une image.

Trois états possibles pour un panneau présent dans l'image, et le système en a besoin des
trois : LU quand le contenu est décodé, REPERE quand un motif carré est trouvé sans être
lisible — un tag repéré est une cible presque garantie, il suffit de s'en approcher — et RIEN.

Ce module ne dépend pas du simulateur : il prend une image et une calibration, et se teste hors
d'Isaac. La calibration doit venir de `Camera.get_intrinsics_matrix()`, jamais d'un calcul à la
main : la valeur lue et la valeur calculée coïncident aujourd'hui, mais c'est au simulateur de
faire foi.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

MODULES_CODE = 21          # version 1
MODULES_MARGE = 2          # `border` passé à qrcode dans qr_tags
MODULES_PANNEAU = MODULES_CODE + 2 * MODULES_MARGE


def taille_code(taille_panneau: float) -> float:
    """Côté du code seul. Les décodeurs rendent les coins du code, pas ceux du panneau :
    utiliser la taille du panneau donnerait des distances 19 % trop grandes."""
    return taille_panneau * MODULES_CODE / MODULES_PANNEAU


class Etat(str, Enum):
    LU = "lu"
    REPERE = "repere"
    RIEN = "rien"


@dataclass
class Lecture:
    """PIÈGE : `position` et `distance` n'ont de sens qu'à l'état LU.

    À l'état REPERE, ils sont calculés en supposant que le quadrilatère trouvé est un QR de la
    taille attendue — ce qui est une hypothèse, pas une mesure. Le chiffre reste utile pour
    décider vers où aller, mais il ne doit jamais entrer dans une statistique de précision :
    mélanger les deux fait passer une précision de 2,5 cm pour une erreur de 1,7 mètre.
    """

    etat: Etat
    texte: str | None
    coins: np.ndarray | None       # 4x2, ordre haut-gauche puis sens horaire
    position: np.ndarray | None    # 3, dans le repère de la caméra
    distance: float | None
    incidence_deg: float | None

    @property
    def position_fiable(self) -> bool:
        return self.etat is Etat.LU and self.position is not None


def _ordonne(coins: np.ndarray) -> np.ndarray:
    """Coins dans un ordre unique : haut-gauche, haut-droit, bas-droit, bas-gauche.
    Indispensable avant `solvePnP` — pyzbar rend ses coins dans le sens inverse des autres, et
    un ordre inversé donne une pose retournée sans lever la moindre erreur."""
    p = np.asarray(coins, dtype=np.float64).reshape(-1, 2)
    centre = p.mean(axis=0)
    angles = np.arctan2(p[:, 1] - centre[1], p[:, 0] - centre[0])
    p = p[np.argsort(angles)]                      # sens horaire, image en y vers le bas
    depart = int(np.argmin(p[:, 0] + p[:, 1]))     # le plus proche du coin haut-gauche
    return np.roll(p, -depart, axis=0)


# --- les décodeurs, chacun ramené à la même interface ---

def _dec_opencv(gray, aruco: bool):
    det = cv2.QRCodeDetectorAruco() if aruco else cv2.QRCodeDetector()
    try:
        ok, textes, points, _ = det.detectAndDecodeMulti(gray)
    except cv2.error:
        return []
    if not ok or points is None:
        return []
    return [(t, np.asarray(p).reshape(-1, 2)) for t, p in zip(textes, points) if t]


def _dec_zxing(gray):
    import zxingcpp

    out = []
    for r in zxingcpp.read_barcodes(gray):
        p = r.position
        out.append((r.text, np.array([
            [p.top_left.x, p.top_left.y], [p.top_right.x, p.top_right.y],
            [p.bottom_right.x, p.bottom_right.y], [p.bottom_left.x, p.bottom_left.y]], float)))
    return out


def _dec_zbar(gray):
    from pyzbar import pyzbar

    out = []
    for r in pyzbar.decode(gray, symbols=[pyzbar.ZBarSymbol.QRCODE]):
        if len(r.polygon) < 4:
            continue
        out.append((r.data.decode(errors="replace"),
                    np.array([[pt.x, pt.y] for pt in r.polygon[:4]], float)))
    return out


def _dec_boof(gray):
    import pyboof as pb

    det = pb.FactoryFiducial(np.uint8).qrcode()
    det.detect(pb.ndarray_to_boof(np.ascontiguousarray(gray)))
    return [(d.message, np.array([[v.x, v.y] for v in d.bounds.vertexes], float))
            for d in det.detections]


DECODEURS = {
    "opencv": lambda g: _dec_opencv(g, aruco=False),
    "opencv_aruco": lambda g: _dec_opencv(g, aruco=True),
    "opencv_aruco_x3": lambda g: _agrandi(g, lambda x: _dec_opencv(x, aruco=True), 3),
    "zxing": _dec_zxing,
    "zbar": _dec_zbar,
    "pyboof": _dec_boof,
}


def _agrandi(gray, fn, facteur: int):
    """Décode sur une image agrandie et ramène les coins à l'échelle d'origine. C'est ce que
    faisait le test 3 ; on le garde comme colonne séparée pour voir ce que l'agrandissement
    apporte réellement."""
    grand = cv2.resize(gray, None, fx=facteur, fy=facteur, interpolation=cv2.INTER_CUBIC)
    return [(t, c / facteur) for t, c in fn(grand)]


def decode_tous(bgr, decodeurs=("zxing",)) -> dict[str, list[tuple[str, np.ndarray]]]:
    """Chaque décodeur voit exactement la même image en niveaux de gris."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    out = {}
    for nom in decodeurs:
        try:
            out[nom] = DECODEURS[nom](gray)
        except Exception:
            out[nom] = []
    return out


# --- repérage sans décodage ---

def repere_motifs(bgr, cote_min: int = 12) -> list[np.ndarray]:
    """Quadrilatères sombres/clairs qui ressemblent à un QR, sans chercher à le lire. C'est la
    portée de repérage : elle dit jusqu'où un drone sait qu'il y a quelque chose à aller voir."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    seuil = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 31, 5)
    contours, _ = cv2.findContours(seuil, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        aire = cv2.contourArea(c)
        if aire < cote_min * cote_min:
            continue
        approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        q = approx.reshape(-1, 2).astype(float)
        cotes = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
        if min(cotes) < cote_min or max(cotes) / max(min(cotes), 1e-6) > 2.5:
            continue
        masque = np.zeros(gray.shape, np.uint8)
        cv2.fillPoly(masque, [approx], 255)
        part_sombre = float((gray[masque > 0] < 128).mean())
        if 0.20 < part_sombre < 0.70:
            out.append(_ordonne(q))
    return out


# --- position 3D ---

def pose_3d(coins: np.ndarray, cote_m: float, K: np.ndarray,
            normale_attendue: np.ndarray | None = None):
    """Position du centre du code dans le repère de la caméra, par `solvePnP`.

    `IPPE_SQUARE` est la méthode dédiée aux marqueurs plans carrés ; elle rend deux solutions,
    car un carré plan vu de loin est ambigu. On tranche avec la normale attendue quand elle est
    connue, sinon par l'erreur de reprojection.
    """
    demi = cote_m / 2.0
    modele = np.array([[-demi, demi, 0.0], [demi, demi, 0.0],
                       [demi, -demi, 0.0], [-demi, -demi, 0.0]])
    img = _ordonne(coins).astype(np.float64)
    ok, rvecs, tvecs, err = cv2.solvePnPGeneric(
        modele, img, K, np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok or not len(tvecs):
        return None, None

    meilleur = 0
    if normale_attendue is not None and len(rvecs) > 1:
        scores = []
        for r in rvecs:
            R, _ = cv2.Rodrigues(r)
            scores.append(float(np.dot(R[:, 2], normale_attendue)))
        meilleur = int(np.argmax(scores))
    elif len(err) > 1:
        meilleur = int(np.argmin(err))

    t = tvecs[meilleur].reshape(3)
    R, _ = cv2.Rodrigues(rvecs[meilleur])
    return t, R


def incidence_deg(centre_cam: np.ndarray, normale_cam: np.ndarray) -> float:
    """Angle entre l'axe de visée du panneau et la direction caméra → panneau."""
    v = -centre_cam / max(np.linalg.norm(centre_cam), 1e-9)
    c = float(np.clip(np.dot(v, normale_cam / max(np.linalg.norm(normale_cam), 1e-9)), -1.0, 1.0))
    return float(np.degrees(np.arccos(abs(c))))


def lire(bgr, K: np.ndarray, taille_panneau: float, cible: str | None = None,
         decodeur: str = "zxing", normale_attendue=None) -> Lecture:
    """La lecture complète d'une image, telle que le système l'utilisera."""
    cote = taille_code(taille_panneau)
    trouves = decode_tous(bgr, (decodeur,))[decodeur]
    for texte, coins in trouves:
        if cible is not None and texte != cible:
            continue
        t, R = pose_3d(coins, cote, K, normale_attendue)
        d = float(np.linalg.norm(t)) if t is not None else None
        a = incidence_deg(t, R[:, 2]) if t is not None else None
        return Lecture(Etat.LU, texte, _ordonne(coins), t, d, a)

    motifs = repere_motifs(bgr)
    if motifs:
        t, R = pose_3d(motifs[0], cote, K, normale_attendue)
        d = float(np.linalg.norm(t)) if t is not None else None
        return Lecture(Etat.REPERE, None, motifs[0], t, d, None)

    return Lecture(Etat.RIEN, None, None, None, None, None)
