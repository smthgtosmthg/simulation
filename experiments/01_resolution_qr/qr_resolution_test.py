"""
Mesure 01 — À quelle résolution un QR code reste-t-il lisible ?

Question : un world model qui encode l'image en 64x64 peut-il voir un QR code ?

Méthode : on génère le MÊME QR que le projet (version 1, correction H, bordure 2),
on le place dans une scène à une distance donnée, on rend l'image à la résolution
voulue, et on essaie de le décoder. On répète avec des décalages sous-pixel et de
petites rotations, parce que le décodage y est très sensible.

Aucun simulateur, aucun GPU. Tourne en quelques minutes sur le processeur.
"""

import csv
import math
from pathlib import Path

import cv2
import numpy as np
import qrcode
from qrcode.constants import ERROR_CORRECT_H

OUT = Path(__file__).parent
FOV_DEG = 60.0
QR_VERSION = 1
BORDER = 2
PAYLOAD = "Item#A14-R5"  # même format que la littérature entrepôt (rack, étagère, rangée)
TRIALS = 30
RNG = np.random.default_rng(0)

try:
    from pyzbar.pyzbar import decode as zbar_decode
    HAS_ZBAR = True
except ImportError:
    HAS_ZBAR = False


def make_qr(payload=PAYLOAD):
    """Reproduit exactement l'appel du projet (scripts/qr_code_system.py).
    Attention : `fit=True` peut relever la version si la charge utile ne rentre pas."""
    qr = qrcode.QRCode(version=QR_VERSION, error_correction=ERROR_CORRECT_H,
                       box_size=20, border=BORDER)
    qr.add_data(payload)
    qr.make(fit=True)
    img = np.array(qr.make_image(fill_color="black", back_color="white").convert("L"))
    total = img.shape[0] // 20
    return img, qr.version, total


QR_MASTER, QR_REAL_VERSION, MODULES = make_qr()


def render_patch(width_px, rot_deg, dx, dy, blur_sigma, noise_std):
    """Rend le QR à `width_px` pixels de large, sur un fond de carton, avec les
    dégradations d'une vraie caméra : flou optique, bruit capteur, sous-pixel."""
    pad = max(8, int(width_px * 0.8))
    canvas = int(width_px + 2 * pad)

    # rendu haute résolution puis réduction : c'est ce que fait un capteur réel
    # (facteur adapté pour que l'image intermédiaire reste raisonnable)
    ss = max(2, min(8, int(2600 / max(canvas, 1))))
    hi = cv2.resize(QR_MASTER, (int(width_px * ss), int(width_px * ss)),
                    interpolation=cv2.INTER_NEAREST)
    scene_hi = np.full((canvas * ss, canvas * ss), 205, np.uint8)  # carton beige clair
    y0 = x0 = pad * ss
    scene_hi[y0:y0 + hi.shape[0], x0:x0 + hi.shape[1]] = hi

    if abs(rot_deg) > 1e-6:
        c = (scene_hi.shape[1] / 2, scene_hi.shape[0] / 2)
        M = cv2.getRotationMatrix2D(c, rot_deg, 1.0)
        scene_hi = cv2.warpAffine(scene_hi, M, scene_hi.shape[::-1],
                                  flags=cv2.INTER_LINEAR, borderValue=205)

    # décalage sous-pixel : la phase d'échantillonnage change tout
    M = np.float32([[1, 0, dx * ss], [0, 1, dy * ss]])
    scene_hi = cv2.warpAffine(scene_hi, M, scene_hi.shape[::-1],
                              flags=cv2.INTER_LINEAR, borderValue=205)

    img = cv2.resize(scene_hi, (canvas, canvas), interpolation=cv2.INTER_AREA)

    if blur_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur_sigma)
    if noise_std > 0:
        img = np.clip(img.astype(np.float32) + RNG.normal(0, noise_std, img.shape),
                      0, 255).astype(np.uint8)
    return img


_det = cv2.QRCodeDetector()
_det_aruco = cv2.QRCodeDetectorAruco() if hasattr(cv2, "QRCodeDetectorAruco") else None


def try_read(img):
    """Renvoie (décodé, repéré). 'Repéré' = on a trouvé le motif sans le lire."""
    decoded = detected = False
    for det in (d for d in (_det_aruco, _det) if d is not None):
        try:
            txt, pts, _ = det.detectAndDecode(img)
            if txt:
                decoded = True
            if pts is not None and len(pts) > 0:
                detected = True
        except cv2.error:
            pass
        if not detected:
            try:
                ok, _ = det.detect(img)
                detected = detected or bool(ok)
            except cv2.error:
                pass
    if not decoded and HAS_ZBAR:
        try:
            if zbar_decode(img):
                decoded = detected = True
        except Exception:
            pass
    return decoded, detected


def measure(width_px, trials=TRIALS, blur_sigma=0.6, noise_std=2.0):
    dec = det = 0
    for i in range(trials):
        img = render_patch(width_px,
                           rot_deg=float(RNG.uniform(-3, 3)),
                           dx=float(RNG.uniform(-0.5, 0.5)),
                           dy=float(RNG.uniform(-0.5, 0.5)),
                           blur_sigma=blur_sigma, noise_std=noise_std)
        d, t = try_read(img)
        dec += d
        det += t
    return dec / trials, det / trials


def px_width(panel_m, dist_m, res_px):
    """Largeur apparente du panneau, en pixels, pour une caméra de FOV horizontal donné."""
    visible_m = 2.0 * dist_m * math.tan(math.radians(FOV_DEG) / 2.0)
    return panel_m * res_px / visible_m


def threshold_from(rows, key, floor=0.5):
    """Plus petite largeur (px/module) où le taux passe et reste au-dessus de `floor`."""
    ok = [r for r in rows if r[key] >= floor]
    return min(r["px_per_module"] for r in ok) if ok else None


def main():
    print(f"Charge utile : {PAYLOAD!r} — version demandée {QR_VERSION}, "
          f"version RÉELLE {QR_REAL_VERSION} (fit=True l'a relevée)")
    print(f"→ {MODULES} modules de large, marge comprise")
    print(f"Décodeurs : OpenCV{'+Aruco' if _det_aruco else ''}"
          f"{'+pyzbar' if HAS_ZBAR else ''} | {TRIALS} essais par point\n")

    # ---------- A. le seuil, en pixels par module ----------
    print("=" * 66)
    print("A. SEUIL DE LISIBILITÉ (indépendant de toute distance)")
    print("=" * 66)
    print(f"{'largeur QR':>11} {'px/module':>10} {'décodé':>9} {'repéré':>9}")
    rows = []
    for w in [12, 16, 20, 25, 30, 35, 40, 50, 60, 75, 90, 110, 140, 175, 220]:
        d, t = measure(w)
        rows.append({"panel_px": w, "px_per_module": round(w / MODULES, 2),
                     "decode": d, "detect": t})
        print(f"{w:>9} px {w / MODULES:>10.2f} {d:>8.0%} {t:>8.0%}")

    k_dec = threshold_from(rows, "decode")
    k_det = threshold_from(rows, "detect")
    print(f"\n  Seuil de DÉCODAGE  : {k_dec} pixels par module")
    print(f"  Seuil de REPÉRAGE  : {k_det} pixels par module")
    if k_dec and k_det:
        print(f"  → repérer est {k_dec / k_det:.2f}× plus facile que décoder (en distance)")

    # ---------- B. taille physique du panneau, déduite de la calibration ----------
    print("\n" + "=" * 66)
    print("B. TAILLE PHYSIQUE DU QR, DÉDUITE DE docs/calibration_gate.csv")
    print("=" * 66)
    print("  Mesure du projet : à 1280 px et FOV 60°, ça décode à 1,50 m, pas à 1,75 m.")
    if k_dec:
        lo = k_dec * MODULES * 2 * 1.50 * math.tan(math.radians(FOV_DEG) / 2) / 1280
        hi = k_dec * MODULES * 2 * 1.75 * math.tan(math.radians(FOV_DEG) / 2) / 1280
        print(f"  → le panneau mesure entre {lo * 100:.1f} cm et {hi * 100:.1f} cm de côté.")
        panel_m = (lo + hi) / 2
    else:
        panel_m = 0.11
    print(f"  → on retient {panel_m * 100:.1f} cm pour la suite.")

    # ---------- C. le vrai test : 64x64 contre pleine résolution ----------
    print("\n" + "=" * 66)
    print("C. LE TEST : PORTÉE DE LECTURE SELON LA RÉSOLUTION D'ENTRÉE")
    print("=" * 66)
    resolutions = [64, 96, 128, 256, 512, 1280]
    dists = [0.05, 0.075, 0.10, 0.15, 0.25, 0.4, 0.6, 0.9, 1.25, 1.5, 2.0, 3.0]
    out = []
    print(f"{'distance':>9} | " + " ".join(f"{r:>6}px" for r in resolutions))
    print("-" * 66)
    for d_m in dists:
        line = f"{d_m:>7.2f} m | "
        for res in resolutions:
            w = px_width(panel_m, d_m, res)
            rate, det = (measure(w, trials=12) if w >= 6 else (0.0, 0.0))
            out.append({"distance_m": d_m, "resolution_px": res,
                        "panel_px": round(w, 1), "px_per_module": round(w / MODULES, 3),
                        "decode_rate": rate, "detect_rate": det})
            line += f"{rate:>7.0%} "
        print(line)

    print("\n  Portée maximale de décodage (taux ≥ 50 %) :")
    ranges = {}
    for res in resolutions:
        ok = [r["distance_m"] for r in out if r["resolution_px"] == res and r["decode_rate"] >= 0.5]
        ranges[res] = max(ok) if ok else 0.0
        print(f"    {res:>5} px  →  {ranges[res]:.3f} m" if ranges[res] else
              f"    {res:>5} px  →  jamais décodé")

    # ---------- D. la conclusion chiffrée ----------
    print("\n" + "=" * 66)
    print("D. CONCLUSION")
    print("=" * 66)
    if k_dec:
        need = k_dec * MODULES * 2 * 1.25 * math.tan(math.radians(FOV_DEG) / 2) / panel_m
        print(f"  Pour décoder à 1,25 m, il faut une image de {need:.0f} px de large.")
        print(f"  Un world model en 64 px est {need / 64:.0f}× en dessous.")
    if ranges.get(1280) and ranges.get(64) is not None:
        r64 = ranges[64]
        print(f"  Portée à 1280 px : {ranges[1280]:.2f} m")
        print(f"  Portée à   64 px : {r64:.3f} m" if r64 else "  Portée à   64 px : nulle")

    with open(OUT / "resultats_seuil.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(OUT / "resultats_portee.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"\n  CSV écrits dans {OUT}")


if __name__ == "__main__":
    main()
