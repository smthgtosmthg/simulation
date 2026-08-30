"""Écriture des images, planches et vidéos. Aucun lien avec le simulateur : ce module
s'importe partout, y compris hors d'Isaac."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


def to_bgr(rgb) -> np.ndarray:
    arr = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)


def label(img: np.ndarray, text: str, bottom: bool = False) -> np.ndarray:
    out = img.copy()
    h = out.shape[0]
    y0, y1 = (h - 34, h) if bottom else (0, 34)
    cv2.rectangle(out, (0, y0), (out.shape[1], y1), (0, 0, 0), -1)
    cv2.putText(out, text, (10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def board(images: list[np.ndarray], path: Path, cols: int = 3, cell: int = 420) -> Path:
    if not images:
        raise ValueError("aucune image")
    rows = math.ceil(len(images) / cols)
    tiles = []
    for img in images:
        h, w = img.shape[:2]
        s = cell / max(h, w)
        tiles.append(cv2.resize(img, (int(w * s), int(h * s))))
    th = max(t.shape[0] for t in tiles)
    tw = max(t.shape[1] for t in tiles)
    canvas = np.full((rows * th, cols * tw, 3), 24, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas[r * th : r * th + t.shape[0], c * tw : c * tw + t.shape[1]] = t
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return path


def save(img: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def video(frames: list[np.ndarray], path: Path, fps: int = 10) -> Path:
    if not frames:
        raise ValueError("aucune image pour la vidéo")
    h, w = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("impossible d'ouvrir l'encodeur vidéo")
    for f in frames:
        writer.write(cv2.resize(f, (w, h)))
    writer.release()
    return path


def side_by_side(*imgs: np.ndarray, gap: int = 8) -> np.ndarray:
    h = max(i.shape[0] for i in imgs)

    def fit(img):
        s = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * s), h))

    parts = []
    for i, img in enumerate(imgs):
        if i:
            parts.append(np.full((h, gap, 3), 40, np.uint8))
        parts.append(fit(img))
    return np.hstack(parts)
