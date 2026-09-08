"""L'œil appris : un petit réseau qui repère les QR et les cartons, sans les lire.

Repérer un carré est plus facile que le déchiffrer, donc le réseau y arrive de beaucoup plus
loin que le décodeur. Ce module ne dépend pas du simulateur : il prend une image et rend des
cadres avec une classe et une confiance. La carte fait le reste — direction du cadre, distance
par le lidar, piste ou carton posé.

Les poids et leurs réglages sont ceux retenus par le banc de l'étape 7 (`experiments/10_detecteur`).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parent / "assets" / "detecteur"
POIDS = ASSETS / "detecteur.pt"
REGLAGES = ASSETS / "detecteur.json"


@dataclass(frozen=True)
class Detection:
    classe: str
    confiance: float
    bbox: np.ndarray          # x0, y0, x1, y1 en pixels

    @property
    def centre(self) -> np.ndarray:
        return np.array([(self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0])

    @property
    def cote_px(self) -> float:
        return float(min(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]))


class Detecteur:
    def __init__(self, poids: str | Path = POIDS, imgsz: int | None = None,
                 conf: float | None = None, device: int | str = 0):
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        from ultralytics import YOLO

        reglages = json.loads(REGLAGES.read_text()) if REGLAGES.exists() else {}
        self.imgsz = int(imgsz or reglages.get("imgsz", 1024))
        self.conf = float(conf if conf is not None else reglages.get("conf", 0.5))
        self.device = device
        self.modele = YOLO(str(poids))
        self.noms = {int(k): str(v) for k, v in self.modele.names.items()}

    def detecte(self, bgr: np.ndarray) -> list[Detection]:
        r = self.modele.predict(bgr, imgsz=self.imgsz, conf=self.conf, half=True,
                                device=self.device, verbose=False)[0]
        b = r.boxes
        return [Detection(self.noms[int(c)], float(s), np.asarray(xyxy, dtype=float))
                for c, s, xyxy in zip(b.cls.tolist(), b.conf.tolist(), b.xyxy.tolist())]
