"""Entraînement du détecteur — étape 7.

  entraine.py --variantes n1024,n640,s1024        entraîne les variantes, dans l'ordre
  entraine.py --liste                             montre les variantes disponibles

Apprentissage sur des entrepôts d'entraînement (graines 0 à 5), validation sur les deux
entrepôts scellés (9033 et 9019) que le réseau ne voit jamais pendant l'apprentissage. Le plus
petit modèle qui suffit sera retenu : il doit tourner pendant le vol sans gêner le reste.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
JEU = HERE / "jeu"
RUNS = HERE / "runs"

TRAIN_SEEDS = (0, 1, 2, 3, 4, 5)
TEST_SEEDS = (9033, 9019)
CLASSES = {0: "qr", 1: "carton"}

VARIANTES = {
    "n1024": {"modele": "yolo11n.pt", "imgsz": 1024, "batch": 8, "epochs": 50},
    "n640": {"modele": "yolo11n.pt", "imgsz": 640, "batch": 16, "epochs": 50},
    "s1024": {"modele": "yolo11s.pt", "imgsz": 1024, "batch": 6, "epochs": 40},
}


def ecrit_yaml() -> Path:
    for s in TRAIN_SEEDS + TEST_SEEDS:
        if not (JEU / f"rendu_{s}" / "images").exists():
            raise RuntimeError(f"jeu manquant : rendu_{s}")
    lignes = [f"path: {JEU}", "train:"] + [f"  - rendu_{s}/images" for s in TRAIN_SEEDS]
    lignes += ["val:"] + [f"  - rendu_{s}/images" for s in TEST_SEEDS]
    lignes += ["names:"] + [f"  {k}: {v}" for k, v in CLASSES.items()]
    p = HERE / "jeu.yaml"
    p.write_text("\n".join(lignes) + "\n")
    return p


def entraine(nom: str, yaml: Path) -> dict:
    from ultralytics import YOLO

    v = VARIANTES[nom]
    os.environ["YOLO_AUTOINSTALL"] = "false"
    modele = YOLO(v["modele"])
    t0 = time.perf_counter()
    modele.train(data=str(yaml), imgsz=v["imgsz"], epochs=v["epochs"], batch=v["batch"],
                 patience=12, workers=4, device=0, amp=True, project=str(RUNS), name=nom,
                 exist_ok=True, seed=0, deterministic=False, plots=True, verbose=False,
                 fliplr=0.0, mosaic=1.0, close_mosaic=10)
    duree = time.perf_counter() - t0
    meilleur = RUNS / nom / "weights" / "best.pt"
    m = YOLO(str(meilleur))
    val = m.val(data=str(yaml), imgsz=v["imgsz"], batch=v["batch"], device=0, half=True,
                plots=False, verbose=False, project=str(RUNS), name=f"{nom}_val", exist_ok=True)
    par_classe = {CLASSES[i]: {"map50": round(float(val.box.ap50[k]), 4),
                               "map5095": round(float(val.box.ap[k]), 4),
                               "rappel": round(float(val.box.r[k]), 4),
                               "precision": round(float(val.box.p[k]), 4)}
                  for k, i in enumerate(val.box.ap_class_index)}
    bilan = {"variante": nom, **v, "poids": str(meilleur), "duree_min": round(duree / 60, 1),
             "parametres": sum(p.numel() for p in m.model.parameters()),
             "map50": round(float(val.box.map50), 4), "map5095": round(float(val.box.map), 4),
             "par_classe": par_classe,
             "ms_inference": round(float(val.speed["inference"]), 2)}
    print(json.dumps(bilan, indent=2))
    return bilan


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variantes", default="n1024,n640,s1024")
    parser.add_argument("--liste", action="store_true")
    a = parser.parse_args()
    if a.liste:
        print(json.dumps(VARIANTES, indent=2))
        sys.exit(0)
    yaml = ecrit_yaml()
    RUNS.mkdir(exist_ok=True)
    bilans = []
    fichier = HERE / "entrainement.json"
    if fichier.exists():
        bilans = json.loads(fichier.read_text())
    for nom in [s for s in a.variantes.split(",") if s]:
        bilans = [b for b in bilans if b["variante"] != nom] + [entraine(nom, yaml)]
        fichier.write_text(json.dumps(bilans, indent=2))
    print("ENTRAINEMENT FINI")
