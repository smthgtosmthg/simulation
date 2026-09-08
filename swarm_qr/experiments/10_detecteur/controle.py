"""Contrôle des cadres avant tout entraînement.

  controle.py --planche jeu/rendu_3 jeu/etape2_optique     dessine les cadres sur des images tirées des jeux
  controle.py --verifie                                    confronte les cadres « qr » de l'étape 2 à la
                                                           projection validée là-bas

Un cadre décalé n'arrête pas l'entraînement : il le fausse en silence. On regarde donc, et on
compare à une mesure indépendante.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from swarm_qr.experiments import _img  # noqa: E402

ETAPE2 = HERE.parent / "07_enveloppe"
COULEURS = {"qr": (0, 220, 0), "carton": (255, 160, 0)}


def charge_jeu(dossier: Path) -> list[dict]:
    return [json.loads(l) for l in (dossier / "manifeste.jsonl").read_text().splitlines() if l.strip()]


def dessine(dossier: Path, m: dict) -> np.ndarray:
    img = cv2.imread(str(dossier / "images" / m["image"]))
    for o in m["objets"]:
        x0, y0, x1, y1 = [int(round(v)) for v in o["bbox"]]
        if o["ignore"]:
            cv2.rectangle(img, (x0, y0), (x1, y1), (128, 128, 128), 1)
            continue
        c = COULEURS[o["classe"]]
        cv2.rectangle(img, (x0, y0), (x1, y1), c, 2)
        if o["classe"] == "qr":
            cv2.putText(img, f"{o['distance']:.1f} m", (x0, max(12, y0 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    n_qr = sum(1 for o in m["objets"] if o["classe"] == "qr" and not o["ignore"])
    n_ca = sum(1 for o in m["objets"] if o["classe"] == "carton" and not o["ignore"])
    return _img.label(img, f"{m['jeu']} {m['image']}  qr {n_qr}  cartons {n_ca}  (gris = ignore)")


def planche(dossiers: list[Path], par_jeu: int, sortie: Path) -> None:
    images = []
    for d in dossiers:
        ms = charge_jeu(d)
        pas = max(1, len(ms) // par_jeu)
        images += [dessine(d, ms[i]) for i in range(0, len(ms), pas)][:par_jeu]
    for k in range(0, len(images), 6):
        p = sortie / f"planche_{k // 6:02d}.jpg"
        _img.board(images[k:k + 6], p, cols=2, cell=760)
        print(f"  {p.name} : {len(images[k:k + 6])} images")


def verifie(jeux: list[str]) -> dict:
    """Pour chaque image de l'étape 2, le panneau visé a une projection validée à quelques
    pixels près. Notre cadre « qr » du même code doit contenir ce centre, avec un côté du
    même ordre. Là où le panneau est dans l'image mais sans cadre, on vérifie que c'est
    parce qu'il est caché, pas parce que l'annotation l'a raté."""
    sys.argv = [sys.argv[0]]
    spec = importlib.util.spec_from_file_location("analyse_etape2", ETAPE2 / "analyse.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    bilan = {}
    for nom in jeux:
        dossier = HERE / "jeu" / f"etape2_{nom}"
        if not dossier.exists():
            continue
        meta = json.loads((ETAPE2 / f"meta_{nom}.json").read_text())
        K = np.array(meta["K"], float)
        ref = json.loads((ETAPE2 / f"poses_{nom}.jsonl").read_text().splitlines()[0])
        dedans, ecarts_cote, sans_cadre, caches, n = 0, [], 0, 0, 0
        for m in charge_jeu(dossier):
            proj = mod.projette_panneau({**ref, "cam_pos": m["cam_pos"], "cam_quat": m["cam_quat"]}, K)
            if proj is None:
                continue
            (u, v), cote_px = proj
            n += 1
            cadres = [o for o in m["objets"] if o["classe"] == "qr" and o.get("code") == m["cible"]]
            vis = [o for o in cadres if not o["ignore"]]
            if not vis:
                if cadres and cadres[0]["visible"] < 0.3:
                    caches += 1
                else:
                    sans_cadre += 1
                continue
            x0, y0, x1, y1 = vis[0]["bbox"]
            if x0 - 2 <= u <= x1 + 2 and y0 - 2 <= v <= y1 + 2:
                dedans += 1
            ecarts_cote.append(abs((x1 - x0) / max(cote_px, 1e-6) - 1.0))
        bilan[nom] = {"panneaux_projetes": n, "cadres": len(ecarts_cote),
                      "centre_dans_le_cadre": round(dedans / max(len(ecarts_cote), 1), 4),
                      "ecart_cote_median": round(float(np.median(ecarts_cote)), 3) if ecarts_cote else None,
                      "caches_par_la_physique": caches, "sans_cadre_inexplique": sans_cadre}
        print(f"  {nom:<10} {n} panneaux projetes : {len(ecarts_cote)} cadres, centre dedans "
              f"{bilan[nom]['centre_dans_le_cadre']:.1%}, ecart de cote median "
              f"{bilan[nom]['ecart_cote_median']}, caches {caches}, sans cadre {sans_cadre}")
    return bilan


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--planche", nargs="*", default=None)
    parser.add_argument("--par-jeu", type=int, default=6)
    parser.add_argument("--sortie", default=str(HERE / "controle"))
    parser.add_argument("--verifie", action="store_true")
    a = parser.parse_args()
    sortie = Path(a.sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    if a.planche is not None:
        planche([Path(p) for p in a.planche], a.par_jeu, sortie)
    if a.verifie:
        b = verifie(["optique", "9019", "vol", "traversee"])
        (sortie / "verification_cadres.json").write_text(json.dumps(b, indent=2))
