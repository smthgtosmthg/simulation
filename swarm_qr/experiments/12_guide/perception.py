"""Que voit vraiment le modèle sur le plan, et que lui coûte la compression ?

Six questions dont la réponse est calculée depuis les données, donc vérifiable sans discussion :
compter les repères, trouver le plus à gauche, le plus à droite, le plus haut, le plus bas, et le
voisin le plus proche d'un repère donné. Aucune n'a de rapport avec la mission : on mesure la
vue, pas le jugement.

Chaque question est posée sur le plan tel qu'il est enregistré, puis sur le plan redessiné, et
pour chaque version du modèle : sans compression, en 8 bits, en 4 bits, et en 4 bits avec
l'encodeur d'images laissé en 16 bits.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))

from swarm_qr.guide import Guide                                        # noqa: E402
from banc import instantanes                                           # noqa: E402
from variantes import melange, plan_lisible, positions_plan_origine, _note  # noqa: E402

DECOR = ("The image is a map of a warehouse seen from above. A few positions are marked with a "
         "red circle containing a white number. ")
ECART_MIN = 25          # px : deux repères plus proches que ça rendent la question ambiguë


def questions(reperes: dict) -> list[dict]:
    """Les six questions et leurs réponses, calculées depuis les pixels des repères."""
    nums = sorted(reperes)
    xs = {n: reperes[n][0] for n in nums}
    ys = {n: reperes[n][1] for n in nums}
    ancre = nums[len(nums) // 2]
    voisins = sorted((float(np.hypot(xs[n] - xs[ancre], ys[n] - ys[ancre])), n)
                     for n in nums if n != ancre)
    out = [
        {"cle": "compter", "texte": "How many numbered circles are on the image?",
         "reponse": len(nums), "ambigu": False},
        {"cle": "gauche", "texte": "Which number is on the leftmost circle?",
         "reponse": min(nums, key=lambda n: xs[n]),
         "ambigu": _serre(sorted(xs.values()))},
        {"cle": "droite", "texte": "Which number is on the rightmost circle?",
         "reponse": max(nums, key=lambda n: xs[n]),
         "ambigu": _serre(sorted(xs.values(), reverse=True))},
        {"cle": "haut", "texte": "Which number is on the topmost circle?",
         "reponse": min(nums, key=lambda n: ys[n]),
         "ambigu": _serre(sorted(ys.values()))},
        {"cle": "bas", "texte": "Which number is on the bottommost circle?",
         "reponse": max(nums, key=lambda n: ys[n]),
         "ambigu": _serre(sorted(ys.values(), reverse=True))},
        {"cle": "voisin", "texte": f"Which numbered circle is closest to the circle numbered {ancre}?",
         "reponse": voisins[0][1],
         "ambigu": len(voisins) > 1 and voisins[1][0] - voisins[0][0] < ECART_MIN},
    ]
    return out


def _serre(valeurs: list[float]) -> bool:
    return len(valeurs) > 1 and abs(valeurs[1] - valeurs[0]) < ECART_MIN


VERSIONS = {
    "sans compression": {"device": "auto", "quantisation": None},
    "8 bits": {"device": "cuda", "quantisation": "8bit"},
    "4 bits": {"device": "cuda", "quantisation": "4bit"},
    "4 bits, vision en 16": {"device": "cuda", "quantisation": "4bit-vision16"},
}


def profil(nom_version: str, modele: str, cas: list[dict]) -> dict:
    import torch

    reglages = VERSIONS[nom_version]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    g = Guide(modele, max_tokens=16, **reglages)
    chargement = time.perf_counter() - t0
    memoire = torch.cuda.max_memory_allocated() / 1e6
    justes = {"origine": 0, "lisible": 0}
    poses = {"origine": 0, "lisible": 0}
    details, latences = [], []
    for k, c in enumerate(cas):
        vue = cv2.imread(c["vue"])
        zones, _, _ = melange(c["zones"], c["verite"], graine=k)
        plans = {"origine": (vue, positions_plan_origine(vue, zones))}
        plans["lisible"] = plan_lisible(vue, zones, c.get("position"), retour_positions=True)
        for nom_plan, (img, reperes) in plans.items():
            for q in questions(reperes):
                if q["ambigu"]:
                    continue
                t1 = time.perf_counter()
                brut = g.repond_images([img], DECOR + q["texte"] + " Answer with the number only.")
                latences.append(time.perf_counter() - t1)
                lu = _note(brut)
                bon = lu is not None and int(lu) == q["reponse"]
                justes[nom_plan] += bon
                poses[nom_plan] += 1
                details.append({"cas": k, "plan": nom_plan, "question": q["cle"],
                                "attendu": q["reponse"], "reponse": None if lu is None else int(lu),
                                "juste": bool(bon), "brut": brut.strip()[:40]})
    del g
    torch.cuda.empty_cache()
    return {"version": nom_version, "memoire_gpu_mo": round(memoire), "chargement_s": round(chargement, 1),
            "latence_mediane_s": round(float(np.median(latences)), 2) if latences else None,
            "justes_plan_origine": f"{justes['origine']}/{poses['origine']}",
            "justes_plan_lisible": f"{justes['lisible']}/{poses['lisible']}",
            "taux_origine": round(justes["origine"] / max(poses["origine"], 1), 3),
            "taux_lisible": round(justes["lisible"] / max(poses["lisible"], 1), 3),
            "details": details}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--missions", nargs="+", required=True)
    ap.add_argument("--modele", required=True)
    ap.add_argument("--cas", type=int, default=2)
    ap.add_argument("--versions", nargs="+", default=["4 bits", "4 bits, vision en 16", "8 bits", "sans compression"])
    ap.add_argument("--sortie", default="resultats_perception.json")
    a = ap.parse_args()

    tous = instantanes([Path("experiments/11_mission") / m for m in a.missions])
    pas = max(len(tous) // max(a.cas, 1), 1)
    cas = tous[::pas][:a.cas]
    print(f"{len(cas)} cas choisis sur {len(tous)} : " +
          ", ".join(f"{c['mission']} instantane {c['k']} drone {c['drone']}" for c in cas))
    bilans = []
    for nom in a.versions:
        try:
            b = profil(nom, a.modele, cas)
        except Exception as e:
            b = {"version": nom, "echec": f"{type(e).__name__}: {str(e)[:120]}"}
            print(f"  {nom:22s} ECHEC {b['echec']}")
        else:
            print(f"  {nom:22s} plan d'origine {b['justes_plan_origine']:>6s} ({b['taux_origine']:.0%}) | "
                  f"plan redessine {b['justes_plan_lisible']:>6s} ({b['taux_lisible']:.0%}) | "
                  f"{b['memoire_gpu_mo']} Mo | {b['latence_mediane_s']} s par question")
        bilans.append(b)
        (HERE / a.sortie).write_text(json.dumps({"modele": a.modele, "cas": len(cas),
                                                 "versions": bilans}, indent=1))
    print("PERCEPTION FINI")


if __name__ == "__main__":
    main()
