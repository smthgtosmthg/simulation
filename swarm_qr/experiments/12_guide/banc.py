"""Banc hors ligne du guide vision-langage — étape 8. Sans simulateur.

  banc.py --missions ../11_mission/nominale ../11_mission/panne ../11_mission/autre --modeles smolvlm smolvlm-2b

Pendant les missions, un instantané est pris toutes les trente secondes : la carte vue de
dessus avec les zones candidates numérotées, l'image de la caméra de chaque drone, et — connu
après coup — la zone qui contient le plus de panneaux non lus et le côté d'où ils se lisent.
Chaque modèle répond aux mêmes instantanés ; on mesure son accord avec la bonne réponse, contre
un choix au hasard. C'est la mesure qui décide si le guide est branché.
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
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from swarm_qr import planning  # noqa: E402


def instantanes(missions: list[Path]) -> list[dict]:
    """Un cas par drone vivant et par instantané qui a au moins deux zones et une bonne réponse."""
    cas = []
    for m in missions:
        journal = json.loads((m / "mission.json").read_text())
        for inst in journal["instantanes"]:
            v = inst["verite"]
            if len(inst["zones"]) < 2 or v["zone"] is None or v["panneaux_restants"] <= 0:
                continue
            vue = m / "instantanes" / f"{inst['k']:03d}_vue.png"
            for d in inst["drones"]:
                cam = m / "instantanes" / f"{inst['k']:03d}_cam{d['i']}.jpg"
                if d["vivant"] and cam.exists() and vue.exists():
                    cas.append({"mission": m.name, "k": inst["k"], "t": inst["t"], "drone": d["i"],
                                "vue": str(vue), "cam": str(cam), "zones": inst["zones"], "verite": v})
    return cas


def juge(nom: str, cas: list[dict]) -> dict:
    import torch
    from swarm_qr.guide import Guide

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    g = Guide(nom)
    chargement = time.perf_counter() - t0
    accords_zone = accords_cote = repondus = cotes_repondus = 0
    hasard_zone = 0.0
    details = []
    for c in cas:
        cam, vue = cv2.imread(c["cam"]), cv2.imread(c["vue"])
        avis = g.conseille(cam, vue, c["zones"])
        v = c["verite"]
        hasard_zone += 1.0 / len(c["zones"])
        zone = None if avis is None else int(avis.phrase.split()[1].rstrip(","))
        cote = None if avis is None else avis.cote
        repondus += int(zone is not None)
        accords_zone += int(zone == v["zone"])
        if cote is not None:
            cotes_repondus += 1
            accords_cote += int(planning.NOMS_COTES[cote] == v["cote"])
        details.append({**{k: c[k] for k in ("mission", "k", "drone")}, "zone": zone,
                        "cote": None if cote is None else planning.NOMS_COTES[cote],
                        "bonne_zone": v["zone"], "bon_cote": v["cote"],
                        "reponses": g.reponses[-2:] if zone is not None else g.reponses[-1:]})
    n = max(len(cas), 1)
    return {"modele": g.nom, "cas": len(cas), "chargement_s": round(chargement, 1),
            "repondus": repondus, "accord_zone": round(accords_zone / n, 4),
            "hasard_zone": round(hasard_zone / n, 4),
            "accord_cote": round(accords_cote / max(cotes_repondus, 1), 4), "cotes_repondus": cotes_repondus,
            "hasard_cote": 0.25, "latence_mediane_s": round(float(np.median(g.latences)), 2),
            "memoire_gpu_mo": round(torch.cuda.max_memory_allocated() / 1e6), "details": details}


def references(cas: list[dict]) -> dict:
    """Ce qu'un modèle doit battre : le hasard, mais aussi les réponses bêtes qui profitent de
    la structure de l'entrepôt — les racks sont tous dans le même sens, donc « toujours ouest »
    a raison une fois sur deux sans rien comprendre — et la zone de plus grande utilité
    géométrique, c'est-à-dire ce que le cerveau choisit déjà sans guide."""
    from collections import Counter
    n = max(len(cas), 1)
    cotes = Counter(c["verite"]["cote"] for c in cas)
    plus_frequent, k = cotes.most_common(1)[0]
    zone_geo = sum(1 for c in cas if c["zones"] and c["zones"][0]["numero"] == c["verite"]["zone"]) / n
    return {"hasard_zone": round(sum(1.0 / len(c["zones"]) for c in cas) / n, 4),
            "zone_la_plus_utile_geometrie": round(zone_geo, 4),
            "hasard_cote": 0.25,
            "toujours_ouest": round(sum(1 for c in cas if c["verite"]["cote"] == "ouest") / n, 4),
            "cote_le_plus_frequent": {"cote": plus_frequent, "part": round(k / n, 4)}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--missions", nargs="+", required=True)
    ap.add_argument("--modeles", nargs="+", default=["smolvlm"])
    a = ap.parse_args()
    cas = instantanes([Path(m) for m in a.missions])
    print(f"{len(cas)} cas (instantane x drone) avec une bonne reponse connue")
    refs = references(cas)
    print("references :", refs)
    bilans = []
    for nom in a.modeles:
        b = juge(nom, cas)
        bilans.append(b)
        print(f"{b['modele']}: zone juste {b['accord_zone']:.0%} (hasard {b['hasard_zone']:.0%}), "
              f"cote juste {b['accord_cote']:.0%} sur {b['cotes_repondus']} reponses (hasard 25 %), "
              f"{b['latence_mediane_s']} s par question, {b['memoire_gpu_mo']} Mo, repondus {b['repondus']}/{b['cas']}")
    (HERE / "resultats.json").write_text(json.dumps({"cas": len(cas), "references": refs, "modeles": bilans}, indent=1))
    print("BANC GUIDE FINI")


if __name__ == "__main__":
    main()
