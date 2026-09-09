"""La durée de chaque cycle de calcul, datée, face à l'inclinaison des drones : un dérèglement
qui suit un ralentissement vient de la simulation, pas du contrôle.

  cycles.py --dossier autre_v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main(dossier: Path) -> None:
    m = json.loads((dossier / "mission.json").read_text())
    cyc = np.array(m.get("cycles_t", []), dtype=float)
    if not len(cyc):
        print("pas de cycles dates dans ce journal")
        return
    inc = {a["i"]: np.array(a["inclinaisons"], dtype=float) for a in m["agents"]}
    chutes = [(e["t"], e["drone"]) for e in m["evenements"] if e["genre"] == "chute"]
    print(f"{len(cyc)} cycles ; mediane {np.median(cyc[:, 1]):.0f} ms ; max {cyc[:, 1].max():.0f} ms ; chutes {chutes}")
    print("fenetre     cycle max   cycle median   inclinaison max (drones 0/1/2)")
    t0 = float(cyc[0, 0])
    for a in np.arange(t0, cyc[-1, 0] + 0.1, 20.0):
        sel = (cyc[:, 0] >= a) & (cyc[:, 0] < a + 20)
        if not sel.any():
            continue
        incl = " / ".join(f"{inc[i][(inc[i][:, 0] >= a) & (inc[i][:, 0] < a + 20)][:, 1].max():4.0f}"
                          if ((inc[i][:, 0] >= a) & (inc[i][:, 0] < a + 20)).any() else "   -" for i in sorted(inc))
        print(f"{a:5.0f}-{a + 20:5.0f} s   {cyc[sel, 1].max():7.0f} ms   {np.median(cyc[sel, 1]):7.0f} ms   {incl}")
    pires = cyc[np.argsort(-cyc[:, 1])[:8]]
    print("les huit cycles les plus longs (t, ms) :", [(round(float(t), 1), int(ms)) for t, ms in pires])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dossier", required=True)
    main(Path(ap.parse_args().dossier))
