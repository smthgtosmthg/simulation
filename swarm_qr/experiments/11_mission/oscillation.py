"""L'oscillation d'attitude, mesurée dans les journaux de bord des autopilotes : l'erreur maximale
entre roulis commandé et roulis réel par fenêtre de 20 s, pour chaque drone. Un vol sain reste
sous 3 degrés ; le vol du 10-09 19:31 passait de 2 à 31 degrés en quarante secondes avant le
retournement.

    oscillation.py --dossier experiments/11_mission/eval_nominal
"""
import argparse
from pathlib import Path

import numpy as np
from pymavlink import DFReader

SEUIL = 3.0


def erreurs(fichier: Path) -> np.ndarray:
    log = DFReader.DFReader_binary(str(fichier))
    t0, out = None, []
    while True:
        m = log.recv_msg()
        if m is None:
            break
        if m.get_type() != "ATT":
            continue
        t0 = m.TimeUS if t0 is None else t0
        out.append(((m.TimeUS - t0) / 1e6, abs(m.Roll - m.DesRoll), abs(m.Pitch - m.DesPitch)))
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dossier", required=True)
    ap.add_argument("--fenetre", type=float, default=20.0)
    a = ap.parse_args()
    sain = True
    for d in sorted((Path(a.dossier) / "ardupilot_logs").glob("drone_*")):
        for f in sorted(d.glob("*.BIN")):
            e = erreurs(f)
            if len(e) == 0:
                print(f"  {d.name} : pas d'attitude dans le journal")
                continue
            pire = 0.0
            lignes = []
            for w in np.arange(0, e[-1, 0], a.fenetre):
                s = e[(e[:, 0] >= w) & (e[:, 0] < w + a.fenetre)]
                if len(s):
                    lignes.append(f"{w:.0f}s:{s[:, 1].max():.1f}/{s[:, 2].max():.1f}")
                    pire = max(pire, s[:, 1].max(), s[:, 2].max())
            sain &= pire < SEUIL * 3
            print(f"  {d.name} : erreur max roulis/tangage par fenetre de {a.fenetre:.0f} s (deg) : " + "  ".join(lignes))
            print(f"  {d.name} : pire erreur {pire:.1f} deg sur {e[-1, 0]:.0f} s -> {'sain' if pire < SEUIL * 3 else 'OSCILLATION'}")
    print("ATTITUDE SAINE" if sain else "ATTITUDE INSTABLE")


if __name__ == "__main__":
    main()
