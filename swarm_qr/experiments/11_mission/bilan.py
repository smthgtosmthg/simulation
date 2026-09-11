"""Le tableau de l'évaluation finale : pour chaque cas, le système complet contre les deux
références, à partir des jugements déjà écrits (`resultats.json`) et des journaux.

    bilan.py            imprime le tableau et écrit bilan.md à côté
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
POLITIQUES = [("eval", "système : géométrie seule"), ("zigzag", "référence : balayage fixe (Pore et al.)"),
              ("glouton", "référence : glouton omniscient")]
CAS = ["nominal", "panne", "9019", "obstacle"]


def ligne(dossier: Path) -> dict | None:
    if not (dossier / "resultats.json").exists() or not (dossier / "mission.json").exists():
        return None
    r = json.loads((dossier / "resultats.json").read_text())
    m = json.loads((dossier / "mission.json").read_text())
    lec, sec = r["lecture"], r["securite"]
    chutes = [e for e in m["evenements"] if e["genre"] == "chute"]
    pm = m.get("paquets_moteurs") or {}
    obs = r.get("obstacle", {})
    return {
        "codes": f"{lec['codes_lus']}/{lec['codes_vrais']}",
        "t90": lec.get("t_90pct_s"),
        "fin": f"{m['fin']} ({m['t_sim_s']:.0f} s)",
        "chutes": len(chutes),
        "rack": f"{sec['dans_un_rack']}/{sec['points']}",
        "dist_min": sec["distance_min_entre_drones_m"],
        "obstacle": ("évité" if obs.get("ok") else f"{obs.get('dedans')} passages") if obs.get("simule") else "-",
        "paquets": f"{pm.get('manques', '?')}/{(pm.get('manques', 0) + pm.get('recus', 0)) or '?'}",
        "calcul_min": m.get("mur_min"),
        "plan": ("%d/%d arrêts servis" % (
            sum(len(a["decisions"]) for a in m["agents"]),
            sum(b["restants"] + b["mis_de_cote"] for b in m["zigzag"].values()) +
            sum(len(a["decisions"]) for a in m["agents"]))) if m.get("zigzag") else "-",
    }


def main() -> None:
    lignes = ["| cas | politique | codes lus | 90 % à | fin | chutes | pts dans un rack | dist. min | obstacle | plan | calcul |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for cas in CAS:
        for tag, nom in POLITIQUES:
            l = ligne(HERE / f"{tag}_{cas}")
            if l is None:
                lignes.append(f"| {cas} | {nom} | pas encore volé | | | | | | | | |")
                continue
            lignes.append(f"| {cas} | {nom} | {l['codes']} | {l['t90']} s | {l['fin']} | {l['chutes']} | {l['rack']} | "
                          f"{l['dist_min']} m | {l['obstacle']} | {l['plan']} | {l['calcul_min']} min |")
    texte = "\n".join(lignes)
    print(texte)
    (HERE / "bilan.md").write_text("# Évaluation finale — système contre références\n\n" + texte + "\n")


if __name__ == "__main__":
    main()
