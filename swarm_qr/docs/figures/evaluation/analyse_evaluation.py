"""Toutes les mesures et toutes les figures du chapitre évaluation.

Lit les quatre vols du système et les quatre vols de la référence, calcule chaque métrique
définie au protocole, écrit `mesures.json` et `tableaux.md`, et produit les figures.

    analyse_evaluation.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = Path(__file__).resolve().parents[3] / "experiments"
ICI = Path(__file__).resolve().parent
SYS = EXP / "11_mission" / "tests of system"
PORE = EXP / "13_pore"

CAS = ["nominal", "panne", "9019", "obstacle"]
TITRES = {"nominal": "Nominal", "panne": "Vehicle loss", "9019": "Unseen warehouse",
          "obstacle": "Obstacle appearing"}
DOSSIERS = {
    "systeme": {c: SYS / f"eval_{c}" for c in CAS},
    "pore": {c: PORE / f"pore_{c}" for c in CAS},
}
# résultats mesurés antérieurement sur les deux approches apprises du projet
APPRIS = {
    "aif": {"nominal": 41.0, "panne": 38.0, "9019": 36.0, "obstacle": 39.0},
    "rl":  {"nominal": 19.0, "panne": 15.0, "9019": 14.0, "obstacle": 17.0},
}

C_SYS, C_PORE, C_AIF, C_RL = "#2F9E44", "#C92A2A", "#B07A22", "#7A3E9D"
ENCRE, DOUCE = "#16201C", "#5F6B66"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#C9D3CF",
    "axes.labelcolor": ENCRE, "text.color": ENCRE, "xtick.color": DOUCE, "ytick.color": DOUCE,
    "axes.grid": True, "grid.color": "#E9EDEB", "grid.linewidth": 0.8, "figure.facecolor": "white",
    "axes.axisbelow": True,
})


def charge(dossier: Path) -> dict:
    m = json.loads((dossier / "mission.json").read_text())
    r = json.loads((dossier / "resultats.json").read_text())
    return {"m": m, "r": r}


def distance(traj) -> float:
    p = np.array([t[1:] for t in traj], dtype=float)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def jalon(courbe, part: float):
    """Instant où la part demandée du total finalement lu est atteinte."""
    c = np.array(courbe, dtype=float)
    if not len(c):
        return None
    cible = part * c[-1, 1]
    k = np.flatnonzero(c[:, 1] >= cible)
    return float(c[k[0], 0]) if len(k) else None


def mesures_vol(d: dict, methode: str) -> dict:
    m, r = d["m"], d["r"]
    lec = r.get("lecture", {})
    sec = r.get("securite", {})
    out = {
        "seed": m.get("seed"),
        "codes_vrais": lec.get("codes_vrais"),
        "codes_lus": lec.get("codes_lus"),
        "part": lec.get("part"),
        "inventes": lec.get("inventes", 0),
        "t_50pct_s": lec.get("t_50pct_s"),
        "t_80pct_s": lec.get("t_80pct_s"),
        "t_90pct_s": lec.get("t_90pct_s"),
        "fin": r.get("fin", {}).get("fin") if isinstance(r.get("fin"), dict) else r.get("fin"),
        "t_sim_s": r.get("fin", {}).get("t_sim_s") if isinstance(r.get("fin"), dict) else r.get("t_sim_s"),
        "points": sec.get("points"),
        "dans_un_rack": sec.get("dans_un_rack"),
        "dist_min_drones_m": sec.get("distance_min_entre_drones_m"),
        "attentes_priorite": sec.get("attentes_de_priorite"),
        "mur_min": r.get("mur_min") or m.get("mur_min"),
        "courbe": m.get("codes_par_t", []),
        "obstacle": r.get("obstacle", {}),
        "methode": methode,
    }
    ag = m.get("agents", [])
    if ag:
        out["distance_m"] = [round(distance(a["trajectoire"]), 1) for a in ag]
        out["distance_totale_m"] = round(sum(out["distance_m"]), 1)
        out["decisions"] = [len(a.get("decisions", [])) for a in ag]
        genres = {}
        for a in ag:
            for dec in a.get("decisions", []):
                g = dec.get("genre") or dec.get("kind") or "?"
                genres[g] = genres.get(g, 0) + 1
        out["decisions_par_genre"] = genres
        out["attentes"] = [a.get("attentes", 0) for a in ag]
        cpt = {}
        for a in ag:
            for k, v in (a.get("compte") or {}).items():
                cpt[k] = cpt.get(k, 0) + v
        out["compte"] = cpt
        ms = {}
        for a in ag:
            for k, v in (a.get("ms") or {}).items():
                ms.setdefault(k, []).append(v)
        out["ms_par_observation"] = {k: round(float(np.mean(v)), 1) for k, v in ms.items()}
        out["inclinaison_max_deg"] = round(max(
            max((i[1] for i in a.get("inclinaisons", [])), default=0.0) for a in ag), 1)
    else:
        out["distance_m"], out["distance_totale_m"] = [], None
    if out["distance_totale_m"] and out["codes_lus"]:
        out["codes_par_100m"] = round(100.0 * out["codes_lus"] / out["distance_totale_m"], 1)
    out["abandons"] = len(r.get("abandons", [])) if isinstance(r.get("abandons"), list) else None
    if methode == "pore":
        out["plan"] = r.get("plan", {})
        out["inventaire"] = r.get("inventaire", {})
    return out


def tout() -> dict:
    res = {"systeme": {}, "pore": {}}
    for methode, cas in DOSSIERS.items():
        for c, d in cas.items():
            if (d / "mission.json").exists():
                res[methode][c] = mesures_vol(charge(d), methode)
    res["aif"] = {c: {"part": APPRIS["aif"][c] / 100.0} for c in CAS}
    res["rl"] = {c: {"part": APPRIS["rl"][c] / 100.0} for c in CAS}
    return res


R = tout()
(ICI / "mesures.json").write_text(json.dumps(R, indent=1, default=str))
print("mesures.json ecrit")
for c in CAS:
    s, p = R["systeme"].get(c), R["pore"].get(c)
    if s:
        print(f"  {c:9s} systeme {s['codes_lus']}/{s['codes_vrais']} "
              f"t90={s['t_90pct_s']} fin={s['t_sim_s']}s dist={s.get('distance_totale_m')}m")
    if p:
        print(f"  {c:9s} pore    {p['codes_lus']}/{p['codes_vrais']} "
              f"t90={p['t_90pct_s']} fin={p['t_sim_s']}s")


# ------------------------------------------------------------------ figures
def sauve(fig, nom):
    fig.tight_layout()
    fig.savefig(ICI / nom, dpi=200)
    plt.close(fig)
    print(f"  {nom}")


def f_methodes():
    """La figure d'ensemble : quatre méthodes, quatre scénarios."""
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    x = np.arange(len(CAS))
    w = 0.2
    series = [("Proposed system", "systeme", C_SYS), ("Pore et al.", "pore", C_PORE),
              ("Active inference", "aif", C_AIF), ("Reinforcement learning", "rl", C_RL)]
    for k, (nom, cle, coul) in enumerate(series):
        y = [100.0 * (R[cle].get(c, {}).get("part") or 0.0) for c in CAS]
        b = ax.bar(x + (k - 1.5) * w, y, w, label=nom, color=coul, edgecolor="white", linewidth=0.8)
        ax.bar_label(b, fmt="%.0f", fontsize=8.5, padding=2, color=coul)
    ax.set_xticks(x, [TITRES[c] for c in CAS])
    ax.set_ylabel("identifiers read (% of the inventory)")
    ax.set_ylim(0, 108)
    ax.axhline(100, color=DOUCE, ls=":", lw=1)
    ax.legend(ncols=4, frameon=False, fontsize=9.5, loc="upper center",
              bbox_to_anchor=(0.5, 1.16))
    sauve(fig, "ev_methodes.png")


def f_lecture_temps():
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.2), sharex=True, sharey=False)
    for ax, c in zip(axes.ravel(), CAS):
        for cle, coul, nom, st in (("systeme", C_SYS, "Proposed system", "-"),
                                   ("pore", C_PORE, "Pore et al.", "--")):
            d = R[cle].get(c)
            if not d or not d.get("courbe"):
                continue
            a = np.array(d["courbe"], dtype=float)
            ax.plot(a[:, 0], a[:, 1], st, color=coul, lw=2, label=nom)
            ax.plot(a[-1, 0], a[-1, 1], "o", color=coul, ms=5)
            ax.annotate(f"{int(a[-1,1])}", (a[-1, 0], a[-1, 1]), textcoords="offset points",
                        xytext=(6, -3), color=coul, fontsize=9, fontweight="semibold")
        n = R["systeme"].get(c, {}).get("codes_vrais")
        if n:
            ax.axhline(n, color=DOUCE, ls=":", lw=1)
            ax.text(8, n + 1.5, f"{n} identifiers present", fontsize=8.5, color=DOUCE)
        if c == "panne":
            ax.axvline(200, color=C_PORE, ls="-.", lw=1.2)
            ax.text(206, 6, "vehicle lost", fontsize=8.5, color=C_PORE, rotation=90, va="bottom")
        if c == "obstacle":
            ax.axvline(200, color=C_PORE, ls="-.", lw=1.2)
            ax.text(206, 6, "obstacle appears", fontsize=8.5, color=C_PORE, rotation=90, va="bottom")
        ax.set_title(TITRES[c], fontsize=11, color=ENCRE, fontweight="semibold")
        ax.set_xlim(0, 600)
    for ax in axes[1]:
        ax.set_xlabel("simulated time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("identifiers read")
    axes[0, 0].legend(frameon=True, framealpha=1, edgecolor="#C9D3CF", loc="lower right")
    sauve(fig, "ev_lecture_temps.png")


def f_jalons():
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    x = np.arange(len(CAS))
    w = 0.16
    for k, (part, alpha) in enumerate([("t_50pct_s", 0.45), ("t_80pct_s", 0.72), ("t_90pct_s", 1.0)]):
        for j, (cle, coul) in enumerate((("systeme", C_SYS), ("pore", C_PORE))):
            y = [R[cle].get(c, {}).get(part) or 0.0 for c in CAS]
            ax.bar(x + (k - 1) * 2 * w + (j - 0.5) * w, y, w, color=coul, alpha=alpha,
                   edgecolor="white", linewidth=0.6)
    ax.set_xticks(x, [TITRES[c] for c in CAS])
    ax.set_ylabel("simulated time to reach the milestone (s)")
    ax.text(0.01, 0.97, "left to right in each group : 50 %, 80 %, 90 %\ngreen = proposed system, "
            "red = Pore et al.   ·   a missing bar means the milestone was never reached",
            transform=ax.transAxes, va="top", fontsize=9, color=DOUCE)
    ax.set_ylim(0, 620)
    sauve(fig, "ev_jalons.png")


def f_scenario(c, nom, texte):
    d, p = R["systeme"][c], R["pore"].get(c)
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    a = np.array(d["courbe"], dtype=float)
    ax.plot(a[:, 0], a[:, 1], "-", color=C_SYS, lw=2.2, label="Proposed system")
    if p and p.get("courbe"):
        b = np.array(p["courbe"], dtype=float)
        ax.plot(b[:, 0], b[:, 1], "--", color=C_PORE, lw=2, label="Pore et al.")
    ax.axvline(200, color=ENCRE, ls="-.", lw=1.4)
    ax.text(204, 4, texte, fontsize=9.5, color=ENCRE, rotation=90, va="bottom")
    avant = float(a[np.searchsorted(a[:, 0], 200) - 1, 1])
    ax.annotate(f"{int(avant)} read before", (200, avant), textcoords="offset points",
                xytext=(-92, 6), fontsize=9, color=C_SYS, fontweight="semibold")
    ax.annotate(f"{int(a[-1,1])} read in total", (a[-1, 0], a[-1, 1]), textcoords="offset points",
                xytext=(-104, 8), fontsize=9, color=C_SYS, fontweight="semibold")
    n = d["codes_vrais"]
    ax.axhline(n, color=DOUCE, ls=":", lw=1)
    ax.set_xlabel("simulated time (s)")
    ax.set_ylabel("identifiers read")
    ax.set_xlim(0, 600)
    ax.legend(frameon=True, framealpha=1, edgecolor="#C9D3CF", loc="lower right")
    sauve(fig, nom)


def f_effort():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    x = np.arange(len(CAS))
    w = 0.25
    for k in range(3):
        y = [(R["systeme"][c].get("distance_m") or [0, 0, 0])[k] for c in CAS]
        a1.bar(x + (k - 1) * w, y, w, label=f"vehicle {k}",
               color=["#2F9E44", "#66C27A", "#A8DDB5"][k], edgecolor="white", linewidth=0.7)
    a1.set_xticks(x, [TITRES[c] for c in CAS], fontsize=9)
    a1.set_ylabel("distance flown (m)")
    a1.legend(frameon=True, framealpha=1, edgecolor="#C9D3CF", fontsize=9)
    a1.set_title("Effort per vehicle", fontsize=11, fontweight="semibold")

    y = [R["systeme"][c].get("codes_par_100m") or 0 for c in CAS]
    b = a2.bar(x, y, 0.55, color=C_SYS, edgecolor="white", linewidth=0.8)
    a2.bar_label(b, fmt="%.1f", fontsize=9.5, padding=3, color=C_SYS)
    a2.set_xticks(x, [TITRES[c] for c in CAS], fontsize=9)
    a2.set_ylabel("identifiers read per 100 m flown")
    a2.set_title("Yield of the flight", fontsize=11, fontweight="semibold")
    sauve(fig, "ev_effort.png")


def f_couts():
    ordre = ["lidar", "couverture", "decodage", "detecteur"]
    noms = {"lidar": "LiDAR integration", "couverture": "coverage integration",
            "decodage": "decoding", "detecteur": "learned detection"}
    coul = ["#0B7285", "#B07A22", "#2F9E44", "#364FC7"]
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    x = np.arange(len(CAS))
    bas = np.zeros(len(CAS))
    for k, o in enumerate(ordre):
        y = np.array([R["systeme"][c].get("ms_par_observation", {}).get(o, 0.0) for c in CAS])
        ax.bar(x, y, 0.5, bottom=bas, label=noms[o], color=coul[k], edgecolor="white", linewidth=0.7)
        bas += y
    for i, t in enumerate(bas):
        ax.text(i, t + 3, f"{t:.0f} ms", ha="center", fontsize=9, color=ENCRE, fontweight="semibold")
    ax.set_xticks(x, [TITRES[c] for c in CAS], fontsize=9)
    ax.set_ylabel("cost of one observation (ms)")
    ax.legend(ncols=2, frameon=True, framealpha=1, edgecolor="#C9D3CF", fontsize=9)
    sauve(fig, "ev_couts.png")


def f_securite():
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(10.6, 3.6))
    x = np.arange(len(CAS))
    for j, (cle, coul, nom) in enumerate((("systeme", C_SYS, "Proposed system"),
                                          ("pore", C_PORE, "Pore et al."))):
        y = [R[cle].get(c, {}).get("dist_min_drones_m") or 0 for c in CAS]
        a1.bar(x + (j - 0.5) * 0.35, y, 0.35, color=coul, label=nom, edgecolor="white", linewidth=0.7)
    a1.axhline(1.5, color=ENCRE, ls="--", lw=1.2)
    a1.text(-0.4, 1.58, "full-stop threshold", fontsize=8.5, color=ENCRE)
    a1.set_ylabel("minimum distance between vehicles (m)")
    a1.legend(fontsize=8.5, frameon=True, framealpha=1, edgecolor="#C9D3CF")

    for j, (cle, coul) in enumerate((("systeme", C_SYS), ("pore", C_PORE))):
        y = [R[cle].get(c, {}).get("dans_un_rack") or 0 for c in CAS]
        a2.bar(x + (j - 0.5) * 0.35, y, 0.35, color=coul, edgecolor="white", linewidth=0.7)
    a2.set_ylabel("trajectory points inside a rack")
    a2.set_ylim(0, 1)
    a2.text(0.5, 0.5, "0 for every flight,\nboth methods", ha="center", va="center",
            transform=a2.transAxes, fontsize=11, color=ENCRE, fontweight="semibold")

    y = [R["systeme"][c].get("attentes_priorite") or 0 for c in CAS]
    b = a3.bar(x, y, 0.55, color=C_SYS, edgecolor="white", linewidth=0.8)
    a3.bar_label(b, fmt="%d", fontsize=9.5, padding=3, color=C_SYS)
    a3.set_ylabel("give-way events")
    for a in (a1, a2, a3):
        a.set_xticks(x, [TITRES[c].split()[0] for c in CAS], fontsize=9)
    sauve(fig, "ev_securite.png")


def f_decisions():
    noms = {"lire": "read a spotted label", "couvrir": "cover an unobserved face",
            "explorer": "extend the known region"}
    coul = {"lire": C_SYS, "couvrir": "#B07A22", "explorer": "#364FC7"}
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    x = np.arange(len(CAS))
    bas = np.zeros(len(CAS))
    for g in ("lire", "couvrir", "explorer"):
        y = np.array([R["systeme"][c].get("decisions_par_genre", {}).get(g, 0) for c in CAS], float)
        ax.bar(x, y, 0.5, bottom=bas, label=noms[g], color=coul[g], edgecolor="white", linewidth=0.7)
        for i, (v, b0) in enumerate(zip(y, bas)):
            if v > 2:
                ax.text(i, b0 + v / 2, f"{int(v)}", ha="center", va="center", fontsize=9,
                        color="white", fontweight="semibold")
        bas += y
    ax.set_xticks(x, [TITRES[c] for c in CAS], fontsize=9)
    ax.set_ylabel("targets selected during the mission")
    ax.legend(frameon=True, framealpha=1, edgecolor="#C9D3CF", fontsize=9)
    sauve(fig, "ev_decisions.png")


print("figures :")
f_methodes()
f_lecture_temps()
f_jalons()
f_scenario("panne", "ev_panne.png", "vehicle 1 stops")
f_scenario("obstacle", "ev_obstacle.png", "obstacle appears")
f_effort()
f_couts()
f_securite()
f_decisions()
