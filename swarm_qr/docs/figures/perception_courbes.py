"""Les deux figures mesurées de la section perception : l'enveloppe de lecture et les deux
portées de repérage. Elles relisent les résultats des étapes 2 et 7 ; aucun chiffre n'est écrit
à la main. Version chapitre conception : une courbe par question, aucun nom de bibliothèque.

    perception_courbes.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = Path(__file__).resolve().parents[2] / "experiments"
ICI = Path(__file__).resolve().parent
ENCRE, DOUCE = "#16201C", "#5F6B66"
LU, APPRIS, CLASSIQUE = "#2F9E44", "#364FC7", "#C92A2A"
LIRE_MIN, LIRE_MAX = 1.5, 4.0

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": "#C9D3CF",
    "axes.labelcolor": ENCRE, "text.color": ENCRE, "xtick.color": DOUCE, "ytick.color": DOUCE,
    "axes.grid": True, "grid.color": "#E6EBE9", "grid.linewidth": 0.8, "figure.facecolor": "white",
})


def centres(bandes):
    return [(b["min"] + b["max"]) / 2 for b in bandes]


def enveloppe():
    """Le taux de lecture contre la distance apparente, et l'enveloppe qu'il définit."""
    r = json.loads((EXP / "07_enveloppe" / "resultats.json").read_text())
    c = r["optique"]["lecteurs"]["zxing"]["courbe_apparente"]
    x, y = centres(c), [b["p_lu"] for b in c]
    bas, haut = [b["ic_bas"] for b in c], [b["ic_haut"] for b in c]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.axvspan(LIRE_MIN, LIRE_MAX, color=LU, alpha=0.09, lw=0)
    ax.axhline(0.9, color=DOUCE, ls=":", lw=1.2)
    ax.fill_between(x, bas, haut, color=LU, alpha=0.16, lw=0)
    ax.plot(x, y, "-o", color=LU, lw=2, ms=5)

    ax.text((LIRE_MIN + LIRE_MAX) / 2, 0.055, "reading envelope $\\mathcal{E}$", ha="center",
            fontsize=10.5, color=LU, fontweight="semibold")
    ax.annotate("", xy=(LIRE_MIN, 0.115), xytext=(LIRE_MAX, 0.115),
                arrowprops=dict(arrowstyle="<->", color=LU, lw=1.2))
    ax.text(max(x) * 0.99, 0.915, "90 %", ha="right", va="bottom", fontsize=9.5, color=DOUCE)
    ax.annotate("mechanical bound : closer than this,\na vehicle cannot hold its heading",
                xy=(LIRE_MIN, 0.80), xytext=(1.02, 0.33), fontsize=9, color=DOUCE, style="italic",
                arrowprops=dict(arrowstyle="->", color=DOUCE, lw=0.9))
    ax.annotate("optical bound : beyond this,\nthe modules cover too few pixels",
                xy=(LIRE_MAX, 0.91), xytext=(4.45, 0.80), fontsize=9, color=DOUCE, style="italic",
                arrowprops=dict(arrowstyle="->", color=DOUCE, lw=0.9))

    ax.set_xlabel("apparent distance   $d_{app} = d\\,/\\cos\\alpha$   (m)")
    ax.set_ylabel("probability of reading, per image")
    ax.set_ylim(0, 1.04)
    ax.set_xlim(0.9, max(x) + 0.4)
    fig.tight_layout()
    fig.savefig(ICI / "fig_enveloppe.png", dpi=200)
    print(f"fig_enveloppe.png : {len(x)} bandes, {sum(b['n'] for b in c)} images")


def portees():
    """Ce que chaque détecteur repère, à la même distance, sur les mêmes images."""
    r = json.loads((EXP / "10_detecteur" / "resultats.json").read_text())
    cl = r["classique"]["optique"]["courbe_distance"]
    ap = [b for b in [v for v in r["variantes"] if v["variante"] == "n1024"][0]
          ["panneau_vise"]["optique"]["conf_0.5"]["par_distance"] if b["taux"] is not None]
    faux_cl = r["classique"]["faux_reperage_sans_qr"]
    faux_ap = [v for v in r["variantes"] if v["variante"] == "n1024"][0]["sans_qr"]["conf_0.5"]["taux"]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.axvspan(0, LIRE_MAX, color=LU, alpha=0.07, lw=0)
    ax.axhline(0.9, color=DOUCE, ls=":", lw=1.2)
    xc, yc = centres(cl), [b["taux"] for b in cl]
    xa, ya = centres(ap), [b["taux"] for b in ap]
    ax.plot(xc, yc, "--s", color=CLASSIQUE, lw=2, ms=5, zorder=2, label="geometric detector")
    ax.plot(xa, ya, "-o", color=APPRIS, lw=2.2, ms=5, zorder=3, label="learned detector")

    ax.text(LIRE_MAX / 2, 0.05, "reading is possible here", ha="center", fontsize=10,
            color=LU, fontweight="semibold")
    ax.annotate(f"{ya[-1]:.0%}", xy=(xa[-1], ya[-1]), xytext=(xa[-1] + 0.25, ya[-1] - 0.03),
                fontsize=10.5, color=APPRIS, fontweight="semibold")
    ax.annotate(f"{yc[-1]:.0%}", xy=(xc[-1], yc[-1]), xytext=(xc[-1] + 0.25, yc[-1] - 0.02),
                fontsize=10.5, color=CLASSIQUE, fontweight="semibold")
    ax.text(0.975, 0.05, f"on a scene holding no code at all,\n"
                         f"the geometric detector reports one on {faux_cl:.0%} of images,\n"
                         f"the learned one on {faux_ap:.1%}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=ENCRE,
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#C9D3CF", lw=0.9))

    ax.set_xlabel("distance to the target panel (m)")
    ax.set_ylabel("fraction of panels detected")
    ax.set_ylim(0, 1.08)
    ax.set_xlim(0.5, 8.6)
    ax.legend(loc="center left", frameon=True, framealpha=1, edgecolor="#C9D3CF")
    fig.tight_layout()
    fig.savefig(ICI / "fig_portees.png", dpi=200)
    print(f"fig_portees.png : appris {len(ap)} bandes, classique {len(cl)} bandes")


enveloppe()
portees()
