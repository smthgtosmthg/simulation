"""Analyse du banc du contrôleur — étape 3. Aucun simulateur.

Lit ce qui existe dans le dossier — freinage.json, poses.jsonl, essaim.json — et produit
resultats.json plus les figures. Toutes les durées sont en secondes simulées.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
IMMOBILE = 0.10          # m/s ; en dessous, le drone est considéré arrêté


def mediane(xs, digits=2):
    return round(float(np.median(xs)), digits) if len(xs) else None


def p90(xs, digits=2):
    return round(float(np.percentile(xs, 90)), digits) if len(xs) else None


# ---------------------------------------------------------------- freinage

def metriques_freinage(run, cible, depart, tol):
    traj = np.array([r[:7] for r in run["traj"]], float)
    t = traj[:, 0] - traj[0, 0]
    p = traj[:, 1:4]
    v = traj[:, 4:7]
    axe = (cible - depart) / max(np.linalg.norm(cible - depart), 1e-6)
    err = np.linalg.norm(cible - p, axis=1)
    au_dela = (p - cible) @ axe
    vitesse = np.linalg.norm(v, axis=1)
    phases = [r[7] for r in run["traj"]]
    k_fin = next((k for k, ph in enumerate(phases) if ph in ("atteint", "abandon")), None)
    fenetre = t <= (t[k_fin] + 1.0 if k_fin is not None else t[-1])
    dedans = err < tol
    t_arrivee = float(t[dedans][0]) if dedans.any() else None
    lent = vitesse < IMMOBILE
    t_stab = None
    for k in range(len(t)):
        if lent[k:].all():
            t_stab = float(t[k])
            break
    sorties = int(np.sum(dedans[:-1] & ~dedans[1:]))
    return {"loi": run["loi"], "rep": run["rep"],
            "t_arrivee": round(t_arrivee, 2) if t_arrivee is not None else None,
            "t_stabilisation": round(t_stab, 2) if t_stab is not None else None,
            "depassement_m": round(float(max(au_dela[fenetre].max(), 0.0)), 3),
            "derive_tenue_m": round(float(err[-1] - err[k_fin]), 3) if k_fin is not None else None,
            "err_tenue_m": round(float(err[k_fin:].max()), 3) if k_fin is not None else None,
            "err_min_m": round(float(err.min()), 3), "err_finale_m": round(float(err[-1]), 3),
            "sorties_de_tolerance": sorties, "phase_finale": run["bilan"].get("phase")}


def volet_freinage(res):
    f = HERE / "freinage.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    cible = np.array(d["cible"], float)
    depart = np.array(d["depart"], float)
    lignes = [metriques_freinage(r, cible, depart, d["tol"]) for r in d["runs"]]
    par_loi = {}
    for loi in dict.fromkeys(l["loi"] for l in lignes):
        sel = [l for l in lignes if l["loi"] == loi]
        arr = [l["t_arrivee"] for l in sel if l["t_arrivee"] is not None]
        stab = [l["t_stabilisation"] for l in sel if l["t_stabilisation"] is not None]
        par_loi[loi] = {
            "essais": len(sel), "passages_dans_tolerance": len(arr),
            "tenues": sum(1 for l in sel if l["phase_finale"] == "atteint"),
            "t_arrivee": mediane(arr), "t_stabilisation": mediane(stab),
            "stabilises": len(stab),
            "depassement_m": mediane([l["depassement_m"] for l in sel], 3),
            "err_finale_m": mediane([l["err_finale_m"] for l in sel], 3),
            "derive_tenue_m": mediane([l["derive_tenue_m"] for l in sel if l["derive_tenue_m"] is not None], 3),
            "err_tenue_max_m": mediane([l["err_tenue_m"] for l in sel if l["err_tenue_m"] is not None], 3),
            "sorties_de_tolerance": mediane([l["sorties_de_tolerance"] for l in sel], 1),
        }
    res["freinage"] = {"tol": d["tol"], "v_approche": d["v_approche"], "gain": d["gain"],
                       "recul_m": round(float(np.linalg.norm(cible - depart)), 2),
                       "par_loi": par_loi, "essais": lignes}
    print("freinage : dépassement / arrivée / arrêt / erreur finale / dérive en tenue / pire écart en tenue")
    for loi, m in par_loi.items():
        print(f"  {loi:14s} {m['depassement_m']:5.2f} m   {str(m['t_arrivee']):>5s} s   "
              f"{str(m['t_stabilisation']):>5s} s   {m['err_finale_m']:.2f} m   "
              f"{str(m['derive_tenue_m']):>6s} m   {str(m['err_tenue_max_m']):>5s} m   "
              f"tient la pose {m['tenues']}/{m['essais']}")
    figure_freinage(d, cible)


def figure_freinage(d, cible):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    couleurs = {"coupe": "tab:red", "proportionnel": "tab:blue", "autopilote": "tab:green"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    vus = set()
    for r in d["runs"]:
        traj = np.array([x[:7] for x in r["traj"]], float)
        t = traj[:, 0] - traj[0, 0]
        err = np.linalg.norm(cible - traj[:, 1:4], axis=1)
        vit = np.linalg.norm(traj[:, 4:7], axis=1)
        lab = r["loi"] if r["loi"] not in vus else None
        vus.add(r["loi"])
        axes[0].plot(t, err, color=couleurs[r["loi"]], alpha=0.8, label=lab)
        axes[1].plot(t, vit, color=couleurs[r["loi"]], alpha=0.8, label=lab)
    axes[0].axhline(d["tol"], color="k", ls=":", lw=1)
    axes[0].set_xlabel("temps (s)"); axes[0].set_ylabel("distance a la cible (m)")
    axes[0].set_title("Distance restante"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].axhline(IMMOBILE, color="k", ls=":", lw=1)
    axes[1].set_xlabel("temps (s)"); axes[1].set_ylabel("vitesse (m/s)")
    axes[1].set_title("Vitesse"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "freinage.png", dpi=130)


# ---------------------------------------------------------------- poses

def volet_poses(res):
    f = HERE / "poses.jsonl"
    if not f.exists():
        return
    lignes = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    meta = json.loads((HERE / "meta_poses.json").read_text()) if (HERE / "meta_poses.json").exists() else {}
    atteintes = [l for l in lignes if l["bilan"]["phase"] == "atteint"]
    abandons = {}
    for l in lignes:
        if l["bilan"]["phase"] != "atteint":
            abandons[l["bilan"]["raison"]] = abandons.get(l["bilan"]["raison"], 0) + 1
    direct = [l for l in atteintes if not l["points"]]
    couloir = [l for l in atteintes if l["points"]]
    lus = [l for l in atteintes if l["lecture"] and l["lecture"]["lu"]]
    err_vue = [abs(l["lecture"]["D_vue"] - l["lecture"]["D_vraie"]) for l in lus
               if l["lecture"]["D_vue"] is not None]

    def cycle(sel):
        b = [l["bilan"] for l in sel]
        return {"n": len(sel),
                "t_total": mediane([x["t_total"] for x in b]), "t_total_p90": p90([x["t_total"] for x in b]),
                "t_transit": mediane([x["t_transit"] for x in b]),
                "t_approche": mediane([x["t_approche"] for x in b]),
                "t_tenue": mediane([x["t_tenue"] for x in b]),
                "longueur_m": mediane([x["longueur"] for x in b]),
                "vitesse_moyenne": mediane([x["longueur"] / max(x["t_total"], 1e-6) for x in b])}

    res["poses"] = {
        "n": len(lignes), "atteintes": len(atteintes),
        "taux_arrivee": round(len(atteintes) / max(len(lignes), 1), 4),
        "abandons": abandons,
        "err_finale_cm": mediane([100 * l["bilan"]["err_finale"] for l in atteintes], 1),
        "err_finale_cm_p90": p90([100 * l["bilan"]["err_finale"] for l in atteintes], 1),
        "cap_err_deg": mediane([math.degrees(l["bilan"]["cap_err_finale"]) for l in atteintes], 1),
        "v_finale": mediane([l["bilan"]["v_finale"] for l in atteintes], 3),
        "cycle": cycle(atteintes), "cycle_direct": cycle(direct), "cycle_couloir": cycle(couloir),
        "lectures": len(lus), "taux_lecture": round(len(lus) / max(len(atteintes), 1), 4),
        "err_distance_vue_cm": mediane([100 * e for e in err_vue], 1),
        "debit_sim_sur_reel": mediane([l["debit"] for l in lignes], 3),
        "parametres": meta,
    }
    r = res["poses"]
    print(f"poses : {r['atteintes']}/{r['n']} atteintes, abandons {abandons}")
    print(f"  erreur finale {r['err_finale_cm']} cm (p90 {r['err_finale_cm_p90']}), "
          f"cap {r['cap_err_deg']} deg, vitesse residuelle {r['v_finale']} m/s")
    c = r["cycle"]
    print(f"  cycle {c['t_total']} s (p90 {c['t_total_p90']}) = transit {c['t_transit']} + "
          f"approche {c['t_approche']} + tenue {c['t_tenue']} ; {c['longueur_m']} m")
    print(f"  meme allee : {r['cycle_direct']['t_total']} s sur {r['cycle_direct']['n']} ; "
          f"par le couloir : {r['cycle_couloir']['t_total']} s sur {r['cycle_couloir']['n']}")
    print(f"  lecture apres arrivee : {r['lectures']}/{r['atteintes']} = {100 * r['taux_lecture']:.0f} %, "
          f"distance vue a {r['err_distance_vue_cm']} cm")
    figure_poses(lignes)


def figure_poses(lignes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [l for l in lignes if l["bilan"]["phase"] == "atteint"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    d = [l["bilan"]["t_total"] for l in ok if not l["points"]]
    c = [l["bilan"]["t_total"] for l in ok if l["points"]]
    axes[0].hist([d, c], bins=20, stacked=True, color=["tab:blue", "tab:orange"],
                 label=["meme allee", "par le couloir"])
    axes[0].set_xlabel("temps de cycle (s)"); axes[0].set_ylabel("poses")
    axes[0].set_title("Temps de cycle"); axes[0].legend(); axes[0].grid(alpha=0.3)
    for l in lignes:
        b = l["bilan"]
        axes[1].scatter(b["longueur"], b["t_total"],
                        color="tab:green" if b["phase"] == "atteint" else "tab:red", s=18)
    axes[1].set_xlabel("longueur du trajet (m)"); axes[1].set_ylabel("temps de cycle (s)")
    axes[1].set_title("Cycle selon la distance (rouge : abandon)"); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "cycles.png", dpi=130)


# ---------------------------------------------------------------- essaim

def volet_essaim(res):
    f = HERE / "essaim.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    manches = []
    for m in d["manches"]:
        drones = [{"drone": x["drone"], "phase": x["bilan"]["phase"], "raison": x["bilan"]["raison"],
                   "t_total": x["bilan"]["t_total"], "ecart_final_m": x["ecart_final"],
                   "lu": bool(x["lecture"] and x["lecture"]["lu"])} for x in m["drones"]]
        # distance minimale entre deux drones au cours de la manche
        trajs = [np.array([r[1:4] for r in x["traj"]], float) for x in m["drones"]]
        n = min(len(t) for t in trajs)
        dmin = min(float(np.linalg.norm(trajs[i][:n] - trajs[j][:n], axis=1).min())
                   for i in range(len(trajs)) for j in range(i + 1, len(trajs)))
        manches.append({"nom": m["nom"], "t_sim": m["t_sim"], "t_mur": m["t_mur"],
                        "vitesse_sim": round(m["t_sim"] / max(m["t_mur"], 1e-6), 3),
                        "distance_min_entre_drones_m": round(dmin, 2), "drones": drones})
        print(f"essaim, {m['nom']} : {m['t_sim']:.0f} s simulees en {m['t_mur']:.0f} s "
              f"({m['t_sim'] / m['t_mur']:.2f}x), drones a {dmin:.2f} m au plus pres")
        for x in drones:
            print(f"  drone {x['drone']}: {x['phase']:8s} {x['raison']:7s} {x['t_total']:5.1f} s  "
                  f"ecart {x['ecart_final_m']:.2f} m  {'lu' if x['lu'] else 'non lu'}")
    res["essaim"] = {"drones": d["drones"], "manches": manches}
    figure_essaim(d)


def figure_essaim(d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import sys
    sys.path.insert(0, str(HERE.parents[2]))
    from swarm_qr.env.config import INTERIOR
    from swarm_qr.env.layout import make_layout

    layout = make_layout(d["seed"])
    couleurs = ["tab:blue", "tab:orange", "tab:green"]
    fig, axes = plt.subplots(1, len(d["manches"]), figsize=(5.2 * len(d["manches"]), 7.5))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, d["manches"]):
        for r in layout.racks:
            x0, x1 = r.x_bounds
            y0, y1 = r.y_bounds
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color="0.75"))
        for k, x in enumerate(m["drones"]):
            T = np.array([r[1:4] for r in x["traj"]], float)
            ax.plot(T[:, 0], T[:, 1], color=couleurs[k], lw=1.5, label=f"drone {k}")
            ax.plot(T[0, 0], T[0, 1], "o", color=couleurs[k])
            ax.plot(x["cible"][0], x["cible"][1], "x", color=couleurs[k], ms=9, mew=2)
        ax.set_xlim(INTERIOR.x_min, INTERIOR.x_max)
        ax.set_ylim(INTERIOR.y_min, INTERIOR.y_max)
        ax.set_aspect("equal")
        ax.set_title(m["nom"])
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
    fig.suptitle("Trois drones, meme boucle de mission : rond = depart, croix = cible")
    fig.tight_layout()
    fig.savefig(HERE / "essaim.png", dpi=130)


# ---------------------------------------------------------------- résumé

res: dict = {}
volet_freinage(res)
volet_poses(res)
volet_essaim(res)
(HERE / "resultats.json").write_text(json.dumps(res, indent=1))
print("resultats.json ecrit")
