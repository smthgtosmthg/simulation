"""Le juge de la référence de Pore et al., et ses figures.

Il mesure les mêmes choses que le juge du système, moins celles qui n'ont pas de sens ici : la
méthode ne construit aucune carte, donc il n'y a ni couverture ni frontières à noter. À la
place, la figure de trajectoires est dessinée sur le **plan connu** de l'entrepôt, qui est
précisément ce que leur système reçoit.

    analyse.py --dossier experiments/13_pore/pore_nominal
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from swarm_qr import pore  # noqa: E402
from swarm_qr.env.config import INTERIOR  # noqa: E402
from swarm_qr.env.layout import make_layout  # noqa: E402

BOUT_VIDE_SUD, BOUT_VIDE_NORD = 0.70, 0.77     # mêmes valeurs que le juge du système
COULEURS = ["#2882ff", "#ff8c00", "#c800c8"]


def lecture(m) -> dict:
    vrais = {t["code"] for t in m["verite"]}
    codes = np.array(m["codes_par_t"], float)
    n = len(vrais)

    def t_pour(part):
        k = int(np.argmax(codes[:, 1] >= part * n))
        return float(codes[k, 0]) if codes[k, 1] >= part * n else None

    lus = set(m["codes"])
    return {"codes_vrais": n, "codes_lus": len(lus), "part": round(len(lus) / n, 4),
            "inventes": sorted(lus - vrais), "manques": len(vrais - lus),
            "t_50pct_s": t_pour(0.5), "t_80pct_s": t_pour(0.8), "t_90pct_s": t_pour(0.9)}


def securite(m, layout) -> dict:
    pts = np.array([p[1:] for a in m["agents"] for p in a["trajectoire"]], float)
    dedans = np.zeros(len(pts), dtype=bool)
    for r in layout.racks:
        (x0, x1), (y0, y1) = r.x_bounds, r.y_bounds
        dedans |= ((pts[:, 0] > x0) & (pts[:, 0] < x1)
                   & (pts[:, 1] > y0 + BOUT_VIDE_SUD) & (pts[:, 1] < y1 - BOUT_VIDE_NORD))
    trajs = {a["i"]: {round(p[0], 1): np.array(p[1:]) for p in a["trajectoire"]} for a in m["agents"]}
    mini = np.inf
    ids = sorted(trajs)
    for k, i in enumerate(ids):
        for j in ids[k + 1:]:
            for t, p in trajs[i].items():
                q = trajs[j].get(t)
                if q is not None:
                    mini = min(mini, float(np.linalg.norm(p - q)))
    return {"points": int(len(pts)), "dans_un_rack": int(dedans.sum()),
            "distance_min_entre_drones_m": round(mini, 2) if np.isfinite(mini) else None,
            "separation_tenue": bool(mini >= pore.SEPARATION_MIN) if np.isfinite(mini) else None,
            "attentes_de_priorite": sum(a["attentes"] for a in m["agents"])}


def obstacle(m) -> dict:
    """Le bloc posé en cours de mission. Un drone qui passe au-dessus, avec la garde qu'exige le
    planificateur, ne le traverse pas : on compte donc séparément les points à hauteur du bloc
    et les survols, sinon la distance minimale vaut zéro sans qu'il y ait eu de contact."""
    o = m.get("obstacle")
    if not o:
        return {"simule": False, "ok": True}
    pts = np.array([[p[0], *p[1:]] for a in m["agents"] for p in a["trajectoire"] if p[0] >= o["t"]], float)
    if not len(pts):
        return {"simule": True, "ok": True, "points_apres": 0, "dedans": 0}
    sur_emprise = ((pts[:, 1] > o["x"][0]) & (pts[:, 1] < o["x"][1])
                   & (pts[:, 2] > o["y"][0]) & (pts[:, 2] < o["y"][1]))
    dedans = sur_emprise & (pts[:, 3] < o["z"][1])
    survols = pts[sur_emprise & (pts[:, 3] >= o["z"][1])]
    bas = pts[pts[:, 3] < o["z"][1] + 0.3]
    dx = np.clip(np.maximum(o["x"][0] - bas[:, 1], bas[:, 1] - o["x"][1]), 0, None)
    dy = np.clip(np.maximum(o["y"][0] - bas[:, 2], bas[:, 2] - o["y"][1]), 0, None)
    return {"simule": True, "ok": int(dedans.sum()) == 0, "t": o["t"], "points_apres": int(len(pts)),
            "dedans": int(dedans.sum()),
            "points_a_hauteur": int(len(bas)),
            "distance_min_m": round(float(np.hypot(dx, dy).min()), 2) if len(bas) else None,
            "survols": int(len(survols)),
            "garde_au_dessus_m": round(float(survols[:, 3].min() - o["z"][1]), 2) if len(survols) else None,
            "deflexions": sum(1 for e in m["evenements"] if e["genre"] == "deflexion")}


def plan(m) -> dict:
    return {"arrets_prevus": sum(a["arrets_prevus"] for a in m["agents"]),
            "arrets_servis": sum(a["arrets_servis"] for a in m["agents"]),
            "non_servis": sum(a.get("arrets_non_servis", 0) for a in m["agents"]),
            "passages": max(a.get("passes", 1) for a in m["agents"])}


def figure_codes(m, dossier: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    codes = np.array(m["codes_par_t"], float)
    n = len({t["code"] for t in m["verite"]})
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(codes[:, 0], 100 * codes[:, 1] / n, lw=2, color="#444")
    for e in m["evenements"]:
        if e["genre"] == "panne":
            ax.axvline(e["t"], color="red", ls="--", label=f"panne du drone {e['drone']}")
        if e["genre"] == "obstacle":
            ax.axvline(e["t"], color="darkorange", ls="--", label="obstacle posé")
    ax.set(xlabel="temps simulé (s)", ylabel="codes lus (%)", ylim=(0, 100),
           title=f"Pore et al. — {m['drones']} drones, entrepôt {m['seed']} : "
                 f"{codes[-1, 1]:.0f} codes sur {n}")
    ax.grid(alpha=0.3)
    if any(e["genre"] in ("panne", "obstacle") for e in m["evenements"]):
        ax.legend()
    fig.tight_layout()
    fig.savefig(dossier / "codes_dans_le_temps.png", dpi=130)
    plt.close(fig)


def figure_trajectoires(m, layout, dossier: Path) -> None:
    """Le plan connu de l'entrepôt, les arrêts prévus, et le vol réel par-dessus. C'est
    l'équivalent de notre carte finale : eux n'en construisent pas, ils reçoivent celle-ci."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(6.4, 9.0))
    ax.add_patch(Rectangle((INTERIOR.x_min, INTERIOR.y_min), INTERIOR.x_max - INTERIOR.x_min,
                           INTERIOR.y_max - INTERIOR.y_min, fill=False, ec="black", lw=1.5))
    for r in layout.racks:
        ax.add_patch(Rectangle((r.x_bounds[0], r.y_bounds[0]), r.x_bounds[1] - r.x_bounds[0],
                               r.y_bounds[1] - r.y_bounds[0], fc="#555", ec="none"))
    plans = pore.attribue(pore.plan_zigzag(layout, m["drones"]),
                          [np.array(a["trajectoire"][0][1:]) for a in m["agents"]])
    for i, p in enumerate(plans):
        if p:
            ax.scatter([a.position[0] for a in p], [a.position[1] for a in p], s=4,
                       color=COULEURS[i % 3], alpha=0.35, marker="s", zorder=2)
    lus = {c: None for c in m["codes"]}
    positions = {t["code"]: t["position"] for t in m["verite"]}
    manques = [positions[c] for c in positions if c not in lus]
    if manques:
        ax.scatter([p[0] for p in manques], [p[1] for p in manques], s=26, color="red",
                   marker="x", label=f"{len(set(positions) - set(lus))} codes non lus", zorder=5)
    for a in m["agents"]:
        t = np.array(a["trajectoire"], float)
        ax.plot(t[:, 1], t[:, 2], lw=1.1, color=COULEURS[a["i"] % 3],
                label=f"drone {a['i']} : {a['arrets_servis']}/{a['arrets_prevus']} arrêts", zorder=3)
    o = m.get("obstacle")
    if o:
        ax.add_patch(Rectangle((o["x"][0], o["y"][0]), o["x"][1] - o["x"][0], o["y"][1] - o["y"][0],
                               fc="none", ec="red", lw=2, zorder=6))
        ax.annotate("obstacle", (o["x"][1], o["y"][1]), color="red", fontsize=8)
    for e in m["evenements"]:
        if e["genre"] == "chute":
            ax.scatter(e["position"][0], e["position"][1], s=90, color="red", marker="*", zorder=7)
    ax.set(xlabel="x (m)", ylabel="y (m)", aspect="equal",
           title=f"Pore et al. — entrepôt {m['seed']}, {m['fin']}\n"
                 f"petits carrés : arrêts prévus ; traits : vol réel")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(dossier / "plan_et_vol.png", dpi=130)
    plt.close(fig)


def main(dossier: Path) -> None:
    m = json.loads((dossier / "mission.json").read_text())
    layout = make_layout(m["seed"])
    res = {"lecture": lecture(m), "securite": securite(m, layout), "obstacle": obstacle(m),
           "plan": plan(m), "inventaire": m["inventaire"], "fin": m["fin"],
           "t_sim_s": m["t_sim_s"], "mur_min": m["mur_min"],
           "chutes": [(e["t"], e["drone"]) for e in m["evenements"] if e["genre"] == "chute"]}
    (dossier / "resultats.json").write_text(json.dumps(res, indent=1))
    figure_codes(m, dossier)
    figure_trajectoires(m, layout, dossier)

    l, s, p, o = res["lecture"], res["securite"], res["plan"], res["obstacle"]
    print("=" * 72)
    print(f"Pore et al. — {dossier.name} : {m['drones']} drones, entrepot {m['seed']}, "
          f"fin « {m['fin']} » a {m['t_sim_s']:.0f} s ({m['mur_min']} min de calcul)")
    print(f"lecture   : {l['codes_lus']}/{l['codes_vrais']} codes ({l['part']:.0%}), "
          f"inventes {l['inventes']}, 50 % a {l['t_50pct_s']} s, 80 % a {l['t_80pct_s']} s, "
          f"90 % a {l['t_90pct_s']} s")
    print(f"plan      : {p['arrets_servis']}/{p['arrets_prevus']} arrets servis, "
          f"{p['non_servis']} non servis, {p['passages']} passage(s)")
    print(f"securite  : {s['dans_un_rack']}/{s['points']} points dans une structure de rack, "
          f"distance min entre drones {s['distance_min_entre_drones_m']} m "
          f"({'≥' if s['separation_tenue'] else '<'} {pore.SEPARATION_MIN} m exigés), "
          f"{s['attentes_de_priorite']} attentes de priorite")
    if o["simule"]:
        print(f"obstacle  : {'EVITE' if o['ok'] else 'TRAVERSE'} ; {o['dedans']} points dedans ; "
              f"a hauteur du bloc, {o['points_a_hauteur']} points, distance min {o['distance_min_m']} m ; "
              f"{o['survols']} points en survol" +
              (f" a {o['garde_au_dessus_m']} m au-dessus" if o["survols"] else "") +
              f" ; {o['deflexions']} deflexions")
    inv = res["inventaire"]
    print(f"inventaire: {inv['lectures']} lectures, {inv['doublons']} doublons ecartes, "
          f"{inv['a_relire']} arrets a relire, confiance moyenne {inv['confiance_moyenne']}")
    print(f"chutes    : {res['chutes']}")
    print("ANALYSE PORE FINIE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dossier", required=True)
    main(Path(ap.parse_args().dossier))
