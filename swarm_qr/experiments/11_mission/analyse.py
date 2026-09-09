"""Le jugement d'une mission — étape 5. Sans simulateur : il lit mission.json et la carte.

  analyse.py --dossier nominale          les cinq vérifications, la lecture, la couverture, les figures

Les cinq vérifications de la porte : aucun blocage, aucun doublon, une dispersion initiale, la
panne absorbée, une fin propre. Puis ce que la mission a produit : codes lus dans le temps,
part de l'entrepôt connue, sécurité du vol.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from swarm_qr import mapping  # noqa: E402
from swarm_qr.env.layout import make_layout  # noqa: E402

BLOCAGE_S = 60.0          # un drone vivant sans cible plus longtemps, alors qu'un autre en a
DOUBLON_M = 2.0           # deux cibles actives plus proches que ça au même instant
DISPERSION_M = 3.0        # les premières cibles doivent être au moins aussi éloignées
BOUT_VIDE_SUD, BOUT_VIDE_NORD = 0.70, 0.77     # mesuré à l'étape 7 : les bouts des racks sont vides


def charge(dossier: Path):
    m = json.loads((dossier / "mission.json").read_text())
    carte = mapping.Carte.charge(dossier / "carte")
    return m, carte


def intervalles(agent, t_mort: float = float("inf")) -> list[tuple[float, float, np.ndarray, str]]:
    """Les cibles actives d'un drone, closes à sa panne ou à sa chute : après, sa zone est à
    reprendre par les autres, ce n'est pas un doublon."""
    out = []
    for d in agent["decisions"]:
        fin = min(d.get("fin", float("inf")), t_mort)
        if fin > d["t"]:
            out.append((d["t"], fin, np.array(d["position"]), d["genre"]))
    return out


def instant_de_mort(m, i: int) -> float:
    return min((e["t"] for e in m["evenements"] if e["genre"] in ("panne", "chute") and e.get("drone") == i),
               default=float("inf"))


def blocages(m) -> dict:
    """Un drone vivant qui reste sans cible plus de BLOCAGE_S alors qu'un coéquipier en a une."""
    sans = [e for e in m["evenements"] if e["genre"] == "sans_cible"]
    t_fin = m["t_sim_s"]
    longs = []
    for e in sans:
        i = e["drone"]
        reprise = next((d["t"] for d in m["agents"][i]["decisions"] if d["t"] > e["t"]), t_fin)
        duree = reprise - e["t"]
        autres = any(any(d["t"] <= e["t"] <= d.get("fin", t_fin) for d in a["decisions"])
                     for a in m["agents"] if a["i"] != i)
        if duree > BLOCAGE_S and autres:
            longs.append({"drone": i, "t": e["t"], "duree_s": round(duree, 1)})
    immobiles = []
    for a in m["agents"]:
        tr = np.array([p[1:] for p in a["trajectoire"]])
        ts = np.array([p[0] for p in a["trajectoire"]])
        for d in a["decisions"]:
            fin = d.get("fin", t_fin)
            sel = (ts >= d["t"]) & (ts <= fin)
            if fin - d["t"] > BLOCAGE_S and sel.sum() > 5 and np.ptp(tr[sel], axis=0).max() < 0.3:
                immobiles.append({"drone": a["i"], "t": d["t"], "genre": d["genre"]})
    return {"sans_cible_prolonge": longs, "immobile_avec_cible": immobiles,
            "ok": not longs and not immobiles}


def doublons(m) -> dict:
    """Deux drones avec des cibles actives à moins de DOUBLON_M l'un de l'autre au même instant."""
    ivs = [(a["i"], iv) for a in m["agents"] for iv in intervalles(a, instant_de_mort(m, a["i"]))]
    cas = []
    for k, (i, (t0, t1, p, g)) in enumerate(ivs):
        for j, (u0, u1, q, h) in ivs[k + 1:]:
            if j == i:
                continue
            if max(t0, u0) < min(t1, u1) and np.linalg.norm(p - q) < DOUBLON_M:
                cas.append({"drones": [i, j], "t": round(max(t0, u0), 1),
                            "distance_m": round(float(np.linalg.norm(p - q)), 2), "genres": [g, h]})
    return {"cas": cas, "ok": not cas}


def dispersion(m) -> dict:
    premieres = [np.array(a["decisions"][0]["position"]) for a in m["agents"] if a["decisions"]]
    if len(premieres) < 2:
        return {"distances_m": [], "ok": True}
    d = [float(np.linalg.norm(p - q)) for k, p in enumerate(premieres) for q in premieres[k + 1:]]
    return {"distances_m": [round(x, 1) for x in d], "min_m": round(min(d), 1), "ok": min(d) >= DISPERSION_M}


def panne(m) -> dict:
    ev = next((e for e in m["evenements"] if e["genre"] == "panne"), None)
    if ev is None:
        return {"simulee": False, "ok": True}
    i, t = ev["drone"], ev["t"]
    apres = [d for a in m["agents"] if a["i"] != i for d in a["decisions"] if d["t"] > t]
    # les coéquipiers ont-ils visité la zone que le drone en panne visait ?
    derniere = next((d for d in reversed(m["agents"][i]["decisions"]) if d["t"] <= t), None)
    reprise = None
    if derniere is not None:
        cible = np.array(derniere["position"])
        proches = [d for d in apres if np.linalg.norm(np.array(d["position"]) - cible) < 4.0]
        reprise = {"n": len(proches), "premiere_t": min((d["t"] for d in proches), default=None)}
    codes = np.array(m["codes_par_t"])
    avant = int(codes[codes[:, 0] <= t][-1, 1]) if (codes[:, 0] <= t).any() else 0
    return {"simulee": True, "drone": i, "t": t, "decisions_des_autres_apres": len(apres),
            "reprise_de_sa_zone": reprise, "codes_avant": avant, "codes_fin": int(codes[-1, 1]),
            "ok": len(apres) > 0 and int(codes[-1, 1]) > avant}


def fin_propre(m) -> dict:
    return {"fin": m["fin"], "t_sim_s": m["t_sim_s"], "ok": m["fin"] in ("plus aucune cible", "budget epuise")}


def securite(m, layout) -> dict:
    pts = np.array([p[1:] for a in m["agents"] for p in a["trajectoire"]], float)
    dedans = np.zeros(len(pts), dtype=bool)
    for r in layout.racks:
        (x0, x1), (y0, y1) = r.x_bounds, r.y_bounds
        y0, y1 = y0 + BOUT_VIDE_SUD, y1 - BOUT_VIDE_NORD
        dedans |= (pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1)
    # rapprochements entre drones
    mini = np.inf
    trajs = {a["i"]: {round(p[0], 1): np.array(p[1:]) for p in a["trajectoire"]} for a in m["agents"]}
    ids = sorted(trajs)
    for k, i in enumerate(ids):
        for j in ids[k + 1:]:
            for t, p in trajs[i].items():
                q = trajs[j].get(t)
                if q is not None:
                    mini = min(mini, float(np.linalg.norm(p - q)))
    return {"points": int(len(pts)), "dans_un_rack": int(dedans.sum()),
            "distance_min_entre_drones_m": round(mini, 2) if np.isfinite(mini) else None,
            "attentes_de_priorite": sum(a["attentes"] for a in m["agents"])}


def lecture(m, carte) -> dict:
    vrais = {t["code"]: np.array(t["position"]) for t in m["verite"]}
    codes = np.array(m["codes_par_t"])
    n = len(vrais)
    def t_pour(part):
        k = np.argmax(codes[:, 1] >= part * n)
        return float(codes[k, 0]) if codes[k, 1] >= part * n else None
    genres = {}
    for a in m["agents"]:
        for d in a["decisions"]:
            g = genres.setdefault(d["genre"], {"n": 0, "atteintes": 0, "lus": 0})
            g["n"] += 1
            g["atteintes"] += int(d.get("phase") == "atteint")
            g["lus"] += int(bool(d.get("lu")))
    return {"codes_vrais": n, "codes_lus": len(carte.codes), "part": round(len(carte.codes) / n, 4),
            "inventes": [c for c in carte.codes if c not in vrais],
            "t_50pct_s": t_pour(0.5), "t_80pct_s": t_pour(0.8), "t_90pct_s": t_pour(0.9),
            "part_connue": m["resume"]["part_connue"], "decisions_par_genre": genres}


def figures(m, carte, dossier: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    codes = np.array(m["codes_par_t"])
    n = len({t["code"] for t in m["verite"]})          # 114 codes, pas 228 faces
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(codes[:, 0], 100 * codes[:, 1] / n, lw=2)
    for e in m["evenements"]:
        if e["genre"] == "panne":
            ax.axvline(e["t"], color="red", ls="--", label=f"panne du drone {e['drone']}")
    ax.set(xlabel="temps simulé (s)", ylabel="codes lus (%)", ylim=(0, 100),
           title=f"{m['drones']} drones, entrepôt {m['seed']} : {codes[-1, 1]:.0f} codes sur {n}")
    ax.grid(alpha=0.3)
    if any(e["genre"] == "panne" for e in m["evenements"]):
        ax.legend()
    fig.tight_layout()
    fig.savefig(dossier / "codes_dans_le_temps.png", dpi=130)


def main(dossier: Path) -> None:
    m, carte = charge(dossier)
    layout = make_layout(m["seed"])
    res = {"blocages": blocages(m), "doublons": doublons(m), "dispersion": dispersion(m),
           "panne": panne(m), "fin": fin_propre(m), "securite": securite(m, layout),
           "lecture": lecture(m, carte), "mur_min": m["mur_min"],
           "abandons": [e for e in m["evenements"] if e["genre"] == "abandon"],
           "ms_par_observation": {a["i"]: a["ms"] for a in m["agents"]}}
    (dossier / "resultats.json").write_text(json.dumps(res, indent=1))
    figures(m, carte, dossier)
    ok = lambda b: "OUI" if b else "NON"
    print("=" * 72)
    print(f"mission {dossier.name} : {m['drones']} drones, entrepot {m['seed']}, fin « {m['fin']} » a {m['t_sim_s']:.0f} s "
          f"({m['mur_min']} min de calcul)")
    print(f"1. aucun blocage      : {ok(res['blocages']['ok'])}  {res['blocages']['sans_cible_prolonge']} {res['blocages']['immobile_avec_cible']}")
    print(f"2. aucun doublon      : {ok(res['doublons']['ok'])}  {len(res['doublons']['cas'])} cas")
    print(f"3. dispersion         : {ok(res['dispersion']['ok'])}  distances entre premieres cibles {res['dispersion']['distances_m']} m")
    p = res["panne"]
    print(f"4. panne absorbee     : {ok(p['ok'])}  " + (f"drone {p['drone']} a {p['t']} s, {p['decisions_des_autres_apres']} decisions des autres ensuite, "
          f"codes {p['codes_avant']} -> {p['codes_fin']}, reprise de sa zone {p['reprise_de_sa_zone']}" if p["simulee"] else "(pas de panne simulee)"))
    print(f"5. fin propre         : {ok(res['fin']['ok'])}  « {m['fin']} »")
    l = res["lecture"]
    print(f"lecture   : {l['codes_lus']}/{l['codes_vrais']} codes ({l['part']:.0%}), inventes {l['inventes']}, "
          f"50 % a {l['t_50pct_s']} s, 80 % a {l['t_80pct_s']} s, 90 % a {l['t_90pct_s']} s ; {l['part_connue']:.0%} connu")
    print(f"decisions : {l['decisions_par_genre']}")
    s = res["securite"]
    print(f"securite  : {s['dans_un_rack']}/{s['points']} points dans une structure de rack, distance min entre drones "
          f"{s['distance_min_entre_drones_m']} m, {s['attentes_de_priorite']} pas d'attente de priorite")
    ab = {}
    for e in m["evenements"]:
        if e["genre"] == "abandon":
            ab[e["raison"]] = ab.get(e["raison"], 0) + 1
    etr = sum(a["compte"].get("codes_etrangers", 0) for a in m["agents"])
    chutes = [e for e in m["evenements"] if e["genre"] == "chute"]
    if m["agents"] and m["agents"][0].get("inclinaisons"):
        incl = {a["i"]: max(v for _, v in a["inclinaisons"]) for a in m["agents"]}
        print(f"inclinaison max par drone (deg) : {incl}")
    print(f"abandons  : {ab} ; codes etrangers ignores : {etr} ; chutes : {[(e['t'], e['drone']) for e in chutes]}")
    if "cycles" in m:
        c = m["cycles"]
        print(f"cycles    : {c['n']} ; mediane {c['mediane_ms']} ms ; max {c['max_ms']} ms ; > 0,5 s : {c['lents_plus_de_500_ms']} ; > 1 s : {c['lents_plus_de_1_s']}")
    print(f"couts     : {res['ms_par_observation']}")
    print("=" * 72)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dossier", required=True)
    main(Path(ap.parse_args().dossier))
