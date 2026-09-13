"""Extrait les quatre vols du système en un seul fichier compact, pour le rejeu en 3D.

Tout vient des vols enregistrés. Une seule grandeur est reconstruite, et c'est dit dans le
fichier : l'instant de lecture de chaque code. Le journal note bien le nombre de codes lus cinq
fois par seconde, mais le nom du code n'apparaît que dans les instantanés, toutes les dix
secondes. On croise donc les deux : l'instantané dit quels codes sont arrivés dans l'intervalle,
la courbe dit à quelles secondes précises le compteur a bougé.

    extrait.py --sortie experiments/14_rejeu/vols.json
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import sys
from pathlib import Path

import numpy as np

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI.parents[2]))
VOLS = ICI.parents[0] / "11_mission" / "tests of system"
CAS = [("nominal", "eval_nominal", "Nominal"),
       ("panne", "eval_panne", "Panne d'un drone"),
       ("inconnu", "eval_9019", "Entrepôt jamais vu"),
       ("obstacle", "eval_obstacle", "Obstacle en cours de mission")]
CELL = 0.25
X0, Y0 = -11.0, -13.0
PAS_FOG = 2                      # une case de brouillard pour deux cases de carte


def instants_de_lecture(m: dict) -> dict[str, float]:
    """Le premier instant où chaque code entre à l'inventaire, reconstruit comme expliqué en tête."""
    sauts = []
    serie = m["codes_par_t"]
    for (t0, n0), (t1, n1) in zip(serie, serie[1:]):
        for _ in range(int(n1 - n0)):
            sauts.append(float(t1))
    connus: set[str] = set()
    out: dict[str, float] = {}
    k = 0
    for inst in m["instantanes"]:
        nouveaux = [p["code"] for p in inst["panneaux"] if p["code"] not in connus]
        vus = []
        for c in nouveaux:
            if c not in connus:
                connus.add(c)
                vus.append(c)
        for c in vus:
            out[c] = sauts[k] if k < len(sauts) else float(inst["t"])
            k += 1
    return out


def cap_par_instant(agent: dict, traj: np.ndarray) -> np.ndarray:
    """Le cap du drone : il regarde sa cible quand il est arrivé devant, sinon là où il va.
    Le journal ne l'enregistre pas cinq fois par seconde, mais les deux sources qui le donnent
    — la cible visée et le sens de marche — suffisent à le reconstituer sans ambiguïté."""
    caps = np.zeros(len(traj))
    for k, (t, x, y, z) in enumerate(traj):
        active = None
        for d in agent["decisions"]:
            if d["t"] <= t <= d.get("fin", 1e9):
                active = d
        vise = None
        if active is not None and active.get("origine") is not None:
            cible = np.array(active["position"][:2], float)
            if float(np.linalg.norm(np.array([x, y]) - cible)) < 1.2:
                vise = np.array(active["origine"][:2], float) - np.array([x, y])
        if vise is None:
            j = min(k + 3, len(traj) - 1)
            vise = traj[j, 1:3] - np.array([x, y])
        n = float(np.linalg.norm(vise))
        caps[k] = math.atan2(vise[1], vise[0]) if n > 1e-3 else (caps[k - 1] if k else 0.0)
    # lissage court : le cap ne saute pas d'un cinquième de seconde à l'autre
    lisse = np.unwrap(caps)
    noyau = np.ones(5) / 5
    lisse = np.convolve(np.pad(lisse, 2, mode="edge"), noyau, mode="valid")
    return lisse


def brouillard(dossier: Path, m: dict) -> dict:
    """Ce que la carte connaît, à chaque instantané, en une image binaire compressée."""
    images, temps = [], []
    for inst in m["instantanes"]:
        f = dossier / "instantanes" / f"{inst['k']:03d}_carte.npz"
        if not f.exists():
            continue
        occ = np.load(f)["occupation"]
        tr = occ[:, :, int(0.6 / CELL):int(5.5 / CELL) + 1]
        connu = ((tr > 1.0).any(axis=2) | (tr < -1.0).any(axis=2))
        petit = connu[::PAS_FOG, ::PAS_FOG]
        images.append(base64.b64encode(np.packbits(petit.ravel())).decode())
        temps.append(round(float(inst["t"]), 1))
    forme = np.zeros((1, 1)) if not images else connu[::PAS_FOG, ::PAS_FOG]
    return {"nx": int(forme.shape[0]), "ny": int(forme.shape[1]),
            "cell": CELL * PAS_FOG, "x0": X0, "y0": Y0, "t": temps, "images": images}


def un_vol(cle: str, dossier: str, titre: str) -> dict:
    d = VOLS / dossier
    m = json.loads((d / "mission.json").read_text())
    r = json.loads((d / "resultats.json").read_text())
    lectures = instants_de_lecture(m)
    codes_vrais = sorted({t["code"] for t in m["verite"]})

    drones = []
    for a in m["agents"]:
        traj = np.array(a["trajectoire"], float)
        caps = cap_par_instant(a, traj)
        drones.append({
            "i": a["i"], "vivant": a["vivant"],
            "t0": round(float(traj[0, 0]), 2),
            "pas": round(float(np.median(np.diff(traj[:, 0]))), 3),
            "xyz": [round(float(v), 2) for v in traj[:, 1:].ravel()],
            "cap": [round(float(v), 3) for v in caps],
            "inclinaison": [round(float(v), 1) for _, v in a["inclinaisons"]],
            "cibles": [{"t": d0["t"], "fin": d0.get("fin", m["t_sim_s"]), "genre": d0["genre"],
                        "p": [round(float(v), 2) for v in d0["position"]],
                        "o": [round(float(v), 2) for v in (d0.get("origine") or d0["position"])],
                        "lu": bool(d0.get("lu")), "phase": d0.get("phase", "")}
                       for d0 in a["decisions"]],
        })

    evenements = [{"t": e["t"], "genre": e["genre"], "drone": e.get("drone")}
                  for e in m["evenements"] if e["genre"] in ("panne", "chute", "obstacle")]
    l = r["lecture"]
    jalons = [{"t": l[f"t_{p}pct_s"], "nom": f"{p} % de l'inventaire"}
              for p in (50, 80, 90) if l.get(f"t_{p}pct_s")]

    return {
        "cle": cle, "titre": titre, "seed": m["seed"], "duree": m["t_sim_s"], "fin": m["fin"],
        "codes_attendus": len(codes_vrais), "codes_lus": l["codes_lus"],
        "t50": l["t_50pct_s"], "t80": l["t_80pct_s"], "t90": l["t_90pct_s"],
        "chutes": len([e for e in m["evenements"] if e["genre"] == "chute"]),
        "dans_rack": r["securite"]["dans_un_rack"], "points": r["securite"]["points"],
        "dist_min": r["securite"]["distance_min_entre_drones_m"],
        "racks": [{"x": [round(v, 2) for v in rk["x"]], "y": [round(v, 2) for v in rk["y"]]}
                  for rk in m["racks"]],
        "panneaux": [{"c": t["code"], "p": [round(float(v), 2) for v in t["position"]],
                      "n": [round(float(v), 2) for v in t["normale"]], "s": t["taille"]}
                     for t in m["verite"]],
        "lectures": {c: round(lectures[c], 1) for c in sorted(lectures)},
        "non_lus": [c for c in codes_vrais if c not in lectures],
        "drones": drones, "evenements": evenements, "jalons": jalons,
        "obstacle": m.get("obstacle"),
        "courbe": [[round(t, 1), int(n)] for t, n in m["codes_par_t"][::5]],
        "brouillard": brouillard(d, m),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sortie", default=str(ICI / "vols.json"))
    a = ap.parse_args()
    from swarm_qr.env.config import INTERIOR, RACKS

    vols = [un_vol(*c) for c in CAS]
    paquet = {
        "entrepot": {"x": [INTERIOR.x_min, INTERIOR.x_max], "y": [INTERIOR.y_min, INTERIOR.y_max],
                     "plafond": INTERIOR.z_ceiling, "rack_profondeur": RACKS.depth,
                     "rack_hauteur": RACKS.height, "etages": list(RACKS.shelf_levels)},
        "vols": vols,
    }
    texte = json.dumps(paquet, separators=(",", ":"), ensure_ascii=False)
    Path(a.sortie).write_text(texte)
    print(f"{len(vols)} vols ecrits dans {a.sortie} : {len(texte) / 1e6:.2f} Mo")
    for v in vols:
        n = sum(len(d["cap"]) for d in v["drones"])
        print(f"  {v['titre']:32s} {v['duree']:.0f} s | {n} poses | "
              f"{len(v['lectures'])}/{v['codes_attendus']} codes dates | "
              f"{len(v['brouillard']['images'])} images de brouillard")


if __name__ == "__main__":
    main()
