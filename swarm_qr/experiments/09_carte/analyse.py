"""Jugement de la carte — étape 4. Aucun simulateur.

La patrouille enregistre la carte, la trajectoire vraie et la vérité des panneaux ; ce fichier
les compare. Le jugement se refait donc en quelques secondes, sans refaire voler le drone.

Les questions sont séparées parce qu'elles n'ont pas les mêmes causes :
  1. le capteur dit-il la vérité ?                         (verification.json)
  2. la carte invente-t-elle des obstacles dans les allées ?
  3. retrouve-t-elle les racks qu'elle a longés ?
  4. les panneaux lus sont-ils au bon endroit, et ses promesses de lisibilité tiennent-elles ?
  5. le drone, guidé par sa seule carte, est-il passé sans toucher un rack ?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from swarm_qr import mapping  # noqa: E402
from swarm_qr.env.config import INTERIOR, RACKS  # noqa: E402
from swarm_qr.env.layout import make_layout  # noqa: E402

import vue3d  # noqa: E402

BANDE_VOL = (0.6, 5.5)
HAUTEUR_PANNEAU = 0.225


# Les racks de l'entrepôt n'ont aucune structure solide aux deux bouts de leur emprise : ni
# montant, ni traverse, ni planche sur 0,70 m au sud et 0,77 m au nord — seulement un panneau
# de signalisation en haut et un pare-chocs bas. Mesuré par rayons physiques sur les trois racks
# (étape 7). L'arbitre juge la structure, pas le rectangle du plan.
BOUT_VIDE_SUD = 0.70
BOUT_VIDE_NORD = 0.77


def emprise_solide(r):
    (x0, x1), (y0, y1) = r.x_bounds, r.y_bounds
    return (x0, x1), (y0 + BOUT_VIDE_SUD, y1 - BOUT_VIDE_NORD)


def dans_un_rack(points, layout, marge: float = 0.0, solide: bool = True) -> np.ndarray:
    p = np.atleast_2d(np.asarray(points, float))
    out = np.zeros(len(p), dtype=bool)
    for r in layout.racks:
        (x0, x1), (y0, y1) = emprise_solide(r) if solide else (r.x_bounds, r.y_bounds)
        out |= ((p[:, 0] > x0 - marge) & (p[:, 0] < x1 + marge)
                & (p[:, 1] > y0 - marge) & (p[:, 1] < y1 + marge))
    return out


def hors_des_murs(points, marge: float = 0.0) -> np.ndarray:
    p = np.atleast_2d(np.asarray(points, float))
    return ~((p[:, 0] > INTERIOR.x_min - marge) & (p[:, 0] < INTERIOR.x_max + marge)
             & (p[:, 1] > INTERIOR.y_min - marge) & (p[:, 1] < INTERIOR.y_max + marge))


def mediane(xs, digits=1):
    return round(float(np.median(xs)), digits) if len(xs) else None


# ---------------------------------------------------------------- les questions

def faux_obstacles(carte, layout) -> dict:
    idx = np.argwhere(carte.occupation > mapping.SEUIL_OCCUPE)
    c = carte.centre(idx) if len(idx) else np.zeros((0, 3))
    c = c[(c[:, 2] > BANDE_VOL[0]) & (c[:, 2] < BANDE_VOL[1])]
    if not len(c):
        return {"cases": 0, "part": None}
    vrai = dans_un_rack(c, layout, marge=0.35) | hors_des_murs(c, marge=-0.35)
    return {"cases": int(len(c)), "part": round(1.0 - float(vrai.mean()), 4)}


def par_etagere(carte, verite) -> dict:
    """Pour chaque vrai panneau, groupé par étagère : la carte a-t-elle posé un obstacle là où
    est le carton, a-t-elle marqué l'espace devant comme lisible, et l'a-t-elle lu ? Les racks
    sont des cadres ouverts : juger des tranches arbitraires compterait du vide comme des
    obstacles manqués."""
    lus = {p.code for p in carte.panneaux}
    groupes: dict[str, dict] = {}
    for t in verite:
        p, n = np.array(t["position"], float), np.array(t["normale"], float)
        niveau = int(np.argmin([abs(p[2] - z - HAUTEUR_PANNEAU) for z in RACKS.shelf_levels]))
        g = groupes.setdefault(f"etagere_{niveau}", {"panneaux": 0, "obstacle": 0, "couvert": 0, "lu": 0})
        g["panneaux"] += 1
        dedans = [p - n * d for d in (0.05, 0.2, 0.4, 0.6)]
        g["obstacle"] += int((carte.etat(dedans) == mapping.OCCUPE).any())
        g["couvert"] += int(carte.couvert_pour([p + n * d for d in (0.3, 0.6, 1.0)], n).any())
        g["lu"] += int(t["code"] in lus and any(
            np.linalg.norm(q.position - p) < mapping.FUSION for q in carte.faces(t["code"])))
    for g in groupes.values():
        n = max(g["panneaux"], 1)
        g["part_obstacle"] = round(g["obstacle"] / n, 4)
        g["part_couvert"] = round(g["couvert"] / n, 4)
        g["part_lu"] = round(g["lu"] / n, 4)
    return dict(sorted(groupes.items()))


def panneaux(carte, verite) -> dict:
    """Chaque carton porte le même code sur ses deux faces : on compare à la face la plus
    proche, sinon on mesurerait l'épaisseur du carton au lieu de l'erreur de la carte."""
    par_code: dict[str, list] = {}
    for t in verite:
        par_code.setdefault(t["code"], []).append(np.array(t["position"], float))
    taille = {t["code"]: t.get("taille", 0.4) for t in verite}
    errs, inventes, par_taille, confirmees = [], [], {}, []
    for p in carte.panneaux:
        if p.code not in par_code:
            inventes.append(p.code)
            continue
        e = min(float(np.linalg.norm(p.position - q)) for q in par_code[p.code])
        errs.append(e)
        if p.lectures >= 2:
            confirmees.append(e)
        par_taille.setdefault(round(taille[p.code], 2), []).append(e)
    tailles_vraies = sorted({round(t.get("taille", 0.4), 2) for t in verite})
    return {
        "codes_vrais": len(par_code), "panneaux_vrais": len(verite),
        "tailles_d_etiquette_m": tailles_vraies,
        "codes_lus": len(carte.codes), "faces_lues": len(carte.panneaux),
        "codes_inventes": inventes,
        "part_codes_lus": round(len(carte.codes) / max(len(par_code), 1), 4),
        "err_cm": mediane([100 * e for e in errs]),
        "err_p90_cm": round(100 * float(np.percentile(errs, 90)), 1) if errs else None,
        "err_max_cm": round(100 * max(errs), 1) if errs else None,
        "err_cm_par_taille": {str(k): mediane([100 * e for e in v]) for k, v in sorted(par_taille.items())},
        "faces_confirmees": len(confirmees),
        "err_max_cm_confirmees": round(100 * max(confirmees), 1) if confirmees else None,
    }


def promesse_de_lisibilite(carte, verite) -> dict:
    """La couverture affirme « un QR posé là aurait été lu ». On le vérifie sur les vrais
    panneaux : parmi ceux dont l'espace devant est marqué couvert, combien ont vraiment été
    lus ? Et parmi les autres ?"""
    lus = {p.code for p in carte.panneaux}
    couverts_lus = couverts = non_couverts_lus = non_couverts = 0
    for t in verite:
        p, n = np.array(t["position"], float), np.array(t["normale"], float)
        couvert = bool(carte.couvert_pour([p + n * d for d in (0.3, 0.6, 1.0)], n).any())
        lu = t["code"] in lus and any(
            np.linalg.norm(q.position - p) < mapping.FUSION for q in carte.faces(t["code"]))
        if couvert:
            couverts += 1
            couverts_lus += int(lu)
        else:
            non_couverts += 1
            non_couverts_lus += int(lu)
    return {"panneaux_couverts": couverts, "lus_parmi_couverts": couverts_lus,
            "part_lus_si_couvert": round(couverts_lus / max(couverts, 1), 4),
            "panneaux_non_couverts": non_couverts, "lus_parmi_non_couverts": non_couverts_lus,
            "part_lus_si_non_couvert": round(non_couverts_lus / max(non_couverts, 1), 4)}


def semantique(carte, verite) -> dict:
    """Le canal sémantique rempli par l'œil appris (étape 7) : les cubes marqués « carton »
    sont-ils sur de vrais cartons ? Un vrai carton porte un panneau sur chacune de ses deux
    faces ; un cube à moins de 60 cm d'un panneau est sur un carton ou contre lui."""
    idx = np.argwhere(carte.semantique == mapping.SEM_CARTON)
    if not len(idx):
        return {"cubes_carton": 0, "sur_un_vrai_carton": 0, "part_fausses": None}
    centres = carte.centre(idx)
    vrais = np.array([t["position"] for t in verite], float)
    d = np.array([np.linalg.norm(vrais - c, axis=1).min() for c in centres])
    return {"cubes_carton": int(len(idx)), "sur_un_vrai_carton": int((d < 0.6).sum()),
            "part_fausses": round(float((d >= 0.6).mean()), 4)}


def pistes(carte, verite, layout=None) -> dict:
    """Les motifs repérés non lus : à quelle distance du vrai panneau le plus proche ? Une piste
    à un mètre d'un panneau, dans un rack, est un vrai carton placé grossièrement ; une piste
    hors de toute emprise de rack est un fantôme."""
    vrais = np.array([t["position"] for t in verite], float)
    d = np.array([np.linalg.norm(vrais - q.position, axis=1).min() for q in carte.pistes])
    n = max(len(d), 1)
    hors = int((~dans_un_rack(np.array([q.position for q in carte.pistes]), layout, marge=0.3)).sum()) \
        if layout is not None and len(d) else 0
    return {"total": len(carte.pistes), "sur_un_vrai_panneau": int((d < 0.5).sum()),
            "part_fausses": round(float((d >= 0.5).mean()) if len(d) else 0.0, 4),
            "part_a_moins_de_1_m": round(float((d < 1.0).mean()) if len(d) else 0.0, 4),
            "part_a_moins_de_2_m": round(float((d < 2.0).mean()) if len(d) else 0.0, 4),
            "hors_de_toute_emprise": hors}


def securite_du_vol(vol, layout) -> dict:
    """Le drone n'a connu que sa carte. Est-il passé sans entrer dans un rack ?"""
    traj = np.array([t[1:] for t in vol["trajectoire"]], float)
    dedans = dans_un_rack(traj, layout)
    bout_vide = dans_un_rack(traj, layout, solide=False) & ~dedans
    dist = np.full(len(traj), np.inf)
    for r in layout.racks:
        (x0, x1), (y0, y1) = emprise_solide(r)
        dx = np.maximum(np.maximum(x0 - traj[:, 0], traj[:, 0] - x1), 0.0)
        dy = np.maximum(np.maximum(y0 - traj[:, 1], traj[:, 1] - y1), 0.0)
        dist = np.minimum(dist, np.hypot(dx, dy))
    allers = vol["allers"]
    transits = [a["transit"] for a in allers if a["transit"]]
    traversees = [a["traversee"] for a in allers if a["traversee"]]
    return {
        "points_de_trajectoire": int(len(traj)),
        "points_dans_un_rack": int(dedans.sum()),
        "points_dans_le_bout_vide_d_un_rack": int(bout_vide.sum()),
        "distance_min_a_un_rack_m": round(float(dist.min()), 2),
        "allers": len(allers),
        "debuts_inaccessibles": sum(1 for a in allers if a["transit"] is None),
        "transits_atteints": sum(1 for b in transits if b["phase"] == "atteint"),
        "traversees_atteintes": sum(1 for b in traversees if b["phase"] == "atteint"),
        "abandons": [b["raison"] for b in transits + traversees if b["phase"] == "abandon"],
        "replanifications": sum(b.get("replanifications", 0) for b in transits),
        "points_de_passage_median": mediane([b["points"] for b in transits], 0),
        "t_transit_median_s": mediane([b["t_total"] for b in transits]),
        "t_traversee_median_s": mediane([b["t_total"] for b in traversees]),
    }


def chemins_sur_la_carte(carte, layout, pas: float = 0.2) -> dict:
    """Douze trajets d'un coin à l'autre, calculés sur la carte finale."""
    z = 1.6
    coins = [np.array([INTERIOR.x_min + 1.5, INTERIOR.y_min + 1.5, z]),
             np.array([INTERIOR.x_max - 1.5, INTERIOR.y_max - 1.5, z]),
             np.array([INTERIOR.x_min + 1.5, INTERIOR.y_max - 1.5, z]),
             np.array([INTERIOR.x_max - 1.5, INTERIOR.y_min + 1.5, z])]
    trouves = propres = par_l_inconnu = sur_obstacle = 0
    for a in coins:
        for b in coins:
            if a is b:
                continue
            route = carte.chemin(a, b)
            if route is None:
                continue
            trouves += 1
            etapes = [a] + list(route) + [b]
            fautif = None
            for p, q in zip(etapes[:-1], etapes[1:]):
                n = max(2, int(np.linalg.norm(q - p) / pas))
                seg = p + (q - p) * np.linspace(0, 1, n)[:, None]
                dedans = dans_un_rack(seg, layout)
                if dedans.any():
                    fautif = seg[dedans]
                    break
            if fautif is None:
                propres += 1
            elif (carte.etat(fautif) == mapping.OCCUPE).any():
                sur_obstacle += 1
            else:
                par_l_inconnu += 1
    return {"essais": 12, "trouves": trouves, "hors_de_toute_emprise": propres,
            "coupent_une_emprise_par_l_inconnu_ou_le_libre": par_l_inconnu,
            "passent_sur_un_obstacle_connu": sur_obstacle}


# ---------------------------------------------------------------- figures

def figure_comparaison(carte, vol, layout, sortie: Path) -> None:
    """La carte vue de dessus, avec la vérité par-dessus : racks en rouge, panneaux en points."""
    import cv2

    ech = 6
    img = mapping.vue_de_dessus(carte, echelle=ech, trajectoire=[t[1:] for t in vol["trajectoire"]])
    ny = carte.forme[1]

    def px(p):
        i, j = carte.indice([p])[0][:2]
        return int((i + 0.5) * ech), int((ny - j - 0.5) * ech)

    for r in layout.racks:
        (x0, x1), (y0, y1) = r.x_bounds, r.y_bounds
        cv2.rectangle(img, px([x0, y0, 0]), px([x1, y1, 0]), (0, 0, 255), 2)
    for t in vol["verite"]:
        cv2.circle(img, px(t["position"]), 2, (0, 0, 255), -1)
    # les codes lus sont à un centimètre de la vérité : dessinés avant elle, le rouge les
    # recouvre et la carte paraît sans vert. On les repasse par-dessus, ainsi que les pistes.
    for q in carte.pistes:
        cv2.circle(img, px(q.position), 2, (0, 140, 255), -1)
    for pan in carte.panneaux:
        cv2.circle(img, px(pan.position), 3, (0, 200, 0), -1)
    cv2.putText(img, "rouge = verite (racks, panneaux) ; vert = lus ; orange = pistes ; bleu = trajet",
                (8, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.imwrite(str(sortie), img)


# ---------------------------------------------------------------- résumé

def main(dossier: Path = HERE) -> None:
    carte = mapping.Carte.charge(dossier / "carte")
    vol = json.loads((dossier / "vol.json").read_text())
    layout = make_layout(vol["seed"])
    verif = json.loads((dossier / "verification.json").read_text()) if (dossier / "verification.json").exists() else {}

    res = {
        "vol": {k: v for k, v in vol.items() if k not in ("verite", "trajectoire", "allers", "racks")},
        "verification": verif,
        "carte": carte.resume(),
        "faux_obstacles": faux_obstacles(carte, layout),
        "par_etagere": par_etagere(carte, vol["verite"]),
        "panneaux": panneaux(carte, vol["verite"]),
        "lisibilite": promesse_de_lisibilite(carte, vol["verite"]),
        "pistes": pistes(carte, vol["verite"], layout),
        "semantique": semantique(carte, vol["verite"]),
        "securite": securite_du_vol(vol, layout),
        "chemins": chemins_sur_la_carte(carte, layout),
    }
    (dossier / "resultats.json").write_text(json.dumps(res, indent=1))
    figure_comparaison(carte, vol, layout, dossier / "comparaison.png")
    vue3d.genere(carte, vol, dossier / "carte_3d.html")
    vue3d.image(carte, vol, dossier / "carte_3d.png")

    c, p, f, s, li, q, ch = (res["carte"], res["panneaux"], res["faux_obstacles"], res["securite"],
                             res["lisibilite"], res["pistes"], res["chemins"])
    print("=" * 72)
    if verif:
        print(f"capteur    : ecart median contre la physique "
              f"{[v['ecart_median_cm'] for v in verif['physique']]} cm aux caps "
              f"{[v['cap_deg'] for v in verif['physique']]} ; rotation {verif['rotation_cm']} cm ; "
              f"{verif['ms_lidar']} ms par tour")
    print(f"carte      : {c['part_connue']:.1%} de l'entrepot connu, {c['couvertes']:,} cases "
          f"lisibles, {c['octets'] / 1e6:.1f} Mo, {res['vol']['ms']} ms par observation")
    print(f"obstacles  : {f['cases']:,} cases occupees a hauteur de vol, {f['part']:.2%} en pleine allee")
    for nom, d in res["par_etagere"].items():
        print(f"  {nom} : {d['panneaux']} panneaux ; obstacle pose sur {d['part_obstacle']:.0%}, "
              f"espace devant lisible sur {d['part_couvert']:.0%}, lus {d['part_lu']:.0%}")
    print(f"panneaux   : {p['codes_lus']}/{p['codes_vrais']} codes lus ({p['faces_lues']} faces sur "
          f"{p['panneaux_vrais']}), inventes {p['codes_inventes']}, position a {p['err_cm']} cm "
          f"(p90 {p['err_p90_cm']}, max {p['err_max_cm']}) ; par taille d'etiquette "
          f"{p['err_cm_par_taille']} ; faces confirmees par 2 lectures ou plus : "
          f"{p['faces_confirmees']}, pire ecart {p['err_max_cm_confirmees']} cm")
    print(f"lisibilite : lus {li['lus_parmi_couverts']}/{li['panneaux_couverts']} = "
          f"{li['part_lus_si_couvert']:.0%} des panneaux couverts, contre "
          f"{li['part_lus_si_non_couvert']:.0%} des non couverts")
    print(f"pistes     : {q['total']} motifs reperes non lus, {q['part_fausses']:.0%} a plus de 50 cm d'un panneau, "
          f"{q['part_a_moins_de_1_m']:.0%} a moins de 1 m, {q['part_a_moins_de_2_m']:.0%} a moins de 2 m, "
          f"{q['hors_de_toute_emprise']} hors de toute emprise de rack")
    sm = res["semantique"]
    if sm["cubes_carton"]:
        print(f"semantique : {sm['cubes_carton']} cubes marques carton, {sm['part_fausses']:.0%} loin de tout carton")
    print(f"securite   : {s['points_dans_un_rack']}/{s['points_de_trajectoire']} points de trajectoire "
          f"dans un rack ({s['points_dans_le_bout_vide_d_un_rack']} dans un bout vide d'emprise), au plus pres {s['distance_min_a_un_rack_m']} m ; transits "
          f"{s['transits_atteints']}/{s['allers'] - s['debuts_inaccessibles']} atteints, "
          f"traversees {s['traversees_atteintes']}, abandons {s['abandons']}, "
          f"{s['debuts_inaccessibles']} debuts inaccessibles, {s['replanifications']} chemins recalcules en route")
    print(f"chemins    : {ch['trouves']}/12 trouves, {ch['hors_de_toute_emprise']} hors de toute "
          f"emprise, {ch['coupent_une_emprise_par_l_inconnu_ou_le_libre']} par un bout d'emprise "
          f"non occupe, {ch['passent_sur_un_obstacle_connu']} SUR UN OBSTACLE CONNU")
    print("=" * 72)
    print("comparaison.png, carte_3d.html, carte_3d.png, resultats.json ecrits")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dossier", default=str(HERE), help="dossier de la carte à juger (carte.npz, vol.json)")
    main(Path(ap.parse_args().dossier))
