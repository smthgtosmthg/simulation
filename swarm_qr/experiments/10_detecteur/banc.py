"""Banc de mesure du détecteur — étape 7. Sans simulateur : il juge sur les jeux annotés.

  banc.py                      juge toutes les variantes entraînées, choisit, écrit resultats.json

Quatre chiffres, dont le premier décide de tout :
  1. la portée de repérage d'un QR, comparée à celle du repérage classique de l'étape 2, sur
     les MÊMES images (le panneau visé, à distance connue) ;
  2. les objets présents et non vus, par distance ;
  3. les fausses alertes : sur les 300 images sans aucun QR de l'étape 2 (le classique en
     donnait 27 %), et sur les entrepôts scellés ;
  4. la vitesse et la mémoire pendant le vol.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
JEU = HERE / "jeu"
ETAPE2 = HERE.parent / "07_enveloppe"

CLASSES = ("qr", "carton")
BINS_DISTANCE = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 12.0]
IOU_OK = 0.5
IOU_IGNORE = 0.3
SEUILS_CONF = (0.25, 0.5)
PORTEE_TAUX = 0.9              # même définition qu'à l'étape 2 : dernière distance tenue à 90 %


def charge(nom: str) -> list[dict]:
    p = JEU / nom / "manifeste.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    aire = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / aire if aire > 0 else 0.0


def predit(modele, dossier: Path, images: list[str], imgsz: int, conf: float) -> dict[str, list]:
    """Toutes les prédictions d'un jeu, au seuil le plus bas ; les seuils plus hauts filtrent."""
    out = {}
    chemins = [str(dossier / "images" / im) for im in images]
    for k in range(0, len(chemins), 16):
        for r in modele.predict(chemins[k:k + 16], imgsz=imgsz, conf=conf, half=True, device=0,
                                verbose=False):
            nom = Path(r.path).name
            b = r.boxes
            out[nom] = [(int(c), float(s), [float(v) for v in xyxy])
                        for c, s, xyxy in zip(b.cls.tolist(), b.conf.tolist(), b.xyxy.tolist())]
    return out


def apparie(preds, objets, conf: float):
    """Chaque vérité visible est trouvée ou manquée ; chaque prédiction est juste, tolérée
    (elle recouvre un objet trop caché pour être annoté) ou fausse."""
    trouves, faux = {}, []
    for classe_i, classe in enumerate(CLASSES):
        verites = [o for o in objets if o["classe"] == classe]
        vis = [o for o in verites if not o["ignore"]]
        pris = set()
        for c, s, box in sorted([p for p in preds if p[0] == classe_i and p[1] >= conf],
                                key=lambda p: -p[1]):
            meilleur, k_best = 0.0, None
            for k, o in enumerate(vis):
                if k in pris:
                    continue
                v = iou(box, o["bbox"])
                if v > meilleur:
                    meilleur, k_best = v, k
            if k_best is not None and meilleur >= IOU_OK:
                pris.add(k_best)
                trouves[id(vis[k_best])] = s
            elif any(iou(box, o["bbox"]) >= IOU_IGNORE for o in verites):
                continue
            else:
                faux.append((classe, s, box))
    return trouves, faux


def bin_de(d: float) -> int:
    for i in range(len(BINS_DISTANCE) - 1):
        if BINS_DISTANCE[i] <= d < BINS_DISTANCE[i + 1]:
            return i
    return len(BINS_DISTANCE) - 2


def courbe_par_distance(compte) -> list[dict]:
    out = []
    for i in range(len(BINS_DISTANCE) - 1):
        n, ok = compte.get(i, (0, 0))
        out.append({"min": BINS_DISTANCE[i], "max": BINS_DISTANCE[i + 1], "n": n,
                    "taux": round(ok / n, 4) if n else None})
    return out


def portee(courbe: list[dict], seuil: float = PORTEE_TAUX):
    """Dernière distance à laquelle le taux tient encore à `seuil`, en partant du plus près."""
    p = None
    for c in courbe:
        if c["taux"] is None or c["n"] < 8:
            continue
        if c["taux"] >= seuil:
            p = c["max"]
        else:
            break
    return p


def juge_jeu(modele, nom: str, imgsz: int) -> dict:
    """Rappel par classe et par distance, fausses alertes par image, aux deux seuils."""
    ms = charge(nom)
    if not ms:
        return {}
    preds = predit(modele, JEU / nom, [m["image"] for m in ms], imgsz, min(SEUILS_CONF))
    bilan = {"images": len(ms)}
    for conf in SEUILS_CONF:
        compte = {c: {} for c in CLASSES}
        n_vis = {c: 0 for c in CLASSES}
        n_trouves = {c: 0 for c in CLASSES}
        n_faux = {c: 0 for c in CLASSES}
        images_avec_faux = 0
        for m in ms:
            trouves, faux = apparie(preds.get(m["image"], []), m["objets"], conf)
            for o in m["objets"]:
                if o["ignore"]:
                    continue
                c = o["classe"]
                n_vis[c] += 1
                ok = id(o) in trouves
                n_trouves[c] += int(ok)
                b = bin_de(o["distance"])
                n, k = compte[c].get(b, (0, 0))
                compte[c][b] = (n + 1, k + int(ok))
            for c, _, _ in faux:
                n_faux[c] += 1
            images_avec_faux += int(bool(faux))
        bilan[f"conf_{conf}"] = {
            c: {"visibles": n_vis[c], "trouves": n_trouves[c],
                "rappel": round(n_trouves[c] / n_vis[c], 4) if n_vis[c] else None,
                "faux": n_faux[c], "faux_par_image": round(n_faux[c] / len(ms), 4),
                "par_distance": courbe_par_distance(compte[c]),
                "portee_m": portee(courbe_par_distance(compte[c]))}
            for c in CLASSES}
        bilan[f"conf_{conf}"]["images_avec_une_fausse_alerte"] = round(images_avec_faux / len(ms), 4)
    return bilan


def juge_panneau_vise(modele, imgsz: int) -> dict:
    """Sur les images de l'étape 2, le panneau visé est-il repéré ? Même définition et mêmes
    images que le repérage classique, donc même courbe et même portée comparables."""
    out = {}
    for nom in ("optique", "9019"):
        ms = charge(f"etape2_{nom}")
        if not ms:
            continue
        preds = predit(modele, JEU / f"etape2_{nom}", [m["image"] for m in ms], imgsz, min(SEUILS_CONF))
        res = {"images": len(ms)}
        for conf in SEUILS_CONF:
            compte, compte_app, n_caches = {}, {}, 0
            for m in ms:
                cible = [o for o in m["objets"] if o["classe"] == "qr" and o.get("code") == m["cible"]]
                if not cible or cible[0]["ignore"]:
                    n_caches += 1
                    continue
                o = cible[0]
                ok = any(p[0] == 0 and p[1] >= conf and iou(p[2], o["bbox"]) >= IOU_OK
                         for p in preds.get(m["image"], []))
                for cle, d in ((compte, m["D_m"]),
                               (compte_app, m["D_m"] / max(np.cos(np.radians(m["alpha_deg"])), 0.2))):
                    b = bin_de(d)
                    n, k = cle.get(b, (0, 0))
                    cle[b] = (n + 1, k + int(ok))
            res[f"conf_{conf}"] = {"par_distance": courbe_par_distance(compte),
                                   "par_distance_apparente": courbe_par_distance(compte_app),
                                   "portee_m": portee(courbe_par_distance(compte)),
                                   "portee_apparente_m": portee(courbe_par_distance(compte_app)),
                                   "panneaux_caches_exclus": n_caches}
        out[nom] = res
    return out


def repere_classique() -> dict:
    """La courbe du repérage classique, mesurée à l'étape 2 sur les mêmes images."""
    r = json.loads((ETAPE2 / "resultats.json").read_text())
    out = {}
    for nom, bloc in (("optique", r["optique"]["lecteurs"]["zxing"]), ("9019", r.get("second_entrepot", {}))):
        courbes = {}
        for cle in ("courbe_distance", "courbe_apparente"):
            if cle not in bloc:
                continue
            pts = []
            for c in bloc[cle]:
                p_rep = next((c[k] for k in c if k.startswith("p_rep")), None)
                pts.append({"min": c["min"], "max": c["max"], "n": c["n"],
                            "taux": p_rep, "taux_lu": c.get("p_lu")})
            courbes[cle] = pts
        out[nom] = {**courbes,
                    "portee_reperage_m": portee(courbes.get("courbe_distance", [])),
                    "portee_reperage_apparente_m": portee(courbes.get("courbe_apparente", [])),
                    "portee_lecture_apparente_m": bloc.get("portee_apparente_m")}
    out["faux_reperage_sans_qr"] = r["sans_qr"]["faux_reperage"]["taux"]
    return out


def juge_sans_qr(modele, imgsz: int) -> dict:
    ms = charge("etape2_sans_qr")
    if not ms:
        return {}
    preds = predit(modele, JEU / "etape2_sans_qr", [m["image"] for m in ms], imgsz, min(SEUILS_CONF))
    out = {"images": len(ms)}
    for conf in SEUILS_CONF:
        alerte = sum(1 for m in ms if any(p[0] == 0 and p[1] >= conf for p in preds.get(m["image"], [])))
        out[f"conf_{conf}"] = {"images_avec_un_qr_invente": alerte,
                               "taux": round(alerte / len(ms), 4)}
    return out


def vitesse(modele, imgsz: int) -> dict:
    import torch

    ms = charge("rendu_9033")[:220]
    chemins = [str(JEU / "rendu_9033" / "images" / m["image"]) for m in ms]
    for c in chemins[:20]:
        modele.predict(c, imgsz=imgsz, half=True, device=0, verbose=False)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for c in chemins[20:]:
        modele.predict(c, imgsz=imgsz, half=True, device=0, verbose=False)
    par_image = (time.perf_counter() - t0) / max(len(chemins) - 20, 1) * 1000
    return {"ms_par_image": round(par_image, 1),
            "memoire_gpu_mo": round(torch.cuda.max_memory_allocated() / 1e6)}


def seuil_mission(b: dict) -> float:
    """Le seuil le plus bas qui garde les fausses alertes sous 1 % des images sans QR."""
    return next((c for c in SEUILS_CONF if b["sans_qr"][f"conf_{c}"]["taux"] <= 0.01), max(SEUILS_CONF))


def choisit(bilans: list[dict]) -> str:
    """Au seuil de mission de chacune : la variante qui trouve le plus de QR sur les entrepôts
    scellés, parmi celles dont la portée tient à 25 cm de la meilleure et qui ne coûtent pas
    plus d'une fois et demie le temps de la plus rapide. Une demi-milliseconde ne vaut pas
    quatre points de rappel."""
    def portee_de(b):
        return b["panneau_vise"]["optique"][f"conf_{seuil_mission(b)}"]["portee_m"] or 0.0

    def rappel_de(b):
        return b["scelles"][f"conf_{seuil_mission(b)}"]["qr"]["rappel"] or 0.0

    meilleure_portee = max(portee_de(b) for b in bilans)
    plus_rapide = min(b["ms_par_image"] for b in bilans)
    ok = [b for b in bilans if portee_de(b) >= meilleure_portee - 0.25 and b["ms_par_image"] <= 1.5 * plus_rapide]
    return max(ok or bilans, key=lambda b: (rappel_de(b), -b["ms_par_image"]))["variante"]


def figure(bilans: list[dict], classique: dict, retenue: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for b in bilans:
        c = b["panneau_vise"]["optique"][f"conf_{seuil_mission(b)}"]["par_distance"]
        x = [(p["min"] + p["max"]) / 2 for p in c if p["taux"] is not None]
        y = [p["taux"] for p in c if p["taux"] is not None]
        axes[0].plot(x, y, marker="o", lw=2.2 if b["variante"] == retenue else 1.0,
                     label=f"appris {b['variante']}" + (" (retenu)" if b["variante"] == retenue else ""))
    cd = classique["optique"].get("courbe_distance", [])
    axes[0].plot([(p["min"] + p["max"]) / 2 for p in cd if p["taux"] is not None],
                 [p["taux"] for p in cd if p["taux"] is not None], "k--", marker="s", label="repérage classique (étape 2)")
    axes[0].plot([(p["min"] + p["max"]) / 2 for p in cd if p["taux_lu"] is not None],
                 [p["taux_lu"] for p in cd if p["taux_lu"] is not None], "k:", label="lecture classique (étape 2)")
    axes[0].axhline(PORTEE_TAUX, color="gray", lw=0.8)
    axes[0].set(xlabel="distance au panneau visé (m)", ylabel="part des panneaux repérés",
                title="Panneau visé repéré, mêmes 2 000 images (seuil de mission)", ylim=(0, 1.02))
    axes[0].legend(fontsize=8)
    for b in bilans:
        for classe, style in (("qr", "-"), ("carton", "--")):
            c = b["scelles"][f"conf_{seuil_mission(b)}"][classe]["par_distance"]
            x = [(p["min"] + p["max"]) / 2 for p in c if p["taux"] is not None]
            y = [p["taux"] for p in c if p["taux"] is not None]
            axes[1].plot(x, y, style, marker="o", lw=2.2 if b["variante"] == retenue else 1.0,
                         label=f"{classe} {b['variante']}")
    axes[1].set(xlabel="distance à l'objet (m)", ylabel="rappel (objets visibles trouvés)",
                title="Entrepôts scellés 9033 et 9019, tous les objets visibles", ylim=(0, 1.02))
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(HERE / "portee.png", dpi=130)


def main() -> None:
    os.environ["YOLO_AUTOINSTALL"] = "false"
    from ultralytics import YOLO

    entrainement = json.loads((HERE / "entrainement.json").read_text())
    classique = repere_classique()
    bilans = []
    for e in entrainement:
        modele = YOLO(e["poids"])
        print(f"\n=== {e['variante']} ({e['parametres'] / 1e6:.1f} M parametres, {e['imgsz']} px) ===")
        b = {"variante": e["variante"], "imgsz": e["imgsz"], "parametres": e["parametres"],
             "map50_val": e["map50"], "par_classe_val": e["par_classe"]}
        scelles = {}
        for nom in ("rendu_9033", "rendu_9019"):
            scelles[nom] = juge_jeu(modele, nom, e["imgsz"])
        # fusion des deux entrepôts scellés
        fusion = {"images": sum(s["images"] for s in scelles.values())}
        for conf in SEUILS_CONF:
            fusion[f"conf_{conf}"] = {}
            for c in CLASSES:
                compte = {}
                vis = tr = faux = 0
                for s in scelles.values():
                    bloc = s[f"conf_{conf}"][c]
                    vis += bloc["visibles"]; tr += bloc["trouves"]; faux += bloc["faux"]
                    for i, p in enumerate(bloc["par_distance"]):
                        n, k = compte.get(i, (0, 0))
                        compte[i] = (n + p["n"], k + round((p["taux"] or 0) * p["n"]))
                courbe = courbe_par_distance(compte)
                fusion[f"conf_{conf}"][c] = {"visibles": vis, "trouves": tr,
                                             "rappel": round(tr / vis, 4) if vis else None,
                                             "faux": faux, "faux_par_image": round(faux / fusion["images"], 4),
                                             "par_distance": courbe, "portee_m": portee(courbe)}
        b["scelles"] = fusion
        b["scelles_par_entrepot"] = scelles
        b["panneau_vise"] = juge_panneau_vise(modele, e["imgsz"])
        b["sans_qr"] = juge_sans_qr(modele, e["imgsz"])
        b.update(vitesse(modele, e["imgsz"]))
        pv = b["panneau_vise"]["optique"]["conf_0.25"]
        print(f"  portee QR (panneau vise) : {pv['portee_m']} m ; apparente {pv['portee_apparente_m']} m")
        print(f"  scelles : rappel qr {b['scelles']['conf_0.25']['qr']['rappel']}, "
              f"carton {b['scelles']['conf_0.25']['carton']['rappel']}, "
              f"faux qr/image {b['scelles']['conf_0.25']['qr']['faux_par_image']}")
        print(f"  sans QR : {b['sans_qr']['conf_0.25']['taux']:.1%} d'images avec un QR invente "
              f"(classique {classique['faux_reperage_sans_qr']:.1%})")
        print(f"  {b['ms_par_image']} ms/image, {b['memoire_gpu_mo']} Mo")
        bilans.append(b)

    retenue = choisit(bilans)
    b = next(b for b in bilans if b["variante"] == retenue)
    # seuil de mission : le plus bas qui garde les fausses alertes sous 1 % des images sans QR
    conf_mission = seuil_mission(b)
    poids = next(e["poids"] for e in entrainement if e["variante"] == retenue)
    resultats = {"classique": classique, "variantes": bilans, "retenue": retenue,
                 "poids_retenus": poids, "conf_mission": conf_mission}
    (HERE / "resultats.json").write_text(json.dumps(resultats, indent=2))
    # les poids retenus deviennent ceux du système (swarm_qr/detecteur.py les charge par défaut)
    import shutil

    from swarm_qr.detecteur import ASSETS, POIDS, REGLAGES

    ASSETS.mkdir(parents=True, exist_ok=True)
    shutil.copy(poids, POIDS)
    REGLAGES.write_text(json.dumps({"variante": retenue, "imgsz": b["imgsz"], "conf": conf_mission,
                                    "classes": list(CLASSES)}, indent=2))
    print(f"poids copies dans {POIDS} (seuil de mission {conf_mission})")
    figure(bilans, classique, retenue)
    print(f"\nvariante retenue : {retenue}")
    print("BANC FINI")


if __name__ == "__main__":
    main()
