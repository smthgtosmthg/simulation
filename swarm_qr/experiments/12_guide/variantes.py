"""Faire choisir la bonne zone à un modèle vision-langage : une variante par cause possible.

Le banc de base (`banc.py`) a montré un modèle proche du hasard. Ce fichier cherche pourquoi,
en changeant une seule chose à la fois, sur les mêmes cas et avec les mêmes références :

  comptage      contrôle de perception : combien de zones vois-tu ? (rien à voir avec la tâche)
  image         la question d'origine, images seules
  lisible       même question, plan redessiné pour un modèle (gros repères, contraste)
  description   plan lisible + les faits de la carte en phrases
  raisonnement  plan lisible + faits + une phrase de raisonnement avant la réponse
  notes         on ne demande plus de choisir : chaque zone est notée de 0 à 10, on prend la meilleure

Les numéros de zone sont MÉLANGÉS : dans le banc d'origine ils suivaient l'ordre d'utilité
géométrique, donc répondre « 1 » valait la géométrie sans rien comprendre.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from swarm_qr import mapping                                            # noqa: E402
from swarm_qr.guide import Guide, decrit                                # noqa: E402

sys.path.insert(0, str(HERE))
from banc import instantanes, references                               # noqa: E402

CONTEXTE = (
    "You help a team of drones that must read QR codes glued on cardboard boxes in a warehouse. "
    "The first image is the side camera of one drone. The second image is the map built so far, "
    "seen from above: dark grey is a shelf or a wall, medium grey is unknown space, light grey is "
    "free space, white has already been inspected, orange dots are boxes seen but not read yet. "
    "Numbered discs are the candidate zones the drone can fly to. "
)
BUT = ("The drone must fly to the zone where it will read the largest number of QR codes that are "
       "still unknown. ")
FORMAT_NOMBRE = "Answer with the zone number only, nothing else."
FORMAT_RAISON = ("First write one short sentence of reasoning, then a new line with "
                 "'ANSWER: <zone number>'.")

COULEUR_ZONE = (40, 40, 220)
COULEUR_TEXTE = (255, 255, 255)


# --------------------------------------------------------------------- le plan

def _pixel(centre_xy, forme, echelle):
    """Monde vers pixel, avec la convention de `mission.vue_annotee`."""
    i = int(np.floor((centre_xy[0] - mapping.MAP.x_min) / mapping.MAP.cell))
    j = int(np.floor((centre_xy[1] - mapping.MAP.y_min) / mapping.MAP.cell))
    return int((i + 0.5) * echelle), int((forme[1] - j - 0.5) * echelle)


def plan_lisible(vue, zones, position=None, recadre: bool = True) -> np.ndarray:
    """Le plan redessiné pour un modèle : disques pleins, gros chiffres blancs, recadrage sur la
    zone utile. Le contrôle de comptage a montré que les cercles fins de 20 pixels avec un
    chiffre de 8 pixels ne sont lus correctement qu'une fois sur trois."""
    img = vue.copy()
    forme = (mapping.MAP.shape[0], mapping.MAP.shape[1])
    echelle = img.shape[1] / forme[0]
    points = []
    for z in zones:
        c = _pixel(z["centre"], forme, echelle)
        points.append(c)
        cv2.circle(img, c, 22, COULEUR_ZONE, -1)
        cv2.circle(img, c, 22, (255, 255, 255), 2)
        texte = str(z["numero"])
        (w, h), _ = cv2.getTextSize(texte, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
        cv2.putText(img, texte, (c[0] - w // 2, c[1] + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    COULEUR_TEXTE, 3, cv2.LINE_AA)
    if position is not None:
        d = _pixel(position, forme, echelle)
        cv2.drawMarker(img, d, (255, 200, 0), cv2.MARKER_TRIANGLE_UP, 26, 3)
        points.append(d)
    if recadre and points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        marge = 60
        x0, x1 = max(min(xs) - marge, 0), min(max(xs) + marge, img.shape[1])
        y0, y1 = max(min(ys) - marge, 0), min(max(ys) + marge, img.shape[0])
        if x1 - x0 > 40 and y1 - y0 > 40:
            img = img[y0:y1, x0:x1]
    h, w = img.shape[:2]
    k = min(1000 / max(h, w), 2.5)
    if k > 1.05:
        img = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_NEAREST)
    return img


# --------------------------------------------------------------- les variantes

def melange(zones: list[dict], verite: dict, graine: int):
    """Renumérote les zones au hasard : le numéro ne doit rien dire de leur qualité."""
    numeros = [z["numero"] for z in zones]
    tire = numeros[:]
    random.Random(graine).shuffle(tire)
    table = dict(zip(numeros, tire))
    neuves = [dict(z, numero=table[z["numero"]]) for z in zones]
    neuves.sort(key=lambda z: z["numero"])
    return neuves, table.get(verite["zone"]), table.get(zones[0]["numero"])


def _reponse_zone(texte: str, zones: list[dict]) -> int | None:
    m = re.search(r"ANSWER\s*[:=]?\s*(\d+)", texte, re.IGNORECASE)
    if m:
        n = int(m.group(1))
        return n if n in {z["numero"] for z in zones} else None
    return Guide.zone_dans(texte, zones)


def _note(texte: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)", texte)
    return float(m.group(1)) if m else None


def demande_choix(g: Guide, cam, vue, zones, question) -> tuple[int | None, str]:
    r = g.repond(cam, vue, question)
    return _reponse_zone(r, zones), r


def variante_comptage(g, cas, cam, vue, zones, *_):
    q = (CONTEXTE + "How many numbered discs are on the second image? " + FORMAT_NOMBRE)
    r = g.repond(cam, plan_lisible(vue, zones, cas.get("position")), q)
    n = _note(r)
    return (int(n) if n is not None else None), r


def variante_comptage_origine(g, cas, cam, vue, zones, *_):
    q = (CONTEXTE + "How many numbered circles are on the second image? " + FORMAT_NOMBRE)
    r = g.repond(cam, vue, q)
    n = _note(r)
    return (int(n) if n is not None else None), r


def variante_image(g, cas, cam, vue, zones, *_):
    q = CONTEXTE + BUT + "Which zone should the drone go to next? " + FORMAT_NOMBRE
    return demande_choix(g, cam, vue, zones, q)


def variante_lisible(g, cas, cam, vue, zones, *_):
    q = CONTEXTE + BUT + "Which zone should the drone go to next? " + FORMAT_NOMBRE
    return demande_choix(g, cam, plan_lisible(vue, zones, cas.get("position")), zones, q)


def variante_description(g, cas, cam, vue, zones, *_):
    faits = decrit(zones, cas.get("position"), cas.get("codes_lus"))
    q = CONTEXTE + BUT + faits + "Which zone should the drone go to next? " + FORMAT_NOMBRE
    return demande_choix(g, cam, plan_lisible(vue, zones, cas.get("position")), zones, q)


def variante_raisonnement(g, cas, cam, vue, zones, *_):
    faits = decrit(zones, cas.get("position"), cas.get("codes_lus"))
    q = (CONTEXTE + BUT + faits
         + "A zone is good when it holds many boxes whose code is still unknown, and when it is "
           "not too far. Which zone should the drone go to next? " + FORMAT_RAISON)
    return demande_choix(g, cam, plan_lisible(vue, zones, cas.get("position")), zones, q)


def variante_notes(g, cas, cam, vue, zones, *_):
    """Juger vaut mieux que choisir : une zone à la fois, une note de 0 à 10."""
    plan = plan_lisible(vue, zones, cas.get("position"))
    faits = decrit(zones, cas.get("position"), cas.get("codes_lus"))
    notes, brut = {}, []
    for z in zones:
        q = (CONTEXTE + BUT + faits
             + f"Rate zone {z['numero']} only, from 0 to 10, on how many unknown QR codes the "
               f"drone would read there. Answer with the number only.")
        r = g.repond(cam, plan, q)
        notes[z["numero"]] = _note(r) if _note(r) is not None else -1.0
        brut.append(f"{z['numero']}:{r.strip()[:6]}")
    meilleure = max(notes, key=notes.get) if notes else None
    if meilleure is not None and notes[meilleure] < 0:
        meilleure = None
    return meilleure, " ".join(brut)


VARIANTES = {
    "comptage_origine": variante_comptage_origine,
    "comptage": variante_comptage,
    "image": variante_image,
    "lisible": variante_lisible,
    "description": variante_description,
    "raisonnement": variante_raisonnement,
    "notes": variante_notes,
}
PERCEPTION = {"comptage", "comptage_origine"}


def juge(nom_variante: str, g: Guide, cas: list[dict]) -> dict:
    fonction = VARIANTES[nom_variante]
    justes = repondus = 0
    comme_geometrie = 0
    details, latences = [], []
    for k, c in enumerate(cas):
        camera, vue = cv2.imread(c["cam"]), cv2.imread(c["vue"])
        zones, bonne, geometrique = melange(c["zones"], c["verite"], graine=k)
        t0 = time.perf_counter()
        reponse, brut = fonction(g, c, camera, vue, zones)
        latences.append(time.perf_counter() - t0)
        if nom_variante in PERCEPTION:
            bonne = len(zones)
        if reponse is not None:
            repondus += 1
            justes += int(reponse == bonne)
            comme_geometrie += int(reponse == geometrique)
        details.append({"mission": c["mission"], "k": c["k"], "drone": c.get("drone"),
                        "reponse": reponse, "bonne": bonne, "geometrie": geometrique,
                        "brut": brut[:120]})
    n = len(cas)
    return {"variante": nom_variante, "cas": n, "repondus": repondus,
            "justes": round(justes / n, 4), "accord_avec_la_geometrie": round(comme_geometrie / max(repondus, 1), 4),
            "latence_mediane_s": round(float(np.median(latences)), 2),
            "details": details}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--missions", nargs="+", required=True)
    ap.add_argument("--modele", required=True)
    ap.add_argument("--variantes", nargs="+", default=["image", "lisible", "description"])
    ap.add_argument("--quant", default="4bit")
    ap.add_argument("--cas", type=int, default=0, help="limiter le nombre de cas (mise au point)")
    ap.add_argument("--sortie", default="resultats_variantes.json")
    a = ap.parse_args()

    cas = instantanes([Path(m) for m in a.missions])
    if a.cas:
        cas = cas[:a.cas]
    refs = references(cas)
    print(f"{len(cas)} cas ; hasard {refs['hasard_zone']:.0%}, geometrie {refs['zone_la_plus_utile_geometrie']:.0%}")
    g = Guide(a.modele, max_tokens=60, quantisation=a.quant)
    bilans = []
    for nom in a.variantes:
        b = juge(nom, g, cas)
        bilans.append(b)
        cible = "zones vues" if nom in PERCEPTION else "bonne zone"
        print(f"  {nom:18s} {b['justes']:.0%} de {cible} ; repondus {b['repondus']}/{b['cas']} ; "
              f"accord avec la geometrie {b['accord_avec_la_geometrie']:.0%} ; {b['latence_mediane_s']} s")
        (HERE / a.sortie).write_text(json.dumps({"cas": len(cas), "references": refs,
                                                 "modele": a.modele, "variantes": bilans}, indent=1))
    print("VARIANTES FINI")


if __name__ == "__main__":
    main()
