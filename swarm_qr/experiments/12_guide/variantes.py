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
from swarm_qr.guide import Guide, decrit, boussole, dossier_zones, epure   # noqa: E402

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

CONTEXTE_CAMERA = (
    "You help a team of drones that must read QR codes glued on cardboard boxes in a warehouse. "
    "The image is the side camera of one drone. The warehouse is divided into candidate zones the "
    "drone can fly to; you get their description in words below. "
)
VOTES = 5
CIBLES_MAX = 25
FRONTIERES_MAX = 20
COULEUR_ZONE = (40, 40, 220)
COULEUR_TEXTE = (255, 255, 255)


# --------------------------------------------------------------------- le plan

def _pixel(centre_xy, forme, echelle):
    """Monde vers pixel, avec la convention de `mission.vue_annotee`."""
    i = int(np.floor((centre_xy[0] - mapping.MAP.x_min) / mapping.MAP.cell))
    j = int(np.floor((centre_xy[1] - mapping.MAP.y_min) / mapping.MAP.cell))
    return int((i + 0.5) * echelle), int((forme[1] - j - 0.5) * echelle)


def plan_lisible(vue, zones, position=None, recadre: bool = True, retour_positions: bool = False):
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
    x0 = y0 = 0
    if recadre and points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        marge = 60
        x0, x1 = max(min(xs) - marge, 0), min(max(xs) + marge, img.shape[1])
        y0, y1 = max(min(ys) - marge, 0), min(max(ys) + marge, img.shape[0])
        if x1 - x0 > 40 and y1 - y0 > 40:
            img = img[y0:y1, x0:x1]
        else:
            x0 = y0 = 0
    h, w = img.shape[:2]
    k = min(1000 / max(h, w), 2.5)
    if k > 1.05:
        img = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_NEAREST)
    if not retour_positions:
        return img
    dx, dy = (x0, y0) if recadre and points else (0, 0)
    repere = {z["numero"]: (int((p[0] - dx) * k), int((p[1] - dy) * k))
              for z, p in zip(zones, points[:len(zones)])}
    return img, repere


def positions_plan_origine(vue, zones) -> dict:
    """Les pixels des repères sur le plan tel qu'il est enregistré, sans redessin."""
    forme = (mapping.MAP.shape[0], mapping.MAP.shape[1])
    echelle = vue.shape[1] / forme[0]
    return {z["numero"]: _pixel(z["centre"], forme, echelle) for z in zones}


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


EXEMPLES = []          # rempli par main() : des cas resolus, jamais ceux qu'on evalue

INDICE = ("Useful knowledge: a QR code already spotted but not read is almost a guaranteed "
          "reading, because the detector has already seen it and the drone only has to come "
          "closer. A shelf section never looked at is only a hope: there may be a box, or not. ")


def dossier(cas, zones, avec_note: bool, niveau: str = "zones") -> dict:
    """Le dossier des zones vient de guide.py (le même qu'en vol) ; les niveaux « listes » et
    « tout » ajoutent les listes brutes du vol « dossier complet »."""
    moi = cas.get("position") or [0.0, 0.0, 0.0]
    autres = [d for d in cas.get("coequipiers", []) if d["i"] != cas.get("drone")]
    out = dossier_zones(cas, zones, avec_note)
    if niveau in ("listes", "tout"):
        # ce que le vol « dossier complet » enregistre en plus : les listes brutes de la carte,
        # en lignes compactes — en JSON verbeux elles font 14 000 jetons, et l'attention du modèle
        # sur cette carte graphique ne tient pas au-delà de ~5 000
        cible = lambda c: (None if not c else f"{c['genre']} at ({c['origine'][0]},{c['origine'][1]},"
                                              f"{c['origine'][2]}) score {c['note']} since {c['depuis']} s")
        out["map_summary"] = cas.get("resume")
        out["this_drone_current_target"] = next((f"{d.get('phase')}: {cible(d.get('cible'))}"
                                                 for d in cas.get("coequipiers", []) if d["i"] == cas.get("drone")), None)
        out["teammates_current_targets"] = [f"drone {d['i']} {d.get('phase')}: {cible(d.get('cible'))}" for d in autres]
        d1 = lambda v: f"{v:.1f}".rstrip("0").rstrip(".")
        out["qr_codes_already_read_positions_x_y_z"] = "; ".join(
            " ".join(d1(v) for v in q["position"]) for q in cas.get("panneaux", []))
        out["qr_codes_spotted_not_read_x_y_z_seen"] = "; ".join(
            " ".join(d1(v) for v in q["position"]) + f" seen {q['vues']}x" for q in cas.get("pistes", []))
        fr = cas.get("frontieres") or {}
        out["frontier"] = {"cells": fr.get("cases"), "examples_x_y_z": "; ".join(
            " ".join(d1(v) for v in f) for f in (fr.get("exemples") or [])[:FRONTIERES_MAX])}
        out["reservations"] = [f"drone {r['drone']} at ({r['cible'][0]},{r['cible'][1]}) until {r['jusqu_a']} s"
                               for r in cas.get("reservations", [])]
    if niveau == "tout":
        cibles = sorted(cas.get("cibles", []), key=lambda c: -c["utilite"])[:CIBLES_MAX]
        out["planner_candidate_targets"] = {
            "note": f"the {len(cibles)} most useful of {len(cas.get('cibles', []))} candidate targets, "
                    "as kind, looked-at point x y z, side, utility",
            "targets": "; ".join(f"{c['genre']} " + " ".join(f"{v:.1f}" for v in c['origine'])
                                 + f" side{c['cote']} u{c['utilite']:.0f}" for c in cibles)}
    return out


def en_phrases(dos: dict) -> str:
    """Le même dossier, mais tout en phrases : chaque nombre écrit dans une phrase complète."""
    m = dos["this_drone"]
    lignes = [f"It is {dos['mission_time_s']} seconds into the mission and {dos['qr_codes_read_so_far']} "
              f"QR codes have been read so far.",
              f"The drone you advise is number {m['id']}. It is at x = {m['position_xyz_m'][0]} m, "
              f"y = {m['position_xyz_m'][1]} m, altitude {m['position_xyz_m'][2]} m, heading "
              f"{m['heading_deg']} degrees. The image is one of its two side cameras."]
    for t in dos["teammates"]:
        lignes.append(f"Teammate {t['id']} is at x = {t['position_xyz_m'][0]} m, y = "
                      f"{t['position_xyz_m'][1]} m, altitude {t['position_xyz_m'][2]} m, "
                      f"{t['distance_from_this_drone_m']} m away from your drone.")
    for r in dos["shelves_mapped_so_far"]:
        lignes.append(f"Shelf {r['name']} occupies x from {r['x_range_m'][0]} to {r['x_range_m'][1]} m "
                      f"and y from {r['y_range_m'][0]} to {r['y_range_m'][1]} m.")
    for z in dos["candidate_zones"]:
        phrase = (f"Zone {z['number']} is centred at x = {z['centre_xy_m'][0]} m, y = "
                  f"{z['centre_xy_m'][1]} m, with a radius of {z['radius_m']} m. It holds "
                  f"{z['qr_codes_spotted_but_not_read']} QR code(s) already spotted by the detector "
                  f"but not read yet, {z['shelf_sections_with_a_box_seen_but_face_never_looked_at']} "
                  f"shelf section(s) where a box was seen but the face was never looked at from the "
                  f"right side, {z['surfaces_never_looked_at_total']} surface(s) never looked at in "
                  f"total, and {z['unexplored_frontier_groups']} unexplored frontier group(s), for "
                  f"{z['total_candidate_targets']} candidate targets of kinds "
                  f"{', '.join(z['target_kinds'])}. The shelf faces there look "
                  f"{z['shelf_faces_look_towards']}. It is {z['distance_from_this_drone_m']} m from "
                  f"your drone, to the {z['direction_from_this_drone']}, and "
                  f"{z['distance_from_nearest_teammate_m']} m from the nearest teammate.")
        if "geometric_score_of_our_planner" in z:
            phrase += f" Our geometric planner scores it {z['geometric_score_of_our_planner']}."
        lignes.append(phrase)
    return " ".join(lignes) + " "


def _dossier_complet(g, cas, cam, zones, forme: str, avec_note: bool, avec_image: bool,
                     niveau: str = "zones"):
    dos = dossier(cas, zones, avec_note, niveau)
    corps = en_phrases(dos) if forme == "phrases" else ("Here is the map data as JSON:\n"
                                                       + json.dumps(dos, indent=1) + "\n")
    tete = CONTEXTE_CAMERA if avec_image else (
        "You help a team of drones that must read QR codes glued on cardboard boxes in a warehouse. "
        "You get no image at all, only the data the map holds. ")
    q = (tete + BUT + corps
         + "Which zone should the drone go to next? Answer on two lines: "
           "'ANSWER: <zone number>' then 'WHY: <one short sentence>'.")
    r = g.repond_images([cam] if avec_image else [], q)
    return _reponse_zone(r, zones), r


def _zones_reordonnees(dos: dict, epurer: bool) -> dict:
    return epure(dos, distracteurs=epurer)


def variante_ordre(g, cas, cam, vue, zones, *_):
    dos = _zones_reordonnees(dossier(cas, zones, False), epurer=False)
    q = (CONTEXTE_CAMERA + BUT + "Here is the map data as JSON:\n" + json.dumps(dos, indent=1)
         + "\nWhich zone should the drone go to next? 'ANSWER: <zone number>' then 'WHY: <one sentence>'.")
    r = g.repond_images([cam], q)
    return _reponse_zone(r, zones), r


def variante_epure(g, cas, cam, vue, zones, *_):
    dos = _zones_reordonnees(dossier(cas, zones, False), epurer=True)
    q = (CONTEXTE_CAMERA + BUT + "Here is the map data as JSON:\n" + json.dumps(dos, indent=1)
         + "\nWhich zone should the drone go to next? 'ANSWER: <zone number>' then 'WHY: <one sentence>'.")
    r = g.repond_images([cam], q)
    return _reponse_zone(r, zones), r


def variante_epure_few_shot(g, cas, cam, vue, zones, *_):
    """Le dossier épuré, précédé de deux cas résolus présentés de la même façon."""
    images, texte = [], CONTEXTE_CAMERA + BUT + "Here are two solved examples.\n"
    for k, ex in enumerate(EXEMPLES, 1):
        images.append(ex["cam"])
        texte += (f"\nEXAMPLE {k}, map data as JSON:\n" + json.dumps(ex["dossier_epure"], indent=1)
                  + f"\nCorrect answer: ANSWER: {ex['bonne']}\n")
    dos = _zones_reordonnees(dossier(cas, zones, False), epurer=True)
    images.append(cam)
    texte += ("\nNOW THE REAL CASE, map data as JSON:\n" + json.dumps(dos, indent=1)
              + "\nWhich zone should the drone go to next? 'ANSWER: <zone number>' then 'WHY: <one sentence>'.")
    r = g.repond_images(images, texte)
    return _reponse_zone(r, zones), r


# ---------------------------------------- la règle donnée indirectement, et des noms parlants

METIER = ("Useful knowledge from the field: a QR code already spotted by the detector but not read "
          "yet is an almost guaranteed reading, the drone only has to come closer. A shelf section "
          "where a box was seen but never looked at from the right side is only a possibility. "
          "Every metre flown costs time. ")
NOMS_PARLANTS = {
    "qr_codes_spotted_but_not_read": "almost_certain_readings_if_the_drone_goes_there",
    "shelf_sections_with_a_box_seen_but_face_never_looked_at": "possible_boxes_not_yet_verified",
    "surfaces_never_looked_at_total": "unverified_surfaces_total",
    "distance_from_this_drone_m": "flight_distance_m_costs_time",
}


def _renomme(dos: dict) -> dict:
    zones = [{NOMS_PARLANTS.get(k, k): v for k, v in z.items()} for z in dos["candidate_zones"]]
    return {**dos, "candidate_zones": zones}


def _epure_variante(g, cas, cam, zones, metier: bool, noms: bool):
    dos = _zones_reordonnees(dossier(cas, zones, False), epurer=True)
    if noms:
        dos = _renomme(dos)
    q = (CONTEXTE_CAMERA + BUT + (METIER if metier else "")
         + "Here is the map data as JSON:\n" + json.dumps(dos, indent=1)
         + "\nWhich zone should the drone go to next? 'ANSWER: <zone number>' then 'WHY: <one sentence>'.")
    r = g.repond_images([cam], q)
    return _reponse_zone(r, zones), r


def variante_epure_metier(g, cas, cam, vue, zones, *_):
    return _epure_variante(g, cas, cam, zones, metier=True, noms=False)


def variante_epure_noms(g, cas, cam, vue, zones, *_):
    return _epure_variante(g, cas, cam, zones, metier=False, noms=True)


def variante_epure_noms_metier(g, cas, cam, vue, zones, *_):
    return _epure_variante(g, cas, cam, zones, metier=True, noms=True)


def variante_extraire(g, cas, cam, vue, zones, *_):
    """Deux étapes dans la même réponse : recopier les chiffres clés de chaque zone dans un petit
    tableau, puis décider. Vise la faiblesse mesurée — il lit mal les nombres dans un long texte."""
    dos = dossier(cas, zones, False)
    q = (CONTEXTE_CAMERA + BUT + "Here is the map data as JSON:\n" + json.dumps(dos, indent=1)
         + "\nFirst, for EACH zone, write one line 'zone N: spotted-unread=S, box-sections=B, "
           "distance=D' by copying the numbers from the data. Then decide. End with "
           "'ANSWER: <zone number>'.")
    r = g.repond_images([cam], q)
    return _reponse_zone(r, zones), r


def variante_question_avant(g, cas, cam, vue, zones, *_):
    """La question et le but d'abord, les données ensuite, un rappel court à la fin."""
    dos = dossier(cas, zones, False)
    q = (CONTEXTE_CAMERA + BUT + "QUESTION: which zone should the drone go to next, to read the "
         "largest number of QR codes still unknown? The data follows.\n" + json.dumps(dos, indent=1)
         + "\nNow answer the question above. 'ANSWER: <zone number>' then 'WHY: <one sentence>'.")
    r = g.repond_images([cam], q)
    return _reponse_zone(r, zones), r


def variante_vote(g, cas, cam, vue, zones, *_):
    """Cinq réponses tirées avec un peu de hasard, la majorité l'emporte."""
    dos = dossier(cas, zones, False)
    q = (CONTEXTE_CAMERA + BUT + "Here is the map data as JSON:\n" + json.dumps(dos, indent=1)
         + "\nWhich zone should the drone go to next? 'ANSWER: <zone number>' then 'WHY: <one sentence>'.")
    votes = []
    for _ in range(VOTES):
        votes.append(_reponse_zone(g.repond_images([cam], q, hasard=0.7), zones))
    valides = [v for v in votes if v is not None]
    if not valides:
        return None, str(votes)
    gagnant = max(set(valides), key=valides.count)
    return gagnant, f"votes {votes}"


def variante_dossier_phrases(g, cas, cam, vue, zones, *_):
    return _dossier_complet(g, cas, cam, zones, "phrases", avec_note=False, avec_image=True)


def variante_dossier_json(g, cas, cam, vue, zones, *_):
    return _dossier_complet(g, cas, cam, zones, "json", avec_note=False, avec_image=True)


def variante_dossier_json_note(g, cas, cam, vue, zones, *_):
    return _dossier_complet(g, cas, cam, zones, "json", avec_note=True, avec_image=True)


def variante_dossier_sans_image(g, cas, cam, vue, zones, *_):
    return _dossier_complet(g, cas, cam, zones, "json", avec_note=False, avec_image=False)


def variante_listes(g, cas, cam, vue, zones, *_):
    """Niveau 2 : les zones, plus les listes brutes — codes lus et leurs positions, codes
    repérés non lus, cibles en cours des drones, frontières, réservations, résumé de la grille."""
    return _dossier_complet(g, cas, cam, zones, "json", avec_note=False, avec_image=True, niveau="listes")


def variante_tout(g, cas, cam, vue, zones, *_):
    """Niveau 3 : tout, y compris les centaines de cibles candidates du planificateur."""
    return _dossier_complet(g, cas, cam, zones, "json", avec_note=False, avec_image=True, niveau="tout")


def variante_listes_sans_image(g, cas, cam, vue, zones, *_):
    return _dossier_complet(g, cas, cam, zones, "json", avec_note=False, avec_image=False, niveau="listes")


def variante_texte_seul(g, cas, cam, vue, zones, *_):
    return _texte_seul(g, cas, cam, zones, indice=False)


def variante_texte_melange(g, cas, cam, vue, zones, *_):
    """Meme question, mais les zones sont enumerees dans un ordre tire au hasard au lieu de
    1 a 6 : le modele n'a jamais repondu 1 sur 102 cas alors que 1 etait juste 15 fois. Si le
    refus du premier element est un biais de position, il doit disparaitre ici."""
    ordre = list(zones)
    random.Random(hash((cas["mission"], cas["k"], cas.get("drone"))) & 0xffff).shuffle(ordre)
    return _texte_seul(g, cas, cam, ordre, indice=False, trier=False)


def variante_texte_seul_indice(g, cas, cam, vue, zones, *_):
    return _texte_seul(g, cas, cam, zones, indice=True)


def variante_few_shot(g, cas, cam, vue, zones, *_):
    """Deux cas deja resolus montres avant la vraie question. Le modele ne s'entraine pas, il
    lit des exemples dans la question meme : c'est le test de faisabilite avant tout
    apprentissage. Si cela n'aide pas, un soft prompt ou LoRA n'aideront pas davantage."""
    images, texte = [], CONTEXTE + BUT + "Here are two solved examples.\n"
    for k, ex in enumerate(EXEMPLES, 1):
        images += [ex["cam"], ex["plan"]]
        texte += (f"\nEXAMPLE {k}. The two images above are its camera and its map. "
                  + ex["faits"] + f"Correct answer: ANSWER: {ex['bonne']}\n")
    plan = plan_lisible(vue, zones, cas.get("position"))
    images += [cam, plan]
    texte += ("\nNOW THE REAL CASE. The last two images are its camera and its map. "
              + decrit(zones, cas.get("position"), cas.get("codes_lus"))
              + "Which zone should the drone go to next? " + FORMAT_NOMBRE)
    r = g.repond_images(images, texte)
    return _reponse_zone(r, zones), r


VARIANTES = {
    "comptage_origine": variante_comptage_origine,
    "comptage": variante_comptage,
    "image": variante_image,
    "lisible": variante_lisible,
    "description": variante_description,
    "raisonnement": variante_raisonnement,
    "notes": variante_notes,
    "few_shot": variante_few_shot,
    "texte_seul": variante_texte_seul,
    "texte_seul_indice": variante_texte_seul_indice,
    "texte_melange": variante_texte_melange,
    "dossier_phrases": variante_dossier_phrases,
    "dossier_json": variante_dossier_json,
    "dossier_json_note": variante_dossier_json_note,
    "dossier_sans_image": variante_dossier_sans_image,
    "listes": variante_listes,
    "tout": variante_tout,
    "listes_sans_image": variante_listes_sans_image,
    "extraire": variante_extraire,
    "question_avant": variante_question_avant,
    "vote": variante_vote,
    "ordre": variante_ordre,
    "epure": variante_epure,
    "epure_few_shot": variante_epure_few_shot,
    "epure_metier": variante_epure_metier,
    "epure_noms": variante_epure_noms,
    "epure_noms_metier": variante_epure_noms_metier,
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
                        "brut": brut[:400]})
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
    ap.add_argument("--quant", default="4bit", help="4bit, 8bit, 4bit-vision16, ou 'aucune'")
    ap.add_argument("--device", default="cuda", help="cuda, cpu ou auto (non compresse, reparti)")
    ap.add_argument("--max-tokens", type=int, default=80, dest="max_tokens")
    ap.add_argument("--cas", type=int, default=0, help="limiter le nombre de cas (mise au point)")
    ap.add_argument("--sortie", default="resultats_variantes.json")
    a = ap.parse_args()

    cas = instantanes([Path(m) for m in a.missions])
    if "few_shot" in a.variantes or "epure_few_shot" in a.variantes:
        # les exemples sont pris a la FIN de la liste, jamais parmi les cas evalues
        for k, ex in enumerate(cas[-2:]):
            zones, bonne, _ = melange(ex["zones"], ex["verite"], graine=1000 + k)
            EXEMPLES.append({"cam": cv2.imread(ex["cam"]),
                             "plan": plan_lisible(cv2.imread(ex["vue"]), zones, ex.get("position")),
                             "faits": decrit(zones, ex.get("position"), ex.get("codes_lus")),
                             "dossier_epure": _zones_reordonnees(dossier(ex, zones, False), epurer=True),
                             "bonne": bonne})
        cas = cas[:-2]
        print(f"2 exemples resolus retires du jeu ; reponses montrees : {[e['bonne'] for e in EXEMPLES]}")
    if a.cas:
        cas = cas[:a.cas]
    refs = references(cas)
    print(f"{len(cas)} cas ; hasard {refs['hasard_zone']:.0%}, geometrie {refs['zone_la_plus_utile_geometrie']:.0%}")
    quant = None if a.quant in ("aucune", "none", "") else a.quant
    g = Guide(a.modele, max_tokens=a.max_tokens, quantisation=quant, device=a.device)
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
