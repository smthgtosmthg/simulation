"""Assemble les vidéos d'une mission enregistrée avec `mission.py --video` : une par caméra
fixe, et une composée — les caméras de couloir en mosaïque, la vue d'ensemble et la caméra de
lecture en médaillons, le temps et le compteur de codes en surimpression. À vitesse réelle : le
simulateur rend cinq images par seconde de vol, et la vidéo les tient un cinquième de seconde.

L'encodage passe par ffmpeg en H.264, vingt-cinq fois plus léger que le MPEG-4 d'OpenCV et
lisible partout ; si ffmpeg manque, OpenCV prend le relais.

    video.py --dossier experiments/11_mission/eval_nominal
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

IPS = 25
CRF = 24                     # qualité H.264 : 18 quasi sans perte, 28 visiblement compressé
MOSAIQUE = ["sud_ouest", "sud_central", "nord_central", "sud_grande"]
TITRES = {"sud_ouest": "couloir ouest, depuis le sud", "sud_central": "couloir central, depuis le sud",
          "nord_central": "couloir central, depuis le nord", "sud_grande": "grande zone, depuis le sud",
          "ensemble": "vue d'ensemble", "lecteur": "camera de lecture"}


class Sortie:
    """Une vidéo en écriture. Les images arrivent au rythme du simulateur ; l'encodeur les
    répète pour atteindre 25 images par seconde, sans les recomposer."""

    def __init__(self, chemin: Path, taille: tuple[int, int], ips_entree: float):
        self.chemin, self.taille = chemin, taille
        self.ffmpeg = shutil.which("ffmpeg")
        if self.ffmpeg:
            self.proc = subprocess.Popen(
                [self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                 "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{taille[0]}x{taille[1]}",
                 "-framerate", f"{ips_entree:.6f}", "-i", "-",
                 "-r", str(IPS), "-c:v", "libx264", "-preset", "veryfast", "-crf", str(CRF),
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(chemin)],
                stdin=subprocess.PIPE)
        else:
            self.repetitions = max(1, round(IPS / ips_entree))
            self.writer = cv2.VideoWriter(str(chemin), cv2.VideoWriter_fourcc(*"mp4v"), IPS, taille)

    def ecrit(self, img: np.ndarray) -> None:
        if self.ffmpeg:
            self.proc.stdin.write(np.ascontiguousarray(img).tobytes())
        else:
            for _ in range(self.repetitions):
                self.writer.write(img)

    def ferme(self) -> float:
        if self.ffmpeg:
            self.proc.stdin.close()
            self.proc.wait()
        else:
            self.writer.release()
        return self.chemin.stat().st_size / 1e6 if self.chemin.exists() else 0.0


def _lit(dossier: Path, cam: str, n: int, taille):
    p = dossier / cam / f"{n:05d}.jpg"
    img = cv2.imread(str(p)) if p.exists() else None
    if img is None:
        img = np.zeros((taille[1], taille[0], 3), np.uint8)
    elif (img.shape[1], img.shape[0]) != tuple(taille):
        img = cv2.resize(img, taille, interpolation=cv2.INTER_AREA)
    return img


def _etiquette(img, texte, y=24):
    cv2.putText(img, texte, (9, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, texte, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def par_camera(dossier: Path, index: dict, ips_entree: float) -> None:
    for cam in sorted(p.name for p in dossier.iterdir() if p.is_dir()):
        premiere = next(iter(sorted((dossier / cam).glob("*.jpg"))), None)
        if premiere is None:
            continue
        h, w = cv2.imread(str(premiere)).shape[:2]
        sortie = Sortie(dossier / f"{cam}.mp4", (w, h), ips_entree)
        for im in index["images"]:
            img = _lit(dossier, cam, im["n"], (w, h))
            _etiquette(img, f"{TITRES.get(cam, cam)}   t = {im['t']:.0f} s   codes {im['codes']}/{index['codes_attendus']}")
            sortie.ecrit(img)
        print(f"  {cam}.mp4 : {len(index['images']) / ips_entree:.0f} s, {sortie.ferme():.1f} Mo")


def panneau_info(taille, index: dict, im: dict, entete: dict) -> np.ndarray:
    """La case libre de la mosaïque dit de quelle mission il s'agit : sans elle, une vidéo
    sortie de son dossier ne veut plus rien dire."""
    img = np.full((taille[1], taille[0], 3), 24, np.uint8)
    lignes = [entete.get("titre", ""), entete.get("entrepot", ""), entete.get("cas", ""), "",
              f"temps de vol      {im['t']:.0f} s",
              f"codes lus         {im['codes']} / {index['codes_attendus']}",
              f"drones en vol     {sum(1 for d in im['drones'] if d['vivant'])}"]
    for k, texte in enumerate(lignes):
        gras = k == 0
        cv2.putText(img, texte, (18, 40 + 30 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.62 if gras else 0.52,
                    (255, 255, 255) if gras else (190, 190, 190), 2 if gras else 1, cv2.LINE_AA)
    return img


def composee(dossier: Path, index: dict, ips_entree: float, entete: dict) -> None:
    """Quatre cases : les caméras de couloir présentes, puis la vue d'ensemble, puis la caméra de
    lecture ; ce qui dépasse quatre passe en médaillon, ce qui manque devient le panneau
    d'identité. Le bandeau occupe sa propre bande, au-dessus, pour ne rien cacher."""
    W, H, BANDE = 1280, 720, 30
    q = (W // 2, H // 2)
    med = (320, 180)
    presentes = [c for c in MOSAIQUE if (dossier / c).is_dir()]
    ordre = presentes + [c for c in ("ensemble", "lecteur") if (dossier / c).is_dir()]
    cases, medaillons = ordre[:4], ordre[4:]
    sortie = Sortie(dossier / "mission.mp4", (W, H + BANDE), ips_entree)
    for im in index["images"]:
        toile = np.zeros((H + BANDE, W, 3), np.uint8)
        for k in range(4):
            x, y = (k % 2) * q[0], BANDE + (k // 2) * q[1]
            if k < len(cases):
                cam = cases[k]
                titre = TITRES[cam] + (f" (drone {im['lecteur']})" if cam == "lecteur" and im.get("lecteur") is not None else "")
                case = _etiquette(_lit(dossier, cam, im["n"], q), titre)
            else:
                case = panneau_info(q, index, im, entete)
            toile[y:y + q[1], x:x + q[0]] = case
        for k, cam in enumerate(medaillons):
            x, y = W - (k + 1) * (med[0] + 8), H + BANDE - med[1] - 8
            vignette = _etiquette(_lit(dossier, cam, im["n"], med), TITRES[cam], y=20)
            cv2.rectangle(vignette, (0, 0), (med[0] - 1, med[1] - 1), (255, 255, 255), 2)
            toile[y:y + med[1], x:x + med[0]] = vignette
        bandeau = (f"{entete.get('titre', '')}   |   t = {im['t']:5.1f} s   |   codes lus "
                   f"{im['codes']}/{index['codes_attendus']}   |   drones en vol "
                   f"{sum(1 for d in im['drones'] if d['vivant'])}")
        cv2.putText(toile, bandeau, (14, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        sortie.ecrit(toile)
    print(f"  mission.mp4 : {len(index['images']) / ips_entree:.0f} s, {W}x{H + BANDE}, {sortie.ferme():.1f} Mo")


NOMS = {"geometrie": "systeme : geometrie seule", "zigzag": "reference : balayage fixe (Pore et al.)",
        "glouton": "reference : glouton omniscient"}


def entete(dossier: Path) -> dict:
    """L'identité de la mission, lue dans son journal."""
    fichier = dossier / "mission.json"
    if not fichier.exists():
        return {}
    m = json.loads(fichier.read_text())
    cas = "nominal"
    if m.get("panne"):
        cas = f"panne du drone {m['panne'].split(':')[0]}"
    elif m.get("obstacle"):
        cas = f"obstacle pose a {m['obstacle']['t']:.0f} s"
    return {"titre": NOMS.get(m.get("politique"), m.get("politique", "")),
            "entrepot": f"entrepot {m.get('seed')}", "cas": f"cas : {cas}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dossier", required=True, help="le dossier de sortie de la mission")
    ap.add_argument("--sans-images", action="store_true", dest="sans_images",
                    help="efface les images une fois les vidéos écrites (le disque est petit)")
    a = ap.parse_args()
    dossier = Path(a.dossier) / "video"
    index = json.loads((dossier / "index.json").read_text())
    ips_entree = 1.0 / index["pas_s"]
    print(f"{len(index['images'])} images, {index['pas_s']} s chacune -> {ips_entree:.1f} i/s a l'entree, {IPS} i/s a la sortie")
    par_camera(dossier, index, ips_entree)
    composee(dossier, index, ips_entree, entete(Path(a.dossier)))
    if a.sans_images:
        for d in [p for p in dossier.iterdir() if p.is_dir()]:
            shutil.rmtree(d)
        print("  images effacees, vidéos gardées")
    print("VIDEO FINI")


if __name__ == "__main__":
    main()
