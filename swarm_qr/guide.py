"""Le guide vision-langage (étape 8) : un avis de bon sens sur où aller et par quel côté.

Le modèle regarde deux images — la caméra du drone et la carte vue de dessus, avec des zones
numérotées — et répond par un numéro de zone, un côté d'abordage et une phrase. Il conseille,
il ne commande pas : la géométrie garde la main sur la pose exacte, et le poids λ de son avis
dans la note peut être nul. Il travaille en arrière-plan pour ne jamais bloquer la boucle de
mission ; en son absence, la décision se fait sans lui.
"""

from __future__ import annotations

import os
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor

import cv2
import numpy as np

from . import planning

MODELES = {
    "smolvlm": "HuggingFaceTB/SmolVLM-500M-Instruct",
    "smolvlm-2b": "HuggingFaceTB/SmolVLM-Instruct",
}
COTES = {"north": 2, "nord": 2, "south": 3, "sud": 3, "east": 0, "est": 0, "west": 1, "ouest": 1}
CONTEXTE = (
    "You guide a drone that must read QR codes glued on cardboard boxes stored on warehouse "
    "shelves. The first image is the drone's side camera. The second image is the map seen from "
    "above: dark grey is a shelf or a wall, white has already been read, light grey is free, "
    "medium grey is unknown, orange dots are boxes seen but not read yet, and the red numbered "
    "circles are the candidate zones. "
)
QUESTION_ZONE = CONTEXTE + "{description}Which zone number should the drone go to next to read the most unread boxes? Answer with the number only."
QUESTION_COTE = CONTEXTE + "{description}The drone goes to zone {n}. From which side should it approach the shelf there: north, south, east or west? Answer with one word."


def decrit(zones: list[dict], position=None, codes_lus: int | None = None) -> str:
    """Ce que la carte sait de chaque zone, en phrases : des faits, jamais la note du cerveau,
    sinon le modèle recopierait le choix de la géométrie."""
    lignes = []
    for z in zones:
        faits = []
        if z.get("n_lire") is not None:
            if z["n_lire"]:
                faits.append(f"{z['n_lire']} QR code(s) spotted but not read yet")
            if z.get("n_couvrir_cartons"):
                faits.append(f"{z['n_couvrir_cartons']} shelf section(s) with boxes never looked at")
            elif z.get("n_couvrir"):
                faits.append(f"{z['n_couvrir']} surface(s) never looked at")
            if z.get("n_explorer"):
                faits.append("unexplored space")
        else:
            genres = set(z.get("genres", []))
            if "lire" in genres:
                faits.append("QR codes spotted but not read yet")
            if "couvrir" in genres:
                faits.append("surfaces never looked at")
            if "explorer" in genres:
                faits.append("unexplored space")
            faits.append(f"{z['cibles']} candidate target(s)")
        if z.get("cote"):
            faits.append(f"shelf faces looking {z['cote']}")
        d = ""
        if position is not None:
            d = f", {float(np.hypot(z['centre'][0] - position[0], z['centre'][1] - position[1])):.0f} m from the drone"
        lignes.append(f"Zone {z['numero']}: " + ", ".join(faits) + d + ".")
    tete = f"So far {codes_lus} codes have been read. " if codes_lus is not None else ""
    return tete + "The map says: " + " ".join(lignes) + " "


MAX_PIXELS = 640 * 28 * 28      # pour les modèles Qwen : une image de caméra vaut ~640 jetons


class Guide:
    def __init__(self, nom: str = "smolvlm", max_tokens: int = 60, device: str = "cuda",
                 quantisation: str | None = None):
        """`quantisation` : None (poids en fp16), "4bit" ou "8bit" (bitsandbytes, sur GPU) —
        un modèle de 7 milliards de paramètres ne tient dans 8 Go de mémoire vidéo qu'en 4 bits.
        Sur `device="cpu"`, les poids sont en bf16 : lent sans AMX, mais sans installation."""
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.nom = MODELES.get(nom, nom)
        options = {"max_pixels": MAX_PIXELS} if "qwen" in self.nom.lower() else {}
        self.processor = AutoProcessor.from_pretrained(self.nom, **options)
        self.dtype = torch.bfloat16 if device == "cpu" else torch.float16
        if quantisation:
            from transformers import BitsAndBytesConfig
            config = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                         bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
                      if quantisation == "4bit" else BitsAndBytesConfig(load_in_8bit=True))
            self.modele = AutoModelForImageTextToText.from_pretrained(
                self.nom, quantization_config=config, device_map={"": device}, dtype=torch.float16)
        else:
            self.modele = AutoModelForImageTextToText.from_pretrained(self.nom, dtype=self.dtype).to(device)
        self.modele.eval()
        self.device, self.max_tokens = device, max_tokens
        self.latences: list[float] = []
        self.reponses: list[str] = []
        self._pool = ThreadPoolExecutor(max_workers=1)

    # ---------------------------------------------------------------- appel

    def repond(self, image_bgr: np.ndarray, vue_bgr: np.ndarray, question: str) -> str:
        """La réponse brute du modèle aux deux images et à une question."""
        import torch
        from PIL import Image

        images = [Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) for im in (image_bgr, vue_bgr)]
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "image"},
                                                 {"type": "text", "text": question}]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        entrees = self.processor(text=prompt, images=images, return_tensors="pt")
        entrees = {k: (v.to(self.device, dtype=self.dtype) if v.dtype.is_floating_point else v.to(self.device))
                   for k, v in entrees.items()}
        t0 = time.perf_counter()
        with torch.no_grad():
            sortie = self.modele.generate(**entrees, max_new_tokens=self.max_tokens, do_sample=False)
        self.latences.append(time.perf_counter() - t0)
        texte = self.processor.batch_decode(sortie[:, entrees["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        texte = texte.strip()
        self.reponses.append(texte)
        return texte

    @staticmethod
    def zone_dans(texte: str, zones: list[dict]) -> int | None:
        """Le numéro de zone lu dans la réponse : « zone 3 », « 3. », « Zone: 3 » ; None si absent
        ou hors liste."""
        numeros = {z["numero"] for z in zones}
        m = re.search(r"zone\s*[:=]?\s*(\d+)", texte, re.IGNORECASE) or re.search(r"(\d+)", texte)
        if not m:
            return None
        n = int(m.group(1))
        return n if n in numeros else None

    @staticmethod
    def cote_dans(texte: str) -> int | None:
        for mot in re.findall(r"[a-zA-Zéè]+", texte.lower()):
            if mot in COTES:
                return COTES[mot]
        return None

    def conseille(self, image_bgr, vue_bgr, zones: list[dict], position=None,
                  description: str | None = None) -> planning.Avis | None:
        """Deux questions courtes plutôt qu'une longue : un petit modèle suit mieux une consigne
        à la fois. D'abord la zone, puis le côté d'abordage pour cette zone. `description` :
        ce que la carte sait des zones, en phrases, en plus des deux images."""
        desc = description or ""
        numero = self.zone_dans(self.repond(image_bgr, vue_bgr, QUESTION_ZONE.format(description=desc)), zones)
        if numero is None:
            return None
        texte_cote = self.repond(image_bgr, vue_bgr, QUESTION_COTE.format(description=desc, n=numero))
        cote = self.cote_dans(texte_cote)
        z = next(z for z in zones if z["numero"] == numero)
        phrase = f"zone {numero}" + (f", cote {planning.NOMS_COTES[cote]}" if cote is not None else "")
        return planning.Avis(centre=np.array([*z["centre"][:2], 0.0]), rayon=float(z["rayon"]), cote=cote,
                             phrase=phrase + f" ({self.reponses[-2][:40]!r} / {texte_cote[:40]!r})")

    def demande(self, image_bgr, vue_bgr, zones, position=None, description: str | None = None) -> Future:
        """Le même avis, calculé en arrière-plan."""
        return self._pool.submit(self.conseille, image_bgr.copy(), vue_bgr.copy(), zones, position, description)

    def bilan(self) -> dict:
        return {"modele": self.nom, "appels": len(self.latences),
                "latence_mediane_s": round(float(np.median(self.latences)), 2) if self.latences else None}
