"""Un petit entraînement QLoRA sur la mission : le modèle en 4 bits reste gelé, de petites couches
apprennent à choisir la zone à partir du dossier épuré. Les exemples viennent de vols
d'entraînement, l'évaluation de vols jamais vus — tous sur l'entrepôt 9033, ce qui est dit dans
le résultat. Chaque cas d'entraînement est présenté plusieurs fois avec des numéros de zones
mélangés différemment : le numéro ne veut rien dire, et le modèle doit l'apprendre.

Sans image : avec le dossier épuré, la photo ne changeait pas le score (79 contre 78 %), et
l'entraînement texte seul tient dans les 8 Go de la carte.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))

from banc import instantanes, references                                            # noqa: E402
from variantes import _reponse_zone, melange                                        # noqa: E402
from swarm_qr.guide import texte_pour_le_guide                                          # noqa: E402

def texte_du_cas(cas: dict, graine: int) -> tuple[str, int, int]:
    """Le texte exactement comme en vol (dossier épuré, sans image), la bonne zone, celle de la
    géométrie."""
    zones, bonne, geo = melange(cas["zones"], cas["verite"], graine=graine)
    return texte_pour_le_guide(cas, zones), bonne, geo


def exemples_d_entrainement(cas: list[dict], permutations: int) -> list[tuple[str, str]]:
    out = []
    for k, c in enumerate(cas):
        for p in range(permutations):
            texte, bonne, _ = texte_du_cas(c, graine=10_000 + k * 97 + p)
            out.append((texte, f"ANSWER: {bonne}"))
    random.Random(0).shuffle(out)
    return out


def charge_modele(modele: str):
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    proc = AutoProcessor.from_pretrained(modele)
    m = AutoModelForImageTextToText.from_pretrained(modele, quantization_config=config,
                                                     device_map={"": "cuda"}, dtype=torch.float16,
                                                     attn_implementation="sdpa")
    return proc, m


def jetons(proc, texte: str, reponse: str | None):
    """Le prompt au format de conversation du modèle ; les étiquettes ne couvrent que la réponse."""
    messages = [{"role": "user", "content": [{"type": "text", "text": texte}]}]
    prompt = proc.apply_chat_template(messages, add_generation_prompt=True)
    ids_prompt = proc.tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    if reponse is None:
        return ids_prompt, None
    ids_rep = proc.tokenizer(reponse + proc.tokenizer.eos_token, return_tensors="pt",
                             add_special_tokens=False)["input_ids"][0]
    ids = torch.cat([ids_prompt, ids_rep])
    labels = torch.cat([torch.full_like(ids_prompt, -100), ids_rep])
    return ids, labels


def entraine(proc, m, exemples, epoques: int, lr: float, accumulation: int, sortie: Path):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=True)
    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)")
    m = get_peft_model(m, lora)
    m.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    m.train()
    pas = 0
    t0 = time.perf_counter()
    for ep in range(epoques):
        perte_ep, n_ep = 0.0, 0
        for i, (texte, rep) in enumerate(exemples):
            ids, labels = jetons(proc, texte, rep)
            ids, labels = ids[None].cuda(), labels[None].cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                perte = m(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels).loss
            (perte / accumulation).backward()
            perte_ep += float(perte); n_ep += 1
            if (i + 1) % accumulation == 0:
                torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad(set_to_none=True); pas += 1
            if (i + 1) % 50 == 0:
                print(f"  epoque {ep + 1} exemple {i + 1}/{len(exemples)} perte moyenne {perte_ep / n_ep:.3f} "
                      f"({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
        print(f"epoque {ep + 1} : perte moyenne {perte_ep / max(n_ep, 1):.3f}", flush=True)
    m.eval()
    m.save_pretrained(str(sortie))
    return m


@torch.no_grad()
def evalue(proc, m, cas: list[dict], etiquette: str) -> dict:
    justes = repondus = geo_ok = 0
    details = []
    for k, c in enumerate(cas):
        texte, bonne, geo = texte_du_cas(c, graine=k)
        ids, _ = jetons(proc, texte, None)
        sortie = m.generate(input_ids=ids[None].cuda(), attention_mask=torch.ones(1, len(ids), device="cuda"),
                            max_new_tokens=12, do_sample=False)
        brut = proc.tokenizer.decode(sortie[0, len(ids):], skip_special_tokens=True)
        zones, _, _ = melange(c["zones"], c["verite"], graine=k)
        rep = _reponse_zone(brut, zones)
        repondus += rep is not None
        justes += rep == bonne
        geo_ok += geo == bonne
        details.append({"mission": c["mission"], "k": c["k"], "drone": c.get("drone"),
                        "reponse": rep, "bonne": bonne, "geometrie": geo, "brut": brut.strip()[:60]})
    n = len(cas)
    print(f"  {etiquette:28s} {justes / n:6.1%} juste | repondus {repondus}/{n} | geometrie {geo_ok / n:.0%}", flush=True)
    return {"etiquette": etiquette, "cas": n, "justes": round(justes / n, 4), "repondus": repondus,
            "geometrie": round(geo_ok / n, 4), "details": details}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrainement", nargs="+", required=True, help="dossiers de mission pour apprendre")
    ap.add_argument("--test", nargs="+", required=True, help="dossiers de mission jamais vus")
    ap.add_argument("--modele", required=True)
    ap.add_argument("--permutations", type=int, default=6)
    ap.add_argument("--epoques", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accumulation", type=int, default=4)
    ap.add_argument("--sortie", default="adaptateur_lora")
    a = ap.parse_args()

    train = instantanes([Path(m) for m in a.entrainement])
    test = instantanes([Path(m) for m in a.test])
    exemples = exemples_d_entrainement(train, a.permutations)
    print(f"entrainement : {len(train)} cas x {a.permutations} permutations = {len(exemples)} exemples ; "
          f"test : {len(test)} cas jamais vus, geometrie {references(test)['zone_la_plus_utile_geometrie']:.0%}")
    proc, m = charge_modele(a.modele)
    avant = evalue(proc, m, test, "avant entrainement, sans image")
    m = entraine(proc, m, exemples, a.epoques, a.lr, a.accumulation, HERE / a.sortie)
    apres = evalue(proc, m, test, "apres entrainement, sans image")
    w = sum(1 for x, y in zip(avant["details"], apres["details"]) if x["reponse"] != x["bonne"] and y["reponse"] == y["bonne"])
    l = sum(1 for x, y in zip(avant["details"], apres["details"]) if x["reponse"] == x["bonne"] and y["reponse"] != y["bonne"])
    print(f"cas gagnes par l'entrainement : {w}, perdus : {l}")
    (HERE / "resultats_entrainement.json").write_text(json.dumps({
        "entrainement": a.entrainement, "test": a.test, "exemples": len(exemples), "epoques": a.epoques,
        "avant": avant, "apres": apres, "gagnes": w, "perdus": l}, indent=1))
    print("ENTRAINEMENT FINI")


if __name__ == "__main__":
    main()
