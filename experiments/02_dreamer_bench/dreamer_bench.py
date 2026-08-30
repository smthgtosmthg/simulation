"""
Mesure 02 — Combien coûte DreamerV3 sur NOTRE machine ?

On reconstruit fidèlement l'architecture de DreamerV3 (encodeur convolutif, RSSM
récurrent, décodeur, têtes récompense/fin, acteur, critique) et on chronomètre un
pas d'entraînement complet : apprentissage du modèle du monde, puis imagination,
puis mise à jour de l'acteur et du critique.

Ce que ça mesure  : le coût d'APPRENTISSAGE de Dreamer, en temps et en mémoire.
Ce que ça ne mesure PAS : le coût du simulateur (Isaac Sim), qui s'ajoute.

Usage :
    ~/isaac5_env/bin/python experiments/02_dreamer_bench/dreamer_bench.py
    ~/isaac5_env/bin/python experiments/02_dreamer_bench/dreamer_bench.py --quick
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path(__file__).parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Tailles officielles de DreamerV3 (Hafner et al. 2023, table des préréglages)
PRESETS = {
    "12M": dict(deter=256, cnn_depth=24, units=256, layers=2),
    "25M": dict(deter=512, cnn_depth=32, units=512, layers=2),
    "50M": dict(deter=1024, cnn_depth=48, units=640, layers=3),
}
STOCH, CLASSES = 32, 32          # 32 variables catégorielles à 32 valeurs
IMG, ACT_DIM = 64, 4             # images 64x64 ; action = vx, vy, vz, omega
PROPRIO_DIM = 7


def mlp(i, o, units, layers):
    net, d = [], i
    for _ in range(layers):
        net += [nn.Linear(d, units), nn.LayerNorm(units), nn.SiLU()]
        d = units
    return nn.Sequential(*net, nn.Linear(d, o))


class Encoder(nn.Module):
    """CNN à 4 étages, division par 2 à chaque fois : 64 -> 32 -> 16 -> 8 -> 4."""

    def __init__(self, depth, n_cams):
        super().__init__()
        ch, layers = 3 * n_cams, []
        for i in range(4):
            out = depth * 2 ** i
            layers += [nn.Conv2d(ch, out, 4, 2, 1), nn.LayerNorm([out, IMG >> (i + 1), IMG >> (i + 1)]), nn.SiLU()]
            ch = out
        self.cnn = nn.Sequential(*layers)
        self.cnn_out = ch * 4 * 4
        self.vec = nn.Sequential(nn.Linear(PROPRIO_DIM, 128), nn.LayerNorm(128), nn.SiLU())
        self.out_dim = self.cnn_out + 128

    def forward(self, img, vec):
        return torch.cat([self.cnn(img).flatten(1), self.vec(vec)], -1)


class Decoder(nn.Module):
    def __init__(self, depth, n_cams, feat):
        super().__init__()
        ch = depth * 8
        self.fc = nn.Linear(feat, ch * 4 * 4)
        self.ch = ch
        layers = []
        for i in range(3, 0, -1):
            out = depth * 2 ** (i - 1)
            layers += [nn.ConvTranspose2d(ch, out, 4, 2, 1), nn.LayerNorm([out, IMG >> i, IMG >> i]), nn.SiLU()]
            ch = out
        layers += [nn.ConvTranspose2d(ch, 3 * n_cams, 4, 2, 1)]
        self.deconv = nn.Sequential(*layers)
        self.vec = nn.Linear(feat, PROPRIO_DIM)

    def forward(self, f):
        return self.deconv(self.fc(f).view(-1, self.ch, 4, 4)), self.vec(f)


class RSSM(nn.Module):
    """Le coeur du modele du monde : une memoire recurrente `deter` + un etat
    stochastique `stoch`. C'est lui qui apprend a predire la suite."""

    def __init__(self, deter, units, layers, embed_dim):
        super().__init__()
        self.deter, self.sd = deter, STOCH * CLASSES
        self.pre = nn.Sequential(nn.Linear(self.sd + ACT_DIM, units), nn.LayerNorm(units), nn.SiLU())
        self.gru = nn.GRUCell(units, deter)
        self.prior = mlp(deter, self.sd, units, layers)
        self.post = mlp(deter + embed_dim, self.sd, units, layers)

    @staticmethod
    def sample(logits):
        lg = logits.view(-1, STOCH, CLASSES)
        p = F.softmax(lg, -1)
        oh = F.one_hot(torch.multinomial(p.view(-1, CLASSES), 1).squeeze(-1), CLASSES).view_as(p)
        return (oh + p - p.detach()).flatten(1), lg   # straight-through

    def step(self, stoch, action, deter, embed=None):
        deter = self.gru(self.pre(torch.cat([stoch, action], -1)), deter)
        prior_lg = self.prior(deter)
        if embed is None:
            s, lg = self.sample(prior_lg)
            return s, deter, lg, None
        post_lg = self.post(torch.cat([deter, embed], -1))
        s, lg = self.sample(post_lg)
        return s, deter, self.prior(deter).view(-1, STOCH, CLASSES), lg


class Dreamer(nn.Module):
    def __init__(self, preset, n_cams):
        super().__init__()
        p = PRESETS[preset]
        self.enc = Encoder(p["cnn_depth"], n_cams)
        self.rssm = RSSM(p["deter"], p["units"], p["layers"], self.enc.out_dim)
        feat = p["deter"] + STOCH * CLASSES
        self.dec = Decoder(p["cnn_depth"], n_cams, feat)
        self.reward = mlp(feat, 255, p["units"], p["layers"])   # symlog two-hot
        self.cont = mlp(feat, 1, p["units"], p["layers"])
        self.actor = mlp(feat, ACT_DIM * 2, p["units"], p["layers"])
        self.critic = mlp(feat, 255, p["units"], p["layers"])
        self.feat = feat


DTYPES = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


def train_step(m, opt, batch, horizon, prec):
    B, T = batch["img"].shape[:2]
    dt = DTYPES[prec]
    with torch.autocast("cuda", dtype=dt or torch.float32, enabled=dt is not None):
        # --- 1. modele du monde : on remonte la sequence pas a pas ---
        embed = m.enc(batch["img"].flatten(0, 1), batch["vec"].flatten(0, 1)).view(B, T, -1)
        stoch = torch.zeros(B, STOCH * CLASSES, device=DEV)
        deter = torch.zeros(B, m.rssm.deter, device=DEV)
        feats, kls = [], []
        for t in range(T):
            stoch, deter, pri, pos = m.rssm.step(stoch, batch["act"][:, t], deter, embed[:, t])
            feats.append(torch.cat([stoch, deter], -1))
            kls.append(F.kl_div(F.log_softmax(pri, -1), F.log_softmax(pos, -1),
                                log_target=True, reduction="batchmean"))
        feat = torch.stack(feats, 1)
        flat = feat.flatten(0, 1)
        img_hat, vec_hat = m.dec(flat)
        loss = (F.mse_loss(img_hat, batch["img"].flatten(0, 1))
                + F.mse_loss(vec_hat, batch["vec"].flatten(0, 1))
                + torch.stack(kls).mean()
                + m.reward(flat).mean() * 0 + m.cont(flat).mean() * 0)

        # --- 2. imagination : on part de CHAQUE etat et on reve `horizon` pas ---
        s, d = flat[:, :STOCH * CLASSES].detach(), flat[:, STOCH * CLASSES:].detach()
        im = []
        for _ in range(horizon):
            f = torch.cat([s, d], -1)
            a = torch.tanh(m.actor(f)[:, :ACT_DIM])
            s, d, _, _ = m.rssm.step(s, a, d)
            im.append(torch.cat([s, d], -1))
        imf = torch.stack(im)
        # --- 3. acteur + critique sur les trajectoires revees ---
        loss = loss + m.critic(imf).mean() * 0 + m.actor(imf).mean() * 0 + m.reward(imf).mean() * 0

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return float(loss.detach())


def bench(preset, n_cams, B, T, horizon, prec, iters):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    m = Dreamer(preset, n_cams).to(DEV)
    opt = torch.optim.Adam(m.parameters(), 1e-4)
    params = sum(p.numel() for p in m.parameters())

    batch = {
        "img": torch.rand(B, T, 3 * n_cams, IMG, IMG, device=DEV),
        "vec": torch.rand(B, T, PROPRIO_DIM, device=DEV),
        "act": torch.rand(B, T, ACT_DIM, device=DEV),
    }
    for _ in range(3):                      # chauffe
        train_step(m, opt, batch, horizon, prec)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        train_step(m, opt, batch, horizon, prec)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters

    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    del m, opt, batch
    torch.cuda.empty_cache()
    return params, dt, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="moins d'itérations, pour un essai rapide")
    a = ap.parse_args()
    iters = 5 if a.quick else 15

    print(f"GPU : {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} Go)")
    print(f"torch {torch.__version__} | CUDA {torch.version.cuda}\n")

    # config de reference DreamerV3 : lot 16 x 64, imagination 15 pas
    configs = [
        ("12M", 2, 16, 64, 15, "fp16"),
        ("12M", 2, 16, 64, 15, "fp32"),
        ("12M", 2, 16, 64, 15, "bf16"),
        ("12M", 1, 16, 64, 15, "fp16"),
        ("25M", 2, 16, 64, 15, "fp16"),
        ("50M", 2, 16, 64, 15, "fp16"),
        ("12M", 2, 32, 64, 15, "fp16"),
        ("12M", 2, 8, 64, 15, "fp16"),
    ]

    print("=" * 84)
    print("A. COÛT D'UN PAS D'ENTRAÎNEMENT")
    print("=" * 84)
    print(f"{'preset':>7} {'cams':>5} {'lot':>8} {'précision':>10} "
          f"{'params':>9} {'VRAM pic':>10} {'temps/pas':>11} {'pas/s':>7}")
    rows = []
    for preset, cams, B, T, H, prec in configs:
        try:
            p, dt, peak = bench(preset, cams, B, T, H, prec, iters)
            rows.append(dict(preset=preset, cams=cams, batch=f"{B}x{T}",
                             precision=prec,
                             params_M=round(p / 1e6, 1), vram_gb=round(peak, 2),
                             sec_per_step=round(dt, 4), steps_per_s=round(1 / dt, 2)))
            print(f"{preset:>7} {cams:>5} {B}x{T:<5} {prec:>10} "
                  f"{p/1e6:>8.1f}M {peak:>9.2f}G {dt*1000:>9.0f}ms {1/dt:>7.2f}")
        except torch.cuda.OutOfMemoryError:
            print(f"{preset:>7} {cams:>5} {B}x{T:<5} {prec:>10}   MÉMOIRE INSUFFISANTE")
            torch.cuda.empty_cache()

    ref = next((r for r in rows if r["preset"] == "12M" and r["cams"] == 2
                and r["precision"] == "fp16" and r["batch"] == "16x64"), rows[0] if rows else None)
    if ref is None:
        print("\nAucune configuration n'a tenu. Arrêt.")
        return

    # ---------- B. tampon de rejeu ----------
    print("\n" + "=" * 84)
    print("B. TAMPON DE REJEU (le point qui inquiétait)")
    print("=" * 84)
    per_drone = IMG * IMG * 3 * 2          # 2 caméras, uint8
    per_team = per_drone * 3
    print(f"  Un pas, un drone, 2 caméras 64x64 : {per_drone/1024:.0f} Ko")
    print(f"  Un pas, les 3 drones             : {per_team/1024:.0f} Ko")
    print(f"\n{'pas stockés':>14} {'taille':>12}   tient dans 19 Go libres ?")
    for steps in [100_000, 250_000, 500_000, 1_000_000, 2_000_000]:
        gb = steps * per_team / 2 ** 30
        print(f"{steps:>14,} {gb:>10.1f} Go   {'oui' if gb < 19 else 'NON'}")

    # ---------- C. duree d'un run ----------
    print("\n" + "=" * 84)
    print("C. DURÉE D'UN ENTRAÎNEMENT COMPLET")
    print("=" * 84)
    print("  DreamerV3 rejoue plusieurs fois chaque pas collecté : c'est le `train_ratio`.")
    print("  Un pas d'entraînement consomme lot x longueur = 1024 pas rejoués.\n")
    sec = ref["sec_per_step"]
    replayed = 16 * 64
    print(f"  Mesuré : {sec*1000:.0f} ms par pas d'entraînement ({ref['preset']}, "
          f"{ref['cams']} caméras, {ref['precision']})\n")
    print(f"{'train_ratio':>12} {'pas env/s':>11} {'1M pas env':>13} {'5M pas env':>13}")
    for tr in [32, 64, 128, 256, 512]:
        env_per_s = replayed / (tr * sec)
        print(f"{tr:>12} {env_per_s:>11.0f} {1e6/env_per_s/3600:>11.1f} h {5e6/env_per_s/3600:>11.1f} h")

    print("\n  Rappel : ces durées ne comptent QUE l'apprentissage.")
    print("  Le simulateur (Isaac Sim + rendu de 6 caméras) s'ajoute par-dessus.")

    with open(OUT / "resultats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(OUT / "resultats.json", "w") as f:
        json.dump({"gpu": torch.cuda.get_device_name(0), "configs": rows}, f, indent=2)
    print(f"\n  Résultats écrits dans {OUT}/resultats.csv et resultats.json")


if __name__ == "__main__":
    main()
