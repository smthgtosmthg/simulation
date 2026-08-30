# Mesure 03 — Débit du simulateur avec les caméras allumées

Date : 29 août 2026 · `render_bench.py` + `run_all.sh` · Isaac Sim 5.1 / Isaac Lab 2.3.2
RTX 2060 SUPER 8 Go · TiledCamera, images RGB, scène avec sol, lumière et deux rangées de racks

## La question

Dreamer alterne collecte et apprentissage, et **la plus lente des deux impose son rythme**.
La mesure 02 donne 296 ms par pas d'apprentissage, sur un lot de 16 × 64 = 1024 pas rejoués.
Avec un `train_ratio` de 64, chaque pas collecté est rejoué 64 fois, donc il faut
1024 ÷ 64 = 16 pas neufs par pas d'apprentissage, soit **54 pas neufs par seconde**.

**Le simulateur peut-il fournir 54 pas par seconde avec les caméras ?**

## Les mesures

| Environnements | Résolution | Caméras | Pas env/s | Images/s |
|---|---|---|---|---|
| 1 | 64 px | 1 | 54,2 | 54,2 |
| 1 | 64 px | 2 | 34,0 | 68,0 |
| **3** (notre essaim) | 64 px | 2 | **93,4** | 186,8 |
| 8 | 64 px | 2 | 245,6 | 491,3 |
| 16 | 64 px | 2 | 409,8 | 819,7 |
| **32** | 64 px | 2 | **736,1** | **1472,2** |
| 8 | **128 px** | 2 | 201,4 | 402,8 |

Toutes les configurations ont produit des images valides (100 % de pixels non noirs,
valeur moyenne ≈ 197).

## Ce que ça établit

**1. Le simulateur n'est pas le goulot — de très loin.**
Il faut 54 pas/s ; on en obtient 736 à 32 environnements. **Marge d'un facteur 13.**
Même la configuration minimale (3 environnements, un par drone) donne 93 pas/s, soit
déjà 1,7 fois le besoin.

**2. Le rendu par tuiles monte presque linéairement.**
31 pas/s par environnement à 3 envs, 31 à 8, 26 à 16, 23 à 32. La légère baisse à partir
de 16 est normale ; il n'y a aucun effondrement.

**3. La résolution coûte peu.**
Passer de 64 à 128 pixels (4 fois plus de pixels) ne coûte que **18 %** de débit
(246 → 201 pas/s). Le coût est dominé par le nombre d'appels de rendu, pas par la surface.
C'est une bonne nouvelle : monter en résolution pour mieux voir est presque gratuit.

**4. On peut baisser le `train_ratio` et aller plus vite.**
Avec 8 environnements (246 pas/s), un `train_ratio` de 16 est soutenable (il en faudrait 216).
Cela ramène un entraînement d'un million de pas à **environ 2,6 heures**.

## Limites de cette mesure

- **La colonne VRAM est inutilisable** (0,01 Go partout) : la mémoire de TiledCamera est
  allouée hors de l'allocateur de PyTorch, donc `torch.cuda.max_memory_allocated()` ne la voit
  pas. Il faudrait relire `nvidia-smi` pour l'avoir. À corriger si la question devient utile.
- **La scène est simple** : un sol, une lumière, deux blocs. Un vrai entrepôt avec 123 cartons
  texturés et un éclairage réaliste sera plus lent. Ces chiffres sont donc un **plafond**.
- **Forte variance** : la même configuration (1 env, 1 caméra) a donné 86 pas/s puis 54 pas/s
  à deux lancements différents. Les valeurs sont des ordres de grandeur, pas des mesures fines.
- **Un blocage passager est survenu** au tout premier essai (8 envs × 2 caméras, bloqué 28 min).
  Il ne s'est pas reproduit. Cause non identifiée. **`timeout -s KILL` est obligatoire :
  Isaac ignore le signal d'arrêt normal.**

## Reproduire

```bash
bash experiments/03_render_bench/run_all.sh          # le balayage complet, ~7 min

# une seule configuration
PYTHONUNBUFFERED=1 ~/isaac5_env/bin/python experiments/03_render_bench/render_bench.py \
    --envs 8 --cams 2 --res 64 --steps 60 \
    --kit_args="--/rtx/verifyDriverVersion/enabled=false"
```

L'avertissement de pilote (535.32 contre 535.129 requis) est **sans effet** : le rendu produit
des images correctes. Le drapeau `verifyDriverVersion` sert seulement à le faire taire.
