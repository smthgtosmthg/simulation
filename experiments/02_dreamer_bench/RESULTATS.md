# Mesure 02 — Ce que coûte DreamerV3 sur notre machine

Date : 29 août 2026 · `dreamer_bench.py` · RTX 2060 SUPER 8 Go, torch 2.7 + CUDA 12.6
Architecture DreamerV3 reconstruite fidèlement : encodeur CNN 4 étages, RSSM (GRU + 32×32
variables catégorielles), décodeur, têtes récompense/fin, acteur, critique. Lot 16×64,
imagination sur 15 pas — les réglages officiels.

## A. Coût d'un pas d'entraînement

| Préréglage | Caméras | Lot | Précision | Params | VRAM pic | Temps/pas |
|---|---|---|---|---|---|---|
| 12M | 2 | 16×64 | **fp16** | 8,9 M | **2,33 Go** | **296 ms** |
| 12M | 2 | 16×64 | fp32 | 8,9 M | 2,51 Go | 318 ms |
| 12M | 2 | 16×64 | **bf16** | 8,9 M | 2,33 Go | **1506 ms** |
| 12M | 1 | 16×64 | fp16 | 8,9 M | 2,18 Go | 281 ms |
| 25M | 2 | 16×64 | fp16 | 18,8 M | 3,38 Go | 335 ms |
| 50M | 2 | 16×64 | fp16 | 39,0 M | 5,36 Go | 486 ms |
| 12M | 2 | 32×64 | fp16 | 8,9 M | 4,45 Go | 404 ms |
| 12M | 2 | 8×64 | fp16 | 8,9 M | 1,26 Go | 264 ms |

**Piège matériel majeur : bf16 est 5 fois plus lent que fp16.** La RTX 2060 SUPER est une
architecture Turing (capacité 7,5) : elle possède des unités fp16, mais **pas de bf16 natif**,
qu'elle émule. Or `torch.cuda.is_bf16_supported()` renvoie `True`, et la plupart des dépôts
DreamerV3 utilisent bf16 par défaut. **Il faut forcer fp16.** Sans ça, tout est 5× plus lent
pour rien.

**La VRAM n'est pas un problème.** Le préréglage 12M avec 2 caméras tient dans 2,33 Go sur les
7,8 Go disponibles. Même le préréglage 50M passe (5,36 Go). On peut aussi doubler le lot.

## B. Tampon de rejeu — la vraie contrainte

Images 64×64 en RGB, 2 caméras, 3 drones, stockage en `uint8` : **72 Ko par pas d'équipe**.

| Pas stockés | Taille | Tient dans 19 Go libres ? |
|---|---|---|
| 100 000 | 6,9 Go | oui |
| **250 000** | **17,2 Go** | **oui, tout juste** |
| 500 000 | 34,3 Go | non |
| 1 000 000 | 68,7 Go | non |

Le disque est plein à 90 %. Le plafond réaliste est donc d'environ **250 000 pas** en l'état.
Trois façons de le repousser, par ordre d'efficacité : une seule caméra (×2), compression JPEG
(×5 à ×10), images 48×48 (×1,8).

## C. Durée d'un entraînement complet

DreamerV3 rejoue chaque pas collecté plusieurs fois — c'est le `train_ratio`. Un pas
d'entraînement consomme 16 × 64 = 1024 pas rejoués.

| `train_ratio` | Pas d'environnement/s | 1 M pas | 5 M pas |
|---|---|---|---|
| 32 | 108 | **2,6 h** | 12,8 h |
| 64 | 54 | **5,1 h** | 25,7 h |
| 128 | 27 | 10,3 h | 51,3 h |
| 256 | 14 | 20,5 h | 102,6 h |
| **512** (défaut DMC) | 7 | **41,1 h** | 205,3 h |

## Ce que cette mesure corrige

**J'avais estimé « 1 à 2 semaines par run ». C'était faux.** Avec fp16 et un `train_ratio`
raisonnable, un run de 1 million de pas prend **entre 3 et 10 heures**. Même au réglage le plus
lourd (512), on reste à 41 heures — moins de deux jours.

**J'avais aussi présenté la VRAM comme un facteur de risque.** Elle ne l'est pas : 2,33 Go sur
7,8 Go disponibles.

Les deux arguments qui pesaient le plus contre Dreamer dans mon analyse initiale sont donc
**annulés par la mesure**.

## Ce qui reste vrai

1. **Le tampon de rejeu plafonne à ~250 000 pas** faute de disque. Contournable, mais c'est du
   travail d'ingénierie à budgéter.
2. **La mémoire récurrente ne tient pas l'épisode.** Le RSSM doit se souvenir des cartons déjà
   lus sur ~1800 pas ; sa portée utile est de l'ordre de quelques dizaines de pas. Une carte
   externe reste nécessaire — et elle retire au modèle du monde une partie de son intérêt.
3. **Le simulateur n'est pas mesuré ici.** C'est désormais **la principale inconnue** : à
   `train_ratio` 64, il faut qu'Isaac Sim produise **54 pas par seconde avec 6 caméras rendues**.
   C'est la prochaine mesure à faire, et c'est elle qui décidera.

## Reproduire

```bash
~/isaac5_env/bin/python experiments/02_dreamer_bench/dreamer_bench.py
```

Sorties : ce fichier, plus `resultats.csv` et `resultats.json` dans le même dossier.
