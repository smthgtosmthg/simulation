# Mesures

Chaque dossier contient un script autonome, ses résultats bruts, et un `RESULTATS.md`
qui explique ce que la mesure établit et ce qu'elle n'établit pas.

Règle : **aucune mesure ne dépend d'Isaac Sim** tant que ce n'est pas indispensable.
Elles tournent en quelques minutes et peuvent être relancées à volonté.

| # | Dossier | Question | Réponse courte |
|---|---|---|---|
| 01 | `01_resolution_qr/` | Quelle résolution faut-il pour lire un QR ? | Décodage à 2,07 px/module. En 64 px d'entrée, la portée tombe à 7,5 cm. Il faut ≥ 1000 px pour lire à 1,25 m. |
| 02 | `02_dreamer_bench/` | Combien coûte DreamerV3 sur notre machine ? | 296 ms/pas, 2,33 Go de VRAM. Un run fait 3 à 40 h, pas 2 semaines. **fp16 obligatoire : bf16 est 5× plus lent.** |
| 03 | `03_render_bench/` | Le simulateur suit-il, caméras allumées ? | Oui, largement : 736 pas/s à 32 environnements pour 54 nécessaires. **Marge ×13.** La résolution coûte peu (+4× de pixels = −18 % de débit). |

## Comment lancer

```bash
cd ~/simulation_mc02

# 01 — résolution et lisibilité des QR (~10 min, processeur seul)
python3 experiments/01_resolution_qr/qr_resolution_test.py

# 02 — coût de DreamerV3 (~3 min, GPU)
~/isaac5_env/bin/python experiments/02_dreamer_bench/dreamer_bench.py

# 03 — débit du simulateur avec caméras (~7 min, Isaac Sim)
bash experiments/03_render_bench/run_all.sh
```

Les deux scripts affichent leurs résultats directement dans le terminal, et écrivent
des fichiers CSV à côté.

Pour un essai rapide du banc Dreamer (moins d'itérations, chiffres plus bruités) :

```bash
~/isaac5_env/bin/python experiments/02_dreamer_bench/dreamer_bench.py --quick
```

## Comment lire les résultats

1. **Le terminal** donne tout : les tableaux sont imprimés au fur et à mesure.
2. **`RESULTATS.md`** dans chaque dossier : l'interprétation, ce qui est établi, et surtout
   ce que la mesure **ne prouve pas**.
3. **`resultats*.csv`** : les chiffres bruts, pour refaire des graphiques ou vérifier.

## Bilan sur Dreamer après les mesures 01 à 03

J'avais donné sept raisons de l'écarter. **Quatre sont mortes, mesurées.**

| Objection initiale | Verdict après mesure |
|---|---|
| Le QR est invisible en 64×64 | **Fausse.** Racks et cartons sont très visibles ; le décodage est externe de toute façon. |
| Un run prend 1 à 2 semaines | **Fausse.** 3 à 40 h selon le réglage — j'étais faux d'un facteur 30 à 80. |
| La mémoire graphique ne suffit pas | **Fausse.** 2,33 Go sur 7,8. |
| Le simulateur ne suivra pas | **Fausse.** 736 pas/s pour 54 nécessaires. |
| Le disque limite le tampon de rejeu | **Vraie**, mais contournable : ~250 000 pas à 2 caméras, le double à 1 caméra, bien plus en JPEG. |
| La mémoire récurrente ne tient pas l'épisode | **Non mesurée ici** — vient de la littérature. Une carte externe la rend caduque. |
| Trop de causes possibles pour déboguer | **Mon jugement**, pas un fait. |

**Dreamer est donc redevenu une option sérieuse.** Ma position initiale reposait sur des
estimations, pas sur des mesures ; les mesures les ont réfutées.

## À faire ensuite

**Mesure 04 — la marge disponible.** Sur notre géométrie, quel écart y a-t-il entre un balayage
fixe et un glouton omniscient ? C'est ce qui dit s'il y a quelque chose à apprendre — et ça vaut
pour les quatre directions, Dreamer compris. **C'est maintenant la mesure la plus importante**,
parce qu'aucune des trois premières ne dit si la tâche a du contenu.
