# SwarmScan-Map — guide complet : de l'installation au lancement de l'entraînement

Ce document décrit, étape par étape, comment installer la machine, vérifier que tout
fonctionne, lancer l'entraînement du modèle SwarmScan-Map, le suivre, le reprendre et
l'évaluer. Toutes les commandes se lancent depuis la racine du dépôt
`~/simulation_mc02` sauf mention contraire.

---

## 1. Ce que fait le modèle

Trois drones volent dans le même entrepôt simulé et doivent lire un maximum de QR codes
collés sur les cartons des racks, en un temps limité.

| Élément | Valeur |
|---|---|
| Simulateur | Isaac Sim 5.1 + Isaac Lab 2.3 |
| Algorithme | PPO à paramètres partagés (un seul réseau pour les 3 drones), bibliothèque `rsl_rl` 3.0 |
| Action (par drone) | 4 nombres continus dans [-1, 1] : `vx`, `vy`, `vz`, vitesse de lacet |
| Observation | 2 cartes égocentriques 32×32 × 9 canaux (18 432 valeurs) + 98 valeurs vectorielles (LiDAR 72 secteurs, état du drone, coéquipiers) |
| Observation du critique seulement | 7 valeurs privilégiées (utilisées à l'entraînement, jamais par la politique) |
| Capteur de lecture | 2 caméras latérales (±90° du cap), portée 1,25 m, angle 50°, vitesse max 0,6 m/s, 2 images consécutives |
| Difficulté | curriculum à 10 crans : le gate part très tolérant (4 m, 60°, 1,45 m/s) et se resserre jusqu'aux valeurs calibrées |

Le drone ne reçoit **jamais** la position des QR codes dans son observation. Il ne voit
que sa carte de couverture, son LiDAR et son état. C'est la contrainte centrale de la
conception (`docs/conception_solution_finale.md`).

---

## 2. Prérequis matériels et logiciels

Configuration sur laquelle le projet tourne aujourd'hui :

| Élément | Valeur requise | Valeur de la machine actuelle |
|---|---|---|
| OS | Ubuntu 22.04 ou plus récent | Ubuntu 22.04.5 LTS |
| GPU NVIDIA | RTX, 8 Go de VRAM minimum | RTX 2060 SUPER, 8 Go |
| Pilote NVIDIA | ≥ 525 (CUDA 12) | 535.288.01 / CUDA 12.2 |
| Python | 3.11 exactement | 3.11.0rc1 |
| Espace disque | 20 Go libres pour l'installation | à surveiller, voir §9.4 |
| Internet | oui, au premier lancement (téléchargement de l'entrepôt USD) | — |

Vérifier le GPU et le pilote :

```bash
nvidia-smi
```

Si cette commande échoue, rien d'autre ne marchera : installer d'abord le pilote NVIDIA.

---

## 3. Installation à partir de zéro

L'environnement de travail s'appelle `~/isaac5_env`. C'est un venv Python 3.11 classique.
Le script `install_isaac_sim.sh` présent à la racine du dépôt est **obsolète** : il
installe Isaac Sim 4.5 dans `~/isaac_sim_env` avec Python 3.10. Ne pas l'utiliser. Suivre
les étapes ci-dessous.

### 3.1 Dépendances système

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git curl wget \
    python3.11 python3.11-venv python3.11-dev \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6
```

### 3.2 Création du venv

```bash
python3.11 -m venv ~/isaac5_env
source ~/isaac5_env/bin/activate
pip install --upgrade pip setuptools wheel
```

### 3.3 Isaac Sim 5.1

Les paquets sont hébergés sur l'index PyPI de NVIDIA. Compter 10 à 25 minutes et environ
12 Go téléchargés.

```bash
export OMNI_KIT_ACCEPT_EULA=YES
pip install "isaacsim[all]==5.1.0"       --extra-index-url https://pypi.nvidia.com
pip install "isaacsim[extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

`[extscache]` met les extensions Kit en cache local. Sans ce paquet, chaque lancement
retélécharge des extensions et prend plusieurs minutes de plus.

### 3.4 Isaac Lab 2.3

```bash
pip install isaaclab==2.3.2.post1
```

Le dossier `~/IsaacLab` (clone du dépôt GitHub) n'est **pas** utilisé par ce projet ;
l'import vient du paquet pip. On peut l'ignorer.

### 3.5 Bibliothèques du projet

```bash
pip install rsl-rl-lib==3.0.1 \
            torch==2.7.0 torchvision==0.22.0 \
            numpy==1.26.0 \
            qrcode pillow opencv-python-headless \
            tensorboard matplotlib pandas scipy \
            pytest pytest-mock
```

Notes :

- `rsl-rl-lib` 3.0 est obligatoire : le code utilise l'API `obs_groups` (`policy` /
  `critic`) introduite dans la version 3. Une version 2.x ne démarrera pas.
- `numpy` doit rester en 1.26 : Isaac Sim 5.1 n'est pas compatible NumPy 2.
- `qrcode` + `pillow` servent à générer les 123 images de QR codes collées sur les cartons.
- `opencv-python-headless` sert au décodeur réel utilisé par `calibrate_gate.py`.

### 3.6 Vérification de l'installation

```bash
~/isaac5_env/bin/python -c "import isaacsim, isaaclab, rsl_rl, torch; \
print('isaacsim', isaacsim.__version__ if hasattr(isaacsim,'__version__') else 'ok'); \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

La sortie doit afficher `cuda True`. Si c'est `False`, PyTorch a été installé en version
CPU : réinstaller avec l'index CUDA (`--index-url https://download.pytorch.org/whl/cu121`).

### 3.7 Le lanceur

Tous les scripts qui ouvrent Isaac Sim passent par `rl_inventory/launch.sh`. Ce script :

- active `~/isaac5_env` ;
- pose les variables EGL / NVIDIA nécessaires au rendu ;
- accepte automatiquement le EULA Omniverse ;
- ajoute le flag obligatoire `--kit_args="--/rtx/verifyDriverVersion/enabled=false"`
  (sans lui, Isaac Sim refuse de démarrer sur ce pilote) ;
- en mode fenêtré, choisit le `DISPLAY` et le `XAUTHORITY`.

Usage :

```bash
bash rl_inventory/launch.sh <script.py> [arguments]            # avec fenêtre
bash rl_inventory/launch.sh <script.py> --headless [arguments] # sans fenêtre
```

**Toujours utiliser `--headless` pour l'entraînement.** Le rendu graphique consomme de la
VRAM et ralentit la simulation sans rien apporter.

### 3.8 Premier lancement : téléchargement de l'entrepôt

L'entrepôt est un fichier USD hébergé chez NVIDIA (`warehouse_multiple_shelves.usd`,
défini dans `rl_inventory/config_rl.py`). Il est téléchargé au premier démarrage puis mis
en cache. Le premier lancement prend donc plusieurs minutes de plus que les suivants, et
demande une connexion internet.

Les images de QR codes sont générées automatiquement dans `rl_inventory/assets/qr/`
(123 fichiers PNG, un par carton) si elles n'existent pas.

---

## 4. Carte des fichiers

```
rl_inventory/
├── launch.sh                  lanceur Isaac Sim (à utiliser pour tout script Isaac)
├── config_rl.py               paramètres bas niveau : LiDAR, action, drone, entrepôt
├── env.py                     arène de base : entrepôt, drones, LiDAR, lecture QR
├── actuator.py                modèle d'actionneur (retard + accélération bornée)
├── qr_task.py                 génère et colle les QR sur les cartons
├── assets/qr/                 les 123 images PNG
├── swarmscan_map/
│   ├── config_map.py          TOUS les paramètres du modèle (gate, curriculum, carte, récompense)
│   ├── env_map.py             l'environnement d'entraînement (obs, récompense, reset)
│   ├── mapping.py             construction des cartes égocentriques
│   ├── layouts.py             générateur de configurations d'inventaire + split train/val/test
│   ├── curriculum.py          curriculum ADR bidirectionnel (10 crans)
│   ├── models.py              réseau acteur-critique (CNN sur les cartes + MLP)
│   ├── flatten_wrapper.py     3 drones × B entrepôts → 3B pseudo-envs pour rsl_rl
│   ├── train.py               ENTRAÎNEMENT
│   ├── eval_map.py            évaluation d'un checkpoint à plusieurs niveaux de bruit
│   ├── verify_env.py          validation de l'environnement AVANT entraînement
│   ├── record_traj.py         enregistrement des trajectoires d'une politique
│   └── calibrate_gate.py      calibration des seuils de lecture sur le décodeur réel
└── tests/
    ├── test_swarmscan_map_pure.py   tests sans Isaac (rapides)
    └── test_swarmscan_map_env.py    smoke-test avec Isaac
```

Les résultats d'entraînement vont dans `swarmscan_runs/<date>_<nom_du_run>/`.

---

## 5. Vérifications avant d'entraîner

Faire les trois dans l'ordre. Elles prennent quelques minutes et évitent de perdre des
heures de calcul.

### 5.1 Tests purs (sans Isaac, ~10 s)

```bash
~/isaac5_env/bin/python rl_inventory/tests/test_swarmscan_map_pure.py
```

Attendu, en dernière ligne : `TOUS LES TESTS PURS PASSENT`.

Ils vérifient le mapper (cônes latéraux, crops égocentriques, overlap d'équipe), le
générateur de configurations (split gelé, déterminisme) et le curriculum (montée,
descente, seuils).

### 5.2 Smoke-test Isaac (~3 min)

```bash
bash rl_inventory/launch.sh rl_inventory/tests/test_swarmscan_map_env.py --headless
```

Il monte un vrai environnement à 2 entrepôts et vérifie six choses : dimensions des
observations, récompense, reset, cartes, panne de drone, terminaison.

### 5.3 Validation de l'environnement (~10 min) — étape la plus importante

```bash
bash rl_inventory/launch.sh rl_inventory/swarmscan_map/verify_env.py --headless \
     --policy random
```

Principe : **un environnement sain est inexploitable par le hasard**. Une politique
aléatoire ne doit lire aucun QR au gate nominal. Si elle en lit, l'entraînement apprendra
à exploiter ce défaut au lieu d'apprendre la tâche. C'est ce qui a coûté quatre semaines
sur ce projet.

Trois verdicts s'affichent à la fin, tous doivent indiquer `PASSE` :

| Verdict | Signification |
|---|---|
| `une politique 'random' ne doit RIEN lire au gate nominal (< 0.01)` | pas de lecture gratuite |
| `accélération réaliste hors chocs (p99 < 0.5 g)` | le modèle d'actionneur est actif, le drone a de l'inertie |
| `lectures non issues de survitesses (< 20 %)` | le gate de vitesse n'est pas contourné |

Autres politiques disponibles : `--policy zero` (immobile), `--policy forward` (plein gaz),
`--policy checkpoint --checkpoint <fichier.pt> --sigma 0` (un modèle entraîné).

Utiliser `--level N` pour tester un cran de curriculum précis (par défaut : gate nominal).

---

## 6. Lancer l'entraînement

### 6.1 Commande de référence

C'est celle du run en cours :

```bash
cd ~/simulation_mc02
source ~/isaac5_env/bin/activate
python rl_inventory/swarmscan_map/train.py --headless \
    --num_envs 64 \
    --max_iterations 30000 \
    --run_name barre088_s42 \
    --seed 42 \
    --kit_args="--/rtx/verifyDriverVersion/enabled=false"
```

Version équivalente via le lanceur (recommandée, elle pose les variables d'environnement
elle-même) :

```bash
bash rl_inventory/launch.sh rl_inventory/swarmscan_map/train.py --headless \
     --num_envs 64 --max_iterations 30000 --run_name barre088_s42 --seed 42
```

Pour ne pas perdre le run si le terminal se ferme :

```bash
nohup bash rl_inventory/launch.sh rl_inventory/swarmscan_map/train.py --headless \
      --num_envs 64 --max_iterations 30000 --run_name barre088_s42 --seed 42 \
      > ~/logs/train_barre088.log 2>&1 &
```

### 6.2 Les arguments

| Argument | Défaut | Rôle |
|---|---|---|
| `--num_envs` | 64 | entrepôts simulés en parallèle. Chacun contient 3 drones, donc 64 → 192 agents |
| `--num_steps` | 32 | pas collectés par env et par itération (fenêtre GAE) |
| `--mini_batches` | 1 | mini-lots par époque. À monter si on monte `--num_steps` |
| `--max_iterations` | 5000 | nombre d'itérations PPO |
| `--seed` | 42 | graine |
| `--run_name` | `v2_map` | nom du dossier de sortie |
| `--resume` | aucun | chemin d'un `.pt` à reprendre |
| `--start_level` | 0 | cran de curriculum au démarrage (à remettre manuellement en cas de reprise) |
| `--freeze_level` | -1 | fige le curriculum à ce cran (ni promotion ni recul) |
| `--entropy` | valeur de config (0.01) | coefficient d'entropie initial |
| `--entropy_final` | 0.001 | entropie de la phase de convergence |
| `--converge_level` | 9 | cran qui déclenche la bascule d'entropie |
| `--headless` | — | sans fenêtre. **Toujours le mettre pour entraîner** |

La bascule d'entropie est automatique : quand le curriculum atteint le cran
`--converge_level`, le coefficient passe de 0.01 à `--entropy_final` et le message
`>>> BASCULE CONVERGENCE` s'affiche. Le bruit d'exploration se retire pour que la politique
moyenne apprenne à viser elle-même.

### 6.3 Ce que la commande produit

Un dossier `swarmscan_runs/<AA-MM-JJ_HH-MM-SS>_<run_name>/` contenant :

- `events.out.tfevents.*` : les courbes TensorBoard ;
- `model_0.pt`, `model_200.pt`, `model_400.pt`… : un checkpoint toutes les 200 itérations
  (`save_interval` dans `train.py`), environ 10,8 Mo chacun ;
- `git/` : l'état du dépôt au lancement.

### 6.4 Durée et coût

Mesures du run en cours (`--num_envs 64`, RTX 2060 8 Go) :

| Grandeur | Valeur mesurée |
|---|---|
| Vitesse | ~26 itérations/minute, soit ~2,3 s par itération |
| 6 400 itérations | 4 h 10 |
| 30 000 itérations | ~19 h 30 |
| VRAM occupée | ~7,0 Go sur 8 Go |
| Disque pour 30 000 itérations | 150 checkpoints × 10,8 Mo ≈ 1,6 Go |

Si la VRAM déborde (`CUDA out of memory`), descendre `--num_envs` à 32 ou 16. La variable
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` est déjà posée par `train.py`.

---

## 7. Suivre l'entraînement

### 7.1 TensorBoard

```bash
~/isaac5_env/bin/python -m tensorboard.main --logdir swarmscan_runs --port 6006
```

Puis ouvrir `http://localhost:6006`. Un seul TensorBoard suffit pour tous les runs : ils
apparaissent côte à côte.

### 7.2 Les courbes qui comptent

| Courbe | Ce qu'elle dit |
|---|---|
| `curriculum/level` | le cran actuel, 0 à 10. **C'est l'indicateur principal de progrès.** Il doit monter |
| `curriculum/ema` | moyenne glissante du taux de lecture. Promotion quand elle dépasse 0,88 pendant 25 épisodes |
| `episode/read_frac_gate` | fraction des tags lisibles effectivement lue, au gate du cran actuel |
| `episode/read_frac_nominal` | même mesure mais au gate nominal calibré. **C'est le score qui compte pour le mémoire** |
| `comportement/couverture_cellules` | surface explorée : dit si les drones balaient ou tournent en rond |
| `comportement/vitesse_mps` | vitesse moyenne. Doit descendre sous 0,6 m/s près des racks pour lire |
| `comportement/contacts_pct` | pourcentage du temps en contact avec un obstacle. Doit rester bas |
| `recompense/*` | décomposition de la récompense en 6 postes : `lectures`, `couverture`, `potentiel`, `chocs`, `separation`, `commande`. Dit quelle composante domine |
| `Policy/mean_noise_std` | l'écart-type σ d'exploration. S'il s'effondre tôt (sous 0,15), la politique arrête d'explorer |
| `Loss/entropy` | même signal, côté perte |
| `Perf/total_fps` | débit de simulation, pour comparer des réglages de `--num_envs` |

Règles de lecture :

- `curriculum/level` bloqué à 0 pendant des milliers d'itérations = le modèle n'atteint pas
  la barre de promotion (0,88). C'est exactement la panne qui a produit 34 runs sans
  résultat ; elle est documentée dans `config_map.py`, section `CurriculumConfig`.
- `read_frac_gate` élevé mais `read_frac_nominal` proche de zéro = le modèle exploite la
  tolérance du cran courant et ne transfère pas. C'est le vrai sujet, mesuré par
  `eval_map.py`.

### 7.3 Console

Le script affiche à chaque itération le résumé standard `rsl_rl` (récompense moyenne,
longueur d'épisode, pertes) et, au reset, les messages de promotion/recul du curriculum.

---

## 8. Reprendre, geler, évaluer

### 8.1 Reprendre un entraînement interrompu

`--start_level` n'est **pas** restauré automatiquement par le checkpoint : le noter avant
d'arrêter le run (dernière valeur de `curriculum/level` dans TensorBoard) et le repasser à
la main.

```bash
bash rl_inventory/launch.sh rl_inventory/swarmscan_map/train.py --headless \
     --num_envs 64 --max_iterations 30000 --run_name barre088_reprise --seed 42 \
     --resume swarmscan_runs/26-07-27_13-05-54_barre088_s42/model_6400.pt \
     --start_level 3
```

### 8.2 Geler le curriculum sur un palier

Pour finir un entraînement à conditions constantes, sans promotion ni recul :

```bash
bash rl_inventory/launch.sh rl_inventory/swarmscan_map/train.py --headless \
     --num_envs 64 --max_iterations 5000 --run_name palier_nominal \
     --resume <checkpoint.pt> --freeze_level 10 --entropy 0.001
```

### 8.3 Évaluer un checkpoint

`eval_map.py` mesure la performance à plusieurs niveaux de bruit σ. σ = 0 signifie
politique déterministe (la moyenne du réseau, sans exploration) : c'est le mode qui compte
pour un résultat publiable.

```bash
bash rl_inventory/launch.sh rl_inventory/swarmscan_map/eval_map.py --headless \
     --checkpoint swarmscan_runs/26-07-27_13-05-54_barre088_s42/model_6400.pt \
     --level -1 --sigmas 0,0.20,0.32,0.42 --split test --num_envs 16
```

| Argument | Rôle |
|---|---|
| `--level` | `-1` = gate nominal (défaut). Un entier = ce cran de curriculum |
| `--sigmas` | liste des écarts-types testés, `0` = déterministe |
| `--split` | `train`, `val` ou `test`. Les configurations de test sont gelées et jamais vues à l'entraînement |
| `--waves` | nombre de vagues d'épisodes complets par σ |
| `--spawn_help` | `0` pour retirer les spawns dirigés près d'un tag |

La sortie donne un tableau `σ → fraction lue (gate) / fraction lue (nominal) / récompense`,
puis le meilleur σ pour la tâche nominale.

### 8.4 Voir ce que fait la politique

Les courbes disent combien le drone lit, jamais où il va.

```bash
bash rl_inventory/launch.sh rl_inventory/swarmscan_map/record_traj.py --headless \
     --checkpoint <checkpoint.pt> --level 0 --num_envs 8 --out /tmp/traj.npz
```

Le fichier `.npz` contient positions, caps, vitesses et instants de lecture, à analyser
hors ligne.

### 8.5 Recalibrer le gate de lecture

Uniquement si les caractéristiques de la caméra changent. Rend de vraies images et tente
un décodage réel avec OpenCV :

```bash
PYTHONUNBUFFERED=1 ~/isaac5_env/bin/python rl_inventory/swarmscan_map/calibrate_gate.py \
    --headless --enable_cameras --kit_args="--/rtx/verifyDriverVersion/enabled=false"
```

Sortie : les seuils mesurés + un CSV dans `docs/`.

---

## 9. Dépannage

### 9.1 Isaac Sim ne démarre pas

- `RTX driver version check failed` : le flag `--kit_args="--/rtx/verifyDriverVersion/enabled=false"`
  manque. Passer par `rl_inventory/launch.sh`, qui l'ajoute toujours.
- Blocage sur le EULA : `export OMNI_KIT_ACCEPT_EULA=YES`.
- Premier démarrage très long (>10 min) : normal, l'entrepôt USD se télécharge. Vérifier la
  connexion internet.

### 9.2 Erreurs Python

- `ImportError: cannot import name ... from rsl_rl` : mauvaise version de `rsl_rl`. Il faut
  `rsl-rl-lib==3.0.1`.
- `numpy.dtype size changed` : NumPy 2 installé. Revenir à `numpy==1.26.0`.
- `ModuleNotFoundError: rl_inventory` : lancer depuis la racine du dépôt. Les scripts
  ajoutent eux-mêmes la racine au `sys.path`, mais seulement à partir de leur propre chemin.

### 9.3 Mémoire GPU

`CUDA out of memory` : baisser `--num_envs` (64 → 32 → 16). Fermer TensorBoard, le
navigateur et toute autre application GPU. Vérifier l'occupation avec `nvidia-smi`.

### 9.4 Disque plein

Chaque run produit ~1,6 Go de checkpoints sur 30 000 itérations. Le disque de la machine
est à 91 % d'occupation. Supprimer les runs anciens :

```bash
du -sh swarmscan_runs/* | sort -h        # voir les plus gros
rm -rf swarmscan_runs/<run_a_supprimer>  # après avoir vérifié qu'il ne sert plus
```

### 9.5 Le curriculum ne monte pas

Ce n'est pas un bug d'installation mais un résultat d'entraînement. Vérifier dans l'ordre :

1. `verify_env.py` passe-t-il les trois verdicts ? Sinon l'environnement est exploitable et
   les mesures ne veulent rien dire.
2. `curriculum/ema` progresse-t-elle, même lentement ? Si elle plafonne loin sous 0,88, la
   barre de promotion est hors d'atteinte pour cette configuration.
3. `episode/read_frac_gate` monte-t-elle encore ou a-t-elle atteint un plateau ? Une courbe
   qui monte encore signifie que le modèle n'a pas fini d'apprendre le cran courant.

---

## 10. Récapitulatif : la séquence minimale

```bash
# 1. installation (une seule fois)
python3.11 -m venv ~/isaac5_env
source ~/isaac5_env/bin/activate
pip install --upgrade pip setuptools wheel
export OMNI_KIT_ACCEPT_EULA=YES
pip install "isaacsim[all]==5.1.0"       --extra-index-url https://pypi.nvidia.com
pip install "isaacsim[extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install isaaclab==2.3.2.post1
pip install rsl-rl-lib==3.0.1 numpy==1.26.0 qrcode pillow opencv-python-headless tensorboard pytest

# 2. vérifications
cd ~/simulation_mc02
~/isaac5_env/bin/python rl_inventory/tests/test_swarmscan_map_pure.py
bash rl_inventory/launch.sh rl_inventory/tests/test_swarmscan_map_env.py --headless
bash rl_inventory/launch.sh rl_inventory/swarmscan_map/verify_env.py --headless --policy random

# 3. entraînement
bash rl_inventory/launch.sh rl_inventory/swarmscan_map/train.py --headless \
     --num_envs 64 --max_iterations 30000 --run_name mon_run --seed 42

# 4. suivi
~/isaac5_env/bin/python -m tensorboard.main --logdir swarmscan_runs --port 6006

# 5. évaluation
bash rl_inventory/launch.sh rl_inventory/swarmscan_map/eval_map.py --headless \
     --checkpoint swarmscan_runs/<run>/model_XXXX.pt --sigmas 0,0.20,0.32 --split test
```
