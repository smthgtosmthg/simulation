# SwarmScan-Map — installer, entraîner, suivre

Python 3.11
`rsl-rl-lib` doit être en 3.0.1
`numpy` en 1.26

---

## 1. Installer

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git curl wget \
    python3.11 python3.11-venv python3.11-dev \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6

python3.11 -m venv ~/isaac5_env
source ~/isaac5_env/bin/activate
pip install --upgrade pip setuptools wheel

export OMNI_KIT_ACCEPT_EULA=YES
pip install "isaacsim[all]==5.1.0"       --extra-index-url https://pypi.nvidia.com
pip install "isaacsim[extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install isaaclab==2.3.2.post1

pip install rsl-rl-lib==3.0.1 torch==2.7.0 torchvision==0.22.0 numpy==1.26.0 \
            qrcode pillow opencv-python-headless tensorboard matplotlib pandas scipy pytest
```



Vérifier — la sortie doit afficher `cuda True` :

```bash
~/isaac5_env/bin/python -c "import isaacsim, isaaclab, rsl_rl, torch; print('cuda', torch.cuda.is_available())"
```

Puis les trois contrôles

```bash
~/isaac5_env/bin/python rl_inventory/tests/test_swarmscan_map_pure.py
bash rl_inventory/launch.sh rl_inventory/tests/test_swarmscan_map_env.py --headless
bash rl_inventory/launch.sh rl_inventory/swarmscan_map/verify_env.py --headless --policy random
```

Les trois verdicts de `verify_env.py` doivent indiquer `PASSE`.

Le premier lancement télécharge l'entrepôt USD et génère les 123 QR codes : compter plusieurs
minutes de plus.

---

## 2. Entraîner

```bash
nohup bash rl_inventory/launch.sh rl_inventory/swarmscan_map/train.py --headless \
      --num_envs 64 --max_iterations 30000 --run_name mon_run --seed 42 \
      > ~/logs/train_mon_run.log 2>&1 &
```

| Argument | Défaut | Rôle |
|---|---|---|
| `--num_envs` | 64 | entrepôts en parallèle, 3 drones chacun (64 → 192 agents) |
| `--max_iterations` | 5000 | itérations PPO |
| `--run_name` | `v2_map` | nom du dossier de sortie |
| `--seed` | 42 | graine |
| `--resume` | — | checkpoint `.pt` à reprendre |
| `--start_level` | 0 | cran de curriculum au démarrage |
| `--freeze_level` | -1 | fige le curriculum, ni promotion ni recul |
| `--headless` | — | sans fenêtre, toujours le mettre |

Sortie : `swarmscan_runs/<date>_<run_name>/`, un checkpoint toutes les 200 itérations.



Reprise :

```bash
bash rl_inventory/launch.sh rl_inventory/swarmscan_map/train.py --headless \
     --num_envs 64 --max_iterations 30000 --run_name reprise --seed 42 \
     --resume swarmscan_runs/<run>/model_6400.pt --start_level 3
```

---

## 3. Suivre

```bash
~/isaac5_env/bin/python -m tensorboard.main --logdir swarmscan_runs --port 6006
```

Ouvrir `http://localhost:6006`. Tous les runs apparaissent côte à côte.

| Courbe | Ce qu'elle dit |
|---|---|
| `curriculum/level` | cran actuel 0→10, indicateur principal de progrès, doit monter |
| `curriculum/ema` | taux de lecture lissé ; promotion si > 0,88 tenue 25 épisodes |
| `episode/read_frac_gate` | fraction lue au gate du cran courant |
| `episode/read_frac_nominal` | fraction lue au gate nominal, c'est le score qui compte |
| `comportement/couverture_cellules` | surface explorée |
| `comportement/vitesse_mps` | doit descendre sous 0,6 m/s près des racks pour lire |
| `comportement/contacts_pct` | temps en contact avec un obstacle, doit rester bas |
| `Policy/mean_noise_std` | σ d'exploration ; sous 0,15 trop tôt, la politique cesse d'explorer |


Évaluer un checkpoint (σ = 0 : politique déterministe, split de test jamais vu à l'entraînement) :

```bash
bash rl_inventory/launch.sh rl_inventory/swarmscan_map/eval_map.py --headless \
     --checkpoint swarmscan_runs/<run>/model_XXXX.pt --sigmas 0,0.20,0.32 --split test
```
