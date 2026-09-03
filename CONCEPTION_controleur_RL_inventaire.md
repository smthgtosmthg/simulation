# Conception & Plan d'Implémentation
## Contrôleur RL pour inspection d'inventaire QR par drone — Isaac Lab

> Document de référence pour l'implémentation. Calé sur la scène entrepôt, le LiDAR,
> la caméra et la logique QR existants, et sur les références du projet.

---

## Principe d'implémentation (à respecter partout)

**Tout doit être fait proprement et fidèlement — aucun raccourci, rien de bâclé.** La qualité
prime sur la vitesse. En particulier :
- Le **baseline Pore** est une réimplémentation **fidèle** (waypoints + champ de risque réglés
  comme dans le papier), pas une version affaiblie : un baseline faible invaliderait toute la
  comparaison.
- Les **tags QR** sont **attachés correctement** à chaque box (cf. §4), pas posés comme des
  panneaux flottants.
- Chaque brique (env, capteurs, récompense, métriques) est implémentée complètement, testée et
  conforme à cette spec avant de passer à la suivante.

---

## 0. Objectif & contribution

**But.** Un contrôleur **appris** qui fait tout — naviguer **et** décider où aller pour lire
tous les QR — et qui bat le contrôleur **classique** de Pore et al. (2026) dans **le même**
environnement.

**Contribution.** Amener un **World Model (DreamerV3)** — avec **PPO** comme contrôleur appris
de référence — sur la tâche d'**inspection d'inventaire QR**, vue comme de la **perception
active** : bouger pour *lire* les tags, **sans carte ni positions connues**, en GPS-denied,
avec une **comparaison équitable** contre la référence du domaine. C'est une contribution de
type « application à un domaine nouveau + évaluation solide ».

**Ce qu'on bat / ce qu'on rapporte.**
- **On bat** : **temps de mission** (à couverture égale) et **adaptabilité** (layout inconnu,
  obstacles mobiles).
- **On rapporte sans chercher à battre** : précision QR % (c'est de la perception/décodage,
  même caméra = même score), latence (réseau), erreur de trajectoire RMS (un chemin fixe gagne
  par construction).

---

## 1. Principe : une arène, plusieurs cerveaux

```
                          ┌──────────────────────────────────────┐
                          │            CERVEAU (au choix)          │
                          │  ┌─────────┐  ┌──────┐  ┌──────────┐  │
                          │  │  Pore   │  │ PPO  │  │ Dreamer  │  │
                          │  │ scripté │  │policy│  │ V3 policy│  │
                          │  └─────────┘  └──────┘  └──────────┘  │
                          └───────▲───────────────────┬──────────┘
                       observation│                   │action (vitesse)
                                  │                   ▼
        ┌─────────────────────────┴───────────────────────────────────┐
        │                   ENVIRONNEMENT (l'arène)                      │
        │   Isaac Lab : scène entrepôt + drone + LiDAR + caméra          │
        │   - applique l'action (commande de vitesse)                    │
        │   - avance la physique                                         │
        │   - construit l'observation                                    │
        │   - calcule la récompense                                      │
        │   - décide la fin d'épisode                                    │
        └────────────────────────────────────────────────────────────────┘
```

**Identique pour les 3 cerveaux** : le monde, l'action, la récompense, les métriques.
**Diffère selon le cerveau** : l'**observation**. Pore et PPO lisent le **LiDAR + l'état** ;
Dreamer lit l'**image caméra**. La comparaison porte sur la **tâche** (même monde, même
récompense, mêmes métriques), pas sur des entrées identiques — chaque méthode utilise l'entrée
qui lui est naturelle.

---

## 2. Ce qu'on construit

**On garde / porte dans Isaac Lab** : la scène entrepôt (USD), le drone (piloté en vitesse),
le LiDAR 3D, la caméra + la logique QR (réutilisée pour l'évaluation).

**On écrit :**

| Fichier | Rôle |
|---|---|
| `env.py` | L'arène : observation, action, fin d'épisode, récompense, reset, step. |
| `qr_task.py` | Pose les N tags QR, suit lesquels sont lus, modèle « QR lu ». |
| `controllers/pore_baseline.py` | Le baseline scripté (waypoints + champ de risque). |
| `controllers/ppo_policy.py` | PPO (réseau + entraînement). |
| `controllers/dreamer_policy.py` | DreamerV3 (wrapper). |
| `metrics.py` | Couverture %, temps mission, collisions, distance mini, sécurité, fluidité. |
| `train_ppo.py` / `train_dreamer.py` / `eval.py` | Entraîner / évaluer n'importe quel cerveau. |
| `config_rl.py` | Paramètres (taille env, LiDAR, bornes, récompense…). |

---

## 3. L'environnement (POMDP)

Le drone ne voit pas tout l'entrepôt d'un coup → c'est un **POMDP** : observation partielle,
action, récompense.

### 3.1 Observation

**Commune (Pore + PPO) :**
- **LiDAR** : une **grille de profondeur azimut × élévation**. **360 colonnes** (une mesure par
  degré, tour complet) × **5 bandes de hauteur**. Chaque case = la distance à l'obstacle le plus
  proche dans **cette direction et cette hauteur**. (C'est une petite « image de profondeur »
  autour du drone : direction et hauteur restent liées dans la même case → pas de trou en
  diagonale.)
- **État drone** : vitesse, cap (yaw), altitude.
- **Voisins** : positions relatives des drones proches (partagées par la comm) → coordination
  et évitement.
- **Mémoire QR** : combien / quels IDs de tags ont **déjà** été décodés. (Aucune position de tag
  — cf. §4.)

**Dreamer, en plus :** l'**image caméra** (profondeur et/ou RGB) — c'est ce qui lui permet
d'exploiter des indices visuels.

### 3.2 Action

**Commande de vitesse continue**, en repère drone, bornée : `a = (vx, vy, vz, yaw_rate)`.
Niveau **vitesse** (le sim suit la consigne), pas les moteurs : la tâche est de la
navigation / perception active, pas de la course agile.

**L'altitude (`vz`) est libre** entre un minimum et un maximum : les tags sont sur des étagères
à **plusieurs hauteurs**, le drone doit **monter et descendre** pour tout scanner.

### 3.3 Fin d'épisode

Le **premier** de :
- **succès** : tous les QR lus (ou couverture cible atteinte) ;
- **collision** : contact avec un mur / rack **ou** un autre drone ;
- **timeout** : nombre max de pas atteint.

### 3.4 Récompense (gabarit de départ)

| Composante | Valeur de départ | Quand |
|---|---|---|
| Nouveau QR lu | `+10` | un tag pas encore lu devient lu |
| Mission complète | `+100` | tous les QR lus |
| Collision | `−50` (+ fin) | contact obstacle / drone |
| Pénalité temps | `−0.5` / pas | pousse à finir vite |
| Pénalité à-coups | `−λ·‖Δaction‖²` | fluidité (λ petit) |
| *(option)* Proximité obstacle | `−petit` | rester à distance de sécurité |

Les pénalités temps + collision + à-coups évitent les comportements dégénérés (tourner en rond,
sur-place). Valeurs à calibrer.

---

## 4. Le modèle « QR lu »

L'agent n'a **ni carte ni positions de tags**. Il **explore** ; la caméra décode les tags qu'il
croise ; la couverture monte ; il retient les **IDs déjà décodés**. C'est de la vraie perception
active dans l'inconnu.

- **L'environnement** connaît les vraies positions des tags — **uniquement** pour calculer la
  récompense. L'agent ne les voit jamais.
- **À l'entraînement** (rapide, parallèle) : un tag est compté **lu** via un **proxy géométrique**
  — dans le champ de vue de la caméra **+** assez proche **+** sous un bon angle **+** ligne de
  vue dégagée. Rapide et parallélisable.
- **À l'évaluation** (réaliste) : le **vrai décodeur** caméra (pyzbar / opencv) mesure la
  précision QR réelle.

**Placement des tags (important).** Chaque box reçoit un **composant QR** rattaché à **son propre
prim** : le tag est posé à plat sur la face de la box et **bouge avec elle** (donc compatible avec
le layout randomisé). Pas de panneau flottant placé à une coordonnée absolue. Chaque box a un
**ID unique** ; l'environnement garde la correspondance **box ↔ ID ↔ position** de son côté (pour
le calcul de la récompense). Pour un décodage fiable : contraste noir/blanc franc, taille
raisonnable. *(Tâche d'implémentation pour Claude Code, au montage de la scène.)*

---

## 5. L'entrepôt & les tags

**Forme du monde** (identique pour tous les cerveaux) : espace intérieur rectangulaire
(~30 × 20 m), rangées d'étagères formant des **allées**, tags QR sur les faces des racks, à
**plusieurs hauteurs**.

**Régime :** layout **randomisé** (positions des racks/tags qui changent par épisode) **+
obstacles mobiles** à l'entraînement. **Évaluation sur les deux** : statique (cas favorable au
classique) **et** stress (randomisé + obstacles mobiles).

---

## 6. Les trois cerveaux

### 6.1 Pore (baseline)

Réimplémentation fidèle du contrôleur de Pore, **dans l'arène**, qui sort une **vitesse**
(donc directement comparable).

1. **Waypoints de scan** : un zig-zag d'arrêts devant les racks (chemin nominal).
2. **Suivi** : un contrôleur P transforme l'erreur `(waypoint − position)` en vitesse (le sim
   lisse le vol entre points).
3. **Champ de risque** : distance au plus proche `r` = min du LiDAR ;
   `Rf = sigmoïde(β·(r_safe − r))`, avec `r_safe = 0.5 m`, `β ∈ [8,12]`, seuil `τ = 0.4`.
4. **Sécurité** : si `Rf > τ` → ralentir et/ou dévier latéralement loin de l'obstacle.
5. **Retour base** : simplifié à « mission finie » en simu.

Le baseline **reçoit la carte** (c'est sa prémisse) ; le contrôleur appris, non. Comparaison :
même quand le classique a la carte gratuitement, l'appris gagne sur le temps et sur les
obstacles mobiles (que le classique ne gère pas).

### 6.2 PPO (contrôleur appris de référence)

- **Observation** : la version LiDAR + état (§3.1). **Action** : vitesse continue.
- **Réseau** : MLP, politique gaussienne (continu).
- **Mécanisme QR** : PPO ne voit pas l'image. C'est la **récompense** qui fait le travail :
  l'environnement détecte un tag lu (proxy §4) → `+récompense` et le compteur de QR monte. À
  force, PPO apprend à **voler et orienter le drone** pour que la caméra balaie les racks. Comme
  l'action inclut `yaw_rate`, il apprend à **viser** en tournant. C'est de la perception active
  apprise par le signal de récompense.
- **Hyperparams de départ** (à régler) : ~4096 envs parallèles, rollout 16–32 pas,
  `lr ≈ 3e-4`, `γ ≈ 0.99`, GAE `λ ≈ 0.95`, clip `0.2`.

### 6.3 Dreamer (contribution principale)

- **Observation** : image caméra + proprioception. C'est là que le world model paie.
- **Algo** : DreamerV3 ; wrapper custom (intégration non fournie d'office).
- **Idée** : il apprend un « simulateur mental » (RSSM) et planifie en imaginant → plus
  sample-efficient, et la perception active peut émerger. Là où PPO apprend à l'aveugle via la
  récompense, Dreamer **voit** et apprend à **regarder intelligemment** (utiliser un bord de
  rack visible pour deviner où sont les tags).

---

## 7. Métriques & évaluation

**Rapportées (familles de Pore) :** précision QR %, latence, erreur trajectoire RMS, **distance
mini** entre drones, **risk factor max Rf**, **% du temps en zone sûre (Rf ≤ τ)**, lectures en
double.

**Sur lesquelles on bat :** **temps de mission**, **réduction du temps en multi-drone**,
**couverture en stress**, **collisions avec obstacles mobiles**.

**Ajoutées :** **taux de réussite** (% d'épisodes : tous les QR lus + sans collision),
**efficacité d'exploration** (QR trouvés par mètre parcouru), **écart de robustesse** (perf en
statique vs randomisé).

**Protocole :** 2 régimes (statique / stress), **N graines aléatoires** (moyenne ± écart-type),
même monde, mêmes tags, même budget de temps pour les 3 cerveaux.

---

## 8. Communication entre drones

- **À l'entraînement** : un **modèle de comm léger** — délai et perte de message tirés
  aléatoirement. Garde la vitesse d'entraînement et rend la politique **robuste** aux
  imperfections réseau.
- **À l'évaluation** : **NS3 réel** (réseau simulé) — les messages passent vraiment, avec les
  vrais délais et pertes. C'est là que la réalité réseau compte.
- **Sans aucune comm** : le **LiDAR voit déjà les autres drones** comme obstacles → l'évitement
  de base fonctionne. La comm ajoute : connaître un coéquipier **hors champ** et **se répartir le
  travail**.

---

## 9. Entraînement

- **Parallèle + headless** : beaucoup d'envs en parallèle dans Isaac Lab.
- **Curriculum** : commencer facile (peu de tags, pas d'obstacle mobile), augmenter la
  difficulté → apprentissage stable.
- **Compute** : un GPU. PPO = heures ; Dreamer = plus long.
- **Checkpoints** + courbes (récompense, couverture, collisions).
- **Multi-drone dès le début** (un test rapide à un seul drone sert juste à valider la
  plomberie de l'env avant d'ajouter la coordination).

---

## 10. Structure du repo

```
rl_inventory/
├── env.py
├── qr_task.py
├── metrics.py
├── controllers/
│   ├── pore_baseline.py
│   ├── ppo_policy.py
│   └── dreamer_policy.py
├── train_ppo.py
├── train_dreamer.py
├── eval.py
├── config_rl.py
└── assets/                 # USD entrepôt, images QR
```

---

## 11. Paramètres de départ (à régler en codant)

- **Grille LiDAR** : 360 colonnes × 5 bandes de hauteur.
- **Bornes altitude** : min / max couvrant la hauteur des étagères.
- **Récompense** : `+10`/QR, `+100` mission, `−50` collision, `−0.5`/pas, `−λ` à-coups.
- **Proxy QR** : distance max, angle max, durée de visée — à caler pour coller au vrai décodeur.
- **N tags** et **nombre de racks** : taille de la mission.

---

## 12. Transmission à Claude Code

Fournir : **ce document** + le **repo** (scène, LiDAR, caméra/QR, bridge NS3) + la **doc
Isaac Lab**. **Pas** les articles de recherche (ils servent à la rédaction, pas au code). Les
hyperparamètres DreamerV3 se prennent dans le **repo** DreamerV3.

---

## 13. Risques & garde-fous

1. **Baseline fidèle** : un Pore réimplémenté trop faible ne prouve rien → l'implémenter
   correctement (waypoints + risque réglés comme dans le papier) et vérifier des perfs crédibles
   en statique.
2. **Proxy « QR lu » calibré** : seuils géométriques alignés sur le vrai décodeur.
3. **Récompense itérée** : ajuster le shaping si comportements dégénérés.
4. **Dreamer long / capricieux** : PPO passe d'abord et donne déjà des résultats.
5. **Étude en simulation** : comparaison dans le même sim ; le hardware reste une extension
   future.

---

*Implémentation en commençant par `env.py` (l'arène), puis les cerveaux.*
