# Conception finale — SwarmScan-Map : inventaire QR par essaim de drones, 100 % RL, sans positions de cibles

**Date** : 8 juillet 2026
**Statut** : conception validée par revue de littérature (9 axes, ~60 papiers, deux passes de recherche + critique adversariale). Remplace la conception « entités QR dans l'observation » (jugée oracle) et corrige la formulation qui a produit le 0 % de l'IPPO.
**Papier à battre** : Pore, Patle & Thorat, *UAV-Based QR Code Scanning and Inventory Synchronization System with Safe Trajectory Planning*, **Symmetry (MDPI) 2026, 18(4):548**, DOI [10.3390/sym18040548](https://doi.org/10.3390/sym18040548) — publié le 24/03/2026, 0 citation à ce jour.

---

## Table des matières

1. [Réponses directes à tes questions](#1-réponses)
2. [La règle d'or : où l'information QR a le droit d'exister](#2-règle-dor)
3. [La formulation correcte du problème](#3-formulation)
4. [L'architecture SwarmScan-Map](#4-architecture)
5. [La récompense](#5-récompense)
6. [Le protocole d'entraînement](#6-entraînement)
7. [Le protocole d'évaluation (comment on teste, proprement)](#7-évaluation)
8. [Pore et al. 2026 : forces, faiblesses, et comment on le bat](#8-pore)
9. [La contribution du PFE (formulation défendable)](#9-contribution)
10. [Pourquoi ça va marcher : correspondance causes d'échec → corrections prouvées](#10-pourquoi)
11. [Risques et mitigations](#11-risques)
12. [Plan d'exécution (phases, jalons go/no-go)](#12-plan)
13. [Place de Dreamer dans le PFE](#13-dreamer)
14. [Note sur le code supprimé le 7 juillet](#14-code)
15. [Bibliographie commentée](#15-biblio)

---

<a name="1-réponses"></a>
## 1. Réponses directes à tes questions

**« Le problème, c'est qu'on n'a pas donné d'infos sur les QR ? Ou le modèle en gros ? »**
Les deux, mais pas comme tu le penses. Le rapport d'échec l'a prouvé : une politique ne peut pas apprendre à aller vers une cible qu'elle ne perçoit pas ET dont l'événement de succès est improbable par hasard (~10⁻⁵). Mais la solution n'est **pas** de donner les positions des QR à l'agent (ça, c'est l'oracle qu'on a rejeté). La solution, validée par toute la littérature ObjectNav/exploration (ANS ICLR 2020, SemExp NeurIPS 2020, Jonnarth ICML 2024, GLEAM ICCV 2025), est de donner à l'agent **sa propre mémoire spatiale** : une carte de ce que *lui et ses coéquipiers ont déjà vu et scanné*, construite en ligne à partir de leurs capteurs. Cette carte ne contient **aucune position de QR non découvert** — ce n'est pas de la triche, c'est la représentation standard du domaine. Ton IPPO avait 1800 rayons bruts et **un seul scalaire** de couverture : la littérature dit explicitement que cette configuration échoue (information spatiale non structurée, aucune mémoire).

**« Est-ce que ça se fait d'entraîner sur un environnement puis tester sur le même ? »**
**Non.** Cobbe et al. (ICML 2019, CoinRun) l'écrivent noir sur blanc : tester sur l'environnement d'entraînement « n'offre que peu d'insight sur la capacité de généralisation », et Kirk et al. (JAIR 2023, la survey de référence) recommandent comme protocole standard : entraîner sur un **ensemble** de configurations, tester sur des configurations **jamais vues** (held-out). Une nuance importante joue pour nous : garder le même **bâtiment** (racks fixes — ta décision D2) est défendable si on le dit explicitement, parce que c'est le scénario de déploiement réel (Verity cartographie chaque site avant d'opérer). Ce qui doit changer entre entraînement et test, c'est la **configuration d'inventaire** : quels emplacements portent des cartons, où sont les QR, d'où partent les drones, où sont les obstacles. C'est le « contexte » au sens de Kirk. Protocole : génération procédurale de configurations, split train / validation / test **gelé**, sélection du checkpoint sur validation uniquement, test utilisé une seule fois.

**« Mes tests de résilience (QR déplacés/ajoutés/supprimés, panne drone, obstacles dynamiques), ça suffit ? »**
C'est un **bon squelette** — la panne d'agent en cours d'épisode est étudiée telle quelle dans la littérature (MAANS ECCV 2022, ACE AAAI 2023, Jiang et al. Aerospace 2024), et injecter obstacles dynamiques/permutation de layout en sim est une pratique établie (Tejero-Ruiz 2024). Mais il manque 6 choses pour que ce soit irréprochable :
1. l'axe **configurations jamais vues** (le held-out ci-dessus) — déplacer des QR sur le layout d'entraînement ne teste pas la généralisation ;
2. des **intensités graduées** par axe (panne à t=25/50/75 % ; 1 puis 2 drones ; 1, 2, 4 obstacles) → courbes de dégradation, pas un chiffre unique ;
3. un axe **bruit capteur** (bruit gaussien + dropout de rayons lidar) ;
4. un axe **taille d'équipe** (entraîné à 3, testé à 2 et 4 — « N-agent ad hoc teamwork », NeurIPS 2024) ;
5. les **baselines soumises aux mêmes perturbations** (c'est là que le planner type Pore s'effondre et que le RL gagne — sans baseline perturbée, pas de preuve) ;
6. le **traitement statistique** : ≥3 seeds d'entraînement (5 si possible), ≥20-100 épisodes par cellule, IQM + intervalles bootstrap 95 % (rliable, Agarwal et al. NeurIPS 2021), seeds d'éval identiques pour toutes les méthodes.

**« Je ne peux pas donner leur place à l'agent. »**
Exact — et c'est la contrainte qui rend ton sujet **publiable**. Mais il faut la formuler précisément, sinon on s'interdit l'outil qui rend la tâche apprenable. Voir la règle d'or, section 2 — c'est LA décision d'arbitrage de cette conception.

---

<a name="2-règle-dor"></a>
## 2. La règle d'or : où l'information QR a le droit d'exister

La contrainte scientifique porte sur ce que la politique **déployée** perçoit. Elle ne porte ni sur la récompense d'entraînement, ni sur le critique. C'est une distinction standard et publiée :

| Endroit | Positions des QR autorisées ? | Justification littérature |
|---|---|---|
| **Observation de l'acteur** (ce que le drone voit en mission) | **NON, jamais.** Ni positions, ni directions, ni distances des QR non lus. | C'est la définition même de la tâche ObjectNav : « la carte ne contient que ce que les capteurs ont vu » (ANS, SemExp). L'ancien système à ~98 % violait ça → oracle, à juste titre rejeté. |
| **Mémoire propre** (carte de couverture accumulée, positions des QR **déjà lus** par l'essaim) | **OUI.** | Une fois qu'un drone a lu un QR, il sait où il l'a lu. La carte accumulée est la mémoire de l'agent, pas un oracle. Représentation standard de tout le domaine 2020-2026. |
| **Récompense (entraînement seulement)** | **OUI.** Le simulateur utilise les positions pour calculer le shaping et détecter l'événement « lu ». | SemExp (NeurIPS 2020) entraîne avec « decrease in distance to nearest goal object » alors que la position n'est **jamais** dans l'obs. Pratique universelle. |
| **Critique (entraînement seulement, CTDE)** | **OUI** (critique asymétrique/privilégié). | D-VAT (RA-L 2024), SimpleFlight (RA-L 2025) : le critique voit des infos privilégiées, l'acteur non. Standard du sim-to-real. |

**En mission (déploiement/éval), la politique n'utilise QUE : lidar + état propre + positions relatives des coéquipiers + sa carte construite en ligne.** Rien d'autre. C'est cette phrase qu'on écrit dans le mémoire, et on ajoute une **ablation** (entraînement sans shaping privilégié) pour montrer que le shaping accélère l'entraînement mais que la politique finale n'en dépend pas au déploiement.

---

<a name="3-formulation"></a>
## 3. La formulation correcte du problème

### 3.1. Ce qu'est vraiment l'inventaire, en langage RL

C'est un **POMDP de couverture orientée perception** : il faut que, à la fin de la mission, chaque **face de rack susceptible de porter un QR** ait été observée dans des conditions qui permettent le décodage (distance, angle, vitesse). Trouver les QR n'est pas un problème séparé de la couverture : **si tu couvres toutes les faces de racks dans le cône de lecture, tu as nécessairement lu tous les QR**. C'est le déverrouillage conceptuel :

> On ne récompense pas « trouver un QR invisible » (improbable, inapprenable).
> On récompense « couvrir de la surface de rack en conditions de lecture » (dense, mesurable à chaque pas, apprenable dès le premier épisode) — et la lecture des QR en découle mécaniquement.

La définition de « cellule couverte » est donc **calquée sur le gate de lecture** : une cellule de façade de rack est couverte quand un drone l'a eue dans son cône caméra, à distance ≤ 3 m, sous incidence ≤ 35°, à vitesse ≤ 0,6 m/s (les seuils exacts = ceux du proxy, calibrés — §6.4). Le risque classique « 95 % de surface vue mais 40 % de QR lus » (couverture surfacique naïve) disparaît par construction, puisque couvrir une cellule = remplir les conditions de lecture dessus. C'est exactement le « face-mesh coverage reward » de Kulkarni et al. (NTNU, 2025) — le papier le plus proche de notre tâche : inspection end-to-end par drone, actions (vx, vy, vz, yaw_rate) continues comme nous, récompense = fraction des faces du mesh vues à bonne distance, 97 % de couverture contre 85 % pour FUEL, entraîné en 6 h sur une A6000, transféré sur un vrai quadrotor.

### 3.2. La mémoire : externalisée dans une carte, pas dans un GRU

Le POMDP exige une mémoire (où suis-je déjà passé ? qu'ai-je déjà scanné ?). Deux options : récurrence (GRU) ou mémoire spatiale externalisée (carte). Le verdict de la littérature est sans appel : **la carte d'abord, le GRU en complément éventuel**. PyroTrack (2024) montre que l'entrée « belief map » bat l'entrée « observations brutes » ; Jonnarth (ICML 2024) prouve par ablation que le canal frontières fait passer la couverture de 72,6 % à 97,8 % ; le recurrent PPO seul sur obs brutes est précisément la configuration qui plafonne. La carte a un autre avantage décisif : elle **se fusionne** entre drones (max-pooling des cartes individuelles — MAANS), ce qui donne la coordination et la résilience à la panne quasiment gratuitement.

### 3.3. Le multi-agent

3 drones homogènes, politique **partagée** (PS-PPO — un seul cerveau, 3× les données, ce que le YAML skrl prétendait faire et ne faisait pas), exécution décentralisée, critique central privilégié (CTDE). L'anti-agglutinement ne vient pas d'une partition imposée (interdite par ta contrainte 100 % RL) mais de la récompense : la **contribution marginale** (l'aire que TU as apportée à l'équipe — un suiveur ne gagne rien) + une pénalité d'overlap programmée. C'est la recette de MAANS, dont la division du travail **émerge** (leur métrique Mutual Overlap : 0,42-0,54, comparable à une partition Voronoi imposée). Note : c'est exactement ton idée D_k (difference reward contrefactuel) — la littérature la valide, implémentée géométriquement sur la carte au lieu du potentiel Φ.

---

<a name="4-architecture"></a>
## 4. L'architecture SwarmScan-Map

### 4.1. Observation par drone (~ carte + vecteur, ZÉRO position de QR non lu)

**Bloc A — cartes égocentriques multi-échelles** (le cœur ; recette Jonnarth ICML 2024 : 4 échelles × 32×32, chaque échelle couvrant 4× la surface de la précédente ; l'agent au centre, orienté avec son cap) :

| Canal | Contenu | Source |
|---|---|---|
| 1 | Occupation (obstacles vus par lidar, accumulés) | lidar propre + fusion essaim |
| 2 | Couverture-scan des façades (cellules de face de rack couvertes en conditions de lecture, fusion essaim) | calcul embarqué (§3.1) |
| 3 | Frontières (bord connu/inconnu de la couverture) | dérivé du canal 2 — **canal critique** (ablation Jonnarth : +25 pts) |
| 4 | QR **déjà lus** par l'essaim (positions mémorisées à la lecture) | mémoire propre |
| 5 | Trajectoires récentes de MOI (décroissance exponentielle) | mémoire propre |
| 6 | Positions/trajectoires des coéquipiers vivants | comm essaim |

En 2.5D pour les racks : les canaux 2-3 sont répétés par **niveau de hauteur** (2-3 bandes d'altitude), ou une grille par plan vertical d'allée — à trancher en phase de proto (P1), c'est le seul morceau de représentation non standard (aucun papier n'a publié la couverture RL de façades multi-niveaux : c'est un des points de nouveauté, §9).

**Bloc B — vecteur bas niveau** : lidar réactif **égocentrique** (correction du bug monde/cap). Important : le capteur **reste 360×5 = 1800 rayons pleine résolution** — il alimente la carte d'occupation (canal 1) à pleine résolution. Ce qui est compacté, c'est uniquement l'entrée *vecteur* de la politique : min par secteur (36-72 secteurs × min des 5 anneaux), c'est-à-dire « la distance de l'obstacle le plus proche dans chaque direction » — exactement l'information dont l'évitement a besoin. Option si besoin de plus de fidélité : CNN 1D circulaire sur les 360 azimuts au lieu du min-pool. S'ajoutent : état propre (vitesses, sin/cos yaw, altitude), 2 coéquipiers en relatif + bit vivant/mort, flags du gate (« suis-je en condition de lecture là, maintenant » — vitesse OK, etc., calculables sans savoir où sont les QR), compteur de temps restant.

**Encodeur** : petit CNN partagé sur les cartes (à la MAANS/ACE) + MLP sur le vecteur + attention sur les entités coéquipiers → torse commun acteur/critique (rsl_rl, obs_groups). GRU optionnel en ablation (rsl_rl `ActorCriticRecurrent` ; POPGym : GRU = meilleure mémoire généraliste ; notre horizon court ne justifie pas un transformer).

**Ce qui disparaît par rapport aux deux tentatives précédentes** : les 1800 rayons bruts *en entrée directe du MLP* (IPPO) et les 16 entités QR oracle (v1). **Code : départ à zéro** (décision utilisatrice) — nouveau module dans `rl_inventory/`, aucun fichier ancien réutilisé ; seules les *techniques* validées sont ré-implémentées à neuf : raycast statique ~50×, gate de lecture 6 conditions (devient le critère de couverture), contrôle vitesse, ADR, dropout de drone.

### 4.2. Action

Inchangée : (vx, vy, vz, yaw_rate) continues dans [−1,1], 100 % RL, aucun planner nulle part. La politique EST le contrôleur. On assume dans le mémoire que c'est **plus dur** que l'état de l'art (ANS/SemExp/MAANS gardent un planner local sous la politique RL) — c'est un choix scientifique revendiqué, compensé par : carte en obs, curriculum, horizon de commande court.

### 4.3. Anti-saturation (verrouillé une fois pour toutes)

Gaussienne rsl_rl avec **σ initial ≈ 0,5**, gain faible sur la dernière couche, **pénalité d'action-difference** ‖aₜ−aₜ₋₁‖² calculée **pré-clamp** (SimpleFlight RA-L 2025 : meilleure que le clipping dur et que le filtre passe-bas, sur Crazyflie), entropie ≤ 0,005, monitoring de la **fraction d'actions clippées** comme alarme. Beta/tanh gardés en ablation seulement (la littérature ne montre pas de gain systématique ; Seyde NeurIPS 2021 : la saturation de l'IPPO était le symptôme de l'absence de gradient, pas la cause).

---

<a name="5-récompense"></a>
## 5. La récompense

Par drone k, par pas (coefficients à calibrer en P1, ordres de grandeur littérature) :

```
r_k =  α · ΔCouvertureÉquipe                 (dense, normalisé par le max atteignable/pas — Jonnarth)
     + β · ΔCouvertureMarginale_k             (l'aire que k a apportée SEUL à l'équipe — MAANS 0.02·ΔArea_k ;
                                               c'est le D_k contrefactuel, version géométrique)
     + 10 · (nouveau QR lu, crédité à k, one-shot par QR)
     + shaping privilégié POTENTIEL vers le gate     [ENTRAÎNEMENT SEULEMENT, γΦ(s')−Φ(s), signé :
        Φ = score continu ∈[0,1] du meilleur candidat (distance, incidence, FOV, vitesse) —
        s'éloigner COÛTE ; jamais de max(0,·) ; Ng et al. 1999]
     − pénalité d'overlap programmée          (MAANS : −0.01·ΔOverlap si couverture<90 %, dégressif)
     − collision graduée (barrière progressive, JAMAIS terminale)
     − c₁ · ‖aₜ−aₜ₋₁‖²  (pré-clamp)
     + bonus par paliers de couverture équipe (50/75/90 %) — remplace la pénalité de temps constante
```

Pourquoi chaque terme, en une ligne :
- **ΔCouverture** est le signal dense qui existe **dès le premier pas d'une politique aléatoire** (bouger = voir du nouveau) — c'est ce qui tue la cause n°2 (événement inéchantillonnable). ANS : ce reward seul suffit à apprendre l'exploration en 10M frames.
- **Contribution marginale** = anti-agglutinement + assignation de crédit (ta contribution D_k, ancrée dans MAANS/difference rewards).
- Le **+10 par QR** reste l'objectif final ; il devient atteignable parce que la couverture amène mécaniquement les drones en position de lecture.
- Le **shaping potentiel privilégié** accélère la découverte du gate (landing/docking : c'est la structure qui atteint 0,44 m de précision d'atterrissage dans la littérature drone) ; il disparaît à l'éval, et l'ablation sans lui est planifiée.
- **Pas de pénalité de temps constante** (cause n°4 : −0,1/pas rendait l'approche perdante) ; γ et les paliers font le travail.

Bonus épisodique = déjà couvert : Henaff (ICML 2023) montre que dans un MDP contextuel (layouts randomisés), les bonus **épisodiques** battent les bonus globaux type RND — or notre ΔCouverture **est** un bonus épisodique aligné tâche. Pas de RND/ICM/E3B en composant central (contre-indiqué : ne cible pas une conjonction de conditions ; détachement en MARL).

---

<a name="6-entraînement"></a>
## 6. Le protocole d'entraînement

### 6.1. Génération procédurale des configurations (le « contexte » Kirk)

À chaque reset, tirage aléatoire de : occupation des emplacements (quels slots portent des cartons, 60-100 %), placement/orientation des QR sur les faces, poses de spawn des drones (reverse curriculum §6.3), masque initial de tags « déjà lus » (0-70 %), obstacles statiques additionnels, (plus tard) obstacles dynamiques, bruit capteurs, perturbations de dynamique (±20-30 % masse/poussée — SimpleFlight/DR standard). Le bâtiment (racks/murs) reste fixe — assumé et justifié (déploiement par site ; le raycast statique l'impose aussi techniquement). **Split** : ~200-500 configurations d'entraînement re-tirées à l'infini, 10-20 de validation, 10-20 de test gelées. Si les autres assets entrepôt NVIDIA (full_warehouse, warehouse_with_forklifts) se chargent proprement, l'un d'eux devient le test **hors distribution** (zéro-shot, à la ProcTHOR/GLEAM) — puissant mais optionnel.

### 6.2. Curriculum ADR sur le gate de lecture (cause n°2, deuxième étage)

Mécanisme OpenAI ADR / DORAEMON (ICLR 2024) : chaque seuil du gate a une borne pilotée par le **taux de lecture mesuré** — démarrer tolérant (distance ≤ 6 m, angle ≤ 60°, pas de contrainte vitesse, dwell 1 pas), resserrer un cran quand le taux dépasse ~70-80 %, relâcher s'il s'effondre (gate bidirectionnel). **Règle absolue** : la SEULE métrique de progrès rapportée est le taux de lecture **au seuil nominal** (3 m/35°/0,6 m/s/0,8 rad/s/2 pas), évaluée périodiquement même pendant les phases tolérantes — sinon on croit progresser alors qu'on apprend des scans « de loin » inutilisables. Le dwell se resserre en **dernier** (condition la plus dure à découvrir).

### 6.3. Reverse curriculum sur les spawns (Florensa CoRL 2017, RFCL ICLR 2024)

Phase précoce : spawner une fraction des drones déjà près des racks, face à des cartons (états où le succès est à 10-90 %) ; élargir vers des spawns génériques. Donne l'effet Go-Explore sans archive d'états, trivial dans Isaac.

### 6.4. Calibration du gate — FAITE (2026-07-09, `swarmscan_map/calibrate_gate.py`)

Mesurée sur images rendues Isaac (caméra déclarée 1280×960, HFOV 60°), décodeur **cv2.QRCodeDetectorAruco + upscale ×3** — la même famille de décodeur qu'utilise Pore (OpenCV), donc la comparaison porte sur le contrôle (Chen 2024 : le décodeur fait varier l'accuracy de 45 à 90 %). Résultats (CSV : `docs/calibration_gate.csv`) :

| Seuil | Supposé | **Mesuré** | Retenu (marge) |
|---|---|---|---|
| Distance de lecture | 3,0 m | décodage jusqu'à **1,5 m** (0 % à 2 m) | **1,25 m** |
| Angle d'incidence | 35° | décodage jusqu'à **55°** | **50°** |
| Cône caméra (demi-angle) | 60° | HFOV 60° calibré | **30°** |
| Vitesse / lacet | 0,6 m/s / 0,8 rad/s | non mesurable en rendu statique | 0,6 / 0,8 (loi Cristiani 2020, déclaré) |

Conséquences appliquées : `alt_max` 3→4 m (tags hauts lisibles jusqu'à ~4,55 m), épisode 90→120 s (couverture ~2× plus lente à 1,25 m qu'à 3 m), normalisation de couverture recalée (~13 cellules/cône). C'est un résultat en soi pour le mémoire : les petits QR (~20 cm) imposent une inspection rapprochée — exactement pourquoi le vol près des racks (et donc l'évitement appris) est central.

### 6.5. Résilience entraînée

Drone-dropout pendant l'entraînement (p→0,3, déjà conçu en v1) + architecture insensible à la taille d'équipe (attention masquée, bit vivant/mort ; ACE est « team-size invariant » par construction). C'est ce qui rend le test « panne à mi-mission » gagnable : MAANS 3→2 en cours d'épisode fait 145 steps là où RRT en met 188.

### 6.6. Échelle et hyperparamètres

- **Budget** : 10-60 M de transitions-agent par run (Jonnarth : 8 M suffisent en mono-agent continu ; la v1 a fait 60 M en ~16 h sur ta 2060 Super — le régime est validé sur ta machine). 3 seeds minimum, 5 si le calendrier tient.
- num_envs au max VRAM (l'obs carte est compacte ; viser 32-64), batch par update ≥ dizaines de milliers, `mini_batches=1` (Yu 2022), γ = 0,995, GAE λ = 0,95, `time_limit_bootstrap=True`, lr schedule fixe si Beta, KL-adaptatif sinon, normalisation obs/valeurs.
- Épisode : 90-120 s (le 45 s de Pore est tenable pour ~2 racks, pas pour 123 tags ; on rapporte de toute façon temps-à-X %). Cible de mission : % du **plafond atteignable** (111/123 tags — 12 sont au-dessus de 4,62 m, mesuré en v1) plutôt que 100 % absolu.
- Sélection de checkpoint : sur **couverture au seuil nominal sur les configurations de validation** — jamais sur la récompense.

---

<a name="7-évaluation"></a>
## 7. Le protocole d'évaluation

### 7.1. La matrice (chaque cellule : ≥20-100 épisodes × 3-5 seeds, seeds d'éval FIXES et identiques pour toutes les méthodes)

| Axe | Cellules | Ce que ça prouve |
|---|---|---|
| **Nominal** | configurations de test held-out | généralisation en distribution |
| **Layout inconnu** | configurations OOD (occupation inhabituelle, éventuellement 2e entrepôt USD) | l'axe « adaptabilité » de la claim |
| **Inventaire perturbé** | QR déplacés / +20 % / −20 % en cours de mission | robustesse au drift d'inventaire (Pore : lecture valide SEULEMENT si l'ID ∈ S_exp connu a priori → il ne peut PAS découvrir un item inattendu, par construction) |
| **Panne drone** | 1 drone à t ∈ {25, 50, 75 %} (t aléatoire), puis 2 drones | résilience (Pore : allocation statique de secteurs → la zone du drone mort n'est JAMAIS couverte) |
| **Obstacles dynamiques** | 1, 2, 4 mobiles | Pore admet explicitement (§2.3.4) ne pas les gérer |
| **Bruit capteur** | lidar bruité faible/fort, dropout de rayons | robustesse sim-to-real |
| **Taille d'équipe** | entraîné à 3, testé à 2 et 4 | scalabilité de la coordination apprise |

Rapporter **par axe** (jamais un score agrégé unique), en **courbes de dégradation** (perf vs intensité), avec IQM + IC bootstrap 95 % (rliable).

### 7.2. Les baselines (décision utilisatrice : Pore, AIF, serpentin — c'est tout)

1. **Planner type Pore dans Isaac** : waypoints serpentin sur carte connue + champ de risque/potentiel + allocation statique de secteurs pour 2-3 drones. Version simplifiée du QP, déclarée comme telle. **Indispensable** : comparer nos chiffres Isaac à leurs chiffres MATLAB/Unreal serait attaqué (simulateurs différents) ; on reproduit leur *principe* chez nous et on compare à armes égales.
2. **AIF** (ton implémentation existante) : baseline « exploration sans apprentissage ».
3. **Serpentin scripté** (lawnmower hover-scan) : borne basse honnête.

Les trois baselines passent dans **la même matrice** de perturbations que notre politique. C'est là que la thèse se prouve : le serpentin et le planner s'effondrent sur panne/obstacles/layout, le RL se réalloue au pas suivant.

### 7.3. Les métriques (mêmes familles que Pore §3.1, mot pour mot, + les nôtres)

- **Couverture/accuracy** : lectures uniques valides / attendues (dédup 1,5 s, seuil de confiance), doublons %, précision/rappel/F1 (Pore les définit sans jamais les rapporter — les rapporter est un avantage).
- **Temps** : mission jusqu'à dernière lecture, **temps-à-50/90/95 %** (métrique principale), courbes couverture-vs-temps, QR/s, **speedup 1→2→3 drones** (leur référence : ~2× à 2 drones ; notre cible : ≥2,5× à 3 drones ET speedup préservé après panne).
- **Sécurité** : clairance min, % temps au-dessus de 0,5 m de clairance (leur « safe set »), séparation inter-drones min (leur seuil : 1,2 m), collisions.
- **Trajectoire** : le RMS vs trajectoire de référence **n'existe pas** en RL (pas de trajectoire de référence) — le dire, et substituer : jerk moyen, longueur de chemin par QR lu, énergie (ton perf_index/énergie existant).
- **Latence** : inférence de la politique par pas (<5 ms sur GPU, vs leur 0,41 s décodage→dashboard — pas la même chose, à présenter honnêtement).
- **Résilience** : latence de récupération (retour à 90 % du débit pré-panne), delta de couverture finale vs nominal.

---

<a name="8-pore"></a>
## 8. Pore et al. 2026 : ce qu'il fait, ses chiffres, et comment on le bat

### 8.1. Leur système (lu intégralement, première main)

Waypoints (x,y,z) **pré-planifiés** en serpentin d'allée avec arrêts hover-and-scan, lissés par un QP convexe (min accélération + yaw-rate, contraintes d'allée « keep-in », bornes cinématiques) ; sécurité par champ de risque sigmoïde sur une **SDF dérivée de la carte 3D connue** (r_safe = 0,5 m), replanification locale à horizon court ; caméra fisheye streamée vers la station sol qui décode avec OpenCV ; multi-UAV = **symétrie miroir** figée dans le QP (2 drones, même rack) ou **secteurs statiques disjoints avec plages d'ID pré-assignées** (2 racks) + règle de priorité à jeton.

### 8.2. Leurs chiffres

| | Sim (5 runs) | Réel (DJI Air 2S) |
|---|---|---|
| QR accuracy Case I (1 UAV) | 95,8 ± 0,6 % | 90,5 ± 0,6 % |
| QR accuracy Case II/III (2 UAV) | ~95,6 / ~95,7 % | 88,5 / 86,1 % |
| Temps mission 2 UAV | 65-68 s | 133,8 s/rack (Case I réel) |
| Réduction vs 1 UAV | 48-52 % (même rack), speedup 1,86-1,95× (racks parallèles) | — |
| Trajectoire RMS | 7,6-9,3 cm | idem (voir red flag) |
| Séparation min | 1,28-1,33 m | — |

### 8.3. Leurs faiblesses structurelles (admises ou démontrables)

1. **Tout est connu a priori** : carte 3D (SDF), waypoints, ET la liste des IDs attendus S_exp — une lecture n'est « valide » que si l'ID appartient à S_exp. Le système **ne peut pas découvrir** un item déplacé/inattendu, par définition.
2. **Obstacles dynamiques non gérés** — admis noir sur blanc dans le papier (§2.3.4, « future work »).
3. **Aucune ré-allocation** : symétrie miroir et secteurs figés. Une panne de drone = sa zone jamais couverte. Le papier ne teste aucune panne.
4. **Protocole faible** : 5 runs, un seul layout, éclairage constant, aucune baseline, aucune ablation, N_expected et dimensions jamais donnés, CIs bootstrap promis jamais montrés.
5. **Red flag** (à manier avec tact en soutenance) : les tableaux « expérimentaux » Case II/III reproduisent **chiffre pour chiffre** les colonnes des tableaux de simulation (latences identiques à la milliseconde) ; seule l'accuracy QR a manifestement été re-mesurée. Et le DJI Air 2S n'a pas de LiDAR embarqué — comment le champ de risque tournait-il en réel ? Non expliqué.
6. Venue : *Symmetry* (journal MDPI de mathématiques générales, pas un venue robotique) ; 0 citation. → Le positionner comme « système représentatif de l'état de la pratique waypoints + champ de risque », pas comme SOTA.

### 8.4. Où on le bat, précisément

| Axe | Eux | Nous (cible) |
|---|---|---|
| Layout/inventaire inconnu ou modifié | N/A par construction (waypoints scannent du vide, IDs inattendus rejetés) | couverture ≥90 % du plafond, découverte des items déplacés |
| Obstacle dynamique | non géré (admis) | évitement réactif appris, dégradation mesurée et bornée |
| Panne d'un drone à mi-mission | zone perdue, couverture plafonnée ~(N−1)/N | ré-allocation émergente au pas suivant, récupération < qq secondes (v1 : 2,0 s), couverture quasi intacte |
| Speedup multi-drone | ~2× à 2 drones (secteurs pré-partitionnés) | ≥2,5× à 3 drones par coordination **apprise**, ET préservé sous perturbation |
| Besoin d'infrastructure | carte 3D + SDF + waypoints + liste d'IDs | **rien** : les drones découvrent l'inventaire |
| Méthodo d'éval | 5 runs, 1 layout | matrice held-out + résilience, 3-5 seeds, IQM+CI |
| Ce qu'on ne bat PAS (à dire) | accuracy de décodage (~même caméra/décodeur → ~même score), RMS de trajectoire (n'existe pas chez nous), latence dashboard | on rapporte sans revendiquer |

Sur **leur** scénario idéal (layout connu, statique, nominal), un planner offline reste imbattable en temps brut (Jonnarth : le RL online reste 35-51 % plus lent que le meilleur planner offline à carte connue) — on ne prétend PAS le contraire, on montre que dès que le monde bouge (leur propre « future work »), leur avantage s'inverse.

---

<a name="9-contribution"></a>
## 9. La contribution du PFE

**Claim principal (formulation vérifiée contre la littérature, chaque qualificatif nécessaire) :**

> « À notre connaissance, la **première politique multi-drone entraînée par apprentissage par renforcement de bout en bout** — des capteurs embarqués (LiDAR + état + mémoire de couverture construite en ligne) aux **commandes de vitesse continues** — pour le **stocktaking d'entrepôt**, **sans positions des tags, sans liste d'IDs attendus, ni carte a priori**, évaluée sur des **configurations jamais vues** à l'entraînement et sous **perturbations** (panne de drone, obstacles dynamiques, inventaire modifié). »

Garde-fous de novelty (vérifiés en ligne, juillet 2026) : citer et écarter soi-même (a) **Lin, Chang & Huang 2024** (Drones 8(6):220 — seul travail « RL + inventaire drone » : leur RL n'ajuste que le point de vue, navigation 100 % classique SLAM/ROS, mono-drone) ; (b) **Corvus Robotics** (stack de navigation learning-based industriel, modulaire, non publié — pas une politique RL) ; (c) la revue **Drones 2026** (10(3):189, >120 papiers) qui déclare le multi-drone d'inventaire « unresolved, single-UAV, simulation-heavy » — notre case est vide. Re-vérifier la veille 1-2 mois avant le dépôt.

**Contributions secondaires** (chacune défendable seule) :
1. **Formulation « couverture en conditions de lecture »** des façades multi-niveaux (2.5D) — aucune couverture RL de faces verticales de racks publiée.
2. **Assignation de crédit contrefactuelle** pour mission de scan (contribution marginale géométrique, lignée difference rewards/MAANS) et son effet mesuré sur l'anti-agglutinement et la panne.
3. **Résilience entraînée** (drone-dropout + invariance à la taille d'équipe) démontrée sur une matrice publiable.
4. **Méthodologie d'évaluation** de l'inventaire par essaim (matrice held-out + perturbations + statistiques rliable) — directement réutilisable par le domaine, et supérieure au protocole du papier de référence.
5. La **baseline négative documentée** (audit du 0 % IPPO) : l'échec end-to-end naïf est *prédit* par la littérature, le chapitre le prouve proprement.

---

<a name="10-pourquoi"></a>
## 10. Pourquoi ça va marcher

| Cause d'échec prouvée (audit) | Correction | Preuve littérature |
|---|---|---|
| n°1 cible invisible | La récompense n'exige plus de percevoir des cibles : couvrir des façades (perçues par lidar) suffit ; mémoire = carte accumulée | ANS/SemExp/Jonnarth/GLEAM : exactement cette formulation, sans position d'objets |
| n°2 événement +10 inéchantillonnable (p~10⁻⁵) | Signal dense dès le pas 1 (ΔCouverture) + ADR sur les seuils + reverse spawns + shaping potentiel privilégié | ANS (10M frames), OpenAI ADR/DORAEMON, Florensa/RFCL, SemExp (reward privilégié standard) |
| n°3 saturation d'actions | σ₀=0,5, action-difference pré-clamp, entropie ≤0,005, monitoring du clip | SimpleFlight (RA-L 2025, Crazyflie), Andrychowicz 2021 ; Seyde 2021 (le vrai coupable était le reward) |
| n°4 optimum « lâche » (fuir les racks) | Approche signée potentielle, collision graduée jamais terminale, pas de pénalité de temps constante, la couverture PAIE d'aller aux racks | Ng 1999 ; conception MAANS des pénalités programmées |
| n°5 échelle (batch 384, obs 1820) | Obs compacte structurée (cartes 32×32), batch ≥10⁴, 10-60M transitions, PS réel (un seul réseau) | standards Isaac/rsl_rl ; MAexp/MAANS/Jonnarth ; v1 a déjà tourné 60M/16h sur TA machine |
| Aucune randomisation (N effectif=1) | Génération procédurale + split train/val/test + DR capteurs/dynamique | Cobbe/Kirk/ProcTHOR/GLEAM ; Explore-Bench (le DRL sous-randomisé est PIRE que frontier — l'échelle de randomisation décide) |
| Agglutinement (v1) | Contribution marginale + overlap penalty + attention coéquipiers | MAANS (Mutual Overlap ≈ Voronoi imposé), ETS-MAPPO |

Et la preuve d'ensemble : **la v1 oracle a atteint 98,6 %** sur cette même infra Isaac. Le pipeline (env, raycast, PPO, curriculum) sait apprendre cette tâche quand le signal existe. Tout ce que fait cette conception, c'est remplacer la source du signal (oracle → carte auto-construite + reward privilégié), avec cinq papiers indépendants qui montrent que ce remplacement marche (Jonnarth, GLEAM, NTNU, MAANS, ANS).

---

<a name="11-risques"></a>
## 11. Risques et mitigations (sortie de la critique adversariale)

| # | Risque | Mitigation |
|---|---|---|
| R1 | Couvrir sans lire (surface vue ≠ QR lus) | Définition « cellule couverte » = conditions de lecture (§3.1) ; métrique pilote = taux de lecture au seuil nominal, jamais le % de surface |
| R2 | Le multi-agent end-to-end continu n'existe pas tel quel dans la littérature (tout le SOTA garde un planner local) → risque d'échec IPPO-bis | Dérisquer en escalier : **mono-drone d'abord** (recette Jonnarth+NTNU validée), puis 2, puis 3 par curriculum. Plan B pré-approuvé : hiérarchie **100 % apprise** (tête lente qui émet un sous-but sur la carte + politique de vitesse apprise — zéro planner algorithmique, la contrainte survit) |
| R3 | Reward hacking du shaping (hover « presque en lecture ») / contestation du shaping privilégié | Forme potentielle exclusivement (boucle fermée = 0), bonus one-shot dominant, monitoring QR/s et temps de hover ; paragraphe mémoire + ablation sans shaping |
| R4 | Généralisation insuffisante → claim principale qui s'effondre | Générateur procédural opérationnel AVANT tout entraînement long ; rapporter le generalization gap comme un résultat ; aucune constante N=123 codée en dur dans obs/reward/terminaison |
| R5 | Budget compute/statistique intenable (matrice × seeds × méthodes sur 8 Go) | Obs compacte (régime 8M steps Jonnarth prouvé sur T4) ; prioriser la matrice : nominal + layout inconnu + panne 1 drone = obligatoires, le reste si temps ; 3 seeds assumés comme limite si nécessaire ; baseline Pore simplifiée déclarée |
| R6 | Sémantique de la panne non spécifiée (que perd l'essaim : le drone ou aussi sa carte ?) | À écrire noir sur blanc AVANT les tests : carte répliquée (fusion continue) → la panne coûte un capteur, pas la mémoire. Variante « perte de mémoire » en scénario bonus |
| R7 | Seuils du gate non calibrés → curriculum convergeant vers un critère arbitraire | Calibration pyzbar/Cristiani en P1 (§6.4), seuils gelés ensuite |

---

<a name="12-plan"></a>
## 12. Plan d'exécution

Ordre logique des travaux (sans calendrier — décision utilisatrice ; chaque étape se valide avant de passer à la suivante) :

1. **Verrouillage** : décision écrite de la règle d'or (§2) ; calibration du gate (§6.4) ; générateur procédural de configurations + split gelé ; ré-implémentation à neuf dans `rl_inventory/` (aucun code ancien repris).
2. **Mono-drone, carte, couverture** : env carte égocentrique multi-échelle (2.5D façades), reward §5 sans multi-agent ; critère de passage : **>80 % du plafond atteignable en config held-out, mono-drone** (recette Jonnarth/NTNU — si ça ne marche pas là, inutile d'aller au multi).
3. **Multi-agent** : PS-PPO 3 drones, fusion de cartes, contribution marginale, dropout ; critère : speedup ≥2× vs mono à couverture égale, sans agglutinement.
4. **Baselines + matrice complète** : planner type Pore dans Isaac, AIF, serpentin ; 3-5 seeds ; toute la matrice §7.
5. **Ablations** : sans canal frontières, sans contribution marginale, sans shaping privilégié, sans ADR, GRU on/off — chaque ablation est un paragraphe de mémoire.
6. **Piste Dreamer bornée (§13) + rédaction.**
À chaque étape : les runs longs lancés par toi ; monitoring = taux de lecture au seuil nominal, fraction d'actions clippées, entropie, couverture val.

---

<a name="13-dreamer"></a>
## 13. Place de Dreamer

Verdict honnête de la recherche : **différenciateur réel mais risque élevé sur 8 Go**. Les succès publiés (Dream to Fly 2025 : 240 h sur GPU 48 Go ; DreamerNav dans Isaac : RTX 4090 24 Go) sont au-dessus de ton matériel ; le multi-agent Dreamer est immature. MAIS un point scientifique fort existe : Plan2Explore/DreamerV3-XP permettent une exploration dirigée par l'incertitude du modèle **calculée en imagination** — l'argument « perception active » parfait pour l'inventaire, et **aucune comparaison publiée Dreamer vs PPO+carte sur une tâche d'inventaire n'existe** : ce serait une contribution originale, pas une réplication.

Scope raisonnable : piste 2 **bornée**, mono-drone, config small (12-25M params), lidar compressé par MLP-VAE (recette arXiv 2512.03429 : 100 % de succès nav lidar là où SAC/TD3 plafonnent <85 %), implémentation sheeprl. Gate go/no-go : le world model prédit-il correctement le lidar imaginé ? Sinon, repli = PPO + intrinsèque type désaccord (l'argument perception active survit à moindre coût). Dreamer peut passer en « perspectives » sans fragiliser le mémoire.

---

<a name="14-code"></a>
## 14. Code : base = le setup `rl_inventory/` existant (décision utilisatrice)

Le code **supprimé** (`swarmscan/`) n'est pas récupéré (pas de décompilation). En revanche, le **setup `rl_inventory/` existant est la base** : `env.py` (chargement entrepôt, drones, lidar, raycast statique, proxy QR), `config_rl.py`, `qr_task.py`, `launch.sh`, tests. La solution v2 (cartes égocentriques, nouveau reward, ADR, PS-PPO) se construit **par-dessus cette base**, en corrigeant au passage les bugs identifiés dans l'audit (`_refreshed`, lidar monde/cap, `time_limit_bootstrap`, etc.).

---

<a name="15-biblio"></a>
## 15. Bibliographie commentée (sélection, ~40 entrées vérifiées en ligne)

### Le papier à battre + inventaire par drone
- **Pore, Patle, Thorat 2026** — *UAV-Based QR Code Scanning and Inventory Synchronization…* — Symmetry 18(4):548. [DOI](https://doi.org/10.3390/sym18040548)
- **Lin, Chang, Huang 2024** — *Development of UAV Navigation and Warehouse Inventory System Based on RL* — Drones 8(6):220. Seul « RL + inventaire » publié (viewpoint seulement). [DOI](https://doi.org/10.3390/drones8060220)
- **Kalinov et al. 2020** — *WareVision* — RA-L 5(4). Trajectoire adaptée aux détections > serpentin fixe (notre thèse, version TSP).
- **Cristiani et al. 2020** — *Inventory Management through Mini-Drones* — WoWMoM. Loi P_scan(v)=1/(v+1)^k_r pour calibrer le gate.
- **Chen, Huang 2024** — *Drone-Assisted QR Code Recognition* — ICNSC. Le décodeur fait 45→90 % : déclarer le nôtre.
- **Martinez-Carranza, Rojas-Perez** — *Warehouse Inspection with an Autonomous MAV* — Unmanned Systems. ~90 % sur 24 QR à 0,25 m/s : la norme caméra réelle.
- **Revue Drones 2026, 10(3):189** — multi-drone inventaire = « unresolved » ; notre case est vide. [MDPI](https://www.mdpi.com/2504-446X/10/3/189)
- **VACNA 2023** (RA-L, Tartu) — navigation coopérative visibility-aware pour inventaire. [DOI](https://doi.org/10.1109/LRA.2023.3312969)

### Couverture/exploration RL sans positions de cibles (le socle de la solution)
- **Jonnarth, Zhao, Felsberg 2024** — *Learning Coverage Paths in Unknown Environments with DRL* — ICML 2024. Cartes égo 4×32×32 + vitesses continues ; bat frontier 1,7-3,2× en T90 ; 8M steps sur T4. [arXiv](https://arxiv.org/abs/2306.16978)
- **Chen et al. 2025** — *GLEAM* — ICCV 2025. Crazyflie dans Isaac, 32 envs PPO, 1024 scènes→128 held-out : 66,5 % vs 56,8 % frontier. [arXiv](https://arxiv.org/abs/2505.20294)
- **Kulkarni et al. 2025 (NTNU)** — *Semantically-driven DRL for Inspection Path Planning* — le plus proche de nous : actions (vx,vy,vz,yaw_rate), reward = faces du mesh vues à bonne distance, 97 % vs 85 % FUEL, transfert réel. [arXiv](https://arxiv.org/abs/2505.14443)
- **Chaplot et al. 2020** — *Active Neural SLAM* — ICLR. Carte accumulée 2 canaux, reward = m² explorés, 10M frames.
- **Chaplot et al. 2020** — *SemExp* — NeurIPS. **Reward privilégié (distance à l'objet) avec position jamais dans l'obs = pratique standard.**
- **ProcTHOR** — NeurIPS 2022 (outstanding). 10k layouts procéduraux → SOTA zero-shot : la preuve « beaucoup de layouts = généralisation ».
- **GenNBV** — CVPR 2024 ; **NextBestPath** — ICLR 2025 ; **ARiADNE** — ICRA 2023 ; **Explore-Bench** — ICRA 2022 (le DRL sous-randomisé est PIRE que frontier).
- **Gervet et al. 2023** — Science Robotics. Modulaire 90 % vs end-to-end 23 % en réel : pourquoi la carte dans l'obs est non négociable.

### Multi-agent
- **Yu et al. 2022** — *MAANS* — ECCV. Spatial-TeamFormer ; contribution marginale 0,02·ΔArea_k ; overlap penalty programmée ; 3→2 drones intra-épisode ; +20,6 %/+8 % vs planners. [arXiv](https://arxiv.org/abs/2110.05734)
- **Yu et al. 2023** — *ACE* — AAAI. Team-size invariant, panne 4→3 : 100 % couverture, ~10 % plus rapide.
- **MAexp** — ICRA 2024. Benchmark 6 algos : **IPPO meilleur en indoor encombré (94,18 %)** — l'échec de ton IPPO venait de l'obs/reward, pas de l'algo.
- **MARVEL** — ICRA 2025. FoV contraint, graph attention, s'adapte à la taille d'équipe sans ré-entraînement, vrais drones.
- **ETS-MAPPO** — Sensors 2024 (belief map dans l'obs : 0,89 vs 0,82) ; **Jiang et al.** — Aerospace 2024 (MARL tolérant aux pannes) ; **IR2** — IROS 2024 ; **MACE** — AAAI 2024.

### Recherche de cibles / gate de détection
- **D-VAT** — RA-L 2024. Shaping dense par condition du gate (distance, centrage, vitesse) + critique asymétrique + DR → zero-shot réel. [arXiv](https://arxiv.org/abs/2308.16874)
- **HRL inspection** — Drones 2025 9(5):352 (qualité d'observation dans le reward, −70 % de waypoints vs CPP) ; **PyroTrack** 2024 (belief map > obs brute) ; **WiSAR DRL** 2024-25 (+160 % vs planners de couverture) ; **Recurrent PPO target localization** arXiv 2412.06231 (93 %/86 %, 80 layouts en rotation).

### Événement rare, curriculum, saturation
- **OpenAI 2019** — *Solving Rubik's Cube* (ADR) ; **DORAEMON** — ICLR 2024 (entropie max sous contrainte de succès) ; **Florensa 2017** (reverse curriculum) ; **RFCL** — ICLR 2024.
- **SimpleFlight** — RA-L 2025 (arXiv 2412.11764). Action-difference > clipping ; zero-shot Crazyflie ; infos privilégiées au critique.
- **Ng, Harada, Russell 1999** — shaping potentiel (la seule forme sûre) ; **Seyde 2021** — bang-bang quasi optimal (la saturation était un symptôme) ; **Henaff 2023** — bonus épisodiques > globaux en MDP contextuel ; **Chou 2017 / Petrazzini 2021** — Beta (en ablation) ; **Andrychowicz et al. 2021** — σ₀≈0,5 (à re-vérifier dans arXiv 2006.05990).

### Protocole d'évaluation
- **Cobbe 2019** (CoinRun) ; **Procgen 2020** ; **Kirk et al. 2023** — JAIR (LA survey généralisation : held-out obligatoire).
- **Henderson 2018** (5 seeds ≠ 5 seeds) ; **Colas 2018** ; **Agarwal et al. 2021** — rliable/IQM (NeurIPS outstanding) ; **Gorsane et al. 2022** — protocole MARL standard ; **Patterson et al. 2024** — JMLR ; **N-Agent Ad Hoc Teamwork** — NeurIPS 2024.
- **Tobin 2017** ; **Muratore 2022** (review DR).

### World models (piste 2)
- **DreamerV3** — arXiv 2301.04104/Nature 2025 ; **Dream to Fly** — arXiv 2501.14377 (240 h/48 Go — borne matérielle) ; **DreamerNav** — Frontiers 2025 (Dreamer dans Isaac, 24 Go, ~25 h/495k steps, ATTENTION : utilise A* en obs) ; **World Models lidar** — arXiv 2512.03429 (MLP-VAE lidar, 100 % vs <85 % model-free — preprint à vérifier) ; **Plan2Explore** — ICML 2020 ; **DreamerV3-XP** — 2025 ; **POPGym** — ICLR 2023 (GRU meilleure mémoire) ; **Ni et al.** — ICML 2022 ; **Memory Gym** — JMLR 2025.

*Chiffres marqués « à re-vérifier » dans les warnings des agents : success rates exacts de RFCL et du two-stage reward curriculum (arXiv 2603.05113) ; détails Andrychowicz. Tout le reste a été extrait des sources par les agents de recherche (juillet 2026).*
