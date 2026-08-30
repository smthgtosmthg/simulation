# Étude comparative des 4 directions — quelle méthode pour découvrir des QR codes dans un entrepôt ?

Date : 28 août 2026
Méthode : 8 agents d'étude de la littérature, puis 4 agents critiques qui ont vérifié chaque
chiffre à la source. Environ 90 sources examinées, une trentaine lues en entier.

---

## 1. Pourquoi cette étude

On repart de zéro. On ne garde que la **problématique**. Tout le reste — architectures,
fichiers de configuration, calibrations, décisions passées, et même la manière d'entraîner —
est remis en question. Le simulateur lui-même peut changer si la solution retenue l'exige.

Le promoteur a proposé quatre directions. Le but de cette étude est simple : **dire laquelle
a le plus de chances de marcher**, avec des preuves, pas des impressions.

### La problématique

Un essaim d'environ 3 drones doit **découvrir et lire des QR codes dans un entrepôt, sans
connaître à l'avance leur position ni leur emplacement**. La solution doit :

- décider où explorer **en fonction de ce qu'elle découvre** — pas de trajectoire pré-calculée ;
- gérer les imprévus : panne d'un drone, inventaire modifié ;
- **généraliser** à des entrepôts jamais vus.

Contraintes physiques (elles décrivent le problème, pas un choix technique) : lire un QR
impose de s'approcher à environ 1 m, d'être à peu près en face, et de voler lentement
(~0,5 m/s). Les cartons sont sur des racks, à plusieurs hauteurs.

Moyens : **GPU de 8 Go**, échéance **2 à 3 mois**.

### Les quatre directions

| # | Direction | Idée du promoteur |
|---|---|---|
| **D1** | Dreamer / World Model | Un modèle du monde visuel apprend depuis les images ; il verrait les QR avec la caméra. |
| **D2** | Visual Models / VLA / LLM | OpenVLA, LLaMA et modèles du même genre. |
| **D3** | RL navigation + module visuel | Un RL apprend à naviguer et explorer ; un modèle visuel lui dit où sont les QR qu'il voit. |
| **D4** | Active Inference + module visuel | L'inférence active pour la navigation, un module visuel pour la découverte. |

---

## 1 bis. À lire d'abord : de quoi parlent ces pourcentages

Ce document critique des méthodes. Il ne dit pas **à quoi elles ressembleraient chez nous** —
c'est le rôle du document compagnon [`architectures_4_directions.md`](architectures_4_directions.md),
qui construit les quatre solutions brique par brique (entrées, modèle, entraînement, sortie,
boucle de décision, contrôle).

**Les probabilités ci-dessous portent sur ces architectures précises, pas sur « la méthode en
général ».** Voici à quoi chaque chiffre se rapporte :

| Chiffre | Ce qu'il mesure exactement |
|---|---|
| **D1 : 10 %** | DreamerV3 préréglage 12M, images 64×64 de 2 caméras + proprioception, **décodeur de QR externe** (obligatoire, mais pas une concession : personne n'apprend à décoder) et **carte de couverture externe** (la mémoire récurrente ne tient pas 1800 pas), acteur sortant les vitesses à 10 Hz, entraîné en imagination sur 15 pas. *Ce qui limite cette direction est la mémoire, le disque et le nombre d'essais — **pas** la résolution : la mesure du 29/08 montre qu'en 64×64 le modèle voit très bien racks et cartons.* |
| **D2 : 10 %** | Un modèle fondation comme **moteur de décision** : soit il pilote (OpenVLA réglé par LoRA sur des démonstrations), soit il choisit une zone toutes les 3-5 s. **Ne mesure pas** le rôle « détecteur de vision », qui lui est utile et vaut 75 % |
| **D3 : 15-25 %** | Détecteur classique à 3 sorties (décodé / repéré non décodé / rien) → carte 3D partagée → **politique apprise de ~1-2 M paramètres qui choisit un point de vue parmi ~50 candidats, une fois par seconde** → contrôleur classique. Entraînement PPO **sans rendu d'image** |
| **D4 : 8 %** | Le même système, mais la décision est un **planificateur d'énergie libre attendue** avec recherche arborescente, sur une croyance probabiliste, sans rien apprendre |

**Le fait le plus important, qui n'apparaît qu'en construisant les architectures : D3 et D4
sont le même système avec deux têtes différentes.** Même œil, même contrôleur, même mémoire,
même coordination. Ils ne diffèrent que sur la brique « décision ». La vraie question n'est donc
pas « quelle direction ? » mais **« la décision doit-elle être apprise ou calculée ? »** — et la
réponse pratique est de construire le socle commun, puis d'y brancher les deux têtes et de les
comparer.

---

## 2. Le résultat en une page

| Direction | Verdict | Chance d'atteindre l'objectif en 2-3 mois | Ce qu'on en garde |
|---|---|---|---|
| **D3 — RL nav + module visuel** | **Mitigé, le meilleur des quatre** | **15-25 %** | **La direction retenue**, mais pas telle qu'elle a été présentée |
| D4 — Active Inference | Mitigé penchant faible | 8 % (complet) / **60-65 % (emprunts seuls)** | **Trois emprunts précis**, greffés sur D3 |
| D1 — Dreamer / World Model | Faible | 10 % | Une idée (le modèle informé), à garder en réserve |
| D2 — VLA / LLM | Faible | 10 % (2 % en contrôle direct) | **Le patron à deux étages** + **la vision comme perception** |

**La conclusion tient en trois phrases.**

1. Les quatre directions ont une probabilité de succès faible, entre 8 % et 25 %. Aucune n'est
   un chemin sûr. Il faut le savoir avant de choisir.
2. **D3 est la moins mauvaise**, et c'est celle qui absorbe naturellement les bonnes idées des
   trois autres. C'est la direction recommandée.
3. **Mais la critique la plus importante de toute l'étude ne porte sur aucune des quatre
   directions : elle porte sur le problème lui-même.** Il se pourrait qu'il n'y ait presque
   rien à découvrir dans un entrepôt tel qu'on l'a défini. Il faut le mesurer en 3 jours,
   avant d'écrire la moindre ligne de code d'apprentissage. C'est le point 6 de ce rapport,
   et c'est le plus important.

---

## 3. Les quatre fiches

### D1 — Dreamer / World Model

**Comment ça marcherait.** Le drone apprend un « modèle du monde » : un réseau qui prédit ce
qu'il va voir ensuite. Il s'entraîne ensuite en imagination, dans sa propre tête, ce qui coûte
beaucoup moins de données réelles. Le promoteur espérait que ce modèle voie les QR codes
directement dans les images de la caméra.

**Les preuves.** Elles existent et elles sont bonnes — mais sur d'autres tâches.

- *Dream to Fly* (arXiv 2501.14377, ICRA 2026) : course de drone depuis des images 64×64.
  **PPO et SAC ne convergent pas après 20 millions de pas ; DreamerV3 converge.** C'est le
  meilleur argument de la direction. Coût : 240 h sur une Quadro RTX 8000.
- *DreamerNav* (Frontiers Robotics AI 2025) : entrepôt sous Isaac Sim, 0,72 de réussite contre
  1,2 but atteint pour A*. 24,8 h sur RTX 4090.
- *Generalization of World Models…* (arXiv 2606.05015, code public) : navigation en salle,
  99,5 % chez soi mais **54,5 % sur un layout inédit** ; avec entraînement varié, 72 %.

**Les trois murs, chacun mesuré.**

1. ~~Un modèle du monde en 64×64 ne peut pas voir un QR.~~ **Corrigé le 29/08 par la mesure.**
   En 64×64, le modèle voit très bien les racks et les cartons (un carton fait 11 px à 2 m) ;
   il ne voit pas le motif du QR (2 px). Mais comme le décodage est de toute façon confié à une
   bibliothèque externe dans les quatre directions, **ce n'est pas un obstacle**. Ce qui reste
   vrai : le modèle ne peut viser que le carton — et chez nous ça suffit, chaque carton portant
   un QR. Détail à garder : la compression latente supprime les petits objets (DIAMOND,
   NeurIPS 2024 ; R2-Dreamer, ICLR 2026), donc ne pas compter sur les pixels pour l'alignement
   fin. Voir `experiments/01_resolution_qr/RESULTATS.md`.
2. **La mémoire.** DreamerV3 tient des délais d'environ **30 pas**. Notre épisode en fait
   **1800**. Memory Maze — la tâche jumelle de la nôtre, retrouver des objets vus plus tôt —
   est précisément celle où il échoue. Le correctif publié (R2I) coûte 400 millions de pas et
   deux semaines de calcul.
3. **L'argument « découverte » est réfuté.** Plan2Explore, l'exploration par désaccord, sous-
   performe sur les labyrinthes 3D : « le désaccord repère la nouveauté mais n'impose pas de
   persistance au niveau de la trajectoire ». Or balayer 120 QR est une routine soutenue, pas
   une chasse ponctuelle.

**Ce que la critique a ajouté.**

- Correction : le dépôt `r2dreamer` n'est pas « 5× plus rapide », le papier annonce **1,59×**.
- **Le vrai mur n'est pas la mémoire vidéo, c'est le disque.** Avec 3 drones × 2 caméras, le
  tampon de rejeu pèse 74 Ko par pas. 500 000 pas font **37 Go**. La machine a **20 Go libres**.
  On serait limité à 30-80 épisodes rejoués en boucle — alors que l'objectif est justement de
  généraliser à des entrepôts jamais vus. Personne n'avait vu ce problème.
- L'argument fort ne se transpose pas : **Swift (Nature, 2023) fait de la course de drone au
  niveau champion du monde avec du PPO**. « Dreamer réussit là où PPO échoue » n'est pas une
  loi générale.

**Ce qui tue la direction.** Chaque faiblesse a un correctif bon marché : décodeur externe pour
la vision, carte externe pour la mémoire, récompense de couverture explicite pour l'exploration.
Mais **ces trois correctifs retirent au modèle du monde exactement ce qui justifiait de le
choisir.** Il ne reste que l'efficacité en données — or à temps de calcul égal, PPO voit 20 à
100 fois plus de situations.

> **Verdict : FAIBLE. 10 %.**
> À garder : l'idée du *modèle informé* (SkyDreamer, arXiv 2510.14783) — donner au modèle la
> position des QR **pendant l'entraînement seulement**, jamais dans l'observation déployée.

---

### D2 — Visual Models / VLA / LLM

**Comment ça marcherait.** Trois rôles possibles, très différents : (a) le modèle pilote
directement le drone ; (b) il sert de « cerveau lent » qui choisit les zones à explorer pendant
qu'un contrôleur rapide vole ; (c) il sert seulement d'œil, pour détecter les cartons et les
codes.

**Il faut le dire d'emblée : le promoteur a pointé le bon papier.** Le seul travail publié qui
pilote un drone avec **exactement notre espace de commande** (Vx, Vy, Vz, ω) s'appelle
**CognitiveDrone** (arXiv 2503.01378), et il est bâti **sur OpenVLA**. Il atteint 59,6 % de
succès, puis 77,2 % avec un module de raisonnement. Ce n'était pas une intuition hasardeuse.

**Le mur est matériel, et il est chiffré** (tous vérifiés dans les sources d'origine) :

| Fait | Chiffre |
|---|---|
| OpenVLA en 4 bits — la seule variante qui tient dans 8 Go | **3 Hz** (il en faut 30) |
| OpenVLA-OFT — la seule variante assez rapide (108,8 Hz) | **15,9 à 19,2 Go** |
| Réglage fin par LoRA d'OpenVLA, minimum officiel | **~27 Go** |
| Entraînement de CognitiveDrone | **4 GPU A100**, 8 000 trajectoires |
| Budget réel disponible chez nous | **2 à 4 Go** (le simulateur consomme le reste) |

**L'argument de fond, reformulé.** Les fiches disaient : « le QR n'a pas de sémantique, donc le
modèle ne peut rien en dire ». La critique a montré que cet argument est **attaquable** — V*/SEAL
atteint 73,82 % sur des cibles occupant moins de 0,05 % de l'image, et la sémantique utile est
dans le *carton* et l'*étagère*, pas dans le QR.

**Le bon argument est ailleurs, et il est solide** : le gain d'un modèle vision-langage vient
de **l'irrégularité des lieux** — des appartements avec des pièces cachées, des culs-de-sac,
là où un planificateur géométrique se trompe. Les deux travaux 2026 qui le démontrent (SAGE et
*Autonomous Frontier-Based Exploration with VLM Guidance*, tous deux de Berkeley) mesurent
+24 % de couverture… **dans des maisons Matterport3D**. Un entrepôt est le cas exactement
inverse : une grille régulière d'allées parallèles, où un balayage classique est déjà proche de
l'optimum. **Le gain attendu tend vers zéro.**

Et dans tous ces travaux, dès qu'il y a un gain, **le modèle ne tourne pas à bord** : SAGE
déporte CLIP hors du drone, l'autre utilise Gemini dans le cloud, GoalSwarm tourne sur une
RTX 4090 de 24 Go et échoue précisément sur les petits objets.

> **Verdict : FAIBLE comme direction principale. 10 %** (dont 2 % pour le contrôle direct).
> **Mais 75 % pour le rôle (c), la perception** — et ce rôle est utile quelle que soit la
> direction retenue. C'est exactement le « module visuel » des directions 3 et 4.
> **À garder : le patron à deux étages** (décision lente en haut, contrôle rapide en bas),
> confirmé sans exception par une dizaine de travaux indépendants.

---

### D3 — RL navigation + module visuel

**Comment ça marcherait.** On découpe le problème en trois étages. Un **module visuel** détecte
les QR dans l'image et dit où ils sont. Une **politique apprise** décide où aller ensuite, en
fonction de ce qui a déjà été vu. Un **contrôleur** exécute le vol.

**Le consensus le plus net de toute l'étude** : sur douze travaux d'exploration apprise
recensés, **douze apprennent « où aller » et aucun n'apprend le contrôle bas niveau**. Active
Neural SLAM sort un but tous les 25 pas, MAANS tous les 15, ARiADNE et MRIPP choisissent un
nœud sur un graphe. Un planificateur classique fait le vol.

**La meilleure preuve, et la seule qui mesure notre métrique.**
**MRIPP** (arXiv 2409.16967, code sous licence MIT) : 3 robots, 185 à 250 cibles de **position
inconnue**, en 3D, et la métrique est bien le **pourcentage de cibles découvertes** sous budget
de trajet. L'action apprise inclut **la direction du regard** — le plus proche de « être en face
du QR ». Résultat : **73,53 %** contre 48,71 %, 47,03 %, 44,74 % pour les méthodes classiques.
Et seulement ~120 000 interactions d'entraînement.

**La rupture décisive.** L'apprentissage ne gagne pas partout, et on sait maintenant où :

| Si la métrique est… | L'apprentissage gagne-t-il ? |
|---|---|
| **La surface couverte** | **Non.** Explore-Bench : le champ de potentiel classique fait 60 s contre 130 s pour le RL, et dans le scénario Corner, 157 s contre **406 s**. Dans l'ablation d'ANS, la frontière classique fait 0,925 contre 0,948, soit +2,3 points. |
| **Le nombre de cibles trouvées sous budget** | **Oui, largement.** +25 points (MRIPP). Une frontière optimise la surface, pas les cibles. |

Notre problème est du second type. C'est ce qui rend la direction crédible.

**Mais la critique a fait mal, sur quatre points.**

1. **Le facteur « repérer de loin » n'existe pas.** La fiche affirmait qu'il est 1,75 à 3,5 fois
   plus facile de *repérer* un QR que de le *décoder*, et toute l'idée « voler vite en repérage,
   ralentir seulement sur cibles confirmées » reposait là-dessus. **C'est une erreur de lecture.**
   L'article source (Sensors 2022) ne mesure jamais la détection seule : sa seule métrique est le
   décodage. Le « seuil de 2 pixels par module » est le point où le **décodage** cesse d'être nul,
   pas où la détection commence. Le facteur réel est probablement **inférieur à 1,5**.
2. **Et même s'il existait, il ne servirait à rien ici.** Le raisonnement venait de MRIPP, où les
   cibles sont **rares et dispersées** (des fenêtres sur des façades). Chez nous, **chaque carton
   porte un QR et les cartons remplissent les racks** : détecter un QR à 5 m ne dit rien de neuf,
   ça dit « il y a un rack ici », et le rack est visible à 20 m. **La carte des cibles est déjà la
   carte des racks.**
3. **MRIPP mesure une tâche bien plus facile que la nôtre**, sur trois axes à la fois : le hasard
   y trouve déjà **40,08 %** des cibles (le plancher est haut) ; la portée du capteur vaut **30 %
   de la taille de l'environnement** (chez nous : 1 m sur 36×26 m, soit ~2 %) ; et **les robots
   n'ont aucune dynamique**, ils vont en ligne droite entre nœuds. Le transfert du « +25 points »
   n'est pas justifié.
4. **Le « trou » sur la panne d'un robot n'existe pas.** Il y a une littérature entière de
   couverture résiliente (Rabban & Tokekar, RA-L 2021 ; *Resilient Multi-Robot Coverage Path
   Redistribution*, Sensors 2024 ; arXiv 2407.19144 avec des chiffres après panne). L'agent avait
   cherché avec le vocabulaire du machine learning au lieu de celui de la robotique. **À ne pas
   présenter comme une découverte dans le mémoire** — un membre du jury le relèvera.

**Ce qui tient malgré tout, et c'est important.**

- La décomposition en étages est un vrai consensus, solidement documenté.
- **L'action discrète sur un graphe de points de vue élimine réellement le piège du bruit.**
  Quand l'action est un choix parmi 80 points de vue candidats, et non une vitesse continue, le
  bruit d'exploration ne peut plus « fabriquer » des lectures. C'est le meilleur apport de la
  direction.
- Le code de MRIPP est réel, en Python simple, sous licence MIT. C'est un point de départ concret
  (réserve : le dépôt ne contient que l'environnement abstrait en grille, pas de scène
  photo-réaliste ni de poids pré-entraînés).
- **BoofCV plutôt que pyzbar** : en gros plan, **100 % contre 12,5 %**. Confirmé, avec une réserve
  sur la neutralité du protocole.
- Sortir le rendu d'images de la boucle d'entraînement est le bon choix : Isaac Lab plante au-delà
  de 48 caméras sur un GPU de 96 Go, soit environ 2 Go par caméra. Intenable sur 8 Go.

> **Verdict : MITIGÉ, le meilleur des quatre. 15-25 %** d'atteindre l'objectif tel qu'énoncé ;
> **70-80 %** de produire un mémoire défendable. Ce sont deux objectifs différents et il faut
> l'assumer dès maintenant.

---

### D4 — Active Inference + module visuel

**Comment ça marcherait.** L'inférence active minimise une quantité appelée « énergie libre
attendue ». Cette quantité se sépare toute seule en deux morceaux : atteindre son but, et
**réduire son incertitude**. L'exploration n'est donc pas ajoutée à la main : elle sort du
calcul. La position des QR n'est pas dans le modèle — elle **est** l'inconnue à deviner, et le
point de départ est « je ne sais rien ».

**L'argument le plus élégant de toute l'étude.** Le terme dit « d'ambiguïté » pousse le drone à
se placer là où son capteur est **net** : proche, de face, lent. C'est exactement la manœuvre
difficile, et elle sortirait du calcul.

**La critique a tranché : le mécanisme est réel, mais l'argument est sur-vendu.** La manœuvre ne
tombe pas du calcul : elle tombe du **modèle du capteur**, qu'il faut écrire et calibrer à la
main. Écrire `P(décodage | distance, angle, vitesse)` demande exactement le même travail, la
même campagne de mesure et le même métier qu'écrire la récompense correspondante. **L'inférence
active ne supprime pas le réglage de la récompense : elle le déplace dans un autre fichier.**
Et un modèle faux est pire qu'une récompense imparfaite, parce que le planificateur planifie
avec assurance contre une réalité qu'il décrit mal.

**Les chiffres qui ferment la porte.**

| Fait vérifié | Chiffre |
|---|---|
| Record d'échelle en robotique (AIMAPP, arXiv 2510.09574) | 325 m², actions discrètes, **4,3 s par décision** |
| Extrapolé à notre échelle (936 m²) | **~19 s par décision** — même une couche haute à 1 Hz ne tient pas |
| Complexité de la planification | `O(|S|·|U|^T)` → 10¹⁸ opérations à horizon 30 |
| Licence du dépôt AIMAPP | **aucune** d'après la vérification par l'API GitHub → juridiquement inutilisable en l'état. *À reconfirmer avant de s'en servir : je n'ai pas pu refaire ce contrôle moi-même, GitHub étant bloqué depuis ma machine.* |
| Planning révisé par la critique | **14 à 20 semaines** pour 8 à 13 disponibles |

**Le résultat théorique qui décide.** Wei (arXiv 2408.06542) démontre que l'énergie libre
attendue **approxime la politique optimale du RL bayésien via la valeur de l'information**.
Autrement dit : **ce n'est pas une méthode différente, c'est une reformulation.** Si les deux
objectifs sont le même objet, on choisit celui qui a dix ans d'ingénierie derrière lui.
La communauté de l'inférence active l'admet d'ailleurs elle-même (arXiv 2508.05619) : les
hybrides « n'ont pas encore égalé » le RL pur à grande échelle.

**Sur Pore et al.** — le travail cité comme référence du domaine (Symmetry 2026, 18(4):548).
Lu et vérifié mot à mot : **la carte est connue à l'avance, la trajectoire d'allée est
pré-planifiée, et le QR encode lui-même son adresse** (`Item#A14-R5` = Rack A, étagère 14,
rangée 5). Ni inférence active, ni RL : des splines et un programme quadratique. Leurs
95,8 % / 90,5 % sont un **taux de décodage**, pas un taux de découverte — le drone passe
forcément devant tous les codes. **Chez eux, le problème dur n'existe pas.** À ne jamais citer
à côté de nos chiffres sans cette précision.

**Deux corrections en faveur de l'inférence active**, que personne n'avait vues : le « planificateur
de frontières bat AIMAPP » est trompeur, parce que ce planificateur n'a que 50 % de réussite
contre 100 % — en espérance, AIMAPP gagne. Et sur les trois entrepôts testés, AIMAPP fait
100/100/83, à égalité avec le meilleur planificateur classique. **L'apport réel et mesuré de
l'inférence active, c'est la robustesse, pas la performance.**

> **Verdict : MITIGÉ penchant FAIBLE. 8 %** comme direction complète.
> **Mais 60-65 % pour ses emprunts seuls** — voir le point 5.

---

## 4. La recommandation

### Direction retenue : **D3 — RL navigation + module visuel**, sous conditions

Trois raisons, dans l'ordre d'importance.

1. **C'est la seule direction dont la métrique correspond à la nôtre.** MRIPP mesure un
   pourcentage de cibles découvertes, en 3D, positions inconnues, plusieurs robots. Toutes les
   autres directions s'appuient sur des preuves de couverture de surface ou de navigation vers
   un but donné.
2. **Elle absorbe le meilleur des trois autres.** Le « module visuel » de D3 est exactement le
   rôle (c) de D2, celui qui a 75 % de chances de marcher. Le patron à deux étages vient aussi
   de D2. Les termes d'exploration de D4 s'y greffent comme récompenses. Choisir D3, ce n'est
   pas rejeter les propositions du promoteur : c'est prendre la partie solide de chacune.
3. **Elle est la seule qui rentre dans le budget.** Les méthodes sur graphe coûtent environ
   120 000 interactions au lieu de 10 millions d'images. C'est la seule famille qui tienne en
   8 Go, sans rendu dans la boucle, et en 2-3 mois.

### Mais pas telle qu'elle a été présentée. Quatre corrections obligatoires.

| # | Correction | Pourquoi |
|---|---|---|
| 1 | **Abandonner l'idée « repérer de loin, ralentir de près »** | Le facteur n'existe pas dans la source, et il ne servirait à rien puisque chaque carton porte un QR. |
| 2 | **Ne pas présenter la résilience aux pannes comme un trou de la littérature** | Il existe une littérature entière de couverture résiliente. Formulation honnête : « nous appliquons pour la première fois ce protocole d'évaluation à une politique apprise de recherche de cibles ». |
| 3 | **Budgéter le nombre de runs, pas la mémoire vidéo** | Une fois le rendu sorti de la boucle, la mémoire n'est plus un problème. Les vraies contraintes sont les 20 fils du processeur et le temps d'horloge : **10 à 15 runs au total**. Il faut donc une discipline stricte sur le nombre d'essais. |
| 4 | **Interdire tout curriculum à seuil** | MRIPP tire sa difficulté **au hasard** dans un intervalle, et c'est le bon remplacement. Un curriculum piloté par un taux de réussite est le mécanisme qui bloque les entraînements. |

### Le risque principal, à énoncer clairement

La direction 3 apprend **où aller**. Elle délègue à un contrôleur classique la tâche de
**se stabiliser devant le rack, à la bonne distance, à la bonne incidence, sous 0,5 m/s**. Or
c'est probablement la partie la plus difficile. Ce n'est pas « aller à un point de passage »,
c'est de l'asservissement visuel. **Aucun des douze travaux recensés n'a de contrainte de
vitesse à la lecture.** Ce morceau doit avoir des semaines dédiées dans le planning, sinon il
sera découvert trop tard.

---

## 5. Ce qu'on emprunte aux trois autres directions

Ce sont les vrais gains de cette étude. Coût total : 1 à 2 semaines.

**De D4 — l'inférence active (les trois emprunts les plus utiles) :**

1. **Le terme d'ambiguïté, transformé en récompense.** Mesurer une fois pour toutes, par
   balayage automatique, la table `P(décodage réussi | distance, angle, vitesse)`. Puis ajouter
   `−H[P(o|s)]` comme récompense **dense et potentielle** (forme `γ·Φ(s') − Φ(s)`, qui garantit
   de ne pas déformer l'optimum). Le drone reçoit un signal continu qui monte quand il se place
   là où le capteur est net, au lieu d'un signal binaire au moment du décodage. Justification
   citable en soutenance : Da Costa et al. 2020 pour la décomposition, Wei 2024 pour l'équivalence.
2. **Le terme épistémique sur la carte de couverture.** Traiter la carte comme une croyance par
   cellule et payer la **baisse d'entropie totale** à chaque pas. C'est littéralement le terme
   épistémique de l'énergie libre, calculé en temps linéaire au lieu d'exponentiel. On capture
   le bénéfice mesuré du domaine (78 % contre 28 % en scénario exigeant d'explorer) sans payer
   les secondes de planification.
3. **« Vu mais non décodé » comme observation à part entière.** Distinguer trois sorties du
   module visuel — décodé / motif repéré non décodé / rien — au lieu de deux. « Vu mais non
   décodé » est l'observation la plus riche : elle prouve qu'il y a une cible ici et crée un
   objectif à courte portée. Coût : quelques heures.

**De D2 — les modèles visuels :**

4. **Le patron à deux étages** : décision lente en haut, contrôle rapide en bas. Confirmé sans
   exception par une dizaine de travaux indépendants. C'est acquis et gratuit.
5. **Un détecteur de vision spécialisé pour la perception**, pas un modèle fondation. La
   littérature drone + codes-barres rapporte 74 à 100 % de détection et 81 à 99 % de décodage.
   Ça tient dans 2 Go.
6. **BoofCV comme décodeur**, pas pyzbar (100 % contre 12,5 % en gros plan).

**De D1 — Dreamer :**

7. **L'idée du modèle informé** (SkyDreamer) : donner la position des QR au réseau **pendant
   l'entraînement seulement**, jamais dans l'observation déployée. Compatible avec la contrainte
   du projet, et réutilisable dans n'importe quelle architecture.

---

## 6. La chose la plus importante de ce rapport

**Avant de choisir une méthode, il faut vérifier qu'il y a quelque chose à découvrir.**

Voici le calcul qu'aucune des huit fiches n'avait fait :

- 120 QR, 3 drones, épisode de 300 s → budget total **900 secondes-drone** ;
- soit **7,5 secondes par QR**, tout compris : déplacement, approche, stabilisation, décodage ;
- or un cycle de lecture réaliste (approche + stabilisation sous 0,5 m/s + une ou deux tentatives
  de décodage) coûte **4 à 8 secondes**.

**Les deux nombres se touchent.** Autrement dit : même avec une trajectoire parfaite fournie par
un oracle qui connaît tout, l'épisode suffit à peine. Dans ce cas, **le facteur qui décide du
résultat n'est pas « où aller », c'est « combien de temps prend une lecture »** — et « où aller »
est précisément ce que la direction 3 apprend.

À quoi s'ajoute un second problème, de sens inverse : dans un entrepôt aux racks alignés où
chaque carton porte un QR, **un balayage systématique des faces de racks est déjà presque
optimal**. La régularité de l'entrepôt, que l'on prenait pour un avantage (« il y a une
structure à apprendre »), est en fait un inconvénient : **une structure forte veut dire qu'un
motif fixe capture déjà presque tout**, et qu'il ne reste rien à gagner pour un réseau.

### L'expérience de 3 jours qui tranche — sans GPU, sans rendu, sans article à lire

**Jour 1.** Coder un modèle de lecture géométrique (distance, incidence, vitesse, occlusion,
champ de vue) sur un entrepôt de 36 × 26 m avec 120 QR, **avec une vraie dynamique de drone**
(accélération bornée, retard d'actionneur). La dynamique n'est pas optionnelle : sans elle, on
mesure un problème plus facile que le vrai.

**Jour 2.** Coder deux politiques **non apprises** :
- (a) un **balayage fixe** de chaque face de rack à 0,4 m/s ;
- (b) un **glouton omniscient** : « aller au QR non lu le plus proche », avec toutes les positions
  connues.
Les évaluer avec 3 drones sur 10 dispositions jamais vues, dans le vrai budget d'épisode.

**Jour 3.** Lire deux nombres.

1. **N_oracle** — ce que lit le glouton omniscient.
   **Si N_oracle est déjà loin de 100 %, la tâche est limitée par le temps, pas par
   l'exploration.** Aucune méthode d'apprentissage n'y changera quoi que ce soit.

2. **Δ = N_oracle − N_balayage** — c'est **toute** la marge disponible pour n'importe quelle
   méthode intelligente.
   - **Δ > 30 points** → la direction 3 mérite ses six semaines d'entraînement. On y va.
   - **Δ entre 15 et 30** → on y va, mais en réduisant l'ambition affichée.
   - **Δ < 15 points** → **ne pas lancer d'entraînement.** Le balayage capture déjà la structure,
     et la littérature montre que le gain de l'apprentissage sur ce genre de tâche est de 0 à
     12 %, c'est-à-dire sous le bruit de mesure.

**Et si Δ est petit, que fait-on ?** On ne change pas de méthode : **on change l'instance du
problème.** Puisqu'on repart de zéro, la taille de l'entrepôt, le nombre de QR, la durée
d'épisode et la disposition des racks sont tous négociables. Pour qu'il y ait quelque chose à
découvrir, il faut au moins un de ces ingrédients :
- **des racks dont la disposition varie** d'un entrepôt à l'autre (pas seulement le contenu) ;
- **un remplissage partiel et irrégulier** — beaucoup d'emplacements vides, donc une vraie
  incertitude sur *où* sont les cartons ;
- **un budget de temps serré** qui force à prioriser au lieu de tout balayer ;
- **des événements en cours de mission** : panne, cartons ajoutés ou déplacés.

C'est une décision de conception de la tâche, et elle doit être prise **avant** de choisir les
hyperparamètres, pas après trois runs ratés.

**Expérience de second rang, 1 jour, juste après :** chronométrer un cycle de lecture complet.
Si le cycle dépasse 5 secondes, c'est ce chiffre-là qu'il faut attaquer en priorité — pas la
stratégie d'exploration.

---

## 7. Les décisions à prendre ensuite (pas tranchées ici)

Ce rapport choisit une **direction**, pas une implémentation. Restent à décider, dans cet ordre :

1. **L'instance du problème** — taille, nombre de QR, budget de temps, ce qui varie d'un
   entrepôt à l'autre. À décider après l'expérience de 3 jours, sur la base de Δ.
2. **Le simulateur** — Isaac Sim n'est plus obligatoire. Comme le rendu d'images sort de la
   boucle d'entraînement, un simulateur léger et rapide suffit pour apprendre, et un simulateur
   réaliste sert seulement à valider. C'est un choix ouvert, à faire une fois l'instance fixée.
3. **La manière d'entraîner** — algorithme, représentation de l'état, structure de récompense.
   Rien n'est repris de l'ancien projet.
4. **Le protocole d'évaluation** — split d'entrepôts jamais vus, matrice de résilience (panne,
   inventaire modifié, obstacles), et **une ligne de base non apprise gardée en permanence**
   (le balayage fixe). Publier la comparaison honnête, y compris si elle est défavorable.

---

## 8. Le plan B

**Enseignant–élève (distillation).** Un « enseignant » qui connaît les positions des QR guide
un « élève » qui ne voit que ce qu'un vrai drone verrait. Cette direction a déjà été étudiée et
validée sur le principe, hors du périmètre de ce rapport. Elle reste la solution de repli si
l'expérience de 3 jours ou les premières semaines d'entraînement tournent mal.

**Condition de bascule à fixer maintenant** : si à mi-parcours la politique apprise ne bat pas
le balayage fixe de plus de 10 points, basculer sur le plan B sans discuter.

---

## Annexe A — Erreurs trouvées et corrigées

Les agents critiques ont vérifié une soixantaine d'affirmations chiffrées à la source.
**Aucune référence n'était inventée** — toutes existent. Mais huit affirmations étaient mal
rapportées, et il faut les corriger **avant** qu'elles entrent dans le mémoire.

| # | Affirmation | Ce qu'il faut écrire à la place |
|---|---|---|
| 1 | « À 2-2,5 pixels par module, la **détection** commence » | **Faux.** L'article ne mesure jamais la détection seule. C'est le point où le **décodage** cesse d'être nul. Le facteur « repérer vs décoder » n'est pas établi. |
| 2 | « La revue arXiv 2607.06706 est catégorique : aucune plateforme aérienne dans Open X-Embodiment » | La revue ne dit pas ça. Écrire : « le site officiel d'Open X-Embodiment décrit ses 22 plateformes comme des bras simples, bimanuels et des quadrupèdes ; aucune plateforme aérienne n'y figure ». |
| 3 | « Personne ne mesure la performance après la panne d'un robot » | **Faux.** Littérature de couverture résiliente : Rabban & Tokekar RA-L 2021, Sensors 2024, arXiv 2407.19144. |
| 4 | « Les VLM font à peine mieux que le hasard en raisonnement spatial » | Tronqué. MINDCUBE monte à **61,3 %** après entraînement adapté. Le « niveau du hasard » de MV-RoboBench vaut pour les modèles de 3-4 milliards sur une seule sous-tâche. |
| 5 | « Le planificateur de frontières FAEL bat AIMAPP » | Trompeur : le chiffre est calculé sur les essais **réussis**, et FAEL n'a que 50 % de réussite contre 100 %. En espérance, AIMAPP gagne. |
| 6 | « r2dreamer est 5× plus rapide » | Le papier annonce **1,59×**. |
| 7 | Détection 74-100 % / décodage 81-99 % attribués à *Scientific Reports* | Mauvaise source. Ces chiffres viennent de ScienceDirect S2667305326000232. *Scientific Reports* donne 92,4 % de mAP. |
| 8 | « SmolVLA : 0,9 Go, 18 ms » | **Absent de la source officielle.** Ne pas citer. |

Deux travaux déclarés inexistants existent bel et bien et doivent être cités : **SAGE**
(arXiv 2605.23160) et **GoalSwarm** (arXiv 2603.12908). Les ignorer serait le point le plus
facile à attaquer en soutenance.

Deux ressources annoncées comme réutilisables ne le sont pas : **AIMAPP** n'a **aucune licence**
et **LiteVLA-H** n'a ni code ni poids publics.

**Contrôle final effectué par mes soins**, en plus des critiques : j'ai revérifié quatre
affirmations centrales directement à la source. **MRIPP** — confirmé (« maximize the number of
discovered stationary targets in an unknown 3D environment », amélioration d'au moins 26,2 %
sur l'état de l'art). **Wei 2408.06542** — confirmé mot pour mot (« We show that EFE
approximates the Bayes optimal RL policy via information value »). **Pore et al.** — confirmé
(Symmetry 2026, 18(4):548, 24 mars 2026 ; 95-96 % en simulation, 86-90,5 % en réel).
**Licence d'AIMAPP** — non revérifiable depuis ma machine, GitHub étant bloqué.

---

## Annexe B — Sources principales

**Direction 1 — Dreamer / World Model**
Dream to Fly, arXiv 2501.14377 · DreamerNav, Frontiers Robotics AI 2025, doi 10.3389/frobt.2025.1655171 ·
Generalization of World Models under Environmental Variability, arXiv 2606.05015 (code : github.com/ntnu-arl/world-model-nav-generalization) ·
NavDreams, arXiv 2203.12299 · Recall to Imagine (R2I) · DIAMOND, NeurIPS 2024 ·
SkyDreamer, arXiv 2510.14783 · Swift, Nature 2023

**Direction 2 — Visual Models / VLA / LLM**
CognitiveDrone, arXiv 2503.01378 · OpenVLA, arXiv 2406.09246 · OpenVLA-OFT · π0 / openpi ·
SmolVLA (huggingface.co/blog/smolvla) · LiteVLA-H, arXiv 2605.00884 · SmolRGPT, arXiv 2509.15490 ·
VLFM, arXiv 2312.03275 · VL-Explore (ex-ClipRover), arXiv 2502.08791 · COMRES-VLM, arXiv 2509.26324 ·
SAGE, arXiv 2605.23160 · Autonomous Frontier-Based Exploration with VLM Guidance, arXiv 2605.23165 ·
GoalSwarm, arXiv 2603.12908 · V*/SEAL, arXiv 2312.14135 · MINDCUBE, arXiv 2506.21458 ·
revue VLA/UAV, arXiv 2607.06706

**Direction 3 — RL navigation + module visuel**
MRIPP, arXiv 2409.16967 (code MIT) · Active Neural SLAM, ICLR 2020 · SemExp · MAANS ·
ARiADNE, ICRA 2023 · CAtNIPP · Explore-Bench, ICRA 2022 ·
QR Code Detectors near Nyquist Limits, Sensors 2022 · benchmark de décodeurs Dynamsoft (536 images) ·
Failure-Resilient Coverage Maximization, RA-L 2021, arXiv 2007.02204 ·
Resilient Multi-Robot Coverage Path Redistribution, Sensors 2024 ·
Collaborative Adaptation for Recovery from Unforeseen Malfunctions, arXiv 2407.19144 ·
ACE, arXiv 2301.03398 · pile modulaire sim-to-real, Science Robotics
Détection de codes sur drone : ScienceDirect S2667305326000232 · Scientific Reports s41598-025-29720-w

**Direction 4 — Active Inference**
AIMAPP, arXiv 2510.09574 (⚠ sans licence) · Pore et al., Symmetry 2026 18(4):548, doi 10.3390/sym18040548 ·
Wei, arXiv 2408.06542 · Champion et al., arXiv 2303.01618 · The Missing Reward, arXiv 2508.05619 ·
complexités de l'EFE, arXiv 2307.00504 · ambiguïté en continu, arXiv 2409.01974 ·
AIF multi-agent, arXiv 2501.03907 · pymdp (MIT) · RxInfer.jl (MIT)
