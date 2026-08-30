# Les 4 directions, construites concrètement pour notre problème

Date : 28 août 2026
Ce document vient **avant** l'étude de la littérature (`etude_4_directions.md`). Il répond à une
seule question : **à quoi ressemblerait chaque solution si on la construisait chez nous ?**
Entrées → modèle → entraînement → sortie → boucle de décision → contrôle → décodage.

Les architectures ci-dessous sont **hypothétiques**. Personne ne les a construites. Je les
propose comme la version la plus raisonnable de chaque idée, celle que je construirais si on
me demandait de le faire. La critique vient après, et elle porte sur **ces** architectures
précises — pas sur « Dreamer en général ».

---

## 0. Le squelette commun — la clé pour comparer

Avant de regarder les différences, il faut voir ce qui est **identique dans les quatre
directions**. Quelle que soit la méthode, un drone qui découvre des QR codes a besoin de six
briques :

| # | Brique | Rôle |
|---|---|---|
| **B1** | **Capteur** | Transformer ce que voit la caméra en information utilisable |
| **B2** | **Mémoire** | Se souvenir de ce qui a déjà été vu et lu |
| **B3** | **Décision** | Choisir où aller ensuite |
| **B4** | **Contrôle** | Voler jusque-là, et tenir la pose de lecture |
| **B5** | **Lecteur** | Décoder le QR |
| **B6** | **Coordination** | Répartir le travail entre les 3 drones |

**B5 n'est jamais apprise.** Décoder un QR est un problème résolu depuis vingt ans par des
bibliothèques (BoofCV, ZBar, OpenCV). Aucune des quatre directions ne propose d'apprendre ça,
et ce serait absurde de le faire.

**Les quatre directions ne diffèrent que sur une chose : quelles briques sont apprises, et
comment.**

| Direction | B1 Capteur | B2 Mémoire | B3 Décision | B4 Contrôle | B6 Coordination |
|---|---|---|---|---|---|
| **D1 Dreamer** | apprise | apprise (dans le réseau) | apprise | apprise | apprise (implicite) |
| **D2 VLA** | pré-entraînée | dans le contexte du modèle | modèle fondation | classique | par le langage |
| **D3 RL + vision** | classique | construite à la main | **apprise** | classique | à la main (carte partagée) |
| **D4 Active Inference** | classique | croyance probabiliste | planificateur EFE | classique | à la main (croyance partagée) |

C'est le tableau le plus utile du document. **D1 apprend tout. D3 et D4 n'apprennent presque
rien — et ils ne diffèrent entre eux que par la brique B3.**

Gardez ça en tête : à la fin, la vraie question ne sera pas « quelle direction ? », mais
**« la brique B3 doit-elle être apprise ou écrite à la main ? »**.

---

## D1 — Dreamer / World Model

### L'idée en une phrase

Le drone apprend un **simulateur mental** : un réseau qui prédit ce qu'il va voir juste après.
Une fois ce simulateur appris, il s'entraîne **dans sa tête**, en imagination, sans avoir
besoin de refaire des millions de vols.

### Deux versions possibles — il faut choisir

**Version A — « pixels purs »**, celle que votre promoteur imaginait : le modèle reçoit les
images brutes et doit tout apprendre, y compris voir les QR codes.

**Version B — « hybride »** : le modèle reçoit les images *et* la sortie d'un détecteur
classique *et* une carte.

Je décris les deux, parce que la différence entre elles est le cœur du jugement.

### Version A — les entrées, exactement

À chaque pas de décision (on tourne la politique à 10 Hz, avec chaque action répétée 3 fois
pour retomber sur les 30 Hz du vol) :

| Entrée | Forme | Taille |
|---|---|---|
| Image caméra gauche | RGB 64 × 64 | 12 288 valeurs |
| Image caméra droite | RGB 64 × 64 | 12 288 valeurs |
| Proprioception | vitesse (3), altitude (1), cap (2 : sin/cos), temps restant (1) | 7 valeurs |
| Action précédente | vx, vy, vz, ω | 4 valeurs |

C'est tout. **Pas de carte, pas de position de QR, pas de détecteur.** C'est la version pure.

### Le modèle, morceau par morceau

DreamerV3 est fait de cinq réseaux qui s'entraînent ensemble :

1. **L'encodeur.** Il prend l'observation et la compresse en un petit code appelé *état
   latent* `z_t`. Dans DreamerV3, ce code est fait de 32 variables catégorielles à 32 valeurs
   chacune — donc environ 1024 bits pour résumer tout ce que le drone voit à cet instant.

2. **Le RSSM** (le cœur du modèle du monde). C'est une mémoire récurrente `h_t`. Elle fait
   deux choses :
   - `h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})` — elle avance dans le temps ;
   - à partir de `h_t` seul, elle **prédit** quel sera `z_t` (c'est le *prior*), avant même
     d'avoir vu l'image.
   L'écart entre ce qu'elle prédit et ce que l'encodeur produit vraiment, c'est ce qui
   l'entraîne. **C'est là que vit « le modèle du monde ».**

3. **Le décodeur.** Il reconstruit l'image à partir de `(h_t, z_t)`. Il sert uniquement à
   forcer le latent à contenir de l'information visuelle.

4. **Les têtes de prédiction.** Deux petits réseaux qui prédisent, à partir de `(h_t, z_t)` :
   la récompense du pas, et si l'épisode continue.

5. **L'acteur et le critique.** L'acteur sort l'action, le critique estime la valeur. Ils ne
   voient **jamais** l'image : ils ne voient que `(h_t, z_t)`.

Taille totale, préréglage « 12M » : environ 12 millions de paramètres, ce qui tient largement
dans 8 Go.

### L'entraînement, concrètement

C'est une boucle à deux vitesses qui tourne en continu :

```
répéter :
    1. COLLECTE  — voler dans le simulateur avec l'acteur actuel,
                   stocker (obs, action, récompense, fin) dans un tampon de rejeu
    2. MODÈLE    — tirer 16 séquences de 64 pas dans le tampon,
                   entraîner encodeur + RSSM + décodeur + têtes
                   (perte = reconstruction d'image + prédiction de récompense + KL)
    3. IMAGINER  — partir de chaque état latent du lot, dérouler 15 pas
                   PUREMENT DANS LE MODÈLE (aucune image, aucun simulateur)
    4. POLITIQUE — entraîner l'acteur et le critique sur ces trajectoires imaginées
```

Le point important, et c'est tout l'intérêt de la méthode : **l'étape 4 ne coûte rien en
simulation.** L'acteur s'améliore en rêvant. C'est pour ça que Dreamer a besoin de beaucoup
moins de vols réels que PPO.

**Ce qu'il apprend exactement :** la physique du drone (« si je commande vx = 0,8, où je serai
dans 3 pas »), l'apparence de l'entrepôt (« un rack, ça ressemble à ça »), et le lien entre ce
qu'il voit et la récompense (« quand cette texture est grande et centrée, un +1 arrive
souvent »).

### La sortie et la boucle d'exécution

Une fois entraîné, il n'y a **aucune planification**. La boucle est purement réactive, avec
mémoire :

```
à chaque pas (10 Hz) :
    z_t     = encodeur(image_gauche, image_droite, proprio)
    h_t     = RSSM(h_{t-1}, z_{t-1}, a_{t-1})
    action  = acteur(h_t, z_t)        →  (vx, vy, vz, ω)
    envoyer l'action au drone, la répéter 3 fois
```

**Il n'y a pas de contrôleur séparé.** L'acteur sort directement les vitesses. Le world model
ne sert **pas** à planifier au moment du vol : il a servi à entraîner l'acteur, puis il ne sert
plus qu'à maintenir la mémoire `h_t`.

C'est un point que beaucoup de gens comprennent mal, et il compte pour la suite : **Dreamer
n'est pas un planificateur. C'est une manière d'entraîner une politique réactive.**

### Comment il découvrirait les QR sans connaître leur position

Par la récompense, et seulement par elle. On paie `+1` chaque fois qu'un QR **nouveau** est
décodé. Rien d'autre n'est dit à l'agent.

L'exploration vient de deux sources :
- le bruit de l'acteur (il tire ses actions au hasard autour de sa moyenne) ;
- optionnellement **Plan2Explore** : on entraîne un ensemble de 5 à 10 petits réseaux qui
  prédisent chacun le prochain latent, et on donne une récompense bonus là où ils sont **en
  désaccord**. L'idée : là où le modèle est incertain, il y a quelque chose à apprendre.

### Qui fait quoi

| Brique | Qui la fait, en version A |
|---|---|
| B1 Capteur | l'encodeur (appris) |
| B2 Mémoire | le RSSM, le vecteur `h_t` (appris) |
| B3 Décision | l'acteur (appris) |
| B4 Contrôle | l'acteur (appris) — pas de contrôleur séparé |
| B5 Lecteur | **externe** : une bibliothèque de décodage tourne en parallèle et déclenche la récompense |
| B6 Coordination | apprise implicitement : chaque drone voit ses coéquipiers dans ses images |

### Ce qui doit être vrai pour que ça marche

C'est la section qui décide de tout. Cinq conditions, et je donne le chiffre pour chacune.

**Condition 1 — que voit-on vraiment dans une image de 64 × 64 ?**
*(Mesuré le 29/08/2026 — voir `experiments/01_resolution_qr/RESULTATS.md`. Une version
antérieure de ce document présentait cette condition comme bloquante : c'était une erreur, et
la mesure l'a corrigée.)*

Il faut séparer trois capacités, qui n'ont pas du tout les mêmes exigences :

| Distance | Un rack | Un carton (~40 cm) | Le QR (7,3 cm mesuré) |
|---|---|---|---|
| 1 m | plein écran | **22 px** | 4 px |
| 2 m | plein écran | **11 px** | 2 px |
| 4 m | bien visible | **5 px** | 1 px |
| 8 m | visible | 3 px | invisible |

Donc, en 64 × 64, le world model **voit parfaitement les racks, les allées et les cartons**.
Il peut naviguer, éviter les obstacles, choisir un carton et s'en approcher. Ce qu'il ne voit
pas, c'est **le QR lui-même** : 2 à 4 pixels, indiscernable d'une étiquette ou d'une ombre.

Seuils mesurés : décodage à **2,07 px/module**, repérage du motif à **1,38 px/module**. En
64 px d'entrée, la portée de décodage tombe à **7,5 cm** et celle de repérage à **10 cm**.

**Conséquence, et elle est moins grave que je ne l'avais écrit :**
- **Le décodage sort du world model** — un décodeur classique tourne en parallèle sur l'image
  en pleine résolution et déclenche la récompense. Ce n'est pas une concession : personne
  n'apprend à décoder un QR, dans aucune des quatre directions.
- **La navigation reste possible en 64 × 64.** La version A n'est donc pas impossible.
- Le world model ne peut **viser que le carton, pas le QR**. Mais chez nous chaque carton porte
  un QR sur sa face : **viser le carton, c'est viser le QR.** La cible n'a pas besoin d'être
  visible.

Ce dernier point est important, et il ne joue pas contre Dreamer — il joue contre l'idée qu'il
y a beaucoup à découvrir. Si les cartons sont visibles à 8 m sur des racks alignés,
l'incertitude sur *où aller* est faible.

**Contrainte matérielle qui en sort, valable pour toutes les directions :** la caméra de
**lecture** doit faire au moins **1000 pixels de large** à 60° de champ pour décoder à 1,25 m.
C'est une caméra séparée de celle qui alimente la politique.

**Condition 2 — la mémoire doit tenir 1800 pas.**
Le drone doit se souvenir de quels QR il a lus depuis le début de l'épisode. Un GRU comme celui
du RSSM tient des dépendances de l'ordre de quelques dizaines de pas. Il n'y a aucune chance
qu'il retienne 111 identifiants pendant 1800 pas.

**Conséquence : il faut une carte externe.** On construit une grille de couverture à la main,
et on l'injecte dans l'observation. Le world model ne mémorise plus rien d'important : la
mémoire est dehors.

**Condition 3 — la récompense doit être atteignable au hasard.**
Au début, l'acteur vole au hasard. Pour toucher `+1`, il faut satisfaire **en même temps**
quatre conditions : être à moins de 1,25 m, être à moins de 50° de la normale, aller à moins
de 0,6 m/s, et tenir ça deux images de suite. La probabilité qu'un vol aléatoire réalise cette
conjonction est très faible.

**Si la récompense n'arrive jamais, le world model apprend parfaitement à prédire un monde
où il ne se passe rien.** L'imagination est alors inutile : l'acteur rêve d'un entrepôt sans
récompense. Il faut donc ajouter une récompense de forme — donc refaire le travail de
conception de récompense qu'on voulait éviter.

**Condition 4 — le tampon de rejeu doit tenir sur le disque.**
Dreamer garde toutes ses images. Avec 3 drones × 2 caméras en 64 × 64 RGB, c'est **74 Ko par
pas**. Pour 500 000 pas : **37 Go**. La machine a **20 Go libres**. On serait donc limité à
environ 150 000 pas, soit **30 à 80 épisodes rejoués en boucle**.

C'est un problème sérieux et personne ne l'avait vu : on demande à l'agent de **généraliser à
des entrepôts jamais vus** alors qu'il ne pourrait s'entraîner que sur quelques dizaines de
dispositions.

**Condition 5 — le temps.** Une itération d'entraînement Dreamer coûte cher : il faut avancer
le simulateur **et** faire tourner l'encodeur, le RSSM et le décodeur sur des séquences. Sur
une 2060 SUPER, un run complet prendrait de l'ordre de **une à deux semaines**. Sur 2-3 mois,
ça laisse **3 à 6 runs**. C'est très peu pour régler une méthode qu'on ne connaît pas.

### La version B — ce qu'il reste après les correctifs

Après avoir appliqué les conditions 1 et 2, l'architecture devient :

| Entrée | Forme |
|---|---|
| Images 64 × 64 (gauche, droite) | pour la navigation et les obstacles seulement |
| **Sortie du détecteur** : jusqu'à 8 QR repérés, chacun (u, v, taille, décodé oui/non) | 8 × 4 |
| **Carte de couverture égocentrique** externe, 3 échelles × 32 × 32 × 4 canaux | construite à la main |
| Proprioception | 7 valeurs |

Et là il faut être honnête sur ce qu'on vient de faire : **on a sorti du world model la
perception fine et la mémoire.** Il ne lui reste qu'à modéliser la physique du drone et
l'apparence générale de l'entrepôt.

### Ce que j'en pense vraiment

Trois choses, indépendamment de ce que disent les papiers.

**1. Le raisonnement du promoteur était juste, à une nuance près.**
L'idée « le world model verra les QR avec la caméra » est presque bonne. La mesure montre qu'il
verra les **racks et les cartons** très bien, mais pas le **motif du QR**. Comme personne
n'apprend à décoder un QR de toute façon, ce n'est pas bloquant : un décodeur classique tourne
à côté. **La résolution n'est donc pas ce qui disqualifie Dreamer** — je l'avais écrit, c'était
faux, et la mesure du 29/08 l'a corrigé.

**2. Après correctifs, l'avantage disparaît.** Le bénéfice de Dreamer, c'est l'efficacité en
données. Mais chez nous les données sont **gratuites** — on a un simulateur, pas un vrai drone
qui casse. L'efficacité en données ne sert à rien quand la contrainte est le temps de calcul,
et là, PPO voit 20 à 100 fois plus de situations par heure de GPU.

**3. Le vrai problème est le débogage.** C'est mon argument principal et il ne vient d'aucun
papier. Avec 3 à 6 runs disponibles, ce qui compte n'est pas la performance théorique, c'est
la **capacité à savoir pourquoi ça rate**. Dans Dreamer, quand le résultat est mauvais, la
cause peut être : l'encodeur, le RSSM, la tête de récompense, l'acteur, le critique, le ratio
imagination/collecte, ou la récompense elle-même. Sept suspects, une seule mesure. Ce projet a
déjà perdu des semaines sur **un seul** bug non localisé. Ajouter six suspects de plus est le
contraire de ce qu'il faut faire.

> **Ce que je retiens de D1 :** une seule idée, transposable partout — le **modèle informé**.
> Donner la position des QR au réseau **pendant l'entraînement**, jamais à l'exécution. Ça ne
> demande pas Dreamer.

---

## D2 — Visual Models / VLA / LLM

### L'idée en une phrase

Utiliser un gros modèle déjà entraîné sur des millions d'images et de textes, en espérant qu'il
apporte du « bon sens » que notre petit réseau n'aurait pas.

Il y a trois façons très différentes de s'en servir. Il faut absolument les séparer, parce
qu'elles n'ont ni le même coût ni les mêmes chances.

### Version A — le modèle pilote le drone

**Entrées.** L'image de la caméra (redimensionnée en 224 × 224 ou 336 × 336) + une phrase :
« scanne les étagères et lis les codes que tu n'as pas encore lus ».

**Le modèle.** OpenVLA : un encodeur d'image (SigLIP + DINOv2) branché sur un modèle de langage
Llama-2 de 7 milliards de paramètres. La sortie n'est pas du texte : les actions sont découpées
en 256 valeurs, et chaque valeur devient un « mot » que le modèle prédit.

**Sortie.** Un bloc de 8 actions d'affilée : `(vx, vy, vz, ω) × 8`. On les rejoue à 30 Hz, ce
qui laisse au modèle ~270 ms pour produire le bloc suivant.

**Entraînement.** On ne l'entraîne pas de zéro. On fait du **réglage fin par LoRA** : on gèle
les 7 milliards de paramètres et on ajoute de petites matrices adaptatrices (~97 millions de
paramètres). Les données d'entraînement sont des **démonstrations** : il faut donc d'abord
écrire un pilote automatique qui fait la tâche correctement, enregistrer quelques milliers de
vols, et apprendre au modèle à l'imiter.

**Ce qui doit être vrai.**

| Condition | Chiffre |
|---|---|
| Faire tourner le modèle | 7,0 Go en 4 bits, mais alors **3 Hz** — il en faut 30 |
| Le faire tourner assez vite (108 Hz) | **16 à 19 Go** |
| Faire le réglage fin par LoRA | **27 Go minimum** (documentation officielle) |
| Mémoire réellement disponible chez nous | **2 à 4 Go** (le simulateur prend le reste) |

**Mon avis, franchement.** Il y a un problème de logique avant même le problème de mémoire.
Pour entraîner cette version, **il faut d'abord écrire le pilote automatique qui réussit la
tâche.** Mais si on a ce pilote, on a déjà résolu le problème. On utiliserait alors un modèle
de 7 milliards de paramètres pour imiter un programme qu'on vient d'écrire — en le rendant
plus lent, plus lourd, et moins bon que l'original.

C'est exactement la distillation enseignant-élève, mais avec un élève 1000 fois trop gros.
Si on veut faire de la distillation, on la fait avec un réseau de 2 millions de paramètres.

### Version B — le modèle comme « cerveau lent »

**Entrées.** Toutes les 3 à 5 secondes : une vue de la carte construite jusqu'ici, ou une image
panoramique, plus une question du type « parmi ces 5 zones candidates, laquelle contient le
plus probablement des cartons non encore scannés ? ».

**Sortie.** Un numéro de zone. Un planificateur classique fait le reste.

**Ce que ça coûte.** Un petit modèle (SmolRGPT, 600 millions de paramètres, conçu justement
pour les entrepôts) tient dans ~2 Go. Techniquement, c'est faisable.

**Mon avis, et c'est mon argument principal contre cette direction.**

Il faut se demander : **qu'est-ce que le modèle sait, que la carte ne sait pas déjà ?**

La carte sait déjà quelles cellules ont été couvertes, où sont les murs vus, où sont les
détections. Le modèle ne peut apporter qu'une chose : un **a priori sur ce qu'on n'a pas encore
vu** — du type « les racks continuent probablement derrière ce pilier ».

Or, **c'est nous qui générons les entrepôts.** Ils sortent de notre propre code de génération
procédurale. Donc tout a priori structurel que le modèle pourrait deviner, **nous l'avons
écrit nous-mêmes** et nous pouvons le donner directement en une ligne. On paierait 600 millions
de paramètres et 2 Go pour faire deviner à une machine une règle que nous avons tapée au
clavier.

C'est différent d'un robot dans un appartement inconnu, où l'a priori « la cuisine est près de
la salle à manger » vient du monde réel et n'est écrit nulle part. Chez nous, il est écrit dans
`layouts.py`.

Deuxième point : un entrepôt est une **grille régulière d'allées parallèles**. C'est le cas où
un balayage géométrique est déjà presque optimal. Le bon sens apporte quelque chose quand la
géométrie trompe — culs-de-sac, pièces cachées. Une allée droite ne trompe personne.

### Version C — le modèle de vision comme œil

**Entrées.** L'image en pleine résolution.
**Sortie.** Des boîtes autour des motifs de QR détectés, avec leur position 3D estimée.
**Le modèle.** Un détecteur spécialisé (type YOLO), quelques millions de paramètres, ~200 Mo.
Pas un modèle fondation.

**Mon avis : c'est utile, ça tient dans notre budget, et il faut le faire.** Mais ce n'est plus
« la direction 2 » — c'est la brique B1 de n'importe quelle architecture. C'est le « module
visuel » des directions 3 et 4.

> **Ce que je retiens de D2 :** le **patron à deux étages** (une décision lente, un contrôle
> rapide) et **la vision comme perception**, avec un petit détecteur spécialisé. Le modèle
> fondation lui-même n'a pas de rôle chez nous.

---

## D3 — RL navigation + module visuel

C'est la direction que je recommande, donc c'est celle que je dois décrire le plus précisément.

### L'idée en une phrase

On sépare le problème en trois étages qui peuvent être testés **séparément** : un œil qui
détecte, une tête qui choisit où aller, des jambes qui volent. Seule la tête apprend.

### B1 — L'œil (pas appris)

Sur chaque image, en pleine résolution (par exemple 1280 × 960), on fait tourner un détecteur
de QR. Il produit **trois sorties possibles**, et cette distinction est la plus importante de
toute l'architecture :

| Sortie | Ce que ça veut dire | Ce qu'on en fait |
|---|---|---|
| **Décodé** | on a l'identifiant du carton | on marque la cellule « lue », `+1` de récompense |
| **Motif repéré, non décodé** | il y a un QR ici, mais trop loin / trop de biais / trop flou | **c'est l'information la plus riche** : on sait où aller, et qu'il y a quelque chose à gagner |
| **Rien** | — | — |

Pour les deux premières, on estime la **position 3D** du code par `solvePnP` (on connaît la
taille physique du QR, donc sa distance et son orientation se déduisent de sa forme dans
l'image).

En parallèle, un capteur de distance (lidar ou profondeur) alimente l'occupation.

### B2 — La mémoire (construite à la main, pas apprise)

Une **carte 3D partagée par les trois drones**, en cellules de 0,25 à 0,5 m. Chaque cellule
porte quelques valeurs :

| Canal | Valeurs |
|---|---|
| Occupation | libre / occupé / inconnu |
| Couverture | cette cellule a-t-elle été observée par une caméra, et sous quel angle |
| Carton non lu | un motif a été repéré ici mais pas décodé (avec sa normale estimée) |
| Lu | un code a été décodé ici |
| Coéquipiers | où sont les autres drones et où ils ont annoncé qu'ils vont |

Point important : **on code à la main la *structure* du monde (il est fait de surfaces
verticales qui portent des cartons), pas son *contenu* (où sont ces surfaces, lesquelles
portent des cartons non lus).** Le contenu reste entièrement à découvrir. Ce n'est pas de la
triche : c'est l'équivalent de savoir qu'une voiture roule sur des routes.

### B3 — La tête (la seule partie apprise)

C'est le cœur de la direction. Elle tourne **environ une fois par seconde**, ou dès que le
drone a atteint son objectif précédent — pas à 30 Hz.

**Observation :**

| Élément | Forme |
|---|---|
| Carte égocentrique multi-échelle | 3 échelles × 32 × 32 × 5 canaux (grossier : tout l'entrepôt ; fin : les 8 m autour) |
| État propre | vitesse, altitude, cap, temps restant |
| État des coéquipiers | position relative, objectif annoncé, vivant ou non |
| Liste des points de vue candidats | ~50 candidats, chacun décrit par : position relative, cap visé, distance, nombre de cellules non lues qu'il permettrait d'observer, occupé par un coéquipier ou non |

**Action : choisir un point de vue parmi les candidats.** Pas une vitesse. Un point de vue,
c'est une position **et** une direction de regard — parce que chez nous, être au bon endroit
mais regarder ailleurs ne sert à rien.

Les candidats sont générés à chaque décision par une règle simple : pour chaque cellule
« carton non lu » ou « inconnue » de la carte, on calcule le point d'où on pourrait la lire
(à 1,08 m de la face, en face). On en garde une cinquantaine, les plus proches et les plus
prometteurs.

**Le réseau.** Un petit encodeur convolutif sur les cartes (≈ 500 000 paramètres), un encodeur
de vecteur sur l'état, puis une **attention** entre l'état et les candidats qui sort un score
par candidat. Total : environ **1 à 2 millions de paramètres**. C'est minuscule, ça tient dans
quelques centaines de Mo, et ça s'entraîne vite.

Ce type de sortie (« choisir un élément dans une liste de taille variable ») s'appelle un
réseau pointeur. L'avantage : ça marche que la liste fasse 10 ou 200 candidats.

**Entraînement.** PPO, dans un simulateur **sans rendu d'image**. Le détecteur est remplacé par
un modèle géométrique calibré : « si le drone est à cette distance, cet angle, cette vitesse,
alors le décodage réussit avec cette probabilité ». Cette table est mesurée une fois pour
toutes par un balayage automatique avec le vrai décodeur.

**Récompense.** `+1` par QR nouvellement décodé par l'équipe, plus une petite pénalité de temps,
plus deux termes de forme empruntés à la direction 4 (voir plus bas). Rien d'autre.

**Ce qu'il apprend exactement.** Une seule chose, mais qui n'est pas triviale : **arbitrer**.
Faut-il finir cette allée, ou partir en commencer une autre ? Faut-il aller chercher ce carton
isolé à 12 m, ou ces trois-là qui sont à 4 m ? Faut-il monter d'un étage maintenant ou au
retour ? Compte tenu du temps restant et de ce que font les deux autres drones.

### B4 — Les jambes (pas apprises)

Reçoivent un point de vue cible. Deux phases :
1. **Transit** : générer une trajectoire lisse jusqu'au point, éviter les obstacles, aller vite.
2. **Lecture** : en approche finale, ralentir sous 0,45 m/s, se stabiliser à la bonne distance
   et au bon cap, tenir quelques images.

**C'est le morceau le plus risqué de toute l'architecture**, et je le dis tout de suite : ce
n'est pas « aller à un point de passage », c'est de l'asservissement. Il doit avoir des
semaines dédiées dans le planning.

### B6 — La coordination

**Carte partagée + réservation.** Chaque drone diffuse ses détections (la carte est commune) et
annonce le point de vue qu'il vise. Les autres le voient dans leur observation et apprennent à
ne pas y aller.

**La panne d'un drone est gérée gratuitement** : les réservations expirent après quelques
secondes sans nouvelles. Les cibles du drone mort redeviennent disponibles, et les deux autres
les reprennent au tour de décision suivant. **Aucun apprentissage spécial n'est nécessaire pour
la résilience** — elle vient de la structure. C'est un point fort qu'il faut assumer comme
tel, pas déguiser en contribution scientifique.

### La boucle complète

```
30 Hz  : les jambes suivent la trajectoire en cours
         le lecteur tourne sur les images, marque les décodages
 5 Hz  : l'œil détecte, estime les positions 3D, met à jour la carte partagée
 1 Hz  : la tête regarde la carte, génère ~50 points de vue candidats,
         en choisit un, l'annonce aux coéquipiers, le passe aux jambes
```

### Ce qui doit être vrai pour que ça marche

**Condition 1 — le détecteur doit repérer un motif de QR plus loin qu'il ne le décode.**
Toute la valeur de la sortie « repéré, non décodé » vient de là. **Ce n'est pas mesuré**, et
la source qu'on citait pour l'affirmer ne dit pas ce qu'on lui faisait dire. À mesurer, c'est
une demi-journée de travail.

**Condition 2 — les jambes doivent tenir la pose de lecture.** Si le contrôleur arrive à
0,9 m/s devant le rack, aucune tête intelligente ne sauvera le résultat.

**Condition 3 — le modèle de capteur simulé ne doit pas être plus généreux que le vrai.**
C'est le piège classique : la politique apprend à exploiter un capteur simulé trop gentil, et
s'effondre sur le vrai. À vérifier en rejouant des trajectoires enregistrées à travers le vrai
détecteur.

**Condition 4 — il doit y avoir quelque chose à arbitrer.** Si la stratégie optimale est
« balayer chaque face de rack dans l'ordre », la tête n'a rien à apprendre. **C'est la
condition la plus importante et c'est celle dont je suis le moins sûr.** Voir la section 3.

### Ce que j'en pense vraiment

**Ce qui me fait choisir cette direction n'est pas dans les papiers.** C'est ceci :

**1. Chaque brique se teste seule.** Je peux mesurer le détecteur sans voler. Je peux mesurer
le contrôleur sans apprendre. Je peux mesurer la carte sans récompense. Quand un chiffre est
mauvais, je sais **quelle brique** est en cause. Sur un projet qui a déjà perdu des semaines
sur un bug non localisé, cette propriété vaut plus que n'importe quel gain de performance.

**2. La partie apprise est minuscule.** 1 à 2 millions de paramètres, une décision par seconde,
pas de rendu d'image dans la boucle. Un run se compte en heures, pas en jours. Ça change tout :
on peut se payer **30 expériences au lieu de 5**. Et sur un problème mal connu, le nombre
d'essais compte plus que la sophistication de la méthode.

**3. L'action discrète supprime un piège réel.** Quand l'action est « choisir le candidat n°17 »
au lieu de « commander 0,63 m/s », le bruit d'exploration ne peut plus fabriquer accidentellement
des lectures. C'est un vrai gain structurel.

**Maintenant ma critique, et elle est sérieuse.**

**La tête pourrait ne rien valoir.** Une fois qu'on a la carte partagée, la génération de
candidats et le contrôleur, la règle « aller au point de vue le plus proche qui n'est pas
réservé, pondéré par le nombre de cellules non lues qu'il découvre » tient en vingt lignes.
Cette règle capture peut-être 90 % de la valeur. Le RL n'apporterait que l'arbitrage fin sous
contrainte de temps.

**Conséquence à assumer maintenant :** il est très possible que la contribution du mémoire soit
**le système**, pas l'apprentissage. Ce serait un résultat honnête et défendable — mais il vaut
mieux le décider maintenant que le découvrir en semaine 10.

Et j'ajoute un doute qui vient de notre situation, pas de la littérature : **le facteur limitant
pourrait être le temps de lecture, pas la stratégie.** Si un cycle complet (approche +
stabilisation + décodage) prend 6 secondes et qu'on a 7,5 secondes par QR de budget, alors la
qualité de la décision ne change presque rien : c'est la vitesse du cycle qu'il faut attaquer.

---

## D4 — Active Inference + module visuel

### L'idée en une phrase

Au lieu d'apprendre par essais et erreurs, le drone **raisonne** : il maintient une croyance
sur ce qu'il ignore, et il choisit l'action qui réduira le plus son ignorance tout en le
rapprochant de son but. L'exploration n'est pas ajoutée — elle sort du calcul.

### La structure, sans les mathématiques

L'inférence active demande de définir quatre objets. Les voici pour notre problème.

**L'état caché (ce qu'on ignore et qu'on veut deviner).**
- Où est le drone : un nœud dans un graphe de positions, construit au fur et à mesure.
- Pour chaque cellule de face de rack : est-ce qu'elle porte un carton non encore lu ? C'est
  une variable binaire, avec une probabilité.

Au départ, **toutes ces probabilités valent 0,5** : c'est la façon de dire « je ne sais rien ».
La position des QR n'est pas dans le modèle — **elle est précisément ce que le modèle cherche
à deviner.** C'est le point élégant de la direction.

**La matrice A — le modèle du capteur.** Elle dit : « si le drone est dans cet état, quelle est
la probabilité de chaque observation ? ». Chez nous : `P(décodage réussi | distance, angle,
vitesse)`. C'est cette matrice qui encode la physique de la lecture.

**La matrice B — les transitions.** « Si je suis ici et que je vais là, où j'arrive ? »

**Le vecteur C — les préférences.** Une seule ligne : « je préfère observer un décodage ». Pas
de bonus de couverture, pas de pénalité de temps bricolée. C'est la discipline de la méthode.

### Comment il décide

À chaque décision (toutes les 2 à 4 secondes), il évalue des séquences d'actions possibles et
calcule pour chacune l'**énergie libre attendue**. Cette quantité se décompose en trois
morceaux, et chacun a un sens concret :

| Terme | Ce qu'il pousse à faire |
|---|---|
| **Pragmatique** | aller là où on a des chances de décoder — l'exploitation |
| **Épistémique** | aller là où la carte est incertaine — l'exploration, **gratuitement** |
| **Ambiguïté** | se placer là où le capteur est **net** : proche, de face, lent |

Le troisième est le plus intéressant pour nous. **Le drone préfère les poses où son capteur
est fiable.** Or notre capteur est fiable exactement quand on est à 1 m, en face, à 0,45 m/s.
Donc la manœuvre de lecture sortirait du calcul, sans qu'on l'ait programmée.

Le calcul se fait par une recherche arborescente (on déroule des séquences de 5 à 10 actions et
on garde la meilleure).

### La boucle complète

```
30 Hz  : un contrôleur classique vole vers le nœud choisi
 5 Hz  : le détecteur met à jour la croyance (bayésien : chaque observation
         modifie la probabilité « il y a un carton non lu ici »)
0,3 Hz : le planificateur évalue les politiques candidates, choisit celle
         qui minimise l'énergie libre attendue
```

### Qui fait quoi

| Brique | Qui |
|---|---|
| B1 Capteur | classique (le même que D3) |
| B2 Mémoire | la croyance probabiliste — **c'est la carte de D3, avec des probabilités au lieu de valeurs** |
| B3 Décision | le planificateur EFE — **écrit à la main, rien n'est appris** |
| B4 Contrôle | classique (le même que D3) |
| B6 Coordination | croyance partagée + allocation séquentielle |

### Ce qui doit être vrai

**Condition 1 — la matrice A doit être calibrée.** Et c'est là que le bel argument s'effondre
partiellement. Pour que le terme d'ambiguïté pousse le drone à se placer correctement, il faut
d'abord lui **donner** `P(décodage | distance, angle, vitesse)`. Il faut donc faire exactement
la même campagne de mesure, avec les mêmes paramètres, que pour écrire une récompense.

**L'inférence active ne supprime pas le réglage à la main. Elle le déplace du fichier
« récompense » vers le fichier « modèle du capteur ».** Et il y a une asymétrie qui joue contre
elle : une récompense imparfaite se rattrape en partie par l'apprentissage, tandis qu'un modèle
faux fait planifier le drone **avec assurance** contre une réalité mal décrite.

**Condition 2 — le calcul doit tenir dans le temps imparti.** C'est le point dur. Le coût de
l'évaluation grandit très vite avec l'horizon de planification. Le meilleur système publié
décide en **4 secondes** sur un espace **trois fois plus petit que le nôtre**, sur un petit
calculateur embarqué. Extrapolé chez nous : de l'ordre de **plusieurs secondes à quelques
dizaines de secondes par décision**. Même la couche lente à 1 Hz devient tendue.

Il existe des versions accélérées (programmation dynamique au lieu de l'arbre complet) qui
ramènent le coût à quelque chose de linéaire. C'est la voie à prendre si on choisit cette
direction — mais c'est du développement, pas une bibliothèque à installer.

**Condition 3 — les outils.** La bibliothèque mûre (`pymdp`) est **discrète et mono-agent**.
Il faudrait donc discrétiser notre problème continu, et écrire soi-même la coordination.

### Ce que j'en pense vraiment

**Le constat qui m'a le plus surpris en construisant ces quatre architectures :**

**D4 et D3 sont presque le même système.** Regardez les deux tableaux « qui fait quoi » : même
œil, même contrôleur, même type de mémoire, même schéma de coordination. **La seule différence
est la brique B3.** D3 apprend la décision ; D4 la calcule à partir d'un modèle.

Ce n'est donc pas « RL contre inférence active ». C'est :

> **la couche de décision doit-elle être apprise, ou écrite à la main comme un planificateur de
> gain d'information ?**

Et posée comme ça, la question a une réponse pratique évidente : **on ne choisit pas, on
construit les deux.** Le planificateur à énergie libre est ce qu'il faut de toute façon comme
point de comparaison honnête. S'il gagne, on l'utilise et on le dit. Si la politique apprise
le bat, c'est la contribution du mémoire. Dans les deux cas, il fallait les deux.

C'est, à mon avis, la conclusion la plus utile de tout cet exercice — et elle n'apparaît que
parce qu'on a construit les architectures avant de les critiquer.

**Ce que je ne recommande pas :** prendre l'inférence active comme cadre **complet**, avec
apprentissage bayésien du modèle en ligne et planification arborescente à chaque pas. Le coût
de calcul et le coût de développement ne rentrent pas dans 2-3 mois, et les outils ne sont pas
prêts pour le continu multi-agent.

> **Ce que je retiens de D4 :** trois choses concrètes, à mettre dans D3.
> 1. **Le terme d'ambiguïté**, transformé en récompense dense calibrée sur (distance, angle,
>    vitesse) — il attaque directement la compétence difficile.
> 2. **Le terme épistémique** = payer la baisse d'incertitude de la carte, au lieu d'inventer
>    un bonus de couverture arbitraire.
> 3. **La discipline d'une seule préférence** : un seul but (« un QR décodé »), tout le reste
>    déduit. C'est la meilleure protection contre le bricolage de récompense.

---

## Récapitulatif — les quatre architectures côte à côte

| | **D1 Dreamer** | **D2 VLA** | **D3 RL + vision** | **D4 Active Inference** |
|---|---|---|---|---|
| **Entrée principale** | images 64×64 + proprio | image + phrase | carte + candidats | carte + croyance |
| **Ce qui est appris** | tout | rien (pré-entraîné) ou imitation | la décision seule | rien |
| **Taille du réseau** | ~12 M | 0,45 à 7 milliards | **~1-2 M** | pas de réseau |
| **Sortie du modèle** | vitesse (30 Hz) | vitesse ou zone | **point de vue (1 Hz)** | nœud (0,3 Hz) |
| **Qui vole** | le réseau | un contrôleur | un contrôleur | un contrôleur |
| **Mémoire des QR lus** | dans le réseau (échoue) | dans le contexte | carte externe | croyance |
| **Détection des QR** | impossible en 64×64 | possible mais lourd | détecteur classique | détecteur classique |
| **Exploration** | bruit + désaccord | a priori du modèle | apprise via récompense | terme épistémique |
| **Coordination** | implicite | par le langage | carte + réservation | croyance partagée |
| **Durée d'un run** | 1 à 2 semaines | jours (si ça tient) | **heures** | pas d'entraînement |
| **Nb d'essais possibles** | 3 à 6 | 2 à 4 | **~30** | illimité (mais dev long) |
| **Débogage** | 7 suspects, 1 mesure | boîte noire | **chaque brique isolable** | modèle inspectable |

---

## Ce que je pense, tout bien pesé

Trois conclusions, dans l'ordre d'importance.

**1. D1 et D2 ne sont pas des directions, ce sont des composants.**
De Dreamer je garde une idée (le modèle informé). De D2 je garde le patron à deux étages et un
détecteur de vision. Aucune des deux ne peut porter le projet, et pour des raisons qui tiennent
à **notre** situation : la mémoire graphique, le disque, le nombre d'essais possibles, et le
fait qu'on génère nous-mêmes les entrepôts.

**2. D3 et D4 sont le même système avec deux têtes différentes.**
Ce n'est pas un choix entre deux directions, c'est un choix sur une seule brique. Et la bonne
réponse est de **construire le système commun d'abord** — l'œil, la carte, le contrôleur, la
coordination — puis d'y brancher successivement les deux têtes : le planificateur de gain
d'information, puis la politique apprise. Le premier est la référence, le second est la
contribution.

**3. Le vrai risque n'est dans aucune des quatre directions.**
Il est dans la brique **B4**, le contrôleur qui doit tenir la pose de lecture, et dans la
**question de savoir s'il y a quelque chose à arbitrer**. Ces deux points ne dépendent d'aucun
choix de méthode. Ils se mesurent en quelques jours, sans GPU, et ils devraient être mesurés
avant d'écrire une ligne de code d'apprentissage.

---

## Où se trouve le reste

- La revue de littérature qui appuie ou contredit ces architectures :
  [`etude_4_directions.md`](etude_4_directions.md).
- Les chances de réussite chiffrées y sont révisées à la lumière de ce document : elles portent
  désormais sur **les architectures ci-dessus**, pas sur « Dreamer en général ».
