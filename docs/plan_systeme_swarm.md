# Plan de construction — Essaim de drones pour l'inventaire par QR codes

**Comment lire ce document.** La partie 1 explique la solution : ce qu'on construit et comment
ça marche. La partie 2 explique la construction : onze étapes dans l'ordre, chacune avec ce
qu'il faut chercher, mesurer, comparer, implémenter, tester et valider. La partie 3 rassemble
les règles qui valent partout.

**Où suis-je à tout moment ?** Chaque étape commence par « Pourquoi maintenant » (ce qui existe
déjà et pourquoi cette étape arrive à ce moment) et finit par « Porte de validation » (les
chiffres à atteindre pour avoir le droit de continuer). Tant qu'une porte n'est pas franchie,
on ne passe pas à la suite.

---

## La règle de départ : on repart de zéro

Tout se construit dans un **nouveau dossier** : `swarm_qr/`.

Il faut séparer deux choses très différentes.

**La plomberie du simulateur : on regarde directement l'ancien projet, et on réutilise.**
Lancer Isaac Sim, charger une scène d'entrepôt, attacher une caméra, régler un paramètre de la
machine — tout cela n'a rien à voir avec notre solution. C'est du branchement d'outil. Refaire
ces recherches serait du temps perdu pour rien. **Quand un problème de ce genre se pose, on va
voir comment c'était déjà fait et on s'en sert.**

**La solution : on repart de zéro, sans exception.** L'architecture, les composants, les
algorithmes, la façon de décider, la façon de coordonner les drones — rien n'est repris de
l'ancien projet.

**Les chiffres : aucun n'est hérité.** Les seuils de lecture, les temps de cycle, les portées,
les taux de réussite : **aucune ancienne mesure n'est une vérité**. Chaque constante dont le
système a besoin sera mesurée ici, par nous, et enregistrée avec le script qui l'a produite.

---

# PARTIE 1 — La solution

## Le problème

Trois drones sont lâchés dans un entrepôt qu'ils ne connaissent pas. Des cartons sont posés sur
des racks à plusieurs hauteurs, et chaque carton porte un QR code sur sa face. Personne ne sait
à l'avance où sont les racks, les cartons, ni les codes.

**La mission : lire le plus de QR possible dans un temps limité.** Le système doit fonctionner
sur des entrepôts de formes différentes, y compris des formes jamais vues, et continuer à
marcher quand quelque chose se passe mal : un drone tombe en panne, un obstacle apparaît,
l'inventaire change en cours de route.

Une contrainte physique fait toute la difficulté : pour décoder un QR, la caméra doit être
**proche, à peu près en face, et lente**. Les seuils exacts ne sont pas connus — **nous les
mesurerons à l'étape 2**, et ils deviendront les constantes du système.

## Le vocabulaire du projet

| Mot | Ce qu'il veut dire ici |
|---|---|
| **Carte** | Une grille 3D qui découpe l'espace en petits cubes de 25 cm. Un cube n'est pas un objet : c'est un morceau d'espace qui dit « ici il y a quelque chose » ou « ici c'est vide » ou « je ne sais pas ». La taille de la grille est fixée à l'avance, plus grande que le plus grand entrepôt testé. |
| **Frontière** | Un endroit de la carte où le connu touche l'inconnu. C'est là qu'il faut aller pour découvrir du nouveau. |
| **Tag vu-non-lu** | Un QR dont on a repéré le motif carré sans réussir à le décoder (trop loin, trop de biais). On connaît sa position approximative : c'est une lecture presque garantie si on s'approche. |
| **Point de vue** | Une position **et** une direction de regard. Par exemple : « à 1 m de cette face de rack, perpendiculairement, à 2 m de haut ». |
| **Réservation** | Quand un drone choisit une cible, il l'annonce aux autres. Elle expire toute seule si le drone ne donne plus de nouvelles. |
| **L'œil** | Le détecteur appris (type YOLO) qui repère les QR et les cartons de loin, et étiquette les zones. |
| **Le guide** | Le modèle vision-langage qui regarde la caméra frontale et la carte, et conseille une zone à explorer. |
| **λ (lambda)** | Le réglage qui dit combien on écoute le guide. λ = 0 : on l'ignore, le système reste purement géométrique. |
| **Balayage fixe** | Politique bête de référence : parcourir toutes les faces de racks dans un ordre fixe. |
| **Oracle** | Politique qui triche pour servir de plafond : elle connaît toutes les positions et va au plus proche non lu. |
| **Hors distribution** | Des entrepôts d'un type que notre générateur habituel ne produit pas. Ils servent à prouver la généralisation. |

## Les sept composants

**Deux modèles de vision, à deux endroits différents.** L'œil regarde **devant le drone** et
remplit la carte. Le guide regarde **la carte entière** et dit où aller. L'un alimente l'autre.

### C1 — Les capteurs
Deux caméras latérales haute résolution (lire les QR de près), une caméra frontale basse
résolution (voir loin, alimenter le guide), un capteur de distance (obstacles).
**Sortie :** images brutes et distances, 30 fois par seconde.

### C2 — L'œil : perception à deux niveaux
**Niveau 1, le décodeur classique.** Un algorithme (pas un réseau) qui lit les QR sur les
images haute résolution. Il ne réussit que si le drone est proche, en face et lent.
**Niveau 2, le détecteur appris.** Un petit réseau de type YOLO, environ 200 Mo. Il ne lit
pas : il **repère**. Repérer un carré est plus facile que le déchiffrer, donc il y arrive de
beaucoup plus loin. Il repère aussi **les cartons eux-mêmes**, et **étiquette les zones**
(allée, rack, mur, espace ouvert).
**Sortie :** quatre états, chacun avec sa position 3D — `décodé`, `vu-non-lu`,
`carton repéré`, `rien` — plus les étiquettes de zone.
**Reçu par :** la carte.

### C3 — La carte partagée
Une grille 3D unique pour les trois drones. Par cube : libre / occupé / inconnu ; déjà observé
ou non ; tag vu-non-lu ; tag lu ; présence des coéquipiers ; **étiquette sémantique**.
Elle tient aussi les réservations avec leur expiration, et la liste noire des cibles
abandonnées.
**Sortie :** l'état des connaissances de l'équipe, et une **vue de dessus** en couleurs (gris =
inconnu, blanc = exploré, orange = vu-non-lu, vert = lu).
**Reçue par :** le générateur de cibles, le guide, le contrôleur.

### C4 — Le générateur de cibles
Il lit la carte et fabrique des propositions de trois sortes : les **frontières** (découvrir),
les **tags vus-non-lus** (convertir en lecture), les **cartons repérés** non encore inspectés.
Pour chacune il calcule le point de vue exact.
**Sortie :** environ 50 points de vue candidats, avec ce que chacun rapporterait.
**Reçu par :** la décision.

### C5 — Le guide vision-langage (optionnel)
Toutes les 3 à 5 secondes. **Entrée : trois choses** — l'image de la caméra frontale, la vue de
dessus de la carte, et 4 à 8 zones numérotées dessinées dessus. **Question posée :** « Quelle
zone contient probablement des étagères non encore inspectées ? Réponds par le numéro et une
phrase. »
**À quoi il sert :** la géométrie sait dire « cette zone est grise », mais pas **ce qu'il y a
probablement dedans**. Le guide a vu des milliers d'entrepôts pendant son pré-entraînement : il
peut reconnaître qu'un couloir continue avec des racks, ou qu'une grande zone vide est un
espace de manœuvre sans rien à scanner.
**Sortie :** un numéro de zone et une phrase d'explication.
**Reçu par :** la décision.

### C6 — La décision
Une fois par seconde, elle note chaque proposition :

> **note = gain d'information + bonus si tag vu-non-lu + λ × avis du guide
> − coût du trajet − pénalité si un coéquipier a réservé**

Elle prend la meilleure note et **réserve** la cible.
**Sortie :** un point de vue cible.
**Reçu par :** le contrôleur.

### C7 — Le contrôleur
**Transit :** aller vite en contournant les obstacles connus. **Approche :** ralentir sous le
seuil de lecture, s'aligner en face, tenir la position stable le temps du décodage.
**Abandon propre** si la cible est inaccessible ou si la lecture échoue plusieurs fois.
**Sortie :** commandes de vitesse à 30 Hz, et le signal « atteint / abandonné » vers la
décision.

### Le schéma

```
  caméras + lidar ──► C2 L'ŒIL ──► détections + étiquettes ──► C3 CARTE PARTAGÉE
                          ▲                                        │        │
                          │                    vue de dessus       │        │ frontières,
                          │                    + zones numérotées  │        │ tags, cartons
  image frontale ─────────┼────────────────────────┐               │        ▼
                          │                        ▼               │   C4 GÉNÉRATEUR
                          │                  C5 LE GUIDE           │        │
                          │                        │ avis          │        │ ~50 candidats
                          │                        └──────► C6 DÉCISION ◄───┘
                          │                                        │
                          │                                        │ cible + réservation
                          │                                        ▼
                          └──────── le drone bouge ◄──────── C7 CONTRÔLEUR
```

Trois rythmes : C7 à 30 Hz, C6 et C4 à 1 Hz, C5 toutes les 3 à 5 secondes. C2 et C3 en continu.

## La vie d'un QR code

C'est la séquence centrale du système. Chaque QR suit ce chemin :

1. **Repéré de loin.** Le détecteur appris voit un carton à 4 m. La carte note « carton repéré ».
2. **Il devient une proposition.** Le générateur calcule le point de vue pour l'inspecter.
3. **Vu.** Le drone approche à 2 m ; le détecteur reconnaît le motif du QR, mais le décodeur
   n'y arrive pas encore. La carte note « vu-non-lu » avec sa position et son orientation.
4. **Il devient prioritaire.** Le bonus le fait passer devant les frontières : une lecture
   presque certaine vaut mieux qu'une découverte incertaine.
5. **Lu.** Le drone approche à 1 m, ralentit, se met en face, se stabilise. Le décodeur
   classique lit l'identifiant. La case passe au vert.

**Les deux modèles se partagent ce travail :** le détecteur appris fait les étapes 1 et 3
(repérer), le décodeur classique fait l'étape 5 (lire).

## Comment les trois drones se répartissent le travail

Il n'y a **pas de chef**, seulement la règle de réservation. Au décollage, les trois voient la
même carte ; le premier qui choisit réserve, les autres voient la pénalité et choisissent
ailleurs. **Ils partent naturellement dans des directions différentes** — c'est une conséquence
de la règle, pas une programmation.

**Si un drone tombe en panne**, il cesse de donner des nouvelles, ses réservations expirent, ses
cibles redeviennent libres, et les deux autres les reprennent. Le même mécanisme disperse les
drones au début et absorbe la panne à la fin.

## Comment la mission se termine

Soit **il ne reste plus rien** : aucune frontière atteignable, aucun tag vu-non-lu, aucun carton
non inspecté. Soit **le temps est écoulé**. Dans les deux cas le système produit son rapport :
codes lus, carte finale, trajectoires, cibles abandonnées avec leur raison, journal des avis du
guide.

## Comment chaque situation difficile est gérée

| Situation | Mécanisme |
|---|---|
| Un drone tombe en panne | Ses réservations expirent, ses cibles redeviennent libres. Rien de spécial à coder. |
| Un obstacle apparaît | La carte se met à jour, le contrôleur contourne. Si la cible devient inatteignable, il abandonne et la décision choisit autre chose. |
| L'inventaire change | Un nouveau motif devient un tag vu-non-lu, donc une nouvelle cible. Un carton disparu redevient de l'espace libre. La carte n'est jamais considérée comme définitive. |
| Un QR est illisible | Plusieurs tentatives en variant le point de vue, puis abandon avec la raison enregistrée. Le drone ne reste jamais bloqué. |
| Une cible est inaccessible | Le contrôleur détecte qu'il ne progresse plus, abandonne, et la cible passe en liste noire temporaire. |
| Deux drones veulent la même cible | Impossible : le premier réserve, l'autre voit la pénalité. |
| Le guide se trompe | Son avis n'est qu'un terme dosé par λ ; la géométrie garde son mot. λ = 0 reste disponible. |
| Le guide est lent ou muet | Son avis est demandé sans jamais l'attendre. Pas d'avis = décision géométrique. |
| L'œil invente une détection | Le drone va vérifier, échoue, abandonne. Coût : un trajet perdu. Le taux de fausses détections est mesuré à l'étape 7. |

## Pourquoi ce système devrait marcher

**La découverte est garantie par construction.** Tant qu'il existe une zone inconnue
atteignable, il existe une frontière, donc une cible, donc un drone qui y ira. Le système ne
peut pas oublier une partie de l'entrepôt.

**La lecture est garantie par la chaîne repéré → vu → lu.** Un tag aperçu reste une cible
prioritaire jusqu'à être lu ou abandonné avec une raison écrite.

**Coordination et résilience viennent d'un seul mécanisme**, la réservation qui expire. Un
mécanisme à tester, trois propriétés obtenues.

**Chaque composant se teste seul.** Quand un chiffre est mauvais, on sait quel composant est en
cause. On n'a jamais deux inconnues superposées.

---

# PARTIE 2 — La construction

## L'ordre d'exécution

Les étapes gardent leur numéro, mais elles se font dans cet ordre-là. Le but est d'obtenir un
**système complet qui vole le plus tôt possible**, puis de le mesurer.

| Rang | Étape | Pourquoi ici |
|---|---|---|
| 1 | **3 — le contrôleur** | Rien ne vole sans lui, et il porte le risque non testé : trois drones en même temps |
| 2 | **4 — la carte partagée** | Tout s'y branche, et elle produit la vue de dessus dont le guide a besoin |
| 3 | **7 — l'œil appris** | Les données d'entraînement existent déjà, et aucun vol nouveau n'est nécessaire |
| 4 | **5 + 8 ensemble** | La décision et le guide sont construits d'un seul tenant |
| | | → **première mission complète à trois drones** |
| 5 | 6 — les deux références | Sert à comparer, pas à faire fonctionner |
| 6 | 9 — l'évaluation complète | Les chiffres du mémoire |
| 7 | 10 — la décision apprise | Optionnelle |
| 8 | 11 — la consolidation | À la fin |

**Les étapes 5 et 8 se construisent ensemble**, et non l'une après l'autre. Le paramètre λ reste
un interrupteur pour isoler une panne pendant la mise au point, pas un protocole d'expérience.

**L'ordre est choisi pour deux raisons.** D'abord le risque : le contrôleur, qui est la partie
physiquement difficile, arrive en premier. Ensuite l'utilité : les quatre premières étapes
suffisent à faire tourner une mission complète ; tout ce qui suit sert à la mesurer ou à
l'améliorer.

---

## Étape 0 — Le cadrage

**Pourquoi maintenant.** Rien n'existe. Il faut fixer par écrit ce qu'on résout et comment on
jugera, pour que plus rien ne bouge pendant la construction.

**Ce qu'on cherche.** Rien à l'extérieur. C'est une décision.

**Ce qu'on décide**, dans `swarm_qr/docs/cadre.md` :
1. **L'instance du problème** : dimensions de l'entrepôt, nombre de racks, de cartons, de QR ;
   durée d'une mission ; nombre de drones. Ces valeurs sont les nôtres, choisies maintenant.
2. **La taille de la carte**, fixée à l'avance et plus grande que le plus grand entrepôt testé.
   Ce n'est pas de la triche : on donne une feuille assez grande pour dessiner, pas le contenu
   de l'entrepôt.
3. **Ce qui varie d'un entrepôt à l'autre** : disposition des cartons, taux de remplissage,
   **et disposition des racks** — c'est elle qui donne son sens au mot « découverte ».
4. **Les deux métriques.** La principale : **le pourcentage de QR décodés** en fin de mission,
   sur des entrepôts jamais utilisés pendant la construction. La seconde : **le temps mis pour
   y arriver**, qui départage deux versions atteignant le même pourcentage.
   Le budget de mission est fixé large, assez pour que le système puisse tout lire : le temps
   sert à mesurer l'efficacité, pas à limiter la mission.
5. **Tous les scénarios d'évaluation, dès maintenant** : mission normale ; panne d'un drone à
   25, 50 et 75 % ; obstacle qui apparaît ; inventaire modifié ; QR dégradés ; cibles
   inaccessibles ; entrepôts hors distribution. Chacun aura son script à l'étape 9.
6. **Les hypothèses assumées**, à écrire comme limites du travail : la position du drone est
   fournie par le simulateur ; la carte est partagée sans simuler les pertes réseau ; la
   physique du drone est simplifiée.

**Porte de validation.** Vous lisez la page et vous la validez.

**Si ça bloque.** Une valeur difficile à fixer (la durée de mission, par exemple) est posée
provisoirement, avec une note disant qu'elle sera re-réglée après l'étape 6, quand l'oracle
aura montré ce qui est atteignable.

---

## Étape 1 — L'environnement de simulation

**Pourquoi maintenant.** Sans monde simulé, aucun composant ne peut être construit ni testé.
Tout ce qui suit s'exécutera dedans.

### L'architecture de l'environnement

| Brique | Ce qu'elle contient |
|---|---|
| **Le simulateur** | Le moteur physique et le rendu des images. |
| **La scène** | L'entrepôt : sol, murs, racks, cartons posés sur les étagères, et une image de QR appliquée sur la face de chaque carton. |
| **Les drones** | Trois appareils identiques, commandés en vitesse (trois vitesses de déplacement et une vitesse de rotation). |
| **Les capteurs de chaque drone** | Deux caméras latérales en haute résolution, une caméra frontale en basse résolution, un capteur de distance. |
| **Le générateur de variantes** | À partir d'un nombre appelé graine, il produit un entrepôt : la disposition des racks, le placement des cartons, le taux de remplissage, et les positions de départ des drones. |
| **Le rapporteur** | Il connaît la vérité (où est chaque QR) et sert uniquement à noter le résultat à la fin. Le système n'y a jamais accès. |

### Les outils

**Isaac Sim** est le simulateur : il calcule la physique et produit les images.
**Isaac Lab** est une couche posée au-dessus, qui fournit des raccourcis pour construire une
scène et brancher plusieurs caméras. Elle évite d'écrire beaucoup de code de branchement.

### Ce qu'on implémente

`swarm_qr/env/` contient quatre choses :
1. la construction de la scène à partir d'une graine ;
2. les drones et leurs capteurs ;
3. le générateur de variantes, qui doit faire changer **la disposition des racks** d'une graine
   à l'autre, et pas seulement le contenu des étagères ;
4. la séparation des graines en deux lots : un lot de travail, et un **lot scellé réservé à
   l'évaluation finale**, qui n'est pas ouvert avant l'étape 9.

### Ce qu'on teste

**Test 1 — Reproductibilité.** On construit deux fois l'entrepôt avec la même graine et on
compare. Ils doivent être identiques. *Ce que ça prouve : deux versions du système pourront être
comparées sur exactement le même terrain.*

**Test 2 — Variation.** On construit dix entrepôts avec dix graines différentes et on compare
leurs dispositions de racks. Elles doivent être nettement différentes. *Ce que ça prouve : les
drones exploreront de vraies formes nouvelles, et l'évaluation finale pourra dire quelque chose
sur la généralisation.*

**Test 3 — Qualité des images.** On capture une planche d'images à plusieurs distances et sous
plusieurs angles, et on la regarde. Les QR doivent être nets de près. *Ce que ça prouve : la
perception aura de quoi travailler.*

**Test 4 — Débit.** On chronomètre combien de pas de simulation par seconde le système tient
avec trois drones et toutes les caméras allumées. *Ce que ça donne : la durée d'une mission,
donc le temps à prévoir pour les campagnes d'évaluation.*

### Ce qu'on confirme pour passer à l'étape 2

Les entrepôts sont reproductibles à graine égale et vraiment différents à graines différentes ;
les QR sont nets de près ; le débit est mesuré et noté.

### Si un test échoue

**Images sombres ou floues :** revoir l'éclairage et les matériaux avant tout le reste, car
toute la perception en dépend.
**Entrepôts trop semblables :** étendre le générateur pour qu'il fasse varier la position, le
nombre et l'orientation des racks.

---

## Étape 2 — Le décodeur classique et l'enveloppe de lecture

**Pourquoi maintenant.** C'est la mesure **fondatrice** du projet. Elle donne les trois seuils —
distance, angle, vitesse — sur lesquels tout le reste s'appuiera : le contrôleur devra les
respecter, le générateur de cibles calculera les points de vue à partir d'eux, l'oracle les
utilisera. Sans ces chiffres, on construirait à l'aveugle.

**Ce qu'on cherche.** Les bibliothèques de décodage de QR disponibles en Python, et la méthode
standard pour obtenir la position 3D d'un marqueur carré de taille connue à partir de ses
quatre coins dans l'image.

**Les options et ce qu'on compare.**

| Décodeur | Avantage | Inconvénient |
|---|---|---|
| OpenCV `QRCodeDetector` | installé partout, zéro dépendance | réputé moyen |
| OpenCV `QRCodeDetectorAruco` | même chose, souvent meilleur | — |
| `pyzbar` (ZBar) | rapide, très répandu | faible sur les images difficiles |
| `pyboof` (BoofCV) | réputé le meilleur | demande Java |

**Comment on choisit : on construit notre propre banc.** Le drone est placé à des poses
contrôlées devant un QR — on balaie la distance, l'angle d'incidence et la vitesse — et **tous
les décodeurs sont mesurés sur exactement les mêmes images**. On garde celui qui a le meilleur
taux de décodage dans la zone utile. On ne se fie à aucun classement extérieur : notre rendu,
notre éclairage, nos QR.

**Ce qu'on mesure — les quatre chiffres fondateurs.**
1. **L'enveloppe de lecture** : distance maximale, angle maximal et vitesse maximale où le
   décodage dépasse 90 %. Ces trois valeurs deviennent **les constantes du projet**.
2. **La portée de repérage du décodeur classique** : jusqu'où il « voit » un motif sans le lire.
   C'est la référence que le détecteur appris devra battre à l'étape 7.
3. **La précision de la position 3D** des tags, en centimètres, à plusieurs distances.
4. **Le taux de fausses détections** sur des images sans QR.

**Ce qu'on implémente.** `swarm_qr/perception.py` (capture, décodage, les trois états, position
3D par `solvePnP`) et le banc `swarm_qr/experiments/02_enveloppe/`.

**Porte de validation.** Les quatre chiffres sont écrits dans
`swarm_qr/docs/constantes_mesurees.md`, avec les images du banc. L'enveloppe est jugée
exploitable : on doit pouvoir lire un QR en volant normalement, pas seulement à l'arrêt collé
au carton.

**Si ça échoue.** Décodage faible même proche et immobile : d'abord la lumière et l'exposition,
puis la résolution de la caméra, puis changer de décodeur. Position 3D imprécise : vérifier la
taille physique du QR utilisée dans le calcul.

**Cohérence.** Ces trois seuils vont maintenant contraindre le contrôleur (étape 3) et le
générateur de cibles (étape 5). Si l'enveloppe est très étroite, il faudra peut-être revoir la
durée de mission fixée à l'étape 0 — c'est une décision à prendre ici, pas plus tard.

---

## Étape 3 — Le contrôleur

**Pourquoi maintenant.** Sans contrôleur, rien ne vole. Il a besoin de l'enveloppe mesurée à
l'étape 2 pour savoir à quelle distance et à quelle vitesse s'arrêter : il vient donc juste
après.

**Test préalable — est-ce qu'il y a seulement un problème ?**

Avant d'écrire quoi que ce soit, on vérifie si le drone a vraiment du mal à s'arrêter. On lui
donne la commande la plus bête possible : « avance vers ce point à vitesse constante, et coupe
tout en arrivant ». Puis on regarde ce qui se passe :

- de combien il **dépasse** le point avant de s'immobiliser ;
- à quelle **vitesse** il passe au point visé ;
- s'il **oscille** autour du point ou s'il se pose franchement ;
- combien de temps il met à se stabiliser.

**Ce que ce test décide.** Si le drone s'arrête proprement du premier coup, il n'y a pas de
problème de freinage et le contrôleur se réduit à presque rien : viser le point, avancer,
couper. Si au contraire il dépasse ou oscille, on sait exactement de combien, et on passe à la
méthode ci-dessous.

**Ce qu'on écrit si le test montre un problème.** Un asservissement classique : le drone
commande une vitesse proportionnelle à l'écart qui le sépare de sa cible, avec un plafond de
vitesse. Loin, il va vite ; près, il ralentit tout seul. Deux modes : transit rapide, puis
approche lente à partir de quelques mètres.

**Ce qu'on mesure — le test des 100 poses.** On tire 100 poses cibles au hasard devant des
racks, à des hauteurs et des orientations variées, et on mesure :
1. **Le taux d'arrivée correcte** : la position et le cap sont atteints avec une petite erreur.
2. **La stabilité** : la pose est tenue sous le seuil de vitesse assez longtemps pour que le
   décodeur ait sa chance.
3. **Le temps de cycle complet** : transit + approche + tenue, chronométré. C'est la mesure
   d'efficacité du contrôleur, et elle servira à comparer les versions du système entre elles.
4. **Le taux d'abandon** : combien de cibles se révèlent inatteignables et en combien de temps
   le drone le comprend.

**Ce qu'on implémente.** `swarm_qr/control.py` : les deux modes, l'abandon propre, le signal
« atteint / abandonné ».

**L'intégration.** Contrôleur + perception ensemble : on donne au drone la position d'un tag,
il vole, se place, et **le décodeur lit pour de vrai**. C'est le premier vrai sous-système :
**un drone qui lit un QR de bout en bout.**

**Porte de validation.** Taux d'arrivée élevé sur les 100 poses ; taux de décodage élevé sur le
test intégré ; temps de cycle chronométré et noté.

**Si ça échoue.** D'abord les réglages : si le drone oscille autour du point, on réduit les
gains ; s'il arrive trop vite, on commence l'approche plus tôt. Cela suffit dans la plupart des
cas.

Si le problème persiste après réglage, deux méthodes plus élaborées existent, à essayer dans
cet ordre :
- **Les trajectoires lissées** : au lieu de réagir instant par instant, on calcule tout le
  chemin à l'avance sous forme d'une courbe douce, en décidant dès le départ à quelle vitesse
  on arrivera. Le drone n'a plus qu'à suivre la courbe.
- **La commande prédictive** : à chaque instant, le drone simule plusieurs futurs possibles sur
  une ou deux secondes, garde la meilleure suite de commandes, en exécute la première, puis
  recommence. C'est la seule méthode qui gère parfaitement « arriver exactement ici » et « ne
  jamais dépasser cette vitesse » en même temps, mais elle demande beaucoup plus de calcul et
  de réglage.

---

## Étape 4 — La carte partagée

**Pourquoi maintenant.** Un drone sait lire. Il lui manque la mémoire, et à l'équipe le moyen
de partager ce qu'elle voit.

**Ce qu'on cherche.** Rien à l'extérieur : c'est une grille et de la comptabilité.

**Les options et ce qu'on compare.**

| Question | Options | Ce qui départage |
|---|---|---|
| Structure | grille pleine / structure creuse | On **calcule la mémoire avant de coder** : si la grille pleine tient confortablement, elle gagne par simplicité |
| Taille de cube | petit (précis, lourd) / grand (léger, grossier) | Le cube doit être nettement plus petit qu'un carton, pour distinguer deux cartons voisins |

**Deux choses que cette étape doit fournir, parce qu'aucune autre ne le fait.**

- **Le capteur de distance.** L'architecture le prévoit, mais il n'est pas encore sur le drone.
  C'est ici qu'on l'installe, parce que la carte est son premier client : sans lui, la case
  « occupé » reste vide et rien ne peut être évité. Comme les caméras à l'étape 2, il est
  **mesuré avant d'être cru** : chaque rayon est refait par le moteur physique, et le banc
  s'arrête si les deux distances ne coïncident pas. Une carte remplie de travers a l'air normale.
- **Le calcul d'itinéraire.** Le contrôleur contourne les obstacles connus, mais il ne les
  connaît que par la carte : c'est donc la carte qui calcule le chemin — des points de passage
  qui évitent les cases occupées, élargies du rayon du drone — et le contrôleur qui le suit.
  Jusqu'ici, le chemin venait du plan connu de l'entrepôt ; il n'existera pas en mission.

**Ce qu'on mesure.**
1. **La mémoire réellement occupée** par la carte à la taille choisie.
2. **Le temps de mise à jour** : combien de millisecondes pour intégrer une observation. Ce
   temps doit rester très petit devant le pas de décision (1 seconde).
3. **Le capteur contre le moteur physique**, rayon par rayon, à plusieurs caps : l'écart doit
   rester de l'ordre du centimètre, et le monde ne doit pas bouger quand le drone tourne.
4. **L'exactitude** : on rejoue une trajectoire connue avec des détections connues, et on
   compare la carte reconstruite à la vérité — bonnes cases occupées, aucun obstacle inventé
   dans les allées, tags au bon endroit.
5. **Les itinéraires** : calculés sur la carte découverte, ils ne traversent jamais une case
   connue comme occupée.

**Ce qu'on implémente.** `swarm_qr/mapping.py` : la grille et ses canaux (occupation,
couverture, coéquipiers, **et le canal sémantique — qui restera vide jusqu'à l'étape 7**),
les tags vus-non-lus et lus **avec leur position exacte** — un cube de 25 cm jetterait la
précision au centimètre mesurée à l'étape 2 —, les réservations avec expiration, la liste noire
des cibles abandonnées, le calcul d'itinéraire, et le rendu **vue de dessus** en couleurs.

Deux faits de l'environnement à respecter. **Chaque carton porte le même code sur ses deux
faces** : deux lectures du même code à deux endroits distants de l'épaisseur d'un carton sont
deux panneaux, pas un seul ; les fusionner placerait le code au milieu du carton. Et **les
étiquettes n'ont pas toutes la même taille**, parce qu'elles suivent la taille des cartons :
la distance déduite de la taille supposée d'un code est fausse du même rapport. La position
d'un code vient donc du capteur de distance, le long de la direction que l'image donne ; la
taille ne sert qu'en repli. C'est aussi ce qu'un vrai système ferait.

**Tests, et ce qu'ils prouvent.** Le test d'exactitude prouve que la mémoire du système est
juste. Un second test simule la mort d'un drone et vérifie que ses réservations expirent et que
ses cibles redeviennent libres : cela prouve que **le mécanisme central de coordination
fonctionne**, avant même qu'on s'en serve. Un troisième vérifie qu'un itinéraire contourne un
mur connu et qu'il traverse l'inconnu — sans quoi un drone ne sortirait jamais de sa zone
explorée ; c'est l'étape 5 qui, par ses frontières, l'y fera entrer par petits pas.

**L'intégration.** Perception + carte sur une mission à un seul drone : la vue de dessus se
remplit correctement pendant le vol, et le drone rejoint ses cibles par des itinéraires calculés
sur ce qu'il a découvert, sans le plan de l'entrepôt.

**Porte de validation.** Les cinq mesures passent, et **la vue de dessus est lisible par un
humain** : vous devez pouvoir suivre la mission à l'œil, sans explication.

**Si ça échoue.** Capteur en désaccord avec la physique : convention de repère ou d'angle du
capteur, à corriger avant tout. Tags mal placés : la précision 3D de l'étape 2, ou les deux
faces confondues. Trous dans la couverture : vérifier le calcul du champ de vue des caméras.

---

## Étape 5 — Le générateur de cibles et la décision

**Pourquoi maintenant.** Le drone sait lire et se souvenir. Il lui manque de savoir **où
aller**. C'est l'étape qui transforme des composants en système.

**Ce qu'on cherche.** Les recettes classiques de sélection de frontières et de « meilleur
prochain point de vue » en robotique d'exploration. Une lecture courte : on veut les idées
standard, pas une méthode exotique.

**Les options et ce qu'on compare.**

| Question | Options | Comment on tranche |
|---|---|---|
| Le score | peu de termes / beaucoup de termes | Le plus simple qui produit le bon comportement sur des cas jouets |
| La coordination | réservation simple / enchères entre drones | Réservation simple d'abord ; enchères seulement si on observe des choix collectifs visiblement mauvais |

**Ce qu'on mesure — les cas jouets d'abord.** Avant de mettre quoi que ce soit en vol, on
dessine à la main une dizaine de petites cartes et on vérifie que le score classe comme on
l'attend :
- un tag vu-non-lu proche doit battre une frontière lointaine ;
- une zone réservée par un coéquipier doit être évitée ;
- entre deux frontières équivalentes, la plus proche doit gagner ;
- quand tout est exploré, la liste de candidats doit être vide (condition de fin de mission).

Ce test prouve que **le cerveau raisonne correctement avant d'être mis dans la boucle**. C'est
beaucoup plus rapide que de déboguer en vol.

**Ce qu'on implémente.** `swarm_qr/planning.py` (candidats, score, réservation) et
`swarm_qr/mission.py`, le chef d'orchestre qui fait tourner les composants aux trois rythmes,
gère le début et la fin de mission, et écrit le rapport final.

**L'intégration — la première mission complète à trois drones.**

**Ce qu'on vérifie, et rien de plus.** Le but ici n'est pas de mesurer une performance : le
système est encore incomplet, ses deux modèles de vision ne sont pas branchés. Le but est de
vérifier que la mécanique tourne et de repérer les gros défauts. Deux ou trois missions
suffisent.

1. **Aucun blocage** : personne ne reste coincé, aucun aller-retour sans fin.
2. **Aucun doublon** : jamais deux drones sur la même cible.
3. **La dispersion initiale** : les trois drones partent-ils vraiment dans des directions
   différentes ?
4. **La panne** : on coupe un drone en cours de mission et on vérifie que les deux autres
   reprennent ses zones.
5. **La fin de mission** : elle se déclenche proprement, dans l'un des deux cas prévus.

**Porte de validation.** La mécanique tourne : pas de blocage, pas de doublon, la panne est
absorbée, la mission se termine proprement.

**Si ça échoue.** Deux drones qui se suivent : la pénalité de réservation est trop faible. Un
drone qui fait des allers-retours : le coût de trajet est trop faible. Des zones jamais
visitées en fin de mission : bug du générateur, une frontière n'est pas détectée.

---

## Étape 6 — Les deux références

**Pourquoi maintenant.** Les deux références s'écrivent une fois pour toutes, pendant qu'on a
les composants sous la main. Elles ne servent pas encore : **elles seront utilisées à
l'étape 9**, pour la comparaison finale avec le système complet. On ne les compare à rien
maintenant.

**Ce qu'on implémente.** `swarm_qr/baselines.py`, deux politiques qui utilisent **le même
contrôleur et la même perception** que le système, pour que la comparaison à venir soit
honnête :
1. **Le balayage fixe** : parcourir toutes les faces de racks dans un ordre fixe, à vitesse de
   lecture, sans jamais regarder la carte. C'est le plancher.
2. **L'oracle** : il reçoit les positions exactes de tous les QR (triche assumée) et va toujours
   au plus proche non lu. C'est le plafond.

**Ce qu'on mesure.** Chaque référence tourne sur quelques entrepôts, et on note son résultat.
Deux chiffres en sortent :

| Politique | % lu |
|---|---|
| Balayage fixe | **A**, le plancher |
| Oracle | **C**, le plafond |

**À quoi ils serviront.** **C** dit ce qu'il est possible d'atteindre au mieux. **C − A** dit
combien de marge existe entre une méthode bête et une méthode parfaite. Ces deux nombres
donneront leur sens aux résultats de l'étape 9, et **C − A** décidera à l'étape 10 si une
décision apprise a un intérêt.

**Porte de validation.** Les deux références tournent correctement et leurs chiffres sont
enregistrés dans `swarm_qr/docs/resultats.md`.

---

## Étape 7 — L'œil appris (le détecteur)

**Pourquoi maintenant.** La base tourne. On ajoute maintenant le premier des deux modèles de
vision, celui qui regarde devant le drone.

**Ce que le détecteur doit apporter, précisément.** Trois choses : repérer les motifs de QR
**plus loin** que le décodeur classique ; repérer **les cartons eux-mêmes**, même sans voir le
QR ; et **étiqueter les zones** (allée, rack, mur, espace ouvert) pour remplir le canal
sémantique de la carte.

**Ce qu'on cherche.** Les familles de détecteurs légers (YOLO et ses variantes récentes, et les
alternatives à taille comparable), leurs tailles, leurs vitesses, leurs licences. Et les outils
d'annotation automatique.

**Le point crucial : les données d'entraînement sont gratuites, et une partie existe déjà.** En
simulation, on connaît la position exacte de chaque carton et de chaque QR. On projette ces
positions dans l'image et **les annotations se génèrent toutes seules**, sans un seul clic.

Les images de l'étape 2 conviennent directement : chacune est enregistrée avec la pose exacte de
la caméra, et la fonction de projection utilisée pour l'analyse est vérifiée à quelques pixels
près. Cette étape démarre donc sur un jeu d'images déjà constitué, et n'a besoin de vols
supplémentaires que pour élargir la variété des points de vue.

**Les options et ce qu'on compare.**

| Question | Options | Ce qui départage |
|---|---|---|
| Les classes à détecter | QR seul / QR + carton / QR + carton + zones sémantiques | **La troisième**, parce que le canal sémantique doit venir de quelque part et que l'ajouter ici coûte presque rien |
| La taille du modèle | très petit / petit / moyen | Le plus petit qui atteint la précision voulue, mesuré |
| La résolution d'entrée | plusieurs valeurs | La plus basse qui garde la portée voulue |

**Le canal sémantique — la décision d'architecture.** Trois solutions étaient possibles : le
détecteur apprend aussi les classes de zones ; un modèle de segmentation séparé ; ou une
déduction purement géométrique. **On retient la première**, parce qu'elle réutilise un modèle
qu'on entraîne de toute façon, qu'elle ne coûte presque rien de plus, et qu'elle produit les
étiquettes à la même fréquence que les détections. Si elle échoue, on se rabat sur la déduction
géométrique (un couloir long et étroit entre deux obstacles est une allée), qui ne demande aucun
apprentissage.

**Ce qu'on mesure — d'abord seul, sur banc.**
1. **La portée de repérage**, comparée à celle du décodeur classique mesurée à l'étape 2.
   **C'est la mesure qui justifie ou non ce composant.**
2. **Le taux de détections manquées** (des QR ou cartons présents et non vus).
3. **Le taux de fausses détections** (des objets vus là où il n'y en a pas). Ce taux compte
   beaucoup : chaque fausse détection coûte un trajet perdu.
4. **La précision des étiquettes de zones.**
5. **La vitesse et la mémoire** : le modèle doit tourner à quelques images par seconde sans
   gêner le reste.

**Puis en intégration.** On rebranche tout et on lance quelques missions, pour vérifier que le
détecteur s'insère correctement : les cartons repérés deviennent bien des cibles, les positions
sont bonnes, et les drones ne partent pas en chasse de détections fantômes.

**Porte de validation.** Le détecteur repère **nettement plus loin** que le décodeur classique ;
son taux de fausses détections est faible ; et les missions tournent normalement avec lui.

**Si ça échoue.** Si la portée gagnée est faible : garder le décodeur classique seul, le
système marche déjà. Si les fausses détections coûtent trop cher : relever le seuil de confiance du
détecteur, quitte à repérer un peu moins loin.

**Cohérence.** Le canal sémantique de la carte, laissé vide à l'étape 4, est maintenant rempli.
Le guide de l'étape 8 pourra s'en servir.

**Ce qui a été fait (2026-09-07) — `experiments/10_detecteur/`, fiche `RESULTATS.md`.**
- *Les exemples se fabriquent seuls, et sont vérifiés.* Projection de la vérité dans l'image,
  puis un rayon du moteur physique par point de surface confirme que rien ne s'interpose ; le
  cadre entoure les points visibles. Huit entrepôts : six d'entraînement (graines 0 à 5), deux
  scellés (9033, 9019) jamais vus à l'apprentissage ; 3 800 images rendues par une caméra libre
  à des poses au hasard, plus les 3 577 images de l'étape 2 ré-annotées et gardées comme juge.
  Contrôle : le centre du panneau visé, projeté par la fonction validée à l'étape 2, tombe dans
  notre cadre à 99,4 % et 100 %.
- *Deux classes, pas trois.* QR et carton. Les zones (allée, rack, mur) ne sont pas des objets
  qu'un cadre délimite : elles viennent de la géométrie de la carte, le repli prévu ci-dessus.
- *Le plus petit réseau suffit.* YOLO11 nano, 2,6 M de paramètres, 1024 px, demi-précision :
  panneau visé repéré à 98 % entre 6 et 8 m sur les mêmes images où le classique tombe à 47 % ;
  98 % des QR visibles trouvés jusqu'à 12 m sur les entrepôts scellés ; 0,7 % d'images sans QR
  avec un QR inventé contre 27 % ; 12 ms par image. Le seuil de mission (0,5) est le plus bas
  qui garde les fausses alertes sous 1 %.
- *Branché sur la carte.* `swarm_qr/detecteur.py` ; dans la patrouille (`--detecteur auto`), un
  QR repéré devient une piste placée par le lidar le long de la direction du cadre, un carton
  repéré marque le canal sémantique (`Carte.marque`). Le repérage classique reste le repli
  sans détecteur.
- *En vol.* Patrouille d'une étagère et deux allées avec le réseau branché : 1 piste sur 143
  hors d'un rack, 99 % à moins de 2 m d'un vrai panneau, 0 point de trajectoire dans une
  structure de rack ; 33 ms par observation pour le détecteur.
- *Deux bugs de la scène trouvés par les rayons.* Les cartons non retenus gardaient leur
  collider (le lidar de l'étape 4 voyait des cartons invisibles ; corrigé, prim désactivé) ; le
  collider d'un carton est plus petit que sa forme visible (visibilité jugée par l'identité de
  l'objet touché, pas par la distance). Et un troisième, mesuré en vol : les racks de
  l'entrepôt n'ont aucune structure solide sur 0,70 m au sud et 0,77 m au nord de leur
  emprise ; l'arbitre juge maintenant la structure, pas le rectangle.

---

## Étape 8 — Le guide vision-langage

**Pourquoi maintenant.** Le système voit bien et décide correctement. On ajoute le deuxième
modèle, celui qui apporte du **bon sens général** là où la géométrie ne sait rien dire — et
c'est la contribution de recherche du projet.

**Ce qu'on cherche.** Les petits modèles vision-langage utilisables localement, et pour chacun :
la mémoire demandée, la vitesse, la licence, la facilité d'installation.

**Sa mission.** Le modèle regarde **l'entrepôt** — l'image de la caméra du drone — **et la
carte** vue de dessus, et il répond à deux questions :

1. **Où aller ?** Il désigne la zone à explorer parmi celles qui lui sont proposées, numérotées
   sur la carte.
2. **Comment l'aborder ?** Par quel côté attaquer le rack, et quel carton repéré viser en
   premier parmi ceux qui attendent d'être lus.

Il répond par ces deux choix et une phrase qui les explique.

**Ce qu'il ne décide pas.** La pose exacte du drone — distance, hauteur, orientation — reste
calculée par la géométrie, à partir des constantes mesurées à l'étape 2. Un modèle de langage
ne juge pas une distance au centimètre ; il juge une situation.

**Le choix retenu : un modèle génératif.** Deux raisons :

- **Il explique son choix.** Il ne dit pas seulement « zone 3 », il dit pourquoi. Un système qui
  justifie ses décisions se montre en démonstration et se défend en soutenance. C'est rare dans
  ce domaine.
- **Il raisonne sur les deux images à la fois**, ce qu'un simple comparateur ne sait pas faire.

**La solution de repli.** Si aucun modèle génératif ne tient dans le budget mémoire, ou si tous
répondent trop lentement, on bascule sur un **encodeur seul** : il ne génère pas de texte, il
calcule un score de ressemblance entre l'image de la caméra et une phrase fixe (« une allée avec
des étagères pleines de cartons »). C'est beaucoup plus rapide, mais on perd l'explication.

**Ce qu'on mesure — hors ligne d'abord, c'est essentiel.** Pendant les missions de l'étape 5, on
a sauvegardé des vues de dessus à intervalles réguliers. On en garde une centaine. Pour chacune,
on connaît **après coup** la bonne réponse : la zone qui contenait le plus de QR restants.

On fait passer chaque candidat sur les cent mêmes cartes et on mesure :
1. **Le taux d'accord avec la bonne réponse**, comparé à un choix au hasard, sur les deux
   questions : la zone choisie, et le côté d'abordage. **C'est la mesure qui décide si on
   branche ce composant.**
2. **La latence** par requête.
3. **La mémoire** occupée.
4. **La qualité des explications**, lues à l'œil sur un échantillon.

**On ne branche le guide que s'il bat nettement le hasard.** Sinon on essaie la solution de
repli. Si rien ne bat le hasard, on **écrit ce résultat négatif** et le système reste à λ = 0 —
il marche déjà. Un résultat négatif mesuré proprement se défend très bien.

**Ce qu'on implémente.** `swarm_qr/advisor.py` : la vue de dessus annotée avec les zones
numérotées, l'image frontale, la question, la lecture de la réponse, **l'appel asynchrone** (la
décision n'attend jamais), et le journal des avis avec leurs explications.

**Puis en intégration : régler λ.** λ dit combien on écoute le guide, et il faut trouver sa
bonne valeur. On lance des missions avec plusieurs valeurs (0 ; 0,5 ; 1 ; 2) sur les mêmes
entrepôts, et on garde celle qui donne le meilleur résultat.

**Le garde-fou.** Aucune valeur de λ ne doit faire pire que λ = 0. Si c'est le cas, le guide
tire le système vers le bas : soit la formule de score est mal équilibrée, soit ses avis sont
mauvais. On corrige avant de continuer.

**Porte de validation.** Un λ est choisi, le guide améliore le résultat ou au moins ne le
dégrade pas, et le journal des explications est lisible.

**Si ça échoue.** Avis au niveau du hasard : simplifier la question posée, essayer un modèle
plus grand, puis la solution de repli, puis rester à λ = 0. Latence trop grande : réduire la
fréquence des appels ou la taille des images envoyées.

**Cohérence.** Le système a maintenant ses sept composants. Il est complet, et prêt pour
l'évaluation.

---

**Ce qui a été fait (2026-09-08) — `experiments/12_guide/`, fiche `RESULTATS.md`.** Le guide
est écrit (`swarm_qr/guide.py`, SmolVLM-500M, deux questions courtes, fil d'arrière-plan, avis
pesé par λ) et jugé hors ligne sur 165 cartes réelles des trois missions finales. Références :
hasard 17 % sur la zone, géométrie seule 42 %, « toujours ouest » 96 % sur le côté. Le modèle :
7 % sur la zone, 42 % sur le côté. **Il n'est pas branché** ; le système final est géométrique.
Le modèle de 2,2 milliards n'a pas pu être téléchargé. Piste non explorée : décrire les zones en
texte plutôt que sur une image.

---

## Étape 9 — L'évaluation complète

**Pourquoi maintenant.** Le système est complet. Il faut produire les chiffres finaux, sur tous
les scénarios définis à l'étape 0 — pas seulement le cas normal.

**Ce qu'on implémente d'abord : le générateur d'entrepôts hors distribution.**
`swarm_qr/env/layouts_ood.py` produit des formes que le générateur habituel ne fait pas : allées
en L, îlots isolés, largeurs variables, zones vides, hauteurs différentes. **Sans lui, on ne peut
pas prouver la généralisation** — c'est donc un composant, pas un détail.

**Ici seulement on ouvre le lot de graines scellé à l'étape 1.**

**C'est ici, et seulement ici, qu'on compare.** Le système complet, les deux références et
l'oracle passent sur les mêmes entrepôts, avec les mêmes graines. C'est la comparaison propre
du projet ; toutes les vérifications faites plus tôt ne servaient qu'à s'assurer que les
composants fonctionnaient.

**Le tableau principal.**

| Politique | % lu | temps pour y arriver |
|---|---|---|
| Balayage fixe (plancher) | | |
| **Le système complet** | | |
| Oracle (plafond) | | |

**Le protocole détaillé, en deux temps.** Une évaluation complète sur tous les scénarios et
toutes les graines coûterait une centaine de missions. On la réduit d'abord à l'essentiel, et
on l'étend plus tard s'il reste du temps.

| Ce qu'on évalue | Scénarios | Graines | Missions |
|---|---|---|---|
| Mission normale | 1 | 5 | 5 |
| Les autres scénarios | 7 | 1 | 7 |
| Balayage et oracle, mission normale | 2 | 5 | 10 |
| Réglage de λ | 3 valeurs | 1 | 3 |
| | | **Total** | **25** |

**Ce que cette réduction change, à écrire dans le mémoire.** Sur la mission normale, cinq
graines donnent une moyenne et un écart-type. Sur les autres scénarios, une seule graine donne
une observation, pas une moyenne : on écrira « sur cet entrepôt, la panne coûte tant de points »
et non « en moyenne ». C'est honnête pour une première passe.

**Réserve à considérer.** Le scénario hors distribution porte l'argument de généralisation,
c'est-à-dire le résultat principal. Une seule graine le rend attaquable. Trois graines y
coûteraient deux missions de plus.

| Scénario | % lu | temps | remarque |
|---|---|---|---|
| Mission normale, entrepôts habituels jamais vus | | | |
| Mission normale, entrepôts **hors distribution** | | | |
| Panne d'un drone à 25 % | | | temps de re-répartition |
| Panne d'un drone à 50 % | | | temps de re-répartition |
| Panne d'un drone à 75 % | | | temps de re-répartition |
| Obstacle apparu en cours de mission | | | |
| Inventaire modifié en cours de mission | | | |
| QR dégradés | | | |

**Ce que chaque ligne prouve.** Les deux premières prouvent la découverte et la généralisation.
Les trois suivantes prouvent la résilience, et on mesure aussi **le temps de re-répartition**
après la panne. Les deux suivantes prouvent l'adaptation au changement. La dernière teste la
robustesse de la perception.

**La prédiction à vérifier.** Si le guide apporte quelque chose, son gain doit être **plus grand
sur les entrepôts hors distribution** que sur les entrepôts habituels. C'est logique : c'est là
que notre géométrie ne sait rien et que sa connaissance générale compte. **Cette comparaison est
le résultat le plus intéressant du projet**, quel que soit son sens.

**Porte de validation.** Le tableau est rempli, les écarts-types sont donnés, et la lecture est
écrite honnêtement — y compris si un composant n'apporte rien.

---

## Étape 10 — Optionnelle : une décision apprise

**Condition d'entrée, à vérifier avant de commencer.** L'étape 6 a montré une marge nette entre
l'oracle et le balayage, **et** le système n'en capture qu'une partie. Si la marge est faible,
cette étape est abandonnée d'avance et on l'écrit : ce n'est pas un renoncement, c'est une
conclusion mesurée.

**Le principe.** On remplace **uniquement** le composant C6, la décision : un petit réseau
apprend à choisir parmi les candidats de C4, à la place de la formule de score. Tout le reste ne
change pas. L'entraînement se fait sans rendu d'images : le capteur est remplacé par son modèle
mesuré à l'étape 2.

**Les garde-fous.** Difficulté tirée au hasard à chaque épisode, jamais une progression par
paliers pilotée par un seuil de réussite. Vérifier **avant tout lancement** qu'une politique
aléatoire touche parfois la récompense — sinon l'apprentissage ne peut pas démarrer. Un budget
d'essais fixé à l'avance.

**Porte de validation.** La politique apprise bat la version géométrique d'une marge nette sur
les entrepôts de test. Sinon on garde la géométrie et on écrit le résultat.

---

## Étape 11 — La consolidation

Les figures : courbe de lecture au cours du temps, cartes avant et après, le tableau de
l'étape 9, le banc hors-ligne du guide. Une vidéo de
mission avec la vue de dessus qui se remplit et les explications du guide affichées. Le chapitre
méthode qui reprend ce plan étape par étape, avec pour chacune le test et le chiffre qui l'ont
validée. Et la liste honnête des limites.

**Porte de validation finale.** Chaque chiffre du mémoire est reproductible par un script du
dossier `swarm_qr/experiments/`, en une commande.

---

# PARTIE 3 — Les règles qui valent partout

1. **Séquentiel.** On ne commence pas une étape tant que la porte de la précédente n'est pas
   franchie.
2. **Chaque test dit ce qu'il prouve.** Tous les tests sont des scripts rangés dans
   `swarm_qr/tests/` ou `swarm_qr/experiments/`, relançables en une commande, avec leurs
   résultats écrits à côté.
3. **Mesurer avant de choisir.** Aucun composant n'est choisi sur sa réputation. On construit un
   banc, on compare les options sur les mêmes données, on garde le meilleur, on écrit pourquoi.
4. **λ = 0 doit rester disponible en permanence.** Aucun composant optionnel ne peut casser le
   système.
5. **Jamais deux inconnues à la fois.** Un composant neuf est testé seul, puis intégré à un
   ensemble déjà validé, et mesuré contre l'état précédent.
6. **Aucune valeur héritée.** Chaque constante vient d'une mesure faite ici, enregistrée dans
   `swarm_qr/docs/constantes_mesurees.md` avec le script qui l'a produite.
7. **Le lot de graines d'évaluation reste scellé** jusqu'à l'étape 9. On ne règle jamais un
   paramètre en regardant les entrepôts de test.
8. **Journal des décisions.** Une ligne par choix — quoi, pourquoi, quelle alternative écartée —
   dans `swarm_qr/docs/journal_decisions.md`. C'est ce qui remplira le chapitre méthode sans
   effort à la fin.

---

# Récapitulatif : où en suis-je ?

Les étapes sont listées dans **l'ordre où elles se font**.

| Rang | Étape | Ce qu'on construit | Ce qui doit être validé pour continuer |
|---|---|---|---|
| — | 0 | Le cadre | Vous avez validé la page |
| — | 1 | L'environnement | Reproductible, physique honnête, images nettes |
| — | 2 | Le décodeur | **L'enveloppe de lecture est mesurée** |
| 1 | 3 | Le contrôleur | **Le drone tient la pose**, et trois drones volent ensemble |
| 2 | 4 | La carte | Capteur vérifié contre la physique, carte exacte, itinéraires sûrs, réservations qui expirent, vue lisible |
| 3 | 7 | L'œil appris | **FAIT** : portée 8 m contre 4 m, 0,7 % de fausses alertes contre 27 % |
| 4 | 5 + 8 | Cibles, décision et guide | **Étape 5 FAITE** : missions nominale et panne 5/5, 110 et 114 codes sur 114 ; entrepôt 9019 ouvert (dérèglement simultané) ; **étape 8** : guide écrit et mesuré, non branché (voir 12_guide) |
| 5 | 6 | Les références | Le tableau balayage / système / oracle |
| 6 | 9 | L'évaluation | Le tableau complet des scénarios |
| 7 | 10 | La décision apprise (option) | Bat la géométrie, sinon on l'écrit |
| 8 | 11 | La consolidation | Chaque chiffre reproductible |
