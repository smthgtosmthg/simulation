# Direction 2, version B — un modèle vision-langage comme guide d'exploration

Date : 29 août 2026
Ce document reprend la version B de la direction 2, que j'avais écartée trop vite, et la
construit sérieusement.

## Pourquoi je reviens dessus

Mon argument contre était : « c'est nous qui générons les entrepôts, donc tout a priori de
structure qu'un modèle pourrait deviner, nous l'avons écrit nous-mêmes ; on paierait cher pour
faire deviner une règle qu'on a tapée au clavier ».

**Cet argument tombe dès qu'on teste sur des entrepôts qu'on n'a pas générés.** Et c'est
justement l'objectif : la solution doit marcher sur des entrepôts inédits, pas seulement sur
les nôtres. Dans ce cas, le modèle apporte une connaissance qui n'est nulle part dans notre
code.

Cette version B n'est donc pas un repli. C'est la direction dont l'argument central est **la
généralisation** — exactement ce qu'on cherche.

## L'idée en une phrase

Le modèle vision-langage ne pilote pas et ne décide pas seul. Il **donne un avis** sur les
zones à explorer, quelques fois par seconde, et cet avis se combine à un calcul géométrique.
Un contrôleur classique fait le vol.

## L'architecture, couche par couche

```
30 Hz   couche 3 : contrôle       vol + maintien de la pose de lecture
 5 Hz   couche 0 : perception     détection/décodage QR, occupation
 1 Hz   couche 2 : décision       utilité géométrique  ×  avis du modèle  →  point de vue
0,3 Hz  couche 1 : avis du modèle quelle zone mérite d'être explorée, et pourquoi
```

### Couche 0 — Perception (classique, rien d'appris)

Trois flux, et c'est un point de conception important : **une caméra ne peut pas tout faire.**

| Caméra | Rôle | Résolution |
|---|---|---|
| 2 latérales | lire les QR | **≥ 1000 px** (mesure 01 : indispensable pour décoder à 1,25 m) |
| 1 frontale | naviguer et alimenter le modèle | 224 à 336 px (l'entrée du modèle) |
| profondeur / lidar | occupation, obstacles | — |

Le détecteur de QR sort **trois états**, et cette distinction porte toute l'architecture :
`décodé` / `motif repéré mais non décodé` / `rien`. Le deuxième est le plus riche : il dit
« il y a une cible ici », avec sa position 3D estimée.

### Couche 1 — La carte (construite en ligne, classique)

Une carte 3D partagée par les trois drones, en cellules de 0,25 à 0,5 m :

| Canal | Contenu |
|---|---|
| occupation | libre / occupé / inconnu |
| couverture | observé par une caméra, et sous quel angle |
| carton non lu | motif repéré, non décodé, avec sa normale |
| lu | code décodé |
| coéquipiers | positions et objectifs annoncés |
| **sémantique** | **l'étiquette donnée par le modèle : allée, mur, zone ouverte, rack** |

Le dernier canal est ce que la direction 2 ajoute. Les autres existent dans toutes les
directions.

### Couche 2 — L'avis du modèle (c'est ici que se joue la direction)

**Quand ?** Pas à chaque pas. Seulement quand une décision de zone se pose : au démarrage,
quand une allée est finie, quand la carte change beaucoup, ou quand un drone tombe en panne.
En pratique **toutes les 3 à 5 secondes**, soit ~0,2-0,3 Hz.

**Entrée.** Trois choses :
1. l'image frontale actuelle ;
2. une **vue de dessus de la carte construite jusqu'ici**, rendue comme une image simple
   (gris = inconnu, blanc = exploré, orange = cartons non lus, vert = lus, croix = drones) ;
3. une liste de 4 à 8 zones candidates, numérotées sur cette vue.

**Question posée.** Quelque chose comme :
> « Voici une carte partielle d'un entrepôt. Les zones grises ne sont pas encore explorées.
> Quelle zone numérotée contient le plus probablement des étagères avec des cartons non
> encore scannés ? Réponds par le numéro, puis une phrase d'explication. »

**Sortie.** Un numéro de zone + une phrase. La phrase ne sert pas à décider — elle sert à
**expliquer**, et c'est utile pour le mémoire.

**Modèle.** Un petit modèle qui tient dans 2 à 4 Go : SmolVLM (0,5 milliard), Florence-2-base,
Moondream2, ou SmolRGPT (0,6 milliard, conçu pour les entrepôts). Aucun ne demande de
réentraînement pour commencer.

**Variante moins chère, à essayer en premier :** pas de génération de texte du tout. On prend
seulement l'encodeur d'images (type CLIP) et on calcule une similarité entre chaque zone et une
phrase fixe (« une allée d'entrepôt avec des étagères remplies de cartons »). C'est 10 à 50 fois
plus rapide, et c'est ce que fait VLFM. À comparer avec la version qui génère.

### Couche 3 — La décision (1 Hz)

C'est ici qu'on combine. Pour chaque point de vue candidat :

```
score = utilité_géométrique  +  λ · avis_du_modèle  −  coût_de_trajet  −  pénalité_coéquipier
```

où `utilité_géométrique` = combien de cellules inconnues ou « carton non lu » ce point de vue
permettrait d'observer.

**λ est le paramètre le plus important de toute l'architecture.** Il dit combien on fait
confiance au modèle. Et il donne l'expérience qui tranche :

- **λ = 0** → planificateur purement géométrique. C'est la référence.
- **λ > 0** → le modèle influence la décision.

**Si le modèle n'apporte rien, λ = 0 gagnera, et on l'aura mesuré.**

### Couche 4 — Le contrôle (30 Hz, classique)

Deux phases : transit rapide vers le point de vue, puis approche finale sous 0,45 m/s à la
bonne distance et au bon cap, maintenue quelques images. C'est le morceau le plus délicat, et
il est identique dans toutes les directions.

### Coordination des 3 drones

Carte partagée + réservation des points de vue. Le modèle est une **ressource commune** : un
seul drone l'interroge à la fois, et sa réponse (« la zone 3 est prometteuse ») est partagée.
En cas de panne, les réservations expirent et les autres reprennent les cibles.

## Ce que cette direction apporte vraiment

**1. La généralisation sans réentraînement.** C'est l'argument central. Une politique apprise
sur nos entrepôts risque de sur-apprendre notre générateur. Le modèle fondation, lui, n'a jamais
vu nos entrepôts et n'en a pas besoin : il raisonne sur ce qu'il voit. **Sur un entrepôt d'un
type inédit, c'est lui qui devrait tenir le mieux.**

**2. L'explication.** Le système peut dire *pourquoi* il va à gauche plutôt qu'à droite. Très
peu de travaux d'inventaire par drone font ça. C'est défendable en soutenance et ça se montre
en démonstration.

**3. Le raisonnement de structure.** Reconnaître « cet entrepôt a des allées parallèles » contre
« c'est un plateau ouvert avec des îlots » et adapter la stratégie. Un planificateur de
frontières ne fait pas cette distinction.

## Ce qui doit être vrai — et comment le vérifier

| # | Condition | Comment la tester |
|---|---|---|
| 1 | Un petit modèle tient dans le budget mémoire restant | mesurer : charger SmolVLM/Florence-2 et lire `nvidia-smi` |
| 2 | Sa latence tient sous ~1 s par requête | chronométrer 50 requêtes |
| 3 | Ses réponses sont **meilleures que le hasard** sur nos cartes | 100 cartes partielles, comparer son choix au meilleur choix connu |
| 4 | **Il apporte plus sur un entrepôt inédit que sur les nôtres** | l'expérience décisive, ci-dessous |
| 5 | Le contrôleur tient la pose de lecture | commun à toutes les directions |

## L'expérience qui tranche — et qui teste directement notre désaccord

Mon argument disait : le modèle n'apporte rien parce que nous générons les entrepôts.
Votre argument dit : mais on testera sur d'autres entrepôts.

**On peut mesurer lequel de nous deux a raison.** Deux jeux de test :

| Jeu de test | Ce que c'est |
|---|---|
| **Dans la distribution** | des entrepôts produits par notre générateur, avec d'autres graines |
| **Hors distribution** | des entrepôts d'un **type différent** : allées en L, îlots, largeurs variables, zones vides, produits par un autre générateur ou dessinés à la main |

Et deux systèmes : λ = 0 (géométrique seul) contre λ > 0 (avec le modèle).

```
                        λ = 0        λ > 0        gain
dans la distribution      A            B          B − A
hors distribution         C            D          D − C
```

- **Si (D − C) > (B − A)** → le modèle aide surtout là où c'est nouveau. **Vous avez raison**,
  et c'est un résultat publiable : le gain d'un modèle fondation croît avec la nouveauté.
- **Si (D − C) ≈ (B − A) ≈ 0** → il n'apporte rien, j'avais raison, et on l'a mesuré au lieu
  d'en discuter.

C'est, à mon avis, **la meilleure expérience de tout le projet** : elle transforme un désaccord
d'opinion en un chiffre, et elle produit un résultat intéressant dans les deux cas.

## Les risques, honnêtement

1. **Les petits modèles sont faibles en raisonnement spatial.** Sur le banc MINDCUBE, ils
   plafonnent à 37,8 % sans entraînement adapté (61,3 % avec). Il faut donc s'attendre à des
   avis médiocres au début, et prévoir la variante « encodeur seul » comme repli.
2. **Le budget mémoire est serré.** Isaac Sim consomme plusieurs gigaoctets. **Mitigation
   simple : le modèle n'a pas besoin d'être dans la boucle d'entraînement.** On entraîne le
   contrôleur et la politique sans lui, et on l'ajoute au moment de l'évaluation. Le conflit
   de mémoire disparaît.
3. **Le modèle peut ajouter du bruit.** D'où la structure en `λ` : il ne décide jamais seul, il
   biaise. Avec λ = 0 toujours mesuré à côté, on ne peut pas se tromper sans le voir.
4. **Notre entrepôt actuel est trop régulier pour que ça se voie.** Sur des allées parallèles,
   un balayage est déjà presque optimal. **C'est pour ça que le jeu de test hors distribution
   est indispensable** — sinon l'expérience ne peut rien montrer.

## Par où commencer, concrètement

| Étape | Durée | Ce qu'on obtient |
|---|---|---|
| 1. Mesurer un petit modèle : mémoire, latence, qualité sur 100 cartes | 2 jours | conditions 1, 2, 3 |
| 2. Écrire le générateur d'entrepôts **hors distribution** | 2 jours | le jeu de test qui rend l'expérience possible |
| 3. Carte + candidats + planificateur géométrique (λ = 0) | 1 semaine | la référence, utile à toutes les directions |
| 4. Contrôleur avec maintien de la pose | 1 semaine | commun à toutes les directions |
| 5. Brancher le modèle, balayer λ, remplir le tableau 2×2 | 1 semaine | **le résultat du mémoire** |

Points importants : les étapes 3 et 4 servent **quelle que soit la direction retenue**. On ne
parie donc pas tout sur le modèle. Et l'étape 1 peut se faire dès maintenant, sans rien casser.

## Lien avec les autres directions

Cette architecture est **la même que la direction 3**, avec une couche en plus. Perception,
carte, décision, contrôle, coordination : identiques. La seule différence est que la couche de
décision reçoit un avis supplémentaire.

Ce qui veut dire qu'on ne choisit pas vraiment entre D2-B et D3 : **on construit le socle
commun, puis on branche successivement le planificateur géométrique, l'avis du modèle, et la
politique apprise.** Chacun est une ligne de plus dans le même tableau de comparaison.
