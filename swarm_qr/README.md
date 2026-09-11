# swarm_qr

Trois drones découvrent un entrepôt inconnu et lisent les QR codes collés sur les cartons.
Système construit, sans apprentissage. Le plan complet est dans `docs/plan_systeme_swarm.md`.

Socle : Isaac Sim 5.1 + Pegasus Simulator + ArduPilot SITL. Chaque drone est un vrai
multirotor Iris piloté par MAVLink — inertie, estimateur, contrôleur de bord réels.

## Organisation

```
env/            le socle
  config.py     les constantes, toutes mesurées dans l'entrepôt
  layout.py     une graine donne un entrepôt (Python pur, testable sans simulateur)
  qr_tags.py    génération des QR et collage sur les cartons
  scene.py      assemblage : entrepôt, racks, cartons, drones Iris, caméras
  pilot.py      la parole au pilote automatique : lien MAVLink, décollage, repères, horloge
perception.py   lire un QR dans une image : lecteurs, coins, position 3D
control.py      le contrôleur : amener un drone à une pose, l'y tenir, dire s'il a réussi
mapping.py      la carte partagée : occupation et couverture orientée, panneaux, réservations,
                frontières, itinéraires, vue de dessus, canal sémantique
detecteur.py    l'œil appris : un réseau qui repère les QR et les cartons sans les lire
observation.py  une observation d'un drone versée dans la carte : lidar, lectures, œil appris, couverture
planning.py     le cerveau géométrique : cibles à lire, à couvrir, à explorer ; note ; choix
guide.py        le guide vision-langage : une zone et un côté conseillés, en arrière-plan
mission.py      le chef d'orchestre : plusieurs drones, trois rythmes, une carte, un rapport
tests/          les tests sans simulateur (contrôleur sur un faux pilote, carte sur un monde de boîtes)
experiments/    un dossier par test, avec ses images et sa fiche de résultats
assets/qr/      les images de QR générées
assets/detecteur/  les poids retenus par le banc de l'étape 7 et leurs réglages
```

Toute installation dans `~/isaac5_env` passe par `pip install -c ../contraintes_isaac.txt` :
sans ce fichier, une dépendance peut remonter numpy et casser le simulateur.

## Lancer les tests de l'étape 1

Chaque test se lance seul — la commande exacte est dans l'en-tête de son `run.py`. Compter 4 à
10 minutes par lancement du simulateur ; les tests en vol lancent le SITL automatiquement et
demandent `DISPLAY=:1`.

## Les quatre tests

| Dossier | Question |
|---|---|
| `01_reproductibilite` | La même graine donne-t-elle le même entrepôt ? |
| `02_variation` | Des graines différentes donnent-elles des entrepôts différents ? |
| `03_images_qr` | Les QR sont-ils nets, et jusqu'à quelle distance se décodent-ils ? |
| `04_debit` | Combien de pas par seconde tient le simulateur ? |

Chaque dossier contient son script, ses images, sa vidéo le cas échéant, et un `RESULTATS.md`.

## Deux pièges du simulateur

La position d'un drone ne se lit que par `scene.positions()`. Les lecteurs USD classiques
renvoient la position de départ, figée, même après des dizaines de pas — les images, elles,
suivent bien le drone, donc l'erreur se voit très tard.

Isaac ignore le signal d'arrêt normal : toujours lancer avec `timeout -s KILL`, et vérifier
`nvidia-smi` avant de relancer, un processus mort peut garder plusieurs gigaoctets.

## Étape 1 — résultats (socle Pegasus + SITL)

| Test | Verdict | Chiffre clé |
|---|---|---|
| 1 Reproductibilité | validé | disposition identique octet par octet, 0 % de géométrie déplacée |
| 2 Variation | validé | paire de graines la plus proche : 1,85 m d'écart ; 62 à 118 cartons |
| 3 Lecture en vol | validé | les six distances lues, de 0,5 à 3 m, arrivées à 4-11 cm ; 6 QR lus en un passage |
| 4 Débit | validé | 206 pas/s en régime de mission (rendu 5 images/s), 0,26× le temps réel |

Huit défauts trouvés et corrigés par ces tests : cartons comptés deux fois, caméra de survol
mal orientée, plafond masquant la vue, rendu jamais ralenti, caméras au centre du drone puis
coque et hélices dans le champ, arrivée validée sans le cap (caméra de travers), et calibration
du repère NED fausse de 10 degrés (l'impulsion de vitesse est polluée par le contrôleur — on
compare maintenant le cap rapporté par ArduPilot au cap vrai du simulateur).

Une constante à retenir : le QR fait 40 cm de côté. La « distance minimale d'un mètre »
annoncée par les premières versions était fausse — elle venait d'une visée décalée de 10 cm ;
bien visé, le code se lit dès 0,5 m.

## Étape 2 — Jusqu'où le drone peut-il lire un QR code ?

3 481 images, chacune avec la position exacte de la caméra, analysées par six lecteurs.
La campagne a été refaite entièrement une seconde fois, code réécrit, pour vérification :
les chiffres se confirment. Détail dans
[`07_enveloppe/RESULTATS.md`](experiments/07_enveloppe/RESULTATS.md).

| Question posée | Réponse mesurée |
|---|---|
| Jusqu'où le drone lit-il ? | zone fiable de **1,5 à 4 m** de distance apparente (≥ 90 %) |
| Faut-il choisir entre distance et angle ? | non : distance apparente = distance ÷ cos(angle) |
| La vitesse gêne-t-elle ? | **non** : 100 % de lecture jusqu'à 1 m/s, panneau dans le cadre |
| Sait-il dire où est le code lu ? | oui, à 0,8 cm près en médiane |
| Invente-t-il des codes ? | jamais, sur 300 images sans le moindre QR |
| Repère-t-il des motifs à tort ? | oui, 27 % — un détecteur appris sera nécessaire (étape 7) |
| Quel lecteur ? | **zxing** retenu, zbar en alternative ; l'ancien perdait onze points |
| Le vol dégrade-t-il la lecture ? | non : le vrai drone lit à 100 % de 1 à 4 m |

La distance apparente augmente avec l'angle : un code vu de biais paraît plus loin. Avec une
visée bien centrée, la lecture marche dès 0,5 m (test 3) ; la borne de 1,5 m couvre une visée
réaliste. Et l'allée entre deux racks limite le recul à 3,6 m : c'est l'angle, presque gratuit,
qui permet de couvrir plusieurs cartons depuis une même position.

## Étape 3 — Le drone sait-il aller quelque part et s'y tenir ?

Un contrôleur qui ne bloque jamais (`control.py`), trois vols de mesure, treize tests sans
simulateur. Détail dans [`08_controle/RESULTATS.md`](experiments/08_controle/RESULTATS.md).

| Question posée | Réponse mesurée |
|---|---|
| Faut-il un contrôleur ? | oui : couper la vitesse fait glisser le drone de 0,79 m |
| Quelle loi ? | vitesse proportionnelle à la distance restante, sur la position vraie : 1 cm en tenue |
| Cent poses au hasard ? | **100 sur 100 atteintes**, aucun abandon, 9 cm à l'arrivée, cap exact |
| Combien de temps par cible ? | **14,6 s** en médiane ; environ 2 s + 0,85 s par mètre de trajet |
| Le QR se lit-il une fois arrivé ? | 93 sur 100, conforme au taux par image de l'étape 2 |
| Trois drones en même temps ? | **oui** : 6 cibles sur 6, 6 lectures sur 6, 0,23 fois le temps réel |

Deux pièges mesurés : une consigne de vitesse nulle laisse dériver le drone (1 à 3 cm/s), et le
pilote automatique tient le cap qu'il *croit* avoir, jusqu'à 8,7 degrés de la vérité — d'où
une tenue active et une boucle de cap sur le cap vrai. Le contrôleur ne calcule pas son chemin
et n'évite pas les autres drones : deux drones se sont croisés à 0,96 m dans le même couloir,
protégés par leurs altitudes différentes. C'est à l'étape 5 de l'interdire.

## Étape 4 — La carte partagée

Une grille pour l'espace, une table pour les panneaux, un lidar sur le drone, des chemins
calculés sur ce que le drone a découvert, et 28 tests sans simulateur. Détail dans
[`09_carte/RESULTATS.md`](experiments/09_carte/RESULTATS.md) ; la carte en trois dimensions
dans `09_carte/carte_3d.html`.

| Question posée | Réponse mesurée |
|---|---|
| Le lidar dit-il la vérité ? | oui : **0,0 cm** d'écart contre le moteur physique, rayon par rayon, à trois caps |
| La carte invente-t-elle des obstacles ? | 0,39 % des cases occupées, en pleine allée |
| Où sont les codes lus ? | à **0,9 cm** près en médiane, 8,3 cm au pire, aucun code inventé |
| Combien de codes en une patrouille ? | **110 sur 114**, les quatre autres portent des étiquettes de 12 cm, trop petites pour la distance de patrouille |
| « Lisible ici » est-il vrai ? | 95 % des panneaux annoncés lisibles ont été lus, contre 37 % des autres |
| Le drone vole-t-il sur sa seule carte ? | oui : 15 transits sur 15, aucun point de trajectoire dans un rack, chemin recalculé en vol quand un obstacle apparaît |
| Un chemin peut-il traverser un obstacle connu ? | jamais, sur 12 trajets d'un coin à l'autre |
| Ce que ça coûte | 1,5 Mo, 24 ms par observation pour la carte |
| La panne d'un drone ? | ses réservations expirent seules, sans code qui la surveille |

Cinq faits découverts en comparant à la vérité : le lidar compte ses angles verticaux **vers le
bas** et ne se rafraîchit qu'au **rendu** ; **les étiquettes n'ont pas toutes la même
taille**, donc la distance d'un code vient du lidar et non de sa taille dans l'image ; chaque
carton porte le même code sur ses deux faces ; une case vue à travers un rack ne rend pas
lisible la face opposée ; et le décodeur lisait les codes-barres imprimés sur le décor.

## Étape 7 — L'œil appris

Un réseau YOLO11 nano qui **repère** les QR et les cartons, sans les lire, de beaucoup plus
loin que le décodeur. Ses exemples se fabriquent sans un seul clic : la vérité de la scène est
projetée dans l'image, et un rayon du moteur physique confirme, point par point, que l'objet
est visible. Six entrepôts pour apprendre, deux entrepôts scellés pour juger, et les images de
l'étape 2 comme juge commun avec le repérage classique. Détail dans
[`10_detecteur/RESULTATS.md`](experiments/10_detecteur/RESULTATS.md).

| Question posée | Réponse mesurée |
|---|---|
| Repère-t-il plus loin que le classique ? | oui : **98 %** des panneaux visés entre 6 et 8 m sur les mêmes images où le classique tombe à 47 % ; 98 % des QR visibles jusqu'à 12 m sur les entrepôts jamais vus |
| Invente-t-il des objets ? | **0,7 %** des images sans QR, contre 27 % pour le classique |
| Voit-il les cartons ? | 94 % des cartons visibles |
| Ce que ça coûte | 2,6 M de paramètres, 63 Mo, 12 ms par image |
| Les cadres appris sont-ils justes ? | le centre validé de l'étape 2 tombe dans notre cadre à 99,4 % et 100 % |
| En vol, fait-il courir les drones après des fantômes ? | non : 1 piste sur 143 hors d'un rack, 99 % à moins de 2 m d'un vrai panneau ; 0 point de trajectoire dans une structure de rack |
| Ce qu'il a révélé | les cartons cachés gardaient leur collider (corrigé) ; le collider d'un carton est plus petit que sa forme visible ; les racks n'ont aucune structure sur 0,7 m à chaque bout de leur emprise |

## Étape 5 — Le cerveau et la mission à trois drones

Chaque drone choisit ses cibles sur la carte que les trois construisent, jamais sur le plan :
pistes à lire, surfaces jamais regardées du bon côté, frontières. Une note par cible, un chemin
sur la carte, des réservations qui expirent, une règle de priorité, et les coéquipiers comme
obstacles mobiles. Détail dans [`11_mission/RESULTATS.md`](experiments/11_mission/RESULTATS.md).

| Question posée | Réponse mesurée |
|---|---|
| Le cerveau raisonne-t-il juste avant de voler ? | 12 cartes jouets, 60 tests verts |
| La mécanique à trois drones tient-elle ? | mission nominale et mission avec panne : **5 vérifications sur 5**, aucune chute, aucun point dans un rack, 2 m au plus près entre drones |
| L'inventaire est-il lu sans plan ? | **110 puis 114 sur 114**, 90 % des codes trois minutes après le décollage |
| La panne d'un drone ? | absorbée : sa zone reprise 20 s après, inventaire complet |
| Sur un entrepôt jamais vu ? | 63 codes sur 65, 5 vérifications sur 5, aucune chute ; un premier vol avait vu les trois drones se dérégler ensemble vers 300 s, non reproduit, journal daté en place |
| Ce que huit missions ont appris | neuf règles, chacune née d'un vol qui l'a rendue nécessaire, listées dans le bilan |

## Étape 9 — L'évaluation : le système complet, ses références, ses vidéos

**Le système complet** = la géométrie plus le guide entraîné : `mission.py --guide entraine --lam 1.0`.
Le guide (`guide.GuideEntraine`) est le modèle Qwen2.5-VL 3B en 4 bits avec l'adaptateur LoRA
appris sur la mission (`experiments/12_guide/adaptateur_lora`, 15 Mo) ; il lit le dossier épuré de
la carte, sans image, et conseille une zone ; son avis vaut λ × 10 points dans la note des cibles.

**L'arrêt de mission** utilise la taille connue de l'inventaire : `--part-arret 0.95 --grace 60
--sans-progres 120` (défauts) — arrêt 60 s après 95 % des codes lus, ou après 120 s sans code
nouveau, ou au budget. `--codes-attendus N` remplace le nombre de cartons de la scène.

**Les références**, même perception, même contrôleur, même juge (`baselines.py`) :
`--politique zigzag` suit la méthode statique de Pore et al. (Symmetry 2026) — arrêts
hover-and-scan tous les 1,5 m le long des faces accessibles, hauteur par paliers d'étagère, faces
réparties entre drones, serpentins en miroir ; `--politique glouton` est l'oracle qui connaît la
position de tous les codes et va au plus proche non lu.

**Les cas** : nominal (9033), panne d'un drone à 200 s (`--panne 1:200`), entrepôt jamais vu (9019),
obstacle apparu à 200 s au milieu du couloir central (`--obstacle -4.96,4.0,200`, un bloc de
1 × 1 × 2 m avec collision ; le juge compte les passages dans son emprise).

**La vidéo** (`--video`) : cinq caméras fixes sur les murs (`scene.CAMERAS_VIDEO`), rendues à chaque
cycle, écrites en JPEG dans `<sortie>/video/` ; `experiments/11_mission/video.py --dossier <sortie>`
assemble une vidéo par caméra et `mission.mp4`, la mosaïque des quatre couloirs avec la vue
d'ensemble et la caméra de lecture en médaillons — à vitesse réelle, cinq images par seconde.

**Tout enchaîner** : `bash experiments/11_mission/evaluation.sh systeme|zigzag|glouton` fait les
quatre cas d'une politique, avec jugement et vidéo, dans `experiments/11_mission/{eval,zigzag,glouton}_<cas>`.
