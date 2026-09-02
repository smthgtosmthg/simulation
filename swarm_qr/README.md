# swarm_qr

Trois drones découvrent un entrepôt inconnu et lisent les QR codes collés sur les cartons.
Système construit, sans apprentissage. Le plan complet est dans `docs/plan_systeme_swarm.md`.

Socle : Isaac Sim 5.1 + Pegasus Simulator + ArduPilot SITL. Chaque drone est un vrai
multirotor Iris piloté par MAVLink — inertie, estimateur, contrôleur de bord réels.

## Organisation

```
env/            le système
  config.py     les constantes, toutes mesurées dans l'entrepôt
  layout.py     une graine donne un entrepôt (Python pur, testable sans simulateur)
  qr_tags.py    génération des QR et collage sur les cartons
  scene.py      assemblage : entrepôt, racks, cartons, drones Iris, caméras
  pilot.py      pilotage MAVLink : connexion, décollage, goto position + cap, calibration
experiments/    un dossier par test, avec ses images et sa fiche de résultats
assets/qr/      les images de QR générées
```

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
