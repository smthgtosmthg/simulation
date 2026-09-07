# Étape 3 — Le drone sait-il aller quelque part et s'y tenir ?

## Ce qu'on voulait savoir

Le drone doit rejoindre une pose de lecture devant un carton, s'y tenir le temps de lire, et
dire s'il a réussi ou s'il abandonne. Il faut aussi que trois drones puissent faire cela en
même temps, car le système final est un essaim. Et il faut connaître le temps d'un cycle,
parce que c'est lui qui décide combien de cartons une mission peut lire.

## Comment on a mesuré

Trois vols, sur le même entrepôt que l'étape 2. D'abord un test de freinage : trois façons de
s'arrêter comparées sur un trajet identique de 4 mètres, trois fois chacune. Ensuite cent poses
tirées au hasard devant les cartons, à des distances, des angles et des hauteurs variés, avec
une lecture réelle du QR après chaque arrivée. Enfin trois drones qui décollent et volent en
même temps vers trois cibles, puis vers trois autres.

La position vraie du drone est enregistrée à chaque pas, et la calibration de la caméra est
vérifiée avant chaque vol. Le contrôleur lui-même a treize tests qui tournent sans simulateur,
sur un faux drone, pour vérifier ses phases et ses abandons en une fraction de seconde.

## Le test de freinage : il fallait bien un contrôleur

On a d'abord donné au drone la commande la plus bête possible : avancer à vitesse constante et
couper tout en arrivant. **Il glisse de 79 centimètres et s'immobilise à 72 centimètres de sa
cible**, trois fois sur trois. Un drone a de l'inertie ; couper la vitesse ne l'arrête pas.

| façon de s'arrêter | glissade au-delà | arrivée | erreur finale | tient la pose |
|---|---|---|---|---|
| couper la vitesse | 0,79 m | — | 0,72 m | non, 0/3 |
| **vitesse proportionnelle à la distance restante** | 0,14 m | 4,7 s | **0,01 m** | **oui, 3/3** |
| consigne de position native d'ArduPilot | 0,20 m | jamais dans la tolérance | 0,25 m | non, 0/3 |

**La loi retenue commande une vitesse proportionnelle à la distance qui reste**, plafonnée à
1 m/s : loin, le drone va vite ; près, il ralentit tout seul. Elle arrive en 4,7 secondes sur
4 mètres et tient ensuite la pose à un centimètre.

La consigne de position native d'ArduPilot, elle, s'arrête toujours à 25 centimètres de la
cible. La raison est instructive : cette consigne est exprimée dans le repère que le pilote
automatique se construit avec son GPS, et ce repère est décalé d'une vingtaine de centimètres
de la position vraie. Notre loi boucle sur la position vraie, donc elle converge.

## Deux pièges trouvés en route

**Une vitesse nulle n'est pas une tenue de position.** Dans un premier essai, une fois arrivé,
le contrôleur envoyait « vitesse zéro ». Le drone dérivait alors de 1 à 3 centimètres par
seconde, jusqu'à 44 centimètres en quatorze secondes. La tenue garde maintenant la loi de
position active, y compris après un abandon, pour ne pas dériver vers un rack.

**Le pilote automatique tient le cap qu'il croit avoir.** Son estimation du cap, faite avec un
compas simulé, s'écarte de la vérité jusqu'à 8,7 degrés selon la direction. Envoyer un cap
absolu ne suffisait donc pas : le drone restait à 8 degrés de la consigne, sans jamais valider
son arrivée. Le contrôleur commande maintenant une vitesse de rotation proportionnelle à
l'écart de cap vrai, et l'erreur finale de cap est tombée à zéro.

## Cent poses : cent arrivées

| ce qu'on a mesuré | valeur |
|---|---|
| Poses atteintes | **100 sur 100**, aucun abandon |
| Erreur de position à l'arrivée | 8,9 cm en médiane, de 3 à 13 cm (tolérance : 15 cm) |
| Erreur de cap | 0 degré |
| Vitesse résiduelle en tenue | 0,10 m/s |
| Temps de cycle, médiane | **14,6 s** (transit 8,2 + approche 7,1 + tenue 0,6) |
| Temps de cycle, 9 poses sur 10 sous | 25 s |
| Dans la même allée (46 poses) | 6,7 s |
| Par le couloir (54 poses) | 21,4 s |
| Vitesse moyenne effective | 1,03 m/s, transit compris |
| QR lu après l'arrivée | **93 sur 100** |

Le temps de cycle suit la longueur du trajet presque exactement : environ 2 secondes plus
0,85 seconde par mètre. C'est le chiffre à utiliser pour estimer ce qu'une mission peut lire.

Les sept lectures manquées ne sont pas des échecs du contrôleur : le drone était en place, à
moins de 12 centimètres de la pose demandée. Elles correspondent au taux de lecture par image
mesuré à l'étape 2 dans la même zone — 48 sur 50 entre 1,5 et 2,5 mètres de distance
apparente, 36 sur 40 entre 2,5 et 3,2, 9 sur 10 au-delà. En mission, le drone prend cinq images
par seconde pendant la tenue, et une seconde image rattrape presque toujours la première.

Cent poses ont demandé 48 minutes de calcul, la simulation tournant à la moitié du temps réel
avec un seul drone.

## Trois drones en même temps

C'était le plus gros risque du projet, jamais testé : trois pilotes automatiques lancés
ensemble, trois liens, trois décollages, et un seul contrôleur par drone dans une seule boucle.

| manche | drone 0 | drone 1 | drone 2 |
|---|---|---|---|
| vers trois allées différentes | atteint, 2 cm, lu | atteint, 10 cm, lu | atteint, 6 cm, lu |
| seconde pose, même allée | atteint, 4 cm, lu | atteint, 11 cm, lu | atteint, 1 cm, lu |

**Six cibles sur six atteintes, six QR sur six lus.** Aucune interférence entre les trois liens.
La simulation tourne alors à 0,23 fois le temps réel : une mission de dix minutes coûte environ
quarante-cinq minutes de calcul.

Un point à retenir pour la suite : dans la première manche, deux drones ont pris le même
couloir en sens opposés et se sont croisés à 0,96 mètre. Ils volaient à des altitudes
différentes, séparées de 70 centimètres, et c'est ce qui les a protégés. Le contrôleur ne sait
pas éviter un autre drone ; c'est la coordination de l'étape 5 qui devra empêcher deux drones
de partager un couloir.

## Ce que le contrôleur sait faire

Il reçoit une pose et des points de passage, et il avance par phases : transit à vitesse
constante de point en point, approche à vitesse proportionnelle, tenue de la pose pendant une
demi-seconde, puis « atteint ». Il abandonne avec une raison — délai dépassé, ou aucun progrès
pendant huit secondes, ce qui est le cas d'un drone qui racle un rack. Sur les cent poses, il
n'a jamais eu à abandonner ; les abandons sont vérifiés sur le faux drone des tests.

Il ne bloque jamais : à chaque tour de boucle, chaque drone calcule et envoie sa commande, puis
le monde avance d'un pas. C'est ce qui permet à trois drones de voler ensemble.

## Ce qu'il ne fait pas encore

Il ne calcule pas son chemin. Pour ces vols, le chemin vient d'une règle simple sur le plan
connu de l'entrepôt : même allée, ligne droite ; sinon, le couloir au bout des racks. Dans le
vrai système, l'entrepôt est inconnu et le chemin viendra de la carte de l'étape 4. Et il
n'évite pas les autres drones.

## Les chiffres à retenir

| ce qu'on a mesuré | valeur |
|---|---|
| Loi retenue | vitesse proportionnelle à la distance restante, plafond 1 m/s, sur la position vraie |
| Précision de tenue | 1 cm quand on lui laisse le temps ; 9 cm au moment où il se déclare arrivé |
| Temps de cycle | 14,6 s en médiane ; environ 2 s + 0,85 s par mètre |
| Arrivées | 100 sur 100, aucun abandon |
| Lecture après arrivée | 93 sur 100, conforme à l'étape 2 |
| Trois drones ensemble | 6 cibles sur 6, 6 lectures sur 6, 0,23 fois le temps réel |

**Images.** `freinage.png` — distance et vitesse pour les trois lois ; `cycles.png` — les
temps de cycle des cent poses ; `essaim.png` — les trajectoires des trois drones.

**Pour refaire.** `bash campagne.sh` relance les trois vols puis l'analyse ; `analyse.py` seul
refait les calculs sur les enregistrements ; `pytest tests/test_control.py` vérifie le
contrôleur sans simulateur.
