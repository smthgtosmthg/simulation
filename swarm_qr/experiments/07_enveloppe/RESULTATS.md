# Étape 2 — Jusqu'où, sous quel angle et à quelle vitesse le drone lit-il un QR code ?

## Ce qu'on voulait savoir

Trois limites, qui serviront de fondations à tout le reste : la distance maximale de lecture,
l'angle maximal, et la vitesse maximale de passage. Le contrôleur devra les respecter, et c'est
avec elles qu'on décidera où envoyer chaque drone.

## Comment on a mesuré

3 481 images, chacune enregistrée avec la position exacte de la caméra, relue dans le
simulateur. Cinq campagnes : 2 000 poses au hasard avec une caméra libre, 300 poses dans un
entrepôt vidé de tous ses codes, 400 poses dans un second entrepôt, 12 positions tenues par le
vrai drone, et quatre traversées du rack à des vitesses différentes. Six lecteurs de QR ont
analysé exactement les mêmes images.

Chaque campagne commence par des contrôles automatiques — calibration de la caméra, mesure du
retard du rendu, lecture à distance connue — et s'arrête d'elle-même si un contrôle échoue.
L'analyse vérifie ensuite, image par image, que la distance vue colle à la position enregistrée
: l'écart médian est de 0,7 à 1,1 centimètre selon les campagnes. Les chiffres qui suivent
reposent donc sur des données dont la cohérence est prouvée, pas supposée.

## Le résultat principal : une seule limite au lieu de deux

Distance et angle ne font qu'une seule limite. Prends une feuille carrée et tourne-la
lentement : elle paraît de plus en plus étroite. C'est ce qui arrive au code vu de biais — ses
petits carrés occupent moins de pixels, exactement comme s'il était plus loin. On ramène donc
tout à la **distance apparente** : la distance qu'aurait le code s'il était vu bien en face.

| distance apparente | lectures réussies | images |
|---|---|---|
| 1,0 à 1,25 m | 75 % | 84 |
| 1,25 à 1,5 m | 89 % | 115 |
| **1,5 à 4 m** | **91 à 99 %** | 925 |
| 4 à 5 m | 74 % | 167 |
| 5 à 6,5 m | 31 % | 194 |
| plus loin | 3 % | 185 |

**La zone fiable va de 1,5 à 4 mètres de distance apparente.** Ces chiffres incluent une visée
imparfaite — le tirage décale volontairement le panneau dans l'image, comme un vrai drone le
ferait. Avec une visée bien centrée, la lecture marche encore plus près : le test 3 lit à 0,5
mètre quand le drone vise juste. La limite basse n'est donc pas une interdiction, c'est une
marge de sécurité pour une visée réaliste.

## La vitesse ne coûte rien

C'est la question qui restait ouverte, et elle est tranchée : le drone a longé le rack à
quatre vitesses en photographiant à la cadence de mission, cinq images par seconde. Sur chaque
image où le panneau était réellement dans le cadre :

| vitesse | images avec le panneau visible | lues |
|---|---|---|
| 0,1 m/s | 45 | **45 — 100 %** |
| 0,3 m/s | 18 | **18 — 100 %** |
| 0,6 m/s | 10 | **10 — 100 %** |
| 1,0 m/s | 5 | **5 — 100 %** |

Aucune lecture manquée, jusqu'à un mètre par seconde. Les échantillons rapides sont petits —
à 1 m/s le panneau ne reste visible que quelques images — mais aucun échec n'apparaît nulle
part. Le simulateur ne modélise pas le flou de bougé d'une vraie caméra : c'est une limite à
documenter, pas un oubli.

## Le vrai drone lit comme la caméra idéale

Les douze positions tenues confirment la carte : **100 % de lecture partout entre 1 et 4
mètres de distance apparente**, et la dégradation commence exactement où la caméra libre la
prévoit — 91 % entre 4 et 5 mètres. Le vol n'ajoute aucune difficulté.

Sa seule faiblesse est connue : à moins d'un mètre du panneau, le drone ne tient plus son
angle. En visant 90 centimètres de face, il s'est stabilisé à 74 centimètres et 29 degrés,
parce qu'un écart latéral ordinaire produit un grand angle quand on est si près. Demander une
position précise à courte distance n'a pas de sens ; il faut demander une zone.

## Le drone n'invente jamais un code

Dans l'entrepôt vidé de ses 228 panneaux, **aucun des six lecteurs n'a jamais prétendu lire un
code** — zéro sur 300 images, pour tous. Une lecture peut donc être crue sans vérification.

## Repérer n'est pas lire

**Lire**, c'est comprendre le contenu. **Repérer**, c'est voir un carré sombre qui ressemble à
un QR. Le repérage porte plus loin que la lecture, mais dans l'entrepôt sans aucun code, le
détecteur croit encore voir un panneau **une fois sur quatre** (27 %) — il confond cartons,
montants et ombres. Un repérage seul ne prouve rien ; c'est le problème que l'étape 7 devra
résoudre avec un détecteur appris.

## Les lecteurs, comparés sur les mêmes images

| lecteur | lectures réussies | portée apparente | erreur de distance |
|---|---|---|---|
| **zbar** | **91,2 %** | **4,50 m** | 0,4 cm |
| **zxing** | 90,2 % | 3,65 m | 0,8 cm |
| pyboof | 87,2 % | 4,50 m | 1,0 cm |
| opencv | 80,8 % | 1,65 m | 1,0 cm |
| opencv aruco ×3 | 79,8 % | — | 1,6 cm |
| opencv aruco *(l'ancien choix)* | 79,3 % | — | 1,8 cm |

zxing et zbar dominent nettement ; zbar lit un peu plus loin sur cette campagne. Les deux sont
excellents et rapides — **zxing reste le lecteur retenu** (17 ms par image), zbar est
l'alternative immédiate. L'ancien lecteur perdait onze points.

## La position d'un code lu est très précise

Erreur médiane de **0,8 centimètre**, et moins de 3 centimètres dans neuf cas sur dix. C'est le
chiffre dont la carte partagée de l'étape 4 a besoin. Il ne vaut que pour un code **lu** : la
position d'un simple motif repéré reste une supposition.

## Les limites de lecture se transportent

Refaite dans un second entrepôt — autre panneau, autre éclairage — la mesure donne **90,4 %
contre 90,2 %**. Ces chiffres sont une propriété de la caméra et du code, pas d'un coin
d'entrepôt. Et l'allée entre deux racks ne fait que 3,92 mètres : le drone ne peut pas reculer
à plus de 3,6 mètres d'un panneau. C'est donc l'angle, pas le recul, qui permet de couvrir
plusieurs cartons depuis une même position — et l'angle est presque gratuit.

## Vérification de la campagne précédente

Cette campagne a été refaite entièrement, code réécrit et vols relancés, pour vérifier la
première. Les chiffres se confirment presque à l'identique : 90,2 % contre 90,2 % en zone
utile, 3,65 m contre 3,65 m de portée, 90,4 % dans les deux cas sur le second entrepôt, zéro
lecture fantôme partout. Deux conclusions de l'ancienne campagne étaient fausses et sont
corrigées : la « distance minimale d'un mètre » venait d'une visée décalée de 10 centimètres,
et l'effet de la vitesse, jusqu'ici non mesurable, est maintenant établi.

## Les chiffres à retenir

| ce qu'on a mesuré | valeur |
|---|---|
| Zone fiable | distance apparente de **1,5 à 4 m** (≥ 90 % par image) |
| Distance apparente | distance réelle ÷ cosinus de l'angle |
| Vitesse | **aucun effet jusqu'à 1 m/s** |
| Recul maximal en allée | 3,6 m — limite physique, pas optique |
| Lecteur retenu | **zxing** (zbar en alternative) |
| Position d'un code lu | 0,8 cm en médiane |
| Codes inventés | aucun |
| Motifs repérés à tort | 27 % — un détecteur appris sera nécessaire |

**Image.** `enveloppe.png` — les trois courbes : distance, angle, et la courbe unifiée.

**Pour refaire.** `bash campagne.sh` relance tout ; `analyse.py` seul refait les calculs sur
les images enregistrées, sans simulateur.
