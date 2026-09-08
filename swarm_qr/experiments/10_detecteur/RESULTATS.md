# Étape 7 — L'œil appris : un réseau qui repère les QR et les cartons

## Ce qu'on voulait

Le décodeur classique lit un code seulement de près, en face et lentement : l'enveloppe mesurée
à l'étape 2 s'arrête à 4 mètres. Cette étape ajoute un petit réseau qui ne lit pas mais
**repère** : un QR de loin devient une piste sur la carte, un carton repéré remplit le canal
sémantique laissé vide à l'étape 4. La question qui décide de tout : repère-t-il nettement plus
loin que le repérage classique, sans inventer d'objets ?

## Comment c'est construit

**Les exemples se fabriquent tout seuls.** En simulation, on connaît la position de chaque
carton et de chaque panneau. Pour chaque image, on projette ces objets dans l'image avec la pose
exacte de la caméra, et le cadre se calcule sans un seul clic. Une projection ne suffit pas :
un carton derrière un autre serait annoté alors qu'on ne le voit pas. Chaque objet est donc
échantillonné en surface (25 points par panneau, 25 par face de carton) et **un rayon du moteur
physique vérifie, pour chaque point, que rien ne s'interpose**. Le cadre entoure les points
visibles, pas l'objet entier ; un objet visible à moins de 30 % (QR) ou 20 % (carton) est
« ignoré » : ni exemple, ni fausse alerte au jugement.

**Huit entrepôts, jamais les mêmes.** Six entrepôts d'entraînement (graines 0 à 5), deux
entrepôts scellés pour le test (9033 et 9019), que le réseau ne voit jamais pendant
l'apprentissage. Dans chacun, une caméra libre prend 500 (ou 400) images à des poses tirées au
hasard dans l'espace de vol : 70 % regardent le rack le plus proche, le reste regarde n'importe
où — allées, murs, bouts de rack — pour apprendre aussi ce qui n'est pas un QR.

**Les 3 577 images de l'étape 2 servent de juge.** Elles sont annotées de la même façon, à
partir de leurs poses relues, et gardées à part : ce sont les images sur lesquelles le repérage
classique a été mesuré. Le réseau est donc comparé à lui **sur exactement les mêmes photos**,
avec le panneau visé à distance connue, et sur les 300 images sans aucun QR où le classique
inventait 27 % de motifs.

**Deux classes.** Le QR (le code seul, sans la marge blanche, comme les décodeurs le rendent) et
le carton. Les zones — allée, rack, mur — ne sont pas des objets qu'un cadre délimite ; elles
viennent de la géométrie de la carte, le repli prévu au plan.

**Le plus petit réseau qui suffit.** YOLO11 nano, 2,6 millions de paramètres, entraîné en
demi-précision sur la carte de 8 Go à deux résolutions d'entrée (1024 et 640 pixels). Un modèle
plus gros n'est entraîné que si le nano ne tient pas la barre.

| le jeu | images | QR annotés | cartons annotés | ignorés |
|---|---|---|---|---|
| six entrepôts d'entraînement | 3 000 | 23 016 | 26 304 | 13 325 |
| deux entrepôts scellés | 800 | 5 917 | 6 495 | 2 580 |
| étape 2, ré-annotées (test) | 3 577 | 19 615 | 24 625 | 14 949 |

Les QR annotés vont jusqu'au fond de l'entrepôt : 9 % à moins de 4 mètres, 36 % au-delà de
12 mètres. Tout ce qui est visible est annoté, jusqu'à 8 pixels de côté ; un QR visible mais
non annoté serait appris comme « rien ».

## Les contrôles, avant d'entraîner

Un cadre décalé n'arrête pas l'entraînement : il le fausse en silence. Trois contrôles.

| contrôle | résultat |
|---|---|
| Calibration relue dans le simulateur | fx = 886,8, la valeur calculée |
| Un panneau dégagé vu de face à 2 m : cadre au centre, côté attendu | 149 px pour 149 attendus, sur les 8 entrepôts |
| Le centre du panneau visé, projeté par la fonction validée à l'étape 2, est dans notre cadre | **99,4 %** (2 000 images), **100 %** (9019, vol, traversée) |
| Planches regardées à l'œil | 24 images, cadres sur les codes et les cartons, gris sur les objets cachés |

## Trois faits découverts en fabriquant les cadres

**Les cartons cachés gardaient leur collider.** Les cartons non retenus par la graine étaient
seulement rendus invisibles ; neuf sur neuf arrêtaient encore les rayons. Le lidar de l'étape 4
voyait donc des cartons que la caméra ne voyait pas — des obstacles fantômes à l'intérieur des
racks. Corrigé à la source (`scene._hide` désactive le prim : ni rendu, ni physique). Les
chiffres de l'étape 4 ont été mesurés avant cette correction.

**Le collider d'un carton est plus petit que sa forme visible.** Un rayon oblique vers un QR
touche son propre carton, mais 10 centimètres plus loin que prévu. Une tolérance de distance
déclarait cachés des panneaux parfaitement visibles (BOX_109 à 2,4 m : 5 points visibles sur
25). La visibilité se juge maintenant par **l'identité du premier objet touché** : si c'est le
carton propriétaire, rien ne s'interpose. Même panneau après : 23 sur 25.

**Un contrôle qui ne choisit pas son témoin ne prouve rien.** Le premier contrôle d'aplomb
prenait le panneau le plus proche de la hauteur de vol ; dans l'entrepôt 5, ce panneau était à
moitié caché et le banc s'est arrêté à tort. Le témoin est maintenant cherché parmi quarante
candidats : caméra hors rack, panneau visible à 95 %.

## Le banc : ce que le réseau vaut

Deux variantes du même réseau nano, jugées au **seuil de mission** — le seuil le plus bas qui
garde les fausses alertes sous 1 % des images sans QR, soit 0,5 pour les deux.

| ce qu'on a mesuré | classique (étape 2) | appris, 1024 px (retenu) | appris, 640 px |
|---|---|---|---|
| Panneau visé repéré, mêmes 2 000 images, jusqu'à 6 m | 100 % jusqu'à 4 m, 75 % à 5–6,5 m | **100 %** | 100 % |
| Panneau visé repéré à 6–8 m | 47 % | **98 %** | 98 % |
| Portée tenue à 90 % (distance réelle) | 4,0 m | **8,0 m**, la limite des images | 8,0 m |
| Images sans aucun QR où un QR est inventé (300 images) | 27,3 % | **0,7 %** | 0,3 % |
| QR visibles trouvés, entrepôts scellés (5 917) | — | **98,3 %** | 94,4 % |
| dont à 8–12 m (4 195) | — | **98,3 %** | 93,4 % |
| Cartons visibles trouvés (6 495) | — | **94,1 %** | 91,4 % |
| Faux QR par image, entrepôts scellés | — | 0,02 | 0,02 |
| Temps par image, une par une, en demi-précision | — | 12 ms | 12 ms |
| Mémoire sur la carte graphique | — | 63 Mo | 49 Mo |
| Paramètres | — | 2,6 M | 2,6 M |
| Entraînement | — | 48 min | 21 min |

**La portée est le chiffre qui justifie le composant.** Sur les images où le repérage
classique perd un panneau sur deux, le réseau en garde 98 sur 100, et sur les entrepôts scellés
il garde 98 % des QR jusqu'à 12 mètres, la limite de ce qui a été annoté. En mission, un drone
qui longe une allée voit donc les codes de toute l'allée, pas seulement ceux à 4 mètres.

**Les fausses alertes sont le chiffre qui rend le composant utilisable.** Chaque objet inventé
coûte un trajet perdu. Le classique inventait un motif sur 27 images sans QR ; le réseau, moins
d'une sur 100. Sur les entrepôts scellés, deux faux QR pour cent images, et la carte en refuse
encore une partie parce qu'un faux cadre tombe souvent hors de l'enveloppe ou hors de l'entrepôt.

**Le choix de la variante.** Les deux tiennent la même portée et le même temps par image : à une
image par observation, le coût est dominé par le transfert, pas par le calcul. La 1024 trouve
quatre points de QR de plus et cinq points de cartons de plus ; elle est retenue. Un modèle plus
gros n'a pas été entraîné : le nano tient la barre.

**Ce que le seuil change.** À 0,25, la 1024 trouve 99,4 % des QR mais invente un QR sur 2,3 % des
images vides ; à 0,5, 98,3 % et 0,7 %. Le seuil de mission prend le second : un point de rappel
contre trois fois moins de fantômes.

`portee.png` — les deux courbes : le panneau visé sur les images de l'étape 2, et tous les objets
visibles des entrepôts scellés, par distance.

## En vol : l'œil appris dans la patrouille

Même patrouille qu'à l'étape 4, réduite à une étagère et deux allées, avec le réseau branché
(`--detecteur auto`) : sur chaque image des deux caméras latérales, un QR repéré devient une
piste, un carton repéré marque le canal sémantique. Le jugement est celui de l'étape 4.

**Premier vol : deux découvertes.** Les pistes étaient toutes dans les racks — une seule sur 152
hors de toute emprise — mais placées grossièrement : 43 % à moins de 50 cm d'un vrai panneau,
71 % à moins d'un mètre, 99 % à moins de deux. La cause n'est pas le réseau, c'est la distance :
à 8 mètres, deux rayons du lidar sont espacés de 28 cm, et le rayon le plus proche de la
direction du cadre tombe sur le carton d'à côté ou, par un trou, sur le fond du rack. **Les
pistes lointaines sont maintenant placées par la carte elle-même**, qui accumule les tours de
lidar : premier cube occupé le long de la direction du cadre.

La seconde découverte a d'abord été mal lue. **7 points de trajectoire « dans un rack »**, contre
zéro à l'étape 4 : le drone avait traversé le bout du rack du milieu à 3,17 m. Premier diagnostic :
le lidar, avec un anneau tous les 4 degrés, ne touche le dessus d'une planche 24 cm sous lui
qu'à 1,4 m devant, et le planificateur ne regardait que ±0,6 m. J'ai donc mis un anneau tous les
2 degrés (3 420 rayons) et une tranche de ±0,85 m, le rayon du drone plus son oscillation
verticale, puis refait le vol : encore 4 points. **Ce diagnostic était faux.** Des rayons
physiques envoyés dans le simulateur montrent que les racks de cet entrepôt n'ont aucune
structure solide aux deux bouts de leur emprise : ni montant, ni traverse, ni planche sur
0,70 m au sud et 0,77 m au nord, seulement un panneau de signalisation en haut et un
pare-chocs bas. Le drone a traversé un rectangle du plan, pas un rack. La carte, elle, avait
raison : cet endroit est libre. L'arbitre juge maintenant la structure solide, mesurée, et non
le rectangle. Les deux changements de capteur et de planificateur sont gardés : ils sont
physiquement fondés et les tests passent, mais ils n'ont corrigé aucun défaut mesuré, et le
lidar coûte 44 ms par observation au lieu de 24.

**Second vol, arbitre corrigé.** Même patrouille, une étagère, deux allées, quatre allers.

| ce qu'on a mesuré | résultat |
|---|---|
| Allers atteints | 4 sur 4, 3 chemins recalculés en route |
| Points de trajectoire dans une structure de rack | **0 sur 819**, au plus près 0,55 m |
| Codes lus, aucun inventé, position | 45 sur 114 (une étagère sur trois, deux allées sur quatre), 1,0 cm en médiane |
| Pistes créées par l'œil appris | 143 |
| Pistes à moins de 1 m / 2 m d'un vrai panneau | **77 %** / **99 %** |
| Pistes hors de toute emprise de rack (les fantômes) | **1 sur 143** |
| Pour comparaison, le repérage classique de l'étape 4 | 47 % / 86 %, 6 fantômes sur 182 |
| Cubes marqués « carton », à moins de 60 cm / 1 m / 2 m d'un vrai carton | 826 ; 68 % / 81 % / 98 % |
| Coût par observation, deux caméras | détecteur 33 ms, lidar 44 ms, carte 4 ms |

**Ce que le vol dit.** Le réseau ne fait pas courir les drones après des objets imaginaires : une
piste sur 143 est hors d'un rack, et 99 % sont à moins de deux mètres d'un vrai panneau. Ce qui
reste grossier, c'est la position d'un objet vu de loin : la direction vient du cadre, la
distance vient du lidar de près et de la carte de loin, à un cube près. Pour la décision de
l'étape 5, une piste à un mètre du panneau suffit : elle dit où aller, et la lecture de près
donne ensuite la position au centimètre. Le premier vol (`vol_avant_correction/`) est gardé
comme pièce à conviction.

## Ce que la carte apprend de plus

Le canal sémantique, vide depuis l'étape 4, porte maintenant les cartons repérés. Les zones —
allée, rack, mur — ne sont pas des classes du détecteur : elles se déduisent de la géométrie de
la carte, et c'est le guide de l'étape 8 qui en aura besoin.

## Les tests sans simulateur

Quarante-cinq tests en dix secondes : les quarante-deux des étapes 3 et 4, plus le canal
sémantique, la géométrie d'une détection, et le premier obstacle le long d'un rayon de la carte.

## Les chiffres à retenir

| ce qu'on a mesuré | valeur |
|---|---|
| Panneau visé repéré à 6–8 m, mêmes images que l'étape 2 | **98 %** contre 47 % |
| QR visibles trouvés, entrepôts jamais vus, jusqu'à 12 m | **98,3 %** |
| Images sans QR où un QR est inventé | **0,7 %** contre 27 % |
| Cartons visibles trouvés | 94 % |
| Temps par image, mémoire | 12 ms, 63 Mo |
| Cadres appris, contrôlés contre l'étape 2 | 99,4 à 100 % |
| En vol : fantômes parmi les pistes | 1 sur 143 |
| En vol : points de trajectoire dans une structure de rack | 0 sur 819 |

**Fichiers.** `rendu.py` fabrique images et cadres ; `controle.py` les vérifie ; `entraine.py`
entraîne ; `banc.py` juge et installe les poids retenus dans `swarm_qr/assets/detecteur/` ;
`campagne.sh` enchaîne les phases (rendu, controle, entraine, banc, vol). `jeu/` contient les
3 800 images et tous les cadres ; `runs/` les entraînements ; `controle/` les planches ;
`portee.png` les courbes ; `vol/` la carte, la trajectoire et le jugement du second vol.
