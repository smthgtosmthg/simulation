# Étape 4 — La carte partagée : la mémoire commune des drones

## Ce qu'on voulait

Jusqu'ici, le drone lisait un code et l'oubliait aussitôt. Il ne savait pas où il était déjà
passé, où étaient les racks, ni ce que faisaient ses coéquipiers. Cette étape lui donne une
mémoire, la partage entre les trois drones, et lui fait calculer ses propres chemins dans un
entrepôt qu'il découvre — sans jamais consulter le plan.

## Comment c'est construit

**Deux structures, pas une.** La grille retient l'espace : l'entrepôt est découpé en cubes de
25 centimètres, et chaque cube retient s'il est occupé, libre ou inconnu, et de quel côté il a
été regardé d'assez près pour qu'un code y soit lisible. Une table retient les panneaux : un
code lu a une identité et une position au centimètre, qu'un cube de 25 centimètres écraserait.

**Un lidar sur le drone.** Il envoie 1 800 rayons à chaque image. Le long de chaque rayon,
l'espace traversé devient libre et le point touché devient occupé. On additionne des preuves
au lieu d'écraser : une case vue libre dix fois puis occupée une fois reste libre.

**La couverture, définie par la lecture.** Une case n'est pas « vue » mais « un code posé là,
face à la caméra, aurait été lisible » : la distance apparente doit tomber dans l'enveloppe
mesurée à l'étape 2, entre 1,5 et 4 mètres, sans obstacle devant, et depuis le bon côté.

**Le chemin vient de la carte.** Elle seule connaît les obstacles ; elle calcule des points de
passage qui les contournent, élargis du rayon du drone. L'inconnu est traversable mais coûte
trois fois plus que le libre connu : sans cela un drone ne sortirait jamais de sa zone
explorée, et sans le coût il couperait tout droit à travers un rack jamais vu. Un chemin déjà
prévu est revérifié toutes les 0,4 seconde en vol et recalculé si un obstacle nouveau le coupe.

**L'équipe.** Chaque drone réserve sa cible ; une réservation non renouvelée expire, et un
drone muet perd les siennes. La panne d'un drone est ainsi gérée sans une ligne de code qui la
surveille. Les cibles abandonnées vont en liste noire, et les frontières — là où le connu
touche l'inconnu — sont prêtes pour la décision de l'étape 5.

## Les contrôles, avant toute mesure

Le banc vérifie sa propre chaîne et s'arrête si un contrôle échoue.

| contrôle | résultat |
|---|---|
| Chaque rayon du lidar refait par le moteur physique, à trois caps | **0,0 cm** d'écart médian, 0 à 3 % de rayons faux |
| Le monde bouge-t-il quand le drone tourne ? | non : 0,0 cm entre les nuages à 0 et 90 degrés |
| Obstacles inventés dans les allées | aucun : 100 % des points à hauteur de vol sur un rack ou un mur |
| Coût d'une observation | 18 ms de lidar, 2 ms de couverture |
| Mémoire | 1,5 Mo pour 258 048 cases |

Ces contrôles ont servi. Le lidar compte ses angles verticaux vers le bas, et il ne se met à
jour qu'au rendu de l'image : sans le contrôle contre la physique, la carte se serait remplie
tête-bêche avec des chiffres d'apparence normale.

## La patrouille : ce que la carte vaut

Le drone longe les cinq faces de rack accessibles, aux trois hauteurs d'étagère, à 2,2 mètres
— 0,96 mètre dans l'allée étroite du mur ouest — en photographiant cinq fois par seconde avec
ses deux caméras latérales. Quinze allers, chaque transit calculé sur la carte découverte.

| ce qu'on a mesuré | résultat |
|---|---|
| Entrepôt connu | **89,5 %** des cases |
| Codes lus | **110 sur 114**, sur 171 faces, toutes confirmées par au moins deux lectures |
| Codes inventés | **aucun** |
| Position d'une face lue | **0,9 cm** en médiane, 1,2 cm dans neuf cas sur dix, 8,3 cm au pire |
| Fausses cases occupées en pleine allée | 0,39 % |
| Cartons dont l'obstacle est posé sur la carte | 82 à 90 % selon l'étagère |
| Promesse de lisibilité tenue | **95 %** des panneaux annoncés lisibles ont été lus, contre 37 % des autres |
| Transits et traversées | 15 sur 15 atteints, aucun abandon, 3 chemins recalculés en route |
| Sécurité du vol sur la seule carte | **aucun** des 2 738 points de trajectoire dans l'emprise d'un rack |
| Chemins d'un coin à l'autre sur la carte finale | 12 sur 12, **aucun ne passe sur un obstacle connu** |
| Coût en vol | 22 ms de lidar + 2 ms de couverture + 19 ms de décodage par observation |

**La position à un centimètre** est le chiffre qui compte pour la suite : c'est la précision
avec laquelle l'inventaire situera chaque carton.

**La promesse de lisibilité** est la vérification de la définition de la couverture : quand
la carte dit « un code posé là aurait été lu », c'est vrai 95 fois sur 100. Les 37 % lus
sans être annoncés lisibles sont les lectures faites plus près que l'enveloppe — l'allée
étroite, et la seconde caméra qui lit la face d'en face à un mètre.

## Cinq faits découverts en comparant à la vérité

Aucun ne provoquait d'erreur ; tous ont été trouvés parce que le jugement compare la carte à
la vérité, chiffre par chiffre.

**Les étiquettes n'ont pas toutes la même taille.** Elles suivent la taille des cartons : 40
centimètres sur les grands, 21 sur certains, moins encore sur d'autres. Un code deux fois plus
petit que prévu paraît deux fois plus loin, et la carte le plaçait à 1,3 mètre de sa vraie
place. Un drone ne peut pas connaître la taille d'une étiquette sur une image. **La distance
vient donc du lidar**, le long de la direction que l'image donne ; l'image ne fournit plus que
l'identité et l'orientation. Toutes les 5 000 positions de la patrouille viennent du lidar.

**Chaque carton porte le même code sur ses deux faces.** Moyenner les deux plaçait le code au
milieu du carton. Une lecture rejoint la face connue la plus proche ; trop loin, c'est l'autre
face. Et un carton n'a que deux faces : une troisième lecture loin des deux connues est une
erreur de décodage — il y en a eu une sur cinq mille — et elle est refusée.

**Une case vue à travers un rack ne rend pas lisible la face opposée.** Les racks sont des
cadres ouverts ; le champ de la caméra passe entre les cartons. La couverture retient donc de
quel côté chaque case a été regardée.

**Le décodeur lisait le décor.** L'entrepôt porte des codes-barres imprimés sur ses montants ;
un lecteur tous formats les faisait entrer dans l'inventaire. Il est restreint aux QR.

**Les montants d'un rack sont fins.** À 8 mètres, deux rayons du lidar sont espacés de 28
centimètres et peuvent les manquer ; le drone, en transit rapide, a frôlé un montant d'angle
avant que le lidar ne le voie de près. D'où la revérification du chemin en vol : sur la
patrouille finale, trois recalculs en route, et plus un seul point de trajectoire dans
l'emprise d'un rack.

## Ce que les fantômes du détecteur deviennent

Le détecteur classique repère 182 motifs qu'il ne lit pas, dont 93 % ne correspondent à aucun
vrai panneau — les 27 % de faux repérages de l'étape 2, accumulés sur 2 800 images. La carte
en refuse déjà beaucoup, parce qu'une fausse détection donne une position hors de l'enveloppe
ou hors de l'entrepôt. Ceux qui restent sont le travail de l'étape 7.

## Ce que la carte ne fait pas

Elle ne décide pas où aller : la patrouille suit un parcours fixe. C'est l'étape 5.

Les quatre codes manquants sont les quatre cartons aux étiquettes de 12 centimètres, sur le
rack du milieu, dans des allées accessibles. Une étiquette trois fois plus petite se lit trois
fois plus près — vers 1,2 mètre si l'on applique la règle de trois à l'enveloppe mesurée — et
la patrouille passait à plus de 2,5 mètres. La carte ne peut pas connaître la taille d'une
étiquette avant de l'avoir lue ; c'est à la décision de l'étape 5 de revenir plus près d'un
carton vu de face sans qu'aucun code y soit lu. Une face de rack sur six est par ailleurs
inaccessible, coincée contre le mur est dans une allée de 75 centimètres ; les codes de ces
cartons ont été lus sur leur face opposée.

## Les tests sans simulateur

Vingt-huit tests vérifient la carte sur un monde de boîtes dont on connaît la vérité, en huit
secondes : l'exactitude de l'occupation, le refus d'inventer derrière un mur, l'enveloppe et le
côté de la couverture, les deux faces d'un carton, le refus des positions absurdes et d'une
troisième face, l'expiration des réservations, le drone muet, les frontières, et les chemins —
qui contournent un mur connu, préfèrent le libre connu à l'inconnu, se déclarent coupés quand
un mur apparaît, et laissent le drone sortir de la marge de sécurité d'un obstacle.

## Les chiffres à retenir

| ce qu'on a mesuré | valeur |
|---|---|
| Lidar contre le moteur physique | 0,0 cm |
| Position d'un code lu | **0,9 cm** en médiane, 8,3 cm au pire |
| Codes lus | 110 sur 114, aucun inventé |
| Promesse de lisibilité | tenue à 95 % |
| Fausses cases occupées | 0,39 % |
| Chemins sur un obstacle connu | aucun |
| Points de trajectoire dans un rack | aucun sur 2 738 |
| Coût d'une observation | 24 ms pour la carte, 19 ms pour le décodage |
| Mémoire | 1,5 Mo |

**Images.** `carte_3d.html` — la carte en trois dimensions, à tourner à la souris, avec le vol
à rejouer et la vérité à superposer ; `carte_3d.png` — la même en image fixe ;
`comparaison.png` — la vue de dessus avec la vérité en rouge ; `carte_qui_se_remplit.mp4` —
la carte aller par aller.

**Pour refaire.** `bash campagne.sh` enchaîne les tests, les contrôles, la patrouille et
l'analyse ; `analyse.py` seul refait le jugement sur la carte enregistrée, sans simulateur.

## Post-scriptum (étape 7, 2026-09-07)

Trois choses apprises après ces mesures. **Les cartons non retenus gardaient leur collider** :
le lidar de cette patrouille voyait des cartons que la caméra ne voyait pas, des obstacles
fantômes à l'intérieur des racks. Corrigé à la source (le prim est désactivé). **Les racks n'ont
aucune structure solide sur 0,70 m au sud et 0,77 m au nord de leur emprise**, mesuré par rayons
physiques : l'arbitre compte maintenant la structure, pas le rectangle du plan. Et le lidar a
maintenant un anneau tous les 2 degrés, le planificateur une tranche de ±0,85 m. Les chiffres
ci-dessus ont été mesurés avant ces changements ; la patrouille complète sera refaite avec la
décision de l'étape 5.
