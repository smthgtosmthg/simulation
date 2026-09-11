# Étape 8 — Le guide vision-langage

## Ce qu'on voulait

Le système voit, se souvient, décide et vole. On ajoute le second modèle d'intelligence
artificielle : un modèle vision-langage qui regarde l'entrepôt et la carte, et donne un avis
de bon sens sur **où aller** et **par quel côté aborder**. Il conseille, il ne commande pas : la
géométrie garde la main sur la pose exacte, et son avis pèse dans la note des cibles avec un
poids λ qui peut être nul. La question qui décide si on le branche : sur des cartes réelles de
mission, a-t-il raison plus souvent que le hasard ?

## Comment c'est construit

**Le modèle.** SmolVLM-500M-Instruct, un modèle vision-langage de 500 millions de paramètres,
chargé avec la bibliothèque transformers déjà présente dans l'environnement. Il tient dans
1,8 Go de mémoire graphique, à côté du simulateur et de l'œil appris. Une version de 2,2
milliards de paramètres est jugée hors ligne pour comparaison.

**Ce qu'il voit.** Deux images : la caméra latérale du drone, et la carte vue de dessus, avec
les zones candidates entourées de rouge et numérotées. Les zones sont les groupes de cibles du
cerveau géométrique, sur une grille de trois mètres, les six plus utiles.

**Ce qu'on lui demande.** Deux questions courtes plutôt qu'une longue, parce qu'un petit modèle
suit mieux une consigne à la fois : « quelle zone ? », puis « par quel côté : nord, sud, est ou
ouest ? ». Une seule question longue lui faisait répondre « 2. » sans le côté.

**Comment son avis entre dans la décision.** Les cibles de la zone conseillée reçoivent
λ × 10 points, et λ × 5 de plus si leur côté d'abordage est celui conseillé. À λ = 0, le système
est purement géométrique ; c'est le repli garanti.

**Il ne bloque jamais la boucle.** Il travaille dans un fil d'arrière-plan ; l'avis sert à la
décision suivante. Une demande toutes les cinq secondes par drone.

**Le banc hors ligne.** Pendant les missions de l'étape 5, un instantané est pris toutes les
trente secondes : la carte annotée, l'image de chaque drone, et — connu après coup — la zone qui
contient le plus de panneaux non lus et le côté d'où ils se lisent. Chaque modèle répond aux
mêmes instantanés. On mesure son accord avec la bonne réponse, contre un choix au hasard.

## Le banc : ce que le guide vaut

165 cas, un par drone vivant et par instantané des trois missions finales, chacun avec une
bonne réponse connue après coup. Quatre références, avant tout modèle.

| référence | bonne zone | bon côté |
|---|---|---|
| un choix au hasard | 17 % | 25 % |
| la zone la plus utile selon la géométrie, ce que le cerveau choisit déjà | **42 %** | — |
| répondre toujours « ouest » | — | **96 %** |

La seconde ligne fixe la vraie barre : un guide n'apporte quelque chose que s'il fait mieux que
le cerveau seul. La troisième dit que la question du côté, telle qu'elle est posée dans cet
entrepôt, n'a pas de valeur : les racks sont tous dans le même sens et les panneaux restants
sont presque tous sur les faces ouest, donc un mot constant suffit.

| modèle | bonne zone | bon côté | temps par question | mémoire |
|---|---|---|---|---|
| SmolVLM-500M-Instruct | **7 %** | 42 % | 0,7 s | 1,7 Go |
| SmolVLM-Instruct, 2,2 milliards | non mesuré : le téléchargement de ses 4,5 Go s'est arrêté à 1,7 Go après trois heures | | | |

**Avec une description en phrases en plus des deux images**, ce que la carte sait de chaque zone
— codes aperçus non lus, surfaces jamais regardées, distance au drone, codes déjà lus — le
petit modèle passe à **13 %** sur la zone et tombe à 12 % sur le côté. Le texte l'aide un peu,
sans le sortir du hasard. Le banc avec description (`banc.py --description`,
`resultats_description.json`) est prêt pour les modèles plus gros.

**Conclusion, écrite telle quelle.** Le petit modèle fait pire que le hasard sur la zone, 7 %
contre 17 %, et six fois moins bien que la géométrie seule, 42 %. Sur le côté, 42 % contre 96 %
pour un mot constant. Lire une carte vue de dessus avec des cercles numérotés est hors de sa
portée. **Le guide n'est pas branché.** Le système final est géométrique ; le guide reste écrit,
testé et mesuré, avec son banc et ses 165 cas, prêt pour un autre modèle ou une autre question.

**Ce qu'il faudrait essayer, et qui n'a pas été fait.** Décrire chaque zone en texte plutôt que
par un cercle sur une image — « zone 3 : 12 cartons repérés, 4 non lus, à 6 mètres » — et ne
demander au modèle que le raisonnement, ce que les petits modèles font mieux que la lecture de
cartes. Et mesurer le modèle de 2,2 milliards une fois téléchargé. Aucun modèle ne sera branché
sans avoir dépassé 42 % sur ces cartes.

**Ce qui est en place.** `swarm_qr/guide.py` : chargement, deux questions courtes, lecture
tolérante de la réponse, fil d'arrière-plan, avis pesé par λ dans la note. `mission.py --guide
smolvlm --lam 1` le branche. `experiments/12_guide/banc.py` juge tout modèle sur les
instantanés de n'importe quelle mission. `resultats.json` porte les 165 cas et les réponses
brutes.

---

# Deuxième campagne (09-10 septembre) — Qwen2.5-VL, et pourquoi un modèle échoue ici

La première campagne concluait « SmolVLM-500M fait 7 % là où la géométrie fait 42 % ». Deux
reproches étaient légitimes : le modèle était minuscule, et personne n'avait vérifié que le banc
lui-même était honnête. Cette campagne répond aux deux, avec Qwen2.5-VL de 7 puis 3 milliards de
paramètres, et elle finit par expliquer l'échec au lieu de le constater.

## Un défaut du banc, trouvé avant de juger le modèle

Les zones étaient numérotées par utilité géométrique décroissante. Répondre « 1 » valait donc
exactement la géométrie, sans rien comprendre au problème. **Les numéros sont désormais tirés au
hasard à chaque cas.** Les 22 % du modèle de 7 milliards, mesurés avant cette correction, ne
veulent rien dire.

## Deux jeux de cas, et pourquoi il ne faut jamais mélanger leurs chiffres

| | jeu ancien | jeu complet |
|---|---|---|
| cas | 165 | 102 |
| instantanés | missions du 08-09 | vols du 09-09 |
| comptages par zone dans l'enregistrement | **absents** | présents |
| moment de mission, médiane | 197 s | 146 s |
| cas après 250 s | 44 % | 0 % |
| codes encore non lus dans l'entrepôt, médiane | 14 | 99 |
| codes non lus dans la meilleure zone, médiane | 8 | 34 |
| **la géométrie y fait** | **42 %** | **85–87 %** |

La même géométrie passe de 42 à 85 % : ce n'est pas un progrès, c'est l'examen qui change. En fin
de mission il reste une quinzaine de codes éparpillés et la meilleure zone n'en contient que huit,
donc toutes les méthodes s'effondrent. Le jeu complet ne couvre que la phase de lecture, où la
bonne zone se détache nettement.

## Tout ce qui a été mesuré, sur le jeu complet

| qui décide | bonne zone | temps par question |
|---|---|---|
| hasard | 17 % | — |
| les deux images seules | 19 % | 0,8 s |
| plan redessiné pour le modèle, sans texte | 25 % | 0,9 s |
| plan redessiné + les faits de la carte en phrases | 40 % | 1,0 s |
| caméra seule + faits enrichis, modèle non compressé | 58 % | 3,3 s |
| idem + une phrase de connaissance du métier | 55 % | 3,4 s |
| idem, zones énumérées dans un ordre mélangé | 55 % | 3,4 s |
| plan + faits + **deux exemples résolus dans la question** | **61 %** | 3,5 s |
| noter chaque zone de 0 à 10, puis prendre la meilleure | 16 % | 6,0 s |
| **notre géométrie** | **85–87 %** | instantané |
| **règle de trois lignes : le plus de codes repérés non lus, sinon le plus proche** | **94 %** | instantané |

Sur le jeu ancien, les mêmes variantes donnent 17, 25, 48, 36 et 16 %, pour une géométrie à 42 %.
Le classement des variantes est le même dans les deux jeux.

**Les deux exemples résolus valent 21 points**, mesurés sur 100 cas appariés : 27 cas gagnés,
6 perdus, probabilité 0,0003 que ce soit du hasard. Sur 30 cas seulement, la même expérience
donnait 0,45, c'est-à-dire rien : il fallait les 100.

## Trois causes mesurées, pas une opinion

**1. Il ne lit pas un plan schématique, et ce n'est pas la compression.** Six questions à réponse
calculée, sans rapport avec la mission : compter les repères, trouver le plus à gauche, à droite,
en haut, en bas, et le voisin le plus proche de l'un d'eux.

| version du modèle de 3 milliards | plan d'origine | plan redessiné | mémoire vidéo |
|---|---|---|---|
| 4 bits | 1/9 | 5/9 | 2 463 Mo |
| 4 bits, encodeur d'images laissé en 16 bits | 1/9 | 5/9 | 4 264 Mo |
| 8 bits | 1/9 | 5/9 | 4 184 Mo |
| aucune compression | 1/9 | **4/9** | 6 748 Mo |

Quatre versions, les mêmes erreurs aux mêmes questions, et la version non compressée est la plus
mauvaise. La compression ne coûte rien ici : AWQ, GPTQ ou GGUF ne changeraient rien. Le redessin
du plan, en revanche, fait passer le comptage de 1 à 5 sur 9 : les cercles fins de 20 pixels avec
un chiffre de 8 pixels n'étaient pas lisibles.

**2. Il connaît la bonne règle et se trompe sur les nombres.** Ses explications sont toujours la
même phrase : « la zone 5, parce qu'elle a le plus de codes non lus ». Quand il se trompe, la
phrase est identique mais le nombre cité n'existe pas dans la description. Ajouter une phrase de
connaissance du métier ne change rien (2 cas gagnés, 5 perdus) : on ne lui apprend pas ce qu'il
sait déjà.

**3. Il refuse le premier élément d'une liste.** Sur 102 cas, il n'a **jamais** répondu 1, alors
que 1 était juste 15 fois. En énumérant les zones dans un ordre mélangé, il répond 1 quatorze
fois et trouve 6 de ces 15 cas. C'est un biais de position, et il est prouvé. Mais le corriger ne
fait pas monter le score (55 contre 58 %) : le verrou reste la lecture des nombres.

## La règle de trois lignes, et ce qu'elle apprend au système

Elle ignore les surfaces et prend la zone qui contient le plus de codes déjà repérés mais non
lus, la plus proche en cas d'égalité. Elle atteint 94 % contre 85 % pour la somme des utilités.
La raison : un code repéré est une lecture presque certaine, une surface jamais regardée n'est
qu'un espoir, et la somme des utilités mélange les deux. Une cible de lecture vaut 30 points, un
groupe de surfaces plafonne aussi à 30 : les deux peuvent donc se retrouver à égalité, et c'est
alors la distance qui tranche.

**Cette comparaison porte sur le classement des zones, pas sur la décision en vol.** En vol, le
drone choisit une cible précise, pas une zone. Rien n'a été changé dans la formule de décision, et
le gain éventuel en vol reste à mesurer par un vol.

## Coût en mémoire, et ce qui pourrait voler

Le simulateur occupe 4,6 Go des 8 Go de la carte pendant un vol. Le modèle de 7 milliards en
4 bits demande 6,3 Go : impossible. Le modèle de 3 milliards en 4 bits demande 2,5 Go : il
tiendrait. En 16 bits sur la carte, il ne se charge pas du tout.

## Ce qu'on en fait

Le guide **n'est pas branché**, λ reste à 0, et la formule de décision est inchangée. Le meilleur
réglage du modèle plafonne à 61 % là où la géométrie fait 85 % et une règle de trois lignes 94 %.

Un constat de structure achève le dossier : sur les 612 zones proposées au modèle, **aucune n'est
une zone d'exploration**, et tous les cas exploitables tombent entre 96 et 208 secondes de
mission. Les zones de lecture passent toujours devant au classement, donc le guide n'est jamais
consulté au moment où il aurait un avantage, quand on ne sait pas encore où sont les codes.

La littérature de 2026 dit la même chose que nos mesures : le raisonnement spatial reste un
point faible reconnu des modèles vision-langage, ouverts comme fermés (OmniSpatial, 8 400
questions ; revue « Spatial intelligence in vision-language models »).

**Les pistes qui restent, dans l'ordre où je les tenterais.** Poser au modèle les questions que
la carte ne sait pas poser, une étagère vide, une allée bouchée, une pancarte à lire, avec une
vérité tirée du simulateur. Mettre les deux exemples résolus à l'intérieur du modèle par un
prompt appris ou par LoRA, ce qui rendrait les 21 points permanents et redonnerait la vitesse.
Et mesurer le plafond avec un modèle de pointe en nuage, pour pouvoir écrire que même le
meilleur modèle disponible ne dépasse pas la géométrie.

**Pour refaire.** `variantes.py --missions ... --modele ... --variantes ... --quant 4bit|aucune
--device cuda|auto` pour toutes les variantes de choix de zone ; `perception.py` pour le profil de
perception et le coût de la compression. Résultats dans `resultats_variantes.json`,
`resultats_variantes_faits.json`, `resultats_fewshot100.json`, `resultats_texte_seul.json`,
`resultats_melange.json`, `resultats_perception.json`.

## Le dossier complet de la carte — l'idée qui change le résultat

Reproche légitime aux mesures ci-dessus : la description ne contenait que quatre chiffres par
zone. Cette série donne au modèle **tout ce que l'enregistrement contient**, sans une once de
vérité du simulateur : l'instant de la mission, les codes déjà lus, la position en trois
dimensions et le cap du drone, la position, le cap et la distance des deux coéquipiers, l'emprise
des cinq racks, et pour chaque zone son centre, son rayon, ses quatre comptages, le nombre et les
genres de ses cibles, le côté de ses faces, sa distance et sa direction depuis le drone, et sa
distance au coéquipier le plus proche. Environ 4 100 caractères par cas. La photo de caméra est
jointe ; **l'image du plan ne l'est plus**.

| ce que reçoit le modèle | bonne zone |
|---|---|
| image du plan + description de quatre chiffres par zone | 40 % |
| caméra + description enrichie | 58 % |
| caméra + dossier complet en phrases | 76 % |
| caméra + dossier complet en JSON | 78 % |
| caméra + dossier complet en JSON + **la note de notre cerveau** | 78 % |
| **dossier complet en JSON, sans aucune image** | **79 %** |
| notre géométrie | 85 % |
| règle de trois lignes | 94 % |

**Quatre conclusions, chacune appuyée sur une comparaison appariée.**

Le contenu fait tout : passer de quatre chiffres par zone au dossier complet vaut **+38 points**.
La prédiction inverse — « trop d'information noiera le modèle » — était fausse.

Le format ne compte pas : JSON contre phrases, +10 cas contre −8, probabilité 0,81 que ce soit du
hasard. On garde le JSON parce qu'il se génère automatiquement.

**L'image de caméra n'apporte rien.** Sans elle, 79 % contre 78 % avec (+6 / −5, probabilité 1,00).
Le modèle vision-langage travaille ici comme un modèle de texte ; sa partie vision est inutile.

**Il ne sait pas prendre le maximum d'une liste.** En lui donnant la note de notre cerveau, la
réponse était le plus grand de six nombres, ce qui vaut 85 % par construction. Il reste à 78 %, et
il ne suit la géométrie que dans 73 cas sur 102. Ses erreurs le disent : « la zone 4, parce
qu'elle a le plus de groupes de frontière inexplorés » alors que ce nombre est nul partout. Il
connaît la règle, il a les chiffres, il se trompe en les comparant.

Le biais de position s'atténue mais ne disparaît pas : 5 réponses « 1 » contre 15 attendues.

**Ce que cela change pour le projet.** Le plafond du modèle passe de 61 à 79 %, à neuf points de la
géométrie et quinze de la règle de trois lignes. Il reste donc au-dessous, et la conclusion tient :
le guide n'est pas branché. Mais la mesure a une valeur en soi pour le mémoire — un modèle de
3 milliards de paramètres, nourri du dossier complet de la carte, retrouve 79 % des choix de notre
planificateur géométrique sans connaître sa formule.

## Les listes brutes de la carte — un vol enregistré pour cela (10 septembre, 02 h à 03 h)

Pour donner au modèle **tout** ce que la carte sait, un vol sur 9033 a enregistré à chaque
instantané la grille compressée, les 149 faces lues avec leur position, les pistes, les 512
cibles candidates du planificateur, les frontières, les réservations, et la cible en cours de
chaque drone (`experiments/11_mission/dossier_complet`, 113/114, 0 chute). 36 cas exploitables,
géométrie à 92 %.

**La carte graphique impose une limite.** Elle est de génération Turing : sans noyaux d'attention
rapides, la mémoire croît avec le carré du texte, et 5 400 jetons débordent déjà les 8 Go. Les
listes sont donc écrites en lignes compactes (positions à une décimale, sans les noms de codes),
et le niveau « tout » garde les 25 cibles les plus utiles sur 512, en le disant au modèle.

| ce que reçoit le modèle | jetons | bonne zone |
|---|---|---|
| dossier des zones (celui de la nuit) | 1 550 | 83 % |
| + codes lus, pistes, cibles des drones, frontières, réservations, résumé de la grille | 4 000 | 78 % |
| la même chose **sans l'image de caméra** | 3 400 | **53 %** |
| + les 25 meilleures cibles du planificateur | 4 500 | 72 % |
| notre géométrie | — | 92 % |

Comparaisons appariées : les listes contre le dossier des zones, +3 / −5 (p = 0,73) ; les cibles en
plus, +1 / −3 (p = 0,62). **Ajouter l'information brute n'apporte rien** ; la tendance est même
légèrement négative, et ses explications commencent à citer « le plus de frontières inexplorées »,
un nombre nul partout. En revanche, **retirer l'image quand le texte est long coûte 25 points**
(0 gagné, 9 perdus, p = 0,004) — alors qu'elle ne comptait pas avec un texte court (79 contre
78 %). Sur 36 cas, ces chiffres restent fragiles, mais le sens est net : le modèle plafonne avec
les comptages par zone, et le surplus de détail le désoriente plus qu'il ne l'aide.

## Matin du 10 septembre — l'introspection, puis les leviers qui en découlent (102 cas, géométrie 85 %)

**L'idée de l'utilisatrice : après sa réponse, demander au modèle quels champs il a utilisés et
quels nombres il a lus.** Sur 15 cas, quand il a raison il cite le bon champ et compare juste.
Quand il se trompe, deux fautes reviennent : il choisit une zone à **zéro** code repéré en
justifiant par « le total des cibles candidates est plus grand, 33 contre 27 » ou par « les
groupes de frontière inexplorés », nuls partout — un gros nombre sans valeur éclipse le petit
nombre décisif ; et il se contredit parfois, la réponse sortant avant le raisonnement.

| levier, toujours le dossier des zones en JSON avec la photo | bonne zone |
|---|---|
| référence, 4 bits | 75 % |
| extraire d'abord les chiffres dans un tableau, puis choisir | 57 % |
| question posée avant les données | 55 % (10 cas sans réponse) |
| champ décisif placé en tête de chaque zone | 76 % |
| **quatre champs distracteurs retirés** (total de cibles, genres, rayon, frontières) | **83 %** |
| vote à cinq | en cours |
| géométrie | 85 % |

**« Extraire puis choisir » est la mesure la plus instructive de toutes.** Le modèle recopie les six
nombres sans une erreur — « zone 1 : 15, zone 2 : 5, zone 3 : 12, zone 4 : 7, zone 5 : 6,
zone 6 : 6 » — puis répond zone 6. La faute n'est pas la lecture des nombres, contrairement à ce
que la nuit précédente laissait croire : c'est l'étape de comparaison, désigner le plus grand de six
nombres écrits côte à côte. Et le levier coûte 18 points (+3 / −21, p < 0,001).

**Retirer les distracteurs vaut +9 points, et c'est prouvé** : +13 / −4 cas contre la référence,
p = 0,049. L'ordre seul ne fait rien (+9 / −7). Le modèle épuré fait alors jeu égal avec la
géométrie : cas par cas, il en gagne 14 et en perd 16. Il ne fait pas mieux, il fait pareil, en
trois secondes là où la géométrie est instantanée.

Sur le même dossier et les mêmes 102 cas, 4 bits contre non compressé : 74,5 % contre 78,4 %,
+6 / −10, p = 0,45 — pas de différence prouvée, un écart de trois points reste possible.

## Trois derniers leviers demandés (10 septembre, 102 cas, 4 bits, référence épuré 83 %)

| levier | bonne zone | cas par cas contre l'épuré |
|---|---|---|
| la règle donnée indirectement, deux phrases de métier avant les données | 71 % | +3 / −16, p < 0,01 |
| des noms de champs porteurs de sens (« lectures presque certaines », « surfaces non vérifiées ») | 66 % | +2 / −20, p < 0,01 |
| les deux ensemble | 53 % | +3 / −34, p < 0,01 |

Tout ce qu'on **ajoute** au dossier épuré le fait baisser, et de façon prouvée. Les noms parlants
ont créé de nouveaux distracteurs : il choisit « la zone qui a le plus de surfaces non vérifiées,
23 », c'est-à-dire de nouveau le plus gros nombre au nom le plus attirant. Les phrases de métier
n'apprennent rien à un modèle qui connaît déjà la règle, et détournent son attention. La leçon
tient en une phrase : ce modèle décide sur le plus grand nombre qu'il voit, et moins il voit de
nombres, mieux il choisit.

## Le test 2 : un petit entraînement QLoRA — le modèle dépasse la géométrie, avec une réserve

Méthode : le modèle en 4 bits reste gelé ; de petites couches LoRA (rang 8, sur les projections
d'attention du modèle de langage seulement, 3,7 millions de paramètres soit 0,1 % du total)
apprennent à répondre « ANSWER: n » à partir du dossier épuré, sans image. Entraînement sur les
vols 1 et 3 du 9 septembre : 66 cas, chacun présenté six fois avec des numéros de zones mélangés
différemment, soit 396 exemples, deux époques, perte de 0,032 à 0,004. Test sur 72 cas jamais vus,
tirés du vol de contrôle et du vol de la nuit. L'adaptateur pèse 15 Mo et se branche ou se
débranche sans toucher au modèle.

| sur les 72 cas de test | bonne zone |
|---|---|
| le plus proche | 25 % |
| notre géométrie | 83 % |
| modèle avant entraînement, sans image | 83 % |
| **modèle après entraînement, sans image** | **96 %** |
| règle de trois lignes | 100 % |

Cas par cas : l'entraînement fait gagner 11 cas et en perdre 2 (p = 0,022) ; contre la géométrie,
+12 / −3 (p = 0,035). **C'est la première variante qui dépasse le planificateur géométrique**, et
ses trois erreurs restantes portent toutes sur la même zone vraie.

**La réserve, et elle est sérieuse.** Entraînement et test viennent du même entrepôt, 9033. Le
modèle a pu apprendre la disposition de cet entrepôt autant que la règle. Le chiffre n'a de valeur
scientifique qu'après un test sur un entrepôt jamais vu, ce qui demande des vols sur d'autres
graines avec l'enregistrement complet. Et la règle de trois lignes, instantanée, fait 100 % sur ces
mêmes cas : ce que le modèle a appris, c'est très probablement cette règle.

**Pour refaire.** `entraine.py --entrainement <vols> --test <vols> --modele <chemin>
[--permutations 6 --epoques 2 --lr 1e-4]` ; résultats dans `resultats_entrainement.json`,
adaptateur dans `adaptateur_lora/`.
