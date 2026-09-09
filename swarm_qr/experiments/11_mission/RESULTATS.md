# Étape 5 — Le cerveau et la première mission complète à trois drones

## Ce qu'on voulait

Le drone savait lire, se souvenir et repérer de loin. Il lui manquait de savoir **où aller**.
Cette étape transforme les composants en système : chaque drone choisit ses cibles sur sa
carte, jamais sur le plan de l'entrepôt ; trois drones se partagent le travail ; la mission
se termine seule. La porte de validation n'est pas une performance mais une mécanique : aucun
blocage, aucun doublon, une dispersion au départ, une panne absorbée, une fin propre.

## Comment c'est construit

**Trois sortes de cibles, toutes tirées de la carte.**
- **Lire** : une piste, un QR repéré de loin par l'œil appris et pas encore lu. La pose de
  lecture se place à 2 mètres devant, du côté où la carte montre de l'espace libre — pas du
  côté d'où le QR a été aperçu, souvent de biais : à 60 degrés, le lecteur ne lit pas.
- **Couvrir** : une surface connue occupée, à hauteur de vol, jamais regardée d'assez près
  depuis le côté libre. Un mètre de surface par cible ; un carton repéré par l'œil appris y
  vaut trois fois plus qu'un mur nu. C'est le premier usage du canal sémantique.
- **Explorer** : une frontière entre le connu et l'inconnu, groupée par deux mètres.

**Une note par cible.** L'utilité (30 pour une piste, un point par cube de surface, 0,15 par
case de frontière), moins un point par mètre de trajet, moins 25 si un coéquipier a réservé
l'endroit. Les six meilleures par la ligne droite reçoivent un vrai chemin sur la carte ; une
cible sans chemin est écartée. Une piste visitée deux fois sans lecture est mise en liste
noire : personne n'y retourne.

**Trois rythmes.** Le contrôleur à chaque pas de physique, l'observation cinq fois par
seconde, la décision quand un drone n'a plus de cible. Devant une cible de lecture, le drone
tient sa pose deux secondes, dix images, pour que le lecteur lise.

**L'équipe.** Une seule carte partagée. Chaque drone annonce sa cible ; les autres l'évitent
par la pénalité. Une réservation non renouvelée expire, un drone muet perd les siennes : la
panne est gérée sans une ligne de code qui la surveille. Et une règle de priorité pour les
croisements : à moins de 1,5 mètre d'un drone de plus petit numéro, on s'arrête et on le laisse
passer.

**La fin.** Quand plus personne n'a de cible pendant trente secondes, ou quand le budget de
temps est épuisé.

**Le cerveau est testé avant de voler.** Dix cartes jouets dessinées à la main, dont on connaît
la bonne réponse : une piste proche bat une frontière lointaine ; une zone réservée est évitée ;
entre deux frontières égales, la plus proche gagne ; quand tout est exploré, la liste est vide ;
une surface jamais regardée du bon côté attire, puis disparaît une fois couverte ; un carton
repéré vaut plus qu'un mur nu ; deux visites sans lecture écartent la piste ; une pose prise dans
la marge d'un obstacle se rapproche du panneau ; l'avis du guide fait pencher la balance.

## Ce que les vols d'essai ont appris avant la campagne

Chaque règle ci-dessous vient d'un vol qui l'a rendue nécessaire. Cinq vols d'essai, un ou deux
drones, trois à cinq minutes chacun.

**La pose de lecture ne se déduit pas de la direction d'aperçu.** Premier essai : 23 cibles de
lecture atteintes, aucune lue. Le drone se plaçait dans la direction d'où il avait aperçu le
code, souvent à 60 degrés de biais ; à cet angle le lecteur ne lit pas. Deuxième essai, avec le
côté libre le plus net selon la carte : près des bouts de rack, le drone se plaçait au bout du
rack, face à son extrémité vide. La règle retenue : **un panneau se lit perpendiculairement au
grand axe de la structure qui le porte**, du côté où l'espace libre est le plus proche. Résultat :
7 codes à 78 secondes contre 1, et 74 codes en trois minutes avec un seul drone.

**Une piste doit être plausible.** Un cadre vu de biais, dont le rayon passe par un trou du rack,
donne un point trop profond. Une piste n'est créée que si sa distance tient entre 0,45 et 1,8
fois la distance déduite de la taille du cadre.

**Trois chances par piste, pas une infinité.** À 2 mètres, puis à 1,2 mètre parce que les
petites étiquettes ne se lisent pas de loin, puis de l'autre côté. Ensuite la piste est écartée
et personne n'y retourne. Les tentatives se comptent par proximité, parce qu'une piste bouge de
quelques centimètres à chaque nouvelle vue.

**On ne survole pas une structure.** Le premier vol à trois drones a coincé un drone dans le
rack du milieu : sa cible était à 5,5 mètres, le chemin passait au-dessus du rack, et le drone a
commencé la traversée avant d'avoir pris l'altitude. Deux règles : tout obstacle connu entre
2 mètres sous le drone et 0,85 mètre au-dessus bloque la colonne, donc un rack se contourne ;
et l'altitude se prend sur place avant le trajet.

**Une cible réservée est exclue.** Le même vol a vu deux drones choisir exactement la même
cible, à 47 centimètres l'un de l'autre : la pénalité de réservation, 25, restait inférieure à
l'utilité d'une piste, 30. Elle vaut maintenant mille, et une cible à moins de 3 mètres d'un
coéquipier en vol coûte 20 de plus. À 80 centimètres, tout le monde s'arrête, prioritaire ou non.

**Un drone immobile abandonne.** Les recalculs de chemin remettaient la patience du contrôleur
à zéro, et un drone coincé recalculait sans fin. Un drone qui n'a pas bougé de 30 centimètres en
quinze secondes abandonne sa cible et l'écarte ; trente recalculs sur une même cible, pareil.

**L'inventaire connaît le format de ses étiquettes.** L'entrepôt porte d'autres QR sur son
décor ; l'un d'eux, « 409518 », était entré dans l'inventaire. Seuls les codes au format
attendu y entrent, les autres sont comptés à part.

Dernier vol d'essai, deux drones, cinq minutes : les cinq vérifications passent, 89 codes sur
114, aucun point dans une structure, 1,02 mètre au plus près entre les deux drones.

## Ce que les missions à trois drones ont appris

Huit missions nominales ont été volées avant d'obtenir un code stable. Chacune est gardée
(`nominale_v0` à `nominale_v8`, plus `diagnostic_3_drones_400s`). Toutes lisent l'inventaire :
110 à 114 codes sur 114, 90 % des codes en 160 à 200 secondes de temps simulé. Ce qui a changé
entre elles, c'est la sécurité du vol.

**Ce qui distingue trois drones d'un seul.** La simulation tourne sept fois plus lentement que
le temps réel : 1,3 seconde de calcul par cycle de 0,2 seconde simulée, dont 0,75 seconde pour
la physique des trois drones, mesurée. Le pilote automatique, dans un processus séparé, suit ce
rythme. Mais le contrôle en vitesse, validé à un drone avec un gain de 0,9, oscille à trois :
±0,5 à ±0,7 m après plusieurs minutes de vol, avec des inclinaisons de 20 à 35°, jusqu'au
contact avec un rack voisin. Le gain de mission est de 0,5 et le transit à 1 m/s.

**Ce que chaque mission a révélé, dans l'ordre.**

1. **Deux drones sur la même cible, à 47 cm.** La pénalité de réservation, 25, restait sous
   l'utilité d'une piste, 30. Une cible réservée est maintenant exclue, et une cible à moins
   de 4 m d'un coéquipier en vol coûte 40.
2. **Un drone coincé dans un rack à 2 m.** Sa cible était à 5,5 m, le chemin passait au-dessus
   du rack, et il a commencé la traversée avant d'avoir pris l'altitude. On ne survole plus une
   structure, et l'altitude se prend avant le trajet.
3. **Les recalculs de chemin remettaient la patience du contrôleur à zéro** : un drone coincé
   recalculait sans fin. Un drone immobile quinze secondes après l'affectation abandonne sa
   cible ; trente recalculs, pareil.
4. **Un drone sans cible dérivait** à vitesse nulle jusque dans un rack, un autre s'est posé
   au sol. Un drone sans cible tient sa position activement, le drone en panne aussi.
5. **Le détecteur de blocage jugeait « immobile » un drone qui venait de tenir sa place** et
   lui faisait abandonner chaque nouvelle cible : 1 622 cibles en une mission. Il compte depuis
   l'affectation, et un drone sans cible ne décide qu'une fois par deux secondes.
6. **Deux drones se sont touchés malgré la règle d'arrêt.** Un drone à l'arrêt dérive, l'autre
   continue vers une cible voisine. Les coéquipiers sont maintenant des obstacles mobiles de
   2 m dans le calcul des chemins ; céder le passage, c'est tenir sa place activement.
7. **Un drone a heurté le panneau de signalisation au sommet d'un rack**, à 5 m, en volant à
   4,8 m, s'est retourné et s'est posé sur le rack sans être déclaré mort. Plafond de vol à
   4,5 m, et un drone incliné à plus de 70° est déclaré en chute.
8. **Un drone est descendu de 4,5 à 2 m à 1,2 m d'un rack** en dérivant d'un mètre, et un autre
   a traversé une zone inconnue à l'intérieur d'un rack que la carte n'avait pas encore vue.
   Le changement d'altitude se fait à 1,5 m de tout obstacle, après vérification du segment
   pour y aller ; une case inconnue coûte vingt fois une case libre, donc l'inconnu ne se
   traverse plus quand un chemin connu existe ; la tolérance d'arrivée passe à 35 cm.
9. **L'oscillation du contrôle**, décrite plus haut, avec trois chutes tardives sans aucun
   autre drone à moins de 5 m. Gain 0,5, transit 1 m/s.

Le vol de diagnostic de 400 secondes qui a validé les règles 6 à 8 est propre sur toute la
ligne : 114 codes sur 114, aucune chute, aucun point dans un rack, 1,70 m au plus près entre
drones, inclinaison maximale de 30°.

**Deux détails du simulateur, sans conséquence sur les vols.** Le pont Pegasus lance le pilote
ArduPilot deux fois par drone, à la construction puis après la remise à zéro du monde : six
fenêtres pour trois drones, dont trois mortes. Et le calcul d'une observation dépend du partage
de la carte graphique entre le rendu et l'œil appris : 30 à 150 ms par image selon le moment.

## La campagne finale : trois missions, même code

Trois drones, 600 secondes de temps simulé dont 95 de décollage, l'œil appris branché, sans
guide. Le jugement est celui de l'arbitre, qui connaît la vérité.

| ce qu'on a mesuré | nominale, 9033 | panne du drone 1 à 200 s, 9033 | entrepôt 9019, jamais vu |
|---|---|---|---|
| Les cinq vérifications | **5 sur 5** | **5 sur 5** | 3 sur 5 |
| Codes lus | 110 / 114 | **114 / 114** | 61 / 65 |
| Codes inventés | 0 | 0 | 0 |
| 50 % / 80 % / 90 % des codes lus à | 137 / 169 / 185 s | 130 / 156 / 163 s | 150 / 169 / 178 s |
| Entrepôt connu | 89 % | 88 % | 89 % |
| Chutes | **aucune** | **aucune** | trois, à 407, 409 et 413 s |
| Inclinaison maximale | 36° | 15° | 103° |
| Points de trajectoire dans une structure de rack | **0** / 7 707 | **0** / 5 707 | 8 / 4 849, après les chutes |
| Distance minimale entre drones | 2,6 m | 1,96 m | 2,22 m |
| Pas d'attente de priorité | 1 | 32 | 27 |
| Panne absorbée | — | oui : 52 décisions des autres ensuite, sa zone reprise 20 s après, codes 108 puis 114 | — |
| Calcul | 57 min | 53 min | 36 min |

**Ce que les deux premières missions prouvent.** La mécanique tient : personne ne se bloque,
jamais deux drones sur la même cible, les trois partent dans trois directions, la fin est
propre, et la panne d'un drone est absorbée sans une ligne de code qui la surveille. Les
drones ne touchent ni les racks ni leurs coéquipiers. Et l'inventaire est lu : 96 % puis 100 %,
avec 90 % des codes en trois minutes après le décollage.

**Ce que la troisième révèle, et qui reste ouvert.** Sur l'entrepôt 9019, les trois drones,
à 5 à 12 mètres les uns des autres, passent au même moment, vers 300 secondes, d'une inclinaison
inférieure à 18° à des oscillations d'attitude de 40 à 55°, et tombent cent secondes plus tard.
Ce n'est ni une collision, ni le choix des cibles : trois pilotes automatiques indépendants qui
se dérèglent ensemble, c'est un événement de la simulation. Les 61 codes sur 65 étaient lus
avant. La mesure qui permettra de dater cet événement, la durée de chaque cycle avec son temps
simulé, est ajoutée au journal pour le prochain vol. Ce point est écrit tel quel : il n'est pas
compris.

**Le régime à trois drones.** La simulation tourne à un huitième du temps réel, 1,3 seconde de
calcul par cycle de 0,2 seconde, dont 0,75 seconde de physique. Le pilote automatique suit ce
rythme, mais le contrôle en position, validé à un drone avec un gain de 0,9, oscillait à trois.
Le gain de mission est 0,5, l'approche à 0,6 m/s, le transit à 1 m/s.
