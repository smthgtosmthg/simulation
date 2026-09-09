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
