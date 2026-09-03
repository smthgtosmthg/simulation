# Test 1 — Est-ce que le même entrepôt revient à l'identique ?

**Ce qu'on veut savoir.** Si on demande deux fois le même entrepôt, obtient-on exactement le
même ? C'est indispensable, parce qu'on voudra plus tard comparer deux versions du système sur
le même terrain.

**Comment on a testé.** L'entrepôt numéro 7 a été construit deux fois, dans deux programmes
séparés, puis photographié d'en haut à chaque fois.

**Ce qu'on a trouvé.** La description de l'entrepôt, qui dit où sont les racks et les cartons,
est identique octet par octet entre les deux constructions. Sur les images, aucun objet n'a
bougé de façon visible : l'écart de géométrie est de 0,000 %, alors qu'on s'autorisait 0,1 %.
L'entrepôt contient 65 cartons et 130 panneaux QR dans les deux cas.

**Un point de méthode important.** La preuve est la description écrite, pas l'image. Le moteur
graphique dessine avec une part de hasard, donc deux images du même entrepôt ne sont jamais
identiques au pixel près, et les comparer directement mène à une fausse conclusion. L'image
sert uniquement à confirmer que rien n'a bougé.

**Verdict : l'entrepôt est reproductible.**

**Image.** `comparaison.jpg` montre les deux constructions côte à côte et la carte de leurs
différences.

**Pour relancer.** `run.py --pass A`, puis `run.py --pass B`, puis `run.py --compare`.
