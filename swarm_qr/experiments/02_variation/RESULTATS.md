# Test 2 — Est-ce que les entrepôts sont vraiment différents entre eux ?

**Ce qu'on veut savoir.** Si tous les entrepôts se ressemblent, dire que notre système
généralise ne veut rien dire, puisqu'il aura toujours vu la même chose.

**Comment on a testé.** Six entrepôts ont été construits avec six numéros différents, puis
photographiés d'en haut. Pour chaque paire d'entrepôts, on a mesuré de combien les trois racks
avaient bougé en moyenne.

**Ce qu'on a trouvé.** Le résultat qui compte est celui du pire cas : les deux entrepôts qui se
ressemblent le plus ont quand même leurs racks déplacés de 1,85 mètre. Les positions varient
de 1,73 mètre d'écart-type en largeur et de 3,27 mètres en profondeur. Le nombre de cartons
change beaucoup lui aussi, entre 62 et 118 selon l'entrepôt.

**Un point de méthode important.** On juge chaque entrepôt dans son ensemble, et non rack par
rack. Avec six entrepôts et trois racks, cela fait quarante-cinq comparaisons : que deux racks
tombent au même endroit par hasard est attendu et sans conséquence.

**Verdict : la variation est suffisante.** Les drones exploreront de vraies formes nouvelles.

**Image.** `planche_variation.jpg` montre les six entrepôts côte à côte.

**Pour relancer.** `run.py --seed N` pour chaque numéro de 1 à 6, puis `run.py --board`.
