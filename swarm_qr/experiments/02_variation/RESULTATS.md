# Test 2 — Variation entre entrepôts

**Question.** Des graines différentes donnent-elles des entrepôts vraiment différents ?

**Méthode.** Six entrepôts construits avec les graines 1 à 6, photographiés de dessus et
assemblés en une planche. On mesure, pour chaque paire de graines, le déplacement moyen des
trois racks.

| Mesure | Valeur |
|---|---|
| Déplacement moyen de la paire la plus ressemblante | 1,85 m (graines 1 et 6) |
| Écart-type des positions en X | 1,73 m |
| Écart-type des positions en Y | 3,27 m |
| Nombre de cartons | de 62 à 118 |

On juge la disposition dans son ensemble, pas rack par rack : avec six graines et trois racks
cela fait quarante-cinq comparaisons, donc deux racks qui tombent au même endroit par hasard
sont attendus et sans conséquence. Ce qui compte est que chaque paire d'entrepôts diffère
nettement, et le pire cas reste à près de deux mètres.

**Verdict : variation suffisante.** Les drones exploreront de vraies formes nouvelles, et
l'évaluation finale pourra dire quelque chose sur la généralisation.

**Images.** `planche_variation.jpg` — les six entrepôts côte à côte.

**Relancer.** `run.py --seed N` pour chaque graine, puis `run.py --board`.
