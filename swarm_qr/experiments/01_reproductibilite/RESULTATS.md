# Test 1 — Reproductibilité

**Question.** Deux constructions avec la même graine donnent-elles le même entrepôt ?

**Méthode.** L'entrepôt de la graine 7 est construit deux fois, dans deux processus séparés, et
photographié de dessus. On compare la disposition écrite en JSON, puis les deux images.

| Mesure | Valeur | Rôle |
|---|---|---|
| Disposition identique | oui, octet par octet | **la preuve** |
| Écart structurel des images | 0,008 % | géométrie déplacée, seuil 0,1 % |
| Écart moyen des pixels | 3,41 sur 255 | informatif |
| Contenu | 65 cartons, 130 panneaux QR | |

La disposition en JSON fait foi : c'est elle qui définit la scène. L'image sert à confirmer
qu'aucune géométrie n'a bougé de façon visible, et 0,008 % représente 46 pixels sur 518 400.
L'écart moyen vient du moteur de rendu, qui échantillonne au hasard, et de la stabilisation des
cartons, le moteur physique n'étant pas parfaitement déterministe d'une exécution à l'autre.

**Verdict : reproductible.** Deux versions du système pourront être comparées sur exactement le
même terrain.

**Images.** `comparaison.jpg` — les deux vues et la carte des différences.

**Relancer.** `run.py --pass A`, puis `--pass B`, puis `--compare`.
