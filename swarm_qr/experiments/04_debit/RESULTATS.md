# Test 4 — Débit du simulateur

**Question.** Combien de pas de simulation par seconde tient la machine, selon le nombre
d'images demandées ?

**Méthode.** Trois drones volent avec leurs neuf caméras. On fait varier la fréquence de rendu
et on chronomètre 240 pas.

| Images/s | Pas/s | Vitesse | Mission de 5 min |
|---|---|---|---|
| 1 | 248,8 | 4,15× le temps réel | 1,2 min |
| **5** | **88,5** | **1,48×** | **3,4 min** |
| 15 | 33,2 | 0,55× | 9,0 min |
| 60 | 8,7 | 0,14× | 34,7 min |

Le coût est presque entièrement dans le rendu : passer de 60 à 5 images par seconde multiplie
la vitesse par dix. Un premier essai avait donné 8,6 pas par seconde quelle que soit la
fréquence demandée, parce que le réglage utilisé limitait la relecture du capteur sans empêcher
le simulateur de rendre. Le bon levier est de demander explicitement `sim.step(render=False)`
la plupart du temps.

**Réglage retenu : 5 images par seconde.** Le simulateur va alors plus vite que le temps réel,
et cinq images par seconde suffisent largement pour un drone à moins d'un mètre par seconde,
qui ne parcourt que 20 cm entre deux images. Les caméras gardent leur pleine résolution : c'est
elle qui rend la lecture des QR possible.

**Image.** `courbe_debit.png` — le débit et la vitesse par rapport au temps réel.

**Relancer.** `run.py --hz N` pour chaque fréquence, puis `run.py --plot`.
