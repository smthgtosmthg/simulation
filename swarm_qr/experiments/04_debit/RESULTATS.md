# Test 4 — Combien de temps coûte une simulation ?

**Ce qu'on veut savoir.** Ce chiffre gouverne tout le projet, puisqu'il dit combien d'heures
coûtera l'évaluation finale.

**Comment on a testé.** La scène complète, avec trois drones et neuf caméras, a été chronométrée
dans trois situations : sans aucune image, avec cinq images par seconde, et avec une image à
chaque pas de calcul.

**Ce qu'on a trouvé.** Sans images, la machine tient 222 pas de simulation par seconde. Avec
cinq images par seconde, elle en tient 206, soit à peine 7 % de moins. Avec une image à chaque
pas, elle s'effondre à 6 pas par seconde.

**Ce que cela nous apprend.** Le coût ne vient pas des caméras, contrairement à ce qu'on
pouvait croire, mais de la physique du pilote automatique, qui doit être calculée huit cents
fois par seconde. La simulation tourne donc environ quatre fois moins vite que le temps réel :
une mission de dix minutes demande à peu près quarante minutes de calcul, et l'évaluation
finale de vingt-cinq missions demandera environ seize heures.

**Réglage retenu : cinq images par seconde.** C'est largement suffisant pour un drone qui vole
sous un mètre par seconde, et les caméras gardent leur pleine résolution, ce qui est la
condition pour lire les QR codes.

**Image.** `courbe_debit.png` compare les trois situations.

**Pour relancer.** `run.py`, puis `run.py --plot` pour afficher le tableau.
