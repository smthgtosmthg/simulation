# Test 3 — Le drone lit-il vraiment les QR codes en volant ?

**Ce qu'on veut savoir.** C'est le cœur de la tâche : si le drone ne sait pas lire, le reste du
système ne sert à rien. Le drone est ici un vrai appareil piloté par ArduPilot, avec son
inertie et ses imprécisions.

**Comment on a testé.** Deux essais différents. Dans le premier, le drone vole jusqu'à six
distances face à un panneau de 40 centimètres, s'arrête, et prend **une seule photo** à chaque
fois. Dans le second, il longe le rack à un demi-mètre par seconde et lit **en continu**, sans
jamais s'arrêter.

**Ce qu'on a trouvé.** À l'arrêt, le code a été lu à 1,1 mètre, à 2 mètres et à 3 mètres, mais
pas à 0,5 mètre, ni à 0,8 mètre, ni à 1,5 mètre. En vol continu, le drone a lu six codes
différents en un seul passage. Au moment de chaque photo, il était placé à moins de 5
centimètres et à moins d'un degré de la position demandée : le pilotage n'est donc plus ce qui
limite la lecture.

**Ce que cela nous apprend.** Une photo unique est fragile, alors que lire en continu est
robuste, puisqu'une image ratée ne coûte rien quand il y en a cinq par seconde. Or la vraie
mission lira en continu. En dessous d'un mètre environ, la lecture est impossible : le panneau
déborde du cadre et la marge blanche autour du code disparaît, alors que le lecteur en a besoin.

**Verdict : la chaîne complète est prouvée**, du décollage jusqu'à l'identité du carton lu.

**Images.** `planche_qr.jpg` réunit les six photos, une par distance, avec le résultat de
lecture écrit sur chacune. `vol_le_long_du_rack.mp4` montre le vol filmé : à gauche ce que voit
la caméra, à droite la vue d'en haut, et un compteur qui monte jusqu'à six.

**Pour relancer.** `DISPLAY=:1 run.py --seed 7`, qui lance le pilote automatique tout seul.
