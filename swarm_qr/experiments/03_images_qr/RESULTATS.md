# Test 3 — Le drone lit-il vraiment les QR codes en volant ?

## Ce qu'on veut savoir

C'est le cœur de la tâche : si le drone ne sait pas lire, le reste du système ne sert à rien.
Le drone est ici un vrai appareil piloté par ArduPilot, avec son inertie et ses imprécisions.

## Comment on a testé

Deux essais dans un même vol. D'abord la planche : le drone rejoint six distances face à un
panneau de 40 centimètres, s'arrête, prend une photo, et on vérifie que **le code visé** est lu
— pas celui d'un carton voisin. Ensuite la vidéo : il longe le rack à un demi-mètre par seconde
en lisant en continu.

Le trajet vers le rack passe par le couloir ouvert au bout des racks : la ligne droite les
traverserait, et l'évitement d'obstacles n'existe pas encore à ce stade du projet.

## Ce qu'on a trouvé

**Les six distances sont lues, de 0,5 à 3 mètres**, avec une arrivée entre 4 et 11 centimètres
du point demandé. À partir de 1,5 mètre, les cartons voisins se lisent en même temps que la
cible — trois codes dans une seule image à 3 mètres.

En vol continu le long du rack, le drone a lu **six codes différents en un seul passage**.

## Ce que cela corrige

Les versions précédentes de ce test concluaient qu'il existait une distance minimale de lecture
autour d'un mètre. C'était faux : la caméra est montée 10 centimètres sur le côté du drone et
11 centimètres plus bas, et l'ancienne visée ne compensait que la hauteur. Le drone se plaçait
donc toujours 10 centimètres trop près, et à courte distance ce décalage suffisait à couper la
marge blanche du code. Avec la visée complète, le panneau tient dans le cadre dès 0,5 mètre et
se lit.

**Verdict : la chaîne complète est prouvée** — décollage, trajet sans collision, visée, capture,
lecture, identité du carton — sur toute la plage de 0,5 à 3 mètres.

## Images

`planche_qr.jpg` réunit les six photos, une par distance, avec le résultat écrit sur chacune.
`vol_le_long_du_rack.mp4` montre le vol : à gauche la caméra du drone avec le compteur de codes
lus, à droite la vue de dessus avec sa position.

## Pour relancer

`DISPLAY=:1 run.py --seed 7` — le pilote automatique se lance tout seul.
