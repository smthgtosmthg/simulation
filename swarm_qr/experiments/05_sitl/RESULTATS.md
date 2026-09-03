# Test 5 — SITL : la chaîne vole, et voici ses chiffres

**Question.** Isaac Sim 5.1 + Pegasus + ArduPilot SITL : est-ce que ça vole, et à quel prix ?

**Réponse : oui, chaîne validée de bout en bout.** Lien de commande établi, estimateur prêt en
36 s, mode guidé confirmé, armement réussi, décollage accepté (acquittement 0) et altitude de
3 m atteinte — le tout automatiquement, sans fenêtre, par `vol_auto.py`.

| Mesure | Valeur |
|---|---|
| Stationnaire 10 s | écart-type X 10,8 / Y 0,9 / Z 1,2 cm — rayon max 20,9 cm |
| Arrêt depuis 1,01 m/s | 6,8 s et 1,06 m (jusqu'à moins de 5 cm/s) |
| Débit (physique 1/800 s, sans rendu) | 413 pas/s, soit 0,52× le temps réel |

L'écart-type en X est pollué par la fin de stabilisation après la montée : la mesure a commencé
4 s seulement après l'arrivée à 3 m, et Y/Z à ~1 cm montrent la vraie tenue. L'arrêt mesure
l'inertie réelle du drone — celle que l'ancien cube à vitesse instantanée n'avait pas — avec un
seuil volontairement strict. Le débit sans rendu donne une mission de 10 min en ~19 min réelles ;
il sera re-mesuré avec les caméras.

**Piège résolu en route :** sur un port MAVLink secondaire, ArduPilot n'envoie presque rien tant
qu'on ne s'annonce pas comme station sol et qu'on ne demande pas les flux — et un décollage
avant que l'estimateur ait son origine GPS est refusé en silence. D'où l'attente du fixe 3D
plus 20 s, comme le faisait déjà le pipeline Gazebo du projet.

**Relancer.** `DISPLAY=:1 ~/isaac5_env/bin/python vol_auto.py` — rapport dans `resultat_vol.json`.
