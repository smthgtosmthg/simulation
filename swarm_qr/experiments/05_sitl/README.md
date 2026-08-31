# Test 5 — SITL : un drone ArduPilot dans l'entrepôt

**Question.** Le montage Isaac Sim 5.1 + Pegasus + ArduPilot SITL fonctionne-t-il, et le drone
vole-t-il ?

**Recette.** Elle vient du pipeline déjà validé du projet (scripts 11 et 12) et de l'exemple
officiel de Pegasus, avec deux corrections issues de l'enquête : le pas de temps physique est
celui du réglage officiel ArduPilot (1/800 s), et le décor est notre entrepôt chargé par son URL
directe déjà en cache — jamais les décors cloud de Pegasus, dont le téléchargement bloquait la
fenêtre et déclenchait le dialogue « l'application ne répond pas ».

**Lancer.** `DISPLAY=:1 ~/isaac5_env/bin/python sonde_gui.py` — puis, dans le terminal MAVProxy
qui s'ouvre tout seul : `mode guided`, `arm throttle`, `takeoff 3`.

**Ce qu'on doit voir.** L'entrepôt avec ses racks, le drone Iris posé dans l'allée, et après les
trois commandes, le drone qui monte à 3 m — l'altitude s'affiche aussi dans le terminal de
lancement, toutes les 2 secondes.
