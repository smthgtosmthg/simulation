# Mesure 01 — Résolution et lisibilité d'un QR code

Date : 29 août 2026 · `qr_resolution_test.py` · CPU seul, aucun simulateur
30 essais par point (décalage sous-pixel, rotation ±3°, flou 0,6 px, bruit σ=2)
Décodeurs : OpenCV + OpenCV-Aruco + pyzbar

## Le QR réellement généré par le projet

Le code demande `version=1`, mais `qr.make(fit=True)` **relève la version** quand la charge
utile ne rentre pas. Avec un identifiant du type `Item#A14-R5` et une correction d'erreur H,
on obtient en réalité une **version 2** : **29 modules de large**, marge comprise (25 + 2×2).

C'est une différence de 16 % sur la finesse à résoudre. À corriger dans toute analyse.

## A. Le seuil de lisibilité

| Critère | Seuil mesuré |
|---|---|
| Décodage fiable (≥ 90 %) | **2,07 pixels par module** |
| Décodage possible (≥ 50 %) | **1,72 pixels par module** |
| Repérage fiable (≥ 90 %) | **1,38 pixels par module** |
| Repérage possible (≥ 50 %) | **1,21 pixels par module** |

**Repérer un QR sans le lire est 1,4 à 1,5 fois plus facile que le décoder** (en distance).

Ce chiffre tranche une question ouverte de l'étude : la fiche D3 annonçait un facteur de
**1,75 à 3,5**, en s'appuyant sur une source qui ne mesurait pas la détection. La mesure donne
**1,42×** au seuil à 50 % et **1,50×** au seuil à 90 %. Le facteur existe, mais il est **environ
deux fois plus petit qu'annoncé**.

## B. Taille physique du QR, déduite

La calibration du projet (`docs/calibration_gate.csv`, vrai rendu Isaac à 1280×960, FOV 60°)
décode à 1,50 m et échoue à 1,75 m. Croisée avec le seuil ci-dessus, elle donne un panneau de
**6,7 à 7,9 cm de côté** — on retient **7,3 cm**.

C'est nettement plus petit que les 10 cm que j'avais supposés dans l'analyse initiale.

## C. Portée de lecture selon la résolution d'entrée

| Résolution | Portée de décodage |
|---|---|
| **64 px** (entrée standard d'un world model) | **7,5 cm** |
| 96 px | 10 cm |
| 128 px | 15 cm |
| 256 px | 25 cm |
| 512 px | 60 cm |
| **1280 px** | **1,50 m** |

**Validation croisée** : le modèle prédit 1,50 m à 1280 px, et la calibration réelle du projet
sous Isaac Sim mesure exactement 1,50 m. Le test synthétique reproduit donc la mesure réelle.

## D. Conclusion

**Pour décoder à 1,25 m, il faut une image d'environ 985 pixels de large.**
Un world model qui encode en 64×64 est **15 fois en dessous**, et sa portée de lecture serait
de **7,5 centimètres**.

La condition 1 de l'architecture D1 (version « pixels purs ») est donc **confirmée par la
mesure** : un world model ne peut pas voir les QR dans son entrée. Un décodeur externe en
pleine résolution est obligatoire, quelle que soit la direction retenue.

## Ce que cette mesure ne prouve pas

- Rendu **synthétique et frontal**. Pas de perspective forte, pas de flou de bougé, pas de
  variation d'éclairage, pas d'occlusion. **Les chiffres sont donc optimistes** : en conditions
  réelles, les portées seront plus courtes.
- Le repérage est mesuré avec le détecteur d'OpenCV. Un détecteur appris (type YOLO) ferait
  probablement mieux, et pourrait relever le facteur de 1,4×. À mesurer si la question devient
  décisive.
- La série de repérage est bruitée aux très petites tailles (faux positifs à 0,69 et 0,86
  px/module). Seuls les seuils à 90 % sont solides.

## Suite

Ces chiffres alimentent la conception : à 1,25 m de distance de lecture, la caméra doit faire
**au moins 1000 px de large** au champ de 60°. C'est une contrainte matérielle à figer avant
de choisir quoi que ce soit d'autre.
