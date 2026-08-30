# swarm_qr

Trois drones découvrent un entrepôt inconnu et lisent les QR codes collés sur les cartons.
Système construit, sans apprentissage. Le plan complet est dans `docs/plan_systeme_swarm.md`.

## Organisation

```
env/            le système
  config.py     les constantes, toutes mesurées dans l'entrepôt
  layout.py     une graine donne un entrepôt (Python pur, testable sans simulateur)
  qr_tags.py    génération des QR et collage sur les cartons
  scene.py      assemblage : entrepôt, racks, cartons, drones, capteurs
experiments/    un dossier par test, avec ses images et sa fiche de résultats
assets/qr/      les images de QR générées
```

## Lancer les tests de l'étape 1

```bash
bash swarm_qr/experiments/run_all.sh
```

Compter environ 17 minutes par lancement du simulateur, soit près de 4 heures pour la campagne
complète. Chaque test peut aussi se lancer seul — voir l'en-tête de son `run.py`.

## Les quatre tests

| Dossier | Question |
|---|---|
| `01_reproductibilite` | La même graine donne-t-elle le même entrepôt ? |
| `02_variation` | Des graines différentes donnent-elles des entrepôts différents ? |
| `03_images_qr` | Les QR sont-ils nets, et jusqu'à quelle distance se décodent-ils ? |
| `04_debit` | Combien de pas par seconde tient le simulateur ? |

Chaque dossier contient son script, ses images, sa vidéo le cas échéant, et un `RESULTATS.md`.

## Deux pièges du simulateur

La position d'un drone ne se lit que par `scene.positions()`. Les lecteurs USD classiques
renvoient la position de départ, figée, même après des dizaines de pas — les images, elles,
suivent bien le drone, donc l'erreur se voit très tard.

Isaac ignore le signal d'arrêt normal : toujours lancer avec `timeout -s KILL`, et vérifier
`nvidia-smi` avant de relancer, un processus mort peut garder plusieurs gigaoctets.

## Étape 1 — résultats

| Test | Verdict | Chiffre clé |
|---|---|---|
| 1 Reproductibilité | validé | disposition identique octet par octet, 0 % de géométrie déplacée |
| 2 Variation | validé | paire de graines la plus proche : 1,85 m d'écart ; 62 à 118 cartons |
| 3 Images et vidéo | validé | décodage réel de 0,8 m à au moins 3 m ; 6 QR lus en vol |
| 4 Débit | validé | 88,5 pas/s à 5 images/s, soit 1,48× le temps réel |

Cinq défauts trouvés et corrigés par ces tests : cartons comptés deux fois, caméra de survol
mal orientée, plafond masquant la vue, caméras placées au centre du drone, et rendu jamais
ralenti. Les deux derniers auraient coûté des semaines s'ils étaient passés inaperçus.

Deux constantes mesurées à retenir pour l'étape 2 : le QR fait 40 cm de côté, et il existe une
**distance minimale** de lecture — trop près, la marge blanche sort du cadre et le décodage
échoue.
