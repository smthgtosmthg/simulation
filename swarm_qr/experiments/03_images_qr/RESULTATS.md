# Test 3 — Netteté des QR et vol du drone

**Question.** Les QR sont-ils lisibles, et à quelle distance ? Le drone vole-t-il vraiment ?

**Méthode.** Le drone est placé face à un QR de 40 cm, à six distances, et on tente le décodage
réel avec OpenCV. Puis il longe un rack à 0,5 m/s pendant que sa caméra et la vue de dessus
sont filmées.

| Distance | Résultat |
|---|---|
| 0,5 m | non décodé |
| 0,8 m | décodé |
| 1,1 m | décodé |
| 1,5 m | décodé |
| 2,0 m | décodé |
| 3,0 m | décodé (un autre QR entre dans le champ) |

L'échec à 0,5 m est instructif : le QR remplit tout le cadre et sa marge blanche est coupée,
alors qu'un décodeur en a besoin. **Il existe donc une distance minimale, pas seulement une
distance maximale.** À 2 et 3 mètres, trois QR tiennent dans la même image, donc plusieurs
cartons pourront être lus d'un coup. La portée maximale n'est pas encore bornée : elle sera
mesurée proprement à l'étape 2.

**Verdict : les QR sont nets et se décodent de 0,8 m à au moins 3 m.** Le drone vole, ses
caméras suivent, et 6 QR ont été lus pendant le vol le long du rack.

**Images.** `planche_qr.jpg` — les six distances. `vol_le_long_du_rack.mp4` — 197 images.

**Relancer.** `run.py --seed 7`.
