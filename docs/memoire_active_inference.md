# La solution par inférence active

## Le critère

Le drone garde une croyance sur ce qui l'entoure et choisit, à chaque cycle, l'action qui minimise
son **énergie libre attendue** G. Cette quantité met sur une seule échelle ce qu'une action ferait
apprendre et ce qu'elle coûterait. Explorer n'est donc pas une consigne : les endroits mal connus
sont simplement ceux dont G est le plus bas.

## La croyance

Une grille de 50 cm couvre le bâtiment. Chaque case porte une probabilité d'occupation, initialisée
à 0,5. Elle est stockée en log-odds, où accumuler des observations revient à additionner :

```
l = log( p / (1 − p) )        p = 1 / (1 + e^(−l))        |l| ≤ 30
```

La borne à ±30 empêche une case de devenir une certitude que plus aucune observation ne pourrait
corriger — c'est elle qui permet à la carte de réagir à un changement du monde.

**Mise à jour par le lidar.** Pour chacun des 360 rayons, on parcourt les cases traversées jusqu'à
l'impact :

```
case traversée :  l ← l − 0,55        case touchée :  l ← l + 0,85
```

**Mesure de l'ignorance.** L'entropie de Bernoulli d'une case, maximale à p = 0,5 et nulle pour une
case sûre :

```
H(p) = − p·log p − (1 − p)·log(1 − p)
```

Sa moyenne sur la grille, notée H̄, est l'ignorance de l'essaim.

## La fusion entre drones

Les croyances se combinent par moyenne des log-odds ramenés à l'a priori (pool d'opinions
indépendantes), puis chaque drone mélange sa propre carte avec le résultat :

```
l_fusion = l_0 + (1/N) · Σ_i ( l_i − l_0 )
l_plan   = (1 − λ)·l_local + λ·l_fusion,       λ = 0,3
```

Un drone garde donc 70 % de ce qu'il a vu lui-même. C'est `l_plan` qui sert à décider.

## Les candidats

Neuf actions : huit directions cardinales et intercardinales à un mètre, plus rester sur place. Un
candidat est écarté d'office s'il sort des bornes navigables ou s'il tombe dans une case dont la
probabilité d'occupation dépasse 0,65.

## Les termes de G

**Gain d'information épistémique.** On se place au candidat et on relance les 360 rayons dans la
croyance courante. Chaque case que le rayon traverserait apporte son entropie, pondérée par la
probabilité que le rayon aille jusque-là :

```
IG(x) = Σ_rayons Σ_pas  p_atteinte · H(p_case)
        avec  p_atteinte ← p_atteinte · (1 − p_case),  arrêt sous 10⁻⁴
```

C'est une anticipation, pas une mesure : elle répond à « si j'y vais, qu'est-ce que j'apprends ».
Le facteur `(1 − p_case)` fait que les cases cachées derrière un obstacle supposé ne comptent
presque plus.

**Attrait de frontière (valeur pragmatique).** Sur une fenêtre de 11 × 11 cases autour du
candidat, la présence de cases encore indécises, pondérée par l'inverse de la distance :

```
F(x) = moyenne sur la fenêtre de  (1 − |2p − 1|) / (distance + 1)
```

**Coût de mouvement.** 0 pour « rester », 1 pour tout déplacement.

**Risque de collision entre drones.**

```
C(x) = Σ_{coéquipiers à moins de 8 m}  1 / (distance + 0,1)
```

**Marge aux obstacles.** Sur un disque de 6 cases de rayon, la somme des cases jugées occupées,
pondérée par l'inverse de la distance :

```
Cl(x) = Σ_{cases occupées, d ≤ 6}  1 / (d + 0,1)
```

**Formule complète, en régime normal :**

```
G(x) = − 2,5·IG(x) − 0,8·F(x) + 0,1·mouvement + 5,0·C(x) + 25,0·Cl(x)
```

Les deux premiers termes tirent vers l'inconnu, les trois derniers freinent.

## Le choix de l'action

Pas un minimum strict, mais un tirage adouci :

```
P(x_i) ∝ exp( − ( G_i − min_j G_j ) / T ),       T = 0,3
```

La température fixe le compromis : à T → 0 on retombe sur le minimum, à T grand le choix devient
aléatoire. Le tirage est nécessaire ici — trois drones aux croyances proches produiraient
exactement la même décision, et le hasard casse cette symétrie. Si aucun candidat n'est valide, le
drone reste sur place.

L'action retenue devient une consigne de position à un mètre, que le pilote de bord rejoint
pendant le cycle suivant.

## La détection de changement et les phases

**L'innovation** mesure l'écart entre ce que le lidar touche et ce que la carte prévoyait :

```
innov = moyenne, sur les impacts, de  |1 − p_prévue(case touchée)|
```

Elle est suivie par une moyenne et une variance glissantes, et une rupture est déclarée quand elle
dépasse son historique de deux écarts-types :

```
µ ← µ + α·(innov − µ)        v ← v + α·((innov − µ)² − v)        α = 0,05
alerte  si  innov > µ + 2·√v
```

Une alerte fait passer le système en phase de **récupération** pour trente cycles, puis en phase
**durable**. Chaque phase a sa propre formule de G :

```
récupération : G = − 3,0·H̄ − 1,2·IG − 2,5·IG − 0,8·F + 0,1·m + 5·C + 25·Cl
durable      : G = − 1,2·H̄ − 2,5·IG − 0,8·F + 0,1·m + 5·C + 25·Cl
                   + 10·max(0, H̄ − 0,44)
```

En récupération, l'information est comptée deux fois : le drone va activement chercher ce qui a
changé. En durable, un terme de maintien pénalise une carte laissée trop incertaine. Le retour à
la phase normale demande que H̄ ≤ 0,44 et innov ≤ 0,16 pendant soixante cycles d'affilée.

## L'implémentation

| fichier | rôle |
|---|---|
| `belief.py` | la grille, la mise à jour lidar, l'entropie, la fusion et le mélange |
| `planner.py` | les neuf candidats, les cinq termes, les trois formules de G, le tirage |
| `agent.py` | un drone : sa croyance, son innovation, l'exécution de la consigne |
| `swarm.py` | le cycle complet de l'essaim et les métriques par pas |
| `resilience.py` | la moyenne glissante, les seuils, la machine à trois phases |
| `math_utils.py` | log-odds, entropie, tirage adouci |

**Un cycle** enchaîne : perception et mise à jour des croyances, calcul de l'innovation et de la
phase, diffusion et fusion des cartes, évaluation des neuf candidats, tirage, envoi de la consigne
au pilote. Le calcul de décision est le même que la décision soit prise par un planificateur
central ou par chaque drone à partir des cartes de ses voisins ; seul change qui exécute la
formule.

## La variante pour l'inventaire

Le même critère a été porté sur la lecture de QR codes, en calcul par lots sur carte graphique.
Trois termes changent, pas la structure :

- **candidats** : huit caps plus le vol stationnaire, projetés 1,6 s en avant à la vitesse de
  transit, au lieu de huit cases voisines ;
- **valeur épistémique** : le nombre de cellules de façade encore non scannées qu'un candidat
  mettrait à portée de caméra, au lieu de l'entropie d'occupation ;
- **valeur pragmatique** : la distance à la cellule de travail la plus proche mesurée sur un champ
  géodésique qui suit les allées, et non à vol d'oiseau, sinon le drone est attiré contre la face
  opposée d'un rack.

Le risque se lit dans un cône de ±25° autour du cap visé, sur les secteurs lidar. La formule de
combinaison reste `G = −w_e·épistémique − w_p·pragmatique + w_r·risque + coût de mouvement`, et le
choix se fait par minimum en évaluation, par tirage adouci sinon.
