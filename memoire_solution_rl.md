# La solution par apprentissage par renforcement

## Ce que la politique apprend

Une politique unique π(a | o) prend l'observation d'un drone et sort une commande de vitesse. Elle
décide tout : où aller, quel carton viser, comment l'aborder, comment se répartir le travail avec
les autres. Les trois drones **partagent les mêmes poids** — un seul réseau produit les trois
décisions, et ce qui les distingue est leur observation.

Contrainte d'information : aucune position de code non lu n'entre dans `o`. Ces positions
n'apparaissent que dans le calcul de la récompense et dans l'entrée du critique, qui n'existent
tous les deux qu'à l'entraînement.

## L'observation

Deux blocs concaténés en un seul vecteur.

**Les cartes (18 432 nombres).** L'essaim tient des grilles partagées de 25 cm à neuf couches :
occupation, exploré, couverture de scan par bande de hauteur (0–1,5 / 1,5–3 / 3–4,7 m), frontière,
codes lus, trace de soi, traces des coéquipiers. On y découpe deux carrés de 32 × 32 cases,
centrés sur le drone et tournés dans son cap : l'un large de 8 m, l'autre de 32 m. Deux échelles
× neuf couches = dix-huit canaux d'image.

**Le vecteur d'état (98 nombres).** Lidar réduit à 72 secteurs égocentriques (minimum par secteur
de 5°), vitesse linéaire et angulaire, sinus et cosinus du cap, altitude, fraction de temps
écoulé, position relative de chaque coéquipier avec son bit « en vol », deux bits disant si la
vitesse et la rotation courantes permettraient une lecture, les quatre seuils courants du
curriculum, et l'identité du drone en code un-parmi-trois.

**Les sept valeurs privilégiées** vont au critique seul : fraction de codes lus, fraction
restante, potentiel courant, direction et distance du code non lu le plus proche, fraction de
l'équipe en vol.

## L'action

Quatre nombres : `a = (vx, vy, vz, ω)`, vitesses dans le repère du drone, produites à 30 Hz.

```
µ = 3·tanh(f_acteur(o) / 3)          moyenne bornée en douceur
σ ∈ [0,05 ; 1,2]                      écart-type appris, encadré
a ~ N(µ, σ), puis a ← clamp(a, −1, 1)
a_xy ← a_xy / max(1, ‖a_xy‖)          la diagonale ne dépasse pas la norme d'un axe
```

La commande passe ensuite par un modèle d'actionneur — premier ordre de constante 0,30 s,
accélération bornée à 3,6 m/s² — avant d'atteindre le drone.

## La récompense

Pour un drone k, à chaque pas :

```
r = couverture + lectures + chocs + potentiel − commande − séparation

couverture = c(t)·[ 1,0·façades + 0,1·(neuves − façades) + 0,5·marginales ] / 13
             − 0,02·recouvrement / 13
lectures   = l(t)·[ 25·codes_neufs + 10·postures_nominales ] + jalons/3
chocs      = −2,0·max(0, (0,5 − d_obstacle)/0,5)² − 5,0·[d_obstacle < 0,25]
potentiel  = 0,5·(γ·Φ' − Φ),  Φ = −d_code/12 + 0,3·lenteur·proximité
commande   = 0,005·‖a_t − a_{t−1}‖² + 0,5·Σ (|a| − 1)₊²
séparation = Σ_{j≠k} 0,25·max(0, (0,6 − d_kj)/0,6)²
```

Les deux horloges `c(t) = 1 − 0,5·t` et `l(t) = 1 + 2·t`, où `t` est l'avancement du curriculum,
font glisser la pression de « balayer » vers « lire ». Le potentiel Φ est calculé à partir de la
distance au code non lu le plus proche : il n'existe qu'à l'entraînement, et sa forme en
différence `γΦ' − Φ` garantit qu'il ne change pas la politique optimale. Un jalon est payé une
fois par équipe quand la fraction lue franchit 50 %, 75 % puis 90 %.

## Le réseau

Acteur et critique ont la même forme, sans poids partagés entre eux :

```
cartes (18 canaux, 32×32) ─► Conv 3×3/2 → 32 ─► Conv 3×3/2 → 64 ─► Conv 3×3/2 → 64
                           ─► aplatissement ─► Linéaire 256
                                                    │
vecteur d'état ─────────────────────────────────────┤ concaténation
                                                    ▼
                                          Linéaire 256 → 128 → sortie
```

Activation ELU partout. Sortie de l'acteur : 4 valeurs. Sortie du critique : 1 valeur, avec en
entrée le vecteur augmenté des sept valeurs privilégiées — c'est un apprentissage centralisé à
exécution décentralisée.

## L'entraînement

**Algorithme.** PPO, implémentation `rsl_rl`. Les trois drones d'un même entrepôt sont vus par
l'algorithme comme trois environnements distincts qui partagent un réseau : un adaptateur aplatit
les 64 entrepôts × 3 drones en 192 environnements parallèles.

| réglage | valeur |
|---|---|
| pas collectés par itération | 32 par environnement, soit 6 144 transitions |
| époques par lot / mini-lots | 5 / 1 |
| coupure du ratio | 0,2 |
| pas d'apprentissage | 3 · 10⁻⁴, fixe |
| facteur d'actualisation γ / GAE λ | 0,995 / 0,95 |
| coefficient de valeur | 1,0 |
| coefficient d'entropie | 0,01, puis 0,001 |
| norme maximale du gradient | 1,0 |

**Le retrait du bruit.** Le coefficient d'entropie reste à 0,01 pendant toute la montée du
curriculum : il maintient un écart-type d'exploration suffisant. Quand le curriculum atteint son
avant-dernier cran, l'environnement écrit lui-même 0,001 dans l'algorithme. La politique doit
alors finir d'apprendre sans le bruit qui la portait — ce qui l'oblige à faire passer la
compétence dans sa moyenne.

**Ce qui change à chaque épisode.** Une configuration d'inventaire tirée dans un jeu gelé de 500
(jamais celles de validation ni de test), les positions et les caps de naissance, et
éventuellement une panne : un drone tiré au sort cesse de voler à un instant tiré entre 25 % et
75 % de l'épisode. L'épisode s'arrête quand 95 % des codes lisibles sont lus, ou au bout du temps
imparti.

## Le curriculum

Les seuils de la tâche ne sont pas fixes. Ils s'interpolent entre un réglage tolérant et le
réglage nominal selon le niveau courant :

```
seuil(niveau) = tolérant + (nominal − tolérant) · niveau/10
```

| seuil | tolérant (niveau 0) | nominal (niveau 10) |
|---|---|---|
| distance de lecture | 4,00 m | 1,25 m |
| angle d'incidence | 60° | 50° |
| vitesse maximale | 1,45 m/s | 0,60 m/s |
| rotation maximale | 1,6 rad/s | 0,8 rad/s |

La règle de progression s'appuie sur une moyenne lissée du taux de lecture,
`E ← 0,95·E + 0,05·taux`, mise à jour à chaque fin d'épisode :

- **monter** d'un cran si `E > 0,88` sur vingt-cinq épisodes consécutifs, et au moins cent vingt
  épisodes passés au cran courant ;
- **redescendre** d'un cran si `E` retombe sous 0,65, après un délai de grâce de deux cent
  cinquante épisodes.

Le niveau pilote aussi trois autres choses : le temps accordé à l'épisode (150 s, plus 15 s par
cran, plafonné à 295 s), la probabilité de naître près d'un carton plutôt qu'au hasard (0,7 → 0,3),
et, une fois le nominal atteint, la probabilité de panne (0 → 0,3).

## L'implémentation

| fichier | rôle |
|---|---|
| `mapping.py` | les grilles partagées et leur découpe égocentrique, en torch pur |
| `env_map.py` | observation, récompense, événement de lecture, resets, pannes |
| `models.py` | acteur-critique convolutionnel, moyenne bornée, écart-type encadré |
| `curriculum.py` | interpolation des seuils, montée et recul |
| `layouts.py` | générateur de configurations et découpe entraînement / validation / test |
| `flatten_wrapper.py` | adaptateur essaim → environnements parallèles pour `rsl_rl` |
| `train.py` | assemblage et boucle d'apprentissage |

Tout ce qui ne dépend pas du simulateur — cartes, curriculum, configurations — est du torch pur et
se teste sans lancer Isaac.
