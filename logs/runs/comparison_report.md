# Comparison report — AIF / Heuristique / NS-3 / Résilience
Generated from `/home/djihene_guitoun/simulation_mc02/logs/runs` (18 runs)

## Résumé

| tag | planner | arch | ns3 | steps | coverage_final | coverage_known_final | entropy_final | innov_final | discovery_rate_final | decisions_per_min_final | active_final | msg_dropped | queue_size | time_to_recovery | durable_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aif_cent_baseline | aif | centralized | wifi | 44 | 93.23 | 0.00 | 0.32 | 0.70 | 0.00 | 0.00 | 3 | 0 | 1 | — | 0 |
| aif_dist_baseline | aif | distributed | wifi | 47 | 93.00 | 0.00 | 0.32 | 0.69 | 0.00 | 0.00 | 3 | 0 | 6 | — | 0 |
| heur_dist_baseline | heuristic | distributed | wifi | 80 | 40.66 | 0.00 | 0.53 | 0.39 | 0.00 | 0.00 | 3 | 0 | 6 | — | 0 |
| aif_cent_kill_d0_s20 | aif | centralized | wifi | 80 | 90.66 | 0.00 | 0.33 | 0.57 | 0.00 | 0.00 | 2 | 0 | 1 | — | 0 |
| heur_cent_kill_d0_s20 | heuristic | centralized | wifi | 80 | 33.00 | 0.00 | 0.56 | 0.12 | 0.00 | 0.00 | 2 | 0 | 1 | — | 0 |
| aif_cent_cut_cloud_s20 | aif | centralized | wifi | 43 | 93.11 | 0.00 | 0.32 | 0.71 | 0.00 | 0.00 | 3 | 0 | 6 | — | 0 |
| aif_dist_cut_links_s20 | aif | distributed | wifi | 64 | 93.04 | 0.00 | 0.31 | 0.58 | 0.00 | 0.00 | 3 | 258 | 0 | — | 0 |
| heur_cent_baseline | heuristic | centralized | wifi | 70 | 93.04 | 0.00 | 0.32 | 0.73 | 0.00 | 0.00 | 3 | 0 | 1 | — | 0 |
| aif_cent_baseline | aif | centralized | wifi | 45 | 93.04 | 93.04 | 0.32 | 0.59 | 0.30 | 180.00 | 3 | 0 | 3 | — | 0 |
| aif_dist_baseline | aif | distributed | wifi | 53 | 93.19 | 86.19 | 0.32 | 0.68 | 0.33 | 180.00 | 3 | 0 | 2 | — | 0 |
| heur_cent_baseline | heuristic | centralized | wifi | 69 | 93.39 | 93.39 | 0.32 | 0.65 | 1.46 | 180.00 | 3 | 0 | 3 | — | 0 |
| aif_cent_cloud_loaded | aif | centralized | wifi | 44 | 93.11 | 93.11 | 0.32 | 0.55 | 0.72 | 180.00 | 3 | 0 | 6 | — | 0 |
| aif_cent_kill_d0_s20 | aif | centralized | wifi | 79 | 93.00 | 93.00 | 0.32 | 0.67 | 0.21 | 120.00 | 2 | 0 | 2 | — | 0 |
| aif_cent_cut_cloud_s20 | aif | centralized | wifi | 64 | 93.00 | 66.78 | 0.32 | 0.45 | 0.08 | 180.00 | 3 | 3 | 0 | — | 0 |
| aif_cent_cut_cloud_s20 | aif | centralized | wifi | 57 | 93.00 | 92.78 | 0.32 | 0.71 | 0.62 | 180.00 | 3 | 3 | 6 | — | 0 |
| aif_dist_cut_links_s20 | aif | distributed | wifi | 52 | 93.15 | 70.54 | 0.32 | 0.52 | 0.75 | 180.00 | 3 | 180 | 0 | — | 0 |
| aif_cent_obstacle_s20 | aif | centralized | wifi | 42 | 93.23 | 93.23 | 0.32 | 0.53 | 1.22 | 180.00 | 3 | 0 | 3 | — | 0 |
| aif_cent_obstacle_s20 | aif | centralized | wifi | 44 | 93.11 | 93.11 | 0.32 | 0.61 | 0.76 | 180.00 | 3 | 0 | 3 | — | 0 |

## Comparaisons systématiques

### AIF vs Heuristique — centralized

Compare la stratégie d'action à environnement égal (sans NS-3, sans cut).

![coverage](comparison_aif_vs_heuristique_—_centralized_coverage.png)

![entropy](comparison_aif_vs_heuristique_—_centralized_entropy.png)

### AIF vs Heuristique — distributed

Même comparaison en architecture distribuée.

![coverage](comparison_aif_vs_heuristique_—_distributed_coverage.png)

![entropy](comparison_aif_vs_heuristique_—_distributed_entropy.png)

### Centralized vs Distributed — AIF

Impact pur de l'architecture, planner identique.

![coverage](comparison_centralized_vs_distributed_—_aif_coverage.png)

![entropy](comparison_centralized_vs_distributed_—_aif_entropy.png)

### Centralized vs Distributed — Heuristique

Idem pour le planner heuristique.

![coverage](comparison_centralized_vs_distributed_—_heuristique_coverage.png)

![entropy](comparison_centralized_vs_distributed_—_heuristique_entropy.png)

### Impact NS-3 WiFi — AIF/centralized

Ajout de la latence WiFi NS-3.

![coverage](comparison_impact_ns-3_wifi_—_aif_centralized_coverage.png)

![entropy](comparison_impact_ns-3_wifi_—_aif_centralized_entropy.png)

### Impact NS-3 WiFi — AIF/distributed

Idem en distribué.

![coverage](comparison_impact_ns-3_wifi_—_aif_distributed_coverage.png)

![entropy](comparison_impact_ns-3_wifi_—_aif_distributed_entropy.png)

## Courbes de résilience par stresseur

### Perte de drone @20 — AIF vs Heuristique

![coverage global](resilience_perte_de_drone_@20_—_aif_vs_heuristique_coverage.png)

![discovery rate](resilience_perte_de_drone_@20_—_aif_vs_heuristique_discovery_rate.png)

![coverage known to planner](resilience_perte_de_drone_@20_—_aif_vs_heuristique_coverage_known.png)

![entropy](resilience_perte_de_drone_@20_—_aif_vs_heuristique_entropy.png)

![innovation](resilience_perte_de_drone_@20_—_aif_vs_heuristique_innovation.png)

### Cut cloud @20 — auto-failover (AIF/centralized)

![coverage global](resilience_cut_cloud_@20_—_auto-failover_(aif_centralized)_coverage.png)

![discovery rate](resilience_cut_cloud_@20_—_auto-failover_(aif_centralized)_discovery_rate.png)

![coverage known to planner](resilience_cut_cloud_@20_—_auto-failover_(aif_centralized)_coverage_known.png)

![entropy](resilience_cut_cloud_@20_—_auto-failover_(aif_centralized)_entropy.png)

![innovation](resilience_cut_cloud_@20_—_auto-failover_(aif_centralized)_innovation.png)

### Cut all drone links @20 — autonomie pure (AIF/distributed)

![coverage global](resilience_cut_all_drone_links_@20_—_autonomie_pure_(aif_distributed)_coverage.png)

![discovery rate](resilience_cut_all_drone_links_@20_—_autonomie_pure_(aif_distributed)_discovery_rate.png)

![coverage known to planner](resilience_cut_all_drone_links_@20_—_autonomie_pure_(aif_distributed)_coverage_known.png)

![entropy](resilience_cut_all_drone_links_@20_—_autonomie_pure_(aif_distributed)_entropy.png)

![innovation](resilience_cut_all_drone_links_@20_—_autonomie_pure_(aif_distributed)_innovation.png)

### Multi-stress (cut cloud@15 + kill d1@30)

![coverage global](resilience_multi-stress_(cut_cloud@15_+_kill_d1@30)_coverage.png)

![discovery rate](resilience_multi-stress_(cut_cloud@15_+_kill_d1@30)_discovery_rate.png)

![coverage known to planner](resilience_multi-stress_(cut_cloud@15_+_kill_d1@30)_coverage_known.png)

![entropy](resilience_multi-stress_(cut_cloud@15_+_kill_d1@30)_entropy.png)

![innovation](resilience_multi-stress_(cut_cloud@15_+_kill_d1@30)_innovation.png)


---
_Generated by `scripts/generate_comparison_report.py`._
