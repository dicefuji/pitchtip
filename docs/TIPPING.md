# What pitch tips actually look like

Research summary that drives pitchtip's feature design. Validation targets with dates and
sources are in [`tip_catalog.csv`](tip_catalog.csv).

## Mechanisms (≈18 documented MLB cases, 2016–2026)
| mechanism | share | examples | pitchtip feature |
|---|---|---|---|
| Ball / grip visible in the glove | ~45% | Schmidt '24, Luzardo '25, Yamamoto '24, Scherzer '25 | `set_hand_bright_frac`, glove-patch PCA (next: DINOv2 crop) |
| Glove / hands height at the set | ~33% | Glasnow '19, Severino '18, Danks | `set_hands_y`, `set_glove_wrist_y`, `set_hands_to_chin` |
| Re-grip / hand motion in the glove | ~22% | Darvish '17, Strasburg '19 | `set_hand_motion_*`, `set_throw_wrist_travel`, `*_std` jitter |
| Timing / tempo | rare | Darvish (Jul '17), Jackson '23 | `set_still_seconds`, `onset_time` |
| Elbows / arms | rare | Helsley '25, Pettitte | `set_elbow_spread`, `*_elbow_out` |
| Head / mouth / gum | rare | Severino glance, Newcomb gum | `head_*` only (mouth not visible from CF) |
| Feet / rubber | rare | historical | not modelled yet |

## Practical rules
- Most modern tips happen **from the stretch with a runner on 2B**, who sees roughly what
  the CF broadcast camera sees. Use `--situation men_on|risp`.
- Tips are **per pitcher and per period** (fixed within days). Evaluate on held-out games,
  ideally chronologically (`--split chrono`), and against a count/situation baseline.
- A RHP in the stretch faces 3B, so CF sees his right side and back. Glove-face flare and
  mouth tells generally need a front view, which the per-pitch clips don't have.
- **False positives are common.** Teams use outcome-independent review: getting hit hard ≠
  tipping (Urías 2023 was reviewed and cleared).

## Prior work
- Ishii, Cal Poly MS thesis 2021: set-position pose + object detection, Glasnow/Darvish/
  Strasburg, ~70%.
- Piergiovanni & Ryoo 2018 (MLB-YouTube): full-pitch video, 36% over 6 pitch types.
- Bright et al. 2026 (arXiv 2603.04874): 80% from 3D pose, but uses release-phase
  features, so it is not a pre-release tip test.

## Validation: blind rediscovery of Glasnow's documented tip
Glasnow confirmed that in 2019 his glove sat higher at the set for fastballs and lower for
curveballs, and that he fixed it for 2020 by varying his glove height. pitchtip was not
told any of this.

| Glasnow, glove/hands height at set | pitches | AUC FF vs CU | p |
|---|---|---|---|
| 2019 regular season (8 games) | 418 | **0.64** | 1e-5 |
| 2019 ALDS G5 (the famous game, 2.2 IP) | 31 | 0.68 | 0.11 (too few pitches) |
| 2020 after the fix (10 games) | 626 | 0.52 | 0.38 |

- In 2019 the blind tell ranking put "hands high in the set → FF 81% (vs 71%)" and "glove
  hand low → CU 40% (vs 29%)" in its top 3 (q ≈ 2e-4).
- In 2020 the tell disappears, and his glove-height spread rises 28%, consistent with
  deliberately varying it.
- The effect is about 0.6% of body height. A hitter can see it live, but it takes
  hundreds of pitches to confirm statistically.

## Darvish 2017 (late season + postseason, 1,197 pitches), fastball vs offspeed
| split | accuracy | base rate | AUC |
|---|---|---|---|
| game-grouped CV | 67.7% | 67.8% | 0.67 |
| chronological (train early, test late incl. WS) | **70.6%** | 65.4% | **0.71** |
| Jev (raw features, same late split) | 61.8% | 62.3% | 0.58 |
| Jev fusion (vision probs + situation) | 67.6% | 62.3% | 0.63 |

Top-decile confident fastball calls: about 90% fastballs vs a 68% base rate.

## Behavior-only results (no count, no runners, no situation)
Features: pose + trajectories around the leg lift, plus DINOv2 embeddings of the glove
region. Model: stacked experts (body trees / glove-appearance / linear posture).
Decision: Jev (`jev-preview` / `jev-latest`) given the expert probabilities, the most
visually similar past deliveries, the tells, and the vision model's track record.

### Leaderboard, fastball vs offspeed (game-grouped CV)
| pitcher-season | n | AUC | acc | base | acc on 25% most confident |
|---|---|---|---|---|---|
| Tyler Glasnow 2019 (documented tip) | 612 | **0.81** | 75.5% | 68.1% | 89.5% |
| Ryan Helsley 2025 (documented tip) | 923 | **0.79** | 71.2% | 53.7% | 87.8% |
| Carlos Rodón 2025 | 1333 | 0.68 | 63.9% | 53.8% | 72.4% |
| Tyler Glasnow 2020 (after his fix) | 767 | 0.66 | 66.2% | 63.1% | 78.5% |
| Clarke Schmidt 2024 | 1170 | 0.64 | 61.8% | 56.8% | 71.2% |
| Jesús Luzardo 2025 | 1511 | 0.63 | 60.5% | 58.2% | 72.1% |
| Yu Darvish 2017 | 1250 | 0.62 | 67.9% | 67.4% | 77.6% |
| Logan Webb 2025 | 931 | 0.58 | 56.1% | 53.9% | 62.5% |

The two pitchers with confirmed, CF-visible tips rank 1 and 2. For Glasnow, the glove-appearance
expert alone reaches AUC 0.86, which fits a glove-position tell.

### Rolling deployment eval (each later block predicted from all earlier games)
| pitcher | n | base | local | **Jev** |
|---|---|---|---|---|
| Glasnow 2019 | 212 | 71.7% | 74.5% | **77.4%** |
| Rodón 2025 | 526 | 52.8% | 64.8% | **66.3%** |
| Helsley 2025 | 372 | 54.8% | 64.8% | **65.3%** |
| Luzardo 2025 | 1006 | 57.9% | 65.0% | 64.5% |
| Schmidt 2024 | 506 | 53.9% | 57.7% | **58.3%** |
| Darvish 2017 | 474 | 68.1% | 67.1% | 65.6% |

Jev decision cost is about $0.003 per 1,000 pitches, with ~120–220 ms latency.

### Live replay: Glasnow, 2019 ALDS Game 5 (the famous tipping game)
`pitchtip reel` stitched the game into one continuous video. `pitchtip live --train-before
2019-10-10` had to find the pitcher, detect the set and leg lift, and call each pitch.
- **32 of 37 called pitches correct (86%)**, against a 68% always-fastball base rate.
- **5 of 6 offspeed calls correct.**
- About 0.2 s per decision, made before release.

### Luzardo 2025 before/after his reported fix (6/11)
AUC 0.695 before vs 0.668 after; with runners on, 0.692 vs 0.658. Same direction as the
report, but the gap is small.
