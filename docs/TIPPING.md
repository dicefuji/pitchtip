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
