# pitchtip

Call the next pitch from a pitcher's **behavior alone**: set position, glove, hands, elbows,
the first instant of the leg lift. It uses no count, runners or game situation. A cheap
decision model, [Jev](https://typesafe.ai), makes the call.

```
broadcast clip / stream
  → YOLO11x-pose (local, Apple MPS): pitcher tracked on the centre-field shot
  → phase segmentation: set position → leg-lift onset (cut ≥0.5 s before release)
  → features: posture, hand/glove/elbow trajectories around the lift (per-game z-scored)
              + DINOv2 embedding of the glove region (grip / glove shape / ball visibility)
  → stacked experts (body trees · glove appearance · linear posture), calibrated
  → Jev Choice: expert probabilities, similar past deliveries, tells, track record
  → call + trust (calibrated) + STRONG flag for the top-quartile calls
```

## Results (behavior only, deployment-style rolling eval; see docs/TIPPING.md)
| pitcher | FB/offspeed base | **Jev** | Jev top-25% (trust-gated) |
|---|---|---|---|
| Tyler Glasnow 2019 (documented tip) | 71.0% | **88.1%** (exact pitch type: 90.0%) | 94% |
| Max Fried 2025 | 55.1% | **72.7%** | 88% |
| Freddy Peralta 2025 | 53.5% | **70.4%** | 86% |
| Yu Darvish 2017 | 68.2% | **69.9%** | — |
| Ryan Helsley 2025 (documented tip) | 53.6% | **69.1%** | 85% |
| Carlos Rodón 2025 | 52.2% | **68.2%** | 82% |
| Jesús Luzardo 2025 | 58.1% | **62.7%** | 79% |
| Zac Gallen 2025 | 55.3% | **61.4%** | 78% |

**Live replays** (one continuous video, trained only on earlier games):
- Glasnow, 2019 ALDS G5: **89–94%** of called pitches, against a 68% base rate.
- Peralta, 2025-09-22: **70%**, against a 55% base rate.

Jev costs about $0.003 per 1,000 pitches and answers in ~0.1–0.2 s.

## Data (no keys needed)
| what | source |
|---|---|
| true pitch type + `playId` for every pitch | MLB Stats API game feed |
| broadcast clip for every pitch (2017→) | Baseball Savant `sporty-videos?playId=…` → mp4 |

Clips are MLB property, for personal research only. `data/` is gitignored. `scan` deletes clips
once their pose file (which keeps the glove crops) is saved, except the last 2 games, which are
kept for demos.

## Usage
```bash
uv sync && cp .env.example .env            # add TYPESAFE_API_KEY
uv run pitchtip scan "Tyler Glasnow:2019" "Ryan Helsley:2025"   # download + pose + glove embeddings
uv run pitchtip leaderboard                 # who is most predictable from behavior
uv run pitchtip patterns                    # league-wide tell families
uv run pitchtip tips "Tyler Glasnow 2019" --mode fb
uv run pitchtip rolling "Tyler Glasnow 2019" --mode type        # deployment-style eval, Jev auto
uv run pitchtip demo "Tyler Glasnow 2019" --game 599341 --mode fb
uv run pitchtip reel "Tyler Glasnow 2019" 599341 --out g5.mp4
uv run pitchtip live g5.mp4 "Tyler Glasnow 2019" --mode fb --train-before 2019-10-10
uv run pitchtip live "https://…stream…" "Tyler Glasnow 2019"     # any yt-dlp-readable stream
```

## Honest caveats
- Tips are specific to a pitcher and a period. Models are per pitcher-season and should be
  retrained as games come in, which is what the rolling eval does.
- Only behavior visible from the CF camera can be learned. Glove-face and mouth tells need a
  front view.
- Live mode needs about 15 fps of YOLO11x pose. A dedicated GPU, or YOLO11l with a matching
  model, is needed for true real time.
- Every live replay is a single game (35–66 calls). The rolling eval is the more reliable number.
