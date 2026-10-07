# pitchtip

Find pitcher **tips** — pre-release tells that give away the pitch — from broadcast video,
then call pitches live with a very cheap decision model ([Jev](https://typesafe.ai) or a local GBM).

```
clip/stream → YOLO11-pose (local, MPS/CPU) → pitcher track on the CF camera shot
            → phase segmentation: set position → leg-lift onset  (cut ≥0.5s before release)
            → features: hands/glove height, wrist gap, elbow angles, set duration, head,
              stance, early-lift deltas + glove-region pixel PCA  (per-game z-scored)
            → decision: local HistGradientBoosting  |  Jev Choice over structured state
            → tells report + per-pitch probabilities
```

## Data (no keys needed)
| what | source |
|---|---|
| true pitch type + `playId` for every pitch | MLB Stats API game feed (`statsapi.mlb.com/api/v1.1/game/{gamePk}/feed/live`) |
| broadcast clip for every pitch (≈2017→today) | Baseball Savant `sporty-videos?playId=…` → mp4 |
| secondary benchmark | MLB-YouTube (Piergiovanni & Ryoo, CVPR'18 workshops) |

Clips are MLB property — personal research only. `data/` is gitignored; never commit video.

## Usage
```bash
uv sync
uv run pitchtip fetch "Yu Darvish" --season 2017 --game-type R --game-type W --max-games 12
uv run pitchtip extract "Yu Darvish"
uv run pitchtip inspect "Yu Darvish"          # eyeball set/onset/cutoff frames
uv run pitchtip tips "Yu Darvish"             # ranked single-feature tells
uv run pitchtip eval "Yu Darvish" --mode fb   # fastball vs offspeed, game-grouped CV + permutation p
uv run pitchtip eval "Yu Darvish" --model jev # needs TYPESAFE_API_KEY
uv run pitchtip train "Yu Darvish"
uv run pitchtip live <youtube-url|file.mp4> "Yu Darvish"
```

## Reading results honestly
Pitchers throw their primary pitch 40–60% of the time, so **accuracy alone means nothing**.
`eval` reports the base rate, log-loss vs. the prior, and a permutation p-value (labels
shuffled within each game). A real tip = accuracy clearly above base rate with p < 0.05.

## Why Jev
Jev is a discriminative "System One" model: state + a `Choice` question → calibrated
probabilities, 70–500 ms, $0.042 / M input tokens (≈ $0.00005 per pitch here). It does
not take pixels, so all vision runs locally and Jev decides over the measured features,
the discovered tells, and the nearest labelled past pitches.
