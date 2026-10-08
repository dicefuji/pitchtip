# pitchtip

Predict a pitcher's next pitch from **body language alone**: the set position, glove and
hand position, arm angles, and the first instant of the leg lift. It uses no count, runners,
or game situation. Video becomes pose and glove-image evidence locally, and a cheap
decision model, [Jev](https://typesafe.ai) by TypeSafe AI, makes the call.

It works on Baseball Savant clips, recorded games, and YouTube or other streams.

```
broadcast clip / stream
  → pose cascade: YOLO11n tracks the pitcher every frame (~6 ms); YOLO11x re-reads the
    set + lift in small batches as frames arrive
  → phase segmentation: coming set → set → leg-lift onset (everything ends before release)
  → behavior features: arm angles, hand/glove/head trajectories, set duration, elbow spread
    + DINOv2 embedding of the glove region (grip, glove shape, ball visibility)
  → stacked expert models (body trees · glove appearance · linear posture), calibrated
  → Jev Choice question: expert opinions, their track record, similar past deliveries, tells
  → call + calibrated trust + STRONG flag (top-quartile trust)
```

## Findings

All numbers are behavior-only. Models are per pitcher-season and are always tested on games
they were not trained on.

### 1. Most pitchers are readable from body language
Rolling evaluation: each later block of games is predicted from all earlier games, the way
the system would run during a season. The task is fastball vs offspeed, over 20
pitcher-seasons and ~9,500 test pitches.

| | accuracy |
|---|---|
| always guess the most common pitch | 57% |
| **Jev (pooled)** | **68%** |
| Jev, top-quartile trust calls only | 80–95% depending on pitcher |

| most readable | base | Jev | | least readable | base | Jev |
|---|---|---|---|---|---|---|
| Tyler Glasnow 2019 | 71% | **88%** | | Garrett Crochet 2025 | 75% | 74% |
| Mason Miller 2026 | 54% | **78%** | | Yusei Kikuchi 2025 | 62% | 64% |
| Ryan Helsley 2024 | 54% | **75%** | | Yoshinobu Yamamoto 2024 | 53% | 60% |
| Max Fried 2025 | 55% | **73%** | | Zac Gallen 2025 | 55% | 61% |

Exact pitch type (full arsenal): Glasnow 90% (base 71%), Helsley 64% over 3 pitches
(base 47%), Fried 31% over 6 pitches (base 21%). Full tables are in
[`docs/TIPPING.md`](docs/TIPPING.md). Charts: [`docs/img/`](docs/img).

### 2. It rediscovers documented tips on its own
- **Glasnow 2019.** His glove sat higher before fastballs (he confirmed it). Without being
  told, the top-ranked tell was "glove-hand height in the set high → fastball 79% vs 68%."
  In 2020, after his reported fix, his tell weakened (AUC 0.93 → 0.75).
- **Helsley 2025.** The reported "arm tick coming set" shows up as glove-elbow flare in the set.
- **Mason Miller, 2026 NLDS G2** (alleged tipping). Replaying the game: 29/37 correct
  (base 51%), and 7/7 STRONG calls correct.
- **Missed: Strasburg, 2019 WS G6.** His tell, reaching into the glove *before* coming set,
  happened outside the original feature window (1st inning 6/12). Fixed by adding "coming
  set" features.

### 3. League-wide patterns
Out of 15 pitcher-seasons, **every one had at least one statistically significant tell**
(FDR q<0.01). The most common kinds are glove-arm and throwing-arm **elbow angle**, then
hand and glove height. Tells are spread across the set, the last frames before the lift,
and the first instant of the lift. Watching only the static set position misses many of them.

### 4. Clips and live streams
| test | correct | base rate | STRONG |
|---|---|---|---|
| Glasnow, 2019 ALDS G5 (stitched into one video; YOLO11m run, YOLO11x got 32/36) | 33/35 (94%) | 68% | 7/7 offspeed calls |
| Peralta, 2025-09-22 (stitched) | 46/66 (70%) | 55% | |
| **Mason Miller, 3 YouTube innings** (after lift starts) | **29/39 (74%)** | 69% | 8/9 |
| **Mason Miller, same innings, called *before* the leg lift** | **28/38 (74%)** | 68% | 10/13 |

- Live mode runs straight from a stream URL at ~1.9x real time on an Apple-silicon laptop.
- Pre-lift calls are ready a median **0.17 s before the leg lift** (~1.2–1.6 s before release).
- Across the five most readable pitchers, set-only calls cost about 6 points on the rolling
  eval but stay 13–19 points over base.

### 5. How Jev is used
- Jev does not see pixels. Each pitch is one `Choice` question whose criteria are the
  pitcher's arsenal. The state carries the expert models' probabilities, how reliable each
  has been for this pitcher, the 25 most similar past deliveries, and the known tells.
  A real request and response is in [`docs/jev_example.json`](docs/jev_example.json).
- **Evidence is chosen per pitcher automatically.** Plain probabilities work best when the
  signal is weak. Track record and similar deliveries help when it is strong.
- **Jev's own confidence is polarized** (almost always near 0 or 1). Rank calls by the
  calibrated vision model's belief in Jev's pick instead; the STRONG flag does this.
- Cost: ~2k input tokens per call ≈ **$0.003 per 1,000 pitches**, 0.1–0.2 s per decision.
- Honest comparison: Jev matches the free local model's accuracy. It never meaningfully
  beat it, because it can only weigh evidence the vision pipeline provides.

### 6. What mattered, and what did not
| change | effect (Glasnow 2019, rolling) |
|---|---|
| pose model YOLO11s → m → l → **x** | AUC 0.78 → 0.86 → 0.89 → **0.93** |
| DINOv2 glove-region embedding | glove expert alone reached AUC ~0.86 on Glasnow |
| trajectories around the lift | +2–3 AUC points |
| DINOv2-base instead of small | worse (glove crops are only ~30 source pixels) |
| Jev multi-question ensemble | no gain |
| pitcher-centred crop for pose (faster) | slightly worse (0.90 vs 0.93) |

Leaks we found and removed, worth knowing if you extend this:
- `onset_time`: Savant clips are cut relative to release, so it encoded delivery length.
- A runners-on flag used the *post*-play base state.
- Clip URLs ending in an HTML-escaped `=` silently dropped every 2019 postseason clip.

## Keys and requirements

| what | needed? | where |
|---|---|---|
| `TYPESAFE_API_KEY` | **yes, for Jev decisions** (the local model works without it) | [console.typesafe.ai](https://console.typesafe.ai), early access |
| MLB Stats API, Baseball Savant | no key | public endpoints |
| `yt-dlp` | only for YouTube / stream URLs | run automatically via `uvx` if installed, else `brew install yt-dlp` |
| GPU | strongly recommended | Apple silicon (MPS) or CUDA; YOLO11x is ~2 s/clip on an M-series laptop |
| disk | ~5 MB per clip while scanning | `scan` deletes clips once their pose file (with glove crops) is saved |

```bash
uv sync
cp .env.example .env        # then add TYPESAFE_API_KEY
```

## Usage

```bash
# 1. Data: labels from the MLB feed, clips from Savant, pose + glove embeddings
uv run pitchtip scan "Mason Miller:2026" "Max Fried:2025:15" --game-type R --game-type D

# 2. Analysis
uv run pitchtip leaderboard                         # who is most readable
uv run pitchtip patterns                            # league-wide tell families
uv run pitchtip tips "Mason Miller 2026" --mode fb  # ranked tells for one pitcher

# 3. Evaluation (deployment-style; Jev picks its evidence automatically)
uv run pitchtip rolling "Mason Miller 2026" --mode fb
uv run pitchtip demo "Mason Miller 2026" --game 849825 --mode fb   # replay one held-out game

# 4. Video and streams
uv run pitchtip ytest video.mp4 "Mason Miller 2026" 823253 --out out/ --innings 9   # score vs MLB feed
uv run pitchtip live "https://www.youtube.com/watch?v=…" "Mason Miller 2026" --set-only
```

Dataset keys are `"<Pitcher Name> <season>"`. Tips belong to a pitcher *and a period*,
so models are per pitcher-season.

## How to improve on this

Ordered by expected payoff:

1. **More games per pitcher.** The YOLO11x re-scan stopped partway, and many pitcher-seasons
   have only 15 games. Accuracy rose with data everywhere we checked. Run `scan` without a game cap.
2. **A glove detector and a higher-resolution glove view.** The most common documented tell
   class (grip visible in the glove) is limited by ~30-pixel glove crops from 720p
   center-field video. A dedicated glove/ball detector, or 1080p sources, should help most.
3. **Better onset detection for live use.** YOLO11n sometimes notices the lift late (2–4 s).
   A small temporal model on the fast keypoints, or a better fast tracker, would make
   pre-lift calls more consistent.
4. **Calibrate or fine-tune the decision step.** Jev's probabilities are polarized.
   Calibrating them, or giving it per-pitcher few-shot context as TypeSafe's tooling
   matures, may let Jev beat the local model rather than match it.
5. **More camera angles.** Glove-face and mouth tells need the front or 1B/3B views, which
   per-pitch Savant clips don't provide.
6. **Drift detection.** Tips change mid-season, and pitchers fix them. Retrain on a rolling
   window and alert when a pitcher's readability jumps. That is also a defensive product:
   tell your own pitchers when they start tipping.
7. **Broader validation.** Run `ytest` on full innings for more pitchers. So far the clip
   tests are mostly Mason Miller.

## Data and limits

- Clips and broadcast video belong to MLB and its broadcasters. Use them for personal research
  only. `data/` is gitignored and no video is committed.
- Each clip or replay result is from one game (35–76 pitches). The rolling evaluation is
  the reliable number.
- Readability is not the same as exploitability. Teams report that knowing the pitch does not
  guarantee hitting it, and false "tipping" alarms are common.

Research notes, the documented-tip catalog, and every experiment are in
[`docs/TIPPING.md`](docs/TIPPING.md) and [`docs/tip_catalog.csv`](docs/tip_catalog.csv).
