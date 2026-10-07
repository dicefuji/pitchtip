"""pitchtip command line."""
from __future__ import annotations

import asyncio
import json
from typing import Optional

import numpy as np
import pandas as pd
import typer

from pitchtip import dataset
from pitchtip.labels import make_labels

app = typer.Typer(add_completion=False, help="Find pitcher tips from broadcast video.")


@app.command()
def fetch(pitcher: str, season: list[int] = typer.Option(..., help="Season(s), repeatable"),
          game_type: list[str] = typer.Option(["R"], help="R=regular, F/D/L/W=postseason rounds"),
          max_games: Optional[int] = None, limit: Optional[int] = None,
          no_download: bool = False, workers: int = 4):
    """Pull pitch labels (MLB Stats API) and download clips (Baseball Savant)."""
    df = dataset.fetch(pitcher, season, max_games, game_type, not no_download, workers, limit)
    typer.echo(f"{len(df)} pitches over {df.game_pk.nunique()} games")
    typer.echo(df.pitch_type.value_counts().to_string())
    if "has_clip" in df:
        typer.echo(f"clips: {df.has_clip.sum()}/{len(df)}")


@app.command()
def extract(pitcher: str, overwrite: bool = False):
    """Run pose + phase segmentation + features over downloaded clips."""
    df = dataset.extract(pitcher, overwrite)
    typer.echo(f"features for {len(df)} pitches")


def _xy(pitcher: str, mode: str, per_game: bool, situation: str = "all"):
    df, hands = dataset.load_features(pitcher, per_game)
    y = make_labels(df.pitch_type, mode)
    keep = (y != "OTHER").to_numpy().copy()
    if situation != "all":
        keep &= df[situation].astype(bool).to_numpy()
    return df[keep].reset_index(drop=True), hands[keep], y[keep].reset_index(drop=True)


@app.command()
def tips(pitcher: str, mode: str = "type", top: int = 10, per_game: bool = True,
         situation: str = typer.Option("all", help="all | men_on | risp")):
    """List the strongest single-feature tells."""
    from pitchtip.tips import find_tells
    df, _, y = _xy(pitcher, mode, per_game, situation)
    t = find_tells(df, y, top)
    for line in t.tell:
        typer.echo(f"• {line}")


@app.command()
def eval(pitcher: str, mode: str = typer.Option("type", help="type | fb (fastball vs offspeed)"),
         model: str = typer.Option("local", help="local | jev | fusion (local vision model -> Jev)"), per_game: bool = True,
         permutations: int = 20, no_hands: bool = False, max_jev: int = 200,
         situation: str = typer.Option("all", help="all | men_on | risp"),
         split: str = typer.Option("cv", help="cv (game-grouped folds) | chrono (train early games, test late)")):
    """Evaluation vs. the pitcher's base rate."""
    from pitchtip.models import local
    df, hands, y = _xy(pitcher, mode, per_game, situation)
    h = None if no_hands else hands
    if model == "local" and split == "chrono":
        games = sorted(df.game_pk.unique())
        tr = (df.game_pk < games[int(len(games) * 0.7)]).to_numpy()
        m = local.fit(df[tr], h[tr] if h is not None else None, y[tr])
        p = m.predict_proba(df[~tr], h[~tr] if h is not None else None)
        s = local.score(y[~tr].reset_index(drop=True), p, list(m.clf.classes_))
    elif model == "local":
        oof, classes = local.cross_val(df, h, y)
        s = local.score(y, oof, classes)
        acc_pred = np.array(classes)[oof.argmax(1)]
        s["permutation_p"] = local.permutation_pvalue(df, h, y, s["accuracy"], permutations) if permutations else None
        s["per_class_recall"] = {c: float((acc_pred[y == c] == c).mean()) for c in classes}
    else:
        from pitchtip.models.jev import JevDecider
        from pitchtip.tips import find_tells
        games = sorted(df.game_pk.unique())
        cut = games[int(len(games) * 0.8)]
        tr, te = df.game_pk < cut, df.game_pk >= cut
        tells = find_tells(df[tr], y[tr], top=12)
        jd = JevDecider(df[tr], y[tr], tells)
        test = df[te].head(max_jev)
        vision = None
        if model == "fusion":
            trm, tem = tr.to_numpy(), np.flatnonzero(te.to_numpy())[:max_jev]
            lm = local.fit(df[tr], h[trm] if h is not None else None, y[tr])
            pv = lm.predict_proba(test, h[tem] if h is not None else None)
            vision = [dict(zip(lm.clf.classes_, row)) for row in pv]
        res = asyncio.run(jd.predict_many(test, pitcher, vision=vision))
        classes = sorted(y.unique())
        proba = np.array([[r["probabilities"].get(c, 0) for c in classes] for r in res])
        s = local.score(y[te].head(max_jev), proba, classes)
        lat = [r["latency_s"] for r in res]
        toks = sum(r["approx_input_tokens"] for r in res)
        s |= {"latency_p50_s": float(np.median(lat)), "approx_cost_usd": toks * 0.042 / 1e6,
              "approx_tokens_per_pitch": toks / len(res)}
    typer.echo(json.dumps(s, indent=2, default=float))


@app.command()
def train(pitcher: str, mode: str = "type", per_game: bool = True):
    """Fit the local model on all games and save it for live use."""
    from pitchtip.models import local
    df, hands, y = _xy(pitcher, mode, per_game)
    m = local.fit(df, hands, y)
    local.save(m, pitcher, mode)
    typer.echo(f"saved model ({len(df)} pitches, classes {m.classes})")


@app.command()
def live(source: str, pitcher: str, mode: str = "type", log: Optional[str] = None):
    """Predict pitches on a stream URL (yt-dlp) or a recorded video file."""
    from pathlib import Path
    from pitchtip.data.mlb import find_pitcher
    from pitchtip.models import local
    from pitchtip import live as live_mod
    p = find_pitcher(pitcher)
    m = local.load(p["name"], mode)
    live_mod.run(source, p["hand"], m, p["name"], log=Path(log) if log else None)


@app.command()
def inspect(pitcher: str, n: int = 12, out: str = "data/inspect.jpg"):
    """Contact sheet of set-start / onset / early-end frames to verify the release cutoff."""
    import cv2
    from pitchtip import config
    from pitchtip.config import slug
    from pitchtip.vision import phases, pose
    df = pd.read_parquet(dataset.pitches_path(pitcher))
    s, hand = slug(df.pitcher_name.iloc[0]), df.pitcher_hand.iloc[0]
    rows = []
    for pid, pt in df.sample(frac=1, random_state=0)[["play_id", "pitch_type"]].itertuples(index=False):
        npz = config.POSE_DIR / s / f"{pid}.npz"
        if not npz.exists():
            continue
        cp = pose.ClipPose.load(npz)
        ph = phases.segment(cp, hand)
        if ph is None:
            continue
        cap = cv2.VideoCapture(str(config.CLIPS_DIR / s / f"{pid}.mp4"))
        tiles = []
        for label, idx in [("set", ph.set_start), ("onset", ph.onset), ("cutoff", ph.early_end - 1)]:
            cap.set(cv2.CAP_PROP_POS_MSEC, cp.t[idx] * 1000)
            ok, fr = cap.read()
            if not ok:
                break
            for x, y_, c in cp.kpts[idx]:
                if np.isfinite(x):
                    cv2.circle(fr, (int(x), int(y_)), 4, (0, 255, 0), -1)
            x1, y1, x2, y2 = cp.boxes[idx].astype(int)
            cv2.rectangle(fr, (x1, y1), (x2, y2), (0, 255, 255), 2)
            fr = cv2.resize(fr, (426, 240))
            cv2.putText(fr, f"{pt} {label} {cp.t[idx]:.1f}s", (8, 24), 0, 0.7, (0, 255, 255), 2)
            tiles.append(fr)
        if len(tiles) == 3:
            rows.append(np.hstack(tiles))
        if len(rows) >= n:
            break
    cv2.imwrite(out, np.vstack(rows))
    typer.echo(out)


if __name__ == "__main__":
    app()
