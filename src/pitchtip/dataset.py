"""On-disk dataset layout and the fetch -> pose -> features pipeline."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from pitchtip import config, features
from pitchtip.config import slug
from pitchtip.data import mlb, savant
from pitchtip.vision import phases, pose


def pitches_path(name: str) -> Path:
    return config.PITCHES_DIR / f"{slug(name)}.parquet"


def fetch(name: str, seasons: list[int], max_games: int | None, game_types: list[str],
          download: bool = True, workers: int = 4, limit: int | None = None) -> pd.DataFrame:
    frames, pitcher = [], None
    for gt in game_types:
        pitcher, df = mlb.pitcher_pitches(name, seasons, max_games, gt)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True).drop_duplicates("play_id")
    if limit:
        df = df.tail(limit)
    out = pitches_path(pitcher["name"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    if download:
        clip_dir = config.CLIPS_DIR / slug(pitcher["name"])
        with ThreadPoolExecutor(workers) as ex:
            got = list(tqdm(ex.map(lambda pid: savant.download_clip(pid, clip_dir), df.play_id),
                            total=len(df), desc="clips"))
        df["has_clip"] = [g is not None for g in got]
        df.to_parquet(out)
    return df


def extract(name: str, overwrite: bool = False) -> pd.DataFrame:
    df = pd.read_parquet(pitches_path(name))
    s = slug(df.pitcher_name.iloc[0])
    hand = df.pitcher_hand.iloc[0]
    rows, patches, skipped = [], [], {"no_clip": 0, "no_phase": 0}
    for r in tqdm(df.itertuples(), total=len(df), desc="pose"):
        clip = config.CLIPS_DIR / s / f"{r.play_id}.mp4"
        npz = config.POSE_DIR / s / f"{r.play_id}.npz"
        if not clip.exists():
            skipped["no_clip"] += 1
            continue
        if npz.exists() and not overwrite:
            cp = pose.ClipPose.load(npz)
        else:
            cp = pose.extract(clip)
            cp.save(npz)
        ph = phases.segment(cp, hand)
        if ph is None:
            skipped["no_phase"] += 1
            continue
        f, patch = features.clip_features(cp, ph, hand)
        rows.append({"play_id": r.play_id, **f})
        patches.append(patch.ravel())
    feats = pd.DataFrame(rows)
    out = df.merge(feats, on="play_id", how="inner")
    config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(config.FEATURES_DIR / f"{s}.parquet")
    np.save(config.FEATURES_DIR / f"{s}_hands.npy", np.array(patches, dtype=np.float32))
    (config.FEATURES_DIR / f"{s}_skipped.json").write_text(json.dumps(skipped))
    return out


def load_features(name: str, per_game: bool = True) -> tuple[pd.DataFrame, np.ndarray]:
    """Load features. per_game=True subtracts each game's mean so a tell is measured
    against the pitcher's own routine that day, not camera placement in that ballpark."""
    s = slug(name)
    df = pd.read_parquet(config.FEATURES_DIR / f"{s}.parquet").reset_index(drop=True)
    hands = np.load(config.FEATURES_DIR / f"{s}_hands.npy")
    if per_game:
        cols = feature_columns(df)
        df[cols] = df[cols] - df.groupby("game_pk")[cols].transform("mean")
        for _, idx in df.groupby("game_pk").indices.items():
            hands[idx] -= hands[idx].mean(axis=0)
    return df, hands


FEATURE_PREFIXES = ("set_", "early_")


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]
