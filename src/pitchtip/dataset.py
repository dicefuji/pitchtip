"""On-disk dataset layout and the fetch -> pose -> features pipeline.

A dataset is keyed by "<Pitcher Name> <season(s)>" (e.g. "Tyler Glasnow 2019") because
tips belong to a pitcher *and a period*. Clips and pose files are shared per pitcher.
"""
from __future__ import annotations

import json
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from pitchtip import config, features
from pitchtip.config import slug
from pitchtip.data import mlb, savant
from pitchtip.vision import phases, pose


def dataset_key(name: str, seasons: list[int]) -> str:
    return f"{name} {'-'.join(str(s) for s in seasons)}"


def pitches_path(key: str) -> Path:
    return config.PITCHES_DIR / f"{slug(key)}.parquet"


def _clip(r) -> Path:
    return config.CLIPS_DIR / slug(r.pitcher_name) / f"{r.play_id}.mp4"


def _npz(r) -> Path:
    return config.POSE_DIR / slug(r.pitcher_name) / f"{r.play_id}.npz"


def fetch_labels(name: str, seasons: list[int], max_games: int | None,
                 game_types: list[str]) -> tuple[str, pd.DataFrame]:
    frames, pitcher = [], None
    for gt in game_types:
        pitcher, df = mlb.pitcher_pitches(name, seasons, max_games, gt)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True).drop_duplicates("play_id")
    key = dataset_key(pitcher["name"], seasons)
    out = pitches_path(key)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    return key, df


def fetch(name: str, seasons: list[int], max_games: int | None, game_types: list[str],
          download: bool = True, workers: int = 4) -> pd.DataFrame:
    key, df = fetch_labels(name, seasons, max_games, game_types)
    if download:
        with ThreadPoolExecutor(workers) as ex:
            got = list(tqdm(ex.map(lambda r: savant.download_clip(r.play_id, _clip(r).parent),
                                   df.itertuples()), total=len(df), desc="clips"))
        df["has_clip"] = [g is not None for g in got]
        df.to_parquet(pitches_path(key))
    return df


def _pose_for(r, hand: str, overwrite: bool) -> pose.ClipPose | None:
    clip, npz = _clip(r), _npz(r)
    if npz.exists() and not overwrite:
        cp = pose.ClipPose.load(npz)
        if len(cp.glove) or not clip.exists():
            return cp
    if not clip.exists():
        return None
    cp = pose.extract(clip, hand)
    cp.save(npz)
    return cp


def build_features(key: str, poses: dict[str, pose.ClipPose]) -> pd.DataFrame:
    df = pd.read_parquet(pitches_path(key))
    hand = df.pitcher_hand.iloc[0]
    rows, patches, gloves, skipped = [], [], [], {"no_clip": 0, "no_phase": 0}
    for r in df.itertuples():
        cp = poses.get(r.play_id)
        if cp is None:
            skipped["no_clip"] += 1
            continue
        ph = phases.segment(cp, hand)
        if ph is None:
            skipped["no_phase"] += 1
            continue
        f, patch = features.clip_features(cp, ph, hand)
        rows.append({"play_id": r.play_id, **f})
        patches.append(patch.ravel())
    feats = pd.DataFrame(rows)
    out = df.merge(feats, on="play_id", how="inner")
    s = slug(key)
    config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(config.FEATURES_DIR / f"{s}.parquet")
    np.save(config.FEATURES_DIR / f"{s}_hands.npy", np.array(patches, dtype=np.float32))
    (config.FEATURES_DIR / f"{s}_skipped.json").write_text(json.dumps(skipped))
    return out


def extract(key: str, overwrite: bool = False) -> pd.DataFrame:
    df = pd.read_parquet(pitches_path(key))
    hand = df.pitcher_hand.iloc[0]
    poses = {}
    for r in tqdm(df.itertuples(), total=len(df), desc="pose"):
        cp = _pose_for(r, hand, overwrite)
        if cp is not None:
            poses[r.play_id] = cp
    return build_features(key, poses)


def scan(name: str, seasons: list[int], max_games: int | None, game_types: list[str],
         workers: int = 8, keep_last_games: int = 2) -> pd.DataFrame:
    """Labels -> parallel clip download -> pose as each clip lands -> features.
    Clips are deleted once their pose file is saved (the pose file keeps the glove crops),
    except for the last `keep_last_games` games, kept for replay/live demos."""
    key, df = fetch_labels(name, seasons, max_games, game_types)
    hand = df.pitcher_hand.iloc[0]
    keep_games = set(df.drop_duplicates("game_pk").sort_values("date").game_pk.tail(keep_last_games))
    ready: queue.Queue = queue.Queue(maxsize=64)

    def producer():
        with ThreadPoolExecutor(workers) as ex:
            for r, got in zip(df.itertuples(), ex.map(
                    lambda r: r if _npz(r).exists() else savant.download_clip(r.play_id, _clip(r).parent),
                    df.itertuples())):
                ready.put(r if got is not None else None)
        ready.put(StopIteration)

    threading.Thread(target=producer, daemon=True).start()
    poses = {}
    with tqdm(total=len(df), desc=f"scan {key}") as bar:
        while (r := ready.get()) is not StopIteration:
            bar.update()
            if r is None:
                continue
            try:
                cp = _pose_for(r, hand, False)
            except Exception as e:  # corrupt clip: skip, keep scanning
                tqdm.write(f"skip {r.play_id}: {e}")
                continue
            if cp is not None:
                poses[r.play_id] = cp
                if r.game_pk not in keep_games and _npz(r).exists():
                    _clip(r).unlink(missing_ok=True)
    return build_features(key, poses)


def load_features(key: str, per_game: bool = True) -> tuple[pd.DataFrame, np.ndarray]:
    """Load features. per_game=True subtracts each game's mean so a tell is measured
    against the pitcher's own routine that day, not camera placement in that ballpark."""
    s = slug(key)
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
