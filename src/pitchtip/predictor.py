"""Causal, behavior-only pitch predictor for clips and live streams.

Normalization is online: each pitch's features are compared with the running average of
the pitcher's earlier deliveries in the same game (camera + routine baseline), so nothing
from the future is used. Decision = BehaviorModel probabilities, optionally fused by Jev.
"""
from __future__ import annotations

import asyncio
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pitchtip import config, dataset, features
from pitchtip.config import slug
from pitchtip.labels import make_labels
from pitchtip.models.behavior import BehaviorModel
from pitchtip.models.jev import JevBehavior
from pitchtip.tips import find_tells
from pitchtip.vision import embed, phases, pose


def raw_training(key: str, mode: str, games: set | None = None):
    """Raw (un-normalized) features/hands/embeddings, optionally restricted to games."""
    df, hands = dataset.load_features(key, per_game=False)
    emb = embed.load(key, per_game=False, df=df)
    y = make_labels(df.pitch_type, mode)
    keep = (y != "OTHER").to_numpy().copy()
    if games is not None:
        keep &= df.game_pk.isin(games).to_numpy()
    return (df[keep].reset_index(drop=True), hands[keep], emb[keep], y[keep].reset_index(drop=True))


def causal_normalize(df: pd.DataFrame, hands: np.ndarray, emb: np.ndarray, warmup: int = 5):
    """Subtract the mean of earlier pitches in the same game (first `warmup` pitches use
    the mean of the first `warmup` pitches)."""
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    df = df.copy()
    cols = dataset.feature_columns(df)
    hands, emb = hands.copy(), emb.copy()
    order = df.sort_values(["game_pk", "at_bat", "pitch_number"]).index
    for g, idx in df.loc[order].groupby("game_pk", sort=False).groups.items():
        idx = np.array(list(idx))
        for arr in ("df", "hands", "emb"):
            X = df.loc[idx, cols].to_numpy(np.float64) if arr == "df" else (hands if arr == "hands" else emb)[idx]
            X = np.where(np.isfinite(X), X, np.nan)
            csum = np.nancumsum(X, axis=0)
            cnt = np.cumsum(np.isfinite(X), axis=0)
            prev_mean = np.vstack([np.full((1, X.shape[1]), np.nan), (csum / np.maximum(cnt, 1))[:-1]])
            warm = np.nanmean(X[:warmup], axis=0)
            prev_mean[:warmup] = warm
            Y = X - prev_mean
            if arr == "df":
                df.loc[idx, cols] = Y
            elif arr == "hands":
                hands[idx] = Y
            else:
                emb[idx] = Y
    return df, np.nan_to_num(hands), np.nan_to_num(emb)


@dataclass
class Predictor:
    key: str
    mode: str = "type"
    use_jev: bool = True
    variant: str = "probs+knn"
    hand: str | None = None
    # running per-game baselines
    _sum: dict = field(default_factory=dict)
    _n: int = 0

    def fit(self, games: set | None = None):
        df, hands, emb, y = raw_training(self.key, self.mode, games)
        self.hand = df.pitcher_hand.iloc[0]
        self.cols = dataset.feature_columns(df)
        df, hands, emb = causal_normalize(df, hands, emb)
        self.model = BehaviorModel().fit(df, hands, emb, y)
        tells = find_tells(df, y, top=6)
        self.jev = JevBehavior(self.model, df, hands, emb, y, tells, variant=self.variant)
        self.reset_game()
        return self

    def reset_game(self):
        self._sum, self._n, self._first = {}, 0, []

    def _normalize(self, f: dict, hands: np.ndarray, e: np.ndarray, warmup: int = 5):
        x = np.array([f.get(c, np.nan) for c in self.cols], np.float64)
        cur = {"f": x, "h": hands.astype(np.float64), "e": e.astype(np.float64)}
        if self._n < warmup:
            self._first.append(cur)
            base = {k: np.nanmean([d[k] for d in self._first], axis=0) for k in cur}
        else:
            base = {k: self._sum[k] / self._n for k in cur}
        for k, v in cur.items():
            self._sum[k] = self._sum.get(k, 0) + np.nan_to_num(v)
        self._n += 1
        row = pd.DataFrame([dict(zip(self.cols, x - base["f"]))])
        return row, np.nan_to_num(cur["h"] - base["h"])[None], np.nan_to_num(cur["e"] - base["e"])[None]

    def features_from_pose(self, cp: pose.ClipPose):
        ph = phases.segment(cp, self.hand)
        if ph is None:
            return None
        f, patch = features.clip_features(cp, ph, self.hand)
        e = embed.clip_embedding(cp, ph)
        e = np.where(np.isfinite(e), e, 0)
        return f, patch.ravel(), e

    def predict_features(self, f, patch, e) -> dict:
        row, h, em = self._normalize(f, patch, e)
        p = self.model.predict_proba(row, h, em)[0]
        out = {"vision": dict(zip(self.model.classes, map(float, p)))}
        if self.use_jev:
            st = self.jev.evidence(row, h, em)
            r = asyncio.run(self.jev.predict(st))[0]
            out["jev"] = {k: float(v) for k, v in r["probabilities"].items()}
            out["jev_latency_s"] = r["latency_s"]
        final = out.get("jev", out["vision"])
        out["call"] = max(final, key=final.get)
        out["confidence"] = final[out["call"]]
        # Jev's own confidence is polarized; the calibrated vision model's belief in the call
        # is the better guide to which calls to act on.
        out["trust"] = out["vision"].get(out["call"], 0.0)
        out["strong"] = out["trust"] >= getattr(self.model, "strong_threshold", 1.0)
        return out

    def predict_clip(self, clip) -> dict | None:
        cp = pose.extract(clip, self.hand)
        feats = self.features_from_pose(cp)
        return None if feats is None else self.predict_features(*feats)
