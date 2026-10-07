"""Jev (TypeSafe AI) as the decision layer.

Jev accepts structured state, not pixels, so we hand it what the vision pipeline
measured: this pitch's features (per-game z-scores), the tells discovered on the
training games, and the most similar past pitches with their true labels. It returns
a calibrated probability for each pitch in the pitcher's arsenal.
"""
from __future__ import annotations

import asyncio
import json
import os
import time

import numpy as np
import pandas as pd

from pitchtip.tips import describe

INSTRUCTIONS = (
    "A baseball pitcher is in the set position, about to throw. Predict which pitch is "
    "coming from body-language measurements taken BEFORE release. Measurements are "
    "z-scores relative to this pitcher's own average in this game (0 = normal routine). "
    "Use the known tells and compare this_pitch with similar_past_pitches (true labels). "
    "If nothing is informative, follow the arsenal base rates."
)


class JevDecider:
    def __init__(self, train: pd.DataFrame, y: pd.Series, tells: pd.DataFrame,
                 n_features: int = 12, k_neighbors: int = 10, model: str | None = None):
        self.base = y.value_counts(normalize=True)
        self.tells = tells
        cols = list(dict.fromkeys(tells.feature.tolist()))[:n_features]
        self.cols = cols
        X = train[cols].to_numpy(np.float64)
        self.mu, self.sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-9
        self.Z = np.nan_to_num((X - self.mu) / self.sd)
        self.y = y.to_numpy()
        self.k = k_neighbors
        self.model = model
        self.criteria = {}
        for c in self.base.index:
            prof = self.Z[self.y == c].mean(0)
            top = np.argsort(-np.abs(prof))[:4]
            sig = ", ".join(f"{describe(cols[i])} {prof[i]:+.2f}" for i in top)
            self.criteria[c] = f"{c}: {self.base[c]:.0%} of this pitcher's pitches. Typical profile: {sig}"

    def state(self, row: pd.Series, pitcher: str, vision: dict | None = None) -> dict:
        """vision: optional local-model probabilities for this pitch (fusion mode)."""
        z = np.nan_to_num((row[self.cols].to_numpy(np.float64) - self.mu) / self.sd)
        nn = np.argsort(((self.Z - z) ** 2).sum(1))[: self.k]
        named = lambda v: {describe(c): round(float(x), 2) for c, x in zip(self.cols, v)}
        st = {
            "pitcher": pitcher,
            "arsenal_base_rates": {c: round(float(p), 3) for c, p in self.base.items()},
            "known_tells": self.tells.tell.tolist(),
            "this_pitch": named(z),
            "similar_past_pitches": [{"pitch": str(self.y[i]), **named(self.Z[i])} for i in nn],
        }
        if "balls" in row:
            st["situation"] = {
                "count": f"{int(row.balls)}-{int(row.strikes)}", "outs": int(row.outs),
                "runners_on": bool(row.men_on), "batter_side": row.batter_side, "inning": int(row.inning),
            }
        if vision is not None:
            st["vision_model_probabilities"] = {k: round(float(v), 3) for k, v in vision.items()}
            st["vision_model_note"] = ("A pose model trained on this pitcher's past games; it "
                                       "ranks pitches well but is overconfident.")
        return st

    def question(self):
        from typesafe_sdk import Choice
        return Choice(instructions=INSTRUCTIONS, criteria=self.criteria)

    async def predict_many(self, rows: pd.DataFrame, pitcher: str, concurrency: int = 8,
                           vision: list[dict] | None = None) -> list[dict]:
        from typesafe_sdk import AsyncTypeSafeClient
        if not os.environ.get("TYPESAFE_API_KEY"):
            raise RuntimeError("Set TYPESAFE_API_KEY (console.typesafe.ai) to use Jev.")
        sem = asyncio.Semaphore(concurrency)
        q = self.question()
        kwargs = {"model": self.model} if self.model else {}

        async def one(client, row, vis):
            st = self.state(row, pitcher, vis)
            async with sem:
                t0 = time.perf_counter()
                res = await client.system_one(state=st, questions={"pitch": q}, **kwargs)
                dt = time.perf_counter() - t0
            a = res.choices["pitch"]
            return {"choice": a.choice, "confidence": a.confidence,
                    "probabilities": dict(a.probabilities), "latency_s": dt,
                    "approx_input_tokens": len(json.dumps(st)) // 4}

        async with AsyncTypeSafeClient() as client:
            vis = vision or [None] * len(rows)
            return await asyncio.gather(*(one(client, r, v) for (_, r), v in zip(rows.iterrows(), vis)))
