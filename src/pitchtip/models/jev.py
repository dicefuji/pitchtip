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


BEHAVIOR_INSTRUCTIONS = (
    "Predict the pitch a baseball pitcher is about to throw using ONLY his body language "
    "before release (set position and the start of the leg lift): glove/hand position, grip "
    "motion, elbows, posture. Evidence: (1) a vision model's calibrated probabilities, "
    "(2) the pitches thrown on the most visually similar past deliveries, (3) this "
    "pitcher's known tells with today's readings. Weigh agreeing evidence; when evidence "
    "is weak, stay close to the arsenal base rates. The vision model's track record on "
    "unseen games tells you how far to trust its confidence: follow it where it has been "
    "reliable, discount it where it has not."
)


class JevBehavior:
    """Jev over behavior-only evidence produced by a BehaviorModel."""

    def __init__(self, model, train_df, train_hands, train_emb, y_train, tells=None,
                 k: int = 25, variant: str = "full", jev_model: str | None = None):
        self.m, self.variant, self.k, self.jev_model = model, variant, k, jev_model
        self.base = y_train.value_counts(normalize=True)
        X = model._views(train_df, train_hands, train_emb)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        self.Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
        self.y = y_train.to_numpy()
        self.tells = tells if tells is not None else pd.DataFrame(columns=["feature", "tell"])
        self.tcols = self.tells.feature.tolist()[:6]
        if self.tcols:
            T = train_df[self.tcols].to_numpy(np.float64)
            self.tmu, self.tsd = np.nanmean(T, 0), np.nanstd(T, 0) + 1e-9
        self.criteria = {c: f"{c} ({self.base[c]:.0%} of this pitcher's pitches)" for c in self.base.index}

    def evidence(self, df, hands, emb) -> list[dict]:
        P = self.m.predict_proba(df, hands, emb)
        EP = self.m.expert_probas(df, hands, emb) if hasattr(self.m, "expert_probas") else {}
        X = (self.m._views(df, hands, emb) - self.mu) / self.sd
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
        sims = X @ self.Z.T
        out = []
        for i in range(len(df)):
            st = {"arsenal_base_rates": {c: round(float(p), 3) for c, p in self.base.items()}}
            if self.variant in ("probs", "probs+knn", "full", "experts"):
                st["vision_model_probabilities"] = {c: round(float(p), 3) for c, p in zip(self.m.classes, P[i])}
            if self.variant in ("experts", "full", "trust+full", "ensemble") and EP:
                st["expert_opinions"] = {
                    {"body": "body position & movement model", "glove": "glove/hands appearance model",
                     "linear": "simple posture model"}[n]: {c: round(float(p), 3) for c, p in zip(self.m.classes, EP[n][i])}
                    for n in EP}
                if getattr(self.m, "expert_weights", None):
                    st["expert_reliability_for_this_pitcher"] = {n: float(w) for n, w in self.m.expert_weights.items()}
            if self.variant in ("trust", "trust+full", "ensemble"):
                st["vision_model_probabilities"] = {c: round(float(p), 3) for c, p in zip(self.m.classes, P[i])}
                st["vision_model_confidence"] = round(float(P[i].max()), 3)
                st["vision_model_track_record_on_unseen_games"] = {
                    "by_confidence": getattr(self.m, "reliability", {}),
                    "accuracy_when_it_calls": getattr(self.m, "per_call", {})}
            if self.variant in ("knn", "probs+knn", "full", "trust", "trust+full", "ensemble"):
                nn = np.argsort(-sims[i])[: self.k]
                votes = pd.Series(self.y[nn]).value_counts()
                st["most_similar_past_deliveries"] = {
                    "k": self.k, "pitch_counts": {c: int(n) for c, n in votes.items()},
                    "closest_5": [str(self.y[j]) for j in nn[:5]]}
            if self.variant in ("tells", "full", "trust+full", "ensemble") and self.tcols:
                z = (df.iloc[i][self.tcols].to_numpy(np.float64) - self.tmu) / self.tsd
                st["known_tells"] = [
                    {"tell": t, "this_pitch_reading_z": round(float(v), 2)}
                    for t, v in zip(self.tells.tell.tolist()[:6], np.nan_to_num(z))]
            out.append(st)
        return out

    ENSEMBLE_QUESTIONS = {
        "overall": BEHAVIOR_INSTRUCTIONS,
        "by_models": ("Which pitch is coming? Rely mainly on vision_model_probabilities and "
                      "expert_opinions, trusting them as far as their track record supports."),
        "by_similar": ("Which pitch is coming? Rely mainly on most_similar_past_deliveries: "
                       "what did this pitcher throw when his set position looked like this?"),
        "by_tells": ("Which pitch is coming? Rely mainly on known_tells and this pitch's "
                     "readings; a strongly deviating reading on a reliable tell is decisive."),
    }

    async def predict(self, states: list[dict], concurrency: int = 16) -> list[dict]:
        from typesafe_sdk import AsyncTypeSafeClient, Choice
        if self.variant == "ensemble":
            qs = {k: Choice(instructions=v, criteria=self.criteria) for k, v in self.ENSEMBLE_QUESTIONS.items()}
        else:
            qs = {"pitch": Choice(instructions=BEHAVIOR_INSTRUCTIONS, criteria=self.criteria)}
        sem = asyncio.Semaphore(concurrency)
        kw = {"model": self.jev_model} if self.jev_model else {}

        async def one(client, st):
            async with sem:
                t0 = time.perf_counter()
                r = None
                for attempt in range(5):
                    try:
                        r = await client.system_one(state=st, questions=qs, timeout=30.0, **kw)
                        break
                    except Exception:
                        await asyncio.sleep(1 + 2 * attempt)
                if r is None:
                    # Jev unreachable: fall back to the vision model's probabilities for this pitch.
                    probs = dict(st.get("vision_model_probabilities") or
                                 {c: float(p) for c, p in st["arsenal_base_rates"].items()})
                    best = max(probs, key=probs.get)
                    return {"choice": best, "confidence": probs[best], "probabilities": probs,
                            "latency_s": time.perf_counter() - t0, "approx_input_tokens": 0,
                            "fallback": True}
                probs = {c: float(np.mean([r.choices[k].probabilities.get(c, 0) for k in qs]))
                         for c in self.criteria}
                best = max(probs, key=probs.get)
                return {"choice": best, "confidence": probs[best], "probabilities": probs,
                        "latency_s": time.perf_counter() - t0,
                        "approx_input_tokens": len(json.dumps(st)) // 4}

        async with AsyncTypeSafeClient() as client:
            return await asyncio.gather(*(one(client, s) for s in states))
