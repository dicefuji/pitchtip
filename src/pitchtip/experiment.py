"""Behavior-only experiments: local model vs Jev variants on a chronological split."""
from __future__ import annotations

import asyncio
import json

import numpy as np
import pandas as pd

from pitchtip import dataset
from pitchtip.labels import make_labels
from pitchtip.models import local
from pitchtip.models.behavior import BehaviorModel, chrono_split
from pitchtip.models.jev import JevBehavior
from pitchtip.tips import find_tells
from pitchtip.vision import embed


def load(key: str, mode: str):
    df, hands = dataset.load_features(key)
    try:
        emb = embed.load(key, df=df)
    except FileNotFoundError:
        emb = None
    y = make_labels(df.pitch_type, mode)
    keep = (y != "OTHER").to_numpy().copy()
    df, hands, y = df[keep].reset_index(drop=True), hands[keep], y[keep].reset_index(drop=True)
    emb = emb[keep] if emb is not None else None
    return df, hands, emb, y


def _sub(a, m):
    return None if a is None else a[m]


def summarize(y: pd.Series, proba: np.ndarray, classes: list[str]) -> dict:
    s = local.score(y.reset_index(drop=True), proba, classes)
    return {k: (round(v, 4) if isinstance(v, float) else
                {a: round(b, 3) for a, b in v.items()} if isinstance(v, dict) else v)
            for k, v in s.items()}


def run(key: str, mode: str = "type", variants=("probs", "knn", "probs+knn", "full"),
        use_emb: bool = True, jev: bool = True, max_n: int = 400) -> dict:
    df, hands, emb, y = load(key, mode)
    if not use_emb:
        emb = None
    tr, va, te = chrono_split(df)
    m = BehaviorModel(n_emb=24 if emb is not None else 0).fit(df[tr], hands[tr], _sub(emb, tr), y[tr])
    res = {"key": key, "mode": mode, "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()),
           "temperature": round(m.temperature, 3)}
    for name, mask in [("val", va), ("test", te)]:
        res[f"local_{name}"] = summarize(y[mask], m.predict_proba(df[mask], hands[mask], _sub(emb, mask)), m.classes)
    if not jev:
        return res
    tells = find_tells(df[tr], y[tr], top=6)
    best, best_acc = None, -1
    for v in variants:
        jb = JevBehavior(m, df[tr], hands[tr], _sub(emb, tr), y[tr], tells, variant=v)
        idx = np.flatnonzero(va)[:max_n]
        out = asyncio.run(jb.predict(jb.evidence(df.iloc[idx], hands[idx], _sub(emb, idx))))
        P = np.array([[o["probabilities"].get(c, 0) for c in m.classes] for o in out])
        s = summarize(y.iloc[idx], P, m.classes)
        s["latency_p50_s"] = round(float(np.median([o["latency_s"] for o in out])), 3)
        res[f"jev_val_{v}"] = s
        if s["accuracy"] > best_acc:
            best, best_acc = v, s["accuracy"]
    jb = JevBehavior(m, df[tr], hands[tr], _sub(emb, tr), y[tr], tells, variant=best)
    idx = np.flatnonzero(te)[:max_n]
    out = asyncio.run(jb.predict(jb.evidence(df.iloc[idx], hands[idx], _sub(emb, idx))))
    P = np.array([[o["probabilities"].get(c, 0) for c in m.classes] for o in out])
    s = summarize(y.iloc[idx], P, m.classes)
    toks = sum(o["approx_input_tokens"] for o in out)
    s |= {"variant": best, "latency_p50_s": round(float(np.median([o["latency_s"] for o in out])), 3),
          "usd_per_1k_pitches": round(toks / len(out) * 1000 * 0.042 / 1e6, 4)}
    res["jev_test"] = s
    return res
