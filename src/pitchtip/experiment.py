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


def run(key: str, mode: str = "type", variants=("probs", "experts", "probs+knn", "full"),
        use_emb: bool = True, jev: bool = True, max_n: int = 400,
        jev_models=("jev-latest", "jev-preview")) -> dict:
    df, hands, emb, y = load(key, mode)
    if not use_emb:
        emb = None
    tr, va, te = chrono_split(df)
    m = BehaviorModel(n_emb=32 if emb is not None else 0).fit(df[tr], hands[tr], _sub(emb, tr), y[tr])
    res = {"key": key, "mode": mode, "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()),
           "temperature": round(m.temperature, 3)}
    for name, mask in [("val", va), ("test", te)]:
        res[f"local_{name}"] = summarize(y[mask], m.predict_proba(df[mask], hands[mask], _sub(emb, mask)), m.classes)
    if not jev:
        return res
    tells = find_tells(df[tr], y[tr], top=6)
    best, best_key = None, (-1, 0)
    idx = np.flatnonzero(va)[:max_n]
    for jm in jev_models:
        for v in variants:
            jb = JevBehavior(m, df[tr], hands[tr], _sub(emb, tr), y[tr], tells, variant=v, jev_model=jm)
            out = asyncio.run(jb.predict(jb.evidence(df.iloc[idx], hands[idx], _sub(emb, idx))))
            P = np.array([[o["probabilities"].get(c, 0) for c in m.classes] for o in out])
            s = summarize(y.iloc[idx], P, m.classes)
            s["latency_p50_s"] = round(float(np.median([o["latency_s"] for o in out])), 3)
            res[f"jev_val_{jm}_{v}"] = s
            k = (s["accuracy"], -s["log_loss"])
            if k > best_key:
                best, best_key = (jm, v), k
    # Calibrate the chosen Jev config's probabilities on the validation games.
    from pitchtip.models.behavior import _temp
    from scipy.optimize import minimize_scalar
    jb = JevBehavior(m, df[tr], hands[tr], _sub(emb, tr), y[tr], tells, variant=best[1], jev_model=best[0])
    outv = asyncio.run(jb.predict(jb.evidence(df.iloc[idx], hands[idx], _sub(emb, idx))))
    Pv = np.array([[o["probabilities"].get(c, 0) for c in m.classes] for o in outv])
    yv = np.array([m.classes.index(c) for c in y.iloc[idx]])
    T = float(minimize_scalar(lambda T: -np.log(np.clip(_temp(Pv, T)[np.arange(len(yv)), yv], 1e-9, 1)).mean(),
                              bounds=(0.5, 50), method="bounded").x)
    res["jev_temperature"] = round(T, 2)
    idx = np.flatnonzero(te)[:max_n]
    out = asyncio.run(jb.predict(jb.evidence(df.iloc[idx], hands[idx], _sub(emb, idx))))
    P = _temp(np.array([[o["probabilities"].get(c, 0) for c in m.classes] for o in out]), T)
    s = summarize(y.iloc[idx], P, m.classes)
    toks = sum(o["approx_input_tokens"] for o in out)
    s |= {"variant": best[1], "jev_model": best[0], "latency_p50_s": round(float(np.median([o["latency_s"] for o in out])), 3),
          "usd_per_1k_pitches": round(toks / len(out) * 1000 * 0.042 / 1e6, 4)}
    res["jev_test"] = s
    return res


def rolling(key: str, mode: str = "fb", test_frac: float = 0.4, blocks: int = 5,
            variant: str = "probs+knn", jev_model: str = "jev-preview", jev: bool = True) -> dict:
    """Deployment-style evaluation: the last `test_frac` of games are split into `blocks`
    chronological blocks; each block is predicted by a model trained on all earlier games."""
    from pitchtip.models.behavior import _temp
    df, hands, emb, y = load(key, mode)
    games = df.drop_duplicates("game_pk").sort_values(["date", "game_pk"]).game_pk.tolist()
    test_games = games[int(len(games) * (1 - test_frac)):]
    P_loc, P_jev, Y, chosen = [], [], [], []
    for blk in np.array_split(np.array(test_games), min(blocks, len(test_games))):
        tr = df.game_pk.isin(games[: games.index(blk[0])]).to_numpy()
        te = df.game_pk.isin(blk).to_numpy()
        m = BehaviorModel(n_emb=32 if emb is not None else 0).fit(df[tr], hands[tr], _sub(emb, tr), y[tr])
        pl = m.predict_proba(df[te], hands[te], _sub(emb, te))
        P_loc.append(pl); Y.append(y[te])
        if jev:
            v = _pick_variant(df, hands, emb, y, tr, jev_model) if variant == "auto" else variant
            chosen.append(v)
            tells = find_tells(df[tr], y[tr], top=6)
            jb = JevBehavior(m, df[tr], hands[tr], _sub(emb, tr), y[tr], tells, variant=v, jev_model=jev_model)
            idx = np.flatnonzero(te)
            out = asyncio.run(jb.predict(jb.evidence(df.iloc[idx], hands[idx], _sub(emb, idx))))
            P_jev.append(np.array([[o["probabilities"].get(c, 0) for c in m.classes] for o in out]))
    yy = pd.concat(Y).reset_index(drop=True)
    classes = sorted(y.unique())
    res = {"key": key, "mode": mode, "n_test": len(yy), "test_games": len(test_games),
           "local": summarize(yy, np.vstack(P_loc), classes)}
    if jev:
        res["jev"] = summarize(yy, np.vstack(P_jev), classes) | {
            "variant": variant if variant != "auto" else "auto:" + ",".join(chosen), "jev_model": jev_model}
    return res


AUTO_VARIANTS = ("probs", "trust", "probs+knn")


def _pick_variant(df, hands, emb, y, tr_mask, jev_model) -> str:
    """Choose the Jev evidence variant on the most recent 25% of the training games
    (model fit on the earlier 75%), scoring accuracy plus ranking quality (AUC)."""
    from sklearn.metrics import roc_auc_score
    g = df[tr_mask].drop_duplicates("game_pk").sort_values(["date", "game_pk"]).game_pk.tolist()
    if len(g) < 4:
        return "probs"
    cut = set(g[int(len(g) * 0.75):])
    inner_tr = tr_mask & ~df.game_pk.isin(cut).to_numpy()
    inner_va = tr_mask & df.game_pk.isin(cut).to_numpy()
    m = BehaviorModel(n_emb=32 if emb is not None else 0).fit(df[inner_tr], hands[inner_tr], _sub(emb, inner_tr), y[inner_tr])
    tells = find_tells(df[inner_tr], y[inner_tr], top=6)
    idx = np.flatnonzero(inner_va)
    yv = y.iloc[idx].to_numpy()
    best, best_score = "probs", -1.0
    for v in AUTO_VARIANTS:
        jb = JevBehavior(m, df[inner_tr], hands[inner_tr], _sub(emb, inner_tr), y[inner_tr], tells, variant=v, jev_model=jev_model)
        out = asyncio.run(jb.predict(jb.evidence(df.iloc[idx], hands[idx], _sub(emb, idx))))
        P = np.array([[o["probabilities"].get(c, 0) for c in m.classes] for o in out])
        acc = float((np.array(m.classes)[P.argmax(1)] == yv).mean())
        try:
            auc = float(np.mean([roc_auc_score(yv == c, P[:, j]) for j, c in enumerate(m.classes) if 0 < (yv == c).sum() < len(yv)]))
        except ValueError:
            auc = 0.5
        score = acc + 0.5 * (auc - 0.5)
        if score > best_score:
            best, best_score = v, score
    return best
