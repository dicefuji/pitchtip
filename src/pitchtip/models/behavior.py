"""Behavior-only pitch model: pose features + glove embeddings, nothing about the count.

Ensemble of gradient-boosted trees (pose features) and a ridge-logistic model (glove
embedding PCA + pose), temperature-calibrated on game-grouped out-of-fold predictions so
its probabilities are honest inputs for Jev.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pitchtip.dataset import feature_columns


def _hgb():
    return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.04, max_leaf_nodes=15,
                                          min_samples_leaf=12, l2_regularization=1.0, random_state=0)


@dataclass
class BehaviorModel:
    n_emb: int = 24
    n_hands: int = 8
    w_tree: float = 0.5
    cols: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    temperature: float = 1.0

    def _views(self, df, hands, emb, fit=False):
        P = df[self.cols].to_numpy(np.float32)
        P = np.where(np.isfinite(P), P, 0)
        parts = [P]
        if hands is not None and self.n_hands:
            if fit:
                self.pca_h = PCA(self.n_hands, random_state=0).fit(hands)
            parts.append(self.pca_h.transform(hands))
        if emb is not None and self.n_emb:
            if fit:
                self.pca_e = PCA(self.n_emb, random_state=0).fit(emb)
            parts.append(self.pca_e.transform(emb))
        return np.hstack(parts)

    def _raw(self, X):
        pt = self.tree.predict_proba(X)
        pl = self.lin.predict_proba(X)
        return self.w_tree * pt + (1 - self.w_tree) * pl

    def fit(self, df, hands, emb, y, calibrate: bool = True):
        self.cols = feature_columns(df)
        self.classes = sorted(y.unique())
        if calibrate and df.game_pk.nunique() >= 3:
            oof = self._oof(df, hands, emb, y)
            yi = np.array([self.classes.index(c) for c in y])
            nll = lambda T: -np.log(np.clip(_temp(oof, T)[np.arange(len(yi)), yi], 1e-9, 1)).mean()
            self.temperature = float(minimize_scalar(nll, bounds=(0.3, 5), method="bounded").x)
        X = self._views(df, hands, emb, fit=True)
        self.tree = _hgb().fit(X, y)
        self.lin = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)).fit(X, y)
        return self

    def _oof(self, df, hands, emb, y):
        oof = np.zeros((len(df), len(self.classes)))
        g = df.game_pk.to_numpy()
        for tr, te in GroupKFold(min(5, len(np.unique(g)))).split(df, y, g):
            m = BehaviorModel(self.n_emb, self.n_hands, self.w_tree).fit(
                df.iloc[tr], _take(hands, tr), _take(emb, tr), y.iloc[tr], calibrate=False)
            p = m.predict_proba(df.iloc[te], _take(hands, te), _take(emb, te))
            for j, c in enumerate(m.classes):
                oof[te, self.classes.index(c)] = p[:, j]
        return oof

    def predict_proba(self, df, hands, emb):
        p = self._raw(self._views(df, hands, emb))
        return _temp(p, self.temperature)


def _take(a, idx):
    return None if a is None else a[idx]


def _temp(p, T):
    lp = np.log(np.clip(p, 1e-9, 1)) / T
    lp -= lp.max(1, keepdims=True)
    e = np.exp(lp)
    return e / e.sum(1, keepdims=True)


def chrono_split(df: pd.DataFrame, fracs=(0.6, 0.2)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train / validation / test masks by game date (no game straddles a split)."""
    games = df.drop_duplicates("game_pk").sort_values(["date", "game_pk"]).game_pk.tolist()
    n = len(games)
    a, b = int(n * fracs[0]), int(n * (fracs[0] + fracs[1]))
    tr, va = set(games[:a]), set(games[a:b])
    g = df.game_pk
    return g.isin(tr).to_numpy(), g.isin(va).to_numpy(), (~g.isin(tr | va)).to_numpy()
