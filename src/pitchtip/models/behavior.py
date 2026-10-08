"""Behavior-only pitch model: pose features + glove embeddings, nothing about the count.

Stacked per-view experts, because pitchers tip in different ways:
  * body   - gradient-boosted trees on pose / trajectory features
  * glove  - ridge-logistic on the glove-region DINOv2 embedding (PCA) + hand patch PCA
  * linear - ridge-logistic on pose features
A meta logistic regression learns, per pitcher, how much to trust each expert from
game-grouped out-of-fold predictions. Output is temperature-calibrated.
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

EXPERTS = ("body", "glove", "linear")


def _hgb():
    return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.04, max_leaf_nodes=15,
                                          min_samples_leaf=12, l2_regularization=1.0, random_state=0)


def _lr(C=0.05):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))


@dataclass
class BehaviorModel:
    n_emb: int = 32
    n_hands: int = 8
    cols: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    temperature: float = 1.0

    # ---- views -------------------------------------------------------------
    def _pose(self, df):
        P = df[self.cols].to_numpy(np.float32)
        return np.where(np.isfinite(P), P, 0)

    def _glove(self, hands, emb, fit=False):
        parts = []
        if emb is not None and self.n_emb:
            if fit:
                self.pca_e = PCA(min(self.n_emb, len(emb) - 1), random_state=0).fit(emb)
            parts.append(self.pca_e.transform(emb))
        if hands is not None and self.n_hands:
            if fit:
                self.pca_h = PCA(self.n_hands, random_state=0).fit(hands)
            parts.append(self.pca_h.transform(hands))
        return np.hstack(parts) if parts else None

    def _views(self, df, hands, emb, fit=False):
        """Concatenated feature space (used for nearest-neighbour evidence)."""
        g = self._glove(hands, emb, fit)
        return self._pose(df) if g is None else np.hstack([self._pose(df), g])

    # ---- experts -----------------------------------------------------------
    def _fit_experts(self, df, hands, emb, y):
        P, G = self._pose(df), self._glove(hands, emb, fit=True)
        self.experts = {"body": _hgb().fit(P, y), "linear": _lr().fit(P, y)}
        if G is not None:
            self.experts["glove"] = _lr(0.02).fit(G, y)

    def expert_probas(self, df, hands, emb) -> dict[str, np.ndarray]:
        P, G = self._pose(df), self._glove(hands, emb)
        out = {}
        for name, m in self.experts.items():
            X = G if name == "glove" else P
            p = m.predict_proba(X)
            full = np.zeros((len(df), len(self.classes)))
            for j, c in enumerate(m.classes_):
                full[:, self.classes.index(c)] = p[:, j]
            out[name] = full
        return out

    def _stack_X(self, ep: dict[str, np.ndarray]) -> np.ndarray:
        return np.hstack([np.log(np.clip(ep[n], 1e-4, 1)) for n in EXPERTS if n in ep])

    # ---- fit / predict -----------------------------------------------------
    def fit(self, df, hands, emb, y, stack: bool = True):
        self.cols = feature_columns(df)
        self.classes = sorted(y.unique())
        self.meta = None
        if stack and df.game_pk.nunique() >= 3:
            oof = self._oof_experts(df, hands, emb, y)
            Xs = self._stack_X(oof)
            self.meta = LogisticRegression(C=1.0, max_iter=2000).fit(Xs, y)
            pm = self._meta_proba(Xs)
            yi = np.array([self.classes.index(c) for c in y])
            nll = lambda T: -np.log(np.clip(_temp(pm, T)[np.arange(len(yi)), yi], 1e-9, 1)).mean()
            self.temperature = float(minimize_scalar(nll, bounds=(0.3, 5), method="bounded").x)
            self.expert_weights = dict(zip([n for n in EXPERTS if n in oof],
                                           np.abs(self.meta.coef_).reshape(len(self.meta.coef_), -1, len(self.classes)).mean((0, 2)).round(3)))
        self._fit_experts(df, hands, emb, y)
        return self

    def _oof_experts(self, df, hands, emb, y):
        g = df.game_pk.to_numpy()
        oof = {}
        for tr, te in GroupKFold(min(5, len(np.unique(g)))).split(df, y, g):
            m = BehaviorModel(self.n_emb, self.n_hands)
            m.cols, m.classes = self.cols, self.classes
            m._fit_experts(df.iloc[tr], _take(hands, tr), _take(emb, tr), y.iloc[tr])
            for name, p in m.expert_probas(df.iloc[te], _take(hands, te), _take(emb, te)).items():
                oof.setdefault(name, np.zeros((len(df), len(self.classes))))[te] = p
        return oof

    def _meta_proba(self, Xs):
        p = self.meta.predict_proba(Xs)
        full = np.zeros((len(Xs), len(self.classes)))
        for j, c in enumerate(self.meta.classes_):
            full[:, self.classes.index(c)] = p[:, j]
        return full

    def predict_proba(self, df, hands, emb):
        ep = self.expert_probas(df, hands, emb)
        if self.meta is None:
            return np.mean(list(ep.values()), axis=0)
        return _temp(self._meta_proba(self._stack_X(ep)), self.temperature)

    def _oof(self, df, hands, emb, y):
        """Game-grouped out-of-fold probabilities of the full stacked model."""
        oof = np.zeros((len(df), len(self.classes)))
        g = df.game_pk.to_numpy()
        for tr, te in GroupKFold(min(5, len(np.unique(g)))).split(df, y, g):
            m = BehaviorModel(self.n_emb, self.n_hands).fit(df.iloc[tr], _take(hands, tr), _take(emb, tr), y.iloc[tr])
            p = m.predict_proba(df.iloc[te], _take(hands, te), _take(emb, te))
            for j, c in enumerate(m.classes):
                oof[te, self.classes.index(c)] = p[:, j]
        return oof


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
