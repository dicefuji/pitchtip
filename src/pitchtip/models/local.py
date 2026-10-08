"""Local per-pitcher decision model: gradient-boosted trees on pose features.

Free, ~1ms per prediction. Evaluated with game-grouped CV so no pitch is ever predicted
by a model that saw other pitches from the same game (same lighting, same camera).
"""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from pitchtip import config
from pitchtip.config import slug
from pitchtip.dataset import feature_columns


@dataclass
class TipModel:
    clf: HistGradientBoostingClassifier
    pca: PCA | None
    columns: list[str]
    classes: list[str]

    def matrix(self, df: pd.DataFrame, hands: np.ndarray | None) -> np.ndarray:
        X = df[self.columns].to_numpy(np.float32)
        if self.pca is not None and hands is not None:
            X = np.hstack([X, self.pca.transform(hands)])
        return X

    def predict_proba(self, df: pd.DataFrame, hands: np.ndarray | None) -> np.ndarray:
        return self.clf.predict_proba(self.matrix(df, hands))


def fit(df: pd.DataFrame, hands: np.ndarray | None, y: pd.Series, n_pca: int = 8) -> TipModel:
    cols = feature_columns(df)
    pca = PCA(n_pca).fit(hands) if hands is not None and n_pca else None
    m = TipModel(None, pca, cols, sorted(y.unique()))
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=10, l2_regularization=1.0,
                                         class_weight=None, random_state=0)
    clf.fit(m.matrix(df, hands), y)
    m.clf = clf
    return m


def cross_val(df: pd.DataFrame, hands: np.ndarray | None, y: pd.Series, n_splits: int = 5,
              n_pca: int = 8) -> tuple[np.ndarray, list[str]]:
    """Out-of-fold class probabilities, grouped by game."""
    classes = sorted(y.unique())
    oof = np.zeros((len(df), len(classes)))
    groups = df.game_pk.to_numpy()
    k = min(n_splits, len(np.unique(groups)))
    for tr, te in GroupKFold(k).split(df, y, groups):
        m = fit(df.iloc[tr], hands[tr] if hands is not None else None, y.iloc[tr], n_pca)
        p = m.predict_proba(df.iloc[te], hands[te] if hands is not None else None)
        for j, c in enumerate(m.clf.classes_):
            oof[te, classes.index(c)] = p[:, j]
    return oof, classes


def score(y: pd.Series, proba: np.ndarray, classes: list[str]) -> dict:
    prior = y.value_counts(normalize=True).reindex(classes).fillna(0).to_numpy()
    pred = np.array(classes)[proba.argmax(1)]
    return {
        "n": len(y),
        "accuracy": accuracy_score(y, pred),
        "base_rate": float(prior.max()),
        "log_loss": log_loss(y, np.clip(proba, 1e-6, 1), labels=classes),
        "prior_log_loss": log_loss(y, np.tile(prior, (len(y), 1)), labels=classes),
        # Tips are usable even when weak if confident calls are reliable: per pitch, how
        # often is it right in the 10% of pitches where the model likes it most?
        "auc_one_vs_rest": {c: float(roc_auc_score(y == c, proba[:, j])) for j, c in enumerate(classes)},
        "top10pct_precision": {c: float((y[proba[:, j] >= np.quantile(proba[:, j], 0.9)] == c).mean())
                               for j, c in enumerate(classes)},
        "base_rates": dict(zip(classes, prior.round(3).tolist())),
        # How a tip is used in practice: only act when the read is clear.
        "selective_accuracy": {f"top{int(c * 100)}pct": float(
            (pred[np.argsort(-proba.max(1))[: max(1, int(len(y) * c))]]
             == np.asarray(y)[np.argsort(-proba.max(1))[: max(1, int(len(y) * c))]]).mean())
            for c in (0.25, 0.5)},
    }


def permutation_pvalue(df, hands, y, observed_acc: float, n: int = 30, n_pca: int = 8,
                       seed: int = 0) -> float:
    """Shuffle labels within each game and re-run CV; p = P(acc_null >= observed)."""
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n):
        ys = y.copy()
        for _, idx in df.groupby("game_pk").indices.items():
            ys.iloc[idx] = rng.permutation(ys.iloc[idx].to_numpy())
        oof, classes = cross_val(df, hands, ys, n_pca=n_pca)
        null.append(accuracy_score(ys, np.array(classes)[oof.argmax(1)]))
    return (1 + sum(a >= observed_acc for a in null)) / (n + 1)


def save(m: TipModel, name: str, mode: str) -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(m, config.MODELS_DIR / f"{slug(name)}_{mode}.joblib")


def load(name: str, mode: str) -> TipModel:
    return joblib.load(config.MODELS_DIR / f"{slug(name)}_{mode}.joblib")
