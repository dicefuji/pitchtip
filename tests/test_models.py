import numpy as np
import pandas as pd

from pitchtip.models.behavior import BehaviorModel, chrono_split
from pitchtip.predictor import causal_normalize
from pitchtip.tips import describe


def _synthetic(n_games=8, per_game=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_games):
        for i in range(per_game):
            fb = rng.random() < 0.6
            rows.append({"game_pk": g, "date": f"2025-05-{g + 1:02d}", "at_bat": i, "pitch_number": 1,
                         "y": "FASTBALL" if fb else "OFFSPEED",
                         # planted tell: hands higher before fastballs, plus a per-game camera offset
                         "set_hands_y_mean": rng.normal(0, 1) + (0.9 if fb else 0) + 3 * g,
                         "set_head_x_mean": rng.normal(0, 1)})
    df = pd.DataFrame(rows)
    hands = rng.normal(size=(len(df), 16)).astype(np.float32)
    emb = rng.normal(size=(len(df), 40)).astype(np.float32)
    return df, hands, emb


def test_causal_normalize_uses_only_past_pitches():
    df, hands, emb = _synthetic(n_games=1, per_game=12)
    out, _, _ = causal_normalize(df, hands, emb, warmup=3)
    x = df.set_hands_y_mean.to_numpy()
    # pitch 7 is compared with the mean of pitches 0..6 only
    assert np.isclose(out.set_hands_y_mean.iloc[7], x[7] - x[:7].mean())
    # changing a future pitch must not change an earlier normalized value
    df2 = df.copy()
    df2.loc[11, "set_hands_y_mean"] += 100
    out2, _, _ = causal_normalize(df2, hands, emb, warmup=3)
    assert np.isclose(out2.set_hands_y_mean.iloc[7], out.set_hands_y_mean.iloc[7])


def test_behavior_model_learns_planted_tell():
    df, hands, emb = _synthetic()
    df["set_hands_y_mean"] -= df.groupby("game_pk").set_hands_y_mean.transform("mean")
    tr, va, te = chrono_split(df)
    y = df.y
    m = BehaviorModel(n_emb=8, n_hands=4).fit(df[tr], hands[tr], emb[tr], y[tr])
    p = m.predict_proba(df[te], hands[te], emb[te])
    acc = (np.array(m.classes)[p.argmax(1)] == y[te].to_numpy()).mean()
    assert acc > 0.68  # base rate ~0.6
    assert set(m.expert_weights) == {"body", "glove", "linear"}


def test_chrono_split_never_straddles_games():
    df, _, _ = _synthetic()
    tr, va, te = chrono_split(df)
    for a, b in [(tr, va), (tr, te), (va, te)]:
        assert not set(df.game_pk[a]) & set(df.game_pk[b])
    assert df.date[tr].max() < df.date[te].min()


def test_describe_trajectory_features():
    assert describe("set_traj_glove_wrist_y_-3") == "glove-hand height 3 frames before leg lift (vs his set)"
    assert describe("set_traj_wrist_gap_+4") == "gap between wrists 4 frames into leg lift (vs his set)"
