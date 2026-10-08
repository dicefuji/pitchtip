"""pitchtip command line."""
from __future__ import annotations

import asyncio
import json
from typing import Optional

import numpy as np
import pandas as pd
import typer

from pitchtip import dataset
from pitchtip.labels import make_labels

app = typer.Typer(add_completion=False, help="Find pitcher tips from broadcast video.")


@app.command()
def fetch(pitcher: str, season: list[int] = typer.Option(..., help="Season(s), repeatable"),
          game_type: list[str] = typer.Option(["R"], help="R=regular, F/D/L/W=postseason rounds"),
          max_games: Optional[int] = None, no_download: bool = False, workers: int = 6):
    """Pull pitch labels (MLB Stats API) and download clips (Baseball Savant)."""
    df = dataset.fetch(pitcher, season, max_games, game_type, not no_download, workers)
    typer.echo(f"{len(df)} pitches over {df.game_pk.nunique()} games")
    typer.echo(df.pitch_type.value_counts().to_string())


@app.command()
def scan(pitchers: list[str] = typer.Argument(..., help='"Name:season[:max_games]" entries'),
         game_type: list[str] = typer.Option(["R"]), workers: int = 8, embed: bool = True):
    """Download + pose-extract (+ glove-embed) many pitcher-seasons."""
    from pitchtip.vision import embed as emb
    for spec in pitchers:
        name, season, *mg = spec.split(":")
        try:
            df = dataset.scan(name, [int(season)], int(mg[0]) if mg else None, game_type, workers)
            key = dataset.dataset_key(df.pitcher_name.iloc[0], [int(season)])
            typer.echo(f"{key}: features for {len(df)} pitches", nl=True)
            if embed:
                emb.build(key)
        except Exception as e:  # keep the batch going
            typer.echo(f"{spec}: FAILED {type(e).__name__}: {e}")


@app.command()
def embed(keys: list[str]):
    """DINOv2 glove-region embeddings for extracted datasets."""
    from pitchtip.vision import embed as emb
    for k in keys:
        a = emb.build(k)
        typer.echo(f"{k}: {a.shape}")


@app.command()
def extract(pitcher: str, overwrite: bool = False):
    """Run pose + phase segmentation + features over downloaded clips."""
    df = dataset.extract(pitcher, overwrite)
    typer.echo(f"features for {len(df)} pitches")


def _xy(pitcher: str, mode: str, per_game: bool, situation: str = "all"):
    df, hands = dataset.load_features(pitcher, per_game)
    y = make_labels(df.pitch_type, mode)
    keep = (y != "OTHER").to_numpy().copy()
    if situation != "all":
        keep &= df[situation].astype(bool).to_numpy()
    return df[keep].reset_index(drop=True), hands[keep], y[keep].reset_index(drop=True)


@app.command()
def tips(pitcher: str, mode: str = "type", top: int = 10, per_game: bool = True,
         situation: str = typer.Option("all", help="all | men_on | risp")):
    """List the strongest single-feature tells."""
    from pitchtip.tips import find_tells
    df, _, y = _xy(pitcher, mode, per_game, situation)
    t = find_tells(df, y, top)
    for line in t.tell:
        typer.echo(f"• {line}")


@app.command()
def eval(pitcher: str, mode: str = typer.Option("type", help="type | fb (fastball vs offspeed)"),
         model: str = typer.Option("local", help="local | jev | fusion (local vision model -> Jev)"), per_game: bool = True,
         permutations: int = 20, no_hands: bool = False, max_jev: int = 200,
         situation: str = typer.Option("all", help="all | men_on | risp"),
         split: str = typer.Option("cv", help="cv (game-grouped folds) | chrono (train early games, test late)")):
    """Evaluation vs. the pitcher's base rate."""
    from pitchtip.models import local
    df, hands, y = _xy(pitcher, mode, per_game, situation)
    h = None if no_hands else hands
    if model == "local" and split == "chrono":
        games = sorted(df.game_pk.unique())
        tr = (df.game_pk < games[int(len(games) * 0.7)]).to_numpy()
        m = local.fit(df[tr], h[tr] if h is not None else None, y[tr])
        p = m.predict_proba(df[~tr], h[~tr] if h is not None else None)
        s = local.score(y[~tr].reset_index(drop=True), p, list(m.clf.classes_))
    elif model == "local":
        oof, classes = local.cross_val(df, h, y)
        s = local.score(y, oof, classes)
        acc_pred = np.array(classes)[oof.argmax(1)]
        s["permutation_p"] = local.permutation_pvalue(df, h, y, s["accuracy"], permutations) if permutations else None
        s["per_class_recall"] = {c: float((acc_pred[y == c] == c).mean()) for c in classes}
    else:
        from pitchtip.models.jev import JevDecider
        from pitchtip.tips import find_tells
        games = sorted(df.game_pk.unique())
        cut = games[int(len(games) * 0.8)]
        tr, te = df.game_pk < cut, df.game_pk >= cut
        tells = find_tells(df[tr], y[tr], top=12)
        jd = JevDecider(df[tr], y[tr], tells)
        test = df[te].head(max_jev)
        vision = None
        if model == "fusion":
            trm, tem = tr.to_numpy(), np.flatnonzero(te.to_numpy())[:max_jev]
            lm = local.fit(df[tr], h[trm] if h is not None else None, y[tr])
            pv = lm.predict_proba(test, h[tem] if h is not None else None)
            vision = [dict(zip(lm.clf.classes_, row)) for row in pv]
        res = asyncio.run(jd.predict_many(test, pitcher, vision=vision))
        classes = sorted(y.unique())
        proba = np.array([[r["probabilities"].get(c, 0) for c in classes] for r in res])
        s = local.score(y[te].head(max_jev), proba, classes)
        lat = [r["latency_s"] for r in res]
        toks = sum(r["approx_input_tokens"] for r in res)
        s |= {"latency_p50_s": float(np.median(lat)), "approx_cost_usd": toks * 0.042 / 1e6,
              "approx_tokens_per_pitch": toks / len(res)}
    typer.echo(json.dumps(s, indent=2, default=float))


@app.command()
def tune(keys: list[str], mode: str = "type", no_jev: bool = False, no_emb: bool = False,
         variants: str = "probs,experts,probs+knn,full", out: Optional[str] = None):
    """Behavior-only: local model vs Jev variants (chosen on val games, scored on test games)."""
    from pitchtip import experiment
    allres = []
    for k in keys:
        r = experiment.run(k, mode, tuple(variants.split(",")), not no_emb, not no_jev)
        allres.append(r)
        typer.echo(json.dumps(r, indent=1, default=float))
    if out:
        with open(out, "a") as f:
            for r in allres:
                f.write(json.dumps(r, default=float) + "\n")


@app.command()
def demo(key: str, game: str = typer.Option("last", help="gamePk or 'last'"), n: int = 30,
         mode: str = "type", jev: bool = True, from_video: bool = False,
         sheet: str = "data/demo.jpg"):
    """Replay one held-out game: call each pitch from video behavior only, in order."""
    import cv2
    from pitchtip import config
    from pitchtip.config import slug
    from pitchtip.predictor import Predictor
    from pitchtip.vision import pose
    pf = pd.read_parquet(config.FEATURES_DIR / f"{slug(key)}.parquet")
    games = pf.drop_duplicates("game_pk").sort_values("date")
    gpk = int(games.game_pk.iloc[-1]) if game == "last" else int(game)
    gdate = games.set_index("game_pk").date[gpk]
    prior = set(games[games.date < gdate].game_pk)
    pr = Predictor(key, mode, use_jev=jev).fit(prior)
    rows = pf[pf.game_pk == gpk].sort_values(["at_bat", "pitch_number"]).head(n)
    hits, tiles = 0, []
    typer.echo(f"{key} game {gpk} ({gdate}); trained on {len(prior)} earlier games")
    for r in rows.itertuples():
        npz = config.POSE_DIR / slug(r.pitcher_name) / f"{r.play_id}.npz"
        clip = config.CLIPS_DIR / slug(r.pitcher_name) / f"{r.play_id}.mp4"
        cp = pose.extract(clip, pr.hand) if from_video or not npz.exists() else pose.ClipPose.load(npz)
        feats = pr.features_from_pose(cp)
        if feats is None:
            continue
        out = pr.predict_features(*feats)
        truth = make_labels(pd.Series([r.pitch_type]), mode).iloc[0]
        ok = out["call"] == truth
        hits += ok
        top = sorted((out.get("jev") or out["vision"]).items(), key=lambda kv: -kv[1])[:3]
        typer.echo(f"  inn {r.inning} AB {r.at_bat:>2} p{r.pitch_number}:  call {out['call']:>8} "
                   f"{'STRONG' if out['strong'] else '      '} trust {out['trust']:.0%}  actual {truth:>8}  {'✓' if ok else '✗'}   "
                   + " ".join(f"{k}:{v:.0%}" for k, v in top))
        if len(cp.glove):
            g = cv2.resize(cp.glove[max(0, len(cp.glove) - 8)], (160, 160))
            cv2.putText(g, f"call {out['call']}", (4, 18), 0, 0.5, (0, 255, 255), 1)
            cv2.putText(g, f"real {truth}", (4, 152), 0, 0.5, (0, 255, 0) if ok else (0, 0, 255), 1)
            tiles.append(g)
    k = len(rows)
    typer.echo(f"accuracy {hits}/{k} = {hits / max(k, 1):.0%}")
    if tiles:
        while len(tiles) % 6:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(sheet, np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]))
        typer.echo(sheet)


@app.command()
def leaderboard(mode: str = "fb", out: str = "data/leaderboard.csv"):
    """Rank every scanned pitcher-season by how predictable their pitches are from behavior."""
    from pitchtip import config
    from pitchtip.experiment import load
    from pitchtip.models.behavior import BehaviorModel
    from pitchtip.models import local
    from pitchtip.tips import find_tells
    rows = []
    for f in sorted(config.FEATURES_DIR.glob("*_emb.npy")):
        key = f.name[: -len("_emb.npy")]
        pq = pd.read_parquet(config.FEATURES_DIR / f"{key}.parquet")
        key = dataset.dataset_key(pq.pitcher_name.iloc[0], sorted({int(d[:4]) for d in pq.date}))
        try:
            df, hands, emb, y = load(key, mode)
            if df.game_pk.nunique() < 4 or y.nunique() < 2:
                continue
            m = BehaviorModel()
            m.cols, m.classes = dataset.feature_columns(df), sorted(y.unique())
            oof = m._oof(df, hands, emb, y)
            w = BehaviorModel().fit(df, hands, emb, y).expert_weights
            s = local.score(y, oof, m.classes)
            t = find_tells(df, y, top=3)
            auc = float(np.mean(list(s["auc_one_vs_rest"].values())))
            rows.append({"key": key, "n": s["n"], "games": df.game_pk.nunique(), "accuracy": round(s["accuracy"], 3),
                         "base_rate": round(s["base_rate"], 3), "auc": round(auc, 3),
                         "top25_acc": round(s["selective_accuracy"]["top25pct"], 3),
                         "expert_weights": w,
                         "top_tell": t.tell.iloc[0] if len(t) else ""})
            typer.echo(f"{key:32s} n={s['n']:5d} auc={auc:.3f} acc={s['accuracy']:.3f} base={s['base_rate']:.3f} "
                       f"top25%={s['selective_accuracy']['top25pct']:.3f}")
        except Exception as e:
            typer.echo(f"{key}: {e}")
    lb = pd.DataFrame(rows).sort_values("auc", ascending=False)
    lb.to_csv(out, index=False)
    typer.echo(lb.to_string(index=False))


@app.command()
def rolling(keys: list[str], mode: str = "fb", variant: str = "auto",
            jev_model: str = "jev-preview", no_jev: bool = False, out: str = "data/rolling_results.jsonl"):
    """Deployment-style eval: predict each later block of games from all earlier games."""
    from pitchtip import experiment
    for k in keys:
        r = experiment.rolling(k, mode, variant=variant, jev_model=jev_model, jev=not no_jev)
        with open(out, "a") as f:
            f.write(json.dumps(r, default=float) + "\n")
        for name in ("local", "jev"):
            if name in r:
                s = r[name]
                typer.echo(f"{k:24s} {name:5s} n={s['n']:4d} acc={s['accuracy']:.3f} base={s['base_rate']:.3f} "
                           f"auc={np.mean(list(s['auc_one_vs_rest'].values())):.3f} "
                           f"top25={s['selective_accuracy']['top25pct']:.3f} top50={s['selective_accuracy']['top50pct']:.3f}"
                           + (f"  gated25={r['jev_gated']['top25pct']:.3f} gated50={r['jev_gated']['top50pct']:.3f}"
                              if name == "jev" and "jev_gated" in r else ""))


@app.command()
def patterns(mode: str = "fb", q: float = 0.01, out: str = "data/patterns.csv"):
    """League-wide view: which tell families recur across pitchers (significant at FDR q)."""
    import re
    from pitchtip import config
    from pitchtip.tips import describe, find_tells
    rows = []
    for f in sorted(config.FEATURES_DIR.glob("*_emb.npy")):
        pq = pd.read_parquet(config.FEATURES_DIR / f"{f.name[:-len('_emb.npy')]}.parquet")
        key = dataset.dataset_key(pq.pitcher_name.iloc[0], sorted({int(d[:4]) for d in pq.date}))
        df, _ = dataset.load_features(key)
        y = make_labels(df.pitch_type, mode)
        keep = (y != "OTHER").to_numpy()
        t = find_tells(df[keep].reset_index(drop=True), y[keep].reset_index(drop=True), top=40)
        t = t[t.q < q]
        for r in t.itertuples():
            fam = re.sub(r"^(set_traj_|set_|early_)", "", r.feature)
            fam = re.sub(r"_(mean|std|delta|[+-]\d+)$", "", fam)
            when = ("trajectory" if "traj" in r.feature else "early lift" if r.feature.startswith("early_")
                    else "set")
            rows.append({"pitcher": key, "family": fam, "when": when, "feature": r.feature,
                         "auc": r.auc, "q": r.q, "pitch": r.pitch, "tell": r.tell})
    t = pd.DataFrame(rows)
    t.to_csv(out, index=False)
    n_p = t.pitcher.nunique()
    agg = (t.groupby("family").agg(pitchers=("pitcher", "nunique"),
                                   mean_strength=("auc", lambda a: float(np.mean(np.abs(a - 0.5)) + 0.5)))
           .sort_values(["pitchers", "mean_strength"], ascending=False))
    typer.echo(f"{n_p} pitcher-seasons with ≥1 significant tell (q<{q})\n")
    typer.echo(agg.to_string())
    typer.echo("\nWhen tells appear:\n" + t.groupby("when").pitcher.nunique().to_string())
    typer.echo("\nStrongest tell per pitcher:")
    for k, g in t.sort_values("q").groupby("pitcher"):
        typer.echo(f"  {k:28s} {g.iloc[0].tell}")


@app.command()
def report(path: str = "data/rolling_results.jsonl"):
    """Latest rolling (deployment-style) result per pitcher and mode, Jev vs local vs base."""
    rows = {}
    for line in open(path):
        r = json.loads(line)
        rows[(r["key"], r["mode"])] = r
    out = []
    for (k, mode), r in sorted(rows.items(), key=lambda kv: -kv[1].get("jev", kv[1]["local"])["accuracy"]):
        j, l = r.get("jev"), r["local"]
        out.append({"pitcher": k, "mode": mode, "n": l["n"], "base": l["base_rate"], "local": l["accuracy"],
                    "jev": j["accuracy"] if j else None,
                    "jev_lift": round(j["accuracy"] - l["base_rate"], 3) if j else None,
                    "jev_top50": j["selective_accuracy"]["top50pct"] if j else None,
                    "variant": j.get("variant") if j else None})
    typer.echo(pd.DataFrame(out).to_string(index=False))


@app.command()
def ytest(video: str, key: str, game: int, out: str = typer.Option(..., help="output dir"),
          mode: str = "type", variant: str = "probs+knn", label: str = "", set_only: bool = False,
          innings: Optional[list[int]] = typer.Option(None, help="restrict the feed to the innings shown in the video"),
          train_all: bool = typer.Option(False, help="train on every other game (incl. later ones), never the test game")):
    """Test on arbitrary broadcast video (e.g. YouTube) and score against the MLB feed."""
    from pathlib import Path
    from pitchtip import config, youtube
    from pitchtip.config import slug
    from pitchtip.labels import make_labels
    from pitchtip.predictor import Predictor
    pf = pd.read_parquet(config.FEATURES_DIR / f"{slug(key)}.parquet")
    gdate = str(pf[pf.game_pk == game].date.iloc[0]) if (pf.game_pk == game).any() else None
    if gdate is None:
        import requests
        gdate = requests.get(f"https://statsapi.mlb.com/api/v1.1/game/{game}/feed/live").json()["gameData"]["datetime"]["officialDate"]
    prior = set(pf[pf.game_pk != game].game_pk) if train_all else set(pf[pf.date < gdate].game_pk)
    pr = Predictor(key, mode, use_jev=True, variant=variant, set_only=set_only).fit(prior)
    typer.echo(f"{key}: trained on {len(prior)} games before {gdate}; classes {pr.model.classes}")
    od = Path(out)
    calls = youtube.run(video, pr, od)
    feed = youtube.feed_pitches(game, int(pf.pitcher_id.iloc[0]), innings)
    feed["label"] = make_labels(feed.pitch_type, mode) if mode == "fb" else feed.pitch_type.where(
        feed.pitch_type.isin(pr.model.classes), "OTHER")
    off, _ = youtube.align(np.array([c["t"] for c, _, _ in calls]), feed.t.to_numpy())
    match = youtube.align_dp(np.array([c["t"] for c, _, _ in calls]), feed.t.to_numpy())
    rows = []
    for i, (c, cp, ph) in enumerate(calls):
        j = match.get(i)
        truth = feed.label.iloc[j] if j is not None else None
        frame = np.load(od / f"call_{i:03d}_frame.npy")
        youtube.render(frame, cp, ph, c | {"jev": c["jev"]}, truth, od / f"call_{i:03d}.jpg",
                       f"{label or key}  t={c['t']:.0f}s" + (f"  {feed.batter.iloc[j]} (count after: {feed['count'].iloc[j]})" if j is not None else ""))
        rows.append({"i": i, "video_t": round(c["t"], 1), "call": c["call"], "trust": round(c["trust"], 3),
                     "called_before_lift_s": c.get("called_before_lift_s"), "after_onset_s": c.get("after_onset_s"),
                     "at_lift_call": (c.get("at_lift") or {}).get("call"),
                     "strong": c["strong"], "jev": c["jev"], "vision": c["vision"],
                     "matched": j is not None, "actual": truth,
                     "batter": feed.batter.iloc[j] if j is not None else None,
                     "count": feed["count"].iloc[j] if j is not None else None})
    df = pd.DataFrame(rows)
    df.to_json(od / "calls.json", orient="records", indent=1)
    m = df[df.matched & (df.actual != "OTHER")]
    acc = (m.call == m.actual).mean() if len(m) else float("nan")
    base = m.actual.value_counts(normalize=True).max() if len(m) else float("nan")
    st = m[m.strong]
    summary = {"video": video, "key": key, "game": game, "date": gdate, "pitches_in_feed": len(feed),
               "calls": len(df), "matched": int(df.matched.sum()), "unmatched_detections": int((~df.matched).sum()),
               "accuracy": round(float(acc), 3), "base_rate": round(float(base), 3),
               "strong_calls": len(st), "strong_accuracy": round(float((st.call == st.actual).mean()), 3) if len(st) else None,
               "offset_s": round(off, 1), "set_only": set_only,
               "called_before_lift": int(df.called_before_lift_s.notna().sum()) if "called_before_lift_s" in df else 0}
    (od / "summary.json").write_text(json.dumps(summary, indent=1))
    typer.echo(json.dumps(summary, indent=1))


@app.command()
def yrescore(out: str, key: str, game: int, mode: str = "type", label: str = "",
             innings: Optional[list[int]] = typer.Option(None)):
    """Re-align saved YouTube calls with the feed and re-render (no video processing)."""
    from pathlib import Path
    from pitchtip import config, youtube
    from pitchtip.config import slug
    from pitchtip.vision import phases, pose
    od = Path(out)
    rows = json.loads((od / "calls.json").read_text())
    pf = pd.read_parquet(config.FEATURES_DIR / f"{slug(key)}.parquet")
    feed = youtube.feed_pitches(game, int(pf.pitcher_id.iloc[0]), innings)
    classes = sorted(set(pf.pitch_type.value_counts(normalize=True).loc[lambda v: v >= 0.05].index))
    feed["label"] = feed.pitch_type.where(feed.pitch_type.isin(classes), "OTHER")
    match = youtube.align_dp(np.array([r["video_t"] for r in rows]), feed.t.to_numpy())
    hand = pf.pitcher_hand.iloc[0]
    for r in rows:
        j = match.get(r["i"])
        r.update(matched=j is not None, actual=feed.label.iloc[j] if j is not None else None,
                 batter=feed.batter.iloc[j] if j is not None else None, count=feed["count"].iloc[j] if j is not None else None)
        fr, pz = od / f'call_{r["i"]:03d}_frame.npy', od / f'call_{r["i"]:03d}_pose.npz'
        if fr.exists() and pz.exists():
            cp = pose.ClipPose.load(pz)
            ph = phases.segment(cp, hand)
            if ph is not None:
                youtube.render(np.load(fr), cp, ph, r, r["actual"], od / f'call_{r["i"]:03d}.jpg',
                               f'{label or key}  t={r["video_t"]:.0f}s' + (f'  {r["batter"]} (count after: {r["count"]})' if j is not None else ""))
    df = pd.DataFrame(rows)
    df.to_json(od / "calls.json", orient="records", indent=1)
    m = df[df.matched & (df.actual != "OTHER")]
    st = m[m.strong]
    s = json.loads((od / "summary.json").read_text())
    s.update(matched=int(df.matched.sum()), unmatched_detections=int((~df.matched).sum()),
             accuracy=round(float((m.call == m.actual).mean()), 3), base_rate=round(float(m.actual.value_counts(normalize=True).max()), 3),
             strong_calls=len(st), strong_accuracy=round(float((st.call == st.actual).mean()), 3) if len(st) else None,
             alignment="order-preserving DP")
    (od / "summary.json").write_text(json.dumps(s, indent=1))
    typer.echo(json.dumps(s, indent=1))


@app.command()
def train(pitcher: str, mode: str = "type", per_game: bool = True):
    """Fit the local model on all games and save it for live use."""
    from pitchtip.models import local
    df, hands, y = _xy(pitcher, mode, per_game)
    m = local.fit(df, hands, y)
    local.save(m, pitcher, mode)
    typer.echo(f"saved model ({len(df)} pitches, classes {m.classes})")


@app.command()
def live(source: str, key: str = typer.Argument(..., help='dataset to learn the pitcher from, e.g. "Tyler Glasnow 2019"'),
         mode: str = "type", jev: bool = True, log: Optional[str] = None,
         train_before: Optional[str] = typer.Option(None, help="only learn from games before this date (YYYY-MM-DD)"),
         variant: str = typer.Option("trust", help="Jev evidence variant"),
         set_only: bool = typer.Option(False, help="call at leg-lift onset from the set alone (earlier, a bit less accurate)")):
    """Call pitches from behavior on a stream URL (yt-dlp) or a recorded video file."""
    from pathlib import Path
    from pitchtip import config, live as live_mod
    from pitchtip.config import slug
    from pitchtip.predictor import Predictor
    games = None
    if train_before:
        pf = pd.read_parquet(config.FEATURES_DIR / f"{slug(key)}.parquet")
        games = set(pf[pf.date < train_before].game_pk)
    pr = Predictor(key, mode, use_jev=jev, variant=variant, set_only=set_only).fit(games)
    typer.echo(f"model ready for {key}: classes {pr.model.classes}")
    live_mod.run(source, pr, log=Path(log) if log else None)


@app.command()
def reel(key: str, game: int, out: str = "data/reel.mp4", n: int = 200):
    """Stitch one game's clips (in pitch order) into a single broadcast-like video."""
    import cv2
    from pitchtip import config
    from pitchtip.config import slug
    df = pd.read_parquet(dataset.pitches_path(key))
    rows = df[df.game_pk == game].sort_values(["at_bat", "pitch_number"]).head(n)
    w = None
    truth = []
    for r in rows.itertuples():
        clip = config.CLIPS_DIR / slug(r.pitcher_name) / f"{r.play_id}.mp4"
        if not clip.exists():
            continue
        cap = cv2.VideoCapture(str(clip))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        if w is None:
            size = (int(cap.get(3)), int(cap.get(4)))
            w = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        while True:
            ok, f = cap.read()
            if not ok:
                break
            w.write(cv2.resize(f, size))
        truth.append(r.pitch_type)
        cap.release()
    w.release()
    typer.echo(f"{out}: {len(truth)} pitches; actual sequence: {' '.join(truth)}")


@app.command()
def inspect(pitcher: str, n: int = 12, out: str = "data/inspect.jpg"):
    """Contact sheet of set-start / onset / early-end frames to verify the release cutoff."""
    import cv2
    from pitchtip import config
    from pitchtip.config import slug
    from pitchtip.vision import phases, pose
    df = pd.read_parquet(dataset.pitches_path(pitcher))
    s, hand = slug(df.pitcher_name.iloc[0]), df.pitcher_hand.iloc[0]
    rows = []
    for pid, pt in df.sample(frac=1, random_state=0)[["play_id", "pitch_type"]].itertuples(index=False):
        npz = config.POSE_DIR / s / f"{pid}.npz"
        if not npz.exists():
            continue
        cp = pose.ClipPose.load(npz)
        ph = phases.segment(cp, hand)
        if ph is None:
            continue
        cap = cv2.VideoCapture(str(config.CLIPS_DIR / s / f"{pid}.mp4"))
        tiles = []
        for label, idx in [("set", ph.set_start), ("onset", ph.onset), ("cutoff", ph.early_end - 1)]:
            cap.set(cv2.CAP_PROP_POS_MSEC, cp.t[idx] * 1000)
            ok, fr = cap.read()
            if not ok:
                break
            for x, y_, c in cp.kpts[idx]:
                if np.isfinite(x):
                    cv2.circle(fr, (int(x), int(y_)), 4, (0, 255, 0), -1)
            x1, y1, x2, y2 = cp.boxes[idx].astype(int)
            cv2.rectangle(fr, (x1, y1), (x2, y2), (0, 255, 255), 2)
            fr = cv2.resize(fr, (426, 240))
            cv2.putText(fr, f"{pt} {label} {cp.t[idx]:.1f}s", (8, 24), 0, 0.7, (0, 255, 255), 2)
            tiles.append(fr)
        if len(tiles) == 3:
            rows.append(np.hstack(tiles))
        if len(rows) >= n:
            break
    cv2.imwrite(out, np.vstack(rows))
    typer.echo(out)


if __name__ == "__main__":
    app()
