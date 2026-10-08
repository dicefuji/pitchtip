"""Test the behavior-only predictor on arbitrary broadcast video (e.g. YouTube) and score
it against the MLB game feed.

Each detected leg lift becomes a call. Ground truth comes from the feed's per-pitch
timestamps: video time and feed time differ by an unknown offset, which we fit by
maximising one-to-one matches, then each call is paired with the nearest pitch.
"""
from __future__ import annotations

import collections
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pitchtip.data.mlb import _get, API
from pitchtip.vision import phases, pose

# BGR. Fastballs blue (#2a78d6), offspeed orange (#eb6834) - same convention as the report.
COLORS = {"FF": (214, 120, 42), "SI": (214, 120, 42), "FC": (214, 120, 42), "FASTBALL": (214, 120, 42),
          "SL": (52, 104, 235), "ST": (52, 104, 235), "CU": (52, 104, 235), "KC": (52, 104, 235),
          "CH": (52, 104, 235), "FS": (52, 104, 235), "CS": (52, 104, 235), "OFFSPEED": (52, 104, 235)}


def feed_pitches(game_pk: int, pitcher_id: int) -> pd.DataFrame:
    feed = _get(f"{API}/v1.1/game/{game_pk}/feed/live")
    rows = []
    for play in feed["liveData"]["plays"]["allPlays"]:
        if play["matchup"]["pitcher"]["id"] != pitcher_id:
            continue
        for ev in play["playEvents"]:
            if ev.get("isPitch") and ev.get("startTime"):
                t = datetime.fromisoformat(ev["startTime"].replace("Z", "+00:00")).timestamp()
                rows.append({"t": t, "pitch_type": (ev["details"].get("type") or {}).get("code"),
                             "batter": play["matchup"]["batter"]["fullName"],
                             "count": f'{ev["count"]["balls"]}-{ev["count"]["strikes"]}',
                             "desc": ev["details"].get("description")})
    return pd.DataFrame(rows).sort_values("t").reset_index(drop=True)


def align(call_t: np.ndarray, pitch_t: np.ndarray, tol: float = 4.0) -> tuple[float, dict[int, int]]:
    """Fit the video→feed offset (release follows leg lift by ~1 s) and match one-to-one."""
    best, best_n = 0.0, -1
    for i in range(len(call_t)):
        for j in range(len(pitch_t)):
            off = pitch_t[j] - call_t[i]
            d = np.abs((call_t[:, None] + off) - pitch_t[None, :])
            n = int((d.min(1) < tol).sum())
            if n > best_n or (n == best_n and np.median(d.min(1)) < np.median(
                    np.abs((call_t[:, None] + best) - pitch_t[None, :]).min(1))):
                best, best_n = off, n
    match, used = {}, set()
    d = np.abs((call_t[:, None] + best) - pitch_t[None, :])
    for i in np.argsort(d.min(1)):
        j = int(np.argmin(np.where([k in used for k in range(len(pitch_t))], np.inf, d[i])))
        if d[i, j] < tol:
            match[int(i)] = j
            used.add(j)
    return best, match


def _bars(img, x, y, w, title, probs: dict, h_bar=26):
    cv2.putText(img, title, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
    y += 12
    for k, v in sorted(probs.items(), key=lambda kv: -kv[1]):
        cv2.rectangle(img, (x + 70, y), (x + 70 + int(v * (w - 130)), y + h_bar - 6), COLORS.get(k, (180, 180, 180)), -1)
        cv2.putText(img, k[:8], (x, y + h_bar - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(img, f"{v:.0%}", (x + 76 + int(v * (w - 130)), y + h_bar - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (230, 230, 230), 1, cv2.LINE_AA)
        y += h_bar
    return y + 14


def render(frame, cp, ph, out: dict, truth: str | None, path: Path, label: str):
    """Broadcast frame at leg-lift onset + side panel: glove crop, Jev & vision probabilities."""
    H = 720
    f = cv2.resize(frame, (1280, 720))
    sx = 1280 / frame.shape[1]
    b = (cp.boxes[ph.onset] * sx).astype(int)
    cv2.rectangle(f, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 2)
    for x, y_, c in cp.kpts[ph.onset]:
        if np.isfinite(x):
            cv2.circle(f, (int(x * sx), int(y_ * sx)), 4, (0, 255, 0), -1)
    panel = np.full((H, 420, 3), 28, np.uint8)
    cv2.putText(panel, label, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    if len(cp.glove):
        g = cv2.resize(cp.glove[max(0, ph.onset - cp.glove_start - 1)], (180, 180), interpolation=cv2.INTER_NEAREST)
        panel[46:226, 16:196] = g
        cv2.putText(panel, "glove at set", (16, 244), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    call_col = COLORS.get(out["call"], (255, 255, 255))
    cv2.putText(panel, "CALL", (214, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(panel, out["call"], (214, 130), cv2.FONT_HERSHEY_DUPLEX, 1.5, call_col, 2, cv2.LINE_AA)
    cv2.putText(panel, f"trust {out['trust']:.0%}" + ("  STRONG" if out["strong"] else ""), (214, 165),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255) if out["strong"] else (200, 200, 200), 1, cv2.LINE_AA)
    if truth:
        ok = truth == out["call"]
        cv2.putText(panel, f"actual {truth}", (214, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (80, 220, 80) if ok else (80, 80, 240), 2, cv2.LINE_AA)
    y = 290
    if "jev" in out:
        y = _bars(panel, 16, y, 400, "Jev decision (Choice probabilities)", out["jev"])
    y = _bars(panel, 16, y, 400, "Vision model (calibrated)", out["vision"])
    cv2.putText(panel, "behavior only - no count/situation", (16, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (140, 140, 140), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), np.hstack([f, panel]))


def cascade_call(predictor, ts, frs, target_fps: float):
    """Re-run the accurate pose model on the buffered frames, segment, featurize."""
    kps, bxs = pose.pose_frames(list(frs))
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frs]
    hs = np.array([pose._hands_patch(g, k, b) if np.isfinite(b).all() else
                   np.zeros((pose.HAND_PATCH, pose.HAND_PATCH), np.uint8) for g, k, b in zip(gray, kps, bxs)])
    cp = pose.ClipPose(target_fps, np.array(ts), kps, bxs, hs, frs[0].shape[1::-1])
    ph = phases.segment(cp, predictor.hand)
    if ph is None:
        return None
    cp.glove = pose.smoothed_glove_crops(list(frs), kps, bxs, ph.set_start, ph.early_end)
    cp.glove_start = ph.set_start
    feats = predictor.features_from_pose(cp)
    return None if feats is None else (cp, ph, feats)


def run(video: str, predictor, out_dir: Path, target_fps: float = 15.0, buffer_seconds: float = 4.0):
    """Detect every leg lift in the video and call it; returns list of (call, ClipPose, Phases).

    Cascade: a tiny pose model tracks the pitcher on every frame and spots the leg lift;
    only then is the accurate model run (batched) over the buffered seconds."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(fps / target_fps)), 1)
    n = int(buffer_seconds * target_fps)
    buf = collections.deque(maxlen=n)
    prev_box, cooldown, i, calls = None, -1.0, 0, []
    hand = predictor.hand
    while cap.grab():
        i += 1
        if i % step:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        t = i / fps
        kp, box = pose.pose_frame(frame, prev_box, weights=pose.FAST_WEIGHTS)
        if kp is None:
            buf.clear(); prev_box = None
            continue
        prev_box = box
        kp = kp.copy(); kp[kp[:, 2] < 0.3, :2] = np.nan
        buf.append((t, kp, box, frame))
        if t < cooldown or len(buf) < n // 2:
            continue
        ts, kps, bxs, frs = zip(*buf)
        cp_fast = pose.ClipPose(target_fps, np.array(ts), np.array(kps), np.array(bxs),
                                np.zeros((len(ts), pose.HAND_PATCH, pose.HAND_PATCH), np.uint8), frame.shape[1::-1])
        ph_fast = phases.segment(cp_fast, hand)
        if ph_fast is None or ph_fast.early_end + 1 >= len(ts):
            continue
        res = cascade_call(predictor, ts, frs, target_fps)
        cooldown = t + 6.0
        buf.clear()
        if res is None:
            continue
        cp, ph, feats = res
        o = predictor.predict_features(*feats)
        c = {"t": float(cp.t[ph.onset]), **{k: o[k] for k in ("call", "confidence", "trust", "strong", "vision")},
             "jev": o.get("jev"), "idx": len(calls)}
        np.save(out_dir / f"call_{len(calls):03d}_frame.npy", np.array(frs[ph.onset]))
        cp.save(out_dir / f"call_{len(calls):03d}_pose.npz")  # lets renders be redone without video
        calls.append((c, cp, ph))
        print(f"[{c['t']:7.1f}s] call {c['call']:>4} trust {c['trust']:.0%}{' STRONG' if c['strong'] else ''}", flush=True)
    cap.release()
    return calls
