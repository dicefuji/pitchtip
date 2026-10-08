"""Run the tip detector on a stream or a long recorded game.

Keeps a rolling pose + glove-crop buffer of the CF shot; when a leg lift starts it calls
the pitch from behavior only (~0.4s after onset, still before release).
"""
from __future__ import annotations

import collections
import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from pitchtip.vision import phases, pose


def resolve_source(src: str) -> str:
    """Local files pass through; web pages (YouTube, Twitch, etc.) go through yt-dlp."""
    if Path(src).exists() or not src.startswith("http") or src.endswith((".mp4", ".m3u8")):
        return src
    if not shutil.which("yt-dlp"):
        raise RuntimeError("yt-dlp is required to open stream pages")
    out = subprocess.run(["yt-dlp", "-g", "-f", "best[height<=720]/best", src],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip().splitlines()[0]


def run(src: str, predictor, target_fps: float = 15.0, buffer_seconds: float = 4.0,
        log: Path | None = None, realtime: bool = False):
    """predictor: a fitted pitchtip.predictor.Predictor (behavior-only, optional Jev)."""
    url = resolve_source(src)
    cap = cv2.VideoCapture(url)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(fps / target_fps)), 1)
    n = int(buffer_seconds * target_fps)
    buf = collections.deque(maxlen=n)
    prev_box, cooldown_until, i = None, -1.0, 0
    log_f = open(log, "a") if log else None
    hand = predictor.hand
    while True:
        ok = cap.grab()
        if not ok:
            break
        i += 1
        if i % step:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        t = i / fps
        kp, box = pose.pose_frame(frame, prev_box)
        if kp is None:
            buf.clear(); prev_box = None
            continue
        prev_box = box
        kp = kp.copy(); kp[kp[:, 2] < 0.3, :2] = np.nan
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        buf.append((t, kp, box, pose._hands_patch(gray, kp, box), frame))
        if t < cooldown_until or len(buf) < n // 2:
            continue
        ts, kps, bxs, hs, frs = zip(*buf)
        ts, kps, bxs, hs = map(np.array, (ts, kps, bxs, hs))
        cp = pose.ClipPose(target_fps, ts, kps, bxs, hs, frame.shape[1::-1])
        ph = phases.segment(cp, hand)
        if ph is None or ph.early_end >= len(ts):
            continue  # fire once the early-lift window is complete
        cp.glove = pose.smoothed_glove_crops(list(frs), kps, bxs, ph.set_start, ph.early_end)
        cp.glove_start = ph.set_start
        feats = predictor.features_from_pose(cp)
        if feats is None:
            continue
        t0 = time.perf_counter()
        out = predictor.predict_features(*feats)
        dt = time.perf_counter() - t0
        probs = out.get("jev") or out["vision"]
        msg = {"t": round(float(ts[ph.onset]), 2), "call": out["call"],
               "confidence": round(out["confidence"], 3), "trust": round(out["trust"], 3),
               "strong": bool(out["strong"]),
               "proba": {k: round(v, 3) for k, v in probs.items()}, "decide_ms": round(dt * 1000)}
        flag = "STRONG" if out["strong"] else "      "
        print(f"[{msg['t']:7.2f}s] leg lift → {out['call']:>8} {flag} trust {out['trust']:.0%}  " +
              "  ".join(f"{k}:{v:.0%}" for k, v in sorted(probs.items(), key=lambda kv: -kv[1])[:4]) +
              f"   [{msg['decide_ms']} ms]", flush=True)
        if log_f:
            log_f.write(json.dumps(msg) + "\n"); log_f.flush()
        cooldown_until = t + 6.0
        buf.clear()
    cap.release()
