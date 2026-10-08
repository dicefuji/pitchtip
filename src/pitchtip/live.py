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
    """Local files pass through; web pages (YouTube, Twitch, etc.) resolve to a direct
    stream URL through yt-dlp (H.264 video-only preferred, which OpenCV decodes)."""
    if Path(src).exists() or not src.startswith("http") or src.endswith((".mp4", ".m3u8")):
        return src
    cmd = ["uvx", "--from", "yt-dlp[default]", "yt-dlp"] if shutil.which("uvx") else ["yt-dlp"]
    out = subprocess.run(cmd + ["-g", "-f", "136/232/398/best[height<=720]", "--", src],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip().splitlines()[0]


def run(src: str, predictor, target_fps: float = 15.0, log: Path | None = None):
    """Call pitches on a stream as they happen (fast/accurate pose cascade).

    Prints each call when the leg lift is ~0.4 s old, before release, plus how far the
    processing is from real time."""
    from pitchtip import youtube

    url = resolve_source(src)
    log_f = open(log, "a") if log else None
    t0 = time.perf_counter()

    def on_call(c):
        lag = (time.perf_counter() - t0) - c["t"]
        probs = c.get("jev") or c["vision"]
        flag = "STRONG" if c["strong"] else "      "
        print(f"[{c['t']:7.1f}s] leg lift -> {c['call']:>4} {flag} trust {c['trust']:.0%}  " +
              "  ".join(f"{k}:{v:.0%}" for k, v in sorted(probs.items(), key=lambda kv: -kv[1])) +
              f"   ready {c['after_onset_s']:.2f}s after lift onset", flush=True)
        if log_f:
            log_f.write(json.dumps({**c, "lag_s": lag}, default=float) + "\n"); log_f.flush()

    calls = youtube.run(url, predictor, Path(log).parent / "live_frames" if log else Path("data/live_frames"),
                        target_fps=target_fps, on_call=on_call)
    wall = time.perf_counter() - t0
    print(f"processed stream in {wall:.0f}s wall clock; {len(calls)} calls", flush=True)
    return calls
