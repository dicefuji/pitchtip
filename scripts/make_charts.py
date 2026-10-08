"""Repo charts from data/rolling_results.jsonl (behavior-only rolling eval)."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

BLUE, ORANGE, GRAY, INK, MUTED, SURF = "#2a78d6", "#eb6834", "#9a9893", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#d6d4ce", "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})
rows = {}
for line in open("data/rolling_results.jsonl"):
    r = json.loads(line)
    if r["mode"] == "fb" and "jev" in r:
        rows[r["key"]] = r
R = sorted(rows.values(), key=lambda r: r["jev"]["accuracy"] - r["local"]["base_rate"])
out = Path("docs/img"); out.mkdir(parents=True, exist_ok=True)

# 1) dot plot: base vs local vs Jev
fig, ax = plt.subplots(figsize=(8.2, 0.36 * len(R) + 1.4))
for i, r in enumerate(R):
    b, l, j = r["local"]["base_rate"], r["local"]["accuracy"], r["jev"]["accuracy"]
    ax.plot([b, max(l, j)], [i, i], color="#e4e2dc", lw=2, zorder=1)
    ax.scatter(b, i, s=46, color=GRAY, zorder=2, label="base rate (always most common)" if i == 0 else None)
    ax.scatter(l, i, s=46, color=ORANGE, zorder=3, label="local vision model" if i == 0 else None)
    ax.scatter(j, i, s=60, color=BLUE, zorder=4, edgecolor=SURF, linewidth=1.5, label="Jev decision" if i == 0 else None)
    ax.text(1.005, i, f"{j:.0%}", va="center", fontsize=9, color=INK, transform=ax.get_yaxis_transform())
ax.set_yticks(range(len(R)), [r["key"] for r in R])
ax.set_xlim(0.45, 0.95); ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.grid(axis="x", color="#ecebe6", lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.set_xlabel("accuracy calling fastball vs offspeed, later games predicted from earlier games")
ax.set_title("Behavior-only pitch calls, 20 pitcher-seasons", loc="left", color=INK, fontsize=12, pad=24)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=9, borderaxespad=0.2)
fig.tight_layout(); fig.savefig(out / "rolling_accuracy.png", dpi=160); plt.close(fig)

# 2) lift over base rate
fig, ax = plt.subplots(figsize=(8.2, 0.36 * len(R) + 1.2))
lifts = [100 * (r["jev"]["accuracy"] - r["local"]["base_rate"]) for r in R]
ax.barh(range(len(R)), lifts, color=BLUE, height=0.62)
for i, v in enumerate(lifts):
    ax.text(v + 0.3 if v >= 0 else v - 0.3, i, f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=9, color=INK)
ax.set_yticks(range(len(R)), [r["key"] for r in R])
ax.axvline(0, color=MUTED, lw=1)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.grid(axis="x", color="#ecebe6", lw=0.8); ax.set_axisbelow(True)
ax.set_xlabel("Jev accuracy minus base rate (percentage points)")
ax.set_title("How much behavior adds over always guessing the most common pitch", loc="left", color=INK, fontsize=12)
fig.tight_layout(); fig.savefig(out / "jev_lift.png", dpi=160); plt.close(fig)
print("wrote", list(out.glob("*.png")))

# 3) pitch-by-pitch probability strip for the NLDS G2 9th inning (YouTube test)
calls = [c for c in json.load(open("data/yt_nlds_g2/calls.json")) if c["matched"]]
fig, ax = plt.subplots(figsize=(8.2, 3.2))
for i, c in enumerate(calls):
    p = (c["jev"] or {}).get("SL", 0)
    ax.bar(i, p - 0.5, bottom=0.5, color=ORANGE if p >= 0.5 else BLUE, width=0.6)
    ax.scatter(i, c["vision"].get("SL", 0), s=22, color=INK, zorder=3)
    ok = c["call"] == c["actual"]
    ax.text(i, -0.12, c["actual"], ha="center", fontsize=8, color=ORANGE if c["actual"] == "SL" else BLUE)
    ax.text(i, -0.22, "✓" if ok else "✗", ha="center", fontsize=9, color="#1a7f37" if ok else "#c43c33")
ax.axhline(0.5, color=MUTED, lw=1, ls="--")
ax.set_ylim(-0.28, 1.02); ax.set_xlim(-0.6, len(calls) - 0.4)
ax.set_yticks([0, 0.5, 1], ["FF", "50%", "SL"]); ax.set_xticks([])
for s in ("top", "right", "bottom"): ax.spines[s].set_visible(False)
ax.set_title("Mason Miller, NLDS Game 2, 9th inning: Jev's call for each pitch", loc="left", color=INK, fontsize=11, pad=22)
ax.text(0, 1.03, "bar up = slider, bar down = fastball · dot = vision model · below = actual pitch",
        transform=ax.transAxes, fontsize=8, color=MUTED)
fig.tight_layout(); fig.savefig(out / "nlds_g2_strip.png", dpi=160); plt.close(fig)

# 4) what Jev was given and what it answered, for one real pitch
ex = json.load(open("docs/jev_example.json"))
st = ex["state"]
fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.8), gridspec_kw={"width_ratios": [1.3, 1.1, 0.9]})
names = list(st["expert_opinions"])
ff = [st["expert_opinions"][n]["FF"] for n in names]
axes[0].barh(range(len(names)), ff, color=BLUE, height=0.55)
axes[0].barh(range(len(names)), [1 - v for v in ff], left=ff, color=ORANGE, height=0.55)
axes[0].set_yticks(range(len(names)), ["body model", "posture model", "glove model"])
axes[0].set_title("1. Expert models\n(blue = fastball, orange = slider)", loc="left", fontsize=10, color=INK)
tr = st["vision_model_track_record_on_unseen_games"]["by_confidence"]
axes[1].bar(range(len(tr)), [v["right"] for v in tr.values()], color=GRAY, width=0.6)
axes[1].set_xticks(range(len(tr)), [k.split("-")[0] + "+" for k in tr], fontsize=8)
axes[1].set_ylim(0, 1); axes[1].set_title("2. How often the vision\nmodel was right, by confidence", loc="left", fontsize=10, color=INK)
pr = ex["answer"]["probabilities"]
axes[2].bar(["FF", "SL"], [pr["FF"], pr["SL"]], color=[BLUE, ORANGE], width=0.55)
axes[2].set_ylim(0, 1); axes[2].set_title(f"3. Jev's answer ({ex['answer']['latency_s'] * 1000:.0f} ms)\nactual pitch: {ex['actual']}", loc="left", fontsize=10, color=INK)
for a in axes:
    for s in ("top", "right"): a.spines[s].set_visible(False)
    a.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}") if a is axes[0] else None
    if a is not axes[0]: a.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
fig.tight_layout(); fig.savefig(out / "jev_example.png", dpi=160); plt.close(fig)

# 5) a few annotated frames from the YouTube tests (downscaled)
import cv2
for src, dst in [("data/yt_yankees_0906/call_002.jpg", "call_yankees_sl_strong.jpg"),
                 ("data/yt_yankees_0906/call_006.jpg", "call_yankees_miss.jpg")]:
    im = cv2.imread(src)
    im = cv2.resize(im, (900, int(im.shape[0] * 900 / im.shape[1])))
    cv2.imwrite(str(out / dst), im, [cv2.IMWRITE_JPEG_QUALITY, 80])
print("extra figures written")
