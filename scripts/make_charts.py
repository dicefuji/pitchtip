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
