"""Build report/index.html (+ images) from YouTube test runs, rolling results and the Jev example."""
import html
import json
from pathlib import Path

import cv2

ROOT = Path(".")
OUT = ROOT / "report"
(OUT / "img").mkdir(parents=True, exist_ok=True)
VIDEOS = [
    ("data/yt_nlds_g2", "NLDS Game 2 · 9th inning", "Oct 4, 2026 · @ MIL", "_-I2N1i9WCM",
     "The game where Brewers hitters were accused of reading his glove. Walk-off loss."),
    ("data/yt_yankees_0906", "Save No. 33 vs Yankees", "Sep 6, 2026", "PAlZF-EHUAU",
     "Full 9th inning, three strikeouts."),
    ("data/yt_braves_0622", "1-0 save vs Braves", "Jun 22, 2026", "-hjITEKROmQ",
     "Full 9th inning of a one-run game."),
]
esc = html.escape


def pct(x):
    return f"{100 * x:.0f}%"


def prob_strip(calls, w=720, h=226):
    """SVG: per call, P(SL) from Jev (bar) and vision (dot); actual marked below."""
    n = max(len(calls), 1)
    pad_l, pad_r, top, ph = 46, 12, 14, 130
    cw = (w - pad_l - pad_r) / n
    y = lambda p: top + (1 - p) * ph
    g = [f'<svg viewBox="0 0 {w} {h}" class="strip" role="img" aria-label="Per-pitch probabilities">']
    for v in (0, 0.5, 1):
        g.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="{"mid" if v == 0.5 else "grid"}"/>')
    g.append(f'<text x="{pad_l - 8}" y="{y(1) + 4:.1f}" class="ax" text-anchor="end">SL</text>')
    g.append(f'<text x="{pad_l - 8}" y="{y(0) + 4:.1f}" class="ax" text-anchor="end">FF</text>')
    g.append(f'<text x="{pad_l - 8}" y="{y(0.5) + 4:.1f}" class="ax" text-anchor="end">50%</text>')
    for i, c in enumerate(calls):
        x0 = pad_l + i * cw
        pj = (c.get("jev") or {}).get("SL", 0.0)
        pv = c["vision"].get("SL", 0.0)
        bw = max(cw * 0.56, 3)
        cls = "sl" if pj >= 0.5 else "ff"
        yb, yt = (y(0.5), y(pj)) if pj >= 0.5 else (y(pj), y(0.5))
        tip = (f'#{i + 1} {c.get("batter") or ""} {c.get("count") or ""}: Jev SL {pct(pj)}, vision SL {pct(pv)}'
               f' · call {c["call"]} · actual {c.get("actual") or "unmatched"}')
        g.append(f'<g class="col{"" if c.get("actual") else " dim"}"><title>{esc(tip)}</title>'
                 f'<rect x="{x0:.1f}" y="{top}" width="{cw:.1f}" height="{h - top}" class="hit"/>'
                 f'<rect x="{x0 + (cw - bw) / 2:.1f}" y="{min(yb, yt):.1f}" width="{bw:.1f}" height="{max(abs(yb - yt), 1.5):.1f}" rx="2" class="{cls}"/>'
                 f'<circle cx="{x0 + cw / 2:.1f}" cy="{y(pv):.1f}" r="3.6" class="vis"/>')
        a = c.get("actual")
        if a:
            ok = a == c["call"]
            g.append(f'<text x="{x0 + cw / 2:.1f}" y="{top + ph + 22}" text-anchor="middle" class="act {"sl" if a == "SL" else "ff"}t">{a}</text>'
                     f'<text x="{x0 + cw / 2:.1f}" y="{top + ph + 40}" text-anchor="middle" class="{"ok" if ok else "bad"}">{"✓" if ok else "✗"}</text>')
        else:
            g.append(f'<text x="{x0 + cw / 2:.1f}" y="{top + ph + 22}" text-anchor="middle" class="ax">–</text>')
        if c.get("strong"):
            g.append(f'<text x="{x0 + cw / 2:.1f}" y="{top + ph + 56}" text-anchor="middle" class="strong">★</text>')
    g.append("</svg>")
    return "".join(g)


def dotplot(rows, w=720):
    rows = sorted(rows, key=lambda r: -(r["jev"]["accuracy"] - r["local"]["base_rate"]))
    rh, top, pl, pr = 26, 30, 190, 56
    h = top + rh * len(rows) + 34
    lo, hi = 0.45, 0.95
    x = lambda v: pl + (v - lo) / (hi - lo) * (w - pl - pr)
    g = [f'<svg viewBox="0 0 {w} {h}" class="dots" role="img" aria-label="Accuracy by pitcher">']
    for v in (0.5, 0.6, 0.7, 0.8, 0.9):
        g.append(f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{top - 8}" y2="{h - 26}" class="grid"/>'
                 f'<text x="{x(v):.1f}" y="{h - 8}" text-anchor="middle" class="ax">{pct(v)}</text>')
    for i, r in enumerate(rows):
        yy = top + i * rh + rh / 2
        b, l, j = r["local"]["base_rate"], r["local"]["accuracy"], r["jev"]["accuracy"]
        tip = f'{r["key"]}: base {pct(b)}, local {pct(l)}, Jev {pct(j)} (n={r["local"]["n"]})'
        g.append(f'<g class="row"><title>{esc(tip)}</title>'
                 f'<rect x="0" y="{yy - rh / 2:.1f}" width="{w}" height="{rh}" class="hit"/>'
                 f'<text x="{pl - 10}" y="{yy + 4:.1f}" text-anchor="end" class="lbl">{esc(r["key"])}</text>'
                 f'<line x1="{x(b):.1f}" x2="{x(max(l, j)):.1f}" y1="{yy:.1f}" y2="{yy:.1f}" class="link"/>'
                 f'<circle cx="{x(b):.1f}" cy="{yy:.1f}" r="5" class="base"/>'
                 f'<circle cx="{x(l):.1f}" cy="{yy:.1f}" r="5" class="loc"/>'
                 f'<circle cx="{x(j):.1f}" cy="{yy:.1f}" r="6" class="jevd"/>'
                 f'<text x="{w - pr + 10}" y="{yy + 4:.1f}" class="val">{pct(j)}</text></g>')
    g.append("</svg>")
    return "".join(g)


def bars(probs, cls_of=lambda k: "sl" if k in ("SL", "OFFSPEED", "ST", "CU", "CH") else "ff"):
    out = []
    for k, v in sorted(probs.items(), key=lambda kv: -kv[1]):
        out.append(f'<div class="pb"><span class="k">{esc(k)}</span><span class="track"><span class="fill {cls_of(k)}" style="width:{100 * v:.1f}%"></span></span><span class="v">{pct(v)}</span></div>')
    return "".join(out)


# ---------- data
sections, all_m, all_ok, all_strong, strong_ok, base_num = [], 0, 0, 0, 0, 0
for d, title, when, yt, blurb in VIDEOS:
    p = Path(d)
    if not (p / "summary.json").exists():
        continue
    s = json.loads((p / "summary.json").read_text())
    calls = json.loads((p / "calls.json").read_text())
    m = [c for c in calls if c["matched"] and c["actual"] in ("FF", "SL")]
    ok = sum(c["call"] == c["actual"] for c in m)
    st = [c for c in m if c["strong"]]
    all_m += len(m); all_ok += ok; all_strong += len(st); strong_ok += sum(c["call"] == c["actual"] for c in st)
    if m:
        base_num += max(sum(c["actual"] == "FF" for c in m), sum(c["actual"] == "SL" for c in m))
    cards, extra = [], []
    for c in calls:
        src = p / f'call_{c["i"]:03d}.jpg'
        if not src.exists():
            continue
        name = f'{p.name}_{c["i"]:03d}.jpg'
        im = cv2.imread(str(src))
        im = cv2.resize(im, (1100, int(im.shape[0] * 1100 / im.shape[1])))
        cv2.imwrite(str(OUT / "img" / name), im, [cv2.IMWRITE_JPEG_QUALITY, 78])
        a = c.get("actual")
        verdict = "" if not a or a == "OTHER" else ("hit" if a == c["call"] else "miss")
        who = esc(c.get("batter") or "no pitch in feed (replay?)")
        cnt = ("count after " + esc(c["count"])) if c.get("count") else ""
        acttag = ('<span class="tag act">actual ' + a + "</span>") if a else ""
        strongtag = "<span class=st>STRONG</span>" if c["strong"] else ""
        (cards if a else extra).append(f'<figure class="card {verdict}"><img loading="lazy" src="img/{name}" alt="Call {c["i"] + 1}: {c["call"]}, actual {a or "unmatched"}">'
                     f'<figcaption><span class="n">#{c["i"] + 1}</span> <b>{who}</b> {cnt}'
                     f'<span class="tag {c["call"].lower()}">call {c["call"]}</span>'
                     f'{acttag}'
                     f'{strongtag}</figcaption></figure>')
    acc = ok / len(m) if m else 0
    base = (max(sum(c["actual"] == "FF" for c in m), sum(c["actual"] == "SL" for c in m)) / len(m)) if m else 0
    sections.append(f'''
<section class="video" id="{p.name}">
  <header class="vh"><div><h3>{esc(title)}</h3><p class="meta">{esc(when)} · <a href="https://www.youtube.com/watch?v={yt}" target="_blank" rel="noopener">YouTube</a> · {esc(blurb)}</p></div>
  <dl class="chips"><div><dt>called</dt><dd>{ok}/{len(m)}</dd></div><div><dt>accuracy</dt><dd>{pct(acc)}</dd></div><div><dt>base rate</dt><dd>{pct(base)}</dd></div><div><dt>STRONG</dt><dd>{sum(c["call"] == c["actual"] for c in st)}/{len(st)}</dd></div><div><dt>pitches in feed</dt><dd>{s["pitches_in_feed"]}</dd></div></dl></header>
  <div class="chart"><div class="legend"><span><i class="sw sl"></i>Jev P(slider), bar up</span><span><i class="sw ff"></i>Jev P(fastball), bar down</span><span><i class="dot"></i>vision model</span><span>★ STRONG call</span></div>{prob_strip(calls)}</div>
  <div class="film">{"".join(cards)}</div>
  {f'<details class="extra"><summary>{len(extra)} detections with no pitch in the feed (replays, cutaways), not scored</summary><div class="film">{"".join(extra)}</div></details>' if extra else ""}
</section>''')

# pre-lift (set-only standing call) comparison on the same three innings
import statistics
pre_rows, pre_tot, pre_ok, pre_base, pre_leads = [], 0, 0, 0, []
for (d, title, when, yt, blurb) in VIDEOS:
    pd_ = Path(d.replace("/yt_", "/ytp_"))
    if not (pd_ / "calls.json").exists():
        continue
    cc = json.loads((pd_ / "calls.json").read_text())
    mm = [c for c in cc if c["matched"] and c["actual"] in ("FF", "SL")]
    ok_ = sum(c["call"] == c["actual"] for c in mm)
    b_ = max(sum(c["actual"] == "FF" for c in mm), sum(c["actual"] == "SL" for c in mm))
    lead = [c["called_before_lift_s"] for c in mm if c.get("called_before_lift_s") is not None]
    pre_rows.append(f"<tr><td>{esc(title)}</td><td>{ok_}/{len(mm)}</td><td>{pct(ok_ / len(mm))}</td><td>{pct(b_ / len(mm))}</td><td>{len(lead)}/{len(mm)}</td></tr>")
    pre_tot += len(mm); pre_ok += ok_; pre_base += b_; pre_leads += lead
pre_html = ""
if pre_tot:
    pre_html = f'''
<section>
  <span class="eyebrow">earlier calls</span><h2>Calling it before the leg lift</h2>
  <p>A call that lands after the lift starts is a real pre-release prediction, but a hitter has no time to use it. In set-only mode, pitchtip keeps a standing call while the pitcher is set. It refreshes about every quarter second from the last 1.2 s of his set position. When the lift begins, the last standing call made <em>before</em> the lift is the prediction. The model is retrained without any lift features.</p>
  <div class="kpis">
    <div class="kpi"><span>Pre-lift accuracy, same three innings</span><b>{pct(pre_ok / pre_tot)}</b><small>{pre_ok} of {pre_tot} · base rate {pct(pre_base / pre_tot)}</small></div>
    <div class="kpi"><span>Called before the lift started</span><b>{len(pre_leads)}/{pre_tot}</b><small>median {statistics.median(pre_leads):.2f} s before the lift (≈1.2–1.6 s before release)</small></div>
    <div class="kpi"><span>Set-only cost, rolling eval</span><b>≈ −6 pts</b><small>5 most readable pitchers; still +13–19 over base</small></div>
  </div>
  <div class="tblwrap"><table><thead><tr><th>inning</th><th>correct</th><th>accuracy</th><th>base</th><th>called pre-lift</th></tr></thead><tbody>{"".join(pre_rows)}</tbody></table></div>
</section>'''

rows = {}
for line in open("data/rolling_results.jsonl"):
    r = json.loads(line)
    if r["mode"] == "fb" and "jev" in r:
        rows[r["key"]] = r
R = list(rows.values())
pooled_n = sum(r["local"]["n"] for r in R)
pooled_jev = sum(r["jev"]["accuracy"] * r["local"]["n"] for r in R) / pooled_n
pooled_base = sum(r["local"]["base_rate"] * r["local"]["n"] for r in R) / pooled_n
ex = json.loads(Path("docs/jev_example.json").read_text())
st = ex["state"]
yt_acc = all_ok / all_m if all_m else 0
yt_base = base_num / all_m if all_m else 0

table_rows = "".join(
    f'<tr><td>{esc(r["key"])}</td><td>{r["local"]["n"]}</td><td>{pct(r["local"]["base_rate"])}</td><td>{pct(r["local"]["accuracy"])}</td>'
    f'<td><b>{pct(r["jev"]["accuracy"])}</b></td><td>{("+%.1f" % (100 * (r["jev"]["accuracy"] - r["local"]["base_rate"])))}</td>'
    f'<td>{pct(r["jev_gated"]["top25pct"]) if r.get("jev_gated") else "–"}</td></tr>'
    for r in sorted(R, key=lambda r: -(r["jev"]["accuracy"] - r["local"]["base_rate"])))

tr = st["vision_model_track_record_on_unseen_games"]["by_confidence"]
track = "".join(f'<div class="pb"><span class="k">{esc(k)}</span><span class="track"><span class="fill neutral" style="width:{100 * v["right"]:.0f}%"></span></span><span class="v">{pct(v["right"])}</span></div>' for k, v in tr.items())
experts = "".join(f'<div class="exp"><span class="en">{esc(name)}</span>{bars(p)}</div>' for name, p in st["expert_opinions"].items())
nn = st["most_similar_past_deliveries"]["pitch_counts"]
nn_dots = "".join(f'<i class="nd {k.lower()}" title="{k}"></i>' for k, v in sorted(nn.items()) for _ in range(v))
tells = "".join(f'<li>{esc(t["tell"].split(" (vs")[0])} <span class="z">z {t["this_pitch_reading_z"]:+.2f}</span></li>' for t in st["known_tells"][:4])

page = f'''<title>Pitchtip Field Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column (max 1120px); scoreboard-style numerals; filmstrips scroll sideways inside their own track. */
:root {{
  --bg:#f5f6f7; --surface:#ffffff; --ink:#11161c; --muted:#5b6571; --line:#dfe3e8; --soft:#eef1f4;
  --ff:#2a78d6; --sl:#eb6834; --neutral:#7d8794; --ok:#1a7f37; --bad:#c43c33; --gold:#b7791f;
  --display:"Barlow Condensed","Arial Narrow",system-ui,sans-serif; --body:"IBM Plex Sans",system-ui,-apple-system,sans-serif; --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg:#0f1317; --surface:#161c22; --ink:#e7ebef; --muted:#9aa5b1; --line:#28313a; --soft:#1c242c;
  --ff:#3987e5; --sl:#d95926; --neutral:#8792a0; --ok:#3fb26b; --bad:#e8675e; --gold:#e0a43c; color-scheme:dark }} }}
:root[data-theme="dark"] {{
  --bg:#0f1317; --surface:#161c22; --ink:#e7ebef; --muted:#9aa5b1; --line:#28313a; --soft:#1c242c;
  --ff:#3987e5; --sl:#d95926; --neutral:#8792a0; --ok:#3fb26b; --bad:#e8675e; --gold:#e0a43c; color-scheme:dark }}
body {{ background:var(--bg); color:var(--ink); font:15px/1.55 var(--body); }}
.wrap {{ max-width:1120px; margin:0 auto; padding-inline:clamp(16px,4vw,40px); padding-block:28px 64px; display:grid; gap:44px; }}
h1,h2,h3 {{ font-family:var(--display); font-weight:600; letter-spacing:.01em; line-height:1.05; text-wrap:balance; margin:0 }}
h1 {{ font-size:clamp(40px,6vw,64px); }} h2 {{ font-size:32px; }} h3 {{ font-size:24px; }}
p {{ margin:0; max-width:68ch }} a {{ color:var(--ff) }}
.eyebrow {{ font:600 12px/1 var(--body); letter-spacing:.12em; text-transform:uppercase; color:var(--muted) }}
.hero {{ display:grid; gap:18px }} .lede {{ font-size:17px; color:var(--muted) }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:1px; background:var(--line); border:1px solid var(--line); border-radius:10px; overflow:hidden }}
.kpi {{ background:var(--surface); padding:16px 18px; display:grid; gap:4px; min-width:0 }}
.kpi b {{ font:600 44px/1 var(--display); font-variant-numeric:tabular-nums }} .kpi span {{ color:var(--muted); font-size:13px }}
.kpi small {{ color:var(--muted); font-size:12px }}
section {{ display:grid; gap:16px; min-width:0 }}
.video {{ background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:18px; gap:14px }}
.vh {{ display:flex; flex-wrap:wrap; gap:14px; justify-content:space-between; align-items:flex-start }}
.vh > div {{ min-width:0; flex:1 1 300px; display:grid; gap:4px }} .meta {{ color:var(--muted); font-size:14px }}
.chips {{ display:flex; flex-wrap:wrap; gap:8px; margin:0 }}
.chips div {{ background:var(--soft); border-radius:8px; padding:6px 10px; display:grid }}
.chips dt {{ font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted) }}
.chips dd {{ margin:0; font:600 22px/1.1 var(--display); font-variant-numeric:tabular-nums }}
.chart {{ overflow-x:auto }} .strip {{ width:100%; min-width:520px; height:auto; display:block }}
.legend {{ display:flex; flex-wrap:wrap; gap:14px; font-size:12px; color:var(--muted); margin-bottom:4px }}
.sw {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; vertical-align:-1px }}
.sw.sl {{ background:var(--sl) }} .sw.ff {{ background:var(--ff) }}
.dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--ink); margin-right:6px }}
svg text {{ font-family:var(--body); }} .ax {{ fill:var(--muted); font-size:11px }}
.grid {{ stroke:var(--line); stroke-width:1 }} .mid {{ stroke:var(--muted); stroke-width:1; stroke-dasharray:3 3 }}
rect.sl {{ fill:var(--sl) }} rect.ff {{ fill:var(--ff) }} .vis {{ fill:var(--ink); stroke:var(--surface); stroke-width:1.5 }}
.hit {{ fill:transparent }} .col:hover .hit, .row:hover .hit {{ fill:var(--soft) }}
.act {{ font:600 10px var(--body) }} .dim {{ opacity:.32 }} .extra {{ display:grid; gap:10px }} .slt {{ fill:var(--sl) }} .fft {{ fill:var(--ff) }}
.ok {{ fill:var(--ok); font-size:12px }} .bad {{ fill:var(--bad); font-size:12px }} .strong {{ fill:var(--gold); font-size:12px }}
.film {{ display:grid; grid-auto-flow:column; grid-auto-columns:minmax(300px,46%); gap:12px; overflow-x:auto; padding-bottom:8px; scroll-snap-type:x mandatory }}
.card {{ margin:0; background:var(--soft); border-radius:10px; overflow:hidden; scroll-snap-align:start; border-top:3px solid var(--line); min-width:0 }}
.card.hit {{ border-top-color:var(--ok) }} .card.miss {{ border-top-color:var(--bad) }}
.card img {{ display:block; width:100%; height:auto }}
.card figcaption {{ padding:8px 10px 10px; font-size:13px; display:flex; flex-wrap:wrap; gap:6px; align-items:center }}
.n {{ color:var(--muted); font-variant-numeric:tabular-nums }}
.tag {{ font:600 11px/1 var(--body); padding:4px 7px; border-radius:99px; background:var(--surface); border:1px solid var(--line) }}
.tag.ff {{ color:var(--ff) }} .tag.sl {{ color:var(--sl) }} .st {{ font:700 10px/1 var(--body); letter-spacing:.08em; color:var(--gold) }}
.how {{ display:grid; gap:18px }}
.pipe {{ overflow-x:auto }} .pipe svg {{ width:100%; min-width:640px; height:auto }}
.pipe .box {{ fill:var(--surface); stroke:var(--line) }} .pipe .jev {{ fill:var(--ff); }} .pipe text {{ fill:var(--ink); font-size:13px }}
.pipe .jevt {{ fill:#fff; font-weight:600 }} .pipe .sub {{ fill:var(--muted); font-size:11px }} .pipe .arrow {{ stroke:var(--muted); stroke-width:1.5; fill:none }}
.panes {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:14px }}
.pane {{ background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:16px; display:grid; gap:10px; align-content:start; min-width:0 }}
.pane h4 {{ margin:0; font:600 13px/1 var(--body); letter-spacing:.1em; text-transform:uppercase; color:var(--muted) }}
.pane .q {{ font-size:13px; color:var(--muted) }}
.pb {{ display:grid; grid-template-columns:56px 1fr 44px; gap:8px; align-items:center; font-size:12px; font-variant-numeric:tabular-nums }}
.pb .k {{ color:var(--muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap }} .pb .v {{ text-align:right }}
.track {{ height:10px; background:var(--soft); border-radius:3px; overflow:hidden }} .fill {{ display:block; height:100%; border-radius:0 3px 3px 0 }}
.fill.ff {{ background:var(--ff) }} .fill.sl {{ background:var(--sl) }} .fill.neutral {{ background:var(--neutral) }}
.exp {{ display:grid; gap:4px }} .en {{ font-size:12px; color:var(--ink) }}
.nn {{ display:flex; flex-wrap:wrap; gap:4px }} .nd {{ width:12px; height:12px; border-radius:50%; display:block }}
.nd.ff {{ background:var(--ff) }} .nd.sl {{ background:var(--sl) }}
.tells {{ margin:0; padding-left:18px; display:grid; gap:4px; font-size:13px }} .z {{ color:var(--muted); font-family:var(--mono); font-size:11px }}
.answer b {{ font:600 56px/1 var(--display); color:var(--ff) }}
.notes {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:14px }}
.note {{ border-left:3px solid var(--ff); padding:2px 0 2px 12px; display:grid; gap:4px }} .note b {{ font-size:14px }} .note p {{ font-size:14px; color:var(--muted) }}
.dotsbox {{ background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:16px; overflow-x:auto }}
.dots {{ width:100%; min-width:560px; height:auto }} .dots .lbl {{ fill:var(--ink); font-size:12px }} .dots .val {{ fill:var(--ink); font-size:12px; font-weight:600 }}
.link {{ stroke:var(--line); stroke-width:3 }} .base {{ fill:var(--neutral) }} .loc {{ fill:var(--sl) }} .jevd {{ fill:var(--ff); stroke:var(--surface); stroke-width:2 }}
.tblwrap {{ overflow-x:auto; background:var(--surface); border:1px solid var(--line); border-radius:12px }}
table {{ border-collapse:collapse; width:100%; font-size:13px; font-variant-numeric:tabular-nums }}
th,td {{ padding:8px 12px; text-align:right; border-bottom:1px solid var(--line); white-space:nowrap }} th:first-child,td:first-child {{ text-align:left }}
th {{ font-weight:600; color:var(--muted); font-size:12px }}
.caveats {{ margin:0; padding-left:18px; display:grid; gap:6px; color:var(--muted); max-width:80ch }}
details summary {{ cursor:pointer; color:var(--muted); font-size:13px }} pre {{ font:12px/1.5 var(--mono); background:var(--soft); padding:12px; border-radius:8px; overflow:auto; max-height:420px }}
:focus-visible {{ outline:2px solid var(--ff); outline-offset:2px }}
@media (prefers-reduced-motion: reduce) {{ * {{ scroll-behavior:auto }} }}
</style>
<main class="wrap">
<header class="hero">
  <span class="eyebrow">pitchtip · behavior-only pitch calling</span>
  <h1>Calling Mason Miller's next pitch from YouTube</h1>
  <p class="lede">Three full innings of broadcast video, never seen in training. For every pitch, a pose model watched the set position and the first instant of the leg lift, and Jev picked fastball or slider before the ball left his hand. No count, no runners, no game situation.</p>
  <div class="kpis">
    <div class="kpi"><span>YouTube pitches called correctly</span><b>{pct(yt_acc)}</b><small>{all_ok} of {all_m} · base rate {pct(yt_base)}</small></div>
    <div class="kpi"><span>STRONG calls correct</span><b>{(str(strong_ok) + "/" + str(all_strong)) if all_strong else "–"}</b><small>top-quartile trust only</small></div>
    <div class="kpi"><span>Rolling eval, 20 pitcher-seasons</span><b>{pct(pooled_jev)}</b><small>{pooled_n:,} later-game pitches · base {pct(pooled_base)}</small></div>
    <div class="kpi"><span>Jev per decision</span><b>~0.2 s</b><small>≈ $0.003 per 1,000 pitches</small></div>
  </div>
</header>

<section>
  <span class="eyebrow">sample runs</span><h2>Three YouTube innings, pitch by pitch</h2>
  <p>Each column is one detected delivery. The bar is Jev's probability: up for slider, down for four-seam fastball; the dot is the calibrated vision model. Below it: the pitch he actually threw, from the MLB game feed. The model for each video was trained only on games before that date. Each card shows the frame where the leg lift started, the glove crop the model read, and both probability readouts.</p>
  {"".join(sections) or "<p>No runs found.</p>"}
</section>

{pre_html}

<section class="how">
  <span class="eyebrow">under the hood</span><h2>How Jev makes the call</h2>
  <p>Jev is a discriminative “System One” model from TypeSafe. It does not see pixels. pitchtip turns video into evidence, then asks Jev one <code>Choice</code> question with the pitcher's arsenal as the options. Jev returns a probability for every option. Below is the real request and answer for the first Miller pitch of the Sep 6 game.</p>
  <div class="pipe"><svg viewBox="0 0 980 120" role="img" aria-label="pipeline">
    <defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="currentColor" style="color:var(--muted)"/></marker></defs>
    <rect class="box" x="4" y="22" width="150" height="76" rx="10"/><text x="79" y="54" text-anchor="middle">Broadcast frames</text><text x="79" y="74" text-anchor="middle" class="sub">CF camera, 15 fps</text>
    <rect class="box" x="190" y="22" width="170" height="76" rx="10"/><text x="275" y="50" text-anchor="middle">Pose cascade</text><text x="275" y="68" text-anchor="middle" class="sub">YOLO11n tracks · YOLO11x</text><text x="275" y="84" text-anchor="middle" class="sub">re-reads the set + lift</text>
    <rect class="box" x="396" y="22" width="180" height="76" rx="10"/><text x="486" y="50" text-anchor="middle">Behavior features</text><text x="486" y="68" text-anchor="middle" class="sub">arm angles, hand path,</text><text x="486" y="84" text-anchor="middle" class="sub">glove image (DINOv2)</text>
    <rect class="box" x="612" y="22" width="170" height="76" rx="10"/><text x="697" y="50" text-anchor="middle">Expert models</text><text x="697" y="68" text-anchor="middle" class="sub">body · glove · posture</text><text x="697" y="84" text-anchor="middle" class="sub">+ similar deliveries</text>
    <rect class="jev" x="818" y="22" width="156" height="76" rx="10"/><text x="896" y="54" text-anchor="middle" class="jevt">Jev Choice</text><text x="896" y="74" text-anchor="middle" class="jevt" style="font-weight:400;font-size:11px">FF 84% · SL 16%</text>
    <path class="arrow" d="M156 60 H186" marker-end="url(#ah)"/><path class="arrow" d="M362 60 H392" marker-end="url(#ah)"/><path class="arrow" d="M578 60 H608" marker-end="url(#ah)"/><path class="arrow" d="M784 60 H814" marker-end="url(#ah)"/>
  </svg></div>
  <div class="panes">
    <div class="pane"><h4>1 · Expert opinions</h4>{experts}<p class="q">Reliability Jev is told for Miller: glove {st["expert_reliability_for_this_pitcher"]["glove"]:.2f}, body {st["expert_reliability_for_this_pitcher"]["body"]:.2f}, posture {st["expert_reliability_for_this_pitcher"]["linear"]:.2f}.</p></div>
    <div class="pane"><h4>2 · Track record on unseen games</h4><p class="q">How often the vision model was right at each confidence level. This tells Jev how far to trust it.</p>{track}</div>
    <div class="pane"><h4>3 · 25 most similar past deliveries</h4><div class="nn">{nn_dots}</div><p class="q">{nn.get("SL", 0)} sliders, {nn.get("FF", 0)} fastballs. Closest five: {" ".join(st["most_similar_past_deliveries"]["closest_5"])}.</p><h4>Known tells (today's reading)</h4><ul class="tells">{tells}</ul></div>
    <div class="pane answer"><h4>Jev's answer · {ex["answer"]["latency_s"] * 1000:.0f} ms</h4><b>{ex["answer"]["choice"]}</b>{bars(ex["answer"]["probabilities"])}<p class="q">Actual pitch: <strong>{ex["actual"]}</strong>. Base rates were SL {pct(st["arsenal_base_rates"]["SL"])}, FF {pct(st["arsenal_base_rates"]["FF"])}.</p></div>
  </div>
  <div class="notes">
    <div class="note"><b>One question, typed answer</b><p>Each call is a single <code>Choice</code> with the arsenal as criteria. Jev returns the pick, a confidence, and a probability per pitch. There is no free text to parse.</p></div>
    <div class="note"><b>Evidence chosen per pitcher</b><p>pitchtip tests evidence bundles on each pitcher's most recent games and keeps the best. Weak signal gets plain probabilities. Strong signal adds the track record and similar deliveries.</p></div>
    <div class="note"><b>Trust comes from the vision model</b><p>Jev's own confidence is close to 0 or 1 on almost every pitch. pitchtip ranks calls by the calibrated vision model's belief in Jev's pick. The top quarter of those calls run 80–95% correct.</p></div>
    <div class="note"><b>Cheap enough for every pitch</b><p>About 2,000 input tokens per call at $0.042 per million: roughly $0.003 per thousand pitches, answered in 0.1–0.2 s.</p></div>
  </div>
  <details><summary>Raw request and response (JSON)</summary><pre>{esc(json.dumps({"question": ex["question"], "state": ex["state"], "answer": ex["answer"]}, indent=1))}</pre></details>
</section>

<section>
  <span class="eyebrow">scale</span><h2>Across 20 pitcher-seasons</h2>
  <p>Fastball vs offspeed. Each later block of games is predicted from all earlier games, the way the system would run during a season. Gray is the base rate (always guess the most common pitch), orange the local vision model, blue Jev.</p>
  <div class="dotsbox">{dotplot(R)}</div>
  <div class="tblwrap"><table><thead><tr><th>pitcher-season</th><th>pitches</th><th>base</th><th>local</th><th>Jev</th><th>lift (pts)</th><th>top-25% trust</th></tr></thead><tbody>{table_rows}</tbody></table></div>
</section>

<section>
  <span class="eyebrow">limits</span><h2>What this does not show</h2>
  <ul class="caveats">
    <li>The YouTube runs are one pitcher in three innings. Treat them as a demonstration. The rolling evaluation over {pooled_n:,} later-game pitches is the reliable number.</li>
    <li>Tips are personal and seasonal. Each model is per pitcher-season and needs retraining as games come in.</li>
    <li>Only what the center-field camera sees can be learned. Glove-face and mouth tells need a front view.</li>
    <li>A detection that matches no pitch in the feed (a replay or a pickoff move) is shown but not scored.</li>
  </ul>
</section>
</main>
'''
(OUT / "index.html").write_text(page)
print("report written;", len(sections), "videos;", all_ok, "/", all_m)
