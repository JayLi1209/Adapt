#!/usr/bin/env python3
"""Render cumulative-return curves (mean +/- 1 s.e.m. over seeds) as an HTML page.

Reads results/trace_curves.json (from build_trace_curves.py) and writes inline
SVG -- no chart library, no external data fetch, no runtime deps.

Series colors are the validated 4-slot categorical palette; oracle and no_adapt
are deliberately neutral grays (they are references, not findings), so the
categorical hues are spent only on the adaptation arms.
"""
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CURVES = os.path.join(HERE, "results", "trace_curves.json")
OUT = os.path.join(HERE, "results", "trace_page.html")

ORDER = ["oracle", "no_adapt", "nl1_forget_qv", "lin1_noforget", "act_rot"]
SLOT = {"oracle": "--ref", "no_adapt": "--ref2",
        "nl1_forget_qv": "--s1", "lin1_noforget": "--s2", "act_rot": "--s3"}

W, H = 640, 350
ML, MR, MT, MB = 62, 150, 14, 42


def nice_ticks(lo, hi, target=5):
    span = hi - lo
    if span <= 0:
        return [lo]
    step = 10 ** math.floor(math.log10(span / target))
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= target + 0.5:
            step *= m
            break
    t = math.ceil(lo / step) * step
    out = []
    while t <= hi + 1e-9:
        out.append(t)
        t += step
    return out


def chart(cell, title, sub):
    arms = cell["arms"]
    present = [a for a in ORDER if a in arms]
    if not present:
        return ""
    T = len(arms[present[0]]["mean"])
    pw, ph = W - ML - MR, H - MT - MB
    sx = pw / max(T - 1, 1)

    lo = min(min(m - s for m, s in zip(arms[a]["mean"], arms[a]["sem"])) for a in present)
    hi = max(max(m + s for m, s in zip(arms[a]["mean"], arms[a]["sem"])) for a in present)
    hi = min(hi, 0.0)
    lo -= (hi - lo) * 0.06 or 1.0

    def X(i):
        return ML + i * sx

    def Y(v):
        return MT + ph - (v - lo) / (hi - lo) * ph

    o = [f'<figure class="fig">',
         f'<figcaption><h3>{title}</h3><p>{sub}</p></figcaption>',
         f'<div class="svg-wrap">',
         f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Cumulative return '
         f'over {T} timesteps, {len(present)} arms, band is one standard error">']

    for tv in nice_ticks(lo, hi):
        y = Y(tv)
        o.append(f'<line class="grid" x1="{ML}" y1="{y:.1f}" x2="{ML+pw}" y2="{y:.1f}"/>')
        o.append(f'<text class="tick ty" x="{ML-9}" y="{y+3.5:.1f}">{tv:,.0f}</text>')
    for xv in (0, T // 2, T - 1):
        o.append(f'<text class="tick tx" x="{X(xv):.1f}" y="{MT+ph+21}">{xv}</text>')
    o.append(f'<text class="axname" x="{ML+pw/2:.1f}" y="{MT+ph+36}">TIMESTEP</text>')

    for a in present:
        v = arms[a]
        up = [(X(i), Y(v["mean"][i] + v["sem"][i])) for i in range(T)]
        dn = [(X(i), Y(v["mean"][i] - v["sem"][i])) for i in range(T)]
        d = ("M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in up) +
             " L" + " L".join(f"{x:.1f},{y:.1f}" for x, y in reversed(dn)) + " Z")
        o.append(f'<path class="band" style="fill:var({SLOT[a]})" d="{d}"/>')

    for a in present:
        v = arms[a]
        d = " ".join(("M" if i == 0 else "L") + f"{X(i):.1f},{Y(v['mean'][i]):.1f}"
                     for i in range(T))
        o.append(f'<path class="ln" style="stroke:var({SLOT[a]})" d="{d}"/>')
        o.append(f'<circle class="dot" style="fill:var({SLOT[a]})" '
                 f'cx="{X(T-1):.1f}" cy="{Y(v["mean"][-1]):.1f}" r="4"/>')

    lab = sorted((Y(arms[a]["mean"][-1]), a) for a in present)
    placed, prev = [], -1e9
    for y, a in lab:
        y = max(y, prev + 15)
        placed.append((y, a))
        prev = y
    for y, a in placed:
        o.append(f'<text class="dlab" style="fill:var({SLOT[a]})" x="{ML+pw+11}" '
                 f'y="{y+3.5:.1f}">{a} {arms[a]["final"]:,.0f}</text>')

    o.append("</svg></div></figure>")
    return "\n".join(o)


def table(cell):
    arms = cell["arms"]
    present = [a for a in ORDER if a in arms]
    rows = "".join(
        f"<tr><td>{a}</td><td>{arms[a]['final']:,.1f}</td>"
        f"<td>{arms[a]['sem'][-1]:.2f}</td><td>{arms[a]['n']}</td></tr>"
        for a in present)
    return ("<table><thead><tr><th>arm</th><th>final return</th>"
            f"<th>s.e.m.</th><th>n</th></tr></thead><tbody>{rows}</tbody></table>")


def main():
    if not os.path.exists(CURVES):
        sys.exit(f"missing {CURVES} -- run build_trace_curves.py first")
    with open(CURVES) as f:
        data = json.load(f)
    for k, c in data.items():
        print(k, {a: v["final"] for a, v in c["arms"].items()})


if __name__ == "__main__":
    main()
