"""Emit the paper table's Pendulum + Reacher rows with error bounds.

Reads the two bounds artefacts (Pendulum/table_row.json, Reacher/table_bounds.json)
and prints the full seven-column table in the paper's `mean ± err` format, plus a
LaTeX body ready to paste into the manuscript.  Cells with no run print as an
em-dash rather than a partial number.
"""
import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
COLS = ["SFIR", "Retrain full", "No retrain", "No forget +no retrain",
        "No adapt", "No forget", "Oracle"]
# (display label, source, key within that source)
ROWS = [
    ("Pendulum-v1", "$m = 1 \\to 4$",   "pend", "m = 1 -> 4"),
    ("Pendulum-v1", "$g = 10 \\to 15$", "pend", "g = 10 -> 15"),
    ("Reacher-v4",  "$k_s = 0.5$",      "reac", "k_s = 0.5"),
    ("Reacher-v4",  "$k_s = 5$",        "reac", "k_s = 5"),
    ("Reacher-v4",  "$\\tau = 90$",     "reac", "tau = 90"),
    ("Reacher-v4",  "$\\tau = 150$",    "reac", "tau = 150"),
]


def load():
    p = json.loads((ROOT / "Pendulum" / "table_row.json").read_text())
    r = json.loads((ROOT / "Reacher" / "table_bounds.json").read_text())
    return {"pend": p, "reac": r}


def fmt(e, tex=False):
    if not e:
        return "—" if not tex else "---"
    pm = "$\\pm$" if tex else "±"
    return f"{e['mean']:.1f} {pm} {e['sem']:.1f}"


def main():
    src = load()
    w = 22
    print("PAPER TABLE — return, mean ± 1 s.e.m.  (n = 100 seeds per cell)\n")
    print(f"{'Environment':<13}{'Setting':<17}" + "".join(f"{c:>{w}}" for c in COLS))
    for env, setting, s, key in ROWS:
        d = src[s].get(key, {})
        cells = "".join(f"{fmt(d.get(c)):>{w}}" for c in COLS)
        plain = (setting.replace("$", "").replace("\\to", "->")
                 .replace("\\tau", "tau").strip())
        print(f"{env:<13}{plain:<17}{cells}")

    print("\n\n% ---- LaTeX body ----")
    for env, setting, s, key in ROWS:
        d = src[s].get(key, {})
        best = min((c for c in COLS if d.get(c) and c != "Oracle"),
                   key=lambda c: -d[c]["mean"], default=None)
        cells = []
        for c in COLS:
            v = fmt(d.get(c), tex=True)
            if c == best and d.get(c):
                v = "\\textbf{" + v + "}"
            cells.append(v)
        print(f"{env} & {setting} & " + " & ".join(cells) + " \\\\")

    missing = [(e, k, c) for e, _, s, k in ROWS
               for c in COLS if not src[s].get(k, {}).get(c)]
    if missing:
        print("\n% missing cells:")
        for e, k, c in missing:
            print(f"%   {e} {k}: {c}")


if __name__ == "__main__":
    main()
