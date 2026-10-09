"""Inject the merged ablation grid (table_row.json) into the HTML report.

NB: the payload MUST NOT be called A or B -- curve()/gain() declare locals
A (=D[scen].arms) and B (=bottom margin), which shadow globals of those names
and silently blank every chart.  Hence ABL / BND.

Idempotent: replaces any previously injected `const A=` payload, so it can be
re-run as more shards land.
"""
import json, pathlib, re

HERE = pathlib.Path(__file__).parent
ROWS = ["SFIR", "Retrain full", "No retrain", "No forget +no retrain",
        "No adapt", "No forget", "Oracle"]
LBL = {"SFIR": "head_1l (SFIR)", "Retrain full": "retrain full",
       "No retrain": "no retrain", "No forget +no retrain": "no forget + no retrain",
       "No adapt": "no adapt", "No forget": "no forget (1L)", "Oracle": "oracle"}
COL = {"SFIR": "var(--s-1l)", "Oracle": "var(--ctx)"}

RENDER = """
const ABL_ROWS=%s,ABL_LBL=%s,ABL_COL=%s;
document.getElementById('tb3').innerHTML=ABL_ROWS.filter(r=>ABL.mass[r]||ABL.grav[r]).map(r=>{
  const cell=e=>{if(!e)return '<td class="sep">—</td><td>—</td><td>—</td>';
    const v=e.vs?((e.vs.diff>0?'+':'')+e.vs.diff.toFixed(1)
      +' <span class="ci">p='+(e.vs.p<1e-4?e.vs.p.toExponential(1):e.vs.p.toFixed(3))+'</span>'):'—';
    return `<td class="sep">${e.mean.toFixed(1)} <span class="pm">± ${e.sem.toFixed(1)}</span></td>`
      +`<td class="ci">[${e.ci95[0].toFixed(0)}, ${e.ci95[1].toFixed(0)}]</td><td>${v}</td>`;};
  return `<tr class="${r==='SFIR'?'hi':''}">`
    +`<td class="arm"><span class="dot" style="background:${ABL_COL[r]||'var(--ctx-soft)'}"></span>${ABL_LBL[r]}</td>`
    +cell(ABL.mass[r])+cell(ABL.grav[r])+'</tr>';}).join('');
"""


def main():
    tr = json.load(open(HERE / "table_row.json"))
    payload = {"mass": tr.get("m = 1 -> 4", {}), "grav": tr.get("g = 10 -> 15", {})}
    p = HERE / "pendulum_depth_report.html"
    s = p.read_text()
    blob = "const ABL=" + json.dumps(payload, separators=(",", ":")) + ";\n" + (
        RENDER % (json.dumps(ROWS), json.dumps(LBL), json.dumps(COL)))
    # Delimited block so re-runs replace rather than stack (a second `const ABL=`
    # is a redeclaration error that kills the whole script).
    start, end = "/*ABL-START*/", "/*ABL-END*/"
    block = f"{start}\n{blob}{end}\n"
    if start in s:
        i, j = s.index(start), s.index(end) + len(end) + 1
        s = s[:i] + block + s[j:]
    else:
        anchor = "function legend(id,items){"
        assert anchor in s
        s = s.replace(anchor, block + "\n" + anchor, 1)
    p.write_text(s)
    have = {k: len(v) for k, v in payload.items()}
    print(f"injected ablation grid: {have}")


if __name__ == "__main__":
    main()
