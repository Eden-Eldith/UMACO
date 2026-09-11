"""Print the per-iteration trajectory of a System U run (json from --out) around its burst/reset events."""
import json, sys

d = json.load(open(sys.argv[1]))
ev = {e[0]: e for e in d["events"]}
print("result", d["result"])
print("it best cur perf   P    |Psi| phase  Hp   pers  live   Rmax  Mmax  div   pw   event")
for r in d["stats"]:
    e = ev.get(r["it"])
    tag = f"{e[1]} {e[2]} g={e[4]}" if e and e[1] == "burst" else (f"reset {e[2]}" if e else "")
    if e or r["it"] % 10 == 0 or r["it"] <= 5:
        print(f"{r['it']:3d} {r['best']:4d} {r['cur_best']:4d} {r['perf']:.3f} {r['P']:.3f} {r['Psi']:.2f} "
              f"{r['phase']:+.2f} {r['Hp']:.2f} {r['pers']:.3f} {r['live']:.3f} {r['Rmax']:6.2f} {r['Mmax']:5.2f} "
              f"{r.get('div', float('nan')):.3f} {r.get('perf_walk', float('nan')):.2f}  {tag}")
