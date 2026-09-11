"""Scaling driver for System U (system_u_v9_walksat.py, same folder): several n, seeds, families; one CSV.
GPU only. Records solve rate, iterations, wall time, and the internal state at the end / at success."""
import argparse, csv, json, os, sys, time
import numpy as np
import cupy as cp

sys.path.insert(0, os.path.dirname(__file__))
from system_u_v9_walksat import SystemU, gen_3sat_sat, host_recount  # noqa: E402


def run(n, ratio, family, seed, K, iters, kw):
    m = int(round(ratio * n))
    V, S, _, sd = gen_3sat_sat(n, m, seed, family == "planted")   # random instances verified SAT by CaDiCaL
    u = SystemU(V, S, n, K=K, seed=seed, **kw)
    res = u.solve(iters, verbose_every=0)
    last = u.stats[-1]
    nb = sum(1 for e in u.events if e[1] == "burst")
    regs = {}
    for e in u.events:
        if e[1] == "burst":
            regs[e[2]] = regs.get(e[2], 0) + 1
    nr = sum(1 for e in u.events if e[1] == "reset")
    # iteration at which best reached its final value
    first_best = next(r["it"] for r in u.stats if r["best"] == last["best"])
    return dict(n=n, m=m, ratio=ratio, family=family, seed=sd, K=K, solved=int(res["solved"]),
                iters=res["iters"], time=round(res["time"], 1), best=last["best"], first_best=first_best,
                bursts=nb, resets=nr, regimes=json.dumps(regs), P=round(last["P"], 3), Psi=round(last["Psi"], 3),
                phase=round(last["phase"], 3), alpha=round(last["alpha"], 3), beta=round(last["beta"], 3),
                rho=round(last["rho"], 3), Hp=round(last["Hp"], 3), live=round(last["live"], 3),
                Rmax=round(last["Rmax"], 3), Mmax=round(last["Mmax"], 3), perf=round(last["perf"], 4),
                walks=res["iters"] * K, sec_per_it=round(np.mean([r["dt"] for r in u.stats]), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=str, default="50,100,200,400")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--families", type=str, default="planted,random")
    ap.add_argument("--ratio", type=float, default=4.26)
    ap.add_argument("--K", type=int, default=128); ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--alpha_w", type=float, default=1.0); ap.add_argument("--flips", type=int, default=30)
    ap.add_argument("--homology_every", type=int, default=1)
    ap.add_argument("--perf_target", type=float, default=0.7)
    ap.add_argument("--kappa_sel", type=float, default=0.0); ap.add_argument("--eps_rel", type=float, default=0.0)
    ap.add_argument("--budget_gain", type=float, default=0.0); ap.add_argument("--flips_max", type=int, default=1000)
    a = ap.parse_args()
    kw = dict(alpha_w=a.alpha_w, flips_base=a.flips, homology_every=a.homology_every, perf_target=a.perf_target,
              kappa_sel=a.kappa_sel, eps_rel=a.eps_rel, budget_gain=a.budget_gain, flips_max=a.flips_max)
    rows = []
    fields = None
    for fam in a.families.split(","):
        for n in [int(x) for x in a.ns.split(",")]:
            for s in [int(x) for x in a.seeds.split(",")]:
                r = run(n, a.ratio, fam, s, a.K, a.iters, kw)
                rows.append(r)
                if fields is None:
                    fields = list(r.keys())
                    fh = open(a.out, "w", newline=""); w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
                w.writerow(r); fh.flush()
                print(f"{fam:8s} n={n:4d} s={s} solved={r['solved']} it={r['iters']:4d} best={r['best']:3d} "
                      f"t={r['time']:6.1f}s bursts={r['bursts']} {r['regimes']} resets={r['resets']} "
                      f"P={r['P']:.2f} Psi={r['Psi']:.2f} ph={r['phase']:+.2f} live={r['live']:.2f}", flush=True)
    fh.close()


if __name__ == "__main__":
    main()
