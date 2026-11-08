"""Dead-ball decoder settings for v31_d1, on DFL + SkillCorner + PFF, then the public window. Aggregate only.

    python tools/dead_bias_sweep.py ext [--every 4]     # DFL 7 + every 4th SkillCorner / PFF match
    $env:MTAH_VARIANT="_kt"; python tools/dead_bias_sweep.py ours --set 0.2,0.5,25

The live / dead decoder (pipeline/refine.decode_dead) scores dead frames beta * logit(P dead) +
gamma (hand D) or - gamma (hand live) + bias, with `cost` per change. v31 uses beta 0.5, gamma 0.5,
cost 25, bias 0. Grid: bias x gamma x cost. Prints spellF1 per set vs the v31 setting (mean, up /
down), D share and stoppages found within 1 s; ours: v31_d1_kt and v31_kt rebuilt with the setting,
written to evaluation/results and scored on the public window.
"""

import argparse
import itertools
import os
import sys
from multiprocessing import Pool

os.environ.setdefault("TC_ROWS", "v2")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

GRID = list(itertools.product((0.0, 0.1, 0.2, 0.3), (0.5, 0.3), (25.0, 15.0)))


def ext_job(job):
    name, g = job
    import possession_metrics as pm
    import eval_dead_v30 as ev
    import eval_touch_run as tr
    import turnover_cls as T
    import miss_profile as M
    from pipeline import refine
    if name == "dfl":
        import dfl_data as dd
        import train_ext_heads as th
        with np.load(os.path.join(dd.CACHE, "refine_{}.npz".format(tr.TEAM_TAG))) as cc:
            p_ref = np.mean([cc["{}__{}".format(g, s)] for s in range(tr.TEAM_SEEDS)], axis=0)
        with np.load(os.path.join(ROOT, "data", "cache", "ext_heads_d1.npz")) as z:
            heads = {k: z[k] for k in z.files if k.startswith("dfl_" + g)}
        lab, hand, p1, f, blk, _, _ = th.dfl_data()[g]
        pdead = th.seeds_mean(heads, "dfl_" + g)
        period = np.zeros(len(lab), np.int8)
        for i, (a, b) in enumerate(blk, start=1):
            period[a:b] = i
        c = {"hand": hand, "p1": p1, "p_ref": p_ref, "vote": tr.run_vote(f, ev.VOTE), "period": period}
    else:
        import run_v31_external as rx
        import pair_cls as P
        c, f = T.load_cache(name, g)
        pdead = P.pdead_of(name, g)
        blk = rx.blocks(c["period"])
        lab = c["labels"]
    runs = pm.valid_runs(lab)
    res = []
    for bias, gamma, cost in GRID:
        refine.DEAD_BIAS = bias
        ev.SETTING = (0.5, gamma, cost)
        codes = T.base_chain(c, f, blk, pdead)
        bd = M.boundaries(lab, pm.fill_abstentions(codes, runs), runs)[1]
        res.append((tr.score(lab, codes), np.mean(codes == 2), bd[1][2] / max(bd[0], 1)))
    return name, np.array(res)


def ext_cmd(args):
    import turnover_cls as T
    import dfl_data as dd
    ext = [(n, g) for n, g, _ in T.ext_list()]
    jobs = [("dfl", g) for g in dd.GAMES] + ext[::args.every]
    with Pool(args.jobs) as p:
        out = p.map(ext_job, jobs, chunksize=1)
    ref = GRID.index((0.0, 0.5, 25.0))
    print("dead decoder grid on v31_d1 (beta 0.5); spellF1 mean (up/down vs v31's setting) | D share | stops in 1 s")
    sets = ("dfl", "skillcorner", "pff")
    print("{:>5} {:>5} {:>5} | ".format("bias", "gamma", "cost") + " | ".join("{:^28}".format(s) for s in sets))
    for i, (bias, gamma, cost) in enumerate(GRID):
        line = "{:>5g} {:>5g} {:>5g} | ".format(bias, gamma, cost)
        for s in sets:
            R = np.array([r for n, r in out if n == s])
            if not len(R):
                continue
            v, r0 = R[:, i, 0], R[:, ref, 0]
            line += "{:.4f} ({:>2}/{:<2}) {:>4.0%} {:>4.0%} | ".format(
                v.mean(), int(np.sum(v > r0 + 1e-9)), int(np.sum(v < r0 - 1e-9)), R[:, i, 1].mean(), R[:, i, 2].mean())
        print(line)


def ours_cmd(args):
    import eval_dead_v30 as ev
    import ours_chain as oc
    from pipeline import refine
    bias, gamma, cost = (float(v) for v in args.set.split(","))
    refine.DEAD_BIAS = bias
    ev.SETTING = (0.5, gamma, cost)
    tag = "db{:g}g{:g}c{:g}".format(bias, gamma, cost).replace(".", "")
    for nm, prefix in (("v31_d1", os.path.join("models", "dead_ext_d1")), ("v31", None)):
        codes, _ = oc.ours_v31(dead_prefix=prefix)
        print("  " + str(oc.write_and_score(codes, "{}_{}{}".format(nm, tag, os.environ.get("MTAH_VARIANT", "")))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("ext", "ours"))
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--set", default="0,0.5,25")
    args = ap.parse_args()
    {"ext": ext_cmd, "ours": ours_cmd}[args.cmd](args)


if __name__ == "__main__":
    main()
