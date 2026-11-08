"""v31 decides dead ball and team; the change net (c1dlbd) may add turnovers. Aggregate only.

    python tools/cn_hybrid.py ext [--tags c1dlbd]                       # DFL + SkillCorner + PFF, held out
    $env:MTAH_VARIANT="_kt"; python tools/cn_hybrid.py ours --tags c1dlbd --rule 0.7,long,1

Candidate: a local maximum (+-6 frames) of the net's P(turnover to team k) >= thr, where the base is
live and on the other team over +-2 frames and has no turnover to k within 1 s. Gates:
  any    no gate;
  long   in the last 2 s the ball (seen frames) travelled >= 20 m, or was >= 2 m high, or >= 15 m/s;
  ld     long, or a duel (both teams' nearest within 2.5 m at the peak +-5 frames).
Extent: from the peak to the base's next change; with ret = 1, an opposite peak >= 0.3 within 5 s
after which the net's class head favours the old team ends it there (a short possession).
Base: v31_d1 (held out on DFL, rebuilt on SkillCorner / PFF); ours: v31_kt and v31_d1_kt.
Several tags are averaged. Prints spellF1 vs the base per set (mean, up / down), turnovers found
within 1 s; ours writes evaluation/results/<base>_hy_<rule>.csv and prints the public line.
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
import short_net as S  # noqa: E402
import change_net as C  # noqa: E402

FPS = 25
H, A, D = 0, 1, 2
RULES = list(itertools.product((0.5, 0.7, 0.85), ("any", "long", "ld"), (0, 1)))
BASE_DIR = os.path.join(C.OUT, "base_v31d1")


def context(x):
    """(long, duel) bool arrays from the change-net inputs (base channel order)."""
    from scipy.ndimage import maximum_filter1d, minimum_filter1d
    names = S.SHARED + ["h_" + k for k in S.BASE_TEAM] + ["a_" + k for k in S.BASE_TEAM]
    col = {k: names.index(k) for k in names}
    seen = x[:, col["seen"]] > 0.5
    bx = np.where(seen, x[:, col["bx"]], np.nan)
    by = np.where(seen, x[:, col["by"]], np.nan)
    n = len(x)
    travel = np.zeros(n)
    for lag in (25, 50):
        d = np.full(n, 0.0)
        d[lag:] = np.nan_to_num(np.hypot(bx[lag:] - bx[:-lag], by[lag:] - by[:-lag]))
        travel = np.maximum(travel, d)
    hz = maximum_filter1d(np.where(seen, x[:, col["bz"]], 0), 51, origin=25)
    sp = maximum_filter1d(np.where(seen, x[:, col["bspd"]], 0), 51, origin=25)
    long_ = (maximum_filter1d(travel, 51, origin=25) >= 20) | (hz >= 2) | (sp >= 15)
    duel = (minimum_filter1d(x[:, col["h_d1"]], 11) <= 2.5) & (minimum_filter1d(x[:, col["a_d1"]], 11) <= 2.5)
    return long_, duel


def hybrid(base, pr, long_, duel, thr, gate, ret):
    from scipy.ndimage import maximum_filter1d
    out = base.copy()
    n = len(base)
    chg = np.flatnonzero(base[1:] != base[:-1]) + 1
    cands = []
    for k, new in ((0, H), (1, A)):
        p = pr[:, k]
        pk = np.flatnonzero((p >= thr) & (p == maximum_filter1d(p, 13)))
        old = A if new == H else H
        bt = np.flatnonzero((base[1:] == new) & (base[:-1] == old)) + 1
        for t in pk:
            if t < 2 or t > n - 3 or np.any(base[t - 2:t + 3] != old):
                continue
            if len(bt) and np.min(np.abs(bt - t)) <= FPS:
                continue
            if gate == "long" and not long_[t]:
                continue
            if gate == "ld" and not (long_[t] or duel[t]):
                continue
            cands.append((t, new, old))
    for t, new, old in sorted(cands):
        if out[t] != old:
            continue
        nx = chg[chg > t]
        e = int(nx[0]) if len(nx) else n
        if ret:
            q = pr[t + 6:min(t + 5 * FPS, e), 1 if new == H else 0]
            if len(q) and q.max() >= 0.3:
                r = t + 6 + int(np.argmax(q))
                co, cn = C.N_TGT + (1 if new == H else 0), C.N_TGT + (0 if new == H else 1)
                aft = pr[r:min(r + 25, n)]
                if len(aft) and aft[:, co].mean() > aft[:, cn].mean():
                    e = r
        seg = out[t:e]
        out[t:e] = np.where(seg == old, new, seg)
    return out


def ext_data(job):
    name, g = job
    import cn_dfl
    if name == "dfl":
        with np.load(cn_dfl.path(g)) as z:
            return name, g, z["lab"], z["base"], z["x"].astype(np.float32)
    import turnover_cls as T
    import run_v31_external as rx
    import pair_cls as P
    path = os.path.join(BASE_DIR, "{}_{}.npy".format(name, g))
    c, f = T.load_cache(name, g)
    if os.path.isfile(path):
        base = np.load(path)
    else:
        base = T.base_chain(c, f, rx.blocks(c["period"]), P.pdead_of(name, g))
        os.makedirs(BASE_DIR, exist_ok=True)
        np.save(path, base)
    x = np.load(os.path.join(S.OUT, "in_{}_{}.npy".format(name, g))).astype(np.float32)
    return name, g, c["labels"], base, x


def ext_eval(job):
    name, g, lab, base, x, pr = job
    import eval_touch_run as tr
    import possession_metrics as pm
    import miss_profile as M
    long_, duel = context(x)
    runs = pm.valid_runs(lab)

    def rec(codes):
        b = M.boundaries(lab, pm.fill_abstentions(codes, runs), runs)[0]
        return b[1][2] / max(b[0], 1)
    res = [(tr.score(lab, base), rec(base))]
    for thr, gate, ret in RULES:
        o = hybrid(base, pr, long_, duel, thr, gate, ret)
        res.append((tr.score(lab, o), rec(o)))
    return name, np.array(res)


def ext_cmd(args):
    import turnover_cls as T
    import dfl_data as dd
    preds = []
    for tg in args.tags:
        with np.load(os.path.join(C.OUT, "pred_{}.npz".format(tg))) as z:
            preds.append({k: z[k] for k in z.files})
    keys = [("dfl", g) for g in dd.GAMES] + [(n, g) for n, g, _ in T.ext_list()][::args.every]
    with Pool(args.jobs) as p:
        data = p.map(ext_data, keys, chunksize=1)
    jobs = [(n, g, lab, base, x, np.mean([pp["{}_{}".format(n, g)].astype(np.float32) for pp in preds], 0))
            for n, g, lab, base, x in data]
    with Pool(args.jobs) as p:
        out = p.map(ext_eval, jobs, chunksize=1)
    sets = ("dfl", "skillcorner", "pff")
    print("hybrid: v31_d1 + {} turnovers; spellF1 (up/down vs base), turnovers found within 1 s".format("+".join(args.tags)))
    head = "{:<16} | ".format("rule") + " | ".join("{:^26}".format(s) for s in sets)
    print(head)
    for i, nm in enumerate(["base v31_d1"] + ["{:g} {} ret{}".format(*r) for r in RULES]):
        line = "{:<16} | ".format(nm)
        for s in sets:
            R = np.array([r for n, r in out if n == s])
            if not len(R):
                continue
            v, r0 = R[:, i, 0], R[:, 0, 0]
            line += "{:.4f} ({:>2}/{:<2}) {:>5.0%}   | ".format(
                v.mean(), int(np.sum(v > r0 + 1e-9)), int(np.sum(v < r0 - 1e-9)), R[:, i, 1].mean())
        print(line)


def ours_cmd(args):
    import ours_chain as oc
    import possession_metrics as pm
    from games import GAME, GAMES
    prs = [np.load(os.path.join(C.OUT, "ours_{}.npz".format(tg)))["pr"].astype(np.float32) for tg in args.tags]
    pr = np.mean(prs, 0)
    x = np.load(os.path.join(C.OUT, "ours_c1dl.npz"))["x"].astype(np.float32)
    long_, duel = context(x)
    var = os.environ.get("MTAH_VARIANT", "")
    rules = [tuple(r.split(",")) for r in args.rule] if args.rule else [(str(a), b, str(c)) for a, b, c in RULES]
    for bname in ("v31", "v31_d1"):
        path = os.path.join(ROOT, "evaluation", "results", "{}{}.csv".format(
            bname if bname == "v31" else "v31_d1_basecheck", var if bname == "v31" else ""))
        base = pm.label_codes(pm.read_label_csv(path))
        base = np.where(base < 0, 3, base).astype(np.int8)
        print("  " + str(oc.write_and_score(base, "{}_hybase{}".format(bname, var))))
        for thr, gate, ret in rules:
            o = hybrid(base, pr, long_, duel, float(thr), gate, int(ret))
            nm = "{}_hy_{}_{}_{}r{}{}".format(bname, "+".join(args.tags), thr, gate, ret, var).replace(".", "")
            print("  {:<44} changed {:.2%} | ".format(nm, np.mean(o != base)) + str(oc.write_and_score(o, nm)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("ext", "ours"))
    ap.add_argument("--tags", nargs="+", default=["c1dlbd"])
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--rule", nargs="*", help="thr,gate,ret (ours); default: the whole grid")
    args = ap.parse_args()
    {"ext": ext_cmd, "ours": ours_cmd}[args.cmd](args)


if __name__ == "__main__":
    main()
