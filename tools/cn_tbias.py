"""Own decoder bias for turnovers (change_net.viterbi tbias) on DFL, chosen leave one match out. Aggregate only.

    python tools/cn_tbias.py [--tag c1dl]

Grid: emission weight w x stop/restart bias x turnover bias, on the held-out change-net prediction of
each DFL match. Prints the grid's mean spellF1 and turnover recall within 1 s, and the nested score
(setting picked on the other 6 matches) vs the fixed default (w 0.5, bias -3, tbias = bias).
"""

import argparse
import itertools
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

W = (0.5, 1.0)
BIAS = (-3.0, -1.5)
TB = (-4.5, -3.0, -2.0, -1.0, 0.0)


def run(job):
    import possession_metrics as pm
    import change_net as C
    import cn_dfl
    import eval_touch_run as tr
    import miss_profile as M
    tag, g = job
    with np.load(cn_dfl.path(g)) as z:
        lab, blk = z["lab"], [tuple(b) for b in z["blk"]]
    pr = cn_dfl.heldout(tag, g)
    runs = pm.valid_runs(lab)
    res = []
    for w, b, t in itertools.product(W, BIAS, TB):
        out = C.viterbi(pr, blk, w, b, tbias=t)
        bd = M.boundaries(lab, pm.fill_abstentions(out, runs), runs)[0]
        res.append((tr.score(lab, out), bd[1][2] / max(bd[0], 1), bd[2] / max(bd[0], 1)))
    return g, np.array(res)


def main():
    import dfl_data as dd
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="c1dl")
    args = ap.parse_args()
    with Pool(7) as pool:
        out = dict(pool.map(run, [(args.tag, g) for g in dd.GAMES]))
    R = np.stack([out[g] for g in dd.GAMES])            # (7, settings, 3)
    sets = list(itertools.product(W, BIAS, TB))
    print("{} on DFL, held out; grid mean over 7 matches".format(args.tag))
    print("{:>4} {:>6} {:>6} | {:>8} {:>10} {:>10}".format("w", "bias", "tbias", "spellF1",
                                                         "turn rec1s", "pred/true"))
    for i, (w, b, t) in enumerate(sets):
        m = R[:, i].mean(axis=0)
        print("{:>4g} {:>6g} {:>6g} | {:>8.4f} {:>9.1%} {:>10.2f}".format(w, b, t, *m))
    d0 = sets.index((0.5, -3.0, -3.0))
    nest = []
    for k in range(len(dd.GAMES)):
        oth = [j for j in range(len(dd.GAMES)) if j != k]
        best = int(np.argmax(R[oth, :, 0].mean(axis=0)))
        nest.append((R[k, best, 0], sets[best]))
    v = np.array([x[0] for x in nest])
    ref = R[:, d0, 0]
    print("nested (picked on the other 6): {:.4f} vs default {:.4f} ({:+.4f}, {}/{} up/down); picks: {}".format(
        v.mean(), ref.mean(), v.mean() - ref.mean(), int(np.sum(v > ref + 1e-9)), int(np.sum(v < ref - 1e-9)),
        sorted(set(x[1] for x in nest))))


if __name__ == "__main__":
    main()
