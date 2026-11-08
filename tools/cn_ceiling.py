"""How much is timing worth? DFL, held-out change net decoded, then oracles. Aggregate only.

    python tools/cn_ceiling.py [--tag c1dlbd]

  snap s   every predicted turnover within s seconds of a true turnover is moved onto it
           (the possession in between rewritten to the new team / old team accordingly);
  add      every true turnover with no predicted one within 2 s is written in (the new team's
           possession from the truth, up to the next true change);
  both     snap 2 s + add.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
import change_net as C  # noqa: E402
import dfl_data as dd  # noqa: E402
import miss_profile as M  # noqa: E402
from pipeline.decode import H, A, D  # noqa: E402

FPS = 25


def snap(lab, out, runs, sec):
    o = out.copy()
    tc = np.array([f for f, k in M.changes(lab, runs) if k == 0])
    pc = [f for f, k in M.changes(pm.fill_abstentions(out, runs), runs) if k == 0]
    for p in pc:
        if not len(tc):
            break
        t = tc[np.argmin(np.abs(tc - p))]
        if 0 < abs(t - p) <= sec * FPS:
            a, b = min(t, p), max(t, p)
            o[a:b] = lab[a:b] if np.all(np.isin(lab[a:b], (H, A))) else o[a:b]
    return o


def add(lab, out, runs, sec=2):
    o = out.copy()
    pc = np.array([f for f, k in M.changes(pm.fill_abstentions(out, runs), runs) if k == 0])
    for a, b, k in pm.possession_spells(lab, runs):
        if a == 0 or lab[a - 1] not in (H, A):
            continue
        if len(pc) and np.min(np.abs(pc - a)) <= sec * FPS:
            continue
        o[a:b] = k
    return o


def main():
    import cn_dfl
    import eval_touch_run as tr
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="c1dlbd")
    args = ap.parse_args()
    rows = []
    for g in dd.GAMES:
        with np.load(cn_dfl.path(g)) as z:
            lab, blk = z["lab"], [tuple(b) for b in z["blk"]]
        out = C.viterbi(cn_dfl.heldout(args.tag, g), blk, 0.5, -3.0)
        runs = pm.valid_runs(lab)
        r = [tr.score(lab, out)]
        for s in (0.5, 1, 2):
            r.append(tr.score(lab, snap(lab, out, runs, s)))
        r.append(tr.score(lab, add(lab, out, runs)))
        r.append(tr.score(lab, add(lab, snap(lab, out, runs, 2), runs)))
        rows.append(r)
    m = np.array(rows).mean(axis=0)
    print("{} on DFL (held out): {:.4f}".format(args.tag, m[0]))
    for nm, v in zip(("snap 0.5 s", "snap 1 s", "snap 2 s", "add missed (> 2 s)", "snap 2 s + add"), m[1:]):
        print("  {:<20} {:.4f} ({:+.4f})".format(nm, v, v - m[0]))


if __name__ == "__main__":
    main()
