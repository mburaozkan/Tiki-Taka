"""Why are short possessions missed? Breakdown on the 84 external matches (SkillCorner + PFF). Aggregate only.

    python tools/short_miss.py [--net s1] [--gen 0.4] [--jobs 4]

Base: v31_d1 (turnover_cls.base_chain), the working base. Short net: its held-out per-frame
P(inside a short possession of H / A) (data/cache/short_net/pred_<net>.npz, each match scored by
the fold model that never saw it). Every true H / A possession shorter than SHORT_S gets one bucket:
  caught            the base has a same-team spell with IoU >= 0.5 with it
  boundary (base)   the base has the team there (any overlap) but its best IoU < 0.5: merged or
                    misplaced, a boundary / split problem rather than missing evidence
  decision          no base overlap; a short-net candidate spell (hysteresis at --gen, as
                    short_check) of that team has IoU >= 0.5: the evidence and edges are there,
                    only the keep / insert decision fails
  boundary (cand.)  a candidate of that team overlaps it but IoU < 0.5: edges too far off
  weak evidence     no candidate, but a touch-network peak of that team within +-0.5 s of its start
                    or P(inside) of that team >= gen - 0.15 somewhere inside it
  no evidence       none of the above
Split by length (< 2 s, 2-5 s) and by context (inside the other team's possession X>[t]>X, after
dead ball D>[t], other). Also: for missed short possessions, how many neighbouring true
possessions (directly before / after) the base misses too (the neighbour damage).
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
from pipeline.decode import H, A, D  # noqa: E402

FPS = 25
SHORT_S = 5.0
BUCKETS = ("caught", "boundary (base)", "decision", "boundary (cand.)", "weak evidence", "no evidence")


def iou(a, b, c, d):
    inter = max(0, min(b, d) - max(a, c))
    return inter / (max(b, d) - min(a, c))


def best_iou(spells, a, b, team):
    best, any_ov = 0.0, False
    for c, d, k in spells:
        if k == team and d > a and c < b:
            any_ov = True
            best = max(best, iou(a, b, c, d))
    return best, any_ov


def match(job):
    name, g, net, gen = job
    import turnover_cls as T
    import short_net as S
    import short_check as SC
    import run_v31_external as rx
    import pair_cls as P
    from pipeline import touch_net as tn
    c, f = T.load_cache(name, g)
    pdead = P.pdead_of(name, g)
    blk = rx.blocks(c["period"])
    base = T.base_chain(c, f, blk, pdead)
    lab = c["labels"]
    with np.load(os.path.join(S.OUT, "pred_{}.npz".format(net))) as z:
        pr = z["{}_{}".format(name, g)].astype(np.float32)
    truth = pm.possession_spells(lab, pm.valid_runs(lab))
    bsp = pm.possession_spells(base, pm.valid_runs(lab))
    cand = [(a, b, H if s == 0 else A) for a, b, s in SC.gen_spells(pr, gen)]
    tp = np.column_stack([1 - f["touch_p_home"] - f["touch_p_away"], f["touch_p_home"], f["touch_p_away"]])
    pk, pt = tn.pick(tp, 0.5)
    hit_long = []
    for a, b, k in truth:
        hit_long.append(best_iou(bsp, a, b, k)[0] >= 0.5)
    rows = []
    for i, (a, b, k) in enumerate(truth):
        ln = (b - a) / FPS
        if ln >= SHORT_S:
            continue
        s = 0 if k == H else 1
        bi, bov = best_iou(bsp, a, b, k)
        if bi >= 0.5:
            bucket = 0
        elif bov:
            bucket = 1
        else:
            ci, cov = best_iou(cand, a, b, k)
            if ci >= 0.5:
                bucket = 2
            elif cov:
                bucket = 3
            else:
                touch = np.any((np.abs(pk - a) <= 12) & (pt == s))
                sig = pr[a:b, s].max() >= gen - 0.15 if b > a else False
                bucket = 4 if (touch or sig) else 5
        prev = lab[a - 1] if a > 0 else -1
        nxt = lab[b] if b < len(lab) else -1
        other = A if k == H else H
        ctx = 0 if (prev == other and nxt == other) else 1 if prev == D else 2
        nb = [j for j in (i - 1, i + 1) if 0 <= j < len(truth)
              and (truth[j][1] == a or truth[j][0] == b)]
        nb_missed = sum(1 for j in nb if not hit_long[j])
        rows.append((bucket, ln < 2, ctx, len(nb), nb_missed))
    return np.array(rows, np.int64).reshape(-1, 5), len(truth)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", default="s1")
    ap.add_argument("--gen", type=float, default=0.4)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    import turnover_cls as T
    jobs = [(n, g, args.net, args.gen) for n, g, _ in T.ext_list()]
    with Pool(args.jobs) as pool:
        res = pool.map(match, jobs, chunksize=1)
    R = np.concatenate([r for r, _ in res])
    ntrue = sum(n for _, n in res)
    print("{} matches; {} true possessions, {} shorter than {:.0f} s ({:.0%})".format(
        len(res), ntrue, len(R), SHORT_S, len(R) / ntrue))
    print("\n{:<18} {:>12} {:>12} {:>12}".format("bucket", "< 2 s", "2-5 s", "all short"))
    for bi, bn in enumerate(BUCKETS):
        m = R[:, 0] == bi
        cells = []
        for sel in (R[:, 1] == 1, R[:, 1] == 0, np.ones(len(R), bool)):
            cells.append("{:>5} {:>4.0%}".format(int((m & sel).sum()), (m & sel).sum() / max(sel.sum(), 1)))
        print("{:<18} {:>12} {:>12} {:>12}".format(bn, *cells))
    print("\ncontext of the missed ones (not caught):")
    miss = R[:, 0] != 0
    for ci, cn in enumerate(("inside the other team's possession (X>[t]>X)", "after dead ball (D>[t])", "other")):
        m = miss & (R[:, 2] == ci)
        row = ", ".join("{} {}".format(BUCKETS[b], int((m & (R[:, 0] == b)).sum())) for b in range(1, 6))
        print("  {:<46} {:>5} ({:.0%}) | {}".format(cn, int(m.sum()), m.sum() / max(miss.sum(), 1), row))
    nb, nbm = R[miss, 3].sum(), R[miss, 4].sum()
    nb_c, nbm_c = R[~miss, 3].sum(), R[~miss, 4].sum()
    print("\nneighbour damage: next to a MISSED short possession the base also misses {} of {} adjacent true "
          "possessions ({:.0%}); next to a CAUGHT one {} of {} ({:.0%})".format(
              nbm, nb, nbm / max(nb, 1), nbm_c, nb_c, nbm_c / max(nb_c, 1)))


if __name__ == "__main__":
    main()
