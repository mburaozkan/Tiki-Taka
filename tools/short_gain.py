"""Why the short-net's catches give less spellF1 than the oracle: decomposition. Aggregate only.

    python tools/short_gain.py [--tag s1] [--hi 0.7] [--over-d]

On the 84 external matches, base v31_d1 and:
  all pasted       the short-net spells as tools/short_net.py eval (the real result)
  right only       only the pasted spells that match a true possession of their team (IoU >= 0.5)
  right, true edges  those same matches written with the true possession's boundaries
  oracle < 5 s     every true possession < 5 s written in (the ceiling of a perfect short detector)
Also: the share of pasted spells that are right, what the false ones land on (inside a true
possession of the other team, of the same team, or in D), and the IoU of the caught short
possessions (spellF1 averages over IoU 0.3 / 0.4 / 0.5 / 0.6 / 0.7, so a catch at IoU 0.5-0.6
counts at 3 of 5 thresholds). Prints aggregates only.
"""

import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from pipeline.decode import H, A, D  # noqa: E402


def iou(a, b, c, e):
    return max(0, min(b, e) - max(a, c)) / float(max(b, e) - min(a, c))


def run(job):
    name, g, tag, hi, over_d = job
    import run_v31_external as rx
    import turnover_cls as T
    import pair_cls as P
    import short_net as S
    import eval_touch_run as tr
    with np.load(os.path.join(S.OUT, "pred_{}.npz".format(tag))) as z:
        pr = z["{}_{}".format(name, g)].astype(np.float32)
    c, f = T.load_cache(name, g)
    pdead = P.pdead_of(name, g)
    blk = rx.blocks(c["period"])
    lab = c["labels"]
    base = T.base_chain(c, f, blk, pdead)
    truth = pm.possession_spells(lab, pm.valid_runs(lab))
    ok = (H, A, D) if over_d else (H, A)
    cand = []
    for s, team in ((0, H), (1, A)):
        for a, b in S.spells_from(pr[:, s], hi, hi - 0.15):
            cand.append((float(pr[a:b, s].mean()), a, b, team))
    cand.sort()

    def write(spells):
        out = base.copy()
        for a, b, team in spells:
            seg = out[a:b]
            out[a:b] = np.where(np.isin(seg, ok), team, seg)
        return out

    right, right_true, land = [], [], {"other team": 0, "same team": 0, "D / ungraded": 0}
    for _, a, b, team in cand:
        best = max(((iou(a, b, ta, tb), (ta, tb)) for ta, tb, k in truth if k == team),
                   default=(0, None))
        if best[0] >= 0.5:
            right.append((a, b, team))
            right_true.append((best[1][0], best[1][1], team))
        else:
            mid = lab[(a + b) // 2]
            land["same team" if mid == team else "other team" if mid in (H, A) else
                 "D / ungraded"] += 1
    oracle = [(a, b, k) for a, b, k in truth if (b - a) / 25.0 < 5]
    allp = write([(a, b, t) for _, a, b, t in cand])
    scores = [tr.score(lab, x) for x in (base, allp, write(right), write(right_true), write(oracle))]
    # IoU of caught short possessions in the real result
    pred = pm.possession_spells(pm.fill_abstentions(allp, pm.valid_runs(lab)), pm.valid_runs(lab))
    ious = []
    for ta, tb, k in truth:
        if (tb - ta) / 25.0 < 5:
            best = max((iou(ta, tb, pa, pb) for pa, pb, j in pred if j == k), default=0)
            ious.append(best)
    return name, scores, len(cand), len(right), land, ious


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s1")
    ap.add_argument("--hi", type=float, default=0.7)
    ap.add_argument("--over-d", action="store_true")
    args = ap.parse_args()
    import turnover_cls as T
    jobs = [(n, g, args.tag, args.hi, args.over_d) for n, g, _ in T.ext_list()]
    with Pool(6) as pool:
        res = pool.map(run, jobs, chunksize=1)
    names = np.array([r[0] for r in res])
    sc = np.array([r[1] for r in res])
    lab = ["base v31_d1", "all pasted (real)", "right only", "right, true edges", "oracle < 5 s"]
    print("setting: hi {:g}{}".format(args.hi, ", also over D" if args.over_d else ""))
    for j, lb in enumerate(lab):
        print("  {:<20} SkillCorner {:.4f} ({:+.4f})  PFF {:.4f} ({:+.4f})".format(
            lb, sc[names == "skillcorner", j].mean(),
            sc[names == "skillcorner", j].mean() - sc[names == "skillcorner", 0].mean(),
            sc[names == "pff", j].mean(), sc[names == "pff", j].mean() - sc[names == "pff", 0].mean()))
    n_c, n_r = sum(r[2] for r in res), sum(r[3] for r in res)
    land = {k: sum(r[4][k] for r in res) for k in res[0][4]}
    print("pasted spells {}: right {:.0%}, false {:.0%} (false ones land: {})".format(
        n_c, n_r / n_c, 1 - n_r / n_c, ", ".join("{} {:.0%}".format(k, v / max(n_c - n_r, 1))
                                                  for k, v in land.items())))
    ious = np.concatenate([r[5] for r in res])
    print("true possessions < 5 s ({}), best IoU of a predicted spell: >= 0.7 {:.0%}, 0.6-0.7 {:.0%}, "
          "0.5-0.6 {:.0%}, 0.3-0.5 {:.0%}, < 0.3 {:.0%}".format(
              len(ious), np.mean(ious >= 0.7), np.mean((ious >= 0.6) & (ious < 0.7)),
              np.mean((ious >= 0.5) & (ious < 0.6)), np.mean((ious >= 0.3) & (ious < 0.5)),
              np.mean(ious < 0.3)))


if __name__ == "__main__":
    main()
