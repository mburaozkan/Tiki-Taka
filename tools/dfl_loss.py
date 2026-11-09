"""Where the submission (v31_d1_c1chdlm_e3k) still loses on DFL, held out. Aggregate only.

    python tools/dfl_loss.py          # needs data/cache/cn_e3k/dfl_*.npz (tools/cn_e3k.py dfl)

1. Oracle gains, each alone (spellF1, mean of 7): dead / live perfect (truth D -> D, a live frame we
   call D -> our previous team); team perfect on frames we call live (truth's team where both live);
   both. Shows whether the remaining loss is the team decision or the dead ball.
2. Every true H / A possession, by length (< 2, 2-5, 5-10, >= 10 s): caught (same team, IoU >= 0.5),
   else what we did there: called D (>= 50% of its frames D), other team (>= 50% the other team),
   merged (our same-team spell covers it and is > 2x longer), shifted / split (the rest).
3. For the missed ones, evidence at the true start (+-0.5 s): a touch-network peak of that team
   (P >= 0.5), the change net's turnover / restart head toward that team >= 0.3, the short net's
   start head of that team >= 0.3; 'none' if all are below. How much of the loss is detectable.
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
from pipeline.decode import H, A, D, X  # noqa: E402

FPS = 25
LENS = ((0, 2), (2, 5), (5, 10), (10, 1e9))


def iou(a, b, c, d):
    return max(0, min(b, d) - max(a, c)) / (max(b, d) - min(a, c))


def main():
    import eval_touch_run as tr
    from pipeline import touch_net as tn
    files = sorted(glob.glob(os.path.join(ROOT, "data", "cache", "cn_e3k", "dfl_*.npz")))
    orc = {k: [] for k in ("submission", "dead/live perfect", "team perfect (live)", "both")}
    rows = []
    for fp in files:
        z = np.load(fp)
        lab, fin = z["lab"], z["final"]
        cpr, spr = z["cpr"].astype(np.float32), z["spr"].astype(np.float32)
        runs = pm.valid_runs(lab)
        filled = pm.fill_abstentions(fin, runs)
        # oracles
        dl = filled.copy()
        dl[lab == D] = D
        live_called_d = np.isin(lab, (H, A)) & (dl == D)
        dl[live_called_d] = X
        dl = pm.fill_abstentions(dl, runs)
        tm = filled.copy()
        both_live = np.isin(lab, (H, A)) & np.isin(tm, (H, A))
        tm[both_live] = lab[both_live]
        bo = dl.copy()
        bl = np.isin(lab, (H, A)) & np.isin(bo, (H, A))
        bo[bl] = lab[bl]
        for k, x in (("submission", filled), ("dead/live perfect", dl), ("team perfect (live)", tm), ("both", bo)):
            orc[k].append(tr.score(lab, x))
        # possessions
        truth = pm.possession_spells(lab, runs)
        pred = pm.possession_spells(filled, runs)
        tp = np.column_stack([1 - z["tph"].astype(np.float32) - z["tpa"].astype(np.float32),
                              z["tph"].astype(np.float32), z["tpa"].astype(np.float32)])
        pk, pt = tn.pick(tp, 0.5)
        for a, b, k in truth:
            ln = (b - a) / FPS
            s = 0 if k == H else 1
            ov = [(iou(a, b, c, d), c, d) for c, d, k2 in pred if k2 == k and d > a and c < b]
            best = max(ov)[0] if ov else 0.0
            seg = filled[a:b]
            if best >= 0.5:
                kind = 0
            elif np.mean(seg == D) >= 0.5:
                kind = 1
            elif np.mean(seg == (A if k == H else H)) >= 0.5:
                kind = 2
            else:
                _, c, d = max(ov) if ov else (0, a, b)
                kind = 3 if (ov and c <= a + 3 and d >= b - 3 and (d - c) > 2 * (b - a)) else 4
            prev = lab[a - 1] if a > 0 else -1
            w = slice(max(a - 12, 0), a + 13)
            touch = bool(np.any((np.abs(pk - a) <= 12) & (pt == s)))
            chead = cpr[w, s].max() if prev in (H, A) else cpr[w, 3 + s].max()   # turnover / restart to that team
            shead = spr[w, 2 + s].max()
            rows.append((kind, ln, touch, chead >= 0.3, shead >= 0.3, prev == D))
    print("DFL, {} matches, held out".format(len(files)))
    print("1. oracle gains (each alone), spellF1 mean:")
    base = np.mean(orc["submission"])
    for k, v in orc.items():
        print("   {:<22} {:.4f}  ({:+.4f})".format(k, np.mean(v), np.mean(v) - base))
    R = np.array(rows, float)
    kinds = ("caught", "called D", "other team", "merged", "shifted / split")
    print("\n2. true possessions by length: share in each outcome")
    print("   {:<10} {:>6} ".format("length", "n") + " ".join("{:>15}".format(k) for k in kinds))
    for lo, hi in LENS:
        sel = (R[:, 1] >= lo) & (R[:, 1] < hi)
        print("   {:<10} {:>6} ".format("{:g}-{:g} s".format(lo, hi) if hi < 1e8 else ">= {:g} s".format(lo), int(sel.sum()))
              + " ".join("{:>14.0%} ".format(np.mean(R[sel, 0] == j)) for j in range(5)))
    sel = np.ones(len(R), bool)
    print("   {:<10} {:>6} ".format("all", len(R)) + " ".join("{:>14.0%} ".format(np.mean(R[:, 0] == j)) for j in range(5)))
    miss = R[:, 0] != 0
    M = R[miss]
    anyev = M[:, 2].astype(bool) | M[:, 3].astype(bool) | M[:, 4].astype(bool)
    print("\n3. missed possessions ({}): evidence at the true start (+-0.5 s)".format(int(miss.sum())))
    print("   touch peak of that team {:.0%}, change-net head toward it {:.0%}, short-net start head {:.0%}; "
          "any {:.0%}, none {:.0%}".format(M[:, 2].mean(), M[:, 3].mean(), M[:, 4].mean(), anyev.mean(), 1 - anyev.mean()))
    for j in range(1, 5):
        m = M[:, 0] == j
        if m.any():
            print("   {:<16} {:>5} ({:.0%} of missed), with any evidence {:.0%}, after a dead ball {:.0%}".format(
                kinds[j], int(m.sum()), m.mean(), anyev[m].mean(), M[m, 5].mean()))


if __name__ == "__main__":
    main()
