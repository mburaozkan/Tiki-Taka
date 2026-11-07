"""How short possession starts look to the turnover classifier, on SkillCorner + PFF. Aggregate only.

    python tools/short_profile.py

1. Coverage: every true start after a team change, by length class, sorted into why it is (not) a
   candidate of tools/turnover_cls.py: candidate / base already had T / base in D / touch peak of T
   within +/- 0.5 s but < CAND_P / no touch peak of T at all within +/- 0.5 s (best P in brackets).
2. Inputs: per input, AUC of short (< 2 s) starts vs negatives, of long (>= 5 s) starts vs negatives,
   and of short vs long starts; medians per group. Top inputs by each.
3. Held-out P(start) per group (classifier tag t1).
Prints only counts, rates, medians and AUCs.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import turnover_cls as T  # noqa: E402
from pipeline.decode import H, A, D  # noqa: E402

CLASSES = (("<2", 0, 2), ("2-5", 2, 5), (">=5", 5, np.inf))


def cls_of(ln):
    return next(c for c, lo, hi in CLASSES if lo <= ln < hi)


def coverage():
    import run_v31_external as rx
    import train_ext_heads as th
    cnt = {c[0]: {} for c in CLASSES}
    with np.load(os.path.join(ROOT, "data", "cache", "ext_heads_d1.npz")) as z:
        heads = {k: z[k] for k in z.files}
    for name, g, _ in T.ext_list():
        c, f = T.load_cache(name, g)
        pdead = th.seeds_mean(heads, name + "_" + g)
        blk = rx.blocks(c["period"])
        base = T.base_chain(c, f, blk, pdead)
        with np.load(os.path.join(T.OUT, "rows_{}_{}.npz".format(name, g))) as z:
            meta = z["meta"]
        cand = {(int(i), int(t)) for i, t in meta[:, :2]}
        tp = np.stack([f["touch_p_home"], f["touch_p_away"]], axis=1)
        k = int(T.LABEL_S * T.FPS)
        bef = int(T.BEFORE_S * T.FPS)
        n = len(base)
        for a, team, ln, chg in T.true_starts(c["labels"]):
            if not chg:
                continue
            side = 0 if team == H else 1
            lo, hi = max(a - k, 0), min(a + k + 1, n)
            near = [i for i in range(lo, hi) if (i, team) in cand]
            pk = float(tp[lo:hi, side].max())
            if near:
                why = "candidate"
            elif base[max(a - bef, 0)] == team:
                why = "base had T"
            elif base[max(a - bef, 0)] == D:
                why = "base in D"
            elif pk >= T.CAND_P:
                why = "peak but base T/D at peak"
            elif pk >= 0.1:
                why = "touch P 0.1-0.3"
            else:
                why = "touch P < 0.1"
            cnt[cls_of(ln)][why] = cnt[cls_of(ln)].get(why, 0) + 1
    print("1. true starts after a team change (84 matches), where they go:")
    for cl, d in cnt.items():
        tot = sum(d.values())
        print("  {:<4} ({:>5}): ".format(cl, tot) + ", ".join(
            "{} {:.0%}".format(k, v / tot) for k, v in sorted(d.items(), key=lambda kv: -kv[1])))


def auc(pos, neg):
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if len(pos) < 20 or len(neg) < 20:
        return np.nan
    x = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    r = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]          # mid-ranks
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def inputs():
    data = T.load_rows()
    names = data[0][5]
    x = np.concatenate([d[3] for d in data])
    m = np.concatenate([d[4] for d in data])
    with np.load(os.path.join(T.OUT, "pred_t1.npz")) as z:
        p = np.concatenate([z["{}_{}".format(d[0], d[1])] for d in data])
    y, ln = m[:, 2] == 1, m[:, 3]
    groups = {"neg": ~y, "short<2": y & (ln < 2), "mid2-5": y & (ln >= 2) & (ln < 5),
              "long>=5": y & (ln >= 5)}
    print("\n3. held-out P(start): " + ", ".join(
        "{} n {} median {:.2f} (>= 0.2: {:.0%})".format(k, int(g.sum()), np.median(p[g]),
                                                       np.mean(p[g] >= 0.2))
        for k, g in groups.items()))
    rows = []
    for j, nm in enumerate(names):
        col = x[:, j]
        rows.append((nm, auc(col[groups["short<2"]], col[groups["neg"]]),
                     auc(col[groups["long>=5"]], col[groups["neg"]]),
                     auc(col[groups["short<2"]], col[groups["long>=5"]]),
                     [np.nanmedian(col[g]) for g in groups.values()]))

    def show(title, key):
        print("\n2. " + title)
        print("   {:<22} {:>9} {:>9} {:>9} | medians neg / short / mid / long".format(
            "input", "sh v neg", "lg v neg", "sh v lg"))
        for r in sorted(rows, key=key)[:14]:
            print("   {:<22} {:>9.3f} {:>9.3f} {:>9.3f} | {}".format(
                r[0], r[1], r[2], r[3], " / ".join("{:.2f}".format(v) for v in r[4])))
    show("best at short starts vs negatives (|AUC - 0.5|):",
         lambda r: -abs((r[1] if not np.isnan(r[1]) else 0.5) - 0.5))
    show("most different between short and long starts:",
         lambda r: -abs((r[3] if not np.isnan(r[3]) else 0.5) - 0.5))


def main():
    if "--inputs" not in sys.argv:
        coverage()
    inputs()


if __name__ == "__main__":
    main()
