"""Why the change net still misses possessions on DFL (7 matches, held out). Aggregate only.

    python tools/cn_dfl_miss.py [--tag c1dl] [--w 1.0] [--bias -3.0]

Decodes the held-out change-net prediction alone (change_net.viterbi), then prints:
  1. true possessions by length: share, caught (mean over IoU 0.3-0.7), oracle gain of that class;
  2. why a true possession is missed (IoU < 0.5), by length: mostly D, mostly the other team,
     merged into a longer spell of its own team, split, shifted / partial;
  3. by how it starts (after a turnover / after a stoppage): share, caught, why missed;
  4. true turnovers (team -> other team): recall within 0.2 / 0.5 / 1 / 2 s, offset of the nearest
     predicted turnover (median, share late), and what the prediction does at the missed ones
     (no change at all: same team carries on / via D / other change type nearby);
     split by ball seen at the turnover (+-0.5 s) and by the length of the new possession.
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
import short_net as S  # noqa: E402
import dfl_data as dd  # noqa: E402
import miss_profile as M  # noqa: E402
from pipeline.decode import H, A, D  # noqa: E402

FPS = 25
WHY = ["caught", "mostly D", "other team", "merged", "split", "shifted"]
TOLS = (5, 12, 25, 50)


def why_rows(lab, filled, truth, pred, flags):
    rows = []
    for i, (a, b, k) in enumerate(truth):
        L = b - a
        seg = filled[a:b]
        same, dead = np.mean(seg == k), np.mean(seg == D)
        other = np.mean((seg != k) & (seg != D))
        v, j = M.best_iou((a, b, k), pred)
        n_same = sum(1 for (c2, e2, m2) in pred if m2 == k and e2 > a and c2 < b
                     and (min(b, e2) - max(a, c2)) >= 0.2 * L)
        if v >= 0.5:
            why = 0
        elif dead >= 0.5:
            why = 1
        elif other >= 0.5:
            why = 2
        elif n_same >= 2:
            why = 4
        elif j >= 0 and (pred[j][1] - pred[j][0]) >= 2 * L and same >= 0.5:
            why = 3
        else:
            why = 5
        prev = lab[a - 1] if a > 0 else -1
        start = 0 if prev in (H, A) else (1 if prev == D else 2)      # turnover / restart / other
        rows.append([M.bin_of(L), flags[:, i].mean(), why, start])
    return rows


def turnover_rows(lab, filled, runs, seen, truth, X=None, col=None, pr=None):
    """per true turnover: offset to nearest predicted turnover, what pred does, seen, new length."""
    tc = [f for f, k in M.changes(lab, runs) if k == 0]
    pc = np.array([f for f, k in M.changes(filled, runs) if k == 0])
    allc = np.array([f for f, k in M.changes(filled, runs)])
    start_len = {a: b - a for a, b, k in truth}
    out = []
    for t in tc:
        off = (pc - t)[np.argmin(np.abs(pc - t))] if len(pc) else 9999
        lo, hi = max(t - FPS, 0), min(t + FPS, len(lab))
        new, old = lab[t], lab[t - 1]
        win = filled[lo:hi]
        if np.any(win == D):
            what = 1                                    # via / into D
        elif np.all(filled[t:hi] == old) or np.all(win == old):
            what = 0                                    # old team carries on
        elif np.all(win == new):
            what = 2                                    # new team already before
        else:
            what = 3
        near_other = len(allc) and np.min(np.abs(allc - t)) <= FPS
        sn = seen[max(t - 12, 0):t + 13].mean()
        row = [off, what, sn, start_len.get(t, 0) / FPS, near_other]
        if X is not None:
            o, nw = ("h_", "a_") if old == H else ("a_", "h_")
            w5 = slice(max(t - 5, 0), t + 6)
            w12 = slice(max(t - 12, 0), t + 13)
            inc = X[max(t - 12, 0):t + 1, col["bspd"]]
            row += [inc.max() if sn > 0.5 else np.nan,
                    X[w5, col[nw + "d1"]].min(), X[w5, col[o + "d1"]].min(),
                    X[w12, col[nw + "touch"]].max(), X[w12, col["bz"]].max(),
                    pr[w12, 0 if lab[t] == H else 1].max()]
        out.append(row)
    return out


def contact_type(r):
    """0 gap, 1 duel (both teams within 2.5 m), 2 pass cut (ball in fast, old team > 4 m),
    3 loose (new team > 3 m from the ball), 4 other."""
    inc, dn, do = r[5], r[6], r[7]
    if r[2] < 0.5:
        return 0
    if dn <= 2.5 and do <= 2.5:
        return 1
    if inc >= 8 and do > 4:
        return 2
    if dn > 3:
        return 3
    return 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="c1dl")
    ap.add_argument("--w", type=float, default=1.0)
    ap.add_argument("--bias", type=float, default=-3.0)
    ap.add_argument("--set", default="dfl", choices=("dfl", "skillcorner", "pff"))
    args = ap.parse_args()
    import eval_touch_run as tr
    import cn_dfl
    names = S.channel_names()
    ci = names.index("seen")
    col = {k: names.index(k) for k in names}
    R, TR, sc, orc = [], [], [], []
    if args.set == "dfl":
        games = list(dd.GAMES)
    else:
        import turnover_cls as T
        games = [g for n, g, _ in T.ext_list() if n == args.set]
        with np.load(os.path.join(C.OUT, "pred_{}.npz".format(args.tag))) as z:
            preds = {k: z[k] for k in z.files if k.startswith(args.set + "_")}
    for g in games:
        if args.set == "dfl":
            with np.load(cn_dfl.path(g)) as z:
                lab, blk, X = z["lab"], [tuple(b) for b in z["blk"]], z["x"].astype(np.float32)
            pr = cn_dfl.heldout(args.tag, g)
        else:
            import run_v31_external as rx
            c, _ = T.load_cache(args.set, g)
            lab = c["labels"]
            blk = rx.blocks(c["period"])
            X = np.load(os.path.join(S.OUT, "in_{}_{}.npy".format(args.set, g))).astype(np.float32)
            pr = preds["{}_{}".format(args.set, g)].astype(np.float32)
        seen = X[:, ci]
        held = pr is not None
        out = C.viterbi(pr, blk, args.w, args.bias)
        runs = pm.valid_runs(lab)
        filled = pm.fill_abstentions(out, runs)
        truth = pm.possession_spells(lab, runs)
        pred = pm.possession_spells(filled, runs)
        ov = pm.spell_overlaps(truth, pred)
        flags = np.stack([pm.matched_gt_flags(ov, len(truth), t) for t in M.IOUS])
        s = tr.score(lab, out)
        sc.append(s)
        o_ = []
        for bi in range(len(M.BINS)):
            o = out.copy()
            for a, b, k in truth:
                if M.bin_of(b - a) == bi:
                    o[a:b] = k
            o_.append(tr.score(lab, o) - s)
        for st in (0, 1):
            o = out.copy()
            for a, b, k in truth:
                prev = lab[a - 1] if a > 0 else -1
                if (st == 0 and prev in (H, A)) or (st == 1 and prev == D):
                    o[a:b] = k
            o_.append(tr.score(lab, o) - s)
        orc.append(o_)
        R += why_rows(lab, filled, truth, pred, flags)
        TR += turnover_rows(lab, filled, runs, seen, truth, X, col, pr)
    R, TR, orc = np.array(R, float), np.array(TR, float), np.array(orc).mean(axis=0)
    print("{} on {} ({}held out), w {:g} bias {:g}: spellF1 {:.4f}".format(
        args.tag, args.set, "" if held else "NOT ", args.w, args.bias, np.mean(sc)))

    print("\n1. true possessions by length (caught = mean over IoU 0.3-0.7)")
    print("{:<9} {:>6} {:>7} {:>8} {:>8}".format("length", "n", "share", "caught", "oracle"))
    for bi, bn in enumerate(M.BIN_NAMES):
        s_ = R[:, 0] == bi
        print("{:<9} {:>6} {:>6.1%} {:>7.1%} {:>+8.4f}".format(bn, int(s_.sum()), s_.mean(),
                                                            R[s_, 1].mean(), orc[bi]))
    print("oracle by start: after a turnover {:+.4f}, after a stoppage {:+.4f}".format(orc[5], orc[6]))

    print("\n2. why missed (IoU < 0.5), share of the class")
    print("{:<9} ".format("length") + " ".join("{:>11}".format(w) for w in WHY))
    for bi, bn in enumerate(M.BIN_NAMES + ["all"]):
        s_ = (R[:, 0] == bi) if bi < len(M.BINS) else np.ones(len(R), bool)
        print("{:<9} ".format(bn) + " ".join("{:>10.1%} ".format(np.mean(R[s_, 2] == k))
                                             for k in range(len(WHY))))

    print("\n3. by start, share / caught / why missed")
    for st, nm in ((0, "after turnover"), (1, "after stoppage")):
        s_ = R[:, 3] == st
        print("  {:<15} {:>5.1%} caught {:>5.1%} | ".format(nm, s_.mean(), R[s_, 1].mean())
              + ", ".join("{} {:.0%}".format(WHY[k], np.mean(R[s_, 2] == k)) for k in range(1, 6)))
        for bi, bn in enumerate(M.BIN_NAMES):
            s2 = s_ & (R[:, 0] == bi)
            if s2.sum():
                print("      {:<9} n {:>5} caught {:>5.1%} | ".format(bn, int(s2.sum()), R[s2, 1].mean())
                      + ", ".join("{} {:.0%}".format(WHY[k], np.mean(R[s2, 2] == k)) for k in range(1, 6)))

    print("\n4. true turnovers: n {}".format(len(TR)))
    off = TR[:, 0]
    print("  recall within 0.2 / 0.5 / 1 / 2 s: " + " / ".join(
        "{:.0%}".format(np.mean(np.abs(off) <= w)) for w in TOLS))
    hit = np.abs(off) <= 25
    print("  found within 1 s: median offset {:+.2f} s, late {:.0%}, early {:.0%}".format(
        np.median(off[hit]) / FPS, np.mean(off[hit] > 2), np.mean(off[hit] < -2)))
    print("  found within 1-2 s: {:.0%} (late {:.0%})".format(
        np.mean((np.abs(off) > 25) & (np.abs(off) <= 50)),
        np.mean((off > 25) & (off <= 50)) / max(np.mean((np.abs(off) > 25) & (np.abs(off) <= 50)), 1e-9)))
    ms = ~(np.abs(off) <= 25)
    names = ["old team carries on", "D nearby", "new team already", "mixed"]
    print("  missed (> 1 s), +-1 s around it: " + ", ".join(
        "{} {:.0%}".format(names[k], np.mean(TR[ms, 1] == k)) for k in range(4))
          + "; another change type within 1 s {:.0%}".format(np.mean(TR[ms, 4] > 0)))
    edges = [-1e9, -250, -125, -50, -25, 25, 50, 125, 250, 1e9]
    lbl = ["< -10", "-10..-5", "-5..-2", "-2..-1", "|1|", "1..2", "2..5", "5..10", "> 10"]
    hst = np.histogram(off, bins=edges)[0] / len(off)
    print("  nearest predicted turnover, offset in s (pred - true): " + ", ".join(
        "{} {:.0%}".format(l_, h_) for l_, h_ in zip(lbl, hst)))
    print("  by ball seen at the turnover (+-0.5 s):")
    for lo, hi, nm in ((0, 0.2, "mostly unseen"), (0.2, 0.8, "partly"), (0.8, 1.01, "seen")):
        s_ = (TR[:, 2] >= lo) & (TR[:, 2] < hi)
        if s_.sum():
            print("    {:<14} share {:>5.1%} recall 1 s {:>5.1%}".format(nm, s_.mean(),
                                                                       np.mean(np.abs(off[s_]) <= 25)))
    print("  by length of the new possession:")
    for lo, hi, nm in ((0, 2, "< 2 s"), (2, 5, "2-5 s"), (5, 10, "5-10 s"), (10, 1e9, ">= 10 s")):
        s_ = (TR[:, 3] >= lo) & (TR[:, 3] < hi)
        if s_.sum():
            print("    {:<14} share {:>5.1%} recall 1 s {:>5.1%}".format(nm, s_.mean(),
                                                                       np.mean(np.abs(off[s_]) <= 25)))

    print()
    print("5. true turnovers by contact type (at the turnover +-0.2 / 0.5 s)")
    ct = np.array([contact_type(r) for r in TR])
    tn = ["ball unseen", "duel (both <= 2.5 m)", "pass cut (fast in, old > 4 m)", "loose (new > 3 m)", "other"]
    print("  {:<31} {:>6} {:>8} | medians, found / missed: new d1, old d1, new touch P, net P".format(
        "type", "share", "rec 1 s"))
    for k, nm in enumerate(tn):
        s_ = ct == k
        if not s_.sum():
            continue
        f_, m_ = s_ & ~ms, s_ & ms
        med = lambda c, sel: np.nanmedian(TR[sel, c]) if sel.sum() else np.nan
        print("  {:<31} {:>5.1%} {:>7.1%} | {:.1f} / {:.1f} m, {:.1f} / {:.1f} m, {:.2f} / {:.2f}, {:.2f} / {:.2f}".format(
            nm, s_.mean(), np.mean(~ms[s_]), med(6, f_), med(6, m_), med(7, f_), med(7, m_),
            med(8, f_), med(8, m_), med(10, f_), med(10, m_)))
    print("  share of all missed turnovers: " + ", ".join(
        "{} {:.0%}".format(nm.split(" (")[0], np.mean(ct[ms] == k)) for k, nm in enumerate(tn)))
    print("  missed with net P peak < 0.1 {:.0%}, 0.1-0.3 {:.0%}, >= 0.3 {:.0%}".format(
        np.mean(TR[ms, 10] < 0.1), np.mean((TR[ms, 10] >= 0.1) & (TR[ms, 10] < 0.3)), np.mean(TR[ms, 10] >= 0.3)))


if __name__ == "__main__":
    main()
