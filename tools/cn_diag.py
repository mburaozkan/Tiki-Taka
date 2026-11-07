"""Why the change network, decoded on its own, loses the public window. Aggregate only.

    $env:MTAH_VARIANT="_kt"; python tools/cn_diag.py [--tag c1] [--w 1] [--bias -3]

Needs data/cache/change_net/ours_<tag>.npz (written by change_net.py ours). Prints:
  A. public window: dead / live agreement and team accuracy on live frames (base v31_d1, the
     class head alone, the decoded net); per true change, the correct change head's peak within
     +/- 0.5 s and +/- 2 s, the class head's P(new team) over the 1 s after, and the offset of the
     nearest predicted change of that type (base, net);
  B. the network's behaviour on our full match vs the 84 external matches (held-out): change-head
     peaks >= 0.3 per live minute by type, true changes per minute (external), the class head's
     confidence, P(D) >= 0.5 share;
  C. input shift: per channel, (our mean - external mean) / external sd, and sd ratio.
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
import miss_profile as M  # noqa: E402
from pipeline.decode import H, A, D  # noqa: E402

FPS = 25
TYPES = ["turnover", "stoppage", "restart"]


def peaks(q, thr=0.3):
    """number of local maxima of q >= thr (at least 0.5 s apart)."""
    from scipy.signal import find_peaks
    return len(find_peaks(q, height=thr, distance=12)[0])


def head_for(ty, team):
    """change-head column for a change of type ty into team (0 H, 1 A)."""
    return {0: team, 1: 2, 2: 3 + team}[ty]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="c1")
    ap.add_argument("--w", type=float, default=1.0)
    ap.add_argument("--bias", type=float, default=-3.0)
    args = ap.parse_args()
    from report import PUBLIC_GT
    z = np.load(os.path.join(C.OUT, "ours_{}.npz".format(args.tag)))
    pr, x, base, blk = z["pr"].astype(np.float32), z["x"].astype(np.float32), z["base"], [tuple(b) for b in z["blk"]]
    net = C.viterbi(pr, blk, args.w, args.bias)
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    runs = pm.valid_runs(gt)
    g = np.zeros(len(gt), bool)
    for a, b in runs:
        g[a:b] = True
    cls = np.array([H, A, D], np.int8)[np.argmax(pr[:, C.N_TGT:], axis=1)]
    print("A. public window ({} graded frames, {} true possessions)".format(int(g.sum()),
                                                                       len(pm.possession_spells(gt, runs))))
    live = g & (gt != D)
    for nm, cd in (("base v31_d1", base), ("class head alone", cls), ("decoded net", net)):
        cdf = pm.fill_abstentions(cd, runs)
        print("  {:<18} dead/live agree {:.1%} (true D called D {:.1%}, true live called live {:.1%}) | "
              "team right on live frames {:.1%}".format(
                  nm, np.mean((cdf[g] == D) == (gt[g] == D)), np.mean(cdf[g & (gt == D)] == D),
                  np.mean(cdf[live] != D), np.mean(cdf[live] == gt[live])))
    p_h = pr[:, C.N_TGT] / np.maximum(pr[:, C.N_TGT] + pr[:, C.N_TGT + 1], 1e-6)
    print("  class head P(H | live) on true H frames: median {:.2f}; on true A frames: median {:.2f}".format(
        float(np.median(p_h[g & (gt == H)])), float(np.median(p_h[g & (gt == A)]))))
    print("  per true change: type, into, head peak +-0.5 s / +-2 s, P(new team | live) 1 s after, "
          "nearest same-type change offset base / net (s)")
    bc = M.changes(pm.fill_abstentions(base, runs), runs)
    nc = M.changes(pm.fill_abstentions(net, runs), runs)
    for i, (t, ty) in enumerate(M.changes(gt, runs)):
        new = gt[t]
        team = 0 if new == H else (1 if new == A else (0 if gt[t - 1] == H else 1))
        col = head_for(ty, team)
        p5 = pr[max(t - 12, 0):t + 13, col].max()
        p2 = pr[max(t - 50, 0):t + 51, col].max()
        if ty == 1:
            pn = float(np.mean(pr[t:t + FPS, C.N_TGT + 2]))
            pn_s = "P(D) {:.2f}".format(pn)
        else:
            q = p_h[t:t + FPS] if new == H else 1 - p_h[t:t + FPS]
            pn_s = "P(new) {:.2f}".format(float(np.mean(q)))

        def off(lst):
            o = [f - t for f, k in lst if k == ty]
            if not o:
                return "  -  "
            v = o[int(np.argmin(np.abs(o)))] / FPS
            return "{:+5.1f}".format(v) if abs(v) <= 10 else " >10 "
        print("    change {:>2} {:<9} into {} | peak {:.2f} / {:.2f} | {} | base {} net {}".format(
            i + 1, TYPES[ty], "HAD"[[H, A, D].index(new)], p5, p2, pn_s, off(bc), off(nc)))

    print("\nB. behaviour: our full match vs the 84 external matches (held-out)")
    import turnover_cls as T
    ext = np.load(os.path.join(C.OUT, "pred_{}.npz".format(args.tag)))
    rows = []
    for n_, g_, _ in T.ext_list():
        q = ext["{}_{}".format(n_, g_)].astype(np.float32)
        c_ = np.load(os.path.join(S.OUT, "c_{}_{}.npy".format(n_, g_)))
        lv = (c_ == 0) | (c_ == 1)
        mins = lv.sum() / FPS / 60
        y = C.targets_from_cls(c_)
        true_rate = [(y[:, k] == 1).sum() / mins for k in range(C.N_TGT)]
        pk = [peaks(np.where(c_ >= 0, q[:, k], 0)) / mins for k in range(C.N_TGT)]
        conf = q[c_ >= 0, C.N_TGT:].max(axis=1).mean()
        pd_ = np.mean(q[c_ >= 0, C.N_TGT + 2] >= 0.5)
        rows.append(true_rate + pk + [conf, pd_])
    R = np.array(rows)
    inb = np.zeros(len(pr), bool)
    for a, b in blk:
        inb[a:b] = True
    live_ours = inb & (base != D)
    mins = live_ours.sum() / FPS / 60
    pk_o = [peaks(np.where(inb, pr[:, k], 0)) / mins for k in range(C.N_TGT)]
    names = ["turn->H", "turn->A", "stop", "restart H", "restart A"]
    print("  per live minute    " + " ".join("{:>10}".format(n_) for n_ in names))
    print("  external true      " + " ".join("{:>10.2f}".format(v) for v in R[:, :5].mean(0)))
    print("  external peaks     " + " ".join("{:>10.2f}".format(v) for v in R[:, 5:10].mean(0)))
    print("  ours peaks         " + " ".join("{:>10.2f}".format(v) for v in pk_o))
    print("  (ours: live minutes from the base's live frames)")
    print("  class head confidence (mean max P): external {:.2f}, ours {:.2f}".format(
        R[:, 10].mean(), pr[inb, C.N_TGT:].max(axis=1).mean()))
    print("  P(D) >= 0.5 share of frames: external {:.1%} (true D on external {:.1%}), ours {:.1%} "
          "(base D {:.1%})".format(R[:, 11].mean(), np.mean([np.mean(np.load(os.path.join(
              S.OUT, "c_{}_{}.npy".format(n_, g_)))[np.load(os.path.join(S.OUT, "c_{}_{}.npy".format(
                  n_, g_))) >= 0] == 2) for n_, g_, _ in T.ext_list()]),
              np.mean(pr[inb, C.N_TGT + 2] >= 0.5), np.mean(base[inb] == D)))
    nf = pm.fill_abstentions(net, [(a, b) for a, b in blk])
    bf = pm.fill_abstentions(base, [(a, b) for a, b in blk])
    print("  frames that differ (net vs base, inside the halves): {:.1%}: base live -> net D {:.1%}, "
          "base D -> net live {:.1%}, team flipped {:.1%}".format(
              np.mean(nf[inb] != bf[inb]), np.mean((bf[inb] != D) & (nf[inb] == D)),
              np.mean((bf[inb] == D) & (nf[inb] != D)),
              np.mean((bf[inb] != D) & (nf[inb] != D) & (bf[inb] != nf[inb]))))

    print("\nC. input shift, ours vs external: (mean difference) / external sd, sd ratio; largest 12")
    xs = np.concatenate([np.load(os.path.join(S.OUT, "in_{}_{}.npy".format(n_, g_)))[::25].astype(np.float32)
                         for n_, g_, _ in T.ext_list()])
    xo = x[inb][::5]
    mu, sd = xs.mean(0), xs.std(0) + 1e-6
    smd = (xo.mean(0) - mu) / sd
    ratio = (xo.std(0) + 1e-6) / sd
    ch = S.channel_names()
    score = np.abs(smd) + np.abs(np.log(ratio))
    for j in np.argsort(-score)[:12]:
        print("  {:<10} shift {:+.2f} sd   sd ratio {:.2f}".format(ch[j], smd[j], ratio[j]))


if __name__ == "__main__":
    main()
