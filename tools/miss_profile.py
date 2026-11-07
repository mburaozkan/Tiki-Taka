"""Where spellF1 is still lost with the best candidate (v31_d1 + e3 short net + check). Aggregate only.

    python tools/miss_profile.py [--net e3] [--gen 0.6] [--tag k1] [--thr 0.3]

On the 84 external matches (SkillCorner + PFF), the candidate is rebuilt as tools/short_check.py eval
(keep >= thr, edges, also over D). Prints:
  1. by true possession length (< 2, 2-5, 5-10, 10-20, >= 20 s): share of possessions, caught rate
     (mean over the 5 IoU thresholds), share of all misses, and the oracle gain of writing just
     that length class in from the truth;
  2. false predicted spells by length;
  3. why a true possession is missed (IoU < 0.5): mostly D, mostly the other team, merged into a
     longer spell of its own team, split, shifted; by length class;
  4. context of the short ones (< 5 s): what comes before / after in the truth (other team / D),
     caught rate per context;
  5. caught vs missed short possessions: median of simple aggregates (ball seen, ball height,
     incoming ball speed, the team's touch-net peak, nearest distance, the net's P peak).
"""

import argparse
import os
import sys
from multiprocessing import Pool

os.environ["TC_ROWS"] = "v2"
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
import turnover_cls as T  # noqa: E402
import short_net as S  # noqa: E402
import short_check as K  # noqa: E402
from pipeline.decode import H, A, D  # noqa: E402

FPS = 25
BINS = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 1e9)]
BIN_NAMES = ["< 2 s", "2-5 s", "5-10 s", "10-20 s", ">= 20 s"]
IOUS = pm.SPELL_IOUS


def bin_of(n):
    s = n / FPS
    for i, (lo, hi) in enumerate(BINS):
        if lo <= s < hi:
            return i
    return len(BINS) - 1


def candidate(job):
    name, g, net, gen, tag, thr = job
    import run_v31_external as rx
    import pair_cls as P
    key = "{}_{}".format(name, g)
    with np.load(os.path.join(S.OUT, "pred_{}.npz".format(net))) as z:
        pr = z[key].astype(np.float32)
    with np.load(os.path.join(K.OUT, "{}_g{:g}_{}_{}.npz".format(net, gen, name, g))) as z:
        meta = z["meta"]
    with np.load(os.path.join(K.OUT, "pred_{}_{}_g{:g}.npz".format(tag, net, gen))) as z:
        p = z[key] if key in z.files else np.zeros(0)
    c, f = T.load_cache(name, g)
    pdead = P.pdead_of(name, g)
    blk = rx.blocks(c["period"])
    lab = c["labels"]
    n = len(lab)
    base = T.base_chain(c, f, blk, pdead)
    out = base.copy()
    for i in np.argsort(p):
        if p[i] < thr:
            continue
        a, b, team = int(meta[i, 0]), int(meta[i, 1]), int(meta[i, 2])
        a, b = K.refine_edges(pr, a, b, 0 if team == H else 1, n)
        out[a:b] = team
    return c, f, lab, base, out, pr


def best_iou(t, spells):
    a, b, k = t
    best, j_best = 0.0, -1
    for j, (c, e, m) in enumerate(spells):
        if m != k or e <= a or b <= c:
            continue
        v = (min(b, e) - max(a, c)) / float(max(b, e) - min(a, c))
        if v > best:
            best, j_best = v, j
    return best, j_best


def run(job):
    import eval_touch_run as tr
    name, g = job[0], job[1]
    c, f, lab, base, out, pr = candidate(job)
    runs = pm.valid_runs(lab)
    filled = pm.fill_abstentions(out, runs)
    truth = pm.possession_spells(lab, runs)
    pred = pm.possession_spells(filled, runs)
    ov = pm.spell_overlaps(truth, pred)
    flags = np.stack([pm.matched_gt_flags(ov, len(truth), t) for t in IOUS])     # (5, n_gt)
    # predicted spells matched at each threshold
    pflags = np.zeros((len(IOUS), len(pred)), bool)
    for ti, t in enumerate(IOUS):
        used_g = set()
        for v, i, j in ov:
            if v < t:
                break
            if i in used_g or pflags[ti, j]:
                continue
            used_g.add(i)
            pflags[ti, j] = True
    score = tr.score(lab, out)
    # oracle per length class
    orc = []
    for bi in range(len(BINS)):
        o = out.copy()
        for a, b, k in truth:
            if bin_of(b - a) == bi:
                o[a:b] = k
        orc.append(tr.score(lab, o) - score)
    # per true spell rows
    x = np.load(os.path.join(S.OUT, "in_{}_{}.npy".format(name, g))).astype(np.float32)
    names = S.channel_names()
    col = {k: names.index(k) for k in names}
    rows = []
    for i, (a, b, k) in enumerate(truth):
        L = b - a
        prev = lab[a - 1] if a > 0 else -1
        nxt = lab[b] if b < len(lab) else -1
        seg = filled[a:b]
        same = np.mean(seg == k)
        dead = np.mean(seg == D)
        other = np.mean((seg != k) & (seg != D))
        v, j = best_iou((a, b, k), pred)
        n_same = sum(1 for (c2, e2, m2) in pred if m2 == k and e2 > a and c2 < b
                     and (min(b, e2) - max(a, c2)) >= 0.2 * L)
        if v >= 0.5:
            why = 0                                          # caught
        elif dead >= 0.5:
            why = 1                                          # mostly D
        elif other >= 0.5:
            why = 2                                          # mostly the other team
        elif n_same >= 2:
            why = 4                                          # split
        elif j >= 0 and (pred[j][1] - pred[j][0]) >= 2 * L and same >= 0.5:
            why = 3                                          # merged into a longer own spell
        else:
            why = 5                                          # shifted / partial
        s = 0 if k == H else 1
        tg = "h_" if k == H else "a_"
        lo, hi = max(a - 2 * FPS, 0), min(b + FPS, len(lab))
        st = slice(max(a - 5, 0), min(a + 5, len(lab)))
        rows.append([bin_of(L), L / FPS, flags[:, i].mean(), why,
                     0 if prev == D else (1 if prev in (H, A) else 2),
                     0 if nxt == D else (1 if nxt in (H, A) else 2),
                     x[a:b, col["seen"]].mean(),
                     x[lo:a + 1, col["bz"]].max() if a + 1 > lo else 0.0,
                     x[st, col["bspd"]].max(),
                     x[a:b, col[tg + "touch"]].max(),
                     x[a:b, col[tg + "d1"]].min(),
                     pr[lo:hi, s].max(),
                     np.mean(base[a:b] == k)])
    prow = []
    for j, (a, b, k) in enumerate(pred):
        mid = lab[(a + b) // 2]
        prow.append([bin_of(b - a), (~pflags[:, j]).mean(),
                     0 if mid == k else (1 if mid in (H, A) else 2)])
    bnd = boundaries(lab, filled, runs)
    return (name, score, np.array(orc), np.array(rows, float), np.array(prow, float),
            flags.sum(axis=1), len(truth), len(pred), bnd)


TOLS = (5, 12, 25, 50)          # frames: 0.2, 0.5, 1, 2 s


def changes(codes, runs):
    """[(frame, type)] type 0 team -> other team, 1 team -> D, 2 D -> team."""
    out = []
    for a, b in runs:
        s = codes[a:b]
        e = np.flatnonzero(s[1:] != s[:-1]) + 1
        for i in e:
            p, q = s[i - 1], s[i]
            if p in (H, A) and q in (H, A):
                out.append((a + i, 0))
            elif p in (H, A) and q == D:
                out.append((a + i, 1))
            elif p == D and q in (H, A):
                out.append((a + i, 2))
    return out


def boundaries(lab, filled, runs):
    """per type: (n_true, hits per tol, n_pred, pred hits per tol, offsets of the nearest pred)."""
    tc, pc = changes(lab, runs), changes(filled, runs)
    res = []
    for ty in (0, 1, 2):
        t = np.array([f for f, k in tc if k == ty])
        p = np.array([f for f, k in pc if k == ty])
        if len(t) == 0 or len(p) == 0:
            res.append((len(t), np.zeros(len(TOLS)), len(p), np.zeros(len(TOLS)), np.zeros(0)))
            continue
        ip = np.clip(np.searchsorted(p, t), 1, len(p) - 1) if len(p) > 1 else np.zeros(len(t), int)
        near = np.where(np.abs(p[ip] - t) < np.abs(p[np.maximum(ip - 1, 0)] - t), p[ip],
                        p[np.maximum(ip - 1, 0)])
        off = near - t
        it = np.clip(np.searchsorted(t, p), 1, len(t) - 1) if len(t) > 1 else np.zeros(len(p), int)
        dp = np.minimum(np.abs(t[it] - p), np.abs(t[np.maximum(it - 1, 0)] - p))
        res.append((len(t), np.array([np.sum(np.abs(off) <= w) for w in TOLS]), len(p),
                    np.array([np.sum(dp <= w) for w in TOLS]), off))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="e3")
    ap.add_argument("--gen", type=float, default=0.6)
    ap.add_argument("--tag", default="k1")
    ap.add_argument("--thr", type=float, default=0.3)
    args = ap.parse_args()
    jobs = [(n, g, args.net, args.gen, args.tag, args.thr) for n, g, _ in T.ext_list()]
    with Pool(6) as pool:
        res = pool.map(run, jobs, chunksize=1)
    names = np.array([r[0] for r in res])
    sc = np.array([r[1] for r in res])
    orc = np.stack([r[2] for r in res])
    R = np.concatenate([r[3] for r in res])
    P_ = np.concatenate([r[4] for r in res])
    print("candidate {} gen {:g} check {} >= {:g}: SkillCorner {:.4f}, PFF {:.4f} ({} matches)".format(
        args.net, args.gen, args.tag, args.thr, sc[names == "skillcorner"].mean(),
        sc[names == "pff"].mean(), len(res)))
    n_gt = sum(r[6] for r in res)
    n_pr = sum(r[7] for r in res)
    tp = np.sum([r[5] for r in res], axis=0)
    print("true possessions {}, predicted {} ({:.2f}); TP per IoU {}".format(
        n_gt, n_pr, n_pr / n_gt, " ".join(str(int(t)) for t in tp)))
    miss_tot = np.sum(1 - R[:, 2])
    fp_tot = np.sum(P_[:, 1])
    print("\n1. true possessions by length (caught = mean over IoU 0.3-0.7)")
    print("{:<9} {:>6} {:>7} {:>8} {:>10} | oracle gain SC / PFF".format(
        "length", "count", "share", "caught", "of misses"))
    for bi, bn in enumerate(BIN_NAMES):
        s = R[:, 0] == bi
        print("{:<9} {:>6} {:>6.1%} {:>7.1%} {:>9.1%}  | {:+.4f} / {:+.4f}".format(
            bn, int(s.sum()), s.mean(), R[s, 2].mean(), np.sum(1 - R[s, 2]) / miss_tot,
            orc[names == "skillcorner", bi].mean(), orc[names == "pff", bi].mean()))
    print("misses (gt, IoU) {:.0f} vs false predicted (pred, IoU) {:.0f}".format(miss_tot, fp_tot))
    print("\n2. predicted spells by length: false rate, share of all false; false ones land on")
    for bi, bn in enumerate(BIN_NAMES):
        s = P_[:, 0] == bi
        if not s.any():
            continue
        w = P_[s, 1]
        land = [np.sum(w * (P_[s, 2] == v)) / max(w.sum(), 1e-9) for v in (0, 1, 2)]
        print("{:<9} {:>6} false {:>5.1%} share {:>5.1%} | own team {:.0%}, other team {:.0%}, D {:.0%}"
              .format(bn, int(s.sum()), w.mean(), w.sum() / fp_tot, *land))
    why = ["caught", "mostly D", "mostly other team", "merged into longer own", "split", "shifted"]
    print("\n3. why missed (IoU < 0.5), share of the class")
    print("{:<9} ".format("length") + " ".join("{:>12}".format(w[:12]) for w in why))
    for bi, bn in enumerate(BIN_NAMES):
        s = R[:, 0] == bi
        print("{:<9} ".format(bn) + " ".join("{:>11.1%} ".format(np.mean(R[s, 3] == k))
                                             for k in range(len(why))))
    sh = R[:, 1] < 5
    ctx = ["D", "other", "none"]
    print("\n4. short (< 5 s) by context (before -> after): share, caught, why missed")
    for p_ in (0, 1):
        for n_ in (0, 1):
            s = sh & (R[:, 4] == p_) & (R[:, 5] == n_)
            if not s.any():
                continue
            print("  {:>5} -> T -> {:<5} {:>5.1%} caught {:>5.1%} | ".format(
                ctx[p_], ctx[n_], s.sum() / sh.sum(), R[s, 2].mean()) +
                ", ".join("{} {:.0%}".format(why[k], np.mean(R[s, 3] == k)) for k in range(1, 6)))
    print("\n5. short (< 5 s), caught (IoU >= 0.5) vs missed: medians")
    feats = ["length s", "ball seen", "max height 2 s before", "ball speed at start",
             "team touch-net peak", "team nearest dist min", "net P peak", "base own share"]
    cau = sh & (R[:, 3] == 0)
    mis = sh & (R[:, 3] != 0)
    for i, ft in enumerate(feats):
        k = [1, 6, 7, 8, 9, 10, 11, 12][i]
        print("  {:<24} caught {:>7.2f}  missed {:>7.2f}".format(ft, np.median(R[cau, k]),
                                                                np.median(R[mis, k])))
    for ft, k, cut in (("team touch-net peak >= 0.3", 9, 0.3), ("net P peak >= 0.3", 11, 0.3),
                       ("ball seen >= 0.5", 6, 0.5)):
        print("  {:<28} caught {:>5.1%}  missed {:>5.1%}".format(ft, np.mean(R[cau, k] >= cut),
                                                                np.mean(R[mis, k] >= cut)))
    print("\n6. changes of state: recall (a predicted change of the same type within tol) / "
          "precision; tol 0.2 / 0.5 / 1 / 2 s")
    for ty, tn in enumerate(("turnover T -> other", "stoppage T -> D", "restart D -> T")):
        for nm in ("skillcorner", "pff"):
            rr = [r[8][ty] for r in res if r[0] == nm]
            nt, ht = sum(x[0] for x in rr), np.sum([x[1] for x in rr], axis=0)
            npr, hp = sum(x[2] for x in rr), np.sum([x[3] for x in rr], axis=0)
            off = np.concatenate([x[4] for x in rr]) / FPS
            ok = np.abs(off) <= 2
            print("  {:<20} {:<11} true {:>5} pred {:>5} | recall {} | precision {} | median "
                  "offset {:+.2f} s, |offset| {:.2f} s".format(
                      tn, nm, nt, npr, " ".join("{:.0%}".format(h / max(nt, 1)) for h in ht),
                      " ".join("{:.0%}".format(h / max(npr, 1)) for h in hp),
                      np.median(off[ok]) if ok.any() else 0, np.median(np.abs(off[ok])) if ok.any() else 0))


if __name__ == "__main__":
    main()
