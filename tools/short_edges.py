"""Boundary misses of short possessions on the current best (v31_d1 + e3k). Aggregate only.

    python tools/short_edges.py diag [--jobs 4]
    python tools/short_edges.py snap [--jobs 4]     # move change points to the short net's edge heads

snap: every H <-> A change point of e3k (frame t, team X before, Y after) is moved to the frame t'
within +-W of t that maximises start-head(Y)[t'] + end-head(X)[t' - 1], if that sum >= P; it never
crosses the neighbouring change points (each spell keeps >= 5 frames). Grid W x P, spellF1
SkillCorner / PFF vs e3k (up / down), short caught, nested 4-fold choice.

e3k on the 84 external matches, as short_check eval (net e3, spells at 0.6, keep >= 0.3 with the
held-out check k1, edges refined, written over D). For every true H / A possession shorter than
5 s that e3k has with the right team but no IoU >= 0.5 (the 'boundary' bucket of
tools/short_miss.py), with the best-overlapping predicted spell of that team:
  merged   the predicted spell covers the true one (within 3 frames at both ends) and is more than
           twice as long: a missing neighbour of the other team, not an edge problem
  shifted  otherwise: partial overlap, edges off
Prints the split, start / end offsets (predicted - true, s) and the share within 0.5 / 1 s, and
for merged ones whether the other team's true neighbour is missed by e3k, and where the short
net's start / end heads peak relative to the true edges (could snapping fix them?).
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
os.environ.setdefault("TC_ROWS", "v2")
import possession_metrics as pm  # noqa: E402
from pipeline.decode import H, A, D  # noqa: E402

FPS = 25
NET, GEN, KEEP = "e3", 0.6, 0.3


def e3k(name, g):
    """(labels, base v31_d1, e3k codes, short-net predictions) for one external match."""
    import turnover_cls as T
    import short_net as S
    import short_check as K
    import run_v31_external as rx
    import pair_cls as P
    c, f = T.load_cache(name, g)
    pdead = P.pdead_of(name, g)
    blk = rx.blocks(c["period"])
    base = T.base_chain(c, f, blk, pdead)
    key = "{}_{}".format(name, g)
    with np.load(os.path.join(S.OUT, "pred_{}.npz".format(NET))) as z:
        pr = z[key].astype(np.float32)
    with np.load(os.path.join(K.OUT, "{}_g{:g}_{}_{}.npz".format(NET, GEN, name, g))) as z:
        meta = z["meta"]
    with np.load(os.path.join(K.OUT, "pred_k1_{}_g{:g}.npz".format(NET, GEN))) as z:
        pk = z[key] if key in z.files else np.zeros(0)
    out = base.copy()
    for i in np.argsort(pk):
        if pk[i] < KEEP:
            continue
        a, b, team = int(meta[i, 0]), int(meta[i, 1]), int(meta[i, 2])
        a, b = K.refine_edges(pr, a, b, 0 if team == H else 1, len(out))
        seg = out[a:b]
        out[a:b] = np.where(np.isin(seg, (H, A, D)), team, seg)
    return c["labels"], base, out, pr


def iou(a, b, c, d):
    return max(0, min(b, d) - max(a, c)) / (max(b, d) - min(a, c))


def diag_match(job):
    name, g = job
    lab, base, codes, pr = e3k(name, g)
    runs = pm.valid_runs(lab)
    truth = pm.possession_spells(lab, runs)
    pred = pm.possession_spells(codes, runs)
    hit = [max([iou(a, b, c, d) for c, d, k2 in pred if k2 == k and d > a and c < b] + [0]) >= 0.5
           for a, b, k in truth]
    rows = []
    for i, (a, b, k) in enumerate(truth):
        if (b - a) / FPS >= 5 or hit[i]:
            continue
        ov = [(iou(a, b, c, d), c, d) for c, d, k2 in pred if k2 == k and d > a and c < b]
        if not ov:
            continue
        _, c, d = max(ov)
        merged = c <= a + 3 and d >= b - 3 and (d - c) > 2 * (b - a)
        s = 0 if k == H else 1
        nb = [j for j in (i - 1, i + 1) if 0 <= j < len(truth) and truth[j][2] not in (k,)
              and (truth[j][1] == a or truth[j][0] == b)]
        nb_missed = int(any(not hit[j] for j in nb))
        w = 12
        sh = pr[max(a - w, 0):a + w, 2 + s]
        eh = pr[max(b - 1 - w, 0):b - 1 + w, 4 + s]
        s_off = (np.argmax(sh) + max(a - w, 0) - a) / FPS if len(sh) and sh.max() >= 0.2 else np.nan
        e_off = (np.argmax(eh) + max(b - 1 - w, 0) - (b - 1)) / FPS if len(eh) and eh.max() >= 0.2 else np.nan
        rows.append((int(merged), (c - a) / FPS, (d - b) / FPS, (b - a) / FPS, nb_missed, s_off, e_off))
    return np.array(rows, float).reshape(-1, 7), sum(1 for a, b, k in truth if (b - a) / FPS < 5), sum(
        1 for i, (a, b, k) in enumerate(truth) if (b - a) / FPS < 5 and hit[i])


def diag_cmd(args):
    import turnover_cls as T
    with Pool(args.jobs) as pool:
        res = pool.map(diag_match, [(n, g) for n, g, _ in T.ext_list()], chunksize=1)
    R = np.concatenate([r[0] for r in res])
    nshort, ncaught = sum(r[1] for r in res), sum(r[2] for r in res)
    print("e3k on 84 matches: {} true short possessions, caught {} ({:.0%}); boundary misses {} ({:.0%})".format(
        nshort, ncaught, ncaught / nshort, len(R), len(R) / nshort))
    for m, nm in ((1, "merged"), (0, "shifted")):
        sel = R[:, 0] == m
        X = R[sel]
        print("\n{}: {} ({:.0%} of boundary misses), true length median {:.1f} s".format(
            nm, len(X), sel.mean(), np.median(X[:, 3])))
        print("  start offset median {:+.2f} s (|.| <= 0.5 s {:.0%}, <= 1 s {:.0%}); end offset median {:+.2f} s "
              "(|.| <= 0.5 s {:.0%}, <= 1 s {:.0%})".format(
                  np.median(X[:, 1]), np.mean(np.abs(X[:, 1]) <= 0.5), np.mean(np.abs(X[:, 1]) <= 1),
                  np.median(X[:, 2]), np.mean(np.abs(X[:, 2]) <= 0.5), np.mean(np.abs(X[:, 2]) <= 1)))
        print("  other-team neighbour also missed: {:.0%}".format(X[:, 4].mean()))
        print("  short-net start head (>= 0.2) near the true start: {:.0%}, within 0.2 s of it {:.0%}; end head near "
              "the true end: {:.0%}, within 0.2 s {:.0%}".format(
                  np.mean(np.isfinite(X[:, 5])), np.mean(np.abs(X[:, 5]) <= 0.2),
                  np.mean(np.isfinite(X[:, 6])), np.mean(np.abs(X[:, 6]) <= 0.2)))


W_GRID = (2.0, 3.0, 4.0, 6.0)
P_GRID = (0.1, 0.2, 0.3)


def snap(codes, pr, w_s, p_min, live=(H, A)):
    out = codes.copy()
    n = len(out)
    t_all = np.flatnonzero(out[1:] != out[:-1]) + 1
    w = int(round(w_s * FPS))
    for i, t in enumerate(t_all):
        x, y = out[t - 1], out[t]
        if x not in live or y not in live or x == y:
            continue
        lo = (t_all[i - 1] if i > 0 else 0) + 5
        hi = (t_all[i + 1] if i + 1 < len(t_all) else n) - 5
        a, b = max(t - w, lo, 1), min(t + w, hi)
        if b <= a:
            continue
        sy, ex = 2 + (0 if y == H else 1), 4 + (0 if x == H else 1)
        sc = pr[a:b + 1, sy] + pr[a - 1:b, ex]
        j = int(np.argmax(sc))
        if sc[j] < p_min:
            continue
        t2 = a + j
        if t2 < t:
            out[t2:t] = y
        elif t2 > t:
            out[t:t2] = x
    return out


def snap_match(job):
    name, g = job
    from decoder_sweep import stats
    lab, base, codes, pr = e3k(name, g)
    res = [stats(lab, codes)]
    for w in W_GRID:
        for p in P_GRID:
            res.append(stats(lab, snap(codes, pr, w, p)))
    return name, g, np.array(res, float)


def snap_cmd(args):
    import turnover_cls as T
    import run_v31_external as rx
    with Pool(args.jobs) as pool:
        res = pool.map(snap_match, [(n, g) for n, g, _ in T.ext_list()], chunksize=1)
    names = np.array([r[0] for r in res])
    folds = np.array([rx.fold_of(r[0], r[1]) for r in res])
    arr = np.stack([r[2] for r in res])
    labels = ["e3k"] + ["snap W {:g} s, P {:g}".format(w, p) for w in W_GRID for p in P_GRID]
    print("{:<20} {:>17} {:>17} | {:>10} {:>9} | {:>9} {:>9}".format(
        "setting", "SkillCorner", "PFF", "caught <2s", "2-5s", "pred/true", "precision"))
    for j, lb in enumerate(labels):
        parts = []
        for nm in ("skillcorner", "pff"):
            v = arr[names == nm, j, 0]
            dl = v - arr[names == nm, 0, 0]
            parts.append("{:.4f} {:>3}/{:<3}".format(v.mean(), int(np.sum(dl > 1e-9)), int(np.sum(dl < -1e-9))))
        t = arr[:, j, 1:].sum(axis=0)
        print("{:<20} {:>17} {:>17} | {:>9.1%} {:>9.1%} | {:>9.2f} {:>9.1%}".format(
            lb, parts[0], parts[1], t[1] / t[0], t[3] / t[2], t[5] / t[4], t[6] / t[5]))
    gain = arr[:, 1:, 0] - arr[:, :1, 0]
    nested = np.zeros(len(arr))
    for k in range(T.FOLDS):
        j = int(np.argmax(gain[folds != k].mean(axis=0)))
        nested[folds == k] = gain[folds == k, j]
    print("nested 4-fold vs e3k: SkillCorner {:+.4f}, PFF {:+.4f} ({}/{})".format(
        nested[names == "skillcorner"].mean(), nested[names == "pff"].mean(),
        int(np.sum(nested > 1e-9)), int(np.sum(nested < -1e-9))))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("diag", "snap"))
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    {"diag": diag_cmd, "snap": snap_cmd}[args.cmd](args)


if __name__ == "__main__":
    main()
