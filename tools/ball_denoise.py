"""Physics denoising of a jittery ball: L1 trend filtering (piecewise straight-line motion, few kinks).

    python tools/ball_denoise.py dfl [--lams 0 1 2 4 8]     # tune on DFL + our real YOLO noise
    python tools/ball_denoise.py ours --lam 2               # -> data/cache/yolo/ball_refined_bambezius_l<lam>.npy

Per seen stretch, x = argmin 0.5 |y - x|^2 + lam * sum_t |x[t+1] - 2 x[t] + x[t-1]|_2 (x, y jointly,
metres), solved by iteratively reweighted least squares. Between contacts the ball moves almost in a
straight line, so jitter is averaged out along a stretch and a kink stays only where the data
clearly bends.

dfl: per DFL match, the ball as in train_touch --yolo-real sharp (sharp ball + the fast part of our
real YOLO corrections, tools/yolo_noise.py; tracking-like elsewhere), filtered at each lam. Prints
the RMS error to the clean sharp ball and the velocity-change peaks (+-3 frames, local max) against
the teacher touches within 0.2 s: best F1 over peak thresholds, with its P / R and peaks per seen
minute. Also the clean sharp ball and the noisy ball unfiltered. Aggregates only, DFL only.
"""
import argparse
import os
import sys

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import splu

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]

FPS = 25
EPS = 1e-3
ITERS = 25


def l1_trend(y, lam, iters=ITERS):
    """y (m, 2) metres, no NaN -> filtered (m, 2)."""
    m = len(y)
    if lam <= 0 or m < 5:
        return y.copy()
    D = sps.diags([np.ones(m - 2), -2 * np.ones(m - 2), np.ones(m - 2)], [0, 1, 2], shape=(m - 2, m)).tocsc()
    I = sps.identity(m, format="csc")
    x = y.copy()
    for _ in range(iters):
        g = np.hypot(*(D @ x).T)
        w = 1.0 / np.maximum(g, EPS)
        A = (I + lam * (D.T @ sps.diags(w) @ D)).tocsc()
        x = splu(A).solve(y)
    return x


def filter_ball(xy_m, lam):
    """(n, 2) metres with NaN gaps -> filtered per seen stretch; gaps stay NaN."""
    out = xy_m.copy()
    seen = np.isfinite(xy_m[:, 0])
    edges = np.flatnonzero(np.diff(np.concatenate([[0], seen.astype(np.int8), [0]])))
    for a, b in zip(edges[::2], edges[1::2]):
        out[a:b] = l1_trend(xy_m[a:b], lam)
    return out


def dv_of(xy, k=3):
    v = np.gradient(xy, axis=0) * FPS
    out = np.full(len(xy), np.nan)
    out[k:-k] = np.hypot(*(v[2 * k:] - v[:-2 * k]).T)
    return out


def peaks(xy, valid):
    from scipy.ndimage import maximum_filter1d
    d = np.nan_to_num(dv_of(xy))
    p = np.flatnonzero((d > 0) & (d == maximum_filter1d(d, 7)) & valid)
    return p, d[p]


def pr(p, h, tf, th, tol=5):
    """P, R, F1 of peaks above th against true frames tf (one to one, nearest, within tol)."""
    p = p[h >= th]
    if len(p) == 0:
        return 0.0, 0.0, 0.0, 0
    used, hit = set(), 0
    for f in tf:
        lo, hi = np.searchsorted(p, [f - tol, f + tol + 1])
        cand = [j for j in range(lo, hi) if j not in used]
        if cand:
            used.add(min(cand, key=lambda j: abs(p[j] - f)))
            hit += 1
    P, R = hit / len(p), hit / max(len(tf), 1)
    return P, R, 2 * P * R / max(P + R, 1e-9), len(p)


def dfl_cmd(args):
    import dfl_data as dd
    import touch_labels as tl
    import train_touch as tt
    tt.REAL, tt.REAL_CUT = "sharp", args.cut
    L, W = dd.PITCH
    ths = np.arange(1.0, 15.01, 0.5)
    names = ["clean sharp", "noisy"] + ["noisy l1 lam {:g}".format(l) for l in args.lams if l > 0]
    acc = {k: [] for k in names}
    for g in dd.GAMES[:args.games]:
        d = dd.load(g)
        noisy = tt.yolo_real(d, seed=tt.TEST_SEED)[:, :2] * np.array([L, W])
        sharp = dd.prepare_ball(d, tt.TEST_SEED, blur=False)[:, :2] * np.array([L, W])
        seen = np.isfinite(noisy[:, 0])
        valid = seen & (d["period"] > 0)
        tf = np.array(sorted(tl.load(g)[0]))
        tf = tf[valid[tf]]
        mins = valid.sum() / FPS / 60
        balls = {"clean sharp": sharp, "noisy": noisy}
        for l in args.lams:
            if l > 0:
                balls["noisy l1 lam {:g}".format(l)] = filter_ball(noisy, l)
        for k, xy in balls.items():
            p, h = peaks(xy, valid)
            rms = np.sqrt(np.nanmean(np.sum((xy - sharp)[valid] ** 2, axis=1)))
            acc[k].append([rms] + [pr(p, h, tf, th)[:3] + (pr(p, h, tf, th)[3] / mins,) for th in ths])
        print("  {} done".format(g), flush=True)
    print("\nmean over {} DFL matches (touches within 0.2 s; best F1 over peak thresholds)".format(args.games))
    print("{:<20} {:>7} | {:>6} {:>5} {:>5} {:>5} {:>8}".format("ball", "RMS m", "th", "P", "R", "F1", "pk/min"))
    for k in names:
        rms = np.mean([r[0] for r in acc[k]])
        tab = np.mean([np.array(r[1:]) for r in acc[k]], axis=0)       # (thresholds, 4)
        i = int(np.argmax(tab[:, 2]))
        print("{:<20} {:>7.2f} | {:>6.1f} {:>5.2f} {:>5.2f} {:>5.2f} {:>8.1f}".format(k, rms, ths[i], *tab[i]))


def ours_cmd(args):
    from games import GAMES, GAME
    L, W = GAMES[GAME]["pitch"]
    p = os.path.join(ROOT, "data", "cache", "yolo", "ball_refined_bambezius.npy")
    ref = np.load(p).astype(np.float64) * np.array([L, W])
    out = filter_ball(ref, args.lam)
    np.save(p.replace(".npy", "_l{:g}.npy".format(args.lam)), (out / np.array([L, W])).astype(np.float32))
    print("saved; frames changed by > 0.1 m: {:.0%} of seen".format(
        np.nanmean(np.hypot(*(out - ref).T)[np.isfinite(ref[:, 0])] > 0.1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("dfl", "ours"))
    ap.add_argument("--lams", type=float, nargs="+", default=[0.5, 1, 2, 4, 8])
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--cut", type=float, default=0.95)
    args = ap.parse_args()
    {"dfl": dfl_cmd, "ours": ours_cmd}[args.cmd](args)


if __name__ == "__main__":
    main()
