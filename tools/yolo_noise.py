"""Real YOLO correction sequences from our match, for transplanting onto DFL's ball. Noise only:
no labels, no events; aggregate printout only.

    python tools/yolo_noise.py [--cut 0.95]

c(t) = refined ball - tracking ball (m), tools/yolo_ball.py refine, on frames where the tracking ball
is seen (0 where YOLO made no correction). Sharp disagreements (|second difference over 3 frames|
above the --cut quantile of corrected frames) are likely real touches the tracking blurs: those
frames +-HALF_S are cut out, and the rest is split into chunks at the cuts and the tracking gaps.
-> data/cache/yolo/yolo_err_<model>_c<cut>.npz (cx, cy: metres, concatenated chunks; starts: chunk
offsets).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]

OUT = os.path.join(ROOT, "data", "cache", "yolo")
HALF_S = 0.5
FPS = 25
MIN_CHUNK = 10


def path(model="bambezius", cut=0.95):
    return os.path.join(OUT, "yolo_err_{}_c{:g}.npz".format(model, cut))


def build(model="bambezius", cut=0.95):
    from atd_cache import load_arrays
    from games import GAMES, GAME
    L, W = GAMES[GAME]["pitch"]
    trk = load_arrays("tactical", raw=True)["ball"][:, :2].astype(np.float64)
    ref = np.load(os.path.join(OUT, "ball_refined_{}.npy".format(model))).astype(np.float64)
    n = min(len(trk), len(ref))
    c = (ref[:n] - trk[:n]) * np.array([L, W])
    seen = np.isfinite(c[:, 0])
    c[~seen] = 0.0
    corr = seen & (np.hypot(*c.T) > 1e-6)
    s = np.zeros(n)
    s[3:-3] = np.hypot(*(c[6:] - 2 * c[3:-3] + c[:-6]).T)
    s[~seen] = 0.0
    th = np.quantile(s[corr], cut) if cut < 1 else np.inf
    bad = np.zeros(n, bool)
    h = int(HALF_S * FPS)
    for k in np.flatnonzero(s > th):
        bad[max(k - h, 0):k + h + 1] = True
    keep = seen & ~bad
    edges = np.flatnonzero(np.diff(np.concatenate([[0], keep.astype(np.int8), [0]])))
    chunks = [c[a:b] for a, b in zip(edges[::2], edges[1::2]) if b - a >= MIN_CHUNK]
    starts = np.cumsum([0] + [len(x) for x in chunks])[:-1]
    allc = np.concatenate(chunks)
    np.savez(path(model, cut), c=allc.astype(np.float32), starts=starts)
    nz = np.hypot(*allc.T) > 1e-6
    print("seen {:.0%}; corrected {:.0%} of seen; cut {:.1%} of seen (threshold {:.2f} m); library {} chunks, "
          "{:.0f} min, corrected share {:.0%}, |c| median {:.2f} m".format(
              seen.mean(), corr[seen].mean(), bad[seen].mean(), th, len(chunks), len(allc) / FPS / 60,
              nz.mean(), np.median(np.hypot(*allc[nz].T))))
    return path(model, cut)


def load(model="bambezius", cut=0.95):
    p = path(model, cut)
    if not os.path.exists(p):
        build(model, cut)
    z = np.load(p)
    return z["c"].astype(np.float64), z["starts"]


def paste(n, seen, lib, rng):
    """(n, 2) metres: library chunks laid end to end over the seen frames (random chunk, random start
    inside it, random x / y sign); 0 on unseen frames."""
    c, starts = lib
    ends = np.append(starts[1:], len(c))
    out = np.zeros((n, 2))
    idx = np.flatnonzero(seen)
    t = 0
    while t < len(idx):
        k = rng.integers(len(starts))
        a = rng.integers(starts[k], max(ends[k] - MIN_CHUNK, starts[k]) + 1)
        seg = c[a:ends[k]] * rng.choice((-1.0, 1.0), size=2)
        m = min(len(seg), len(idx) - t)
        out[idx[t:t + m]] = seg[:m]
        t += m
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cut", type=float, default=0.95)
    ap.add_argument("--model", default="bambezius")
    a = ap.parse_args()
    build(a.model, a.cut)
