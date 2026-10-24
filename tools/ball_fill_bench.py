"""How well do the simple ball guesses (pipeline/ball_fill.py) work? Hide the ball where it is seen.

Inside the halves, cut fake gaps out of seen stretches, with lengths drawn from our real gaps
(3 s and longer), guess the ball with each method and compare with the real track. Aggregate only:

    python tools/ball_fill_bench.py [--n 2000] [--seed 0]

Per gap-length bucket and method:
  median / p75 error (m)    distance from the guess to the real ball
  within 5 m                share of frames
  team right                frames where a player is within CONTROL_M of the real ball (someone
                            has it): the nearest player to the guess is on that player's team
  team right, first 2 s     the same, over the first 2 s of each gap only
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from atd_cache import load_arrays  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402
from pipeline.ball_fill import fill_gap, METHODS, VEL_WINDOW, FPS, LENGTH, WIDTH, _runs  # noqa: E402

# Params
MIN_GAP_S = 3.0
CONTROL_M = 2.0
EARLY_S = 2.0
BUCKETS = ((3, 10), (10, 30), (30, 1e9))


def nearest_team(px, py, team, x, y):
    """Team (0/1) of the nearest tracked player to (x, y) per frame, and that distance."""
    d = np.hypot(px - x[:, None], py - y[:, None])
    d = np.where(np.isnan(d), np.inf, d)
    i = np.argmin(d, axis=1)
    return team[i], d[np.arange(len(i)), i]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=2000, help="fake gaps")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    atd = load_arrays("tactical")
    ball, players, team = atd["ball"], atd["players"], atd["team"]
    x = ball[:, 0].astype(np.float64) * LENGTH
    y = ball[:, 1].astype(np.float64) * WIDTH
    px = players[:, :, 0].astype(np.float64) * LENGTH
    py = players[:, :, 1].astype(np.float64) * WIDTH
    seen = ~np.isnan(x)
    in_half = np.zeros(len(x), bool)
    for _, first, last in half_video_blocks(GAME):
        in_half[first:last + 1] = True

    real = np.array([b - a for a, b in _runs(~seen & in_half) if b - a >= MIN_GAP_S * FPS])
    stretches = [(a, b) for a, b in _runs(seen & in_half)]
    # A fake gap needs VEL_WINDOW + 1 seen frames before it and one after, inside the stretch.
    lengths = np.array([b - a for a, b in stretches])

    rows = {m: [] for m in METHODS}   # (bucket, err, team_ok or -1, early)
    made = 0
    for _ in range(args.n * 20):
        if made >= args.n:
            break
        g = int(rng.choice(real))
        fits = np.flatnonzero(lengths >= g + VEL_WINDOW + 2)
        if not len(fits):
            continue
        a, b = stretches[int(rng.choice(fits, p=lengths[fits] / lengths[fits].sum()))]
        start = int(rng.integers(a + VEL_WINDOW + 1, b - g))
        stop = start + g
        lo, hi = start - VEL_WINDOW - 1, stop + 1
        hx, hy = x[lo:hi].copy(), y[lo:hi].copy()
        s, e = start - lo, stop - lo
        hx[s:e], hy[s:e] = np.nan, np.nan
        tx, ty = x[start:stop], y[start:stop]
        true_team, true_d = nearest_team(px[start:stop], py[start:stop], team, tx, ty)
        has = true_d <= CONTROL_M
        early = np.arange(g) < EARLY_S * FPS
        bucket = next(i for i, (l, h) in enumerate(BUCKETS) if l <= g / FPS < h)
        for m in METHODS:
            gx, gy = fill_gap(hx, hy, s, e, m)
            err = np.hypot(gx - tx, gy - ty)
            guess_team, _ = nearest_team(px[start:stop], py[start:stop], team, gx, gy)
            ok = np.where(has, (guess_team == true_team).astype(int), -1)
            rows[m].append((np.full(g, bucket), err, ok, early))
        made += 1

    print("FAKE GAPS cut from seen ball: {}  (real gaps >= {:g} s: {}, lengths drawn from them)"
          .format(made, MIN_GAP_S, len(real)))
    print("  {:<12} {:<11} {:>7} {:>8} {:>8} {:>9} {:>11} {:>14}".format(
        "gap", "method", "frames", "med m", "p75 m", "<=5 m", "team right", "team right 2s"))
    for i, (l, h) in enumerate(BUCKETS):
        name = "{:g}-{} s".format(l, "" if h > 1e8 else "{:g}".format(h))
        for m in METHODS:
            parts = [r for r in rows[m]]
            bk = np.concatenate([p[0] for p in parts])
            sel = bk == i
            if not sel.any():
                continue
            err = np.concatenate([p[1] for p in parts])[sel]
            ok = np.concatenate([p[2] for p in parts])[sel]
            early = np.concatenate([p[3] for p in parts])[sel]
            okm = ok >= 0
            print("  {:<12} {:<11} {:>7} {:>8.1f} {:>8.1f} {:>8.1f}% {:>10.1f}% {:>13.1f}%".format(
                name, m, int(sel.sum()), np.median(err), np.percentile(err, 75),
                100.0 * np.mean(err <= 5.0), 100.0 * np.mean(ok[okm] == 1),
                100.0 * np.mean(ok[okm & early] == 1)))
    print("  (team right: frames where someone is within {:g} m of the real ball; "
          "a coin flip is ~50%)".format(CONTROL_M))


if __name__ == '__main__':
    main()
