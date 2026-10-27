"""Why does pipeline/track_fix miss an impostor keeper? Aggregates over a public-possession window.

    python tools/keeper_window.py 9 13

For each goal, over the window (true possessions #first..#last, as in tools/spell_table.py), in
seconds: how long a slot of the attacking team is in that goal's box / within 25 m of the goal,
how long it is the deepest of the moving defenders, how long a defender stands still (several
thresholds), and per attacking-team piece touching the box: its length and share in the box.
Only seconds and shares are printed: no frames, positions or slot ids.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from report import PUBLIC_GT, FPS  # noqa: E402
from pipeline import track_fix as tf  # noqa: E402

STILL_M = (0.15, 0.3, 0.6, 1.0)


def main():
    first, last = int(sys.argv[1]), int(sys.argv[-1])
    from games import GAME, GAMES, half_video_blocks
    from atd_cache import load_arrays
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    spells = pm.possession_spells(gt, pm.valid_runs(gt))
    a, b = spells[first - 1][0], spells[last - 1][1]
    atd = load_arrays()
    players, team = atd["players"], np.asarray(atd["team"])
    length, width = GAMES[GAME]["pitch"]
    half = next((s, e + 1) for _, s, e in half_video_blocks(GAME) if s <= a < e + 1)
    px = players[:, :, 0].astype(np.float64) * length
    py = players[:, :, 1].astype(np.float64) * width
    tracked = ~np.isnan(px)
    goal = {}
    for side in (0, 1):
        cols = team == side
        mx = np.nanmedian(np.nanmean(np.where(tracked[half[0]:half[1]][:, cols],
                                              px[half[0]:half[1]][:, cols], np.nan), axis=1))
        goal[side] = 0.0 if mx < length / 2.0 else length
    w = int(tf.STILL_S * FPS)
    import pandas as pd
    rng = np.maximum(*[(lambda r: (r.max() - r.min()).values)(
        pd.DataFrame(v[a - w:b + w]).rolling(w, center=True, min_periods=w)) for v in (px, py)])[w:-w]
    sl = slice(a, b)
    print("window #{}-#{}: {:.1f} s".format(first, last, (b - a) / float(FPS)))
    for side, name in ((0, "Barcelona's goal"), (1, "Athletic's goal")):
        depth = np.abs(px[sl] - goal[side])
        central = np.abs(py[sl] - width / 2.0)
        box = tracked[sl] & (depth <= tf.BOX_DEPTH_M) & (central <= tf.BOX_HALF_M)
        near25 = tracked[sl] & (np.hypot(depth, central) <= 25.0)
        att, dfd = team != side, team == side
        print("\n" + name)
        print("  attacking-team slot in the box: {:.1f} s; within 25 m of the goal: {:.1f} s".format(
            box[:, att].any(axis=1).sum() / FPS, near25[:, att].any(axis=1).sum() / FPS))
        print("  defending-team slot in the box: {:.1f} s".format(box[:, dfd].any(axis=1).sum() / FPS))
        for m in STILL_M:
            still = np.where(np.isnan(rng), False, rng < m)
            moving_def = box & dfd[None, :] & ~still
            d_def = np.min(np.where(moving_def, depth, np.inf), axis=1)
            deepest = (box & att[None, :] & (depth < d_def[:, None])).any(axis=1)
            ghost = (box & dfd[None, :] & still).any(axis=1)
            print("  still < {:.2f} m: still defender in box {:.1f} s; attacker deepest of the moving "
                  "{:.1f} s; both {:.1f} s".format(m, ghost.sum() / FPS, deepest.sum() / FPS,
                                                    (ghost & deepest).sum() / FPS))
        pieces = []
        for i in np.flatnonzero(att):
            for s, e in tf._runs(tracked[:, i]):
                if e > a and s < b:
                    in_box_w = box[max(s, a) - a:min(e, b) - a, i]
                    if in_box_w.any():
                        d = np.abs(px[s:e, i] - goal[side])
                        c = np.abs(py[s:e, i] - width / 2.0)
                        whole = (d <= tf.BOX_DEPTH_M) & (c <= tf.BOX_HALF_M)
                        pieces.append(((e - s) / FPS, whole.mean(), in_box_w.sum() / FPS))
        for p in sorted(pieces, key=lambda p: -p[2]):
            print("  attacking-team piece: {:.1f} s long, {:.0f}% of it in the box, {:.1f} s in the "
                  "box inside the window".format(*p[:1], 100 * p[1], p[2]))


if __name__ == '__main__':
    main()
