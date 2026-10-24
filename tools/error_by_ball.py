"""Where do a submission's mistakes happen: with the ATD ball seen, or inside a ball gap?

Public window only, aggregate only. For each group of frames (ball seen, short gap, long gap) it
prints how many graded frames fall there and what share of them are wrong, split by kind of error.
Also counts true possession changes that happen inside a gap vs with the ball seen.

    python tools/error_by_ball.py evaluation/results/v2.csv

If most wrong frames sit inside gaps, a better ball tracker is worth building. If they sit where
the ball is seen, the problem is the decoding of the evidence, not the ball track.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from atd_cache import load_arrays  # noqa: E402
from report import PUBLIC_GT, FPS  # noqa: E402

# Params
SHORT_GAP_S = 3.0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submission")
    args = parser.parse_args()

    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    runs = pm.valid_runs(gt)
    pred = pm.fill_abstentions(pm.label_codes(pm.read_label_csv(args.submission)), runs)

    ball = load_arrays("tactical")["ball"]
    seen = ~np.isnan(ball[:, 0])
    gap_len = np.zeros(len(seen))
    edges = np.diff(np.concatenate(([0], (~seen).astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        gap_len[a:b] = (b - a) / float(FPS)

    graded = gt != pm.X
    alive_t = (gt == pm.H) | (gt == pm.A)
    alive_p = (pred == pm.H) | (pred == pm.A)
    groups = (("ball seen", seen),
              ("gap < {:g}s".format(SHORT_GAP_S), ~seen & (gap_len < SHORT_GAP_S)),
              ("gap >= {:g}s".format(SHORT_GAP_S), ~seen & (gap_len >= SHORT_GAP_S)))

    total_wrong = np.sum(graded & (pred != gt))
    print("PUBLIC WINDOW  {}  ({} graded frames, {} wrong)".format(
        os.path.basename(args.submission), int(graded.sum()), int(total_wrong)))
    print("  {:<12} {:>7} {:>7} {:>9} {:>11} {:>11} {:>11}".format(
        "frames", "share", "wrong", "of all err", "wrong team", "dead->alive", "alive->dead"))
    for name, mask in groups:
        m = graded & mask
        n = int(m.sum())
        if not n:
            print("  {:<12} {:>6.1f}%".format(name, 0.0))
            continue
        wrong = m & (pred != gt)
        team = m & alive_t & alive_p & (pred != gt)
        d2a = m & (gt == pm.D) & alive_p
        a2d = m & alive_t & (pred == pm.D)
        print("  {:<12} {:>6.1f}% {:>6.1f}% {:>8.1f}% {:>10.1f}% {:>10.1f}% {:>10.1f}%".format(
            name, 100.0 * n / graded.sum(), 100.0 * wrong.sum() / n,
            100.0 * wrong.sum() / max(total_wrong, 1),
            *(100.0 * x.sum() / n for x in (team, d2a, a2d))))

    # Ball-seen frames only: which situations carry the errors? Buckets by ball speed and by the
    # distance from the ball to the nearest tracked player of either team.
    from pipeline.features import build_features
    f = build_features()
    speed = f["ball_speed"]
    near = np.fmin(f["d_home"], f["d_away"])
    base = graded & seen
    for title, values, edges in (("ball speed m/s", speed, (0, 2, 8, 15, 1e9)),
                                 ("nearest player m", near, (0, 1.5, 4, 8, 1e9))):
        print("  ball seen, by {}:".format(title))
        for low, high in zip(edges[:-1], edges[1:]):
            m = base & (values >= low) & (values < high)
            n = int(m.sum())
            if not n:
                continue
            team = m & alive_t & alive_p & (pred != gt)
            d2a = m & (gt == pm.D) & alive_p
            print("    {:>4g}-{:<4g} {:>5.1f}% of seen   wrong team {:5.1f}%   dead->alive {:5.1f}%"
                  "   true D {:5.1f}%".format(low, high if high < 1e9 else np.inf,
                                             100.0 * n / base.sum(),
                                             100.0 * team.sum() / n, 100.0 * d2a.sum() / n,
                                             100.0 * np.sum(m & (gt == pm.D)) / n))

    # Where the true changes of possession happen: is the ball seen at that moment?
    kinds = {"seen": 0, "in gap": 0}
    for frame, before, after in pm.transitions(gt, runs):
        kinds["seen" if seen[frame] and seen[frame - 1] else "in gap"] += 1
    print("  true label changes: ball seen {}   inside/at a gap {}".format(
        kinds["seen"], kinds["in gap"]))


if __name__ == '__main__':
    main()
