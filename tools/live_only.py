"""Score the team decision alone, with the true stoppages taken out. Public window, aggregate only.

    python tools/live_only.py [--set k=v ...]

For each ball-fill method (none, hold, extrapolate, interp):
  live-only     the normal decode, scored with every true D frame made ungraded (X): spellF1,
                macroF1, and on true live frames the share called the right team / wrong team / D
  oracle D      the decoder is told the true stoppages (D forced on true D frames and ruled out
                on true live ones), so only H vs A is left to decide; then scored as usual

This uses the ground truth inside the decoder, so it is an analysis, never a submission.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from report import PUBLIC_GT  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402
from pipeline.features import build_features  # noqa: E402
from pipeline.decode import emissions, transition_cost, touch_discount, viterbi, H, A, D, X  # noqa: E402
from pipeline.run import parse_overrides  # noqa: E402

# Params
FORCE = 1e3
METHODS = (None, "hold", "extrapolate", "interp")


def decode_with(scores, f, p):
    cost = transition_cost(p)
    discount = touch_discount(f, p)
    codes = np.full(len(scores), X, dtype=np.int8)
    for _, first, last in half_video_blocks(GAME):
        codes[first:last + 1] = viterbi(scores[first:last + 1], cost, discount[first:last + 1])
    return codes


def score(gt, pred):
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    spell = pm.spell_scores(gt, filled, runs)
    cm3 = pm.scored_confusion(pm.confusion(gt, pred))
    return spell["mean_f1"], pm.macro_f1(cm3)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="override PARAMS")
    args = parser.parse_args()
    p = parse_overrides(args.set)

    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    n = len(gt)
    live_t = (gt == H) | (gt == A)
    gt_live = np.where(gt == D, X, gt)

    print("PUBLIC WINDOW, team decision only ({} true live frames, {} true D frames made X)"
          .format(int(live_t.sum()), int(np.sum(gt == D))))
    print("  {:<12} {:>18} {:>11} {:>9} {:>7}   {:>18}".format(
        "fill", "live-only spF1/mF1", "team right", "wrong", "as D", "oracle-D spF1/mF1"))
    for method in METHODS:
        f = build_features(ball_fill=method)
        scores = emissions(f, p)[:n]
        pred = decode_with(scores, f, p)[:n]
        s1, m1 = score(gt_live, pred)
        right = np.mean(pred[live_t] == gt[live_t])
        wrong = np.mean(np.isin(pred[live_t], (H, A)) & (pred[live_t] != gt[live_t]))
        as_d = np.mean(pred[live_t] == D)

        forced = scores.copy()
        graded = gt != X
        forced[graded & (gt == D), D] += FORCE
        forced[graded & live_t, D] -= FORCE
        oracle = decode_with(forced, f, p)[:n]
        s2, m2 = score(gt, oracle)
        print("  {:<12} {:>9.4f} / {:.4f} {:>10.1f}% {:>8.1f}% {:>6.1f}%   {:>9.4f} / {:.4f}".format(
            method or "none", s1, m1, 100 * right, 100 * wrong, 100 * as_d, s2, m2))


if __name__ == '__main__':
    main()
