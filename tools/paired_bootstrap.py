"""Is B really better than A on the public window, or is the gap noise? Paired bootstrap.

Resamples the true possessions with replacement -- the same draw for both submissions, which is
what makes it paired -- and scores spellF1 the way possession_metrics.bootstrap_spell_f1 does.
Prints the observed gap, its 5-95% interval and how often B beats A. Aggregate only.

    python tools/paired_bootstrap.py evaluation/results/v3_wt2.csv evaluation/results/v5.csv

Seventeen possessions: an interval that straddles 0 means the public window cannot tell them
apart, whatever the point estimates say.
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
from report import PUBLIC_GT  # noqa: E402

# Params
N_DRAWS = 2000
SEED = 0


def flags_of(path, gt, runs, gt_spells):
    """(matched flags per IoU threshold x true possession, number of predicted possessions)."""
    pred = pm.fill_abstentions(pm.label_codes(pm.read_label_csv(path)), runs)
    pred_spells = pm.possession_spells(pred, runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)
    flags = np.vstack([pm.matched_gt_flags(overlaps, len(gt_spells), t) for t in pm.SPELL_IOUS])
    return flags, len(pred_spells)


def score(flags, n_pred, picks):
    """spellF1 on one resample, as in possession_metrics.bootstrap_spell_f1."""
    n_gt = len(picks)
    f1s = []
    for row in flags:
        tp = int(row[picks].sum())
        fp, fn = max(0, n_pred - tp), n_gt - tp
        f1s.append(2.0 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return np.mean(f1s)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("a")
    parser.add_argument("b")
    args = parser.parse_args()

    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    runs = pm.valid_runs(gt)
    gt_spells = pm.possession_spells(gt, runs)
    n_gt = len(gt_spells)
    fa, na = flags_of(args.a, gt, runs, gt_spells)
    fb, nb = flags_of(args.b, gt, runs, gt_spells)

    everything = np.arange(n_gt)
    observed = score(fb, nb, everything) - score(fa, na, everything)
    rng = np.random.RandomState(SEED)
    gaps = np.empty(N_DRAWS)
    for draw in range(N_DRAWS):
        picks = rng.randint(0, n_gt, n_gt)
        gaps[draw] = score(fb, nb, picks) - score(fa, na, picks)
    low, high = np.percentile(gaps, (5, 95))
    print("B - A spellF1 (public, {} possessions, {} paired draws)".format(n_gt, N_DRAWS))
    print("  A {}   B {}".format(os.path.basename(args.a), os.path.basename(args.b)))
    print("  observed {:+.4f}   5-95% [{:+.4f}, {:+.4f}]   B > A in {:.0f}% of draws".format(
        observed, low, high, 100.0 * np.mean(gaps > 0)))


if __name__ == '__main__':
    main()
