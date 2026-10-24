"""Which features separate dead from live, and home from away? Aggregate answer only.

For each pipeline feature, its mean and quartiles on public-window frames grouped by the ground
truth label (H, A, D). Prints summary statistics only, never a frame number or a value per frame.

    python tools/feature_table.py

Seventeen possessions and four dead stretches: this checks that a signal points the right way and
is roughly the right size. It is not for fine-tuning weights.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from pipeline.features import build_features  # noqa: E402
from pipeline.decode import emissions  # noqa: E402
from report import PUBLIC_GT  # noqa: E402

NAMES = {pm.H: "H", pm.A: "A", pm.D: "D"}


def stats(values):
    """'mean (p25 / p50 / p75) nan%' of one group."""
    ok = values[~np.isnan(values)]
    nan_share = 100.0 * (1 - len(ok) / float(len(values))) if len(values) else 0.0
    if not len(ok):
        return "{:>26}  nan {:3.0f}%".format("-", nan_share)
    q = np.percentile(ok, (25, 50, 75))
    return "{:6.2f} ({:5.2f}/{:5.2f}/{:5.2f})  nan {:3.0f}%".format(ok.mean(), *q, nan_share)


def main():
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    graded = np.flatnonzero(gt != pm.X)
    sl = slice(graded[0], graded[-1] + 1)
    truth = gt[sl]

    f = build_features()
    f["d_away - d_home"] = f["d_away"] - f["d_home"]
    scores = emissions(f)
    f["score H-A"] = (scores[:, 0] - scores[:, 1]).astype(np.float32)
    f["score D-max(H,A)"] = (scores[:, 2] - np.max(scores[:, :2], axis=1)).astype(np.float32)

    counts = "  ".join("{} {}".format(NAMES[k], int(np.sum(truth == k))) for k in NAMES)
    print("PUBLIC WINDOW  frames by truth: {}".format(counts))
    print("  {:<18} {}".format("feature", "   ".join(
        "{:<40}".format("GT " + NAMES[k] + ": mean (p25/p50/p75)") for k in NAMES)))
    for name, values in f.items():
        window = values[sl].astype(np.float64)
        print("  {:<18} {}".format(name, "   ".join(
            "{:<40}".format(stats(window[truth == k])) for k in NAMES)))

    # The team signals as classifiers on alive frames: how often do they point at the right side?
    alive = (truth == pm.H) | (truth == pm.A)
    home = truth == pm.H
    for name, values in (("nearest player", f["d_away - d_home"][sl]),
                         ("tag_team", f["tag_team"][sl])):
        speaks = alive & ~np.isnan(values) & (values != 0)
        right = np.where(home, values > 0, values < 0)
        print("  {:<15} speaks on {:5.1f}% of alive frames, right side {:5.1f}% of those".format(
            name, 100.0 * speaks.sum() / max(alive.sum(), 1),
            100.0 * right[speaks].mean() if speaks.any() else 0.0))


if __name__ == '__main__':
    main()
