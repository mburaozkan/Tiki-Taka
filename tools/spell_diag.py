"""Why does spellF1 come out where it does? Aggregate diagnosis of possession matching.

spellF1 matches each true possession to at most one predicted possession of the same team, and
counts it only if their temporal IoU clears a threshold (0.3 ... 0.7, averaged). So the score is
decided by a handful of possessions crossing or missing those thresholds, not by frame accuracy.
This prints, per submission and for the public window only:

  - the best same-team IoU each true possession reaches, in buckets;
  - for those below 0.5, why: no same-team prediction at all, split into pieces, swallowed by a
    longer prediction (merged), or a single prediction that is too short or shifted;
  - the match rate by true possession length (are the misses the short ones?);
  - start/end offsets of the best match (frames, + = predicted later than truth);
  - spurious predicted possessions (overlap no true possession of their team), by length.

    python tools/spell_diag.py evaluation/results/v3_wt2.csv evaluation/results/v4_mix.csv
    python tools/spell_diag.py --profile          # is there evidence inside the true possessions?

Counts and medians only. Seventeen possessions: this says which kind of failure dominates, not
which submission is better.
"""

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from report import PUBLIC_GT, FPS  # noqa: E402

# Params
IOU_EDGES = (0.0, 0.3, 0.5, 0.7, 1.01)
LENGTH_EDGES_S = (0, 5, 15, 1e9)
STRETCH = 1.5  # a prediction this many times longer/shorter than the truth counts as merged/short


def classify(gt_spell, pred_spells, overlaps):
    """(best_iou, cause, best_pred) for one true possession. overlaps: [(iou, j)] same team."""
    if not overlaps:
        return 0.0, "no same-team pred", None
    best_iou, j = max(overlaps)
    if best_iou >= 0.5:
        return best_iou, "matched >=0.5", j
    gs, ge, _ = gt_spell
    ps, pe, _ = pred_spells[j]
    gt_len, pred_len = ge - gs, pe - ps
    if len(overlaps) >= 2 and pred_len < gt_len:
        cause = "split"
    elif pred_len > STRETCH * gt_len:
        cause = "merged (pred too long)"
    elif pred_len * STRETCH < gt_len:
        cause = "pred too short"
    else:
        cause = "shifted"
    return best_iou, cause, j


def bucket(value, edges):
    for low, high in zip(edges[:-1], edges[1:]):
        if low <= value < high:
            return low, high
    return edges[-2], edges[-1]


def diagnose(path, gt):
    runs = pm.valid_runs(gt)
    pred = pm.fill_abstentions(pm.label_codes(pm.read_label_csv(path)), runs)
    gt_spells = pm.possession_spells(gt, runs)
    pred_spells = pm.possession_spells(pred, runs)
    pairs = pm.spell_overlaps(gt_spells, pred_spells)
    by_gt = {i: [] for i in range(len(gt_spells))}
    for iou, i, j in pairs:
        by_gt[i].append((iou, j))

    result = pm.evaluate(gt, pred)
    print("=" * 90)
    print("{}   spellF1 {:.4f}   possessions pred {} / gt {}".format(
        os.path.basename(path), result["spell_f1_mean"], len(pred_spells), len(gt_spells)))

    rows = [classify(gt_spells[i], pred_spells, by_gt[i]) for i in range(len(gt_spells))]

    counts = {}
    for iou, _, _ in rows:
        key = bucket(iou, IOU_EDGES)
        counts[key] = counts.get(key, 0) + 1
    print("  best same-team IoU per true possession:  " + "  ".join(
        "[{:.1f},{:.1f}): {}".format(low, min(high, 1.0), counts.get((low, high), 0))
        for low, high in zip(IOU_EDGES[:-1], IOU_EDGES[1:])))

    causes = {}
    for _, cause, _ in rows:
        causes[cause] = causes.get(cause, 0) + 1
    print("  outcome:  " + "   ".join("{} {}".format(k, v) for k, v in sorted(causes.items())))

    print("  matched (IoU >= 0.5) by true possession length:")
    for low, high in zip(LENGTH_EDGES_S[:-1], LENGTH_EDGES_S[1:]):
        group = [rows[i] for i, (s, e, _) in enumerate(gt_spells)
                 if low <= (e - s) / float(FPS) < high]
        if group:
            print("    {:>3g}-{:<5g}s  {} of {}".format(
                low, high if high < 1e9 else np.inf,
                sum(1 for iou, _, _ in group if iou >= 0.5), len(group)))

    starts = [pred_spells[j][0] - gt_spells[i][0] for i, (_, _, j) in enumerate(rows)
              if j is not None]
    ends = [pred_spells[j][1] - gt_spells[i][1] for i, (_, _, j) in enumerate(rows)
            if j is not None]
    if starts:
        print("  best-match offsets (frames, + = pred later):  start median {:+.0f} "
              "(|.| median {:.0f})   end median {:+.0f} (|.| median {:.0f})".format(
                  np.median(starts), np.median(np.abs(starts)),
                  np.median(ends), np.median(np.abs(ends))))

    touched = {j for _, _, j in pairs}
    spurious = [(e - s) / float(FPS) for j, (s, e, _) in enumerate(pred_spells)
                if j not in touched]
    print("  spurious predicted possessions: {}   lengths (s): {}".format(
        len(spurious), "  ".join("{:>3g}-{:<4g}: {}".format(
            low, high if high < 1e9 else np.inf,
            sum(1 for x in spurious if low <= x < high))
            for low, high in zip(LENGTH_EDGES_S[:-1], LENGTH_EDGES_S[1:]))))


def profile(gt):
    """Is there evidence inside the true possessions? Short vs long, model-independent.

    For the frames of each group of true possessions: how often the ball is seen, and how often
    each team signal points at the right side when it speaks. If the signals are right inside the
    short possessions, the decoder is swallowing them; if not, the evidence is missing.
    """
    from pipeline.features import build_features
    f = build_features()
    runs = pm.valid_runs(gt)
    spells = pm.possession_spells(gt, runs)
    signals = (("nearest", f["d_away"] - f["d_home"]), ("carrier", f["c_away"] - f["c_home"]),
               ("tag_team", f["tag_team"]), ("last_touch", f["last_touch"]))
    print("=" * 90)
    print("TRUE POSSESSION PROFILE (no submission involved)")
    for name, low, high in (("short <5s", 0, 5), ("5-15s", 5, 15), ("long >=15s", 15, 1e9)):
        group = [(s, e, k) for s, e, k in spells if low <= (e - s) / float(FPS) < high]
        if not group:
            continue
        frames = np.concatenate([np.arange(s, e) for s, e, _ in group])
        home = np.concatenate([np.full(e - s, k == pm.H) for s, e, k in group])
        lengths = [(e - s) / float(FPS) for s, e, _ in group]
        parts = ["{:<11} n={:<2} median {:4.1f}s  ball seen {:3.0f}%  ball speed med {:4.1f}".format(
            name, len(group), np.median(lengths), 100.0 * f["ball_seen"][frames].mean(),
            np.nanmedian(f["ball_speed"][frames]) if np.any(~np.isnan(f["ball_speed"][frames]))
            else np.nan)]
        for label, values in signals:
            v = values[frames]
            speaks = ~np.isnan(v) & (v != 0)
            right = np.where(home, v > 0, v < 0) & speaks
            parts.append("{} speaks {:3.0f}% right {:3.0f}%".format(
                label, 100.0 * speaks.mean(), 100.0 * right.sum() / max(speaks.sum(), 1)))
        print("  " + parts[0])
        print("      " + "   ".join(parts[1:3]))
        print("      " + "   ".join(parts[3:]))

    # The touches themselves: how many, and do they name the side that truly has the ball?
    graded = np.flatnonzero(gt != pm.X)
    sl = slice(graded[0], graded[-1] + 1)
    touch, truth = f["touch"][sl], gt[sl]
    alive = (truth == pm.H) | (truth == pm.A)
    at = (touch != 0) & alive
    right = np.where(truth == pm.H, touch > 0, touch < 0) & at
    print("  touches: {} on live frames ({:.1f} per live minute; {} on dead frames)   "
          "touching side = true owner on {:.0f}%".format(
              int(at.sum()), at.sum() / (alive.sum() / float(FPS) / 60.0),
              int(np.sum((touch != 0) & (truth == pm.D))),
              100.0 * right.sum() / max(at.sum(), 1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submissions", nargs="*", help="frame,label CSVs (globs allowed)")
    parser.add_argument("--profile", action="store_true",
                        help="also profile the true possessions: is there evidence inside them?")
    args = parser.parse_args()
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    if args.profile:
        profile(gt)
    for path in sorted({p for pattern in args.submissions for p in glob.glob(pattern)}):
        diagnose(path, gt)
    print("=" * 90)


if __name__ == '__main__':
    main()
