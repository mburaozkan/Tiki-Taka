"""One row per true possession in the public window: how each submission gets it right or wrong.

Built from labels only -- the published public ground truth and our own submissions. No frame
numbers, no clock times, no tracking values: possessions are numbered in order (#1, #2, ...) and
every timing is relative to the true possession, in seconds. Safe to share.

    python tools/spell_table.py evaluation/results/baseline_smooth50.csv evaluation/results/v5.csv

Columns per true possession:
  context   the true labels before and after it (e.g. D>[H]>A), length, team
  per submission:
    IoU     best same-team overlap (spellF1 counts a hit at 0.3 ... 0.7)
    start   predicted start - true start (s), + = late
    end     predicted end - true end (s), + = runs on past the true end
    k       number of same-team predicted pieces overlapping it (k>1 = split)
    else    when there is no same-team prediction: what was predicted over it instead
Then the spurious predictions of each submission: team, length, and which true possession they
sit inside or between.
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

NAMES = {pm.H: "H", pm.A: "A", pm.D: "D", pm.X: "X"}


def label_mix(codes):
    """'A 80% D 20%' -- the predicted labels over a stretch, largest first."""
    shares = [(np.mean(codes == k), k) for k in (pm.H, pm.A, pm.D)]
    return " ".join("{} {:.0f}%".format(NAMES[k], 100 * s)
                    for s, k in sorted(shares, reverse=True) if s >= 0.1)


def cell(gt_spell, pred_spells, pred):
    """The per-submission cell for one true possession."""
    gs, ge, gk = gt_spell
    same = [(ps, pe) for ps, pe, pk in pred_spells if pk == gk and ps < ge and gs < pe]
    if not same:
        return "{:<30}".format("MISS  else: " + label_mix(pred[gs:ge]))
    best = max(same, key=lambda s: (min(s[1], ge) - max(s[0], gs)) / float(max(s[1], ge) -
                                                                           min(s[0], gs)))
    ps, pe = best
    iou = (min(pe, ge) - max(ps, gs)) / float(max(pe, ge) - min(ps, gs))
    return "IoU {:.2f} st {:+5.1f} en {:+5.1f} k{}".format(
        iou, (ps - gs) / float(FPS), (pe - ge) / float(FPS), len(same))


def where(spell, gt_spells, gt):
    """Which true possession a spurious prediction sits in, by its midpoint."""
    mid = (spell[0] + spell[1]) // 2
    for i, (gs, ge, gk) in enumerate(gt_spells):
        if gs <= mid < ge:
            return "inside #{} ({})".format(i + 1, NAMES[gk])
    before = [i for i, (gs, ge, _) in enumerate(gt_spells) if ge <= mid]
    after = [i for i, (gs, ge, _) in enumerate(gt_spells) if gs > mid]
    gap = "between #{} and #{}".format(before[-1] + 1 if before else "-",
                                       after[0] + 1 if after else "-")
    return "{} (true {})".format(gap, NAMES[int(gt[mid])])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submissions", nargs="+", help="frame,label CSVs (globs allowed)")
    args = parser.parse_args()
    paths = [p for pattern in args.submissions for p in sorted(glob.glob(pattern))]

    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    runs = pm.valid_runs(gt)
    segs = pm.segments(gt, runs)
    gt_spells = pm.possession_spells(gt, runs)
    preds = [pm.fill_abstentions(pm.label_codes(pm.read_label_csv(p)), runs) for p in paths]
    pred_spells = [pm.possession_spells(p, runs) for p in preds]

    names = [os.path.splitext(os.path.basename(p))[0][:30] for p in paths]
    print("{:<4} {:<11} {:>5}  ".format("#", "context", "len") + "  ".join(
        "{:<30}".format(n) for n in names))
    for i, spell in enumerate(gt_spells):
        index = next(j for j, s in enumerate(segs) if s[0] == spell[0])
        before = NAMES[segs[index - 1][2]] if index > 0 and segs[index - 1][1] == spell[0] else "|"
        after = (NAMES[segs[index + 1][2]] if index + 1 < len(segs)
                 and segs[index + 1][0] == spell[1] else "|")
        context = "{}>[{}]>{}".format(before, NAMES[spell[2]], after)
        print("{:<4} {:<11} {:>4.1f}s  ".format("#{}".format(i + 1), context,
                                                (spell[1] - spell[0]) / float(FPS)) +
              "  ".join(cell(spell, ps, p) for ps, p in zip(pred_spells, preds)))

    dead = [(b - a) / float(FPS) for a, b, k in segs if k == pm.D]
    print("\ntrue dead stretches: {}  lengths (s): {}".format(
        len(dead), " ".join("{:.1f}".format(x) for x in dead)))

    for name, spells, pred in zip(names, pred_spells, preds):
        spurious = []
        for s in spells:
            if not any(gk == s[2] and gs < s[1] and s[0] < ge for gs, ge, gk in gt_spells):
                spurious.append(s)
        print("\n{}: {} predicted possessions, {} spurious (overlap no true possession of "
              "their team)".format(name, len(spells), len(spurious)))
        for s in spurious:
            print("    {} {:5.1f}s  {}".format(NAMES[s[2]], (s[1] - s[0]) / float(FPS),
                                              where(s, gt_spells, gt)))


if __name__ == '__main__':
    main()
