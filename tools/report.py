"""Summarise one or more submissions in aggregate numbers only -- safe to share outside the team.

The competition terms forbid making the match data, or any extract of it, available to a hosted
LLM. This report is built so its output can be pasted into such a conversation: it prints counts,
rates, durations and scores, and never a frame number, a coordinate or a per-frame label.

    python tools/report.py evaluation/results/baseline_smooth50.csv
    python tools/report.py evaluation/results/*.csv          # several, one block each

Two sections per file:

  1. FULL MATCH -- what the submission claims about the whole game, per half. No ground truth is
     involved, so this is the only view of the 95% of the match the public window cannot see.
  2. PUBLIC WINDOW -- scored against the 5-minute public ground truth, broken down by *kind* of
     error rather than only by score. Seventeen possessions: read it as a diagnosis, never as a
     ranking (EVALUATION.md).

Durations are in seconds. Everything temporal is computed inside each half, never across
halftime, and with abstentions filled the way the official scorer fills them.
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
from games import GAME, GAMES, half_video_blocks  # noqa: E402

# Params
FPS = GAMES[GAME]["fps"]
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "{}_possession_gt_public.csv".format(GAME))
# Possession-length buckets, in seconds. The short end is where flicker shows up.
DURATION_EDGES = (0, 1, 2, 5, 10, 20, 40, 1e9)
NAMES = {pm.H: "H", pm.A: "A", pm.D: "D", pm.X: "X"}


def half_runs():
    """[(half, start, stop)] half-open, from the published half boundaries."""
    return [(half, first, last + 1) for half, first, last in half_video_blocks(GAME)]


def bucket_counts(lengths_s):
    """Counts of durations per DURATION_EDGES bucket, as a compact one-line string."""
    counts, _ = np.histogram(lengths_s, bins=DURATION_EDGES)
    labels = ["<{:g}".format(DURATION_EDGES[1])] + [
        "{:g}-{:g}".format(a, b) for a, b in zip(DURATION_EDGES[1:-2], DURATION_EDGES[2:-1])
    ] + [">{:g}".format(DURATION_EDGES[-2])]
    return "  ".join("{}s:{}".format(label, count) for label, count in zip(labels, counts))


def describe_spells(lengths_s):
    """count, mean, median, p10, p90 of a list of durations, as one line."""
    if not len(lengths_s):
        return "n=0"
    return "n={:<4} mean {:5.1f}  median {:5.1f}  p10 {:5.1f}  p90 {:5.1f}".format(
        len(lengths_s), np.mean(lengths_s), np.median(lengths_s),
        np.percentile(lengths_s, 10), np.percentile(lengths_s, 90))


def transition_kinds(codes, runs):
    """Counter of 'H>A', 'D>H', ... over all label changes inside runs."""
    kinds = {}
    for _, before, after in pm.transitions(codes, runs):
        key = "{}>{}".format(NAMES[before], NAMES[after])
        kinds[key] = kinds.get(key, 0) + 1
    return kinds


def full_match_section(codes):
    """What the submission claims about the whole match, per half. No ground truth used."""
    print("  FULL MATCH (no ground truth -- what the file claims)")
    for half, start, stop in half_runs():
        raw = codes[start:stop]
        runs = [(start, stop)]
        filled = pm.fill_abstentions(codes, runs)
        n = stop - start

        shares = "  ".join("{} {:5.1f}%".format(NAMES[k], 100.0 * np.sum(raw == k) / n)
                           for k in (pm.H, pm.A, pm.D, pm.X))
        print("    half {}  {:>6} frames   {}".format(half, n, shares))

        segs = pm.segments(filled, runs)
        spells = [(b - a) / float(FPS) for a, b, k in segs if k in pm.POSSESSION_LABELS]
        dead = [(b - a) / float(FPS) for a, b, k in segs if k == pm.D]
        print("      possessions  {}".format(describe_spells(spells)))
        print("                   {}".format(bucket_counts(spells)))
        print("      dead spells  {}".format(describe_spells(dead)))

        kinds = transition_kinds(filled, runs)
        direct = kinds.get("H>A", 0) + kinds.get("A>H", 0)
        via_dead = sum(v for k, v in kinds.items() if k.startswith("D>"))
        print("      changes      H<>A direct {}   restarts (D>H/A) {}   stoppages (>D) {}".format(
            direct, via_dead, kinds.get("H>D", 0) + kinds.get("A>D", 0)))
        alive = np.sum((filled[start:stop] == pm.H) | (filled[start:stop] == pm.A))
        if alive:
            print("      home share of alive frames {:5.1f}%".format(
                100.0 * np.sum(filled[start:stop] == pm.H) / alive))


def overlap_counts(gt_spells, pred_spells):
    """For each GT possession, how many same-team predicted possessions overlap it."""
    out = []
    for gs, ge, gk in gt_spells:
        out.append(sum(1 for ps, pe, pk in pred_spells if pk == gk and ps < ge and gs < pe))
    return out


def public_section(codes, gt):
    """Scores against the public 5 minutes, broken down by kind of error."""
    result = pm.evaluate(gt, codes)
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(codes, runs)

    print("\n  PUBLIC WINDOW (5 min, {} GT possessions -- diagnosis, not ranking)".format(
        result["n_gt_spells"]))
    print("    spellF1 {:.4f}   macroF1 {:.4f}   accuracy {:.4f}   alive/dead F1 {:.4f}".format(
        result["spell_f1_mean"], result["macro_f1"], result["accuracy"],
        result["alive_dead_f1"]))
    print("    possessions  pred {} vs gt {}   mean {:.1f}s vs {:.1f}s".format(
        result["n_pred_spells"], result["n_gt_spells"],
        result["mean_spell_duration_s"], result["gt_mean_spell_duration_s"]))

    print("    per class    " + "   ".join(
        "{} P {:.2f} R {:.2f} F1 {:.2f}".format(name, v["precision"], v["recall"], v["f1"])
        for name, v in result["per_class"].items()))

    # Where the wrong frames went. Rows = truth, columns = prediction.
    cm = np.array(result["confusion"])
    print("    confusion (rows truth H/A/D, cols pred H/A/D/X)")
    for k in (pm.H, pm.A, pm.D):
        print("      {}  {}".format(NAMES[k], "  ".join("{:>5}".format(v) for v in cm[k, :4])))
    graded = cm[:3, :4].sum()
    team_swap = cm[pm.H, pm.A] + cm[pm.A, pm.H]
    alive_as_dead = cm[pm.H, pm.D] + cm[pm.A, pm.D]
    dead_as_alive = cm[pm.D, pm.H] + cm[pm.D, pm.A]
    abstain = cm[:3, pm.X].sum()
    print("    wrong frames  wrong team {:.1f}%   alive called dead {:.1f}%   "
          "dead called alive {:.1f}%   abstained {:.1f}%".format(
              *(100.0 * v / graded for v in (team_swap, alive_as_dead, dead_as_alive, abstain))))

    print("    spell match  " + "   ".join(
        "IoU{:.1f} {}/{} ({} spurious)".format(iou, v["n_matched"], result["n_gt_spells"],
                                               v["n_spurious"])
        for iou, v in sorted(result["spell_f1"].items())))

    gt_spells = pm.possession_spells(gt, runs)
    pred_spells = pm.possession_spells(filled, runs)
    pieces = overlap_counts(gt_spells, pred_spells)
    print("    GT possessions split into k predicted pieces: " + "  ".join(
        "k={}:{}".format(k, pieces.count(k)) for k in sorted(set(pieces))))

    gt_kinds = transition_kinds(gt, runs)
    pred_kinds = transition_kinds(filled, runs)
    keys = sorted(set(gt_kinds) | set(pred_kinds))
    print("    changes by kind (pred/gt)  " + "  ".join(
        "{} {}/{}".format(k, pred_kinds.get(k, 0), gt_kinds.get(k, 0)) for k in keys))

    for tol, t in sorted(result["transitions"].items()):
        latency = ("median {:+.0f}  mean|.| {:.1f}  frames".format(
            t["median_latency"], t["mean_abs_latency"]) if t["median_latency"] is not None
            else "n/a")
        print("    transitions +/-{:<3} P {:.2f} R {:.2f}  matched {} missed {} spurious {}   "
              "latency {}".format(tol, t["precision"], t["recall"], t["n_matched"],
                                  t["n_missed"], t["n_spurious"], latency))


BRIEF_HEADER = ("{:<28} {:>5} {:>7} {:>6} {:>6} {:>7}   {:>8} {:>8} {:>9}".format(
    "submission", "poss", "median", "direct", "D", "home", "pub spF1", "pub mF1", "pub poss"))


def brief_line(name, codes, gt):
    """One line: full-match plausibility first, then the public-window scores."""
    runs = [(start, stop) for _, start, stop in half_runs()]
    filled = pm.fill_abstentions(codes, runs)
    spells = [(b - a) / float(FPS) for a, b, k in pm.segments(filled, runs)
              if k in pm.POSSESSION_LABELS]
    kinds = transition_kinds(filled, runs)
    direct = kinds.get("H>A", 0) + kinds.get("A>H", 0)
    graded = np.concatenate([filled[a:b] for a, b in runs])
    alive = np.sum(graded != pm.D)
    result = pm.evaluate(gt, codes)
    return "{:<28} {:>5} {:>6.1f}s {:>5.0f}% {:>5.1f}% {:>6.1f}%   {:>8.4f} {:>8.4f} {:>5}/{:<3}".format(
        name[:28], len(spells), np.median(spells) if spells else 0.0,
        100.0 * direct / max(sum(kinds.values()), 1), 100.0 * np.mean(graded == pm.D),
        100.0 * np.sum(graded == pm.H) / max(alive, 1),
        result["spell_f1_mean"], result["macro_f1"], result["n_pred_spells"],
        result["n_gt_spells"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submissions", nargs="+", help="frame,label CSVs (globs allowed)")
    parser.add_argument("--gt", default=PUBLIC_GT, help="ground truth CSV (default: public)")
    parser.add_argument("--brief", action="store_true",
                        help="one line per file: possessions, median length, direct-change "
                             "share, D share, home share, public spellF1/macroF1/count")
    args = parser.parse_args()

    if args.brief:
        paths = sorted({p for pattern in args.submissions for p in glob.glob(pattern)})
        gt = pm.label_codes(pm.read_label_csv(args.gt))
        print(BRIEF_HEADER)
        for path in paths:
            table = pm.read_label_csv(path)
            reason = pm.validate_submission(table, GAMES[GAME]["n_video_frames"])
            name = os.path.splitext(os.path.basename(path))[0]
            print("{:<28} INVALID: {}".format(name, reason) if reason
                  else brief_line(name, pm.label_codes(table), gt))
        return

    paths = sorted({p for pattern in args.submissions for p in glob.glob(pattern)})
    if not paths:
        sys.exit("no files matched")
    gt = pm.label_codes(pm.read_label_csv(args.gt))

    for path in paths:
        print("=" * 100)
        print(os.path.basename(path))
        table = pm.read_label_csv(path)
        reason = pm.validate_submission(table, GAMES[GAME]["n_video_frames"])
        if reason:
            print("  INVALID: {}".format(reason))
            continue
        codes = pm.label_codes(table)
        full_match_section(codes)
        public_section(codes, gt)
    print("=" * 100)


if __name__ == '__main__':
    main()
