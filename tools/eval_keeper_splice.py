"""Keeper fix only where it applies: v30's inputs everywhere, the keeper-fixed tracking's inputs
(MTAH_VARIANT=_kd, pipeline/track_fix own_flip_deepest) only on frames where the keeper rule
fires, widened by margin_s; then v30's chain once (tools/eval_turnover_switch.v30, no discount).
Our match only (DFL has clean team ids). Aggregate only.

    python tools/eval_keeper_splice.py                       # grid, public scores
    python tools/eval_keeper_splice.py --write "margin_s=2 goal=both" --name v30_ks

Needs data/cache/v25_ours_prob_kd.npz (in_domain_team.py --ours ... under MTAH_VARIANT=_kd).
goal: "both", "ath" (Athletic's goal: Barcelona slots given to Athletic) or "bar".
Spliced: the hand path, the first-stage probabilities, the second stage, and every feature.
"""

import argparse
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
import dfl_data as dd  # noqa: E402
from eval_turnover_switch import v30  # noqa: E402

# Params
VARIANT = "_kd"         # --variant _ke: the rule with evidence (doubled / ghost keeper)
GRID = {"margin_s": (0.0, 1.0, 2.0, 4.0, 8.0), "goal": ("both", "ath", "bar")}


def inputs(variant):
    from possession_view import signals
    os.environ["MTAH_VARIANT"] = variant
    try:
        s = signals()
        with np.load(os.path.join(dd.DATA_DIR, "cache", "v25_ours_prob{}.npz".format(variant))) as c:
            prob = c["prob"]
    finally:
        os.environ["MTAH_VARIANT"] = ""
    return s, prob


def keeper_frames(pitch, blk, evidence=False):
    """{goal: (n,) bool} frames where the _kd rule moves a slot, by the goal it happens at."""
    from atd_cache import load_arrays
    from pipeline.track_fix import keeper_status
    raw = load_arrays(raw=True)
    team = np.asarray(raw["team"])
    _, own, cases = keeper_status(raw["players"], team, blk, pitch, own_keeper=True,
                                  deepest_only=True, evidence=evidence)
    # a Barcelona (home, 0) slot flipped: Athletic's goal
    ath = cases[:, team == 0].any(axis=1)
    bar = cases[:, team == 1].any(axis=1)
    return {"ath": ath, "bar": bar, "both": ath | bar}


def splice(mask, s0, prob0, s1, prob1):
    f = {k: (np.where(mask, s1["f"][k], v) if np.ndim(v) == 1 and len(v) == len(mask) else v)
         for k, v in s0["f"].items()}
    return (np.where(mask, s1["hand"], s0["hand"]), np.where(mask[:, None], prob1, prob0),
            np.where(mask, s1["p_ref"], s0["p_ref"]), f)


def widen(mask, k):
    if k <= 0:
        return mask
    return np.convolve(mask.astype(np.float64), np.ones(2 * k + 1), mode="same") > 0


def main():
    from games import GAME, GAMES, half_video_blocks
    from report import brief_line, PUBLIC_GT
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", help='"margin_s=.. goal=.."')
    parser.add_argument("--name", default="v30_ks")
    parser.add_argument("--variant", default=VARIANT, choices=("_kd", "_ke"))
    args = parser.parse_args()
    blk = [(a, b + 1) for _, a, b in half_video_blocks(GAME)]
    s0, prob0 = inputs("")
    s1, prob1 = inputs(args.variant)
    frames = keeper_frames(s0["pitch"], blk, evidence=args.variant == "_ke")
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    base = v30(s0["hand"], prob0, s0["p_ref"], s0["f"], blk)
    live = np.isin(s0["hand"], (0, 1))
    print("keeper rule fires on {:.1f} s (Athletic's goal {:.1f} s, Barcelona's {:.1f} s)".format(
        frames["both"].sum() / dd.FPS, frames["ath"].sum() / dd.FPS, frames["bar"].sum() / dd.FPS))
    print("  " + brief_line("v30", base, gt))
    if args.write:
        from pipeline.run import write_submission, RESULTS_DIR
        kw = dict(t.split("=") for t in args.write.split())
        mask = widen(frames[kw["goal"]], int(float(kw["margin_s"]) * dd.FPS))
        codes = v30(*splice(mask, s0, prob0, s1, prob1), blk)
        path = write_submission(os.path.join(RESULTS_DIR, args.name + ".csv"), codes)
        reason = pm.validate_submission(pm.read_label_csv(path), GAMES[GAME]["n_video_frames"])
        print("{} ({}) -> {} ({}); differs from v30 on {:.2f}% of live frames".format(
            args.name, args.write, os.path.relpath(path, ROOT), reason or "valid",
            100 * np.mean(codes[live] != base[live])))
        print("  " + brief_line(args.name, codes, gt))
        return
    for margin, goal in itertools.product(*GRID.values()):
        mask = widen(frames[goal], int(margin * dd.FPS))
        hand, prob, p_ref, f = splice(mask, s0, prob0, s1, prob1)
        codes = v30(hand, prob, p_ref, f, blk)
        print("  " + brief_line("m{:g} {}".format(margin, goal), codes, gt) +
              "  differs {:.2f}%".format(100 * np.mean(codes[live] != base[live])), flush=True)


if __name__ == "__main__":
    main()
