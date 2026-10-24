"""Describe the provided match data in aggregate numbers only -- safe to share outside the team.

Prints rates, counts and distributions. Never a frame number, a coordinate, a row, or anything
that could reconstruct the data -- the terms forbid making extracts available to a hosted LLM.

    python tools/data_report.py                  # tactical: ATD, homography, Smart Tagging
    python tools/data_report.py --broadcast      # also the broadcast ATD and homography

The first run parses the ATD into data/cache/ (a minute or two); later runs take seconds.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, half_video_blocks  # noqa: E402
from atd_cache import load_arrays  # noqa: E402

# Params
FPS = GAMES[GAME]["fps"]
LENGTH, WIDTH = GAMES[GAME]["pitch"]


def runs_of(mask):
    """Lengths of maximal True runs in a boolean array."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def dist(values, unit=""):
    """n, mean, median, p90, max of an array, as one line."""
    if not len(values):
        return "n=0"
    return "n={:<5} mean {:6.2f}{u}  median {:6.2f}{u}  p90 {:6.2f}{u}  max {:7.2f}{u}".format(
        len(values), np.mean(values), np.median(values), np.percentile(values, 90),
        np.max(values), u=unit)


def atd_block(players, ball, team, label):
    """Ball visibility, ball gaps, ball position/speed sanity, and tracked-object counts."""
    n = len(ball)
    seen = ~np.isnan(ball[:, 0])
    print("  {}  ({} frames)".format(label, n))
    print("    ball visible       {:5.1f}%".format(100.0 * seen.mean()))
    gaps = runs_of(~seen) / float(FPS)
    print("    ball gaps (s)      {}".format(dist(gaps, "s")))
    print("    ball gaps longer than 1s / 2s / 5s / 10s:  {} / {} / {} / {}".format(
        *(int(np.sum(gaps > t)) for t in (1, 2, 5, 10))))

    if seen.any():
        bx, by = ball[seen, 0], ball[seen, 1]
        outside = (bx < 0) | (bx > 1) | (by < 0) | (by > 1)
        print("    ball outside pitch {:5.1f}% of visible frames".format(100.0 * outside.mean()))
        print("    ball 3rd column    {}".format(dist(ball[seen, 2])))
        step = np.hypot(np.diff(ball[:, 0]) * LENGTH, np.diff(ball[:, 1]) * WIDTH) * FPS
        step = step[~np.isnan(step)]
        print("    ball speed (m/s, consecutive visible frames)  {}".format(dist(step)))
        print("    ball jumps > 40 m/s (likely tracking errors): {:.2f}% of steps".format(
            100.0 * np.mean(step > 40)))

    present = ~np.isnan(players[:, :, 0])
    for side, name in ((0, "home"), (1, "away")):
        count = present[:, team == side].sum(axis=1)
        print("    {} tracked per frame  mean {:5.2f}   <9: {:4.1f}%   >11: {:4.1f}%".format(
            name, count.mean(), 100.0 * np.mean(count < 9), 100.0 * np.mean(count > 11)))
    slot_life = [runs_of(present[:, s]) / float(FPS) for s in range(present.shape[1])]
    slot_life = np.concatenate([s for s in slot_life if len(s)]) if slot_life else np.array([])
    print("    tracklet lifetimes (s)  {}".format(dist(slot_life, "s")))


def tactical_atd():
    """Per-half ATD summary for the tactical recording, graded frames only."""
    atd = load_arrays("tactical")
    print("TACTICAL ATD   {} track slots (home {}, away {})".format(
        len(atd["team"]), int(np.sum(atd["team"] == 0)), int(np.sum(atd["team"] == 1))))
    for half, first, last in half_video_blocks(GAME):
        sl = slice(first, last + 1)
        atd_block(atd["players"][sl], atd["ball"][sl], atd["team"], "half {}".format(half))


def broadcast_atd():
    """Whole-file ATD summary for the broadcast recording (it has no half boundaries)."""
    atd = load_arrays("broadcast")
    print("\nBROADCAST ATD   {} track slots".format(len(atd["team"])))
    atd_block(atd["players"], atd["ball"], atd["team"], "whole file")


def homography(source):
    """Valid share, invalid-stretch lengths and reprojection error distribution."""
    from load_homography import load
    _, error, valid = load(source=source)
    print("\nHOMOGRAPHY {}   valid {:5.1f}%".format(source, 100.0 * valid.mean()))
    print("  invalid stretches (s)   {}".format(dist(runs_of(~valid) / float(FPS), "s")))
    print("  reprojection error      {}".format(dist(error[valid])))


def smart_tagging():
    """Counts by code and team, interval durations by code. Already-published style numbers."""
    from load_smart_tagging import load
    events = load()
    print("\nSMART TAGGING   {} instances".format(len(events)))
    codes = sorted({e["code"] for e in events})
    print("  {:<32} {:>5} {:>5} {:>5} {:>5}   duration (s)".format(
        "code", "n", "BAR", "ATH", "N/A"))
    for code in codes:
        group = [e for e in events if e["code"] == code]
        teams = [(e["tags"].get("Team") or "").lower() for e in group]
        durations = np.array([e["end_s"] - e["start_s"] for e in group])
        print("  {:<32} {:>5} {:>5} {:>5} {:>5}   median {:5.1f}  p90 {:5.1f}".format(
            code, len(group), sum("barcelona" in t for t in teams),
            sum("atletic" in t or "athletic" in t for t in teams),
            sum(t in ("n/a", "") for t in teams),
            np.median(durations), np.percentile(durations, 90)))
    # How much of each half the tagging covers at all -- a hint at how far it can be relied on.
    for half, first, last in half_video_blocks(GAME):
        covered = np.zeros(last - first + 1, dtype=bool)
        for e in events:
            a, b = max(e["start_frame"], first), min(e["end_frame"], last + 1)
            if b > a:
                covered[a - first:b - first] = True
        print("  half {} frames inside at least one tag: {:5.1f}%".format(
            half, 100.0 * covered.mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--broadcast", action="store_true", help="include the broadcast recording")
    args = parser.parse_args()

    tactical_atd()
    if args.broadcast:
        broadcast_atd()
    homography("tactical")
    if args.broadcast:
        homography("broadcast")
    smart_tagging()


if __name__ == '__main__':
    main()
