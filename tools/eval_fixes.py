"""Hand fixes (pipeline/fixes.py) on top of v25, on realistic DFL and on our match. Aggregate only.

    python tools/eval_fixes.py                          # DFL grid, every fix against v25
    python tools/eval_fixes.py --skip-dfl --ours "A max_s=3" "B w=1 vote_s=5" "A max_s=3 + B w=1"

DFL: the v25 pipeline held out (tools/eval_mbr.v25_inputs: hand path, blend + players-only vote),
reference decode_hybrid at switch 10. Each fix setting: spellF1 per match, mean, delta, folds won /
lost. For B also how often "the other team than the last toucher" is the true restart team.

--ours: each spec is one or more fixes joined by "+", each "<A|B|C|D> key=value ..."; writes
evaluation/results/v27_<n>.csv (n = position in the list) and prints a brief report line.
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
from eval_mbr import v25_inputs  # noqa: E402
from train_learned_dfl import match_features, test_seed, blocks  # noqa: E402
from pipeline import learned, fixes  # noqa: E402
from pipeline.decode import H, A, D  # noqa: E402

# Params
SWITCH = 10.0
GRID = {
    "A": {"max_s": (2.0, 3.0, 4.0, 6.0), "slow_ms": (1.5, 2.0)},
    "B": {"w": (0.5, 1.0, 2.0), "vote_s": (3.0, 5.0)},
    "C": {"w": (0.5, 1.0, 2.0)},
    "D": {"w": (0.5, 1.0, 2.0), "vote_s": (1.0, 2.0)},
}


def apply(spec, hand, prob, f, ball, pitch, blk):
    """Codes after the fixes in spec: [(letter, {param: value})]."""
    vote = np.zeros(len(hand))
    handover = None
    for letter, kw in spec:
        kw = dict(kw)
        if letter == "A":
            handover = kw
            continue
        w = kw.pop("w", 1.0)
        if letter == "B":
            vote += w * fixes.out_vote(hand, f, **kw)
        elif letter == "C":
            vote += w * fixes.landing_vote(ball, np.asarray(f["d_home"], np.float64),
                                           np.asarray(f["d_away"], np.float64), pitch, **kw)
        elif letter == "D":
            vote += w * fixes.margin_vote(f, **kw)
    p = fixes.add_vote(prob, vote, 1.0) if np.any(vote) else prob
    codes = learned.decode_hybrid(hand, p, blk, SWITCH)
    if handover is not None:
        codes = fixes.restart_handover(codes, f, **handover)
    return codes


def parse(text):
    spec = []
    for part in text.split("+"):
        tokens = part.split()
        spec.append((tokens[0], {k: float(v) for k, v in (t.split("=") for t in tokens[1:])}))
    return spec


def dfl():
    data = v25_inputs()
    ctx = {}
    for g in dd.GAMES:
        d = dd.load(g)
        f = match_features(g, test_seed(g))
        ball = dd.prepare_ball(d, test_seed(g))
        ctx[g] = (f, ball, blocks(d["period"]))
    ref = [pm.evaluate(data[g][0], learned.decode_hybrid(data[g][1], data[g][2], ctx[g][2], SWITCH),
                       bootstrap_n=0)["spell_f1_mean"] for g in dd.GAMES]
    print("reference v25: {} | {:.3f}".format(" ".join("{:.3f}".format(x) for x in ref),
                                             np.mean(ref)), flush=True)

    # B's premise: the restart after the ball went out goes to the other team than the last toucher.
    hits = total = 0
    for g in dd.GAMES:
        lab, hand, _ = data[g]
        vote = fixes.out_vote(hand, ctx[g][0])
        starts = np.flatnonzero((vote[1:] != 0) & (vote[:-1] == 0)) + 1
        for s in starts:
            seg = lab[s:s + 3 * dd.FPS]
            live = seg[np.isin(seg, (H, A))]
            if len(live):
                total += 1
                hits += int((live[0] == H) == (vote[s] > 0))
    print("B premise: {} restarts after ball out, other team than the last toucher = true first "
          "team {:.1f}%".format(total, 100.0 * hits / max(total, 1)), flush=True)

    for letter, grid in GRID.items():
        for values in itertools.product(*grid.values()):
            kw = dict(zip(grid.keys(), values))
            s = []
            for g in dd.GAMES:
                lab, hand, prob = data[g]
                f, ball, blk = ctx[g]
                s.append(pm.evaluate(lab, apply([(letter, kw)], hand, prob, f, ball, dd.PITCH, blk),
                                     bootstrap_n=0)["spell_f1_mean"])
            won = sum(x > y + 1e-9 for x, y in zip(s, ref))
            lost = sum(x < y - 1e-9 for x, y in zip(s, ref))
            print("{} {:<24} {} | {:.3f} | {:+.4f} | {}/{}".format(
                letter, " ".join("{}={:g}".format(k, v) for k, v in kw.items()),
                " ".join("{:.3f}".format(x) for x in s), np.mean(s), np.mean(s) - np.mean(ref),
                won, lost), flush=True)


def ours(specs):
    from games import GAME, GAMES, half_video_blocks
    from atd_cache import load_arrays
    from pipeline.features import build_features, LENGTH, WIDTH
    from pipeline import touch_net
    from pipeline.run import write_submission, RESULTS_DIR
    from report import brief_line, PUBLIC_GT
    with np.load(os.path.join(dd.DATA_DIR, "cache", "v25_ours_prob.npz")) as c:
        hand, prob = c["hand"], c["prob"]
    f = build_features(events=[], touch_model=touch_net.load(os.path.join(ROOT, touch_net.MODEL_PATH)))
    ball = load_arrays()["ball"]
    blk = [(a, b + 1) for _, a, b in half_video_blocks(GAME)]
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    for i, text in enumerate(specs, start=1):
        codes = apply(parse(text), hand, prob, f, ball, (LENGTH, WIDTH), blk)
        path = write_submission(os.path.join(RESULTS_DIR, "v27_{}.csv".format(i)), codes)
        reason = pm.validate_submission(pm.read_label_csv(path), GAMES[GAME]["n_video_frames"])
        print("{:<40} {}".format(text, reason or "valid"))
        print("  " + brief_line("v27_{}".format(i), codes, gt))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-dfl", action="store_true")
    parser.add_argument("--ours", nargs="*", default=None)
    args = parser.parse_args()
    if not args.skip_dfl:
        dfl()
    if args.ours:
        ours(args.ours)


if __name__ == '__main__':
    main()
