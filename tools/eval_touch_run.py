"""Runs of same-team touches as a team vote (pipeline/fixes.run_vote), on top of v29.

    python tools/eval_touch_run.py
    python tools/eval_touch_run.py --ours "w=2 min_n=2 max_gap_s=3 hold_s=2 turnover=1" --name v29_tr

The vote is added to v29's blend (first stage + 0.5 x second stage, in log-odds) before the
team Viterbi (tools/tune_refine.decode: switch 20, restart handover); D vs live is untouched.
  DFL (blur + our gaps, 7 matches): v29's held-out second-stage probabilities (refine_b2.npz),
      per setting mean spellF1, delta vs v29, folds won / lost, and nested.
  --ours: v29's second stage (bin 2, 5 seeds) on our match plus the vote; writes
      evaluation/results/<name>.csv and prints the brief report line.
Aggregate only.
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
from eval_refine import stage1  # noqa: E402
from train_learned_dfl import match_features, test_seed, blocks  # noqa: E402
from pipeline import learned, fixes  # noqa: E402

# Params
TEAM_TAG, TEAM_SEEDS, ALPHA, SWITCH = "b2", 5, 0.5, 20.0
GRID = {"w": (0.5, 1.0, 2.0, 3.0), "min_n": (2, 3), "max_gap_s": (2.0, 4.0),
        "hold_s": (1.0, 2.0, 4.0), "turnover": (1, 0)}


def decode(hand, p1, p_ref, f, blk, vote):
    q = np.clip(p_ref, 1e-4, 1 - 1e-4)
    p = fixes.add_vote(p1, ALPHA * np.log(q / (1 - q)) + vote, 1.0)
    return fixes.restart_handover(learned.decode_hybrid(hand, p, blk, SWITCH), f, max_s=3.0,
                                  slow_ms=2.0)


def run_vote(f, kw):
    kw = dict(kw)
    w = kw.pop("w")
    return w * fixes.run_vote(f, min_n=int(kw["min_n"]), max_gap_s=kw["max_gap_s"],
                              hold_s=kw["hold_s"], turnover=bool(kw["turnover"]))


def score(lab, codes):
    return pm.evaluate(lab, codes, bootstrap_n=0)["spell_f1_mean"]


def dfl():
    data = v25_inputs()
    with np.load(os.path.join(dd.CACHE, "refine_{}.npz".format(TEAM_TAG))) as c:
        team = {k: c[k] for k in c.files}
    settings = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    res = [[] for _ in settings]
    ref = []
    for g in dd.GAMES:
        d = dd.load(g)
        f = match_features(g, test_seed(g))
        blk = blocks(d["period"])
        lab, hand, prob = data[g]
        p1 = stage1(prob, f)
        p_ref = np.mean([team["{}__{}".format(g, s)] for s in range(TEAM_SEEDS)], axis=0)
        ref.append(score(lab, decode(hand, p1, p_ref, f, blk, 0.0)))
        for k, kw in enumerate(settings):
            res[k].append(score(lab, decode(hand, p1, p_ref, f, blk, run_vote(f, kw))))
        print("  DFL", g, flush=True)
    ref = np.array(ref)
    res = [np.array(r) for r in res]
    print("v29 reference {} | {:.3f}".format(" ".join("{:.3f}".format(x) for x in ref), ref.mean()))
    order = sorted(range(len(settings)), key=lambda k: -res[k].mean())
    for k in order[:15] + order[-3:]:
        s = res[k]
        print("{:<52} {} | {:.4f} | {:+.4f} | {}/{}".format(
            " ".join("{}={:g}".format(a, b) for a, b in settings[k].items()),
            " ".join("{:.3f}".format(x) for x in s), s.mean(), s.mean() - ref.mean(),
            int(np.sum(s > ref + 1e-9)), int(np.sum(s < ref - 1e-9))))
    nested = np.array([res[max(range(len(settings)), key=lambda k: np.delete(res[k], i).mean())][i]
                       for i in range(len(dd.GAMES))])
    print("nested {} | {:.4f} | {:+.4f} | {}/{}".format(
        " ".join("{:.3f}".format(x) for x in nested), nested.mean(), nested.mean() - ref.mean(),
        int(np.sum(nested > ref + 1e-9)), int(np.sum(nested < ref - 1e-9))))


def ours(text, name):
    from games import GAME, GAMES, half_video_blocks
    from pipeline.run import write_submission, RESULTS_DIR
    from report import brief_line, PUBLIC_GT
    from possession_view import signals
    import eval_refine  # noqa: F401  (path set up for stage1 inside signals)
    kw = {k: float(v) for k, v in (t.split("=") for t in text.split())}
    s = signals()
    with np.load(os.path.join(dd.DATA_DIR, "cache", "v25_ours_prob{}.npz".format(
            os.environ.get("MTAH_VARIANT", "")))) as c:
        prob = c["prob"]
    hand, f = s["hand"], s["f"]
    blk = [(a, b + 1) for _, a, b in half_video_blocks(GAME)]
    p1 = stage1(prob, f)
    v29 = decode(hand, p1, s["p_ref"], f, blk, 0.0)
    codes = decode(hand, p1, s["p_ref"], f, blk, run_vote(f, kw))
    path = write_submission(os.path.join(RESULTS_DIR, name + ".csv"), codes)
    reason = pm.validate_submission(pm.read_label_csv(path), GAMES[GAME]["n_video_frames"])
    live = np.isin(hand, (0, 1))
    print("{} ({}) -> {} ({}); team differs from v29 on {:.2f}% of live frames".format(
        name, text, os.path.relpath(path, ROOT), reason or "valid",
        100 * np.mean(codes[live] != v29[live])))
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    print("  " + brief_line(name, codes, gt))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ours", help='"w=.. min_n=.. max_gap_s=.. hold_s=.. turnover=0|1"')
    parser.add_argument("--name", default="v29_tr")
    args = parser.parse_args()
    if args.ours:
        ours(args.ours, args.name)
    else:
        dfl()


if __name__ == "__main__":
    main()
