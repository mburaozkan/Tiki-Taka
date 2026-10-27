"""Backdating a change of team to an uncertain earlier touch (pipeline/fixes.backdate), on top of
the submission v29_tr_a (v29 + touch-run vote). Aggregate only.

    python tools/eval_backdate.py
    python tools/eval_backdate.py --ours "p_max=0.8 max_s=6 loose_m=2" --name v29_bd

DFL as tools/eval_touch_run.py (v29's held-out second stage, refine_b2): per setting mean
spellF1, delta vs v29_tr_a, folds won / lost, nested. --ours writes evaluation/results/<name>.csv.
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
import eval_touch_run as tr  # noqa: E402
from eval_mbr import v25_inputs  # noqa: E402
from eval_refine import stage1  # noqa: E402
from train_learned_dfl import match_features, test_seed, blocks  # noqa: E402
from pipeline import fixes  # noqa: E402

# Params
SUBMISSION = {"w": 2.0, "min_n": 2, "max_gap_s": 2.0, "hold_s": 4.0, "turnover": 1}
GRID = {"p_max": (0.7, 0.8, 0.9, 1.01), "max_s": (3.0, 6.0, 9.0), "loose_m": (0.0, 1.5, 2.5)}


def dfl():
    data = v25_inputs()
    with np.load(os.path.join(dd.CACHE, "refine_{}.npz".format(tr.TEAM_TAG))) as c:
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
        p_ref = np.mean([team["{}__{}".format(g, s)] for s in range(tr.TEAM_SEEDS)], axis=0)
        base = tr.decode(hand, p1, p_ref, f, blk, tr.run_vote(f, SUBMISSION))
        ref.append(tr.score(lab, base))
        for k, kw in enumerate(settings):
            res[k].append(tr.score(lab, fixes.backdate(base, f, **kw)))
        print("  DFL", g, flush=True)
    ref = np.array(ref)
    res = [np.array(r) for r in res]
    print("v29_tr_a reference {} | {:.4f}".format(" ".join("{:.3f}".format(x) for x in ref), ref.mean()))
    for k in sorted(range(len(settings)), key=lambda k: -res[k].mean()):
        s = res[k]
        print("{:<34} {} | {:.4f} | {:+.4f} | {}/{}".format(
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
    kw = {k: float(v) for k, v in (t.split("=") for t in text.split())}
    s = signals()
    with np.load(os.path.join(dd.DATA_DIR, "cache", "v25_ours_prob{}.npz".format(
            os.environ.get("MTAH_VARIANT", "")))) as c:
        prob = c["prob"]
    hand, f = s["hand"], s["f"]
    blk = [(a, b + 1) for _, a, b in half_video_blocks(GAME)]
    base = tr.decode(hand, stage1(prob, f), s["p_ref"], f, blk, tr.run_vote(f, SUBMISSION))
    codes = fixes.backdate(base, f, **kw)
    path = write_submission(os.path.join(RESULTS_DIR, name + ".csv"), codes)
    reason = pm.validate_submission(pm.read_label_csv(path), GAMES[GAME]["n_video_frames"])
    live = np.isin(hand, (0, 1))
    print("{} ({}) -> {} ({}); differs from v29_tr_a on {:.2f}% of live frames".format(
        name, text, os.path.relpath(path, ROOT), reason or "valid",
        100 * np.mean(codes[live] != base[live])))
    print("  " + brief_line(name, codes, pm.label_codes(pm.read_label_csv(PUBLIC_GT))))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ours", help='"p_max=.. max_s=.. loose_m=.."')
    parser.add_argument("--name", default="v29_bd")
    args = parser.parse_args()
    if args.ours:
        ours(args.ours, args.name)
    else:
        dfl()


if __name__ == "__main__":
    main()
