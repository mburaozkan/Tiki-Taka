"""Rival touches (pipeline/fixes.rival_touches) under the submission v30 (v29 + touch-run vote +
backdating). Aggregate only.

    python tools/eval_rival_touch.py
    python tools/eval_rival_touch.py --ours "min_p=0.3 within_s=0.5" --name v30_rt

DFL as tools/eval_backdate.py (v29's held-out second stage, refine_b2, unchanged): the extra
touches enter the touch-margin vote, the touch-run vote, backdating and the restart handover; per
setting mean spellF1, delta vs v30, folds won / lost, nested. --ours writes
evaluation/results/<name>.csv (the second stage on the original features, as on DFL).
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
from eval_backdate import SUBMISSION  # noqa: E402
from eval_mbr import v25_inputs  # noqa: E402
from eval_refine import stage1  # noqa: E402
from train_learned_dfl import match_features, test_seed, blocks  # noqa: E402
from pipeline import fixes  # noqa: E402

# Params
BACKDATE = {"p_max": 0.7, "max_s": 9.0, "loose_m": 1.5}
GRID = {"min_p": (0.2, 0.3, 0.4, 0.5), "within_s": (0.3, 0.5, 1.0)}


def v30(hand, prob, p_ref, f, blk):
    base = tr.decode(hand, stage1(prob, f), p_ref, f, blk, tr.run_vote(f, SUBMISSION))
    return fixes.backdate(base, f, **BACKDATE)


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
        p_ref = np.mean([team["{}__{}".format(g, s)] for s in range(tr.TEAM_SEEDS)], axis=0)
        ref.append(tr.score(lab, v30(hand, prob, p_ref, f, blk)))
        for k, kw in enumerate(settings):
            res[k].append(tr.score(lab, v30(hand, prob, p_ref, fixes.rival_touches(f, **kw), blk)))
        print("  DFL", g, flush=True)
    ref = np.array(ref)
    res = [np.array(r) for r in res]
    print("v30 reference {} | {:.4f}".format(" ".join("{:.3f}".format(x) for x in ref), ref.mean()))
    for k in sorted(range(len(settings)), key=lambda k: -res[k].mean()):
        s = res[k]
        print("{:<24} {} | {:.4f} | {:+.4f} | {}/{}".format(
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
    base = v30(hand, prob, s["p_ref"], f, blk)
    f2 = fixes.rival_touches(f, **kw)
    codes = v30(hand, prob, s["p_ref"], f2, blk)
    path = write_submission(os.path.join(RESULTS_DIR, name + ".csv"), codes)
    reason = pm.validate_submission(pm.read_label_csv(path), GAMES[GAME]["n_video_frames"])
    live = np.isin(hand, (0, 1))
    print("{} ({}) -> {} ({}); differs from v30 on {:.2f}% of live frames; touches {} -> {}".format(
        name, text, os.path.relpath(path, ROOT), reason or "valid",
        100 * np.mean(codes[live] != base[live]), int(np.sum(f["touch"] != 0)),
        int(np.sum(f2["touch"] != 0))))
    print("  " + brief_line(name, codes, pm.label_codes(pm.read_label_csv(PUBLIC_GT))))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ours", help='"min_p=.. within_s=.."')
    parser.add_argument("--name", default="v30_rt")
    args = parser.parse_args()
    if args.ours:
        ours(args.ours, args.name)
    else:
        dfl()


if __name__ == "__main__":
    main()
