"""Dead-ball head (dh1: d1n, add-only) on top of the submission v30. Aggregate only.

    python tools/eval_dead_v30.py                               # DFL, leave one match out
    python tools/eval_dead_v30.py --ours 0.5 0.5 25 --name v30_dh   # our match, public check
    python tools/eval_dead_v30.py --gate                          # DFL, + remove gate grid
    python tools/eval_dead_v30.py --ours 0.5 0.5 25 --gate 0.3 0 --name v30_dhg

v30 = v29's chain + touch-run vote + backdating (tools/eval_backdate.py). Here the hand path's
D vs live is first extended by the dead-ball head (tools/eval_dead_refine.py, held-out
probabilities from data/cache/dfl/dead_d1n.npz, deployed models models/dead_d1n_s0..2.pt), then
v30's chain runs unchanged. Per setting: mean spellF1, delta vs v30, folds won / lost; nested.
Remove gate (--gate): afterwards every D stretch whose mean head P(dead) is below T (and, if
MAX_S > 0, no longer than MAX_S seconds) is made live, so the team decoder fills it.
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
from eval_dead_refine import constrain, split_errors, BETAS, GAMMAS, COSTS  # noqa: E402
from train_learned_dfl import match_features, test_seed, blocks  # noqa: E402
from pipeline import refine, fixes  # noqa: E402
from pipeline.decode import H, A, D  # noqa: E402

# Params
DEAD_TAG, DEAD_SEEDS, MODE = "d1n", 3, "add"
VOTE = {"w": 2.0, "min_n": 2, "max_gap_s": 2.0, "hold_s": 4.0, "turnover": 1}   # v29_tr_a
BACKDATE = {"p_max": 0.7, "max_s": 9.0, "loose_m": 1.5}                          # v30
SETTING = (0.5, 0.5, 25.0)                                                       # dh1's head setting
GATE_T = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
GATE_MAX_S = (0.0, 10.0, 20.0)


def remove_gate(hand2, pdead, blk, t, max_s, fps):
    """Make live every D stretch of hand2 with mean P(dead) < t (and length <= max_s if max_s > 0)."""
    out = hand2.copy()
    for a0, b0 in blk:
        dead = (hand2[a0:b0] == D).astype(np.int8)
        e = np.diff(np.concatenate(([0], dead, [0])))
        for s0, e0 in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
            a, b = a0 + s0, a0 + e0
            if np.mean(pdead[a:b]) < t and (max_s <= 0 or (b - a) / fps <= max_s):
                out[a:b] = H                # any live code: decode_hybrid re-decides the team
    return out


def chain(hand, p1, p_ref, f, blk, vote):
    return fixes.backdate(tr.decode(hand, p1, p_ref, f, blk, vote), f, **BACKDATE)


def dfl(gate):
    data = v25_inputs()
    with np.load(os.path.join(dd.CACHE, "refine_{}.npz".format(tr.TEAM_TAG))) as c:
        team = {k: c[k] for k in c.files}
    with np.load(os.path.join(dd.CACHE, "dead_{}.npz".format(DEAD_TAG))) as c:
        pdead = {g: np.mean([c["{}__{}".format(g, s)] for s in range(DEAD_SEEDS)], axis=0)
                 for g in dd.GAMES}
    ctx, ref, base = {}, [], {}
    for g in dd.GAMES:
        d = dd.load(g)
        f = match_features(g, test_seed(g))
        blk = blocks(d["period"])
        lab, hand, prob = data[g]
        p1 = stage1(prob, f)
        p_ref = np.mean([team["{}__{}".format(g, s)] for s in range(tr.TEAM_SEEDS)], axis=0)
        ctx[g] = (f, blk, p1, p_ref, tr.run_vote(f, VOTE))
        base[g] = chain(hand, *ctx[g][2:4], f, blk, ctx[g][4])
        ref.append(tr.score(lab, base[g]))
        print("  DFL", g, flush=True)
    ref = np.array(ref)
    print("v30 reference {} | {:.4f}".format(" ".join("{:.3f}".format(x) for x in ref), ref.mean()))
    res, preds = {}, {}
    hand2 = {}
    for g in dd.GAMES:
        hand = data[g][1]
        hand2[g] = constrain(hand, refine.decode_dead(hand, pdead[g], ctx[g][1], *SETTING), MODE)
    if gate:
        grid = [(None, None)] + list(itertools.product(GATE_T, GATE_MAX_S))
    else:
        grid = list(itertools.product(BETAS, GAMMAS, COSTS))
    for key in grid:
        s, pr = [], {}
        for g in dd.GAMES:
            f, blk, p1, p_ref, vote = ctx[g]
            if gate:
                h = hand2[g] if key[0] is None else remove_gate(hand2[g], pdead[g], blk, *key, dd.FPS)
            else:
                hand = data[g][1]
                h = constrain(hand, refine.decode_dead(hand, pdead[g], blk, *key), MODE)
            pr[g] = chain(h, p1, p_ref, f, blk, vote)
            s.append(tr.score(data[g][0], pr[g]))
        s = np.array(s)
        res[key], preds[key] = s, pr
        label = ("v30_dh (no gate)" if key[0] is None else "gate T {:<4g} max_s {:<4g}".format(*key))             if gate else "beta {:<4g} gamma {:<4g} cost {:<4g}".format(*key)
        print("{:<28} {} | {:.4f} | {:+.4f} | {}/{}".format(
            label, " ".join("{:.3f}".format(x) for x in s), s.mean(), s.mean() - ref.mean(),
            int(np.sum(s > ref + 1e-9)), int(np.sum(s < ref - 1e-9))), flush=True)
    nested = np.array([res[max(res, key=lambda k: np.delete(res[k], i).mean())][i]
                       for i in range(len(dd.GAMES))])
    best = max(res, key=lambda k: res[k].mean())
    print("nested {} | {:.4f} | {:+.4f} | {}/{}   best {}: {:.4f}".format(
        " ".join("{:.3f}".format(x) for x in nested), nested.mean(), nested.mean() - ref.mean(),
        int(np.sum(nested > ref + 1e-9)), int(np.sum(nested < ref - 1e-9)), best, res[best].mean()))
    e0 = np.sum([split_errors(data[g][0], base[g]) for g in dd.GAMES], axis=0)
    for key in grid:
        e = np.sum([split_errors(data[g][0], preds[key][g]) for g in dd.GAMES], axis=0)
        print("{}: D>live {:.1f}% -> {:.1f}%, live>D {:.1f}% -> {:.1f}% (v30 -> this)".format(
            key, 100 * e0[0] / e0[1], 100 * e[0] / e[1], 100 * e0[2] / e0[3], 100 * e[2] / e[3]))


def ours(beta, gamma, cost, name, gate=None):
    from games import GAME, GAMES, half_video_blocks
    from pipeline.run import write_submission, RESULTS_DIR
    from report import brief_line, PUBLIC_GT
    from possession_view import signals
    s = signals()
    with np.load(os.path.join(dd.DATA_DIR, "cache", "v25_ours_prob{}.npz".format(
            os.environ.get("MTAH_VARIANT", "")))) as c:
        prob = c["prob"]
    hand, f, cols = s["hand"], s["f"], s["cols"]
    blk = [(a, b + 1) for _, a, b in half_video_blocks(GAME)]
    p1 = stage1(prob, f)
    bundles = [refine.load(os.path.join(ROOT, "models", "dead_{}_s{}.pt".format(DEAD_TAG, k)))
               for k in range(DEAD_SEEDS)]
    refine.BIN = bundles[0][0].cfg["BIN"]
    xd = (refine.dead_inputs(p1, hand, f, cols, False, False),
          refine.dead_inputs(p1, hand, f, cols, True, False))
    pdead = np.mean([refine.predict(b, xd[0], xd[1], len(hand), symmetric=True) for b in bundles],
                    axis=0)
    hand2 = constrain(hand, refine.decode_dead(hand, pdead, blk, beta, gamma, cost), MODE)
    if gate:
        n_d = np.sum(hand2 == D)
        hand2 = remove_gate(hand2, pdead, blk, gate[0], gate[1], dd.FPS)
        print("remove gate T {:g} max_s {:g}: {:.1f}% of the head path's D made live".format(
            gate[0], gate[1], 100 * (1 - np.sum(hand2 == D) / n_d)))
    codes = chain(hand2, p1, s["p_ref"], f, blk, s["vote"])
    base = s["codes"]
    path = write_submission(os.path.join(RESULTS_DIR, name + ".csv"), codes)
    reason = pm.validate_submission(pm.read_label_csv(path), GAMES[GAME]["n_video_frames"])
    inside = np.zeros(len(hand), bool)
    for a, b in blk:
        inside[a:b] = True

    def summary(c):
        c = c[inside]
        n = int(np.sum(np.diff(c) != 0) + 1)
        return "D {:.1f}%, H {:.1f}%, A {:.1f}%, label runs {}".format(
            100 * np.mean(c == D), 100 * np.mean(c == H), 100 * np.mean(c == A), n)
    print("{} (beta {:g}, gamma {:g}, cost {:g}) -> {} ({})".format(
        name, beta, gamma, cost, os.path.relpath(path, ROOT), reason or "valid"))
    print("  full match v30: " + summary(base))
    print("  full match {}: ".format(name) + summary(codes))
    print("  frames that differ from v30: {:.2f}% (live -> D {:.2f}%, D -> live {:.2f}%, team {:.2f}%)".format(
        100 * np.mean(codes[inside] != base[inside]),
        100 * np.mean((base[inside] != D) & (codes[inside] == D)),
        100 * np.mean((base[inside] == D) & (codes[inside] != D)),
        100 * np.mean((base[inside] != D) & (codes[inside] != D) & (codes[inside] != base[inside]))))
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    print("  " + brief_line("v30", base, gt))
    print("  " + brief_line(name, codes, gt))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ours", nargs=3, type=float, metavar=("BETA", "GAMMA", "COST"))
    parser.add_argument("--gate", nargs="*", type=float, metavar="T MAX_S",
                        help="DFL: gate grid; --ours: T MAX_S")
    parser.add_argument("--name", default="v30_dh")
    args = parser.parse_args()
    if args.ours:
        ours(*args.ours, args.name, args.gate)
    else:
        dfl(args.gate is not None)


if __name__ == "__main__":
    main()
