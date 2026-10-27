"""Second-stage dead-ball head on DFL, on top of v29. Aggregate only.

    python tools/eval_dead_refine.py --tag d1                  # with the first stage's P(dead)
    python tools/eval_dead_refine.py --tag d1n --no-pd         # without it (transfer check)
    python tools/eval_dead_refine.py --tag d1 --ours BETA GAMMA COST   # train on all 7, our match

Leave one DFL match out: the refiner network (v29's configuration, 12.5 fps) trained on the other
six to say whether the ball is dead (refine.dead_inputs / dead_targets), --seeds seeds averaged
(cached in data/cache/dfl/dead_<tag>.npz). Then refine.decode_dead re-decides D vs live from its
log-odds (beta), agreement with the hand path (gamma) and a cost per change, and the team is
decoded as v29 (its second-stage probabilities from refine_b2.npz, alpha 0.5, switch 20, restart
handover). Per setting: mean spellF1, delta vs v29, folds won / lost; nested picks the setting on
the other six. Also D>live and live>D for the best setting and per-bin accuracy of the head.
"""

import argparse
import itertools
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
import dfl_data as dd  # noqa: E402
import in_domain_team as idt  # noqa: E402
from eval_mbr import v25_inputs  # noqa: E402
from eval_refine import stage1  # noqa: E402
from tune_refine import decode  # noqa: E402
from train_learned_dfl import match_features, test_seed, blocks  # noqa: E402
from pipeline import refine  # noqa: E402
from pipeline.decode import H, A, D, X  # noqa: E402

# Params
TEAM_TAG, TEAM_SEEDS, ALPHA, SWITCH = "b2", 5, 0.5, 20.0
BETAS = (0.5, 1.0)
GAMMAS = (0.0, 0.5, 1.0, 2.0)
COSTS = (10.0, 25.0, 50.0, 100.0)
NEAR_S = 3.0                # --mode near: changes only within this many seconds of a hand D edge


def constrain(hand, hand2, mode):
    """free: the head's split as is; add: the hand path's D kept, the head may only add D;
    near: the head's split only within NEAR_S of the hand path's D edges (boundary moves)."""
    if mode == "free":
        return hand2
    if mode == "add":
        return np.where(hand == D, D, hand2).astype(np.int8)
    dead = (hand == D).astype(np.int8)
    edges = np.flatnonzero(np.diff(dead)) + 1
    near = np.zeros(len(hand), bool)
    k = int(NEAR_S * dd.FPS)
    for e in edges:
        near[max(0, e - k):e + k] = True
    return np.where(near & (hand != X), hand2, hand).astype(np.int8)


def split_errors(lab, pred):
    g = lab != X
    td, tl = g & (lab == D), g & np.isin(lab, (H, A))
    return np.sum(td & np.isin(pred, (H, A))), td.sum(), np.sum(tl & (pred == D)), tl.sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--no-pd", action="store_true", help="leave out the first stage's P(dead)")
    parser.add_argument("--mode", choices=("free", "add", "near"), default="free")
    parser.add_argument("--ours", nargs=3, type=float, metavar=("BETA", "GAMMA", "COST"))
    args = parser.parse_args()
    refine.BIN = 2
    use_pd = not args.no_pd
    data = v25_inputs()
    with np.load(os.path.join(dd.CACHE, "refine_{}.npz".format(TEAM_TAG))) as c:
        team = {k: c[k] for k in c.files}
    ctx, ex = {}, {}
    for g in dd.GAMES:
        d = dd.load(g)
        f = match_features(g, test_seed(g))
        cols = idt.player_cols_cached(g, d)
        lab, hand, prob = data[g]
        p1 = stage1(prob, f)
        p_ref = np.mean([team["{}__{}".format(g, s)] for s in range(TEAM_SEEDS)], axis=0)
        ctx[g] = (f, blocks(d["period"]), p1, p_ref)
        ex[g] = [(refine.dead_inputs(p1, hand, f, cols, flip, use_pd),) + refine.dead_targets(lab, flip)
                 for flip in (False, True)]
    if args.ours:
        ours(ex, args, use_pd, *args.ours)
        return

    cache = os.path.join(dd.CACHE, "dead_{}.npz".format(args.tag))
    probs = {}
    if os.path.isfile(cache):
        with np.load(cache) as c:
            probs = {k: c[k] for k in c.files}
    for g in dd.GAMES:
        for seed in range(args.seeds):
            key = "{}__{}".format(g, seed)
            if key in probs:
                continue
            t0 = time.time()
            bundle = refine.train([e for o in dd.GAMES if o != g for e in ex[o]], seed=seed,
                                  log=lambda *_: None)
            probs[key] = refine.predict(bundle, ex[g][0][0], ex[g][1][0], len(data[g][0]),
                                        symmetric=True)
            np.savez(cache, **probs)
            print("  {} seed {} trained in {:.0f} s".format(g, seed, time.time() - t0), flush=True)
    pdead = {g: np.mean([probs["{}__{}".format(g, s)] for s in range(args.seeds)], axis=0)
             for g in dd.GAMES}

    # per-bin accuracy on pure bins: the head, the hand path, the first stage's own D call
    acc = {"head": [], "hand": [], "stage 1": []}
    for g in dd.GAMES:
        lab, hand, prob = data[g]
        y, v = refine.dead_targets(lab)
        n = len(y) * refine.BIN

        def binned(x):
            return x[:n].reshape(-1, refine.BIN).mean(axis=1)
        acc["head"].append(np.mean((binned(pdead[g]) >= 0.5)[v] == (y[v] == 1)))
        acc["hand"].append(np.mean((binned((hand == D).astype(float)) >= 0.5)[v] == (y[v] == 1)))
        acc["stage 1"].append(np.mean((binned(prob[:, D]) >= 0.5)[v] == (y[v] == 1)))
    print("\nper-bin dead vs live accuracy: " + ", ".join(
        "{} {:.1f}%".format(k, 100 * np.mean(v)) for k, v in acc.items()))

    ref = np.array([pm.evaluate(data[g][0], decode(data[g][1], ctx[g][2], ctx[g][3], ctx[g][0], ctx[g][1],
                                                   ALPHA, SWITCH), bootstrap_n=0)["spell_f1_mean"]
                    for g in dd.GAMES])
    print("v29 reference {} | {:.3f}".format(" ".join("{:.3f}".format(x) for x in ref), ref.mean()))
    res, preds = {}, {}
    for beta, gamma, cost in itertools.product(BETAS, GAMMAS, COSTS):
        s, pr = [], {}
        for g in dd.GAMES:
            f, blk, p1, p_ref = ctx[g]
            hand2 = constrain(data[g][1], refine.decode_dead(data[g][1], pdead[g], blk, beta, gamma,
                                                             cost), args.mode)
            pr[g] = decode(hand2, p1, p_ref, f, blk, ALPHA, SWITCH)
            s.append(pm.evaluate(data[g][0], pr[g], bootstrap_n=0)["spell_f1_mean"])
        s = np.array(s)
        res[(beta, gamma, cost)], preds[(beta, gamma, cost)] = s, pr
        print("beta {:<4g} gamma {:<4g} cost {:<4g} {} | {:.3f} | {:+.4f} | {}/{}".format(
            beta, gamma, cost, " ".join("{:.3f}".format(x) for x in s), s.mean(), s.mean() - ref.mean(),
            int(np.sum(s > ref + 1e-9)), int(np.sum(s < ref - 1e-9))), flush=True)
    nested = np.array([res[max(res, key=lambda k: np.delete(res[k], i).mean())][i]
                       for i in range(len(dd.GAMES))])
    best = max(res, key=lambda k: res[k].mean())
    e = np.sum([split_errors(data[g][0], preds[best][g]) for g in dd.GAMES], axis=0)
    e0 = np.sum([split_errors(data[g][0], decode(data[g][1], ctx[g][2], ctx[g][3], ctx[g][0], ctx[g][1],
                                                 ALPHA, SWITCH)) for g in dd.GAMES], axis=0)
    print("nested {} | {:.3f} | {:+.4f} | {}/{}   best beta {:g} gamma {:g} cost {:g}: {:.3f}".format(
        " ".join("{:.3f}".format(x) for x in nested), nested.mean(), nested.mean() - ref.mean(),
        int(np.sum(nested > ref + 1e-9)), int(np.sum(nested < ref - 1e-9)), *best, res[best].mean()))
    print("D>live {:.1f}% -> {:.1f}%, live>D {:.1f}% -> {:.1f}% (v29 -> best)".format(
        100 * e0[0] / e0[1], 100 * e[0] / e[1], 100 * e0[2] / e0[3], 100 * e[2] / e[3]))


def ours(ex, args, use_pd, beta, gamma, cost):
    """Dead heads trained on all 7 DFL matches on top of v29 on our match."""
    from games import GAME, GAMES, half_video_blocks
    from atd_cache import load_arrays
    from pipeline.features import build_features, LENGTH, WIDTH
    from pipeline import touch_net
    from pipeline.run import write_submission, RESULTS_DIR
    from report import brief_line, PUBLIC_GT
    bundles = []
    for seed in range(args.seeds):
        path = os.path.join(ROOT, "models", "dead_{}_s{}.pt".format(args.tag, seed))
        if not os.path.isfile(path):
            refine.save(refine.train([e for g in dd.GAMES for e in ex[g]], seed=seed,
                                     log=lambda *_: None), path)
        bundles.append(refine.load(path))
    team_bundles = [refine.load(os.path.join(ROOT, "models", "refine_v29_s{}.pt".format(s)))
                    for s in range(TEAM_SEEDS)]
    with np.load(os.path.join(dd.DATA_DIR, "cache", "v25_ours_prob{}.npz".format(os.environ.get("MTAH_VARIANT", "")))) as c:
        hand, prob = c["hand"], c["prob"]
    f = build_features(events=[], touch_model=touch_net.load(os.path.join(ROOT, touch_net.MODEL_PATH)))
    blk = [(a, b + 1) for _, a, b in half_video_blocks(GAME)]
    groups = np.zeros(len(hand), dtype=np.int8)
    for i, (a, b) in enumerate(blk, start=1):
        groups[a:b] = i
    atd = load_arrays()
    cols = idt.player_columns(atd["players"], atd["team"], groups, (LENGTH, WIDTH))
    p1 = stage1(prob, f)
    xs = (refine.inputs(p1, hand, f, cols), refine.inputs(p1, hand, f, cols, True))
    p_ref = np.mean([refine.predict(b, xs[0], xs[1], len(hand)) for b in team_bundles], axis=0)
    xd = (refine.dead_inputs(p1, hand, f, cols, False, use_pd),
          refine.dead_inputs(p1, hand, f, cols, True, use_pd))
    pdead = np.mean([refine.predict(b, xd[0], xd[1], len(hand), symmetric=True) for b in bundles],
                    axis=0)
    hand2 = constrain(hand, refine.decode_dead(hand, pdead, blk, beta, gamma, cost), args.mode)
    codes = decode(hand2, p1, p_ref, f, blk, ALPHA, SWITCH)
    name = "v30_{}_{}".format(args.tag, args.mode)
    out = write_submission(os.path.join(RESULTS_DIR, name + ".csv"), codes)
    reason = pm.validate_submission(pm.read_label_csv(out), GAMES[GAME]["n_video_frames"])
    inside = groups > 0
    print("{} (beta {:g}, gamma {:g}, cost {:g}) -> {} ({}); D vs live differs from the hand path "
          "on {:.1f}% of frames".format(name, beta, gamma, cost, os.path.relpath(out, ROOT),
                                        reason or "valid",
                                        100 * np.mean((hand2[inside] == D) != (hand[inside] == D))))
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    print("  " + brief_line(name, codes, gt))


if __name__ == '__main__':
    main()
