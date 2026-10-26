"""Tune the second-stage team model (pipeline/refine.py) on DFL. Aggregate only.

    python tools/tune_refine.py --bin 5 --tag b5                     # the v28 configuration
    python tools/tune_refine.py --bin 1 --conv 2 --tag b1c2          # full 25 fps
    python tools/tune_refine.py --bin 1 --conv 2 --hidden 128 --crop-s 240 --tag ...
    python tools/tune_refine.py --bin 2 --seeds 5 --stage1 clean3 --tag b2s5c3   # leak-free, 3 gap patterns

Leave one DFL match out as tools/eval_refine.py, but each fold is trained with --seeds seeds and
the held-out P(home) averaged over them (cached per seed in data/cache/dfl/refine_<tag>.npz, so a
rerun only trains what is missing). Then for every (alpha, switch): v27 log-odds + alpha x the
refined log-odds, decode_hybrid, v27's restart handover. Reports per setting the mean, delta vs
v27 and folds won / lost; "nested" picks (alpha, switch) on the other six matches for each held-out
match, for the seed ensemble and for seed 0 alone.
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
from eval_reaction import v27_codes, timing  # noqa: E402
from eval_refine import stage1  # noqa: E402
from train_learned_dfl import match_features, test_seed, blocks  # noqa: E402
from pipeline import learned, fixes, refine  # noqa: E402

# Params
ALPHAS = (0.25, 0.5, 0.75, 1.0)
SWITCHES = (10.0, 15.0, 20.0)


def decode(hand, p1, p_ref, f, blk, alpha, sw):
    q = np.clip(p_ref, 1e-4, 1 - 1e-4)
    codes = learned.decode_hybrid(hand, fixes.add_vote(p1, alpha * np.log(q / (1 - q)), 1.0), blk, sw)
    return fixes.restart_handover(codes, f, max_s=3.0, slow_ms=2.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bin", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--conv", type=int, default=0)
    parser.add_argument("--players", type=int, default=0,
                        help="per-player inputs: the K players per team nearest the ball")
    parser.add_argument("--crop-s", type=float, default=120.0)
    parser.add_argument("--stage1", choices=("lomo", "clean", "clean3"), default="lomo",
                        help="second-stage training inputs: lomo = eval_mbr.v25_inputs (leaks the "
                             "held-out match through the first stage); clean = first stage trained "
                             "without the held-out match too (tools/clean_stage1.py); clean3 = clean "
                             "on 3 gap patterns per match")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--ours", nargs=2, type=float, metavar=("ALPHA", "SWITCH"),
                        help="train --seeds models on all 7 (models/refine_<tag>_s<seed>.pt), write "
                             "evaluation/results/<tag>.csv on our match, and stop")
    args = parser.parse_args()
    refine.BIN, refine.HIDDEN, refine.LAYERS, refine.CONV = args.bin, args.hidden, args.layers, args.conv
    refine.PLAYERS = args.players
    refine.CROP_S = args.crop_s
    cache = os.path.join(dd.CACHE, "refine_{}.npz".format(args.tag))

    data = v25_inputs()
    ctx, ex = {}, {}
    for g in dd.GAMES:
        d = dd.load(g)
        f = match_features(g, test_seed(g))
        cols = idt.player_cols_cached(g, d)
        lab, hand, prob = data[g]
        p1 = stage1(prob, f)
        ctx[g] = (f, blocks(d["period"]), p1)
        ex[g] = [(refine.inputs(p1, hand, f, cols, flip),) + refine.targets(lab, flip)
                 for flip in (False, True)]
        if args.players:
            ball = dd.prepare_ball(d, test_seed(g))     # the ball the features were built on
            ex[g] = [e + (refine.player_inputs(d["players"], d["team"], ball, d["period"],
                                               dd.PITCH, flip=flip),)
                     for e, flip in zip(ex[g], (False, True))]

    if args.ours:
        ours(ex, args, *args.ours)
        return
    probs, clean_ex = {}, {}
    if os.path.isfile(cache):
        with np.load(cache) as c:
            probs = {k: c[k] for k in c.files}
    for g in dd.GAMES:
        for seed in range(args.seeds):
            key = "{}__{}".format(g, seed)
            if key in probs:
                continue
            t0 = time.time()
            if args.stage1 == "lomo":
                train_ex = [e for o in dd.GAMES if o != g for e in ex[o]]
            else:
                if g not in clean_ex:
                    clean_ex.clear()
                    clean_ex[g] = clean_examples([o for o in dd.GAMES if o != g], args, g)
                train_ex = clean_ex[g]
            bundle = refine.train(train_ex, steps=args.steps, seed=seed, log=lambda *_: None)
            pp = (ex[g][0][3], ex[g][1][3]) if args.players else (None, None)
            probs[key] = refine.predict(bundle, ex[g][0][0], ex[g][1][0], len(data[g][0]), *pp)
            np.savez(cache, **probs)
            print("  {} seed {} trained in {:.0f} s".format(g, seed, time.time() - t0), flush=True)

    ens = {g: np.mean([probs["{}__{}".format(g, s)] for s in range(args.seeds)], axis=0)
           for g in dd.GAMES}
    one = {g: probs["{}__0".format(g)] for g in dd.GAMES}
    ref = np.array([pm.evaluate(data[g][0], v27_codes(data[g][1], data[g][2], ctx[g][0], ctx[g][1]),
                                bootstrap_n=0)["spell_f1_mean"] for g in dd.GAMES])
    acc = np.mean([np.mean((ens[g][np.isin(data[g][0], (0, 1))] >= 0.5)
                           == (data[g][0][np.isin(data[g][0], (0, 1))] == 0)) for g in dd.GAMES])
    print("\nconfig {}  steps {}  crop {:g} s  seeds {}: per-frame team accuracy (ensemble) {:.1f}%"
          .format(refine.config(), args.steps, args.crop_s, args.seeds, 100 * acc))
    print("v27 reference {} | {:.3f}".format(" ".join("{:.3f}".format(x) for x in ref), ref.mean()))

    for name, pr in (("ensemble", ens), ("seed 0", one)):
        res = {}
        for alpha, sw in itertools.product(ALPHAS, SWITCHES):
            res[(alpha, sw)] = np.array([pm.evaluate(
                data[g][0], decode(data[g][1], ctx[g][2], pr[g], ctx[g][0], ctx[g][1], alpha, sw),
                bootstrap_n=0)["spell_f1_mean"] for g in dd.GAMES])
        if name == "ensemble":
            for (alpha, sw), s in res.items():
                print("alpha {:<5g} switch {:<4g} {} | {:.3f} | {:+.4f} | {}/{}".format(
                    alpha, sw, " ".join("{:.3f}".format(x) for x in s), s.mean(), s.mean() - ref.mean(),
                    int(np.sum(s > ref + 1e-9)), int(np.sum(s < ref - 1e-9))))
        nested = []
        for i in range(len(dd.GAMES)):
            best = max(res, key=lambda k: np.delete(res[k], i).mean())
            nested.append(res[best][i])
        nested = np.array(nested)
        best = max(res, key=lambda k: res[k].mean())
        preds = {g: decode(data[g][1], ctx[g][2], pr[g], ctx[g][0], ctx[g][1], *best) for g in dd.GAMES}
        n_true, offs, false, longf = 0, [], 0, 0
        for g in dd.GAMES:
            nt, o, fa, lf = timing(data[g][0], preds[g])
            n_true, offs, false, longf = n_true + nt, offs + o, false + fa, longf + lf
        offs = np.array(offs)
        print("{:<8} nested {} | {:.3f} | {:+.4f} | {}/{}   best alpha {:g} switch {:g}: {:.3f}   "
              "found {:.0f}% within 1 s {:.0f}% false/h {:.1f}".format(
                  name, " ".join("{:.3f}".format(x) for x in nested), nested.mean(),
                  nested.mean() - ref.mean(), int(np.sum(nested > ref + 1e-9)),
                  int(np.sum(nested < ref - 1e-9)), best[0], best[1], res[best].mean(),
                  100.0 * len(offs) / max(n_true, 1), 100.0 * np.mean(np.abs(offs) <= 1),
                  false / (longf / dd.FPS / 3600.0)), flush=True)


def clean_examples(games, args, held=None):
    """Second-stage training examples from tools/clean_stage1.py: the first stage for each match
    trained without it and without `held` (None: without it only), on 1 or 3 gap patterns."""
    import clean_stage1 as cs
    out = []
    for o in games:
        d = dd.load(o)
        cols = idt.player_cols_cached(o, d)
        excluded = (o,) if held is None else (o, held)
        for v in (cs.VERSIONS if args.stage1 == "clean3" else (0,)):
            lab, hand, prob = cs.inputs(o, excluded, v)
            f = match_features(o, test_seed(o) + v)
            p1 = stage1(prob, f)
            for flip in (False, True):
                e = (refine.inputs(p1, hand, f, cols, flip),) + refine.targets(lab, flip)
                if args.players:
                    ball = dd.prepare_ball(d, test_seed(o) + v)
                    e += (refine.player_inputs(d["players"], d["team"], ball, d["period"],
                                               dd.PITCH, flip=flip),)
                out.append(e)
    return out


def ours(ex, args, alpha, sw):
    """The seed ensemble trained on all 7 DFL matches, on top of v27 on our match."""
    from games import GAME, GAMES, half_video_blocks
    from atd_cache import load_arrays
    from pipeline.features import build_features, LENGTH, WIDTH
    from pipeline import touch_net
    from pipeline.run import write_submission, RESULTS_DIR
    from report import brief_line, PUBLIC_GT
    bundles = []
    for seed in range(args.seeds):
        path = os.path.join(ROOT, "models", "refine_{}_s{}.pt".format(args.tag, seed))
        if os.path.isfile(path):
            bundles.append(refine.load(path))
        else:
            train_ex = ([e for g in dd.GAMES for e in ex[g]] if args.stage1 == "lomo"
                        else clean_examples(dd.GAMES, args))
            b = refine.train(train_ex, steps=args.steps, seed=seed, log=lambda *_: None)
            refine.save(b, path)
            bundles.append(refine.load(path))      # predict exactly as a rebuild will
    with np.load(os.path.join(dd.DATA_DIR, "cache", "v25_ours_prob.npz")) as c:
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
    pp = ((refine.player_inputs(atd["players"], atd["team"], atd["ball"], groups, (LENGTH, WIDTH)),
           refine.player_inputs(atd["players"], atd["team"], atd["ball"], groups, (LENGTH, WIDTH),
                                flip=True)) if args.players else (None, None))
    p = np.mean([refine.predict(b, xs[0], xs[1], len(hand), *pp) for b in bundles], axis=0)
    codes = decode(hand, p1, p, f, blk, alpha, sw)
    v27 = decode(hand, p1, np.full(len(hand), 0.5), f, blk, 0.0, 10.0)
    out = write_submission(os.path.join(RESULTS_DIR, args.tag + ".csv"), codes)
    reason = pm.validate_submission(pm.read_label_csv(out), GAMES[GAME]["n_video_frames"])
    live = np.isin(hand, (0, 1)) & (groups > 0)
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    print("{} (alpha {:g}, switch {:g}, {} seeds) -> {} ({}); team differs from v27 on {:.1f}% of "
          "live frames".format(args.tag, alpha, sw, len(bundles), os.path.relpath(out, ROOT),
                               reason or "valid", 100 * np.mean(codes[live] != v27[live])))
    print("  " + brief_line(args.tag, codes, gt))


if __name__ == '__main__':
    main()
