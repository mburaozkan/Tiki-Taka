"""Tune the second-stage team model (pipeline/refine.py) on DFL. Aggregate only.

    python tools/tune_refine.py --bin 5 --tag b5                     # the v28 configuration
    python tools/tune_refine.py --bin 1 --conv 2 --tag b1c2          # full 25 fps
    python tools/tune_refine.py --bin 1 --conv 2 --hidden 128 --crop-s 240 --tag ...

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
    parser.add_argument("--crop-s", type=float, default=120.0)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    refine.BIN, refine.HIDDEN, refine.LAYERS, refine.CONV = args.bin, args.hidden, args.layers, args.conv
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
            bundle = refine.train([e for o in dd.GAMES if o != g for e in ex[o]], steps=args.steps,
                                  seed=seed, log=lambda *_: None)
            probs[key] = refine.predict(bundle, ex[g][0][0], ex[g][1][0], len(data[g][0]))
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


if __name__ == '__main__':
    main()
