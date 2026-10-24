"""Learned evidence (GBM, GRU, blend) trained on DFL as our match looks, one match held out.

    python tools/train_learned_dfl.py                  # 7 folds, GBM + GRU + blend
    python tools/train_learned_dfl.py --model gbm      # GBM only (faster)

Every DFL match is prepared like ours (dfl_data.prepare_ball: dead ball animated, blurred,
gapped), features from pipeline.features with the touch network trained WITHOUT that
match (touch_net_<match>.pt), so its touch probabilities are never in-sample. Labels: the
controlled-possession labels. The test version of each match uses the same gap seed as
tools/sweep.py, so the hand model's numbers there (v20: 0.567) are on the same data; training
uses TRAIN_VERSIONS other gap patterns.

Each held-out match is decoded by the Viterbi in log-probability units over COST_GRID; the grid is
small, but it is chosen on the same folds it is reported on (mildly optimistic). Prints spellF1 /
macroF1 per match and the mean for the hand model (v20), GBM, GRU and the 50/50 blend.
Features are cached in data/cache/dfl/feat_<match>_<seed>.npz, the held-out probabilities in
data/cache/dfl/learned_probs.npz. --hybrid then also scores learned.decode_hybrid (hand v20 for
D vs live, the learned model for the team) over HYBRID_SWITCH; --reuse skips training and reads
the saved probabilities.
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
from gap_model import degrade  # noqa: E402
import pipeline.features as features  # noqa: E402
from pipeline import learned, touch_net  # noqa: E402
from pipeline.decode import PARAMS, emissions, decode_block, X  # noqa: E402

# Params
TRAIN_VERSIONS = 2
COST_GRID = {"switch": (6.0, 10.0, 15.0, 20.0, 30.0), "to_dead": (6.0, 10.0, 15.0),
             "from_dead": (2.0, 4.0)}
MODEL_DIR = os.path.join(dd.DATA_DIR, "cache", "models")
PROBS = os.path.join(dd.CACHE, "learned_probs_anim.npz")
HYBRID_SWITCH = (2.0, 4.0, 6.0, 10.0, 15.0, 20.0)


def test_seed(game):
    return sum(map(ord, str(game)))     # the seed tools/sweep.py uses


def match_features(game, seed, rebuild=False):
    path = os.path.join(dd.CACHE, "feat_{}_{}_anim.npz".format(game, seed))
    if not rebuild and os.path.isfile(path):
        with np.load(path) as c:
            return {k: c[k] for k in c.files}
    d = dd.load(game)
    ball = dd.prepare_ball(d, seed)        # dead ball animated, blurred, our gaps
    model = touch_net.load(os.path.join(MODEL_DIR, "touch_net_{}.pt".format(game)))
    length, width = features.LENGTH, features.WIDTH
    features.LENGTH, features.WIDTH = dd.PITCH
    f = features.build_features({"players": d["players"], "ball": ball, "team": d["team"]},
                                events=[], touch_model=model)
    features.LENGTH, features.WIDTH = length, width
    np.savez(path, **f)
    return f


def blocks(period):
    out = []
    for per in (1, 2):
        idx = np.flatnonzero(period == per)
        if len(idx):
            out.append((idx[0], idx[-1] + 1))
    return out


def score(gt, pred):
    r = pm.evaluate(gt, pred, bootstrap_n=0)
    return r["spell_f1_mean"], r["macro_f1"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=("gbm", "gru", "both"), default="both")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--reuse", action="store_true", help="read saved held-out probabilities")
    args = parser.parse_args()

    print("building features ...", flush=True)
    data = {}
    for g in dd.GAMES:
        d = dd.load(g)
        test = match_features(g, test_seed(g), args.rebuild)
        train = [match_features(g, test_seed(g) + 1 + v, args.rebuild)
                 for v in range(TRAIN_VERSIONS)]
        data[g] = (test, train, d["controlled"], blocks(d["period"]))

    # Hand model v20 on the same test features.
    hand, hand_codes = {}, {}
    for g, (f, _, lab, blk) in data.items():
        sc = emissions(f, PARAMS)
        pred = np.full(len(lab), X, dtype=np.int8)
        for a, b in blk:
            pred[a:b] = decode_block(sc, f, PARAMS, a, b)
        hand[g] = score(lab, pred)
        hand_codes[g] = pred

    probs = {"gbm": {}, "gru": {}}
    if args.reuse:
        with np.load(PROBS) as c:
            for k in c.files:
                name, g = k.split("__")
                probs.setdefault(name, {})[g] = c[k]
    for held in ([] if args.reuse else dd.GAMES):
        print("held out {} ...".format(held), flush=True)
        train = [(f, data[g][2]) for g in dd.GAMES if g != held for f in data[g][1]]
        if args.model in ("gbm", "both"):
            model = learned.train_gbm([learned.training_rows(f, lab, step=3) for f, lab in train])
            probs["gbm"][held] = learned.predict_proba(model, data[held][0])
        if args.model in ("gru", "both"):
            from pipeline.sequence import train_gru, predict_gru
            trained = train_gru(train, log=lambda *_: None)
            probs["gru"][held] = predict_gru(trained, data[held][0])
    if not args.reuse:
        np.savez(PROBS, **{"{}__{}".format(name, g): p for name, pr in probs.items()
                           for g, p in pr.items()})
    if probs.get("gbm") and probs.get("gru"):
        probs["blend"] = {g: (probs["gbm"][g] + probs["gru"][g]) / 2 for g in dd.GAMES}

    print("\n{:<8} {}".format("", "  ".join("{:>13}".format(g) for g in dd.GAMES)) + "   mean")
    print("{:<8} {}   {:.3f} / {:.3f}".format(
        "hand v20", "  ".join("{:.3f} / {:.3f}".format(*hand[g]) for g in dd.GAMES),
        *np.mean([hand[g] for g in dd.GAMES], axis=0)))
    for name, pr in probs.items():
        if not pr:
            continue
        best = None
        for sw, td, fd in itertools.product(*COST_GRID.values()):
            costs = {"switch": sw, "to_dead": td, "from_dead": fd}
            res = {g: score(data[g][2], learned.decode_probs(pr[g], data[g][3], costs))
                   for g in dd.GAMES}
            mean = np.mean([res[g] for g in dd.GAMES], axis=0)
            if best is None or mean[0] > best[0][0]:
                best = (mean, costs, res)
        mean, costs, res = best
        print("{:<8} {}   {:.3f} / {:.3f}   costs {}".format(
            name, "  ".join("{:.3f} / {:.3f}".format(*res[g]) for g in dd.GAMES), *mean,
            "/".join("{:g}".format(v) for v in costs.values())), flush=True)
        if args.hybrid:
            best = None
            for sw in HYBRID_SWITCH:
                res = {g: score(data[g][2], learned.decode_hybrid(hand_codes[g], pr[g],
                                                                  data[g][3], sw))
                       for g in dd.GAMES}
                mean = np.mean([res[g] for g in dd.GAMES], axis=0)
                if best is None or mean[0] > best[0][0]:
                    best = (mean, sw, res)
            mean, sw, res = best
            print("{:<8} {}   {:.3f} / {:.3f}   switch {:g}".format(
                "hyb_" + name, "  ".join("{:.3f} / {:.3f}".format(*res[g]) for g in dd.GAMES),
                *mean, sw), flush=True)


if __name__ == '__main__':
    main()
