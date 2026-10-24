"""Semi-Markov decoding vs Viterbi on DFL (prepared like ours), one match held out. Aggregate only.

    python tools/eval_hsmm.py                  # hand scores (v20) and the learned blend

For every held-out DFL match, the length priors come from the other six matches' controlled
labels only. The per-frame scores are the hand model's (pipeline.decode.emissions, as v20) and the
learned blend's log-probabilities (data/cache/dfl/learned_probs_anim.npz, written by
tools/train_learned_dfl.py). Each is decoded by Viterbi (as now) and by pipeline.semimarkov over a
small grid of prior weights and cost scales; reported: spellF1 / macroF1 mean, and recall of true
possessions under 4 s. The grid is chosen on the same folds (mildly optimistic, for both).
"""

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
from train_learned_dfl import match_features, test_seed, blocks, PROBS  # noqa: E402
from sweep import recall_by_length  # noqa: E402
from pipeline import learned, semimarkov  # noqa: E402
from pipeline.decode import PARAMS, emissions, transition_cost, decode_block, X  # noqa: E402

# Params
HAND_WEIGHTS = (1.0, 3.0, 10.0)
HAND_COST_SCALES = (0.0, 0.3, 1.0)
LEARNED_WEIGHTS = (0.3, 1.0)
LEARNED_COSTS = ((0.0, 0.0, 0.0), (4.0, 6.0, 2.0), (10.0, 15.0, 4.0))


def evaluate(preds, data):
    sp, mf, short = [], [], np.zeros(2)
    for g in dd.GAMES:
        lab, blk = data[g][1], data[g][2]
        r = pm.evaluate(lab, preds[g], bootstrap_n=0)
        sp.append(r["spell_f1_mean"])
        mf.append(r["macro_f1"])
        rec, _ = recall_by_length(lab, preds[g], blk, dd.FPS)
        short += np.array(rec[0])
    return np.mean(sp), np.mean(mf), short[0] / max(short[1], 1)


def main():
    data = {}
    for g in dd.GAMES:
        d = dd.load(g)
        data[g] = (match_features(g, test_seed(g)), d["controlled"], blocks(d["period"]))
    priors = {g: semimarkov.duration_priors([data[h][1] for h in dd.GAMES if h != g])
              for g in dd.GAMES}
    line = "{:<34} spellF1 {:.3f}  macroF1 {:.3f}  short (<4 s) found {:.0f}%"

    # Hand scores.
    hand_sc = {g: emissions(data[g][0], PARAMS) for g in dd.GAMES}
    vit = {}
    for g in dd.GAMES:
        f, lab, blk = data[g]
        pred = np.full(len(lab), X, dtype=np.int8)
        for a, b in blk:
            pred[a:b] = decode_block(hand_sc[g], f, PARAMS, a, b)
        vit[g] = pred
    s, m, r = evaluate(vit, data)
    print(line.format("hand, Viterbi (v20)", s, m, 100 * r), flush=True)
    best = None
    for w, k in itertools.product(HAND_WEIGHTS, HAND_COST_SCALES):
        costs = transition_cost(PARAMS) * k
        preds = {g: semimarkov.decode_blocks(hand_sc[g], data[g][2], priors[g], costs, w,
                                             n=len(data[g][1])) for g in dd.GAMES}
        res = evaluate(preds, data)
        print(line.format("  hand, semi-Markov w {:g} cost x{:g}".format(w, k), res[0], res[1],
                          100 * res[2]), flush=True)
        if best is None or res[0] > best[0][0]:
            best = (res, w, k)
    print(line.format("hand, semi-Markov best (w {:g}, x{:g})".format(best[1], best[2]),
                      best[0][0], best[0][1], 100 * best[0][2]), flush=True)

    # Learned blend log-probabilities.
    if not os.path.isfile(PROBS):
        print("no learned probabilities yet ({})".format(PROBS))
        return
    with np.load(PROBS) as c:
        blend = {g: (c["gbm__" + g] + c["gru__" + g]) / 2 for g in dd.GAMES}
    for costs in LEARNED_COSTS:
        cdict = dict(zip(("switch", "to_dead", "from_dead"), costs))
        if costs[0] == 0:
            continue
        preds = {g: learned.decode_probs(blend[g], data[g][2], cdict) for g in dd.GAMES}
        res = evaluate(preds, data)
        print(line.format("blend, Viterbi costs {}".format("/".join("{:g}".format(v)
                                                                   for v in costs)), *res[:2],
                          100 * res[2]), flush=True)
    for w, costs in itertools.product(LEARNED_WEIGHTS, LEARNED_COSTS):
        cm = np.zeros((3, 3))
        cm[0, 1] = cm[1, 0] = costs[0]
        cm[0, 2] = cm[1, 2] = costs[1]
        cm[2, 0] = cm[2, 1] = costs[2]
        preds = {g: semimarkov.decode_blocks(np.log(np.clip(blend[g], 1e-4, 1)), data[g][2],
                                             priors[g], cm, w, n=len(data[g][1]))
                 for g in dd.GAMES}
        res = evaluate(preds, data)
        print(line.format("blend, semi-Markov w {:g} costs {}".format(
            w, "/".join("{:g}".format(v) for v in costs)), res[0], res[1], 100 * res[2]),
            flush=True)


if __name__ == '__main__':
    main()
