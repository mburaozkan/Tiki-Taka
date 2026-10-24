"""Team network: which team has the ball, frame by frame, during live play. DFL, one match out.

    python tools/train_team.py            # 7 folds + the model on all 7, then the decoder test

The oracle test (JOURNAL.md, 2026-10-25) put the bottleneck here: perfect team evidence lifts
DFL spellF1 from 0.534 to 0.73-0.76, perfect change points only to 0.57-0.59. So this network is
trained for that question alone:
  inputs   pipeline.touch_net.inputs (ball, nearest players, ball path) plus the touch network's
           P(home touch), P(away touch) from the touch network trained WITHOUT this match
  target   the team in controlled possession, on live frames only (stoppages stay with the hand
           model, as in the hybrid decoder)
  context  dilations up to 64, about +/- 5 s
DFL prepared like ours (dfl_data.prepare_ball); VERSIONS gap patterns per training match, each also
with the teams swapped. Reports per-frame team accuracy on the held-out matches, then the hybrid
decoder (hand D vs live from pipeline.decode, team from this network) against the touch-network
pipeline, over HYBRID_SWITCH. Models: data/cache/models/team_net_<match>.pt and team_net_all.pt;
held-out probabilities: data/cache/dfl/team_probs.npz.
"""

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
from train_learned_dfl import match_features, test_seed, blocks  # noqa: E402
from pipeline import touch_net as tn, learned  # noqa: E402
from pipeline.decode import PARAMS, emissions, decode_block, X, H, A  # noqa: E402

# Params
MODEL_DIR = os.path.join(dd.DATA_DIR, "cache", "models")
PROBS = os.path.join(dd.CACHE, "team_probs.npz")
VERSIONS = 2
DILATIONS = (1, 2, 4, 8, 16, 32, 64)
HYBRID_SWITCH = (2.0, 4.0, 6.0, 10.0, 15.0, 20.0)


def team_inputs(d, ball, touch_model, flip=False):
    """touch_net inputs plus the touch network's two probabilities (swapped when flipped)."""
    x = tn.inputs(d["players"], ball, d["team"], dd.PITCH, flip=flip)
    p = tn.predict(touch_model, tn.inputs(d["players"], ball, d["team"], dd.PITCH,
                                          path=len(touch_model[1]) > tn.n_base_inputs()))
    tp = p[:, [2, 1]] if flip else p[:, 1:]
    return np.concatenate([x, tp.astype(np.float32)], axis=1)


def team_targets(lab, flip=False):
    """(y, valid): 1 home / 2 away on live frames (swapped when flipped); valid = live."""
    home, away = (2, 1) if flip else (1, 2)
    y = np.where(lab == H, home, np.where(lab == A, away, 0)).astype(np.int64)
    return y, np.isin(lab, (H, A))


def main():
    tms = {g: tn.load(os.path.join(MODEL_DIR, "touch_net_{}.pt".format(g))) for g in dd.GAMES}
    print("building inputs ...", flush=True)
    train_sets, test = {}, {}
    for g in dd.GAMES:
        d = dd.load(g)
        sets = []
        for v in range(VERSIONS):
            ball = dd.prepare_ball(d, seed=v)
            for flip in (False, True):
                y, valid = team_targets(d["controlled"], flip)
                sets.append((team_inputs(d, ball, tms[g], flip), y, valid))
        train_sets[g] = sets
        test[g] = team_inputs(d, dd.prepare_ball(d, seed=test_seed(g)), tms[g])

    probs, acc = {}, []
    for held in dd.GAMES:
        print("held out {} ...".format(held), flush=True)
        bundle = tn.train([s for g in dd.GAMES if g != held for s in train_sets[g]],
                          log=lambda *_: None, dilations=DILATIONS, weights=(1.0, 1.0, 1.0))
        tn.save(bundle, os.path.join(MODEL_DIR, "team_net_{}.pt".format(held)))
        p = tn.predict(bundle, test[held])
        lab = dd.load(held)["controlled"]
        live = np.isin(lab, (H, A))
        call = np.where(p[:, 1] >= p[:, 2], H, A)
        acc.append(np.mean(call[live] == lab[live]))
        print("  per-frame team accuracy {:.1f}%".format(100 * acc[-1]), flush=True)
        # as P(H, A, D) for decode_hybrid (only the H vs A ratio is used there)
        probs[held] = np.stack([p[:, 1], p[:, 2], p[:, 0]], axis=1)
    np.savez(PROBS, **probs)
    print("training on all 7 ...", flush=True)
    tn.save(tn.train([s for g in dd.GAMES for s in train_sets[g]], log=lambda *_: None,
                     dilations=DILATIONS, weights=(1.0, 1.0, 1.0)),
            os.path.join(MODEL_DIR, "team_net_all.pt"))
    print("\nper-frame team accuracy, mean of 7 held-out: {:.1f}%  (hand 82.1%, learned blend "
          "88.6%)".format(100 * np.mean(acc)))

    # Decoder: the touch-network pipeline (v20-style) vs the hybrid with this network's team.
    line = "{:<30} spellF1 {:.3f}  macroF1 {:.3f}"
    base, hand_codes, data = [], {}, {}
    for g in dd.GAMES:
        d = dd.load(g)
        f = match_features(g, test_seed(g))
        lab, blk = d["controlled"], blocks(d["period"])
        sc = emissions(f, PARAMS)
        pred = np.full(len(lab), X, dtype=np.int8)
        for a, b in blk:
            pred[a:b] = decode_block(sc, f, PARAMS, a, b)
        hand_codes[g], data[g] = pred, (lab, blk)
        r = pm.evaluate(lab, pred, bootstrap_n=0)
        base.append((r["spell_f1_mean"], r["macro_f1"]))
    print(line.format("hand + touch network", *np.mean(base, axis=0)))
    for sw in HYBRID_SWITCH:
        res = []
        for g in dd.GAMES:
            lab, blk = data[g]
            pred = learned.decode_hybrid(hand_codes[g], probs[g], blk, sw)
            r = pm.evaluate(lab, pred, bootstrap_n=0)
            res.append((r["spell_f1_mean"], r["macro_f1"]))
        print(line.format("hybrid, team net, switch {:g}".format(sw), *np.mean(res, axis=0)),
              flush=True)


if __name__ == '__main__':
    main()
