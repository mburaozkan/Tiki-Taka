"""First-stage inputs for the second-stage network without the stacking leak, on more gap patterns.

    python tools/clean_stage1.py            # build everything missing (pairs + singles), aggregate only

tools/eval_mbr.v25_inputs gives each DFL match the first stage held out from THAT match only, so
when the second stage is validated on match g, its training inputs for another match o came from
first-stage models that saw g's labels (the leak). Here the first stage for (o, held-out g) is
trained without both o and g: GBM + GRU (tools/train_learned_dfl.py), the players-only vote with
DFL labels from the other five matches + o's own pseudo-labels (in_domain_team), combined as v25
(weight W). Every target is also built on the GBM's training gap patterns (version 1, 2 =
test_seed + 1, + 2), which the pair model never saw: three gap patterns per match instead of one.

`inputs(target, excluded, v)` -> (labels, hand codes, combined prob), cached per call in
data/cache/dfl/s1_<excluded>_<target>_<v>.npz. excluded = (g, o) sorted for the leak-free
second stage on held-out g; excluded = (o,) for the deployed second stage (trained on all 7).
"""

import itertools
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import dfl_data as dd  # noqa: E402
import in_domain_team as idt  # noqa: E402
from eval_mbr import W  # noqa: E402
from train_learned_dfl import match_features, test_seed, blocks, TRAIN_VERSIONS  # noqa: E402
from pipeline import learned  # noqa: E402
from pipeline.decode import PARAMS, emissions, decode_block, X, H, A  # noqa: E402

# Params
VERSIONS = (0, 1, 2)        # gap patterns: test_seed(g) + v


def _path(excluded, target, v):
    return os.path.join(dd.CACHE, "s1_{}_{}_{}.npz".format("+".join(excluded), target, v))


def _hand(g, v, f, lab, blk):
    path = os.path.join(dd.CACHE, "hand_{}_{}.npy".format(g, v))
    if os.path.isfile(path):
        return np.load(path)
    sc = emissions(f, PARAMS)
    hand = np.full(len(lab), X, dtype=np.int8)
    for a, b in blk:
        hand[a:b] = decode_block(sc, f, PARAMS, a, b)
    np.save(path, hand)
    return hand


def inputs(target, excluded, v):
    with np.load(_path(tuple(sorted(excluded)), target, v)) as c:
        return c["lab"], c["hand"], c["prob"]


def build(excluded, targets, log=print):
    """Train the first stage without `excluded`, write inputs for each target x version."""
    excluded = tuple(sorted(excluded))
    todo = [(t, v) for t in targets for v in VERSIONS if not os.path.isfile(_path(excluded, t, v))]
    if not todo:
        return
    t0 = time.time()
    train_games = [g for g in dd.GAMES if g not in excluded]
    train = [(match_features(g, test_seed(g) + 1 + k), dd.load(g)["controlled"])
             for g in train_games for k in range(TRAIN_VERSIONS)]
    gbm = learned.train_gbm([learned.training_rows(f, lab, step=3) for f, lab in train])
    from pipeline.sequence import train_gru, predict_gru
    gru = train_gru(train, log=lambda *_: None)
    del train
    po_sets = []
    for g in train_games:
        lab = dd.load(g)["controlled"]
        po_sets.append((idt.player_cols_cached(g, dd.load(g)), lab, np.isin(lab, (H, A))))
    for t, v in todo:
        d = dd.load(t)
        f = match_features(t, test_seed(t) + v)
        lab, blk = d["controlled"], blocks(d["period"])
        hand = _hand(t, v, f, lab, blk)
        pg = learned.predict_proba(gbm, f)
        blend = (pg + predict_gru(gru, f)) / 2
        team, conf, weak = idt.pseudo_labels(f, hand)
        dteam, dmask = idt.decoded_pseudo(learned.decode_hybrid(hand, pg, blk, 20.0), pg)
        cols = idt.player_cols_cached(t, d)
        own = (cols, np.where(conf, team, dteam), conf | dmask)
        p_po = idt.predict_po(idt.train_po_multi(po_sets + [own]), cols)
        prob = idt.combine(blend, p_po, weak, W)
        np.savez(_path(excluded, t, v), lab=lab, hand=hand, prob=prob.astype(np.float32))
    log("  without {}: {} inputs in {:.0f} s".format("+".join(excluded), len(todo), time.time() - t0))


def main():
    for pair in itertools.combinations(dd.GAMES, 2):
        build(pair, pair)
    for g in dd.GAMES:
        build((g,), (g,))
    # aggregate check: per-frame team accuracy on live frames, pair vs single exclusion
    for name, ex in (("pair", lambda t: (t, dd.GAMES[0] if t != dd.GAMES[0] else dd.GAMES[1])),
                     ("single", lambda t: (t,))):
        acc = []
        for t in dd.GAMES:
            for v in VERSIONS:
                lab, _, prob = inputs(t, ex(t), v)
                live = np.isin(lab, (H, A))
                acc.append(np.mean((prob[live, H] >= prob[live, A]) == (lab[live] == H)))
        print("{:<6} first-stage team accuracy on live frames, mean over matches x versions: {:.1f}%"
              .format(name, 100 * np.mean(acc)), flush=True)


if __name__ == '__main__':
    main()
