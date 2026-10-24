"""Train the learned evidence model on Metrica, leave one match out, compare with the hand model.

    python tools/train_external.py                  # LightGBM, leave-one-match-out on the 3 games

For each Metrica game: train on the other two (each also home/away flipped), predict this one,
decode with the Viterbi at a few cost settings, score with the competition's evaluate(). Prints
spellF1 / macroF1 per game and the mean, next to the hand model (v15) on the same labels. Also
the per-frame argmax with no decoder, to show what the decoder adds.
Caches features per game in data/cache/metrica/feat_game<g>.npz.
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
import metrica_data as md  # noqa: E402
import pipeline.features as features  # noqa: E402
from pipeline import learned  # noqa: E402

# Params
COST_GRID = {"switch": (10.0, 20.0, 30.0, 45.0), "to_dead": (8.0, 15.0, 25.0),
             "from_dead": (4.0, 8.0)}


def game_features(g, rebuild=False):
    path = os.path.join(md.CACHE, "feat_game{}.npz".format(g))
    if not rebuild and os.path.isfile(path):
        with np.load(path) as c:
            return {k: c[k] for k in c.files}
    d = md.load(g)
    features.LENGTH, features.WIDTH = md.PITCH
    f = features.build_features({"players": d["players"], "ball": d["ball"], "team": d["team"]},
                                events=[])
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
    r = pm.evaluate(gt, pred)
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=400)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--stack", action="store_true", help="add the hand model's scores and path")
    parser.add_argument("--model", choices=("gbm", "gru"), default="gbm")
    args = parser.parse_args()

    data = {}
    for g in md.GAMES:
        d = md.load(g)
        data[g] = (game_features(g, args.rebuild), d["labels"], d["period"])
    blk = {g: (blocks(per) if args.stack else None) for g, (_, _, per) in data.items()}
    rows = ({g: learned.training_rows(f, lab, blocks=blk[g]) for g, (f, lab, _) in data.items()}
            if args.model == "gbm" else {g: (None, np.zeros(0)) for g in data})
    names = learned.matrix(data[1][0], blk[1])[1]
    print("{} input columns, {} training rows per game (with the flipped copy)".format(
        len(names), " / ".join(str(len(rows[g][1])) for g in md.GAMES)))

    probs = {}
    for g in md.GAMES:
        if args.model == "gru":
            from pipeline.sequence import train_gru, predict_gru
            print("fold: test game {}".format(g))
            trained = train_gru([(data[h][0], data[h][1]) for h in md.GAMES if h != g])
            probs[g] = predict_gru(trained, data[g][0])
            continue
        model = learned.train_gbm([rows[h] for h in md.GAMES if h != g], rounds=args.rounds)
        probs[g] = learned.predict_proba(model, data[g][0], blk[g])
        if g == md.GAMES[-1]:
            imp = model.feature_importance("gain")
            top = np.argsort(imp)[::-1][:12]
            print("top inputs by gain (last fold): " + ", ".join(names[i] for i in top))

    # Frame argmax, no decoder.
    print("\nno decoder (argmax per frame):")
    for g in md.GAMES:
        gt, per = data[g][1], data[g][2]
        pred = np.full(len(gt), md.X, dtype=np.int8)
        for a, b in blocks(per):
            pred[a:b] = np.argmax(probs[g][a:b], axis=1)
        s, m, n_p, n_t = score(gt, pred)
        print("  game {}  spellF1 {:.3f}  macroF1 {:.3f}  poss {} / {}".format(g, s, m, n_p, n_t))

    # Decoder cost grid; report every setting's mean, then the best per game.
    print("\nwith the Viterbi decoder (costs in log-prob units):")
    results = []
    for sw, td, fd in itertools.product(*COST_GRID.values()):
        costs = {"switch": sw, "to_dead": td, "from_dead": fd}
        per_game = []
        for g in md.GAMES:
            gt, per = data[g][1], data[g][2]
            per_game.append(score(gt, learned.decode_probs(probs[g], blocks(per), costs)))
        mean = np.mean([x[0] for x in per_game]), np.mean([x[1] for x in per_game])
        results.append((mean, costs, per_game))
        print("  switch {:4.1f} to_dead {:3.1f} from_dead {:3.1f}  spellF1 {:.3f}  macroF1 {:.3f}"
              .format(sw, td, fd, *mean))
    best = max(results, key=lambda r: r[0][0])
    print("\nbest costs {}:".format(best[1]))
    for g, (s, m, n_p, n_t) in zip(md.GAMES, best[2]):
        print("  game {}  spellF1 {:.3f}  macroF1 {:.3f}  poss {} / {}".format(g, s, m, n_p, n_t))
    print("  MEAN    spellF1 {:.3f}  macroF1 {:.3f}   (compare: tools/run_external.py)".format(
        *best[0]))
    np.savez(os.path.join(md.CACHE, "{}{}_lomo_probs.npz".format(args.model, "_stack" if args.stack else "")),
             **{"game{}".format(g): probs[g] for g in md.GAMES})


if __name__ == '__main__':
    main()
