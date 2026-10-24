"""Train and test the touch network on DFL, one match held out at a time. Aggregate only.

    python tools/train_touch.py                 # 7 folds
    python tools/train_touch.py --folds 2       # the first 2 held-out matches only

Saves each fold's model as data/cache/models/touch_net_<held-out match>.pt (for testing the
pipeline on DFL with a network that never saw that match), and with --final also a model
trained on all 7 as touch_net_all.pt (the one used on our match).

Every match's ball is made to look like ours (dfl_data.prepare_ball: dead ball animated,
blurred, our gap pattern); training uses VERSIONS random gap patterns per match, each also with the
teams swapped. The held-out match gets one fixed gap pattern, and both the network and our current
rule (pipeline.features.touches on the same ball) are scored against the teacher touches
(tools/touch_labels.py):
  P / R / F1   touches matched one to one, same team, within TOL_S
  turnover R   recall of the true touches whose team differs from the previous true touch's:
               the touches that change possession, which is what the decoder needs
  offset       median (predicted - true) of matched touches, s
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import dfl_data as dd  # noqa: E402
import touch_labels as tl  # noqa: E402
from gap_model import degrade  # noqa: E402
import pipeline.features as features  # noqa: E402
from pipeline import touch_net as tn  # noqa: E402

# Params
MODEL_DIR = os.path.join(dd.DATA_DIR, "cache", "models")
VERSIONS = 2            # gap patterns per training match
TEST_SEED = 100
TOLS_S = (0.2, 0.5)
THRESHOLDS = (0.3, 0.5, 0.7)


def changes(lab):
    """(frames, teams) where controlled possession starts: H or A after anything else graded."""
    f = np.flatnonzero(np.isin(lab[1:], (dd.H, dd.A)) & (lab[1:] != lab[:-1])
                       & (lab[:-1] != dd.X)) + 1
    return f, (lab[f] == dd.A).astype(np.int8)


def prepared(game, seed, target="touch"):
    """Ball made to look like ours for one match, plus the target events (touches, or possession
    changes for the turnover network)."""
    d = dd.load(game)
    ball = dd.prepare_ball(d, seed)        # dead ball animated, blurred, our gaps
    frames, teams = (tl.load(game)[:2] if target == "touch" else changes(d["controlled"]))
    valid = d["period"] > 0
    return d, ball, frames, teams, valid


def match(pf, pt, tf, tt, tol):
    """(matched pred, matched true, offsets) greedy one-to-one, same team, nearest first."""
    pairs = []
    for i, (f, t) in enumerate(zip(tf, tt)):
        lo, hi = np.searchsorted(pf, [f - tol, f + tol + 1])
        for j in range(lo, hi):
            if pt[j] == t:
                pairs.append((abs(pf[j] - f), i, j, pf[j] - f))
    pairs.sort()
    used_t, used_p, offs = set(), set(), []
    for _, i, j, off in pairs:
        if i in used_t or j in used_p:
            continue
        used_t.add(i)
        used_p.add(j)
        offs.append(off)
    return used_p, used_t, offs


def scores(pf, pt, tf, tt):
    """{tol: (P, R, F1, turnover R, median offset s)}."""
    turnover = np.concatenate(([False], tt[1:] != tt[:-1]))
    out = {}
    for tol_s in TOLS_S:
        mp, mt, offs = match(pf, pt, tf, tt, int(round(tol_s * dd.FPS)))
        p = len(mp) / max(len(pf), 1)
        r = len(mt) / max(len(tf), 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        tr = np.mean([i in mt for i in np.flatnonzero(turnover)]) if turnover.any() else np.nan
        out[tol_s] = (p, r, f1, tr, np.median(offs) / dd.FPS if offs else np.nan)
    return out


def rule_touches(d, ball):
    length, width = features.LENGTH, features.WIDTH
    features.LENGTH, features.WIDTH = dd.PITCH
    f = features.build_features({"players": d["players"], "ball": ball, "team": d["team"]},
                                events=[])
    features.LENGTH, features.WIDTH = length, width
    frames = np.flatnonzero(f["touch"] != 0)
    return frames, (f["touch"][frames] < 0).astype(np.int8)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folds", type=int, default=len(dd.GAMES))
    parser.add_argument("--epochs", type=int, default=tn.EPOCHS)
    parser.add_argument("--final", action="store_true", help="also train on all 7 and save")
    parser.add_argument("--target", choices=("touch", "turnover"), default="touch",
                        help="touches (touch_net_*) or controlled possession changes "
                             "(turnover_net_*)")
    args = parser.parse_args()
    stem = "touch_net" if args.target == "touch" else "turnover_net"
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("building inputs ...", flush=True)
    train_sets = {}
    for g in dd.GAMES:
        sets = []
        for v in range(VERSIONS):
            d, ball, frames, teams, valid = prepared(g, seed=v, target=args.target)
            for flip in (False, True):
                x = tn.inputs(d["players"], ball, d["team"], dd.PITCH, flip=flip)
                sets.append((x, tn.targets(len(x), frames, teams, flip=flip), valid))
        train_sets[g] = sets

    rows = {"rule": [], **{"net@{}".format(t): [] for t in THRESHOLDS}}
    chg = {"rule": [], **{"net@{}".format(t): [] for t in THRESHOLDS}}
    for held in dd.GAMES[:args.folds]:
        print("held out {} ...".format(held), flush=True)
        bundle = tn.train([s for g in dd.GAMES if g != held for s in train_sets[g]],
                          epochs=args.epochs, log=lambda *_: None)
        tn.save(bundle, os.path.join(MODEL_DIR, "{}_{}.pt".format(stem, held)))
        d, ball, tf, tt, valid = prepared(held, seed=TEST_SEED, target=args.target)
        cf, ct = changes(d["controlled"])
        prob = tn.predict(bundle, tn.inputs(d["players"], ball, d["team"], dd.PITCH))
        rf, rt = rule_touches(d, ball)
        rows["rule"].append(scores(rf, rt, tf, tt))
        chg["rule"].append(scores(rf, rt, cf, ct))
        for th in THRESHOLDS:
            pf, pt = tn.pick(prob, th)
            rows["net@{}".format(th)].append(scores(pf, pt, tf, tt))
            chg["net@{}".format(th)].append(scores(pf, pt, cf, ct))
        r, n = rows["rule"][-1][0.5], rows["net@0.5"][-1][0.5]
        print("  +/-0.5 s  rule F1 {:.2f} turnover R {:.2f} | net@0.5 F1 {:.2f} turnover R {:.2f}"
              .format(r[2], r[3], n[2], n[3]), flush=True)

    if args.final:
        print("training on all 7 ...", flush=True)
        bundle = tn.train([s for g in dd.GAMES for s in train_sets[g]], epochs=args.epochs,
                          log=lambda *_: None)
        tn.save(bundle, os.path.join(MODEL_DIR, "{}_all.pt".format(stem)))

    print("\nmean over {} held-out matches".format(args.folds))
    print("{:<10} {:>6} {:>32} {:>32}".format("", "", "within 0.2 s: P  R  F1  turnR  off",
                                              "within 0.5 s: P  R  F1  turnR  off"))
    for title, table in (("against its own target ({})".format(args.target), rows),
                         ("against the true possession changes (controlled labels)", chg)):
        print(title)
        for name, rs in table.items():
            line = "{:<10}".format(name)
            for tol in TOLS_S:
                m = np.nanmean([r[tol] for r in rs], axis=0)
                line += "   {:.2f} {:.2f} {:.2f} {:.2f} {:+.2f}s".format(*m)
            print(line)


if __name__ == '__main__':
    main()
