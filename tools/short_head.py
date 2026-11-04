"""Short-possession head on top of v31 (candidate (b)). Aggregate only.

    python tools/short_head.py                 # 4-fold CV on SkillCorner + PFF (cached v31 runs)
    python tools/short_head.py --ours T MINLEN --name N   # train on all, apply to our match

A gradient-boosted frame classifier reads v31's own outputs and evidence around each frame, seen
from the team v31 gives the frame (T) and the other team (O): O's / T's touch probabilities,
nearest-player distances, co-movement, distance to goal, ball height / speed / seen, gap length,
the refiner's and stage 1's P(T), the touch-run vote, P(dead), where the frame sits in v31's
spell; plus past / future means (0.5, 1, 2 s) of the main ones. Target: the truth is O there and
that truth spell is < MAX_SPELL_S long, i.e. a short possession of the other team that v31 missed.

Insertion: P smoothed over +/- SMOOTH frames; every run of P > T inside one v31 spell of team T,
at least MINLEN s and at most MAX_RUN_S long, is relabelled O.

Cross-validation: the 4 fixed folds of run_v31_external.fold_of, inputs from data/cache/ext_v31
(run tools/run_v31_external.py <set> --save first). Per setting: mean spellF1, delta vs v31,
matches up / down, short spells caught; nested = the setting chosen on the other folds.
"""

import argparse
import glob
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
import run_v31_external as rx  # noqa: E402
import eval_touch_run as tr  # noqa: E402
from pipeline import learned  # noqa: E402
from pipeline.decode import H, A, D, X  # noqa: E402

# Params
FPS = 25
MAX_SPELL_S = 3.0
NEG_STEP = 4
SMOOTH = 2
MAX_RUN_S = 4.0
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
MIN_LENS = (0.2, 0.5, 1.0)
CONTEXT_S = (0.5, 1.0, 2.0)


def spell_pos(codes):
    """(seconds since the spell start, until its end, spell length) of each frame's label run."""
    n = len(codes)
    edges = np.flatnonzero(np.diff(codes)) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [n]))
    since, until, length = np.empty(n), np.empty(n), np.empty(n)
    for a, b in zip(starts, stops):
        r = np.arange(a, b)
        since[a:b], until[a:b], length[a:b] = (r - a) / FPS, (b - 1 - r) / FPS, (b - a) / FPS
    return since, until, length


def inputs(E, codes):
    """(n, k) float32 inputs oriented to the team of codes (frames not H / A get T = H)."""
    T = np.where(codes == A, A, H)
    home = T == H
    s = np.where(home, 1.0, -1.0)

    def pair(name):
        h, a = E["f_" + name + "_home"].astype(np.float64), E["f_" + name + "_away"].astype(np.float64)
        return np.where(home, h, a), np.where(home, a, h)

    tpT, tpO = pair("touch_p")
    dT, dO = pair("d")
    cT, cO = pair("c")
    mT, mO = pair("m")
    gT, gO = pair("g")
    p_ref = E["p_ref"].astype(np.float64)
    pT_ref = np.where(home, p_ref, 1 - p_ref)
    p1 = E["p1"].astype(np.float64)
    live = np.maximum(p1[:, H] + p1[:, A], 1e-9)
    pT_1 = np.where(home, p1[:, H], p1[:, A]) / live
    since, until, length = spell_pos(codes)
    with np.errstate(invalid="ignore"):
        dd_ = dO - dT
    cols = [tpT, tpO, dT, dO, dd_, cT, cO, mT, mO, gT, gO, pT_ref, pT_1, 1 - live,
            s * E["vote"], E["pdead"], E["f_ball_seen"], E["f_ball_speed"], E["f_gap_s"],
            E["ball_z"], E["f_player_speed"], E["f_ball_out"], E["f_keeper_hold"],
            s * E["f_touch"], s * E["f_last_touch"], since, until, length,
            (codes == D).astype(np.float64)]
    for x in (tpO, tpT, dd_, E["ball_z"].astype(np.float64), E["f_ball_speed"].astype(np.float64),
              pT_ref):
        for c in CONTEXT_S:
            w = int(c * FPS)
            cols += [learned.window(x, -w, 0), learned.window(x, 1, w + 1)]
    return np.stack([np.asarray(c, dtype=np.float32) for c in cols], axis=1)


def target(lab, codes):
    """1 where the truth is the other team than codes and that truth spell is short."""
    other = np.where(codes == H, A, np.where(codes == A, H, -1))
    _, _, length = spell_pos(lab)
    return (lab == other) & (length < MAX_SPELL_S)


def train(rows):
    import lightgbm as lgb
    X_ = np.concatenate([r[0] for r in rows])
    y = np.concatenate([r[1] for r in rows]).astype(np.float64)
    params = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63,
              "min_data_in_leaf": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8,
              "bagging_freq": 1, "lambda_l2": 1.0, "seed": 0, "verbose": -1}
    return lgb.train(params, lgb.Dataset(X_, y), num_boost_round=400)


def rows_of(E):
    codes, lab = E["v31"], E["labels"]
    x = inputs(E, codes)
    y = target(lab, codes)
    ok = np.isin(codes, (H, A)) & (lab != X)
    idx = np.flatnonzero(ok & (y | (np.arange(len(y)) % NEG_STEP == 0)))
    return x[idx], y[idx]


def insert(codes, p, t, min_len):
    """codes with every run of smoothed p > t inside one H / A spell (min_len..MAX_RUN_S s)
    given to the other team."""
    q = learned.window(p.astype(np.float64), -SMOOTH, SMOOTH + 1)
    out = codes.copy()
    hot = (q > t) & np.isin(codes, (H, A))
    e = np.diff(np.concatenate(([0], hot.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        if len(np.unique(codes[a:b])) != 1:
            continue
        if not (min_len * FPS <= b - a <= MAX_RUN_S * FPS):
            continue
        out[a:b] = A if codes[a] == H else H
    return out


def load_all():
    data = {}
    for name in rx.SETS if hasattr(rx, "SETS") else ("skillcorner", "pff"):
        for g in rx.games_of(name):
            path = os.path.join(rx.INTER, "v31_{}_{}.npz".format(name, g))
            if os.path.isfile(path):
                with np.load(path) as c:
                    data[(name, g)] = {k: c[k] for k in c.files}
    return data


def caught_n(lab, codes):
    short = rx.short_spells(lab)
    sp = pm.possession_spells(codes, pm.valid_runs(lab))
    return len(short), sum(rx.caught(s, sp) for s in short)


def cv():
    data = load_all()
    print("{} cached matches".format(len(data)), flush=True)
    grid = list(itertools.product(THRESHOLDS, MIN_LENS))
    keys = sorted(data)
    base = {k: tr.score(data[k]["labels"], data[k]["v31"]) for k in keys}
    res = {s: {} for s in grid}
    cat = {s: [0, 0] for s in grid}
    cat0 = [0, 0]
    rows = {k: rows_of(data[k]) for k in keys}
    for fold in range(rx.FOLDS):
        test = [k for k in keys if rx.fold_of(*k) == fold]
        model = train([rows[k] for k in keys if rx.fold_of(*k) != fold])
        for k in test:
            E = data[k]
            p = model.predict(inputs(E, E["v31"]))
            n0, c0 = caught_n(E["labels"], E["v31"])
            cat0[0] += n0
            cat0[1] += c0
            for s in grid:
                codes = insert(E["v31"], p, *s)
                res[s][k] = tr.score(E["labels"], codes)
                cat[s][1] += caught_n(E["labels"], codes)[1]
                cat[s][0] += n0
        print("  fold {} done ({} test matches)".format(fold, len(test)), flush=True)
    b = np.array([base[k] for k in keys])
    print("v31: {:.4f} | short caught {} / {}".format(b.mean(), cat0[1], cat0[0]))
    for name in ("skillcorner", "pff"):
        print("  v31 {}: {:.4f}".format(name, np.mean([base[k] for k in keys if k[0] == name])))
    for s in grid:
        v = np.array([res[s][k] for k in keys])
        by = {name: np.mean([res[s][k] - base[k] for k in keys if k[0] == name])
              for name in ("skillcorner", "pff")}
        print("T {:<4g} min {:<4g} | {:.4f} | {:+.4f} (SC {:+.4f}, PFF {:+.4f}) | {}/{} | caught {}"
              .format(*s, v.mean(), v.mean() - b.mean(), by["skillcorner"], by["pff"],
                      int(np.sum(v > b + 1e-9)), int(np.sum(v < b - 1e-9)), cat[s][1]), flush=True)
    nested = []
    for fold in range(rx.FOLDS):
        others = [k for k in keys if rx.fold_of(*k) != fold]
        best = max(grid, key=lambda s: np.mean([res[s][k] - base[k] for k in others]))
        nested += [(res[best][k] - base[k]) for k in keys if rx.fold_of(*k) == fold]
        print("  nested fold {}: {}".format(fold, best))
    nested = np.array(nested)
    print("nested: {:+.4f} | {}/{}".format(nested.mean(), int(np.sum(nested > 1e-9)),
                                          int(np.sum(nested < -1e-9))))


def ours(t, min_len, name):
    import ours_chain as oc
    data = load_all()
    model = train([rows_of(E) for E in data.values()])
    model.save_model(os.path.join(rx.INTER, "short_head_all.txt"))
    codes, E = oc.ours_v31()
    p = model.predict(inputs(E, codes))
    new = insert(codes, p, t, min_len)
    inside = np.zeros(len(codes), bool)
    for a, b in E["blk"]:
        inside[a:b] = True
    print("frames changed {:.2f}%, inserted runs {}".format(
        100 * np.mean(new[inside] != codes[inside]),
        int(np.sum(np.diff((new != codes).astype(np.int8)) == 1))))
    print(oc.write_and_score(codes, "v31_check"))
    print(oc.write_and_score(new, name))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ours", nargs=2, type=float, metavar=("T", "MINLEN"))
    parser.add_argument("--name", default="v31_sh")
    args = parser.parse_args()
    if args.ours:
        ours(*args.ours, args.name)
    else:
        cv()


if __name__ == "__main__":
    main()
