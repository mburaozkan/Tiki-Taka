"""Insert short-net spells by expected spellF1 gain (neighbours included). Aggregate only.

    python tools/short_insert.py delta [--net e3 --gen 0.6]   # each candidate's real effect, alone
    python tools/short_insert.py train [--net e3 --gen 0.6]   # regressor, 4 fixed folds (held out)
    python tools/short_insert.py eval  [--net e3 --gen 0.6]   # greedy insertion vs base / keep check

Candidates: the short-net spells of tools/short_check.py build (data/cache/short_net/check/
<net>_g<gen>_<set>_<match>.npz: inputs x, meta a, b, team, ...), base v31_d1 (turnover_cls.base_chain).
delta: each candidate pasted ALONE over the base's frames (H / A / D, as short_check's over_d) and
the match's spellF1 recomputed; target = (spellF1 with it - base) x the match's number of true
possessions, i.e. the change in "possessions' worth" of score: the spell itself, the neighbours it
repairs, and the split / false-spell damage when it is wrong, all measured, nothing modelled.
train: LightGBM regression on x (+ short_check's held-out P(keep) k1 as one more input), the 4
fixed external folds; every candidate gets the prediction of the model that never saw its match.
eval: per match, candidates with predicted gain >= tau pasted greedily, highest first, skipping any
that overlaps one already pasted; spellF1 SkillCorner / PFF (up / down vs base), caught < 2 s /
2-5 s, predicted / true, precision; nested 4-fold choice of tau; and the old keep-check rule
(P(keep) >= thr, short_check eval) on the same matches for comparison.
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
from pipeline.decode import H, A, D  # noqa: E402

OUT = os.path.join(ROOT, "data", "cache", "short_insert")
TAUS = (-0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
KEEP_THR = (0.2, 0.3, 0.4, 0.5)


def ctx_of(name, g):
    import turnover_cls as T
    import run_v31_external as rx
    import pair_cls as P
    c, f = T.load_cache(name, g)
    pdead = P.pdead_of(name, g)
    blk = rx.blocks(c["period"])
    return c["labels"], T.base_chain(c, f, blk, pdead)


def check_path(net, gen, name, g):
    import short_check as SC
    return os.path.join(SC.OUT, "{}_g{:g}_{}_{}.npz".format(net, gen, name, g))


def paste(base, spells):
    out = base.copy()
    for a, b, team in spells:
        seg = out[a:b]
        out[a:b] = np.where(np.isin(seg, (H, A, D)), team, seg)
    return out


def delta_match(job):
    name, g, net, gen = job
    path = os.path.join(OUT, "delta_{}_g{:g}_{}_{}.npy".format(net, gen, name, g))
    if os.path.isfile(path):
        return path
    import eval_touch_run as tr
    import possession_metrics as pm
    lab, base = ctx_of(name, g)
    n_true = len(pm.possession_spells(lab, pm.valid_runs(lab)))
    with np.load(check_path(net, gen, name, g)) as z:
        meta = z["meta"]
    s0 = tr.score(lab, base)
    d = np.array([(tr.score(lab, paste(base, [(int(m[0]), int(m[1]), int(m[2]))])) - s0) * n_true
                  for m in meta], np.float32)
    np.save(path, d)
    return path


def jobs(args):
    import turnover_cls as T
    return [(n, g, args.net, args.gen) for n, g, _ in T.ext_list()]


def delta_cmd(args):
    os.makedirs(OUT, exist_ok=True)
    with Pool(args.jobs) as pool:
        paths = pool.map(delta_match, jobs(args), chunksize=1)
    d = np.concatenate([np.load(p) for p in paths])
    print("{} candidates: real gain alone > 0 for {:.1%}, < 0 for {:.1%}; mean {:+.3f} possessions' worth; "
          "sum of positives {:.0f}, of negatives {:.0f}".format(len(d), np.mean(d > 1e-6), np.mean(d < -1e-6),
                                                             d.mean(), d[d > 0].sum(), d[d < 0].sum()))


def load_all(args):
    import turnover_cls as T
    import short_check as SC
    keep = np.load(os.path.join(SC.OUT, "pred_k1_{}_g{:g}.npz".format(args.net, args.gen)))
    rows = []
    for n, g, fold in T.ext_list():
        with np.load(check_path(args.net, args.gen, n, g)) as z:
            x, meta = z["x"], z["meta"]
        key = "{}_{}".format(n, g)
        pk = keep[key] if key in keep.files else np.zeros(len(meta))
        d = np.load(os.path.join(OUT, "delta_{}_g{:g}_{}_{}.npy".format(args.net, args.gen, n, g)))
        rows.append((n, g, fold, np.column_stack([x, pk]) if len(x) else np.zeros((0, x.shape[1] + 1)), meta, d, pk))
    return rows


def train_cmd(args):
    import lightgbm as lgb
    rows = load_all(args)
    folds = sorted({r[2] for r in rows})
    preds = {}
    for k in folds:
        tr_x = np.concatenate([r[3] for r in rows if r[2] != k])
        tr_y = np.concatenate([r[5] for r in rows if r[2] != k])
        model = lgb.train(dict(objective="huber", alpha=1.0, learning_rate=0.03, num_leaves=31,
                               min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
                               bagging_freq=1, verbose=-1, seed=0),
                          lgb.Dataset(tr_x, tr_y), num_boost_round=600)
        for r in rows:
            if r[2] == k and len(r[3]):
                preds["{}_{}".format(r[0], r[1])] = model.predict(r[3]).astype(np.float32)
    np.savez(os.path.join(OUT, "pred_{}_g{:g}.npz".format(args.net, args.gen)), **preds)
    y = np.concatenate([r[5] for r in rows])
    p = np.concatenate([preds.get("{}_{}".format(r[0], r[1]), np.zeros(0)) for r in rows])
    print("held-out: corr(predicted, real gain) {:.3f}; candidates predicted > 0: {:.1%}, of them really > 0: {:.1%}"
          .format(np.corrcoef(p, y)[0, 1], np.mean(p > 0), np.mean(y[p > 0] > 1e-6)))


def eval_match(job):
    name, g, net, gen, pred, pk = job
    import possession_metrics as pm
    from decoder_sweep import stats
    lab, base = ctx_of(name, g)
    with np.load(check_path(net, gen, name, g)) as z:
        meta = z["meta"]
    res = [stats(lab, base)]

    def greedy(score, thr):
        chosen, taken = [], np.zeros(len(lab), bool)
        for i in np.argsort(-score):
            if score[i] < thr:
                break
            a, b, team = int(meta[i, 0]), int(meta[i, 1]), int(meta[i, 2])
            if taken[a:b].any():
                continue
            taken[a:b] = True
            chosen.append((a, b, team))
        return chosen

    for tau in TAUS:
        res.append(stats(lab, paste(base, greedy(pred, tau)) if len(meta) else base))
    for thr in KEEP_THR:                      # the old rule: P(keep) >= thr, weakest first, overwrite
        sel = [i for i in np.argsort(pk) if pk[i] >= thr]
        res.append(stats(lab, paste(base, [(int(meta[i, 0]), int(meta[i, 1]), int(meta[i, 2])) for i in sel])
                         if len(meta) else base))
    return name, g, np.array(res, float)


def eval_cmd(args):
    import run_v31_external as rx
    import turnover_cls as T
    rows = load_all(args)
    pz = np.load(os.path.join(OUT, "pred_{}_g{:g}.npz".format(args.net, args.gen)))
    jb = [(r[0], r[1], args.net, args.gen, pz["{}_{}".format(r[0], r[1])] if "{}_{}".format(r[0], r[1]) in pz.files
           else np.zeros(0), r[6]) for r in rows]
    with Pool(args.jobs) as pool:
        res = pool.map(eval_match, jb, chunksize=1)
    names = np.array([r[0] for r in res])
    folds = np.array([rx.fold_of(r[0], r[1]) for r in res])
    arr = np.stack([r[2] for r in res])
    labels = (["base v31_d1"] + ["gain >= {:+g}".format(t) for t in TAUS]
              + ["old keep >= {:g}".format(t) for t in KEEP_THR])
    print("net {}, spells at {:g}".format(args.net, args.gen))
    print("{:<18} {:>17} {:>17} | {:>10} {:>9} | {:>9} {:>9}".format(
        "rule", "SkillCorner", "PFF", "caught <2s", "2-5s", "pred/true", "precision"))
    for j, lb in enumerate(labels):
        parts = []
        for nm in ("skillcorner", "pff"):
            v = arr[names == nm, j, 0]
            dlt = v - arr[names == nm, 0, 0]
            parts.append("{:.4f} {:>3}/{:<3}".format(v.mean(), int(np.sum(dlt > 1e-9)), int(np.sum(dlt < -1e-9))))
        t = arr[:, j, 1:].sum(axis=0)
        print("{:<18} {:>17} {:>17} | {:>9.1%} {:>9.1%} | {:>9.2f} {:>9.1%}".format(
            lb, parts[0], parts[1], t[1] / t[0], t[3] / t[2], t[5] / t[4], t[6] / t[5]))
    gain = arr[:, 1:, 0] - arr[:, :1, 0]
    for nm, cols in (("gain rule", range(0, len(TAUS))), ("old keep rule", range(len(TAUS), len(TAUS) + len(KEEP_THR)))):
        cols = list(cols)
        nested = np.zeros(len(arr))
        for k in range(T.FOLDS):
            j = cols[int(np.argmax(gain[folds != k][:, cols].mean(axis=0)))]
            nested[folds == k] = gain[folds == k, j]
        print("nested 4-fold, {:<13}: SkillCorner {:+.4f}, PFF {:+.4f} ({}/{})".format(
            nm, nested[names == "skillcorner"].mean(), nested[names == "pff"].mean(),
            int(np.sum(nested > 1e-9)), int(np.sum(nested < -1e-9))))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("delta", "train", "eval"))
    ap.add_argument("--net", default="e3")
    ap.add_argument("--gen", type=float, default=0.6)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    {"delta": delta_cmd, "train": train_cmd, "eval": eval_cmd}[args.cmd](args)


if __name__ == "__main__":
    main()
