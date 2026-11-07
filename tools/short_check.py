"""Keep / drop check for the short-net spells, and their edges from the start / end heads. Aggregate only.

    python tools/short_check.py build --net s1 [--gen 0.4]   # spells + inputs per external match
    python tools/short_check.py train --net s1 [--tag k1]     # XGBoost GPU, 4 fixed folds + all
    python tools/short_check.py eval --net s1 [--tag k1]      # paste kept spells over v31_d1

Spells: hysteresis runs of the short net's held-out P(inside) per team (>= gen, extended while
>= gen - 0.15), 0.2-5 s (tools/short_net.py spells_from), generated loosely so the check decides.
Label: a true possession of that team with IoU >= 0.5. Inputs: length; P(inside) mean / max / min
/ at the edges; the other team's P(inside) over it; start / end head peaks near the edges; the
net's class P (team / other / D) over the spell and 1 s before / after; the base's codes over
it and just before / after; hand D, P(dead); the single-touch inputs (v2) at the start oriented to
the team and at the end oriented to the other team.
Edges: start moved to the start head's peak within +/- EDGE_S of the spell start (if >= EDGE_P),
end likewise with the end head.
eval: spells with P(keep) >= thr pasted (strongest last), also over D; spellF1 vs base, up /
down, nested 4-fold, caught < 2 s / 2-5 s, predicted / true, precision.
"""

import argparse
import itertools
import os
import sys
from multiprocessing import Pool

os.environ["TC_ROWS"] = "v2"
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
import turnover_cls as T  # noqa: E402
import short_net as S  # noqa: E402
from pipeline.decode import H, A, D  # noqa: E402

FPS = 25
EDGE_S = 0.5
EDGE_P = 0.2
OUT = os.path.join(S.OUT, "check")
GRID = {"thr": (0.0, 0.1, 0.2, 0.3, 0.4, 0.5), "edges": (0, 1), "over_d": (1,)}


def iou(a, b, c, e):
    return max(0, min(b, e) - max(a, c)) / float(max(b, e) - min(a, c))


def gen_spells(pr, gen):
    out = []
    for s in (0, 1):
        for a, b in S.spells_from(pr[:, s].astype(np.float32), gen, gen - 0.15):
            out.append((a, b, s))
    return out


def spell_inputs(ctx, pr, a, b, s):
    n = ctx["n"]
    base, c = ctx["base"], ctx["c"]
    team, other = (H, A) if s == 0 else (A, H)
    q, o = pr[:, s], pr[:, 1 - s]
    w = int(EDGE_S * FPS)
    x = {"len_s": (b - a) / FPS, "q_mean": float(q[a:b].mean()), "q_max": float(q[a:b].max()),
         "q_min": float(q[a:b].min()), "q_a": float(q[a]), "q_b": float(q[b - 1]),
         "o_max": float(o[a:b].max()), "o_mean": float(o[a:b].mean()),
         "st_peak": float(pr[max(a - w, 0):a + w, 2 + s].max()),
         "en_peak": float(pr[max(b - 1 - w, 0):b - 1 + w, 4 + s].max()),
         "st_off": float(np.argmax(pr[max(a - w, 0):a + w, 2 + s]) - min(w, a)) / FPS,
         "en_off": float(np.argmax(pr[max(b - 1 - w, 0):b - 1 + w, 4 + s]) - min(w, b - 1)) / FPS}
    cl_t, cl_o = 6 + s, 7 - s
    for tag, sl in (("in", slice(a, b)), ("pre", slice(max(a - FPS, 0), a)),
                    ("post", slice(b, min(b + FPS, n)))):
        if sl.stop <= sl.start:
            x.update({"cl_t_" + tag: np.nan, "cl_o_" + tag: np.nan, "cl_d_" + tag: np.nan,
                      "base_t_" + tag: np.nan, "base_o_" + tag: np.nan, "base_d_" + tag: np.nan})
            continue
        x["cl_t_" + tag] = float(pr[sl, cl_t].mean())
        x["cl_o_" + tag] = float(pr[sl, cl_o].mean())
        x["cl_d_" + tag] = float(pr[sl, 8].mean())
        x["base_t_" + tag] = float(np.mean(base[sl] == team))
        x["base_o_" + tag] = float(np.mean(base[sl] == other))
        x["base_d_" + tag] = float(np.mean(base[sl] == D))
    x["handD"] = float(np.mean(c["hand"][a:b] == D))
    x["dead"] = float(np.mean(ctx["pdead"][a:b]))
    x.update({"s_" + k: v for k, v in T.touch_feats(ctx, a, s, max(a - 5, 0)).items()})
    x.update({"e_" + k: v for k, v in T.touch_feats(ctx, min(b, n - 1), 1 - s,
                                                     max(b - 5, 0)).items()})
    return x


def build_match(job):
    name, g, fold, net, gen = job
    path = os.path.join(OUT, "{}_g{:g}_{}_{}.npz".format(net, gen, name, g))
    if os.path.isfile(path):
        return path
    import dfl_data as dd
    import run_v31_external as rx
    import pair_cls as P
    from ext_stage1 import cols_for
    with np.load(os.path.join(S.OUT, "pred_{}.npz".format(net))) as z:
        pr = z["{}_{}".format(name, g)].astype(np.float32)
    c, f = T.load_cache(name, g)
    pdead = P.pdead_of(name, g)
    blk = rx.blocks(c["period"])
    base = T.base_chain(c, f, blk, pdead)
    d, pitch, ball = rx.load_set(name, g)
    ctx = T.match_ctx(c, f, base, pdead, d, dd.blur_ball(ball), pitch, cols_for(name, g))
    truth = pm.possession_spells(c["labels"], pm.valid_runs(c["labels"]))
    rows, meta, names = [], [], None
    for a, b, s in gen_spells(pr, gen):
        x = spell_inputs(ctx, pr, a, b, s)
        if names is None:
            names = list(x)
        rows.append([x[k] for k in names])
        team = H if s == 0 else A
        best, ln = 0.0, np.nan
        for ta, tb, k in truth:
            if k == team and tb > a and ta < b:
                v = iou(a, b, ta, tb)
                if v > best:
                    best, ln = v, (tb - ta) / FPS
        meta.append((a, b, team, float(best >= 0.5), ln, best))
    os.makedirs(OUT, exist_ok=True)
    np.savez(path, x=np.array(rows, np.float32).reshape(len(rows), -1),
             meta=np.array(meta, np.float64).reshape(len(meta), 6), names=np.array(names or []),
             fold=fold)
    return path


def build_cmd(args):
    jobs = [(n, g, f, args.net, args.gen) for n, g, f in T.ext_list()]
    with Pool(6) as pool:
        paths = pool.map(build_match, jobs, chunksize=1)
    m = np.concatenate([np.load(p)["meta"] for p in paths])
    print("spells {}, right {:.1%} (of those < 2 s: {}, 2-5 s: {})".format(
        len(m), m[:, 3].mean(), int(np.sum((m[:, 3] == 1) & (m[:, 4] < 2))),
        int(np.sum((m[:, 3] == 1) & (m[:, 4] >= 2)))))


def load(net, gen):
    out = []
    for n, g, f in T.ext_list():
        with np.load(os.path.join(OUT, "{}_g{:g}_{}_{}.npz".format(net, gen, n, g))) as z:
            out.append((n, g, f, z["x"], z["meta"], list(z["names"])))
    return out


def train_cmd(args):
    import xgboost as xgb
    data = [d for d in load(args.net, args.gen) if len(d[4])]
    preds = {}
    for k in range(T.FOLDS):
        tr_ = [d for d in data if d[2] != k]
        bst = T.train_xgb(np.concatenate([d[3] for d in tr_]), np.concatenate([d[4][:, 3] for d in tr_]))
        for d in data:
            if d[2] == k:
                preds["{}_{}".format(d[0], d[1])] = bst.predict(xgb.DMatrix(d[3]))
    bst = T.train_xgb(np.concatenate([d[3] for d in data]), np.concatenate([d[4][:, 3] for d in data]))
    os.makedirs(os.path.join(OUT, "models"), exist_ok=True)
    bst.save_model(os.path.join(OUT, "models", "{}_{}_g{:g}_all.json".format(args.tag, args.net, args.gen)))
    np.savez(os.path.join(OUT, "pred_{}_{}_g{:g}.npz".format(args.tag, args.net, args.gen)), **preds)
    y = np.concatenate([d[4][:, 3] for d in data])
    p = np.concatenate([preds["{}_{}".format(d[0], d[1])] for d in data])
    q = np.concatenate([d[3][:, d[5].index("q_mean")] for d in data])
    print("held-out AUC {:.3f} (net's mean P(inside) alone {:.3f}), base rate {:.3f}".format(
        T.roc_auc_score(y, p), T.roc_auc_score(y, q), y.mean()))
    for thr in (0.3, 0.5, 0.7):
        sel = p >= thr
        print("  keep p >= {:.1f}: kept {:.0%} of spells, precision {:.2f}, recall of right spells {:.2f}"
              .format(thr, sel.mean(), y[sel].mean(), sel[y == 1].mean()))
    imp = bst.get_score(importance_type="gain")
    names = data[0][5]
    print("top inputs (gain): " + ", ".join(names[int(k[1:])] for k, _ in
                                            sorted(imp.items(), key=lambda kv: -kv[1])[:15]))


def refine_edges(pr, a, b, s, n):
    w = int(EDGE_S * FPS)
    lo = max(a - w, 0)
    sh = pr[lo:min(a + w, n), 2 + s]
    if len(sh) and sh.max() >= EDGE_P:
        a2 = lo + int(np.argmax(sh))
    else:
        a2 = a
    lo = max(b - 1 - w, 0)
    eh = pr[lo:min(b - 1 + w, n), 4 + s]
    b2 = lo + int(np.argmax(eh)) + 1 if len(eh) and eh.max() >= EDGE_P else b
    return (a2, b2) if b2 - a2 >= 5 else (a, b)


def eval_match(job):
    name, g, net, gen, tag, settings = job
    import run_v31_external as rx
    import pair_cls as P
    from decoder_sweep import stats
    key = "{}_{}".format(name, g)
    with np.load(os.path.join(S.OUT, "pred_{}.npz".format(net))) as z:
        pr = z[key].astype(np.float32)
    with np.load(os.path.join(OUT, "{}_g{:g}_{}_{}.npz".format(net, gen, name, g))) as z:
        meta = z["meta"]
    with np.load(os.path.join(OUT, "pred_{}_{}_g{:g}.npz".format(tag, net, gen))) as z:
        p = z[key] if key in z.files else np.zeros(0)
    c, f = T.load_cache(name, g)
    pdead = P.pdead_of(name, g)
    blk = rx.blocks(c["period"])
    lab = c["labels"]
    n = len(lab)
    base = T.base_chain(c, f, blk, pdead)
    res = [stats(lab, base)]
    for st in settings:
        out = base.copy()
        ok = (H, A, D) if st["over_d"] else (H, A)
        order = np.argsort(p)
        for i in order:
            if p[i] < st["thr"]:
                continue
            a, b, team = int(meta[i, 0]), int(meta[i, 1]), int(meta[i, 2])
            if st["edges"]:
                a, b = refine_edges(pr, a, b, 0 if team == H else 1, n)
            seg = out[a:b]
            out[a:b] = np.where(np.isin(seg, ok), team, seg)
        res.append(stats(lab, out))
    return name, g, np.array(res, float)


def eval_cmd(args):
    import run_v31_external as rx
    settings = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    jobs = [(n, g, args.net, args.gen, args.tag, settings) for n, g, _ in T.ext_list()]
    with Pool(6) as pool:
        res = pool.map(eval_match, jobs, chunksize=1)
    names = np.array([r[0] for r in res])
    folds = np.array([rx.fold_of(r[0], r[1]) for r in res])
    arr = np.stack([r[2] for r in res])
    gain = arr[:, 1:, 0] - arr[:, :1, 0]
    print("net {}, spells generated at {:g}, check {}".format(args.net, args.gen, args.tag))
    print("{:<26} {:>17} {:>17} | {:>10} {:>10} | {:>9} {:>9}".format(
        "setting", "SkillCorner", "PFF", "caught <2s", "2-5s", "pred/true", "precision"))
    labels = ["base v31_d1"] + ["keep >= {:g}{}{}".format(s["thr"], ", edges" if s["edges"] else "",
                                                          ", over D" if s["over_d"] else "")
                                for s in settings]
    for j, lb in enumerate(labels):
        parts = []
        for nm in ("skillcorner", "pff"):
            v = arr[names == nm, j, 0]
            d = v - arr[names == nm, 0, 0]
            parts.append("{:.4f} {:>3}/{:<3}".format(v.mean(), int(np.sum(d > 1e-9)), int(np.sum(d < -1e-9))))
        t = arr[:, j, 1:].sum(axis=0)
        print("{:<26} {:>17} {:>17} | {:>9.1%} {:>9.1%}  | {:>9.2f} {:>9.1%}".format(
            lb, parts[0], parts[1], t[1] / t[0], t[3] / t[2], t[5] / t[4], t[6] / t[5]))
    nested = np.zeros(len(arr))
    for k in range(T.FOLDS):
        j = int(np.argmax(gain[folds != k].mean(axis=0)))
        nested[folds == k] = gain[folds == k, j]
    print("nested 4-fold: SkillCorner {:+.4f}, PFF {:+.4f} ({}/{})".format(
        nested[names == "skillcorner"].mean(), nested[names == "pff"].mean(),
        int(np.sum(nested > 1e-9)), int(np.sum(nested < -1e-9))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "train", "eval"))
    ap.add_argument("--net", default="s1")
    ap.add_argument("--gen", type=float, default=0.4)
    ap.add_argument("--tag", default="k1")
    args = ap.parse_args()
    {"build": build_cmd, "train": train_cmd, "eval": eval_cmd}[args.cmd](args)


if __name__ == "__main__":
    main()
