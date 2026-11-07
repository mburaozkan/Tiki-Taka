"""Pair proposals for short possessions, trained on SkillCorner + PFF. Aggregate only.

    python tools/pair_cls.py build [--jobs 6]      # pair candidates + inputs per match
    python tools/pair_cls.py train [--tag p1]       # XGBoost GPU, 4 fixed folds + all
    python tools/pair_cls.py eval [--tag p1]        # into the v31_d1 chain, with / without t1

A pair is a whole short spell proposed at once: a touch-network peak of team B (P >= END_P) at t1
while the working base (v31_d1) has the other team A, and one of A's next MAX_ENDS peaks
(P >= END_P) at t2, 0.2-MAX_GAP_S s later. Target: a true possession of B with IoU >= 0.5 to
[t1, t2) (what spellF1's catch needs). Inputs: the single-touch inputs of tools/turnover_cls.py
(v2) at t1 oriented to B and at t2 oriented to A, plus the span: length, touch P of both teams
inside, number of B peaks inside, first stage / refiner / vote / hand D / P(dead) over it, ball
travel and gain toward B's attack, whether the base is back with A after t2.
eval: proposals with P >= p0 (best pair per t1) put b per frame toward B over [t1, t2) (at least
tot in total) on top of the base, optionally together with the t1 single-touch proposals (chosen
setting). spellF1 vs base (SkillCorner, PFF, up / down), nested 4-fold, caught < 2 s and 2-5 s,
and the ceiling: share of true short spells that have a positive pair.
"""

import argparse
import itertools
import os
import sys
from multiprocessing import Pool

os.environ["TC_ROWS"] = "v2"          # per-touch inputs with the interception set
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
import turnover_cls as T  # noqa: E402
from pipeline.decode import H, A, D  # noqa: E402

# Params
END_P = 0.2
MAX_GAP_S = 3.0
MIN_GAP_S = 0.2
MAX_ENDS = 2
IOU = 0.5
OUT = os.path.join(T.OUT, "pairs")
SINGLE_TAG = "t1"
GRID = {"p0": (0.1, 0.15, 0.2, 0.25), "b": (8.0,), "tot": (45.0,), "single": (0, 1)}
CHOSEN = {"p0": 0.2, "b": 8.0, "tot": 45.0, "single": 1}    # 2026-11-07, nested +0.012


def pdead_of(name, g):
    import train_ext_heads as th
    with np.load(os.path.join(ROOT, "data", "cache", "ext_heads_{}.npz".format(T.DEAD_TAG))) as z:
        return th.seeds_mean({k: z[k] for k in z.files if k.startswith(name + "_" + g + "__")},
                             name + "_" + g)


def pair_rows(ctx, lab):
    """(x, meta, names); meta rows: t1, t2, B, label, true spell length (s, nan if none)."""
    c, f, base, n, tp = ctx["c"], ctx["f"], ctx["base"], ctx["n"], ctx["tp"]
    L, W, ball, sign_home = ctx["L"], ctx["W"], ctx["ball"], ctx["sign_home"]
    half = int(round(T.PEAK_HALF_S * T.FPS))
    pk = [T.peaks(tp[:, s], END_P, half) for s in (0, 1)]
    spells = pm.possession_spells(lab, pm.valid_runs(lab)) if lab is not None else []
    by_team = {H: [s for s in spells if s[2] == H], A: [s for s in spells if s[2] == A]}
    lo, hi = int(MIN_GAP_S * T.FPS), int(MAX_GAP_S * T.FPS)
    rows, meta, names = [], [], None
    for b_side in (0, 1):
        B, Aa = (H, A) if b_side == 0 else (A, H)
        a_side = 1 - b_side
        for t1 in pk[b_side]:
            if base[max(t1 - 5, 0)] != Aa:
                continue
            ends = pk[a_side][(pk[a_side] > t1 + lo) & (pk[a_side] <= t1 + hi)][:MAX_ENDS]
            if not len(ends):
                continue
            xs = touch = None
            for t2 in ends:
                if xs is None:
                    xs = {"s_" + k: v for k, v in T.touch_feats(ctx, t1, b_side, max(t1 - 5, 0)).items()}
                xe = {"e_" + k: v for k, v in T.touch_feats(ctx, t2, a_side, max(t2 - 5, 0)).items()}
                x = dict(xs)
                x.update(xe)
                sl = slice(t1 + 1, t2)
                inner = slice(t1 + 3, max(t2 - 3, t1 + 4))
                sg = sign_home[t1] if b_side == 0 else -sign_home[t1]
                pb = ctx["ph1"][sl] if b_side == 0 else 1 - ctx["ph1"][sl]
                lq = np.log(ctx["q"][sl] / (1 - ctx["q"][sl]))
                lq = lq if b_side == 0 else -lq
                vt = c["vote"][sl] if b_side == 0 else -c["vote"][sl]
                x.update({
                    "gap_s": (t2 - t1) / T.FPS,
                    "in_b_pmax": tp[inner, b_side].max(initial=0),
                    "in_a_pmax": tp[inner, a_side].max(initial=0),
                    "in_b_peaks": float(np.sum((pk[b_side] > t1) & (pk[b_side] < t2))),
                    "in_a_peaks": float(np.sum((pk[a_side] > t1) & (pk[a_side] < t2))),
                    "end_rank": float(np.flatnonzero(ends == t2)[0]),
                    "span_p1_mean": float(np.mean(pb)), "span_p1_max": float(np.max(pb)),
                    "span_ref_mean": float(np.mean(lq)), "span_vote_mean": float(np.mean(vt)),
                    "span_handD": float(np.mean(c["hand"][sl] == D)),
                    "span_dead": float(np.mean(ctx["pdead"][sl])),
                    "span_travel": float(np.hypot((ball[t2, 0] - ball[t1, 0]) * L,
                                                  (ball[t2, 1] - ball[t1, 1]) * W)),
                    "span_adv": float((ball[t2, 0] - ball[t1, 0]) * L * sg),
                    "base_a_after": float(base[min(t2 + 10, n - 1)] == Aa),
                    "s_picked": float(f["touch"][t1] != 0), "e_picked": float(f["touch"][t2] != 0),
                })
                if names is None:
                    names = list(x)
                rows.append([x[k] for k in names])
                y, ln = 0, np.nan
                best = 0.0
                for a_, b_, _ in by_team[B]:
                    if b_ > t1 and a_ < t2:
                        iou = (min(b_, t2) - max(a_, t1)) / float(max(b_, t2) - min(a_, t1))
                        if iou > best:
                            best, ln = iou, (b_ - a_) / T.FPS
                y = int(best >= IOU)
                meta.append((t1, t2, B, y, ln))
    return (np.array(rows, dtype=np.float32).reshape(len(rows), -1),
            np.array(meta, dtype=np.float64).reshape(len(meta), 5), names or [])


def build_match(job):
    name, g, fold = job
    path = os.path.join(OUT, "pairs_{}_{}.npz".format(name, g))
    if os.path.isfile(path):
        return path
    import dfl_data as dd
    import run_v31_external as rx
    from ext_stage1 import cols_for
    c, f = T.load_cache(name, g)
    pdead = pdead_of(name, g)
    blk = rx.blocks(c["period"])
    base = T.base_chain(c, f, blk, pdead)
    d, pitch, ball = rx.load_set(name, g)
    ctx = T.match_ctx(c, f, base, pdead, d, dd.blur_ball(ball), pitch, cols_for(name, g))
    x, meta, names = pair_rows(ctx, c["labels"])
    os.makedirs(OUT, exist_ok=True)
    np.savez(path, x=x, meta=meta, names=np.array(names), fold=fold)
    return path


def short_truth(lab):
    """True spells < 2 s and 2-5 s that follow a spell of the other team (team change)."""
    sp = pm.possession_spells(lab, pm.valid_runs(lab))
    out = []
    for p, s in zip(sp, sp[1:]):
        if p[1] == s[0] and p[2] != s[2]:
            ln = (s[1] - s[0]) / T.FPS
            if ln < 5:
                out.append((s, "<2" if ln < 2 else "2-5"))
    return out


def build_cmd(args):
    jobs = T.ext_list()
    with Pool(args.jobs) as pool:
        paths = pool.map(build_match, jobs, chunksize=1)
    tot = pos = 0
    cov = {"<2": [0, 0], "2-5": [0, 0]}
    for (name, g, _), p in zip(jobs, paths):
        with np.load(p) as z:
            m = z["meta"]
        tot += len(m)
        pos += int(m[:, 3].sum()) if len(m) else 0
        c, _ = T.load_cache(name, g)
        good = [(int(r[0]), int(r[1]), int(r[2])) for r in m if r[3] == 1]
        for (a, b, k), cl in short_truth(c["labels"]):
            cov[cl][0] += 1
            if any(kk == k and min(b, t2) - max(a, t1) > 0 and
                   (min(b, t2) - max(a, t1)) / float(max(b, t2) - min(a, t1)) >= IOU
                   for t1, t2, kk in good):
                cov[cl][1] += 1
    print("pairs {}, positive {} ({:.1%})".format(tot, pos, pos / max(tot, 1)))
    print("ceiling: true short spells after a team change with a positive pair: " + ", ".join(
        "{} {}/{} ({:.0%})".format(k, v[1], v[0], v[1] / max(v[0], 1)) for k, v in cov.items()))


def load_pairs():
    out = []
    for name, g, fold in T.ext_list():
        with np.load(os.path.join(OUT, "pairs_{}_{}.npz".format(name, g))) as z:
            if len(z["meta"]):
                out.append((name, g, fold, z["x"], z["meta"], list(z["names"])))
    return out


def train_cmd(args):
    import xgboost as xgb
    data = load_pairs()
    out = {}
    for k in range(T.FOLDS):
        tr_ = [d for d in data if d[2] != k]
        bst = T.train_xgb(np.concatenate([d[3] for d in tr_]),
                          np.concatenate([d[4][:, 3] for d in tr_]))
        for d in data:
            if d[2] == k:
                out["{}_{}".format(d[0], d[1])] = bst.predict(xgb.DMatrix(d[3]))
        print("fold {} done".format(k), flush=True)
    bst = T.train_xgb(np.concatenate([d[3] for d in data]), np.concatenate([d[4][:, 3] for d in data]))
    os.makedirs(os.path.join(OUT, "models"), exist_ok=True)
    bst.save_model(os.path.join(OUT, "models", "{}_all.json".format(args.tag)))
    np.savez(os.path.join(OUT, "pred_{}.npz".format(args.tag)), **out)
    y = np.concatenate([d[4][:, 3] for d in data])
    p = np.concatenate([out["{}_{}".format(d[0], d[1])] for d in data])
    ln = np.concatenate([d[4][:, 4] for d in data])
    print("held-out AUC {:.3f}, AP {:.3f} (base rate {:.3f})".format(
        T.roc_auc_score(y, p), T.average_precision_score(y, p), y.mean()))
    for thr in (0.3, 0.5, 0.7):
        sel = p >= thr
        print("  p >= {:.1f}: precision {:.2f}, recall {:.2f} (true spell < 2 s {:.2f}, 2-5 s {:.2f})".format(
            thr, y[sel].mean() if sel.any() else np.nan, sel[y == 1].mean(),
            sel[(y == 1) & (ln < 2)].mean(), sel[(y == 1) & (ln >= 2)].mean()))
    imp = bst.get_score(importance_type="gain")
    names = data[0][5]
    print("top inputs (gain): " + ", ".join(names[int(k[1:])] for k, _ in
                                            sorted(imp.items(), key=lambda kv: -kv[1])[:15]))


def pair_boost(n, meta, p, p0, b, tot):
    """(n,) home log-odds toward B over [t1, t2) for the best pair per t1 with P >= p0."""
    out = np.zeros(n)
    best = {}
    for (t1, t2, B), pi in zip(meta[:, :3].astype(int), p):
        if pi >= p0 and (t1 not in best or pi > best[t1][0]):
            best[t1] = (pi, t2, B)
    for t1, (pi, t2, B) in best.items():
        vi = b * (pi - p0) / (1.0 - p0)
        if tot > 0:
            vi = max(vi, tot / float(t2 - t1))
        sgn = 1.0 if B == H else -1.0
        seg = out[t1:t2]
        out[t1:t2] = np.where(np.abs(seg) < vi, sgn * vi, seg)
    return out


def eval_match(job):
    name, g, tag, settings = job
    import eval_touch_run as tr
    import run_v31_external as rx
    from ext_stage1 import length_caught
    c, f = T.load_cache(name, g)
    pdead = pdead_of(name, g)
    blk = rx.blocks(c["period"])
    lab = c["labels"]
    n = len(lab)
    with np.load(os.path.join(OUT, "pairs_{}_{}.npz".format(name, g))) as z:
        meta = z["meta"]
    key = "{}_{}".format(name, g)
    with np.load(os.path.join(OUT, "pred_{}.npz".format(tag))) as z:
        p = z[key] if key in z.files else np.zeros(0)
    with np.load(os.path.join(T.OUT, "rows_{}_{}.npz".format(name, g))) as z:
        smeta = z["meta"]
    with np.load(os.path.join(T.OUT, "pred_{}.npz".format(SINGLE_TAG))) as z:
        sp = z[key]
    s0 = T.CHOSEN
    single = T.span_boost(n, smeta, sp, f, s0["b"], s0["p0"], s0["max_s"])
    base = T.base_chain(c, f, blk, pdead)
    res = [(tr.score(lab, base), length_caught(lab, base))]
    for s in settings:
        bst = pair_boost(n, meta, p, s["p0"], s["b"], s["tot"]) if len(p) else np.zeros(n)
        if s["single"]:
            bst = np.where(np.abs(bst) >= np.abs(single), bst, single)
        codes = T.base_chain(c, f, blk, pdead, None, bst)
        res.append((tr.score(lab, codes), length_caught(lab, codes)))
    return name, g, res


def eval_cmd(args):
    import run_v31_external as rx
    settings = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    settings.insert(0, {"p0": 2.0, "b": 8.0, "tot": 0.0, "single": 1})   # t1 alone (reference)
    jobs = [(nm, g, args.tag, settings) for nm, g, _ in T.ext_list()]
    with Pool(args.jobs) as pool:
        res = pool.map(eval_match, jobs, chunksize=1)
    sc = np.array([[r[0] for r in x[2]] for x in res])
    lc = np.array([[np.array(r[1]) for r in x[2]] for x in res])
    names = np.array([x[0] for x in res])
    folds = np.array([rx.fold_of(x[0], x[1]) for x in res])
    gain = sc[:, 1:] - sc[:, :1]

    def caught(col):
        t = lc[:, col].sum(axis=0)
        return "< 2 s {:.1%}, 2-5 s {:.1%}".format(t[0, 1] / max(t[0, 0], 1), t[1, 1] / max(t[1, 0], 1))

    def line(j, label):
        parts = []
        for nm in ("skillcorner", "pff"):
            gg = gain[names == nm, j]
            parts.append("{} {:+.4f} ({}/{})".format(nm[:5], gg.mean(), int(np.sum(gg > 1e-9)),
                                                     int(np.sum(gg < -1e-9))))
        print("{:<36} {} | all {:+.4f} | caught {}".format(label, "  ".join(parts),
                                                            gain[:, j].mean(), caught(j + 1)))

    print("base v31_d1: SkillCorner {:.4f}, PFF {:.4f} | caught {}".format(
        sc[names == "skillcorner", 0].mean(), sc[names == "pff", 0].mean(), caught(0)))
    line(0, "t1 single proposals only (candidate)")
    order = np.argsort(-gain[:, 1:].mean(axis=0)) + 1
    for j in order[:10]:
        s = settings[j]
        line(j, "pairs p0 {:g} tot {:g}{}".format(s["p0"], s["tot"], " + t1" if s["single"] else ""))
    for single in (0, 1):
        idx = [j for j in range(1, len(settings)) if settings[j]["single"] == single]
        nested = np.zeros(len(sc))
        for k in range(T.FOLDS):
            j = idx[int(np.argmax(gain[folds != k][:, idx].mean(axis=0)))]
            nested[folds == k] = gain[folds == k, j]
        print("nested 4-fold, pairs{}: {:+.4f} ({}/{})".format(
            " + t1" if single else " alone", nested.mean(), int(np.sum(nested > 1e-9)),
            int(np.sum(nested < -1e-9))))


def ours_cmd(args):
    """Our match (MTAH_VARIANT=_kt): v31_d1 + t1 single proposals + pair proposals (CHOSEN)."""
    import xgboost as xgb
    import dfl_data as dd
    import ours_chain as oc
    import in_domain_team as idt
    from atd_cache import load_arrays
    from pipeline.features import LENGTH, WIDTH
    codes, ex = oc.ours_v31(dead_prefix=os.path.join("models", "dead_ext_d1"))
    f = {k[2:]: v for k, v in ex.items() if k.startswith("f_")}
    blk = [tuple(b) for b in ex["blk"]]
    n = len(codes)
    period = np.zeros(n, dtype=np.int8)
    for i, (a, b) in enumerate(blk, start=1):
        period[a:b] = i
    c = dict(ex, period=period)
    base = T.base_chain(c, f, blk, ex["pdead"])
    print("base_chain equals ours_v31: {}".format(np.array_equal(base, codes)))
    atd = load_arrays()
    cols = idt.player_columns(atd["players"], atd["team"], period, (LENGTH, WIDTH))
    d = {"players": atd["players"], "team": atd["team"], "period": period}
    ball = dd.blur_ball(atd["ball"])
    # t1 single proposals (original inputs)
    T.V2 = False
    xs, smeta, _ = T.match_rows(c, f, base, ex["pdead"], d, ball, (LENGTH, WIDTH), cols)
    b1 = xgb.Booster()
    b1.load_model(os.path.join(T.OUT, "models", "{}_all.json".format(SINGLE_TAG)))
    sp = b1.predict(xgb.DMatrix(xs))
    s0 = T.CHOSEN
    single = T.span_boost(n, smeta, sp, f, s0["b"], s0["p0"], s0["max_s"])
    # pairs (v2 inputs)
    T.V2 = True
    ctx = T.match_ctx(c, f, base, ex["pdead"], d, ball, (LENGTH, WIDTH), cols)
    x, meta, _ = pair_rows(ctx, None)
    b2 = xgb.Booster()
    b2.load_model(os.path.join(OUT, "models", "{}_all.json".format(args.tag)))
    p = b2.predict(xgb.DMatrix(x))
    s = CHOSEN
    pb = pair_boost(n, meta, p, s["p0"], s["b"], s["tot"])
    both = np.where(np.abs(pb) >= np.abs(single), pb, single)
    t1 = T.base_chain(c, f, blk, ex["pdead"], None, single)
    new = T.base_chain(c, f, blk, ex["pdead"], None, both)
    print("single proposals {} of {}; pairs {} (P >= p0: {}); frames changed vs base {:.2f}%".format(
        int(np.sum(sp >= s0["p0"])), len(sp), len(p), int(np.sum(p >= s["p0"])),
        100 * np.mean(new != base)))
    for nm, cd in (("base", base), ("t1", t1), ("pairs + t1", new)):
        sp_ = pm.possession_spells(cd, pm.valid_runs(cd))
        print("{}: {} possessions".format(nm, len(sp_)))
    if args.name:
        print(oc.write_and_score(new, args.name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "train", "eval", "ours"))
    ap.add_argument("--name")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--tag", default="p1")
    args = ap.parse_args()
    {"build": build_cmd, "train": train_cmd, "eval": eval_cmd, "ours": ours_cmd}[args.cmd](args)


if __name__ == "__main__":
    main()
