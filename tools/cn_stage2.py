"""Second-stage judge for the turnover peaks the decoder did not take. Aggregate only.

    python tools/cn_stage2.py [--tag c1dlbd]

Candidates: local maxima (+-6 frames) of the change net's P(turnover to H) / P(turnover to A) >= 0.2
with no decoded turnover of that direction within 1 s. Label: a true turnover of that direction
within 12 frames. Features: peak height (and logit), class head before / after (new team over the
next 1 s and 1-3 s, old team over the last 1 s, D around), the decoded state before, the strongest
opposite / same-direction peak within 3 s and its offset, ball speed / height / seen, nearest
distances and touch-net peaks of both teams. Logistic model (L2), always fitted without the match
it is applied to: DFL leave one match out (SkillCorner + PFF + the other 6 DFL matches).
An accepted candidate writes the new team from its frame until the next decoded change, or until an
opposite peak (>= 0.2, within 3 s) after which the class head favours the old team.
Prints: candidate counts, AUC of peak height vs the judge, DFL spellF1 vs the decoded net for a grid
of acceptance thresholds, and nested (threshold picked on the training matches' DFL part).
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
import change_net as C  # noqa: E402
import short_net as S  # noqa: E402
import miss_profile as M  # noqa: E402
from turnover_signals import auc, logit_fit  # noqa: E402
from pipeline.decode import H, A, D  # noqa: E402

FPS = 25
THR = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def match_data(name, g, preds):
    import cn_dfl
    if name == "dfl":
        with np.load(cn_dfl.path(g)) as z:
            lab, blk, x = z["lab"], [tuple(b) for b in z["blk"]], z["x"].astype(np.float32)
    else:
        import turnover_cls as T
        import run_v31_external as rx
        c, _ = T.load_cache(name, g)
        lab, blk = c["labels"], rx.blocks(c["period"])
        x = np.load(os.path.join(S.OUT, "in_{}_{}.npy".format(name, g))).astype(np.float32)
    return lab, blk, x, preds["{}_{}".format(name, g)].astype(np.float32)


def candidates(lab, blk, x, pr):
    from scipy.ndimage import maximum_filter1d
    n = len(lab)
    dec = C.viterbi(pr, blk, 0.5, -3.0)
    runs = pm.valid_runs(lab)
    filled = pm.fill_abstentions(dec, runs)
    names = S.channel_names()
    col = {k: names.index(k) for k in names if k in names}
    tch = [(f, filled[f]) for f, k in M.changes(filled, runs) if k == 0]
    tru = [(f, lab[f]) for f, k in M.changes(lab, runs) if k == 0]
    rows, meta = [], []
    for k, new in ((0, H), (1, A)):
        p = pr[:, k]
        q = pr[:, 1 - k]
        mx = maximum_filter1d(p, 13)
        peaks = np.flatnonzero((p >= 0.2) & (p == mx))
        taken = np.array([f for f, s in tch if s == new])
        truth = np.array([f for f, s in tru if s == new])
        old = A if new == H else H
        cn, co = C.N_TGT + (0 if new == H else 1), C.N_TGT + (1 if new == H else 0)
        tn, to = ("h_", "a_") if new == H else ("a_", "h_")
        for t in peaks:
            if len(taken) and np.min(np.abs(taken - t)) <= FPS:
                continue
            y = bool(len(truth) and np.min(np.abs(truth - t)) <= 12)
            w1, w3, wb = slice(t, min(t + 25, n)), slice(min(t + 25, n - 1), min(t + 75, n)), slice(max(t - 25, 0), t)
            fw = slice(t + 1, min(t + 76, n))
            opp = q[fw]
            jo = int(np.argmax(opp)) if len(opp) else 0
            same = p[max(t - 75, 0):max(t - 6, 0)]
            near = slice(max(t - 5, 0), min(t + 6, n))
            wide = slice(max(t - 12, 0), min(t + 13, n))
            st = filled[t - 1] if t > 0 else D
            rows.append([p[t], np.log(p[t] / (1 - p[t] + 1e-6)),
                         pr[w1, cn].mean(), pr[w3, cn].mean() if w3.stop > w3.start else 0.5,
                         pr[wb, co].mean() if t > 0 else 0.5, pr[wide, C.N_TGT + 2].mean(),
                         float(st == old), float(st == D),
                         opp.max() if len(opp) else 0.0, (jo + 1) / FPS if len(opp) else 3.0,
                         same.max() if len(same) else 0.0,
                         x[near, col["bspd"]].max(), x[wide, col["bz"]].max(), x[wide, col["seen"]].mean(),
                         x[near, col[tn + "d1"]].min(), x[near, col[to + "d1"]].min(),
                         x[wide, col[tn + "touch"]].max(), x[wide, col[to + "touch"]].max()])
            meta.append((t, new, y, (t + 1 + jo) if len(opp) and opp.max() >= 0.2 else -1))
    return filled, np.array(rows, np.float64).reshape(-1, 18), meta


def apply(filled, pr, meta, score, thr, runs):
    out = filled.copy()
    chg = np.array([f for f, _ in M.changes(filled, runs)] + [len(filled)])
    order = np.argsort([m[0] for m in meta])
    for i in order:
        t, new, _, ret = meta[i]
        if score[i] < thr or out[t] == new:
            continue
        e = chg[chg > t][0] if np.any(chg > t) else len(out)
        if ret > 0 and ret < e:
            co = C.N_TGT + (1 if new == H else 0)
            cn = C.N_TGT + (0 if new == H else 1)
            after = pr[ret:min(ret + 25, len(pr))]
            if len(after) and after[:, co].mean() > after[:, cn].mean():
                e = ret
        seg = out[t:e]
        out[t:e] = np.where(seg == D, D, new)
    return out


def main():
    import turnover_cls as T
    import dfl_data as dd
    import eval_touch_run as tr
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="c1dlbd")
    args = ap.parse_args()
    with np.load(os.path.join(C.OUT, "pred_{}.npz".format(args.tag))) as z:
        preds = {k: z[k] for k in z.files}
    keys = [(n, g) for n, g, _ in T.ext_list()] + [("dfl", g) for g in dd.GAMES]
    data = {}
    for n, g in keys:
        lab, blk, x, pr = match_data(n, g, preds)
        filled, X, meta = candidates(lab, blk, x, pr)
        data[(n, g)] = (lab, blk, pr, filled, X, meta)
    allX = np.concatenate([v[4] for v in data.values()])
    ally = np.concatenate([[m[2] for m in v[5]] for v in data.values()]).astype(bool)
    print("{} candidates ({:.1%} true), DFL {} ({:.1%} true)".format(
        len(allX), ally.mean(), sum(len(data[("dfl", g)][5]) for g in dd.GAMES),
        np.mean([m[2] for g in dd.GAMES for m in data[("dfl", g)][5]])))
    res = {thr: [] for thr in THR}
    base, aucs = [], []
    for g in dd.GAMES:
        tr_keys = [k for k in keys if k != ("dfl", g)]
        Xt = np.concatenate([data[k][4] for k in tr_keys])
        yt = np.concatenate([[m[2] for m in data[k][5]] for k in tr_keys]).astype(bool)
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
        w = logit_fit((Xt - mu) / sd, yt)
        lab, blk, pr, filled, X, meta = data[("dfl", g)]
        s = 1 / (1 + np.exp(-(((X - mu) / sd) @ w[1:] + w[0])))
        y = np.array([m[2] for m in meta], bool)
        aucs.append((auc(y, X[:, 0]), auc(y, s), len(y), y.sum()))
        runs = pm.valid_runs(lab)
        base.append(tr.score(lab, filled))
        for thr in THR:
            res[thr].append(tr.score(lab, apply(filled, pr, meta, s, thr, runs)))
    a = np.array(aucs)
    print("DFL candidates per match {:.0f} ({:.0f} true); AUC peak height {:.3f}, judge {:.3f}".format(
        a[:, 2].mean(), a[:, 3].mean(), np.nanmean(a[:, 0]), np.nanmean(a[:, 1])))
    b = np.array(base)
    print("DFL {} decoded: {:.4f}".format(args.tag, b.mean()))
    for thr in THR:
        v = np.array(res[thr])
        print("  judge >= {:.1f}: {:.4f} ({:+.4f}, {}/{} up/down)".format(
            thr, v.mean(), v.mean() - b.mean(), int(np.sum(v > b + 1e-9)), int(np.sum(v < b - 1e-9))))
    R = np.array([res[t] for t in THR])                 # (thr, 7)
    nest = []
    for k in range(len(dd.GAMES)):
        oth = [j for j in range(len(dd.GAMES)) if j != k]
        nest.append(R[int(np.argmax(R[:, oth].mean(1))), k])
    v = np.array(nest)
    print("nested (threshold picked on the other 6): {:.4f} ({:+.4f}, {}/{})".format(
        v.mean(), v.mean() - b.mean(), int(np.sum(v > b + 1e-9)), int(np.sum(v < b - 1e-9))))


if __name__ == "__main__":
    main()
