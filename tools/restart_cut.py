"""End our dead-ball stretches at the change net's restart signal (DFL loss #1). Aggregate only.

    python tools/restart_cut.py dfl      # 7 DFL matches, held out (cache from tools/cn_e3k.py dfl)
    python tools/restart_cut.py ext      # 84 external matches (held-out change-net predictions)
    python tools/restart_cut.py ours --p 0.3 [--touch]   # our match: public window, writes the file

On DFL 39% of the possessions the submission misses are called D, 56% of those right after a dead
ball: the stoppage runs on through the first (short) possession after the restart. Rule, applied to
the submission's output (v31_d1 + change-net mix c1chdl + e3k): in each stretch we call D, the frame t
where the change net's restart head toward team k (k = H, A) peaks; if that peak >= P (and, with
--touch, a touch-network peak of team k lies within +-0.5 s of t), frames [t, end of the D stretch)
become k. Stretches shorter than MIN_D_S are left alone, and t must leave >= MIN_KEEP_S of D before it.
Grid P x touch; spellF1 vs the submission (up / down), short catches; nested 4-fold on the external sets.
"""
import argparse
import glob
import itertools
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
os.environ.setdefault("TC_ROWS", "v2")
import possession_metrics as pm  # noqa: E402
from pipeline.decode import H, A, D  # noqa: E402

FPS = 25
MIN_D_S = 1.0
MIN_KEEP_S = 0.4
P_GRID = (0.2, 0.3, 0.4, 0.5, 0.6)
TOUCH_GRID = (0, 1)


def cut(codes, cpr, p_min, touch_frames=None):
    """codes with each D stretch ended at a strong restart-head peak (see module doc).
    touch_frames: (frames, teams) of touch-network peaks, or None for no touch check."""
    out = codes.copy()
    isd = (codes == D).astype(np.int8)
    e = np.flatnonzero(np.diff(np.concatenate(([0], isd, [0]))))
    keep = int(MIN_KEEP_S * FPS)
    for a, b in zip(e[::2], e[1::2]):
        if b - a < MIN_D_S * FPS or b - a <= keep + 2:
            continue
        lo = a + keep
        best = None
        for k, team in ((0, H), (1, A)):
            seg = cpr[lo:b, 3 + k]
            j = int(np.argmax(seg))
            v = float(seg[j])
            if v < p_min:
                continue
            t = lo + j
            if touch_frames is not None:
                tf, tt = touch_frames
                if not np.any((np.abs(tf - t) <= 12) & (tt == k)):
                    continue
            if best is None or v > best[0]:
                best = (v, t, team)
        if best is not None:
            out[best[1]:b] = best[2]
    return out


def settings():
    return list(itertools.product(P_GRID, TOUCH_GRID))


def score_all(lab, final, cpr, tf):
    from decoder_sweep import stats
    res = [stats(lab, final)]
    for p, tch in settings():
        res.append(stats(lab, cut(final, cpr, p, tf if tch else None)))
    return np.array(res, float)


def table(rows, sets, nested_folds=None):
    names = np.array([r[0] for r in rows])
    arr = np.stack([r[2] for r in rows])
    labels = ["submission"] + ["restart cut P {:g}{}".format(p, ", touch" if t else "") for p, t in settings()]
    print("{:<26} ".format("setting") + " ".join("{:>18}".format(s) for s in sets)
          + " | {:>10} {:>9} | {:>9}".format("caught <2s", "2-5s", "pred/true"))
    for j, lb in enumerate(labels):
        parts = []
        for s in sets:
            v = arr[names == s, j, 0]
            d = v - arr[names == s, 0, 0]
            parts.append("{:.4f} {:>3}/{:<3}".format(v.mean(), int(np.sum(d > 1e-9)), int(np.sum(d < -1e-9))))
        t = arr[:, j, 1:].sum(axis=0)
        print("{:<26} ".format(lb) + " ".join("{:>18}".format(p) for p in parts)
              + " | {:>9.1%} {:>9.1%} | {:>9.2f}".format(t[1] / t[0], t[3] / t[2], t[5] / t[4]))
    if nested_folds is not None:
        import turnover_cls as T
        gain = arr[:, 1:, 0] - arr[:, :1, 0]
        nested = np.zeros(len(arr))
        for k in range(T.FOLDS):
            j = int(np.argmax(gain[nested_folds != k].mean(axis=0)))
            nested[nested_folds == k] = gain[nested_folds == k, j]
        print("nested 4-fold vs submission: " + ", ".join("{} {:+.4f}".format(s, nested[names == s].mean()) for s in sets)
              + " ({}/{})".format(int(np.sum(nested > 1e-9)), int(np.sum(nested < -1e-9))))


def dfl_cmd(args):
    from pipeline import touch_net as tn
    rows = []
    for fp in sorted(glob.glob(os.path.join(ROOT, "data", "cache", "cn_e3k", "dfl_*.npz"))):
        z = np.load(fp)
        tp = np.column_stack([1 - z["tph"].astype(np.float32) - z["tpa"].astype(np.float32),
                              z["tph"].astype(np.float32), z["tpa"].astype(np.float32)])
        tf = tn.pick(tp, 0.5)
        rows.append(("dfl", fp, score_all(z["lab"], z["final"], z["cpr"].astype(np.float32), tf)))
    table(rows, ("dfl",))


def ext_match(job):
    name, g, tag = job
    import turnover_cls as T
    import run_v31_external as rx
    import pair_cls as P
    import change_net as C
    import cn_e3k as CE
    from pipeline import touch_net as tn
    c, f = T.load_cache(name, g)
    pdead = P.pdead_of(name, g)
    blk = rx.blocks(c["period"])
    base = T.base_chain(c, f, blk, pdead)
    with np.load(os.path.join(C.OUT, "pred_{}.npz".format(tag))) as z:
        cpr = z["{}_{}".format(name, g)].astype(np.float32)
    final = CE.paste(CE.mixed(cpr, base, blk), CE.ext_spells(name, g, len(base)))
    tp = np.column_stack([1 - f["touch_p_home"] - f["touch_p_away"], f["touch_p_home"], f["touch_p_away"]])
    return name, g, score_all(c["labels"], final, cpr, tn.pick(tp, 0.5))


def ext_cmd(args):
    import turnover_cls as T
    import run_v31_external as rx
    with Pool(args.jobs) as pool:
        rows = pool.map(ext_match, [(n, g, args.tag) for n, g, _ in T.ext_list()], chunksize=1)
    table(rows, ("skillcorner", "pff"), np.array([rx.fold_of(r[0], r[1]) for r in rows]))


def ours_cmd(args):
    import change_net as C
    import cn_e3k as CE
    import short_net as S
    import ours_chain as oc
    from pipeline import touch_net as tn
    z = np.load(os.path.join(C.OUT, "ours_{}.npz".format(args.tag)))
    cpr, base, blk = z["pr"].astype(np.float32), z["base"], [tuple(b) for b in z["blk"]]
    sp = [tuple(int(v) for v in r) for r in np.load(os.path.join(S.OUT, "ours_spells_{}.npy".format(args.spells)))]
    final = CE.paste(CE.mixed(cpr, base, blk), sp)
    tf = None
    if args.touch:
        x = z["x"].astype(np.float32)
        names = S.channel_names()
        th, ta = x[:, names.index("h_touch_p")], x[:, names.index("a_touch_p")]
        tf = tn.pick(np.column_stack([1 - th - ta, th, ta]), 0.5)
    print("  " + str(oc.write_and_score(final, "v31_d1_c1chdlm_e3k_chk")))
    out = cut(final, cpr, args.p, tf)
    print("  frames changed {:.2%}".format(np.mean(out != final)))
    print("  " + str(oc.write_and_score(out, "v31_d1_c1chdlm_e3k_rc{:g}{}".format(args.p, "t" if args.touch else "").replace(".", ""))))


# ---- fix 2: accept signalled turnovers (DFL loss #2: 'other team', 31% of misses, 67% with evidence)
TURN_P = (0.3, 0.4, 0.5, 0.6)
TURN_END = (0.2, 0.3)
CAP_S = 10.0


def turn(codes, cpr, tf, p_min, p_end, cap_s=CAP_S):
    """inside each H / A stretch of team X: where the turnover head toward Y peaks >= p_min with a touch
    peak of Y within +-0.5 s, Y from there until the turnover head back to X peaks >= p_end, or our next
    change, whichever comes first; skipped if neither comes within cap_s."""
    out = codes.copy()
    n = len(codes)
    ftf, ftt = tf
    order = np.argsort(ftf)
    ftf, ftt = ftf[order], ftt[order]
    cap = int(cap_s * FPS)
    i = 0
    while i < len(ftf):
        t, k = int(ftf[i]), int(ftt[i])
        i += 1
        y = H if k == 0 else A
        x = A if y == H else H
        if t <= 0 or t >= n or out[t] != x:
            continue
        lo, hi = max(t - 12, 0), min(t + 13, n)
        j = lo + int(np.argmax(cpr[lo:hi, k]))
        if cpr[j, k] < p_min or out[j] != x:
            continue
        nxt = j + 1
        while nxt < n and out[nxt] == x and nxt - j <= cap:
            nxt += 1
        back = cpr[j + 5:min(nxt, j + cap), 1 - k]
        end = None
        if len(back) and back.max() >= p_end:
            end = j + 5 + int(np.argmax(back))
        elif nxt - j <= cap:
            end = nxt
        if end is None or end - j < 5:
            continue
        out[j:end] = y
    return out


def turn_all(lab, final, cpr, tf):
    from decoder_sweep import stats
    res = [stats(lab, final)]
    for p, pe in itertools.product(TURN_P, TURN_END):
        res.append(stats(lab, turn(final, cpr, tf, p, pe)))
    return np.array(res, float)


def turn_table(rows, sets, folds=None):
    names = np.array([r[0] for r in rows])
    arr = np.stack([r[2] for r in rows])
    labels = ["submission"] + ["turnover P {:g}, end {:g}".format(p, pe) for p, pe in itertools.product(TURN_P, TURN_END)]
    for j, lb in enumerate(labels):
        parts = []
        for s_ in sets:
            v = arr[names == s_, j, 0]
            d = v - arr[names == s_, 0, 0]
            parts.append("{:.4f} {:>3}/{:<3}".format(v.mean(), int(np.sum(d > 1e-9)), int(np.sum(d < -1e-9))))
        t = arr[:, j, 1:].sum(axis=0)
        print("{:<26} ".format(lb) + " ".join("{:>18}".format(q) for q in parts)
              + " | caught <2s {:>5.1%} 2-5s {:>5.1%} | pred/true {:.2f}".format(t[1] / t[0], t[3] / t[2], t[5] / t[4]))
    if folds is not None:
        import turnover_cls as T
        gain = arr[:, 1:, 0] - arr[:, :1, 0]
        nested = np.zeros(len(arr))
        for kf in range(T.FOLDS):
            jb = int(np.argmax(gain[folds != kf].mean(axis=0)))
            nested[folds == kf] = gain[folds == kf, jb]
        print("nested 4-fold vs submission: " + ", ".join("{} {:+.4f}".format(s_, nested[names == s_].mean()) for s_ in sets)
              + " ({}/{})".format(int(np.sum(nested > 1e-9)), int(np.sum(nested < -1e-9))))


def turn_dfl_cmd(args):
    from pipeline import touch_net as tn
    rows = []
    for fp in sorted(glob.glob(os.path.join(ROOT, "data", "cache", "cn_e3k", "dfl_*.npz"))):
        z = np.load(fp)
        tp = np.column_stack([1 - z["tph"].astype(np.float32) - z["tpa"].astype(np.float32),
                              z["tph"].astype(np.float32), z["tpa"].astype(np.float32)])
        rows.append(("dfl", fp, turn_all(z["lab"], z["final"], z["cpr"].astype(np.float32), tn.pick(tp, 0.5))))
    turn_table(rows, ("dfl",))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("dfl", "ext", "ours", "turn-dfl"))
    ap.add_argument("--tag", default="c1chdl")
    ap.add_argument("--spells", default="v31_d1_e3k_rb")
    ap.add_argument("--p", type=float, default=0.3)
    ap.add_argument("--touch", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    {"dfl": dfl_cmd, "ext": ext_cmd, "ours": ours_cmd, "turn-dfl": turn_dfl_cmd}[args.cmd](args)


if __name__ == "__main__":
    main()
