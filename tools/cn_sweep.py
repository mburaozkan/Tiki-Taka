"""Change-net variants x mix weight, each + e3k: a sweep without training. Aggregate only.

    python tools/cn_sweep.py ext  [--tags c1dl c1dlbd c1chdl] [--alphas 0.3 0.4 0.5 0.6]
    python tools/cn_sweep.py dfl  [...]     # needs data/cache/cn_e3k/dfl_*.npz (tools/cn_e3k.py dfl)
    python tools/cn_sweep.py ours [...]     # public window; writes v31_d1_<tag>m<alpha>[b<beta>]_e3k
    --betas 0 0.5 1   base's own changes added to the change heads (change_net.mix_pr beta), so
                      confident base turnovers survive the mix

Same construction as tools/cn_e3k.py (base v31_d1, change_net.mix_pr alpha, beta 0, viterbi w 1 bias -3,
then e3k's kept spells pasted; beta 0 unless --betas). alpha = weight on the base's codes (0.5 = the submission for c1chdl).
Held-out change-net predictions everywhere (external folds; DFL keys of the tag's pred file, else
the all-84 model never saw DFL). Reports spellF1 per set vs the submission (c1chdl 0.5 + e3k).
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
os.environ.setdefault("TC_ROWS", "v2")


def combos(args):
    return [(t, a, b) for t in args.tags for a in args.alphas for b in args.betas]


def label(t, a, b):
    return "{} alpha {:g}{} + e3k".format(t, a, " beta {:g}".format(b) if b else "")


def ext_match(job):
    name, g, cmb = job
    import turnover_cls as T
    import run_v31_external as rx
    import pair_cls as P
    import change_net as C
    import cn_e3k as CE
    from decoder_sweep import stats
    c, f = T.load_cache(name, g)
    pdead = P.pdead_of(name, g)
    blk = rx.blocks(c["period"])
    base = T.base_chain(c, f, blk, pdead)
    lab = c["labels"]
    sp = CE.ext_spells(name, g, len(base))
    preds = {}
    for t in {t for t, _, _ in cmb} | {"c1chdl"}:
        with np.load(os.path.join(C.OUT, "pred_{}.npz".format(t))) as z:
            preds[t] = z["{}_{}".format(name, g)].astype(np.float32)
    res = [stats(lab, CE.paste(CE.mixed(preds["c1chdl"], base, blk, 0.5), sp))]
    for t, a, b in cmb:
        res.append(stats(lab, CE.paste(CE.mixed(preds[t], base, blk, a, b), sp)))
    return name, g, np.array(res, float)


def show(rows, sets, cmb, folds=None):
    names = np.array([r[0] for r in rows])
    arr = np.stack([r[2] for r in rows])
    labels = ["submission (c1chdl 0.5)"] + [label(*c) for c in cmb]
    for j, lb in enumerate(labels):
        parts = []
        for s in sets:
            v = arr[names == s, j, 0]
            d = v - arr[names == s, 0, 0]
            parts.append("{:.4f} {:>3}/{:<3}".format(v.mean(), int(np.sum(d > 1e-9)), int(np.sum(d < -1e-9))))
        t = arr[:, j, 1:].sum(axis=0)
        print("{:<34} ".format(lb) + " ".join("{:>18}".format(p) for p in parts)
              + " | caught <2s {:>5.1%} 2-5s {:>5.1%} | pred/true {:.2f}".format(t[1] / t[0], t[3] / t[2], t[5] / t[4]),
              flush=True)


def ext_cmd(args):
    import turnover_cls as T
    cmb = combos(args)
    with Pool(args.jobs) as pool:
        rows = pool.map(ext_match, [(n, g, cmb) for n, g, _ in T.ext_list()], chunksize=1)
    show(rows, ("skillcorner", "pff"), cmb)


def dfl_final(g, tag, alpha, beta=0.0):
    """DFL game g: (z, change-net pr, output) for tag / alpha / beta + e3k, as dfl_cmd (held out)."""
    import cn_dfl
    import cn_e3k as CE
    z = dict(np.load(os.path.join(ROOT, "data", "cache", "cn_e3k", "dfl_{}.npz".format(g))))
    zc = dict(np.load(cn_dfl.path(g)))
    base, blk = z["base"], [tuple(b) for b in zc["blk"]]
    mix_sub = CE.mixed(z["cpr"].astype(np.float32), base, blk, 0.5)
    diff = z["final"] != mix_sub
    e = np.flatnonzero(np.diff(np.concatenate(([0], diff.astype(np.int8), [0]))))
    sp = [(a, b, int(z["final"][a])) for a, b in zip(e[::2], e[1::2])]
    pr = cn_dfl.heldout(tag, g)
    if pr is None:
        pr = cn_dfl.predict_tag(tag, zc["x"])
    return z, pr, CE.paste(CE.mixed(pr, base, blk, alpha, beta), sp)


def dfl_cmd(args):
    import cn_dfl
    import cn_e3k as CE
    import dfl_data as dd
    from decoder_sweep import stats
    cmb = combos(args)
    rows = []
    for g in dd.GAMES:
        z = np.load(os.path.join(ROOT, "data", "cache", "cn_e3k", "dfl_{}.npz".format(g)))
        zc = dict(np.load(cn_dfl.path(g)))
        lab, base, blk = z["lab"], z["base"], [tuple(b) for b in zc["blk"]]
        final = z["final"]
        mix_sub = CE.mixed(z["cpr"].astype(np.float32), base, blk, 0.5)
        # e3k's spells are the frames where final differs from the submission's mix, pasted the same way:
        # recover them as the paste of runs of (final != mix_sub) with final's team
        diff = final != mix_sub
        e = np.flatnonzero(np.diff(np.concatenate(([0], diff.astype(np.int8), [0]))))
        sp = [(a, b, int(final[a])) for a, b in zip(e[::2], e[1::2])]
        res = [stats(lab, final)]
        prs = {}
        for t, a, b in cmb:
            if t not in prs:
                prs[t] = cn_dfl.heldout(t, g)
                if prs[t] is None:
                    prs[t] = cn_dfl.predict_tag(t, zc["x"])
            res.append(stats(lab, CE.paste(CE.mixed(prs[t], base, blk, a, b), sp)))
        rows.append(("dfl", g, np.array(res, float)))
    show(rows, ("dfl",), cmb)


def ours_cmd(args):
    import change_net as C
    import cn_e3k as CE
    import short_net as S
    import ours_chain as oc
    sp = [tuple(int(v) for v in r) for r in np.load(os.path.join(S.OUT, "ours_spells_v31_d1_e3k_rb.npy"))]
    for t, a, b in combos(args):
        p = os.path.join(C.OUT, "ours_{}.npz".format(t))
        if not os.path.isfile(p):
            print("  no ours_{}.npz".format(t))
            continue
        z = np.load(p)
        pr, base, blk = z["pr"].astype(np.float32), z["base"], [tuple(b) for b in z["blk"]]
        out = CE.paste(CE.mixed(pr, base, blk, a, b), sp)
        nm = "v31_d1_{}m{:g}{}_e3k".format(t, a, "b{:g}".format(b) if b else "").replace(".", "")
        print("  " + str(oc.write_and_score(out, nm)), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("ext", "dfl", "ours"))
    ap.add_argument("--tags", nargs="+", default=["c1dl", "c1dlbd", "c1chdl"])
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.3, 0.4, 0.5, 0.6])
    ap.add_argument("--betas", nargs="+", type=float, default=[0.0])
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    {"ext": ext_cmd, "dfl": dfl_cmd, "ours": ours_cmd}[args.cmd](args)


if __name__ == "__main__":
    main()
