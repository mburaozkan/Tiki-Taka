"""Pair proposals written straight over the base, no decoder; plus the perfect-classifier oracle.
Aggregate only.

    python tools/pair_paste.py [--tag p1]

On the 84 external matches: base v31_d1, then the best pair per t1 with P >= p0 written as B over
[t1, t2) (live frames only, or also over D), optionally with the t1 single proposals pasted too.
Oracle: every pair whose label is 1 (a perfect pair classifier). Per setting: spellF1, caught < 2 s
and 2-5 s, predicted / true possessions, precision of predicted possessions. Aggregates only.
"""

import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pair_cls as P  # noqa: E402
import turnover_cls as T  # noqa: E402
from decoder_sweep import stats, paste as paste_single  # noqa: E402
from pipeline.decode import H, A, D  # noqa: E402

SETTINGS = [("oracle", None, False), ("oracle, also over D", None, True)] + \
    [("p >= {:g}".format(p0), p0, False) for p0 in (0.05, 0.1, 0.2, 0.3, 0.5)] + \
    [("p >= {:g}, also over D".format(p0), p0, True) for p0 in (0.1, 0.2)]


def paste_pairs(base, meta, p, p0, over_d):
    out = base.copy()
    best = {}
    for (t1, t2, B), pi, y in zip(meta[:, :3].astype(int), p, meta[:, 3]):
        score = y if p0 is None else pi
        if (p0 is None and y == 1) or (p0 is not None and pi >= p0):
            if t1 not in best or score > best[t1][0]:
                best[t1] = (score, t2, B)
    ok = (H, A, D) if over_d else (H, A)
    for t1, (_, t2, B) in best.items():
        seg = out[t1:t2]
        out[t1:t2] = np.where(np.isin(seg, ok), B, seg)
    return out


def run(job):
    name, g, tag = job
    import run_v31_external as rx
    c, f = T.load_cache(name, g)
    pdead = P.pdead_of(name, g)
    blk = rx.blocks(c["period"])
    lab = c["labels"]
    base = T.base_chain(c, f, blk, pdead)
    with np.load(os.path.join(P.OUT, "pairs_{}_{}.npz".format(name, g))) as z:
        meta = z["meta"]
    key = "{}_{}".format(name, g)
    with np.load(os.path.join(P.OUT, "pred_{}.npz".format(tag))) as z:
        p = z[key]
    res = [stats(lab, base)]
    for _, p0, over_d in SETTINGS:
        res.append(stats(lab, paste_pairs(base, meta, p, p0, over_d)))
    return name, np.array(res, dtype=float)


def main():
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "p1"
    jobs = [(nm, g, tag) for nm, g, _ in T.ext_list()]
    with Pool(6) as pool:
        res = pool.map(run, jobs, chunksize=1)
    names = np.array([r[0] for r in res])
    arr = np.stack([r[1] for r in res])
    labels = ["base v31_d1"] + [s[0] for s in SETTINGS]
    print("{:<26} {:>8} {:>8} | {:>12} {:>12} | {:>9} {:>9}".format(
        "pairs pasted, no decoder", "SkillC", "PFF", "caught < 2s", "caught 2-5s", "pred/true",
        "precision"))
    for j, lb in enumerate(labels):
        sc = [arr[names == nm, j, 0].mean() for nm in ("skillcorner", "pff")]
        t = arr[:, j, 1:].sum(axis=0)
        print("{:<26} {:>8.4f} {:>8.4f} | {:>11.1%} {:>11.1%}  | {:>9.2f} {:>9.1%}".format(
            lb, sc[0], sc[1], t[1] / t[0], t[3] / t[2], t[5] / t[4], t[6] / t[5]))


if __name__ == "__main__":
    main()
