"""How many short possessions the evidence holds without the decoder's smoothing. Aggregate only.

    python tools/decoder_sweep.py [--tag t1] [--rows ""]

On the 84 external matches (SkillCorner + PFF), the v31_d1 chain with:
  switch S  : the team Viterbi's switch cost S (base 20; 0 = per-frame argmax of the evidence,
              p1 + refiner + run vote), handover + backdating as usual;
  paste p0  : the base, then every turnover-classifier proposal (P >= p0) written straight over it
              as T from the touch to the other team's next pick (<= 5 s), no decoder.
Per setting: mean spellF1 (SkillCorner, PFF), possessions caught (IoU >= 0.5) < 2 s and 2-5 s,
predicted possessions per true possession, and the share of predicted possessions that match a true
one (precision at IoU >= 0.5). Prints aggregates only.
"""

import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from pipeline.decode import H, A  # noqa: E402

SWITCHES = (20.0, 10.0, 5.0, 3.0, 2.0, 1.0, 0.0)
PASTE = (0.2, 0.3, 0.5, 0.7)


def chain_switch(c, f, blk, pdead, switch):
    import eval_dead_v30 as ev
    import eval_touch_run as tr
    import train_ext_heads as th
    from pipeline import fixes, learned
    h2 = th.rebuild_hand2(c["hand"], pdead, f, blk)
    q = np.clip(c["p_ref"], 1e-4, 1 - 1e-4)
    p = fixes.add_vote(c["p1"], tr.ALPHA * np.log(q / (1 - q)) + c["vote"], 1.0)
    codes = learned.decode_hybrid(h2, p, blk, max(switch, 1e-6))
    codes = fixes.restart_handover(codes, f, max_s=3.0, slow_ms=2.0)
    return fixes.backdate(codes, f, **ev.BACKDATE)


def paste(base, meta, p, f, p0, max_s=5.0):
    out = base.copy()
    tch = np.asarray(f["touch"])
    picks = np.flatnonzero(tch != 0)
    side = np.where(tch[picks] > 0, H, A)
    n = len(base)
    for (i, T), pi in zip(meta[:, :2].astype(int), p):
        if pi < p0:
            continue
        nxt = picks[(picks > i + 2) & (side != T)]
        e = min(int(nxt[0]) if len(nxt) else n, i + int(max_s * 25), n)
        seg = out[i:e]
        out[i:e] = np.where(np.isin(seg, (H, A)), T, seg)
    return out


def stats(lab, codes):
    import eval_touch_run as tr
    import run_v31_external as rx
    runs = pm.valid_runs(lab)
    truth = pm.possession_spells(lab, runs)
    pred = pm.possession_spells(pm.fill_abstentions(codes, runs), runs)
    out = [tr.score(lab, codes)]
    for lo, hi in ((0, 2), (2, 5)):
        sel = [t for t in truth if lo <= (t[1] - t[0]) / 25.0 < hi]
        out += [len(sel), sum(rx.caught(t, pred) for t in sel)]
    out += [len(truth), len(pred), sum(rx.caught(q, truth) for q in pred)]
    return out


def run(job):
    name, g, rows, tag = job
    os.environ["TC_ROWS"] = rows
    import turnover_cls as T
    import run_v31_external as rx
    import train_ext_heads as th
    c, f = T.load_cache(name, g)
    with np.load(os.path.join(ROOT, "data", "cache", "ext_heads_d1.npz")) as z:
        pdead = th.seeds_mean({k: z[k] for k in z.files if k.startswith(name + "_" + g + "__")},
                              name + "_" + g)
    blk = rx.blocks(c["period"])
    lab = c["labels"]
    res = [stats(lab, chain_switch(c, f, blk, pdead, s)) for s in SWITCHES]
    base = chain_switch(c, f, blk, pdead, 20.0)
    with np.load(T.rows_path(name, g)) as z:
        meta = z["meta"]
    with np.load(os.path.join(T.OUT, "pred_{}.npz".format(tag))) as z:
        p = z["{}_{}".format(name, g)]
    res += [stats(lab, paste(base, meta, p, f, p0)) for p0 in PASTE]
    return name, np.array(res, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="t1")
    ap.add_argument("--rows", default="")
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    os.environ["TC_ROWS"] = args.rows
    import turnover_cls as T
    jobs = [(nm, g, args.rows, args.tag) for nm, g, _ in T.ext_list()]
    with Pool(args.jobs) as pool:
        res = pool.map(run, jobs, chunksize=1)
    names = np.array([r[0] for r in res])
    arr = np.stack([r[1] for r in res])          # (matches, settings, 9)
    labels = ["switch {:g}{}".format(s, " (base)" if s == 20 else "") for s in SWITCHES] + \
        ["paste p >= {:g}".format(p0) for p0 in PASTE]
    print("{:<18} {:>8} {:>8} | {:>14} {:>14} | {:>10} {:>10}".format(
        "setting", "SkillC", "PFF", "caught < 2 s", "caught 2-5 s", "pred/true", "precision"))
    for j, lb in enumerate(labels):
        sc = [arr[names == nm, j, 0].mean() for nm in ("skillcorner", "pff")]
        t = arr[:, j, 1:].sum(axis=0)
        print("{:<18} {:>8.4f} {:>8.4f} | {:>13.1%} {:>13.1%}  | {:>10.2f} {:>10.1%}".format(
            lb, sc[0], sc[1], t[1] / t[0], t[3] / t[2], t[5] / t[4], t[6] / t[5]))


if __name__ == "__main__":
    main()
