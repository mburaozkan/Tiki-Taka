"""What happens to turnover-classifier proposals at true short starts (SkillCorner + PFF). Aggregate only.

    python tools/short_proposals.py --rows v2 --tag t2w6 --p0 0.2

For every candidate that is a true start of a possession < 5 s with P >= p0 (a proposal), after the
v31_d1 chain with the proposals (b 8, span <= 5 s): caught (a predicted spell of T with IoU >= 0.5),
or why not: no predicted T spell overlaps / overlaps with IoU < 0.5 because the predicted spell is
too long (ends late) / too short (ends early) / starts off. Median offsets of predicted vs true end,
and of the proposal span end (next picked touch of the other team) vs the true end.
False proposals (P >= p0, not a true start): share, and how many frames they change.
Prints counts, shares and medians only.
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
from pipeline.decode import H, A  # noqa: E402


def run(job):
    name, g, rows, tag, p0 = job
    os.environ["TC_ROWS"] = rows
    import turnover_cls as T
    T.ROWS, T.V2 = rows, rows == "v2"
    import run_v31_external as rx
    import train_ext_heads as th
    import eval_touch_run as tr
    c, f = T.load_cache(name, g)
    with np.load(os.path.join(ROOT, "data", "cache", "ext_heads_d1.npz")) as z:
        pdead = th.seeds_mean({k: z[k] for k in z.files if k.startswith(name + "_" + g + "__")},
                              name + "_" + g)
    with np.load(T.rows_path(name, g)) as z:
        meta = z["meta"]
    with np.load(os.path.join(T.OUT, "pred_{}.npz".format(tag))) as z:
        p = z["{}_{}".format(name, g)]
    blk = rx.blocks(c["period"])
    lab = c["labels"]
    n = len(lab)
    s = dict(T.CHOSEN, p0=p0)
    boost = T.span_boost(n, meta, p, f, s["b"], s["p0"], s["max_s"])
    codes = T.base_chain(c, f, blk, pdead, None, boost)
    # stages, for the no-overlap cases
    import eval_dead_v30 as ev
    from pipeline import fixes, learned
    h2 = th.rebuild_hand2(c["hand"], pdead, f, blk)
    q = np.clip(c["p_ref"], 1e-4, 1 - 1e-4)
    pp = fixes.add_vote(c["p1"], tr.ALPHA * np.log(q / (1 - q)) + c["vote"] + boost, 1.0)
    st1 = learned.decode_hybrid(h2, pp, blk, tr.SWITCH)
    st2 = fixes.restart_handover(st1, f, max_s=3.0, slow_ms=2.0)
    runs = pm.valid_runs(lab)
    truth = pm.possession_spells(lab, runs)
    pred = pm.possession_spells(pm.fill_abstentions(codes, runs), runs)
    t_by_start = {(a, k): (a, b, k) for a, b, k in truth}
    tch = np.asarray(f["touch"])
    picks = np.flatnonzero(tch != 0)
    side = np.where(tch[picks] > 0, H, A)
    out = []
    for (i, team, y, ln, _), pi in zip(meta, p):
        if pi < p0:
            continue
        i, team = int(i), int(team)
        if y != 1 or ln >= 5:
            out.append(("false" if y != 1 else "long", np.nan, np.nan, np.nan))
            continue
        # the true spell this candidate starts
        best = min((t for t in truth if t[2] == team and abs(t[0] - i) <= 13),
                   key=lambda t: abs(t[0] - i), default=None)
        if best is None:
            continue
        a, b, k = best
        nxt = picks[(picks > i + 2) & (side != team)]
        span_end = min(int(nxt[0]) if len(nxt) else n, i + int(s["max_s"] * T.FPS))
        ov = [(c_, e_) for c_, e_, j in pred if j == k and e_ > a and c_ < b]
        if not ov:
            if np.any(h2[a:b] == 2) and np.mean(h2[a:b] == 2) > 0.5:
                st = "hand D"
            elif not np.any(st1[a:b] == k):
                st = "decoder"
            elif not np.any(st2[a:b] == k):
                st = "handover"
            else:
                st = "backdate"
            out.append(("{}: no overlap, lost at {}".format("short" if ln < 2 else "mid", st),
                        np.nan, np.nan, (span_end - b) / T.FPS))
            continue
        c_, e_ = max(ov, key=lambda q: min(b, q[1]) - max(a, q[0]))
        iou = (min(b, e_) - max(a, c_)) / float(max(b, e_) - min(a, c_))
        cls = "short" if ln < 2 else "mid"
        if iou >= 0.5:
            why = "caught"
        elif (e_ - c_) > (b - a):
            why = "pred too long"
        else:
            why = "pred too short"
        out.append(("{}: {}".format(cls, why), (e_ - b) / T.FPS, (c_ - a) / T.FPS,
                    (span_end - b) / T.FPS))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="v2")
    ap.add_argument("--tag", default="t2w6")
    ap.add_argument("--p0", type=float, default=0.2)
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    os.environ["TC_ROWS"] = args.rows
    import turnover_cls as T
    from multiprocessing import Pool
    jobs = [(nm, g, args.rows, args.tag, args.p0) for nm, g, _ in T.ext_list()]
    with Pool(args.jobs) as pool:
        res = [r for x in pool.map(run, jobs, chunksize=1) for r in x]
    kinds = sorted({r[0] for r in res})
    tot = len(res)
    print("proposals (P >= {}): {}".format(args.p0, tot))
    for k in kinds:
        sel = [r for r in res if r[0] == k]
        e = np.array([r[1] for r in sel], float)
        st = np.array([r[2] for r in sel], float)
        sp = np.array([r[3] for r in sel], float)
        print("  {:<22} {:>5} ({:.0%})  median end offset {:+.2f} s, start offset {:+.2f} s, "
              "span end vs true end {:+.2f} s".format(k, len(sel), len(sel) / tot,
                                                      np.nanmedian(e) if np.any(~np.isnan(e)) else np.nan,
                                                      np.nanmedian(st) if np.any(~np.isnan(st)) else np.nan,
                                                      np.nanmedian(sp) if np.any(~np.isnan(sp)) else np.nan))


if __name__ == "__main__":
    main()
