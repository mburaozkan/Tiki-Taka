"""Where is live play called D on DFL, for v30 and v30_dh? Aggregate only.

    python tools/dead_overcall.py

v30 and v30_dh as tools/eval_dead_v30.py (head setting 0.5 / 0.5 / 25). True-live frames called D
are split by the called D stretch they sit in:
  early     the stretch overlaps a true stoppage, frame before that stoppage starts
  late      the stretch overlaps a true stoppage, frame after it ends (restart called late)
  phantom   the stretch overlaps no true stoppage (D out of nowhere)
Per kind: share of all true-live frames, phantom stretches per 90 min and median length.
Oracle: spellF1 if the frames of one kind took the labelled team instead (upper bound of a fix).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import dfl_data as dd  # noqa: E402
import eval_touch_run as tr  # noqa: E402
import eval_dead_v30 as dv  # noqa: E402
from eval_mbr import v25_inputs  # noqa: E402
from eval_refine import stage1  # noqa: E402
from eval_dead_refine import constrain  # noqa: E402
from train_learned_dfl import match_features, test_seed, blocks  # noqa: E402
from pipeline import refine  # noqa: E402
from pipeline.decode import H, A, D, X  # noqa: E402

# Params
SETTING = (0.5, 0.5, 25.0)
KINDS = ("early", "late", "phantom")


def runs(mask):
    e = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def classify(lab, pred):
    """Per frame: 0 none, 1 early, 2 late, 3 phantom (true live called D). Phantom stretch lengths."""
    kind = np.zeros(len(lab), np.int8)
    true_d = lab == D
    over = (pred == D) & np.isin(lab, (H, A))
    lens = []
    for a, b in runs(pred == D):
        seg = over[a:b]
        if not seg.any():
            continue
        td = np.flatnonzero(true_d[a:b])
        if len(td) == 0:
            kind[a:b][seg] = 3
            lens.append((b - a) / dd.FPS)
            continue
        idx = np.arange(a, b)
        kind[idx[seg & (idx < a + td[0])]] = 1
        kind[idx[seg & (idx > a + td[-1])]] = 2
        kind[idx[seg & (idx > a + td[0]) & (idx < a + td[-1])]] = 3   # live gaps inside, rare
    return kind, lens


def main():
    data = v25_inputs()
    with np.load(os.path.join(dd.CACHE, "refine_{}.npz".format(tr.TEAM_TAG))) as c:
        team = {k: c[k] for k in c.files}
    with np.load(os.path.join(dd.CACHE, "dead_{}.npz".format(dv.DEAD_TAG))) as c:
        pdead = {g: np.mean([c["{}__{}".format(g, s)] for s in range(dv.DEAD_SEEDS)], axis=0)
                 for g in dd.GAMES}
    stats = {v: {"live": 0, "min": 0.0, "n": np.zeros(4), "lens": [], "score": [],
                 "oracle": {k: [] for k in KINDS}} for v in ("v30", "v30_dh")}
    for g in dd.GAMES:
        d = dd.load(g)
        f = match_features(g, test_seed(g))
        blk = blocks(d["period"])
        lab, hand, prob = data[g]
        p1 = stage1(prob, f)
        p_ref = np.mean([team["{}__{}".format(g, s)] for s in range(tr.TEAM_SEEDS)], axis=0)
        vote = tr.run_vote(f, dv.VOTE)
        hand2 = constrain(hand, refine.decode_dead(hand, pdead[g], blk, *SETTING), dv.MODE)
        for v, h in (("v30", hand), ("v30_dh", hand2)):
            pred = dv.chain(h, p1, p_ref, f, blk, vote)
            kind, lens = classify(lab, pred)
            st = stats[v]
            st["live"] += int(np.sum(np.isin(lab, (H, A))))
            st["min"] += np.sum(lab != X) / dd.FPS / 60
            st["n"] += np.bincount(kind, minlength=4)
            st["lens"] += lens
            st["score"].append(tr.score(lab, pred))
            for k, name in enumerate(KINDS, start=1):
                fixed = np.where(kind == k, lab, pred).astype(np.int8)
                st["oracle"][name].append(tr.score(lab, fixed))
        print("  DFL", g, flush=True)
    for v, st in stats.items():
        base = np.mean(st["score"])
        print("\n{}: spellF1 {:.4f}; true live called D {:.1f}%".format(
            v, base, 100 * st["n"][1:].sum() / st["live"]))
        for k, name in enumerate(KINDS, start=1):
            print("  {:<8} {:5.1f}% of live frames | oracle spellF1 {:.4f} ({:+.4f})".format(
                name, 100 * st["n"][k] / st["live"], np.mean(st["oracle"][name]),
                np.mean(st["oracle"][name]) - base))
        lens = np.array(st["lens"])
        if len(lens):
            print("  phantom stretches: {:.1f} per 90 min, median {:.1f} s, >= 3 s {:.0f}%".format(
                len(lens) / st["min"] * 90, np.median(lens), 100 * np.mean(lens >= 3)))


if __name__ == "__main__":
    main()
