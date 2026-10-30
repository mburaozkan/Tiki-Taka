"""What do phantom D stretches look like on DFL (v30_dh)? Aggregate only.

    python tools/phantom_diag.py

Frames of three groups, pooled over the 7 DFL matches (v30_dh as tools/eval_dead_v30.py):
  phantom   called D, true live, in a called D stretch that overlaps no true stoppage
  stoppage  called D and true D
  live ok   called live and true live
Per group: the hand rule's inputs (ball seen, gap length, ball speed, player speed, ball out,
keeper hold) and the dead-ball head's P(dead). Per phantom stretch: share with the ball missing
throughout / partly / never, and how the head's mean P(dead) separates phantom stretches from the
called D stretches that hold a true stoppage (share below a few thresholds).
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
from dead_overcall import runs, SETTING  # noqa: E402
from eval_mbr import v25_inputs  # noqa: E402
from eval_refine import stage1  # noqa: E402
from eval_dead_refine import constrain  # noqa: E402
from train_learned_dfl import match_features, test_seed, blocks  # noqa: E402
from pipeline import refine  # noqa: E402
from pipeline.decode import H, A, D  # noqa: E402

# Params
COLS = ("ball_seen", "gap_s", "ball_speed", "player_speed", "ball_out", "keeper_hold", "p_dead")
THRESH = (0.3, 0.5, 0.7)


def main():
    data = v25_inputs()
    with np.load(os.path.join(dd.CACHE, "refine_{}.npz".format(tr.TEAM_TAG))) as c:
        team = {k: c[k] for k in c.files}
    with np.load(os.path.join(dd.CACHE, "dead_{}.npz".format(dv.DEAD_TAG))) as c:
        pdead = {g: np.mean([c["{}__{}".format(g, s)] for s in range(dv.DEAD_SEEDS)], axis=0)
                 for g in dd.GAMES}
    vals = {k: {c: [] for c in COLS} for k in ("phantom", "stoppage", "live ok")}
    miss = np.zeros(3)                 # phantom stretches: ball missing all / partly / never
    head = {"phantom": [], "stoppage": []}
    for g in dd.GAMES:
        d = dd.load(g)
        f = match_features(g, test_seed(g))
        blk = blocks(d["period"])
        lab, hand, prob = data[g]
        p1 = stage1(prob, f)
        p_ref = np.mean([team["{}__{}".format(g, s)] for s in range(tr.TEAM_SEEDS)], axis=0)
        hand2 = constrain(hand, refine.decode_dead(hand, pdead[g], blk, *SETTING), dv.MODE)
        pred = dv.chain(hand2, p1, p_ref, f, blk, tr.run_vote(f, dv.VOTE))
        live = np.isin(lab, (H, A))
        phantom = np.zeros(len(lab), bool)
        for a, b in runs(pred == D):
            if np.any(lab[a:b] == D):
                head["stoppage"].append(np.mean(pdead[g][a:b]))
                continue
            if not live[a:b].any():
                continue
            phantom[a:b] = True
            seen = f["ball_seen"][a:b] > 0
            miss[0 if not seen.any() else 2 if seen.all() else 1] += 1
            head["phantom"].append(np.mean(pdead[g][a:b]))
        masks = {"phantom": phantom & live, "stoppage": (pred == D) & (lab == D),
                 "live ok": (pred != D) & live}
        cols = dict(f, p_dead=pdead[g])
        for k, m in masks.items():
            for c in COLS:
                x = np.asarray(cols[c], dtype=np.float64)[m]
                vals[k][c].append(np.abs(x) if c == "keeper_hold" else x)
        print("  DFL", g, flush=True)
    print("\nmean per frame (nan ignored)      " + " ".join("{:>12}".format(c) for c in COLS))
    for k, v in vals.items():
        print("{:<10} ({:>7} frames)     ".format(k, sum(len(x) for x in v[COLS[0]])) + " ".join(
            "{:>12.2f}".format(np.nanmean(np.concatenate(v[c]))) for c in COLS))
    n = miss.sum()
    print("\nphantom stretches {:.0f}: ball missing throughout {:.0f}%, partly {:.0f}%, seen throughout {:.0f}%".format(
        n, *(100 * miss / n)))
    ph, st = np.array(head["phantom"]), np.array(head["stoppage"])
    print("head mean P(dead) per stretch: phantom median {:.2f}, true-stoppage stretches median {:.2f}".format(
        np.median(ph), np.median(st)))
    for t in THRESH:
        print("  below {:.1f}: phantom {:.0f}%, true-stoppage stretches {:.0f}%".format(
            t, 100 * np.mean(ph < t), 100 * np.mean(st < t)))


if __name__ == "__main__":
    main()
