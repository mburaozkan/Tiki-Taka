"""Where does the dead-ball (D vs live) side lose on DFL? Aggregate only.

    python tools/eval_dead.py

On the realistic DFL matches (v25 inputs, v29's second stage from data/cache/dfl/refine_b2.npz,
seeds 0-4, alpha 0.5, switch 20), per match and overall:
  D>live / live>D   share of true D frames called live, of true live frames called D
  stoppages         true D stretches >= 2 s: share overlapped by a called D at all (found),
                    signed start / end offsets of the called D (s, + = late), median
  oracle bounds     spellF1 with the D vs live split taken from the labels and the team as v29
                    ("oracle D"), and v29's D split with the team from the labels ("oracle team")
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
import dfl_data as dd  # noqa: E402
from eval_mbr import v25_inputs  # noqa: E402
from eval_refine import stage1  # noqa: E402
from tune_refine import decode  # noqa: E402
from train_learned_dfl import match_features, test_seed, blocks  # noqa: E402
from pipeline import learned, fixes  # noqa: E402
from pipeline.decode import H, A, D, X  # noqa: E402

# Params
TAG, SEEDS, ALPHA, SWITCH = "b2", 5, 0.5, 20.0
MIN_STOP_S = 2.0


def runs(mask):
    e = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def v29_path(g, data, probs):
    lab, hand, prob = data[g]
    f = match_features(g, test_seed(g))
    blk = blocks(dd.load(g)["period"])
    p_ref = np.mean([probs["{}__{}".format(g, s)] for s in range(SEEDS)], axis=0)
    p1 = stage1(prob, f)
    return decode(hand, p1, p_ref, f, blk, ALPHA, SWITCH), hand, p1, p_ref, f, blk


def main():
    data = v25_inputs()
    with np.load(os.path.join(dd.CACHE, "refine_{}.npz".format(TAG))) as c:
        probs = {k: c[k] for k in c.files}
    rows, tot = [], {"dl": [0, 0], "ld": [0, 0], "found": [0, 0], "so": [], "eo": []}
    print("{:<8} {:>7} {:>7} {:>7} {:>8} {:>8} {:>7} {:>9} {:>11}".format(
        "match", "D>live", "live>D", "found", "start s", "end s", "v29", "oracle D", "oracle team"))
    for g in dd.GAMES:
        lab = data[g][0]
        pred, hand, p1, p_ref, f, blk = v29_path(g, data, probs)
        graded = lab != X
        td, tl = graded & (lab == D), graded & np.isin(lab, (H, A))
        pd_, pl = pred == D, np.isin(pred, (H, A))
        dl, ld = np.sum(td & pl), np.sum(tl & pd_)
        tot["dl"][0] += dl
        tot["dl"][1] += td.sum()
        tot["ld"][0] += ld
        tot["ld"][1] += tl.sum()
        found, so, eo = 0, [], []
        stops = [(a, b) for a, b in runs(td) if b - a >= MIN_STOP_S * dd.FPS]
        called = runs(pd_)
        for a, b in stops:
            ov = [(c, e) for c, e in called if c < b and e > a]
            if ov:
                found += 1
                so.append((ov[0][0] - a) / dd.FPS)
                eo.append((ov[-1][1] - b) / dd.FPS)
        tot["found"][0] += found
        tot["found"][1] += len(stops)
        tot["so"] += so
        tot["eo"] += eo
        # oracle D: labels' D vs live split, team as v29 inside the true live stretches
        q = np.clip(p_ref, 1e-4, 1 - 1e-4)
        p_team = fixes.add_vote(p1, ALPHA * np.log(q / (1 - q)), 1.0)
        oracle_hand = np.where(lab == X, X, np.where(lab == D, D, H)).astype(np.int8)
        oracle_d = learned.decode_hybrid(oracle_hand, p_team, blk, SWITCH)
        oracle_t = np.where(np.isin(pred, (H, A)) & np.isin(lab, (H, A)), lab, pred)
        s = [pm.evaluate(lab, p, bootstrap_n=0)["spell_f1_mean"] for p in (pred, oracle_d, oracle_t)]
        rows.append(s)
        print("{:<8} {:>6.1f}% {:>6.1f}% {:>6.0f}% {:>8.1f} {:>8.1f} {:>7.3f} {:>9.3f} {:>11.3f}".format(
            g, 100 * dl / td.sum(), 100 * ld / tl.sum(), 100 * found / max(len(stops), 1),
            np.median(so) if so else np.nan, np.median(eo) if eo else np.nan, *s), flush=True)
    m = np.mean(rows, axis=0)
    print("{:<8} {:>6.1f}% {:>6.1f}% {:>6.0f}% {:>8.1f} {:>8.1f} {:>7.3f} {:>9.3f} {:>11.3f}".format(
        "all", 100 * tot["dl"][0] / tot["dl"][1], 100 * tot["ld"][0] / tot["ld"][1],
        100 * tot["found"][0] / tot["found"][1], np.median(tot["so"]), np.median(tot["eo"]), *m))
    so, eo = np.array(tot["so"]), np.array(tot["eo"])
    print("stoppages >= {:g} s: {}; called start within 1 s {:.0f}%, within 3 s {:.0f}%; end within 1 s "
          "{:.0f}%, within 3 s {:.0f}%".format(MIN_STOP_S, tot["found"][1], 100 * np.mean(np.abs(so) <= 1),
                                              100 * np.mean(np.abs(so) <= 3), 100 * np.mean(np.abs(eo) <= 1),
                                              100 * np.mean(np.abs(eo) <= 3)))


if __name__ == '__main__':
    main()
