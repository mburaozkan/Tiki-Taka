"""Why does the dead-ball head miss a stoppage of ours? Coarse per-stoppage profile. Aggregate only.

    python tools/stoppage_profile.py

Public window: one row per true stoppage, numbered in order (S1, S2, ...; no frames or times):
length, ball seen share, longest gap, mean team speed (both teams, cols speed_*), the hand path's
D share and the head's mean P(dead) (d1n, deployed models). DFL (held-out head probabilities,
true stoppages >= MIN_S): the same columns as percentiles, and the head's mean P(dead) by ball
seen share, so a stoppage of ours can be placed in DFL's distribution.
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
import in_domain_team as idt  # noqa: E402
import eval_dead_v30 as dv  # noqa: E402
from dead_overcall import runs  # noqa: E402
from eval_mbr import v25_inputs  # noqa: E402
from eval_refine import stage1  # noqa: E402
from train_learned_dfl import match_features, test_seed  # noqa: E402
from pipeline import refine  # noqa: E402
from pipeline.decode import D  # noqa: E402

# Params
MIN_S = 2.0
SEEN_BINS = (0.0, 0.2, 0.5, 0.8, 1.01)
COLS = ("len_s", "seen", "max_gap_s", "speed", "hand_D", "p_dead")


def profile(lab, hand, f, cols, pdead, fps):
    rows = []
    seen = np.asarray(f["ball_seen"]) > 0
    speed = 0.5 * (np.nan_to_num(np.asarray(cols["speed_home"], np.float64))
                   + np.nan_to_num(np.asarray(cols["speed_away"], np.float64)))
    for a, b in runs(lab == D):
        if (b - a) / fps < MIN_S:
            continue
        miss = runs(~seen[a:b])
        rows.append(((b - a) / fps, seen[a:b].mean(),
                     max([(e - s) / fps for s, e in miss], default=0.0),
                     speed[a:b].mean(), np.mean(hand[a:b] == D), np.mean(pdead[a:b])))
    return np.array(rows)


def main():
    from games import GAME
    from possession_view import signals
    from report import PUBLIC_GT
    s = signals()
    with np.load(os.path.join(dd.DATA_DIR, "cache", "v25_ours_prob{}.npz".format(
            os.environ.get("MTAH_VARIANT", "")))) as c:
        prob = c["prob"]
    hand, f, cols = s["hand"], s["f"], s["cols"]
    p1 = stage1(prob, f)
    bundles = [refine.load(os.path.join(ROOT, "models", "dead_{}_s{}.pt".format(dv.DEAD_TAG, k)))
               for k in range(dv.DEAD_SEEDS)]
    refine.BIN = bundles[0][0].cfg["BIN"]
    xd = (refine.dead_inputs(p1, hand, f, cols, False, False),
          refine.dead_inputs(p1, hand, f, cols, True, False))
    pdead = np.mean([refine.predict(b, xd[0], xd[1], len(hand), symmetric=True) for b in bundles],
                    axis=0)
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    lab = np.full(len(hand), -1, np.int8)
    lab[:len(gt)] = gt[:len(hand)]
    ours = profile(lab, hand, f, cols, pdead, dd.FPS)
    print("public window, true stoppages >= {:g} s".format(MIN_S))
    print("      " + " ".join("{:>10}".format(c) for c in COLS))
    for i, r in enumerate(ours, start=1):
        print("S{:<4} ".format(i) + " ".join("{:>10.2f}".format(x) for x in r))

    data = v25_inputs()
    with np.load(os.path.join(dd.CACHE, "dead_{}.npz".format(dv.DEAD_TAG))) as c:
        pd_dfl = {g: np.mean([c["{}__{}".format(g, k)] for k in range(dv.DEAD_SEEDS)], axis=0)
                  for g in dd.GAMES}
    rows = []
    for g in dd.GAMES:
        d = dd.load(g)
        fg = match_features(g, test_seed(g))
        lab_g, hand_g, _ = data[g]
        rows.append(profile(lab_g, hand_g, fg, idt.player_cols_cached(g, d), pd_dfl[g], dd.FPS))
        print("  DFL", g, flush=True)
    rows = np.concatenate(rows)
    print("\nDFL true stoppages >= {:g} s: {}".format(MIN_S, len(rows)))
    print("pct   " + " ".join("{:>10}".format(c) for c in COLS))
    for q in (10, 25, 50, 75, 90):
        print("p{:<4} ".format(q) + " ".join("{:>10.2f}".format(x) for x in np.percentile(rows, q, axis=0)))
    print("share of DFL stoppages with head mean P(dead) < 0.1: {:.1f}%".format(
        100 * np.mean(rows[:, 5] < 0.1)))
    print("\nhead mean P(dead) by ball seen share (DFL):")
    for lo, hi in zip(SEEN_BINS[:-1], SEEN_BINS[1:]):
        m = (rows[:, 1] >= lo) & (rows[:, 1] < hi)
        if m.any():
            print("  seen {:.1f}-{:.1f}: {:4d} stoppages, median P(dead) {:.2f}, < 0.1 {:.0f}%".format(
                lo, min(hi, 1.0), m.sum(), np.median(rows[m, 5]), 100 * np.mean(rows[m, 5] < 0.1)))
    for i, r in enumerate(ours, start=1):
        print("S{}: percentile in DFL  ".format(i) + " ".join(
            "{}={:.0f}".format(c, 100 * np.mean(rows[:, k] < r[k])) for k, c in enumerate(COLS)))


if __name__ == "__main__":
    main()
