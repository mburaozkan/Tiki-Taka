"""Why does v29 not switch into a true public possession? Relative seconds and aggregates only.

    python tools/switch_diag.py 2 15
    python tools/switch_diag.py 15 --sub evaluation/results/v29.csv

For every true possession #k given (numbered as in tools/spell_table.py), over the spell and a
PAD_S margin around it:
  hand      the hand path's live / D runs (s from the true start), which bound the team Viterbi
  sub       the submission's runs over the same stretch
  P(team)   mean P(true team | live) over the spell: first stage, second stage, v29 blend, and the
            share of spell frames where the blend favours the true team
  gain      the summed blend log-odds toward the true team over the spell, and the best gain of
            any sub-stretch inside the margin, against the cost of the switches it needs (SWITCH
            per change; one change when the stretch touches a D or the end of a live run)
  touches   picked touches: offset (s), side, touch network P, toucher's margin over the nearest
            opponent (m)
No frame numbers, clock times or tracking values.
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
from report import PUBLIC_GT, FPS  # noqa: E402
from possession_view import signals, SUB  # noqa: E402

# Params
PAD_S = 8.0
SWITCH = 20.0           # v29's switch cost
NAMES = {pm.H: "H", pm.A: "A", pm.D: "D", pm.X: "X"}


def runs_text(codes, a, origin):
    edges = np.flatnonzero(np.diff(codes)) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [len(codes)]))
    return "  ".join("{} {:+.1f}..{:+.1f}".format(NAMES[int(codes[s])], (a + s - origin) / FPS,
                                                  (a + e - origin) / FPS)
                     for s, e in zip(starts, stops))


def best_gain(x):
    """Largest sum of a contiguous stretch of x, with its (start, stop)."""
    best, cur, lo, span = -np.inf, 0.0, 0, (0, 0)
    for i, v in enumerate(x):
        if cur <= 0:
            cur, lo = 0.0, i
        cur += v
        if cur > best:
            best, span = cur, (lo, i + 1)
    return best, span


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("numbers", type=int, nargs="+")
    parser.add_argument("--sub", default=SUB)
    args = parser.parse_args()
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    runs = pm.valid_runs(gt)
    spells = pm.possession_spells(gt, runs)
    sub = pm.label_codes(pm.read_label_csv(args.sub))
    s = signals()
    hand, f = s["hand"], s["f"]
    hand_live = np.where(np.isin(hand, (pm.H, pm.A)), pm.H, hand)
    touch = np.asarray(f["touch"], dtype=np.float64)
    th, ta = np.asarray(f["touch_p_home"]), np.asarray(f["touch_p_away"])
    dh, da = np.asarray(f["d_home"], np.float64), np.asarray(f["d_away"], np.float64)

    def logit(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))

    z = logit(s["blend"])
    for k in args.numbers:
        a0, b0 = spells[k - 1][0], spells[k - 1][1]
        team = int(gt[a0])
        sign = 1.0 if team == pm.H else -1.0
        run = next(r for r in runs if r[0] <= a0 < r[1])
        a, b = max(a0 - int(PAD_S * FPS), run[0]), min(b0 + int(PAD_S * FPS), run[1])
        print("\n#{}  true {}  {:.1f} s".format(k, NAMES[team], (b0 - a0) / FPS))
        print("  truth  " + runs_text(gt[a:b], a, a0))
        print("  hand   " + runs_text(np.where(hand_live[a:b] == pm.H, pm.X, pm.D), a, a0)
              .replace("X ", "live "))
        print("  sub    " + runs_text(sub[a:b], a, a0))

        def team_p(p):
            return float(np.mean(p[a0:b0] if team == pm.H else 1 - p[a0:b0]))
        live = np.isin(hand[a0:b0], (pm.H, pm.A))
        print("  P(true team) over the spell: first {:.2f}  second {:.2f}  blend {:.2f}   blend "
              "favours it on {:.0f}% of frames; hand live on {:.0f}%".format(
                  team_p(s["p1"]), team_p(s["p_ref"]), team_p(s["blend"]),
                  100 * np.mean(sign * z[a0:b0] > 0), 100 * np.mean(live)))
        g = sign * z * np.isin(hand, (pm.H, pm.A))
        ends_dead = hand[b0] == pm.D if b0 < len(hand) else True
        starts_dead = hand[a0 - 1] == pm.D if a0 > 0 else True
        need = SWITCH * (2 - int(ends_dead) - int(starts_dead))
        print("  gain over the spell {:+.1f} (switches needed {:.0f}: {}starts after D, {}ends in D)"
              .format(g[a0:b0].sum(), need, "" if starts_dead else "not ", "" if ends_dead else "not "))
        best, (lo, hi) = best_gain(g[a:b])
        print("  best sub-stretch {:+.1f} at {:+.1f}..{:+.1f} s".format(
            best, (a + lo - a0) / FPS, (a + hi - a0) / FPS))
        for lo_s, hi_s in ((-PAD_S, 0), (0, (b0 - a0) / FPS), ((b0 - a0) / FPS, (b0 - a0) / FPS + PAD_S)):
            sl = slice(max(a, a0 + int(lo_s * FPS)), min(b, a0 + int(hi_s * FPS)))
            print("    blend log-odds toward {} {:+.1f}..{:+.1f}: sum {:+.1f}, mean {:+.2f}".format(
                NAMES[team], lo_s, hi_s, g[sl].sum(), g[sl].mean() if sl.stop > sl.start else 0))
        for i in np.flatnonzero(touch[a:b] != 0) + a:
            m = abs(dh[i] - da[i]) if not (np.isnan(dh[i]) or np.isnan(da[i])) else float("nan")
            print("  touch {:+.1f} s  {}  P(H touch) {:.2f}  P(A touch) {:.2f}  margin {:.1f} m".format(
                (i - a0) / FPS, "H" if touch[i] > 0 else "A", th[i], ta[i], m))


if __name__ == "__main__":
    main()
