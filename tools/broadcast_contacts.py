"""Does the broadcast ball show the contact that starts a possession, where the tactical ball does
not? Public window only, flags and shares only (no positions).

    python tools/broadcast_contacts.py
    python tools/broadcast_contacts.py --window 1.0

For every true possession start in the public ground truth, within +/- window s, per source
(tactical ATD; the broadcast ATD mapped onto tactical frames by tools/broadcast_align.py):
  aligned   share of the window with a broadcast frame
  ball      share with that source's ball seen
  peak      a velocity peak of the touch rule (possession_view.touch_peaks: >= 2 m/s, local max
            over +/- 3 frames) in the window
  new <=3m  at such a peak, the nearest player within 3 m is of the possession's (new) team
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
from games import DATA_DIR  # noqa: E402
from atd_cache import load_arrays  # noqa: E402
import pipeline.features as features  # noqa: E402
from possession_view import ball_track, touch_peaks  # noqa: E402

ALIGN = os.path.join(DATA_DIR, "cache", "broadcast_align.npz")


def contact(ball, players, team, lo, hi, side):
    """(peak in [lo, hi), a peak there with the nearest player within reach of team `side`)."""
    _, dv = ball_track(ball)
    peaks = np.flatnonzero(touch_peaks(dv)[lo:hi]) + lo
    if not len(peaks):
        return False, False
    d_home, d_away = features.nearest_by_team(players[peaks], team, ball[peaks])
    dh = np.nan_to_num(d_home, nan=np.inf)
    da = np.nan_to_num(d_away, nan=np.inf)
    mine, other = (dh, da) if side == 0 else (da, dh)
    return True, bool(np.any((mine <= features.TOUCH_R_M) & (mine < other)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--window", type=float, default=0.5)
    args = parser.parse_args()
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    spells = pm.possession_spells(gt, pm.valid_runs(gt))
    tac = load_arrays("tactical")
    bro = load_arrays("broadcast")
    with np.load(ALIGN) as c:
        match = c["match"]
    n = len(match)
    on = match >= 0
    bball = np.full((n, tac["ball"].shape[1]), np.nan, dtype=np.float32)
    bball[on] = bro["ball"][match[on]]
    bplayers = np.full((n,) + bro["players"].shape[1:], np.nan, dtype=np.float32)
    bplayers[on] = bro["players"][match[on]]
    k = int(args.window * FPS)
    print("possession starts, +/- {:g} s".format(args.window))
    print("{:<5} {:>4} {:>8} {:>9} {:>9} | {:>8} {:>9} | {:>8} {:>9}".format(
        "#", "team", "aligned", "ball tac", "ball bro", "tac peak", "new <=3m", "bro peak",
        "new <=3m"))
    rows = []
    for i, (a, b, code) in enumerate(spells, start=1):
        lo, hi = max(a - k, 0), min(a + k + 1, n)
        side = 0 if code == pm.H else 1
        tp, tn = contact(tac["ball"], tac["players"], np.asarray(tac["team"]), lo, hi, side)
        bp, bn = contact(bball, bplayers, np.asarray(bro["team"]), lo, hi, side)
        row = (on[lo:hi].mean(), (~np.isnan(tac["ball"][lo:hi, 0])).mean(),
               (~np.isnan(bball[lo:hi, 0])).mean(), tp, tn, bp, bn)
        rows.append(row)
        yn = lambda v: "yes" if v else "-"
        print("#{:<4} {:>4} {:>7.0%} {:>9.0%} {:>9.0%} | {:>8} {:>9} | {:>8} {:>9}".format(
            i, "H" if side == 0 else "A", row[0], row[1], row[2], yn(tp), yn(tn), yn(bp), yn(bn)))
    r = np.array(rows, dtype=float)
    print("all   {:>4} {:>7.0%} {:>9.0%} {:>9.0%} | {:>8.0%} {:>9.0%} | {:>8.0%} {:>9.0%}".format(
        "", *r.mean(axis=0)))


if __name__ == "__main__":
    main()
