"""Does the ball's location near a goal push the decoder to D? Aggregate answer only.

    python tools/zone_diag.py evaluation/results/v8.csv

Splits graded, ball-seen frames by the ball's zone (six-yard box, rest of the penalty area,
elsewhere) and prints per zone: share of frames, predicted D share (full match), true and predicted
D share (public window), mean nearest-player distance, the 'alone' share (no player within 8 m),
ball-out share and ball speed. Then goalkeeper coverage: how often each team has any tracked player
within GK_M of a goal it could be defending. Shares and means only.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "starter"))
import possession_metrics as pm  # noqa: E402
from report import PUBLIC_GT  # noqa: E402
from pipeline.features import build_features, LENGTH, WIDTH  # noqa: E402
from atd_cache import load_arrays, HOME, AWAY  # noqa: E402

# Params
SIX_DEPTH, SIX_HALF = 5.5, 9.16       # six-yard box: depth, half-width (m)
BOX_DEPTH, BOX_HALF = 16.5, 20.16     # penalty area
GK_M = 16.5                           # a keeper is within this of its goal centre in live play


def zone(bx, by):
    """0 six-yard box, 1 rest of penalty area, 2 elsewhere (NaN -> 2)."""
    depth = np.fmin(bx, LENGTH - bx)
    lateral = np.abs(by - WIDTH / 2.0)
    six = (depth <= SIX_DEPTH) & (lateral <= SIX_HALF)
    box = (depth <= BOX_DEPTH) & (lateral <= BOX_HALF)
    return np.where(six, 0, np.where(box, 1, 2))


def keepers(players, team):
    """(n, 2) keeper positions (m) for home, away: per frame, the team's player nearest the goal
    that team defends (the goal on the side of its own players' mean x); NaN if none within GK_M.
    Also (n, 2) the x of that goal."""
    px, py = players[:, :, 0] * LENGTH, players[:, :, 1] * WIDTH
    n = len(px)
    pos = np.full((n, 2, 2), np.nan)
    goal_x = np.zeros((n, 2))
    for j, side in enumerate((HOME, AWAY)):
        cols = team == side
        mean_x = np.nanmean(px[:, cols], axis=1)
        gx = np.where(mean_x < LENGTH / 2.0, 0.0, LENGTH)
        goal_x[:, j] = gx
        d = np.hypot(px[:, cols] - gx[:, None], py[:, cols] - WIDTH / 2.0)
        d = np.where(np.isnan(d), np.inf, d)
        i = np.argmin(d, axis=1)
        ok = d[np.arange(n), i] <= GK_M
        sub_x, sub_y = px[:, cols], py[:, cols]
        pos[ok, j, 0] = sub_x[np.flatnonzero(ok), i[ok]]
        pos[ok, j, 1] = sub_y[np.flatnonzero(ok), i[ok]]
    return pos, goal_x


def mix(codes, team_code=None):
    codes = codes[codes != pm.X]
    if not len(codes):
        return "-"
    text = "H {:3.0f}% A {:3.0f}% D {:3.0f}%".format(*(100 * np.mean(codes == k)
                                                       for k in (pm.H, pm.A, pm.D)))
    return text + " ({:.0f}s)".format(len(codes) / 25.0)


def keeper_section(pred, gt, f, ball, players, team, graded, public):
    """Ball at a keeper (in his own box) and gaps by how the ball was last seen."""
    pos, goal_x = keepers(players, team)
    bx, by = ball[:, 0] * LENGTH, ball[:, 1] * WIDTH
    print("\nBALL AT A KEEPER (ball seen within 2.5 m of a team's deepest player, in its own box)")
    gt_k = {}
    for j, (label, code) in enumerate((("home keeper", pm.H), ("away keeper", pm.A))):
        d = np.hypot(bx - pos[:, j, 0], by - pos[:, j, 1])
        in_box = (np.abs(bx - goal_x[:, j]) <= BOX_DEPTH) & (np.abs(by - WIDTH / 2.0) <= BOX_HALF)
        m = (d <= 2.5) & in_box & graded
        gt_k[code] = m
        print("  {}: full match pred {}   public truth {}   public pred {}".format(
            label, mix(pred[m]), mix(gt[m & public]), mix(pred[m & public])))

    # Gaps of 2 s or more, classed by the last seen ball before the gap.
    seen = f["ball_seen"] > 0
    from pipeline.features import runs_of
    classes = {}
    for a, b in runs_of(~seen):
        if b - a < 50 or a == 0 or not graded[a]:
            continue
        lx, ly = bx[a - 1], by[a - 1]
        cls = "elsewhere"
        if f["ball_out"][a - 1] > 0 or np.any(f["ball_out"][max(0, a - 25):a] > 0):
            end_line = min(lx, LENGTH - lx) < 1.5
            cls = "out over end line" if end_line else "out over touchline"
        else:
            for j in range(2):
                if np.hypot(lx - pos[a - 1, j, 0], ly - pos[a - 1, j, 1]) <= 3.0:
                    cls = "at a keeper"
        classes.setdefault(cls, []).append((a, b))
    print("\nGAPS >= 2 s by how the ball was last seen   n  median s   full-match pred"
          "                      public truth")
    for cls, runs in sorted(classes.items()):
        m = np.zeros(len(pred), dtype=bool)
        for a, b in runs:
            m[a:b] = True
        m &= graded
        print("  {:<20} {:4d}  {:6.1f}   {:<36} {}".format(
            cls, len(runs), np.median([(b - a) / 25.0 for a, b in runs]), mix(pred[m]),
            mix(gt[m & public])))


def main():
    pred = pm.label_codes(pm.read_label_csv(sys.argv[1]))
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    atd = load_arrays("tactical")
    f = build_features(atd)
    ball, players, team = atd["ball"], atd["players"], atd["team"]
    bx, by = ball[:, 0] * LENGTH, ball[:, 1] * WIDTH
    z = zone(bx, by)
    graded = pred != pm.X
    seen = (f["ball_seen"] > 0) & graded
    public = gt != pm.X
    nearest = np.fmin(f["d_home"], f["d_away"])
    names = ["six-yard box", "penalty area", "elsewhere"]

    print("BALL SEEN, by zone   frames  pred D   pub frames  true D  pred D   nearest m  "
          "none<8m  out  speed")
    for k, name in enumerate(names):
        m = seen & (z == k)
        pm_ = m & public
        print("  {:<16} {:6.1f}%  {:5.1f}%   {:9d}  {:5.1f}%  {:5.1f}%   {:9.1f}  {:6.1f}%  {:3.0f}%  "
              "{:5.1f}".format(
                  name, 100 * m.sum() / seen.sum(), 100 * np.mean(pred[m] == pm.D),
                  int(pm_.sum()),
                  100 * np.mean(gt[pm_] == pm.D) if pm_.any() else np.nan,
                  100 * np.mean(pred[pm_] == pm.D) if pm_.any() else np.nan,
                  np.nanmedian(nearest[m]), 100 * np.mean(nearest[m] > 8.0),
                  100 * np.mean(f["ball_out"][m]), np.nanmedian(f["ball_speed"][m])))

    # Goalkeeper coverage: per team, any tracked player within GK_M of either goal centre.
    px, py = players[:, :, 0] * LENGTH, players[:, :, 1] * WIDTH
    d_left = np.hypot(px, py - WIDTH / 2.0)
    d_right = np.hypot(px - LENGTH, py - WIDTH / 2.0)
    print("\nGOAL COVERAGE (graded frames): share with a tracked player of the team within "
          "{:.1f} m of each goal".format(GK_M))
    for side, label in ((HOME, "home"), (AWAY, "away")):
        cols = team == side
        left = np.any(np.nan_to_num(d_left[:, cols], nan=1e9) <= GK_M, axis=1)
        right = np.any(np.nan_to_num(d_right[:, cols], nan=1e9) <= GK_M, axis=1)
        either = left | right
        print("  {}: left goal {:4.1f}%  right goal {:4.1f}%  either {:4.1f}%   tracked players "
              "per frame {:.1f}".format(label, 100 * left[graded].mean(), 100 * right[graded].mean(),
                                         100 * either[graded].mean(),
                                         np.mean(np.sum(~np.isnan(px[:, cols]), axis=1)[graded])))
    keeper_section(pred, gt, f, ball, players, team, graded, public)

    # With the ball in a six-yard box: is any player of either team tracked within 8 m?
    m = seen & (z == 0)
    print("\n  ball in a six-yard box: no tracked player within 8 m on {:.0f}% of frames, "
          "within 3 m on {:.0f}%".format(100 * np.mean(nearest[m] > 8), 100 * np.mean(nearest[m] <= 3)))


if __name__ == '__main__':
    main()
