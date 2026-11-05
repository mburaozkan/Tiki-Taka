"""How common are teleporting slots, and what would fix_teleports (MTAH_VARIANT=_kt) change?

    python tools/teleport_diag.py

Runs pipeline.track_fix.fix_teleports on the raw tactical ATD of the whole match and prints
aggregates only (counts, seconds, shares; no frames, positions or slot ids):
  - teleports: a slot reappearing >= TELEPORT_MIN_M away, faster than TELEPORT_MS;
  - keeper swaps: teleported pieces that play as the other team's keeper, flipped to that team;
  - how long the ball is within 10 m of a flipped piece (where it can change a label);
  - ghost seconds ignored, and how much of it falls in the public window and in #9-#13.
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
from report import PUBLIC_GT, FPS  # noqa: E402
from pipeline import track_fix as tf  # noqa: E402


def main():
    from games import GAME, GAMES, half_video_blocks
    from atd_cache import load_arrays
    raw = load_arrays(raw=True)
    players, team, ball = raw["players"], np.asarray(raw["team"]), raw["ball"]
    length, width = GAMES[GAME]["pitch"]
    blk = [(x, y + 1) for _, x, y in half_video_blocks(GAME)]
    graded = np.zeros(len(players), dtype=bool)
    for a, b in blk:
        graded[a:b] = True
    out, new_team, st = tf.fix_teleports(players, team, blk, (length, width), details=True)
    k0 = len(team)

    print("teleports (a slot reappearing >= {:.0f} m away, faster than {:.0f} m/s): {}".format(
        tf.TELEPORT_MIN_M, tf.TELEPORT_MS, st["teleports"]))
    print("keeper swaps flipped: {} pieces, {:.1f} s ({:.2f}% of the graded match); from "
          "Barcelona slots {}, from Athletic slots {}".format(
              st["pieces"], st["seconds"], 100 * st["seconds"] / (graded.sum() / FPS),
              *st["per_team"]))
    print("ghosts ignored (a still slot in its own box while its keeper plays elsewhere): "
          "{:.1f} s".format(st["ghost_seconds"]))
    print("keepers bridged through the swap gaps (no real position in the data): {:.1f} s".format(
        st["bridged_seconds"]))
    if st["pieces"]:
        bx = ball[:, 0].astype(np.float64) * length
        by = ball[:, 1].astype(np.float64) * width
        ex = out[:, k0:, 0].astype(np.float64) * length
        ey = out[:, k0:, 1].astype(np.float64) * width
        with np.errstate(invalid="ignore"):
            near = (np.hypot(ex - bx[:, None], ey - by[:, None]) <= 10.0).any(axis=1)
        print("ball within 10 m of a flipped keeper: {:.1f} s".format(near.sum() / FPS))
        for side, name in ((0, "Barcelona"), (1, "Athletic")):
            cols = np.flatnonzero(new_team[k0:] == side)
            secs = (~np.isnan(out[:, k0 + cols, 0])).any(axis=1).sum() / FPS if len(cols) else 0.0
            print("  added {} slots (flipped or bridged keepers): {}, {:.1f} s".format(
                name, len(cols), secs))
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    runs = pm.valid_runs(gt)
    spells = pm.possession_spells(gt, runs)
    pub = np.zeros(len(gt), dtype=bool)
    for a, b in runs:
        pub[a:b] = True
    a, b = spells[8][0], spells[12][1]
    flipped = st["flipped"]
    print("flipped keeper in the public window: {:.1f} s; in #9-#13: {:.1f} s".format(
        (flipped[:len(pub)] & pub).sum() / FPS, flipped[a:b].sum() / FPS))
    # every reappearance (a new tracked run or a one-frame jump) starting inside #9-#13
    length_m, width_m = length, width
    px = players[:, :, 0].astype(np.float64) * length_m
    py = players[:, :, 1].astype(np.float64) * width_m
    tracked = ~np.isnan(px)
    h0, h1 = next((x, y) for x, y in blk if x <= a < y)
    print("reappearances starting in #9-#13 (gap, distance, speed, team, near the other team's goal "
          "at the start):")
    goal = tf._goals(px, tracked, team, blk, length_m)
    for i in range(len(team)):
        pieces = tf._pieces(px[h0:h1, i], py[h0:h1, i], tracked[h0:h1, i])
        for (ps, pe), (qs, qe) in zip(pieces, pieces[1:]):
            lf, ff = h0 + pe - 1, h0 + qs
            if not a <= ff < b:
                continue
            d = np.hypot(px[ff, i] - px[lf, i], py[ff, i] - py[lf, i])
            gap = (ff - lf) / float(FPS)
            og = 1 - team[i]
            dg = np.hypot(px[ff, i] - goal[ff, og], py[ff, i] - width_m / 2.0)
            print("  gap {:.2f} s, moved {:.0f} m, {:.1f} m/s, {} slot, {:.0f} m from {}'s goal, "
                  "next piece {:.1f} s{}".format(
                      gap, d, d / gap, ("Barcelona", "Athletic")[team[i]], dg,
                      ("Barcelona", "Athletic")[og], (qe - qs) / float(FPS),
                      "  <- teleport" if d >= tf.TELEPORT_MIN_M and d / gap > tf.TELEPORT_MS
                      else ""))
    print("teleported pieces overlapping the public window (piece length, stay near the other "
          "team's goal from the reappearance, share of it deeper than every moving player, flipped):")
    for s, e, secs, stay, share, deeper, ok in st["details"]:
        if pub[s:e].any():
            print("  {}{:.1f} s long, stay {:.1f} s, deeper {:.0f}%, {}".format(
                "#9-#13: " if s < b and e > a else "", secs, stay, 100 * deeper,
                "flipped" if ok else "not flipped"))

if __name__ == '__main__':
    main()