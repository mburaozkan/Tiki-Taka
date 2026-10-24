"""Does an external ball track move like ours? Aggregate statistics only, side by side.

    python tools/ball_realism.py

For our match and each DFL / Metrica match, over frames with the ball seen and players moving
(mean pace above MOVING_MS, a proxy for live play that needs no labels):
  speed       median ball speed (m/s)
  jitter      median |acceleration| frame to frame (m/s^2): tracker noise and smoothing
  near        median distance from the ball to the nearest player (m)
  touches     touches per minute of ball-seen live time, by our current rule (pipeline.features)
A model trained on a provider whose numbers differ a lot from ours will not transfer (Metrica's
ball sat on the carrier and was far smoother than ours, which is why v16 failed).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import pipeline.features as features  # noqa: E402

# Params
MOVING_MS = 2.0


def stats(ball, players, team, pitch, fps=25):
    length, width = features.LENGTH, features.WIDTH
    features.LENGTH, features.WIDTH = pitch
    f = features.build_features({"players": players, "ball": ball, "team": team}, events=[])
    features.LENGTH, features.WIDTH = length, width
    x, y = ball[:, 0].astype(np.float64) * pitch[0], ball[:, 1].astype(np.float64) * pitch[1]
    v = np.full(len(x), np.nan)
    v[1:] = np.hypot(np.diff(x), np.diff(y)) * fps
    acc = np.full(len(x), np.nan)
    acc[1:] = np.abs(np.diff(v)) * fps
    live = (f["ball_seen"] > 0) & (f["player_speed"] > MOVING_MS)
    near = np.fmin(f["d_home"], f["d_away"])
    minutes = live.sum() / fps / 60
    return (np.nanmedian(v[live]), np.nanmedian(acc[live]), np.nanmedian(near[live]),
            np.sum((f["touch"] != 0) & live) / max(minutes, 1e-9), 100 * live.mean())


def main():
    from atd_cache import load_arrays
    from games import GAME, GAMES
    import dfl_data as dd
    import metrica_data as md
    row = "{:<10} {:>8.1f} {:>9.1f} {:>7.2f} {:>9.1f} {:>9.0f}%"
    print("{:<10} {:>8} {:>9} {:>7} {:>9} {:>10}".format(
        "match", "speed", "jitter", "near", "touch/min", "frames used"))
    atd = load_arrays("tactical")
    print(row.format("ours", *stats(atd["ball"], atd["players"], atd["team"], GAMES[GAME]["pitch"])))
    for g in dd.GAMES:
        d = dd.load(g)
        print(row.format(g, *stats(d["ball"], d["players"], d["team"], dd.PITCH)), flush=True)
    for g in md.GAMES:
        d = md.load(g)
        print(row.format("metrica" + str(g), *stats(d["ball"], d["players"], d["team"], md.PITCH)),
              flush=True)


if __name__ == '__main__':
    main()
