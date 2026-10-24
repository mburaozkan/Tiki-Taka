"""Guess the ball's position inside ATD ball gaps with simple kinematics, no learning.

    from pipeline.ball_fill import fill_ball
    filled = fill_ball(ball, method="extrapolate")   # same shape as ball, gaps filled

Methods
-------
hold         the last seen position, held through the gap
extrapolate  the last seen position moved on with the last seen velocity (direction and speed
             over the last VEL_WINDOW frames), slowing down exponentially with FRICTION_S, kept
             on the pitch
interp       straight line from the last seen position to the position where the ball reappears

Only the position is filled. Whether a frame is a gap (ball_seen, gap_s) is still read from the
real track, so the dead-ball evidence is unchanged.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES  # noqa: E402

# Params
FPS = GAMES[GAME]["fps"]
LENGTH, WIDTH = GAMES[GAME]["pitch"]
VEL_WINDOW = 10         # frames (0.4 s) before the gap used for the last velocity
FRICTION_S = 2.0        # time constant (s) of the ball slowing down: travel = v * T * (1 - e^-t/T)
MAX_SPEED_MS = 40.0     # faster than this is a tracker jump: no velocity, hold instead
METHODS = ("hold", "extrapolate", "interp")


def _runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def last_velocity(x, y, last):
    """(vx, vy) in m/s over the VEL_WINDOW frames ending at `last`; zero if not usable."""
    first = last - VEL_WINDOW
    if first < 0 or np.isnan(x[first]) or np.isnan(x[last]):
        return 0.0, 0.0
    vx = (x[last] - x[first]) * FPS / VEL_WINDOW
    vy = (y[last] - y[first]) * FPS / VEL_WINDOW
    if np.hypot(vx, vy) > MAX_SPEED_MS:
        return 0.0, 0.0
    return vx, vy


def fill_gap(x, y, start, stop, method):
    """(gx, gy) in metres for frames start..stop-1 of a gap; x, y are metre tracks with NaN gaps."""
    n = stop - start
    last, nxt = start - 1, stop
    has_last = last >= 0
    has_next = nxt < len(x)
    if not has_last and not has_next:
        return np.full(n, np.nan), np.full(n, np.nan)
    if not has_last:  # gap at the very start: hold the reappearing position
        return np.full(n, x[nxt]), np.full(n, y[nxt])
    x0, y0 = x[last], y[last]
    t = np.arange(1, n + 1) / float(FPS)
    if method == "interp" and has_next:
        frac = np.arange(1, n + 1) / float(n + 1)
        return x0 + frac * (x[nxt] - x0), y0 + frac * (y[nxt] - y0)
    if method == "extrapolate":
        vx, vy = last_velocity(x, y, last)
        travel = FRICTION_S * (1.0 - np.exp(-t / FRICTION_S))
        return (np.clip(x0 + vx * travel, 0.0, LENGTH), np.clip(y0 + vy * travel, 0.0, WIDTH))
    return np.full(n, x0), np.full(n, y0)


def fill_ball(ball, method="extrapolate", seen=None):
    """Copy of ball (n, 3; x, y normalised 0-1) with the x, y of every gap filled.

    seen: optional mask of frames to treat as seen (the benchmark hides seen stretches with it).
    """
    if method not in METHODS:
        raise ValueError("method must be one of {}".format(METHODS))
    seen = ~np.isnan(ball[:, 0]) if seen is None else seen
    x = np.where(seen, ball[:, 0].astype(np.float64) * LENGTH, np.nan)
    y = np.where(seen, ball[:, 1].astype(np.float64) * WIDTH, np.nan)
    out = ball.copy()
    for start, stop in _runs(~seen):
        gx, gy = fill_gap(x, y, start, stop, method)
        out[start:stop, 0] = gx / LENGTH
        out[start:stop, 1] = gy / WIDTH
    return out
