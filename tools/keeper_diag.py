"""How often does the tracking get the keepers wrong? Aggregate only (counts, seconds, shares).

    python tools/keeper_diag.py

The user saw in the public window (JOURNAL.md, 2026-10-27): a frozen "keeper" left standing in
goal (ghost), the real keeper carried by a slot of the other team (impostor), and the visible
keeper doubled for both teams (duplicate). Over the graded frames of the full match, per goal
(the goal each team defends, from the team's median x in each half):
  ghost       a tracked slot that moves less than FROZEN_M over FROZEN_S (all slots, and in a box)
  impostor    the deepest tracked player in a box belongs to the attacking team, at least
              IMPOSTOR_MARGIN_M deeper than every defender, within KEEPER_DEPTH_M of the goal line
              (also with frozen defenders ignored: a ghost left in goal hides the impostor)
  no keeper   no defender of that goal inside its box
  duplicate   two slots within DUP_M of each other (same team / different teams / in a keeper zone)
  jump        a slot moving more than JUMP_M between two frames (an identity swap)
Each line also gives the share of those frames with the ball seen within NEAR_BALL_M, and the
number of episodes of at least MIN_EPISODE_S. Nothing per frame, per slot or per position.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

# Params
FPS = 25
FROZEN_S = 3.0
FROZEN_M = (0.05, 0.15, 0.3)
IMPOSTOR_MARGIN_M = 2.0
KEEPER_DEPTH_M = 6.0
BOX_DEPTH_M, BOX_HALF_M = 16.5, 20.16
DUP_M = (1.0, 0.3)
JUMP_M = 1.0                # 25 m/s at 25 fps
NEAR_BALL_M = 10.0
MIN_EPISODE_S = 1.0


def episodes(mask, min_len):
    """Number of runs of True at least min_len frames long."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return int(np.sum(stops - starts >= min_len))


def rolling_range(v, w):
    """Max - min of v over a centred window of w frames (NaN if any frame in it is missing)."""
    import pandas as pd
    frame = pd.DataFrame(v)
    roll = frame.rolling(w, center=True, min_periods=w)
    return (roll.max() - roll.min()).values


def diagnose(players, team, ball, graded, pitch):
    """{name: (seconds, near-ball share, episodes)} over graded frames; see the module docstring."""
    length, width = pitch
    px = players[:, :, 0].astype(np.float64) * length
    py = players[:, :, 1].astype(np.float64) * width
    bx = ball[:, 0].astype(np.float64) * length
    by = ball[:, 1].astype(np.float64) * width
    n, k = px.shape
    tracked = ~np.isnan(px)
    dball = np.hypot(px - bx[:, None], py - by[:, None])      # NaN without ball
    near = np.where(np.isnan(dball), False, dball <= NEAR_BALL_M)
    min_ep = int(MIN_EPISODE_S * FPS)
    out = {}

    def add(name, mask, near_mask):
        mask = mask & graded
        sec = mask.sum() / float(FPS)
        share = near_mask[mask].mean() if mask.any() else np.nan
        out[name] = (sec, share, episodes(mask, min_ep))

    # Goal each team defends, per graded block (constant within a half).
    goal_x = np.full((n, 2), np.nan)
    labels = np.cumsum(np.diff(np.concatenate(([0], graded.astype(np.int8)))) == 1) * graded
    for blk in np.unique(labels[labels > 0]):
        rows = labels == blk
        for side in (0, 1):
            mx = np.nanmedian(np.nanmean(np.where(tracked[rows][:, team == side],
                                                  px[rows][:, team == side], np.nan), axis=1))
            goal_x[rows, side] = 0.0 if mx < length / 2.0 else length

    # Ghosts.
    w = int(FROZEN_S * FPS)
    rng = np.maximum(rolling_range(px, w), rolling_range(py, w))
    in_box_any = np.zeros((n, k), dtype=bool)
    for side in (0, 1):
        gx = goal_x[:, side][:, None]
        in_box_any |= (np.abs(px - gx) <= BOX_DEPTH_M) & (np.abs(py - width / 2.0) <= BOX_HALF_M)
    ghost = np.where(np.isnan(rng), False, rng < FROZEN_M[1])
    for m in FROZEN_M:
        frozen = np.where(np.isnan(rng), False, rng < m)
        add("ghost < {:g} m, any slot".format(m), frozen.any(axis=1), (frozen & near).any(axis=1))
        fb = frozen & in_box_any
        add("ghost < {:g} m, in a box".format(m), fb.any(axis=1), (fb & near).any(axis=1))

    # Impostor and missing keeper, per defended goal.
    for side, name in ((0, "home goal"), (1, "away goal")):
        gx = goal_x[:, side][:, None]
        depth = np.abs(px - gx)
        box = tracked & (depth <= BOX_DEPTH_M) & (np.abs(py - width / 2.0) <= BOX_HALF_M)
        dfd = box & (team == side)[None, :]
        att = box & (team != side)[None, :]
        d_att = np.min(np.where(att, depth, np.inf), axis=1)
        att_near = (att & near & (depth <= KEEPER_DEPTH_M)).any(axis=1)
        for label, defenders in (("", dfd), (", ghosts ignored", dfd & ~ghost)):
            d_def = np.min(np.where(defenders, depth, np.inf), axis=1)
            imp = (d_att <= KEEPER_DEPTH_M) & (d_att + IMPOSTOR_MARGIN_M <= d_def)
            add("impostor keeper, " + name + label, imp, imp & att_near)
        nok = ~dfd.any(axis=1) & np.isfinite(goal_x[:, side])
        add("no defender in box, " + name, nok, (box & near).any(axis=1))

    # Duplicates (and in the keeper zone: within KEEPER_DEPTH_M of a goal line, inside the box).
    keeper_zone = np.zeros((n, k), dtype=bool)
    for side in (0, 1):
        gx = goal_x[:, side][:, None]
        keeper_zone |= (np.abs(px - gx) <= KEEPER_DEPTH_M) & (np.abs(py - width / 2.0) <= BOX_HALF_M)
    for dm in DUP_M:
        hits = {key: np.zeros(n, dtype=bool) for key in ("same", "diff", "zone", "near", "zn")}
        for i in range(k):
            d = np.hypot(px[:, i + 1:] - px[:, i:i + 1], py[:, i + 1:] - py[:, i:i + 1])
            close = np.where(np.isnan(d), False, d <= dm)
            if not close.any():
                continue
            same = team[i + 1:] == team[i]
            hits["same"] |= close[:, same].any(axis=1)
            hits["diff"] |= close[:, ~same].any(axis=1)
            hit = close.any(axis=1)
            hits["zone"] |= hit & keeper_zone[:, i]
            hits["near"] |= hit & near[:, i]
            hits["zn"] |= hit & keeper_zone[:, i] & near[:, i]
        add("duplicate <= {:g} m, same team".format(dm), hits["same"], hits["near"])
        add("duplicate <= {:g} m, different teams".format(dm), hits["diff"], hits["near"])
        add("duplicate <= {:g} m, keeper zone".format(dm), hits["zone"], hits["zn"])
    step = np.hypot(np.diff(px, axis=0), np.diff(py, axis=0))
    jump = np.vstack([np.zeros((1, k), dtype=bool), np.where(np.isnan(step), False, step > JUMP_M)])
    add("jump > {:g} m/frame, any slot".format(JUMP_M), jump.any(axis=1), (jump & near).any(axis=1))
    jb = jump & in_box_any
    add("jump > {:g} m/frame, in a box".format(JUMP_M), jb.any(axis=1), (jb & near).any(axis=1))
    out["_graded_s"] = graded.sum() / float(FPS)
    return out


def report(out):
    total = out.pop("_graded_s")
    print("graded: {:.0f} s".format(total))
    print("{:<44} {:>8} {:>8} {:>10} {:>9}".format("", "seconds", "% graded", "near ball",
                                                    "episodes"))
    for name, (sec, share, ep) in out.items():
        print("{:<44} {:>8.0f} {:>7.1f}% {:>9.0f}% {:>9}".format(
            name, sec, 100 * sec / total, 100 * share if np.isfinite(share) else 0, ep))


def synthetic():
    """A tiny made-up match exercising every check (no real data)."""
    rng = np.random.default_rng(0)
    n, k = 600, 6
    team = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
    players = np.full((n, k, 3), np.nan, dtype=np.float32)
    t = np.arange(n) / FPS
    for i in range(k):
        base = 0.3 if team[i] == 0 else 0.7
        players[:, i, 0] = base + 0.05 * np.sin(t + i) + rng.normal(0, 0.002, n)
        players[:, i, 1] = 0.5 + 0.1 * np.cos(t * 0.7 + i)
    players[:, 0, 0], players[:, 0, 1] = 0.01, 0.5               # home keeper at x = 0
    players[:, 3, 0], players[:, 3, 1] = 0.995, 0.5              # away keeper, frozen: a ghost
    players[200:400, 1, 0], players[200:400, 1, 1] = 0.985, 0.52  # home slot in the away goal
    players[400:500, 4, :2] = players[400:500, 5, :2]            # duplicate
    players[300, 2, 0] = 0.9                                     # a jump
    ball = np.stack([players[:, 1, 0], players[:, 1, 1], np.zeros(n)], axis=1)
    return players, team, ball, np.ones(n, dtype=bool), (105.0, 68.0)


def main():
    if "--synthetic" in sys.argv:
        report(diagnose(*synthetic()))
        return
    from games import GAME, half_video_blocks
    from atd_cache import load_arrays
    from pipeline.features import LENGTH, WIDTH
    atd = load_arrays()
    graded = np.zeros(len(atd["ball"]), dtype=bool)
    for _, a, b in half_video_blocks(GAME):
        graded[a:b + 1] = True
    report(diagnose(atd["players"], np.asarray(atd["team"]), atd["ball"], graded, (LENGTH, WIDTH)))


if __name__ == '__main__':
    main()
