"""Give impostor keepers back to the team they play for (the tracking's slot team is fixed).

    from pipeline.track_fix import fix_keepers
    players, team, stats = fix_keepers(players, team, blocks, pitch)

The ATD gives every slot one team for the whole match, but slots swap people across gaps
(JOURNAL.md, 2026-10-27: Athletic's keeper carried by a Barcelona slot, next to a frozen
"Athletic keeper" left in goal). Swaps happen across gaps, not as jumps, so the unit is a piece: a
slot's continuous tracked run. A piece of team T is the other team's keeper when, inside one half,
  - it spends at least IN_BOX of its frames inside the box of the goal the other team defends, and
  - on at least DEEPEST of its frames it is deeper (closer to that goal line) than every defender
    in the box that is moving (a defender still to within STILL_M over STILL_S may be a ghost),
  - and it lasts at least MIN_PIECE_S.
Attackers do not live in the opponent's box behind the keeper for seconds on end. Second pass,
for a swap without a gap (two players crossing): inside a piece that fails the test, every run of
at least STRETCH_S as the deepest in the box, while a still defender stands in that box (the ghost
the user saw; without it, an attacker deepest while the keeper is untracked would be flipped), is
flipped, widened to the whole stay within EXTEND_M of the goal (the keeper steps out of his
box to play the ball; user, #9–#13). A flipped piece
moves to a new slot of the other team (its old slot is NaN there), so every consumer of
(players, team) works unchanged. The goal each team defends comes from the team's median x per
half. Barcelona's untracked keeper is not invented: he only matters with the ball in his box.
"""

import numpy as np

# Params
FPS = 25
BOX_DEPTH_M, BOX_HALF_M = 16.5, 20.16
IN_BOX = 0.8
DEEPEST = 0.5
MIN_PIECE_S = 2.0
STRETCH_S = 1.0         # second pass: deepest in the box for at least this long
EXTEND_M = 30.0         # ...flipped for its whole stay within this of the goal (a keeper steps out)
STILL_S, STILL_M = 3.0, 0.15


def _runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _still(px, py, w):
    import pandas as pd
    out = []
    for v in (px, py):
        roll = pd.DataFrame(v).rolling(w, center=True, min_periods=w)
        out.append((roll.max() - roll.min()).values)
    rng = np.maximum(out[0], out[1])
    return np.where(np.isnan(rng), False, rng < STILL_M)


def fix_keepers(players, team, blocks, pitch):
    """(players, team, stats): see the module docstring. blocks = [(first, stop)] per half;
    stats = {"pieces": flipped pieces, "seconds": their length, "per_team": flipped from H, A}."""
    length, width = pitch
    team = np.asarray(team)
    px = players[:, :, 0].astype(np.float64) * length
    py = players[:, :, 1].astype(np.float64) * width
    tracked = ~np.isnan(px)
    still = _still(px, py, int(STILL_S * FPS))
    flips = []
    for a, b in blocks:
        goal = {}
        for side in (0, 1):
            cols = team == side
            mx = np.nanmedian(np.nanmean(np.where(tracked[a:b][:, cols], px[a:b][:, cols], np.nan),
                                         axis=1))
            goal[side] = 0.0 if mx < length / 2.0 else length
        for side in (0, 1):                 # the defending side of a goal; impostors are 1 - side
            depth = np.abs(px[a:b] - goal[side])
            box = tracked[a:b] & (depth <= BOX_DEPTH_M) & (np.abs(py[a:b] - width / 2.0) <= BOX_HALF_M)
            moving_def = box & (team == side)[None, :] & ~still[a:b]
            d_def = np.min(np.where(moving_def, depth, np.inf), axis=1)
            ghost = (box & (team == side)[None, :] & still[a:b]).any(axis=1)
            near_goal = tracked[a:b] & (np.hypot(depth, py[a:b] - width / 2.0) <= EXTEND_M)
            for i in np.flatnonzero(team != side):
                for s, e in _runs(tracked[a:b, i]):
                    if e - s < MIN_PIECE_S * FPS:
                        continue
                    in_box = box[s:e, i]
                    deepest = in_box & (depth[s:e, i] < d_def[s:e])
                    if in_box.mean() >= IN_BOX and deepest.mean() >= DEEPEST:
                        flips.append((i, a + s, a + e, side))
                        continue
                    # A swap without a gap: flip the stretch around each run of being the deepest
                    # in the box for STRETCH_S, as far as the piece stays in the box.
                    for ds, de in _runs(deepest & ghost[s:e]):
                        if de - ds < STRETCH_S * FPS:
                            continue
                        lo, hi = ds, de
                        near = near_goal[s:e, i]
                        while lo > 0 and near[lo - 1]:
                            lo -= 1
                        while hi < len(near) and near[hi]:
                            hi += 1
                        if not flips or flips[-1][:2] != (i, a + s + lo):
                            flips.append((i, a + s + lo, a + s + hi, side))
    if not flips:
        return players, team, {"pieces": 0, "seconds": 0.0, "per_team": (0, 0)}
    extra = np.full((len(players), len(flips), players.shape[2]), np.nan, dtype=players.dtype)
    out = players.copy()
    for j, (i, s, e, side) in enumerate(flips):
        extra[s:e, j] = players[s:e, i]
        out[s:e, i] = np.nan
    new_team = np.concatenate([team, np.array([side for _, _, _, side in flips], dtype=team.dtype)])
    stats = {"pieces": len(flips), "seconds": sum(e - s for _, s, e, _ in flips) / float(FPS),
             "per_team": (sum(team[i] == 0 for i, _, _, _ in flips),
                          sum(team[i] == 1 for i, _, _, _ in flips))}
    return np.concatenate([out, extra], axis=1), new_team, stats
