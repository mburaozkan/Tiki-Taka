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


# The user's rule (2026-10-27), kept apart from fix_keepers: nothing is flipped, impostors are
# ignored. A team's keeper is visible when its deepest player, inside its own box, stands at least
# LINE_GAP_M behind the rest of the team (the defensive line). While it is, a player of the other
# team behind that line and within EXTEND_M of that goal for at least IGNORE_MIN_S is the doubled or
# swapped keeper from the far side of the pitch, and is ignored (NaN). No visible keeper: no change.
LINE_GAP_M = 2.0
IGNORE_MIN_S = 2.0
DUP_M = 1.0             # evidence: the other team's slot this close to the team's deepest player
EVIDENCE_SHARE = 0.5


def _goals(px, tracked, team, blocks, length):
    """(n, 2) x of the goal each team defends (NaN outside the blocks): median team x per half."""
    goal = np.full((len(px), 2), np.nan)
    for a, b in blocks:
        for side in (0, 1):
            cols = team == side
            with np.errstate(invalid="ignore"):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    mx = np.nanmedian(np.nanmean(np.where(tracked[a:b][:, cols], px[a:b][:, cols],
                                                          np.nan), axis=1))
            goal[a:b, side] = 0.0 if mx < length / 2.0 else length
    return goal


def keeper_status(players, team, blocks, pitch, own_keeper=False, deepest_only=False,
                  evidence=False):
    """(visible (n, 2) bool per team, ignored (n, k) bool): see the comment above.
    own_keeper=True: the same cases, but the team's own "visible keeper" (its deepest player) is the one
    ignored, since there the other team's slot is the real keeper (user, 2026-10-27, #4); the
    third value is then the other team's slots in those cases (n, k), for own_flip.
    deepest_only: per frame only the deepest of those slots is a case (the real keeper); the
    others are the team's attackers pressing him and keep their team (user, 2026-10-29).
    evidence: a case run is kept only if, on at least EVIDENCE_SHARE of it, the slot doubles the
    team's deepest player (within DUP_M) or that player is a still ghost (STILL_M over STILL_S):
    a tracking error, not an attacker behind the line (the rule fired on 1454 s, the measured
    impostors are ~290 s; JOURNAL.md 2026-10-29)."""
    length, width = pitch
    team = np.asarray(team)
    px = players[:, :, 0].astype(np.float64) * length
    py = players[:, :, 1].astype(np.float64) * width
    tracked = ~np.isnan(px)
    goal = _goals(px, tracked, team, blocks, length)
    n, k = px.shape
    still = _still(px, py, int(STILL_S * FPS)) if evidence else None
    visible = np.zeros((n, 2), dtype=bool)
    ignored = np.zeros((n, k), dtype=bool)
    own_ign = np.zeros((n, k), dtype=bool)
    cases = np.zeros((n, k), dtype=bool)
    for side in (0, 1):
        gx = goal[:, side][:, None]
        with np.errstate(invalid="ignore"):
            depth = np.where(tracked, np.abs(px - gx), np.inf)
            central = np.abs(py - width / 2.0)
        own = np.where((team == side)[None, :], depth, np.inf)
        part = np.partition(own, 1, axis=1) if k > 1 else np.full((n, 2), np.inf)
        d1, d2 = part[:, 0], part[:, 1]
        deepest = np.argmin(own, axis=1)
        in_box = (d1 <= BOX_DEPTH_M) & (central[np.arange(n), deepest] <= BOX_HALF_M)
        with np.errstate(invalid="ignore"):
            visible[:, side] = in_box & (d2 - d1 >= LINE_GAP_M) & np.isfinite(gx[:, 0])
        with np.errstate(invalid="ignore"):
            near = np.hypot(depth, central) <= EXTEND_M
        behind = visible[:, side][:, None] & (depth < d2[:, None]) & near & (team != side)[None, :]
        side_cases = np.zeros((n, k), dtype=bool)
        if evidence:
            rows = np.arange(n)
            ghost = still[rows, deepest] & np.isfinite(d1)
        for i in np.flatnonzero(team != side):
            for s, e in _runs(behind[:, i]):
                if e - s < IGNORE_MIN_S * FPS:
                    continue
                if evidence:
                    dj = deepest[s:e]
                    with np.errstate(invalid="ignore"):
                        dup = np.hypot(px[s:e, i] - px[rows[s:e], dj],
                                       py[s:e, i] - py[rows[s:e], dj]) <= DUP_M
                    if np.mean(dup | ghost[s:e]) < EVIDENCE_SHARE:
                        continue
                side_cases[s:e, i] = True
        if deepest_only:
            rows = np.flatnonzero(side_cases.any(axis=1))
            pick = np.argmin(np.where(side_cases[rows], depth[rows], np.inf), axis=1)
            side_cases[rows] = False
            side_cases[rows, pick] = True
        ignored |= side_cases
        cases |= side_cases
        rows = np.flatnonzero(side_cases.any(axis=1))
        own_ign[rows, deepest[rows]] = True
    if own_keeper:
        return visible, own_ign, cases
    return visible, ignored


def ignore_impostors(players, team, blocks, pitch, mode="opponent"):
    """(players, team, stats {"seconds", "per_team", "visible"}). mode "opponent": the other team's
    slot behind the line is ignored (NaN); "own": the team's own deepest player is ignored instead;
    "own_flip": as "own", and the other team's slot is moved to a new slot of the defending team
    (the real keeper with his right team); "own_flip_deepest": as "own_flip", but only the deepest
    such slot per frame is moved (an attacker pressing the keeper keeps his team);
    "own_flip_evidence": as "own_flip_deepest", only where keeper_status(evidence=True) finds a
    doubled or ghost keeper."""
    team = np.asarray(team)
    if mode == "opponent":
        visible, ignored = keeper_status(players, team, blocks, pitch)
        cases = None
    else:
        visible, ignored, cases = keeper_status(
            players, team, blocks, pitch, own_keeper=True,
            deepest_only=mode in ("own_flip_deepest", "own_flip_evidence"),
            evidence=mode == "own_flip_evidence")
    out = players.copy()
    out[ignored] = np.nan
    k0 = len(team)
    if mode in ("own_flip", "own_flip_deepest", "own_flip_evidence") and cases.any():
        slots = np.flatnonzero(cases.any(axis=0))
        extra = np.full((len(players), len(slots), players.shape[2]), np.nan, dtype=players.dtype)
        for j, i in enumerate(slots):
            extra[cases[:, i], j] = players[cases[:, i], i]
            out[cases[:, i], i] = np.nan
        out = np.concatenate([out, extra], axis=1)
        team = np.concatenate([team, (1 - team[slots]).astype(team.dtype)])
    graded = np.zeros(len(players), dtype=bool)
    for a, b in blocks:
        graded[a:b] = True
    stats = {"seconds": ignored.any(axis=1).sum() / float(FPS),
             "per_team": tuple(ignored[:, team[:k0] == s].any(axis=1).sum() / float(FPS)
                               for s in (0, 1)),
             "visible": tuple(visible[graded, s].mean() for s in (0, 1))}
    return out, team, stats
