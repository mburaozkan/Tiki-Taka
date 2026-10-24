"""Per-frame evidence for the decoder, one array per feature, indexed by video frame.

    from pipeline.features import build_features
    f = build_features()          # dict of float32 arrays, each (n_frames,)

Nothing here looks at the ground truth. Every feature is computed over the whole video, and the
decoder only ever reads it inside the two halves.

Features
--------
ball_seen      1 where the ATD has a ball position
d_home/d_away  distance (m) from the ball to the nearest tracked player of each team; NaN if no ball.
               With ball_fill, measured to the guessed ball inside gaps (pipeline/ball_fill.py)
c_home/c_away  carrier distance (m): how close the best player of each team stayed to the ball over
               +/- CARRIER_HALF frames; NaN if no player stays tracked with the ball seen
touch          +1 / -1 on frames where a home / away player touches the ball (velocity-change peak
               next to a player), 0 elsewhere
last_touch     the side of the most recent touch, carried forward; 0 before any and after a long gap
ball_out       1 where the ball is outside the pitch (beyond OUT_MARGIN_M)
ball_speed     ball speed (m/s), median over +/- 0.5 s of visible frames; NaN if no ball
gap_s          length (s) of the ball gap a frame sits in; 0 where the ball is seen
player_speed   mean speed (m/s) of every tracked player, averaged over +/- 1 s
tag_setpiece   1 inside a Smart Tagging SET PIECES interval
tag_team       +1 / -1 where the Smart Tagging phases point at home / away having the ball, 0 else
m_home/m_away  co-movement: how well the team's best player within COMOVE_R_M of the ball moves
               with it, exp(-|v_player - v_ball| / COMOVE_SCALE_MS) in [0, 1], averaged over
               +/- COMOVE_HALF frames; NaN where the ball is unseen or slower than COMOVE_MIN_MS
shape          log(home spread / away spread), spread = std(x) * std(y) of each team's tracked
               players, averaged over +/- SHAPE_HALF frames: the team in possession spreads out,
               the defending team compacts. Works with the ball missing. NaN with < 5 players a side
g_home/g_away  distance (m) from the ball to the goal the home / away team defends
keeper_hold    +1 / -1 while the home / away keeper holds the ball: a ball gap that starts with the
               ball at the keeper inside his own box (a catch), up to KEEPER_HOLD_S; not after the
               ball went out (that is a goal kick being set up)
tag_restart    1 from a SET PIECES start until the first attack/defence phase starts (the restart
               is being set up), capped at RESTART_MAX_S
tag_restart_team  +1 / -1 for RESTART_TEAM_S after that window: the team the set piece is tagged
               for takes it (0 for throw-ins and kick-offs, which are tagged N/A)
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES  # noqa: E402
from atd_cache import load_arrays, HOME, AWAY  # noqa: E402

# Params
FPS = GAMES[GAME]["fps"]
LENGTH, WIDTH = GAMES[GAME]["pitch"]
OUT_MARGIN_M = 0.5          # tolerance before a ball counts as out, for tracking noise on the line
BALL_SPEED_HALF = FPS // 2  # +/- frames for the ball-speed median
PLAYER_SPEED_HALF = FPS     # +/- frames for the player-speed mean
CARRIER_HALF = 6            # +/- frames a carrier must stay with the ball (~0.5 s window)
# Touch detection: a touch is a peak in the ball's velocity change |v(t+K) - v(t-K)| above
# TOUCH_DV_MS, with a player within TOUCH_R_M. Positions are smoothed over +/- TOUCH_SMOOTH frames
# first, so tracker jitter does not read as a touch.
TOUCH_SMOOTH = 2
TOUCH_K = 3
TOUCH_DV_MS = 2.0         # 3.0 in v5: missed dribble touches and deflections
TOUCH_R_M = 3.0           # 2.0 in v5
# A contested touch (nearest opponent less than this much farther than the toucher) is a coin flip
# (43% right in the public window), so it does not change last_touch. 0 = every touch counts (v10).
TOUCH_MIN_MARGIN_M = 0.0
TOUCH_MAX_MS = 40.0         # faster than this is a tracker jump, not a ball
TOUCH_RESET_GAP_S = 3.0     # after a gap this long, the last touch is forgotten

# Smart Tagging codes and which side of the Team tag they say has the ball. +1: the tagged team is
# in possession. -1: the tagged team is defending, so the other one has it. Codes not listed
# (SET PIECES, GOALS, PLAYERS IN THE BOX) say nothing about the team in possession here.
# These signs are a reading of the code names, to be checked with tools/feature_table.py.
TAG_SIGN = {
    "build up": 1, "progression": 1, "ball in final third": 1, "ball in the box": 1,
    "creating chances": 1, "long ball": 1, "attacking transition": 1,
    "defending in defensive third": -1, "defending in middle third": -1,
    "defending in attacking third": -1, "defensive transition": -1,
}
SETPIECE_CODE = "set pieces"
# Optional extra team readings for codes left out of TAG_SIGN (sign unknown; see tag_alignment.py).
EXTRA_TAG_SIGN = {}
# Co-movement: the player carrying the ball moves with it (same direction and speed); a defender
# closing in is near it but moves differently.
COMOVE_R_M = 2.5
COMOVE_SCALE_MS = 2.0       # velocity difference (m/s) at which the match falls to 1/e
COMOVE_MIN_MS = 1.5         # a still ball matches every standing player: no reading below this
COMOVE_HALF = 6             # +/- frames (~0.5 s) of averaging
PLAYER_MAX_MS = 12.0        # faster than this is a track swap, not a player

SHAPE_HALF = 12             # +/- frames (~0.5 s)
SHAPE_MIN_PLAYERS = 5

# Keeper hold: the "keeper" is the team's player nearest the goal it defends (the goal on the side
# of its players' mean x), if within KEEPER_ZONE_M of it.
KEEPER_ZONE_M = 16.5
KEEPER_BALL_M = 3.0         # last seen ball within this of the keeper
BOX_DEPTH_M, BOX_HALF_M = 16.5, 20.16
KEEPER_HOLD_S = 600.0       # the whole gap (user, 2026-10-23: a keeper holding it is never D); 8 s in v9
KEEPER_OUT_S = 5.0          # ball out within this before the gap: a goal kick, not a hold
RESTART_MAX_S = 20.0        # a restart window never runs longer than this
RESTART_TEAM_S = 5.0        # how long after the restart the tagged team gets the vote


def runs_of(mask):
    """[(start, stop)] half-open True runs."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def window_mean(values, half):
    """Centred mean over +/- half frames, ignoring NaN. NaN where the window has no value."""
    ok = ~np.isnan(values)
    total = np.concatenate(([0.0], np.cumsum(np.where(ok, values, 0.0))))
    count = np.concatenate(([0], np.cumsum(ok)))
    index = np.arange(len(values))
    low, high = np.maximum(0, index - half), np.minimum(len(values), index + half + 1)
    n = count[high] - count[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (total[high] - total[low]) / np.maximum(n, 1), np.nan)


def window_median(values, half):
    """Centred median over +/- half frames, ignoring NaN, computed per visible stretch."""
    out = np.full(len(values), np.nan)
    for start, stop in runs_of(~np.isnan(values)):
        chunk = values[start:stop]
        if stop - start <= 2 * half:
            out[start:stop] = np.median(chunk)
            continue
        padded = np.pad(chunk, half, mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
        out[start:stop] = np.median(windows, axis=1)
    return out


def ball_features(ball):
    """ball_seen, ball_out, ball_speed, gap_s from the ball track."""
    seen = ~np.isnan(ball[:, 0])
    bx, by = ball[:, 0] * LENGTH, ball[:, 1] * WIDTH
    with np.errstate(invalid="ignore"):
        out = ((bx < -OUT_MARGIN_M) | (bx > LENGTH + OUT_MARGIN_M)
               | (by < -OUT_MARGIN_M) | (by > WIDTH + OUT_MARGIN_M))

    step = np.full(len(ball), np.nan)
    step[1:] = np.hypot(np.diff(bx), np.diff(by)) * FPS
    step[0] = step[1] if len(step) > 1 else np.nan
    step[~seen] = np.nan
    speed = window_median(step, BALL_SPEED_HALF)
    speed[~seen] = np.nan

    gap = np.zeros(len(ball))
    for start, stop in runs_of(~seen):
        gap[start:stop] = (stop - start) / float(FPS)
    return seen, out & seen, speed, gap


def velocity(x, y, smooth=TOUCH_SMOOTH):
    """Centred velocity (m/s) of a position track (m), smoothed over +/- smooth frames."""
    xs, ys = window_mean(x, smooth), window_mean(y, smooth)
    xs[np.isnan(x)], ys[np.isnan(y)] = np.nan, np.nan
    vx, vy = np.full(len(x), np.nan), np.full(len(x), np.nan)
    vx[1:-1], vy[1:-1] = (xs[2:] - xs[:-2]) * FPS / 2.0, (ys[2:] - ys[:-2]) * FPS / 2.0
    return vx, vy


def comove_by_team(players, team, ball):
    """(m_home, m_away): best co-movement score of each team's players near the ball."""
    n = len(ball)
    bx, by = ball[:, 0].astype(np.float64) * LENGTH, ball[:, 1].astype(np.float64) * WIDTH
    bvx, bvy = velocity(bx, by)
    bspeed = np.hypot(bvx, bvy)
    usable = (bspeed >= COMOVE_MIN_MS) & (bspeed <= TOUCH_MAX_MS)
    out = []
    for side in (HOME, AWAY):
        best = np.full(n, np.nan)
        for slot in np.flatnonzero(team == side):
            px = players[:, slot, 0].astype(np.float64) * LENGTH
            py = players[:, slot, 1].astype(np.float64) * WIDTH
            near = np.hypot(px - bx, py - by) <= COMOVE_R_M
            if not np.any(near & usable):
                continue
            vx, vy = velocity(px, py)
            ok = near & usable & (np.hypot(vx, vy) <= PLAYER_MAX_MS)
            score = np.where(ok, np.exp(-np.hypot(vx - bvx, vy - bvy) / COMOVE_SCALE_MS), np.nan)
            best = np.fmax(best, score)
        # Frames with a usable ball but nobody of this side near: 0, not unknown.
        best = np.where(usable & np.isnan(best), 0.0, best)
        out.append(window_mean(best, COMOVE_HALF))
    return out


def team_shape(players, team):
    """log(home spread / away spread), smoothed; NaN where either side has too few players."""
    out = []
    for side in (HOME, AWAY):
        x = players[:, team == side, 0].astype(np.float64) * LENGTH
        y = players[:, team == side, 1].astype(np.float64) * WIDTH
        count = np.sum(~np.isnan(x), axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                area = np.nanstd(x, axis=1) * np.nanstd(y, axis=1)
        area = np.where((count >= SHAPE_MIN_PLAYERS) & (area > 0), area, np.nan)
        out.append(np.log(area))
    return window_mean(out[0] - out[1], SHAPE_HALF)


def keeper_positions(players, team):
    """(n, 2, 2) keeper x, y (m) for home, away (NaN if none within KEEPER_ZONE_M of its goal),
    and (n, 2) the x of the goal each team defends."""
    px, py = players[:, :, 0] * LENGTH, players[:, :, 1] * WIDTH
    n = len(px)
    pos = np.full((n, 2, 2), np.nan)
    goal_x = np.zeros((n, 2))
    for j, side in enumerate((HOME, AWAY)):
        cols = team == side
        # Mean x of the team's tracked players (inf where none: the goal choice is then arbitrary).
        tracked = ~np.isnan(px[:, cols])
        mean_x = np.where(tracked.any(axis=1), np.nansum(px[:, cols], axis=1)
                          / np.maximum(tracked.sum(axis=1), 1), np.inf)
        gx = np.where(mean_x < LENGTH / 2.0, 0.0, LENGTH)
        goal_x[:, j] = gx
        d = np.hypot(px[:, cols] - gx[:, None], py[:, cols] - WIDTH / 2.0)
        d = np.where(np.isnan(d), np.inf, d)
        i = np.argmin(d, axis=1)
        rows = np.flatnonzero(d[np.arange(n), i] <= KEEPER_ZONE_M)
        pos[rows, j, 0] = px[:, cols][rows, i[rows]]
        pos[rows, j, 1] = py[:, cols][rows, i[rows]]
    return pos, goal_x


def goal_distances(players, team, ball):
    """Distance (m) from the ball to the goal each team defends (NaN without ball)."""
    _, goal_x = keeper_positions(players, team)
    bx, by = ball[:, 0] * LENGTH, ball[:, 1] * WIDTH
    return [np.hypot(bx - goal_x[:, j], by - WIDTH / 2.0) for j in range(2)]


def keeper_hold(ball, players, team, seen, out):
    """+1 / -1 over the start of ball gaps that begin with the ball at a keeper in his own box."""
    n = len(ball)
    hold = np.zeros(n, dtype=np.float32)
    pos, goal_x = keeper_positions(players, team)
    bx, by = ball[:, 0] * LENGTH, ball[:, 1] * WIDTH
    out_recent = int(KEEPER_OUT_S * FPS)
    for start, stop in runs_of(~seen):
        if start == 0:
            continue
        last = start - 1
        if out[max(0, start - out_recent):start].any():
            continue
        for j, sign in enumerate((1.0, -1.0)):
            d = np.hypot(bx[last] - pos[last, j, 0], by[last] - pos[last, j, 1])
            in_box = (abs(bx[last] - goal_x[last, j]) <= BOX_DEPTH_M
                      and abs(by[last] - WIDTH / 2.0) <= BOX_HALF_M)
            if d <= KEEPER_BALL_M and in_box:
                hold[start:min(stop, start + int(KEEPER_HOLD_S * FPS))] = sign
                break
    return hold


def nearest_by_team(players, team, ball):
    """Distance (m) from the ball to the nearest player of each team. NaN without ball/players."""
    dx = (players[:, :, 0] - ball[:, None, 0]) * LENGTH
    dy = (players[:, :, 1] - ball[:, None, 1]) * WIDTH
    dist = np.hypot(dx, dy)
    out = []
    for side in (HOME, AWAY):
        nearest = np.min(np.where(np.isnan(dist[:, team == side]), np.inf,
                                  dist[:, team == side]), axis=1)
        out.append(np.where(np.isinf(nearest), np.nan, nearest))
    return out


def carrier_by_team(players, team, ball, half=CARRIER_HALF):
    """Per team, the distance (m) a player stayed within of the ball over +/- half frames.

    For each track slot, the worst (largest) ball distance over the window; then the best slot of
    each team. A player running with the ball keeps it small; one the ball merely passes by, or an
    opponent closing in for a single frame, does not. A frame without ball or player inside the
    window counts as infinitely far, so this says nothing next to a gap. NaN where no slot stays.
    """
    dx = (players[:, :, 0] - ball[:, None, 0]) * LENGTH
    dy = (players[:, :, 1] - ball[:, None, 1]) * WIDTH
    dist = np.hypot(dx, dy).astype(np.float32)
    dist[np.isnan(dist)] = np.inf
    worst = dist.copy()
    for shift in range(1, half + 1):
        np.maximum(worst[shift:], dist[:-shift], out=worst[shift:])
        np.maximum(worst[:-shift], dist[shift:], out=worst[:-shift])
    out = []
    for side in (HOME, AWAY):
        best = worst[:, team == side].min(axis=1)
        out.append(np.where(np.isinf(best), np.nan, best))
    return out


def touches(ball, d_home, d_away, gap):
    """(touch, last_touch): +1/-1 on frames where home/away touches the ball (0 elsewhere), and
    the side of the most recent touch carried forward (0 before any, and after a long gap).
    """
    n = len(ball)
    x = window_mean(ball[:, 0].astype(np.float64) * LENGTH, TOUCH_SMOOTH)
    y = window_mean(ball[:, 1].astype(np.float64) * WIDTH, TOUCH_SMOOTH)
    seen = ~np.isnan(ball[:, 0])
    x[~seen], y[~seen] = np.nan, np.nan

    vx, vy = np.full(n, np.nan), np.full(n, np.nan)
    vx[1:-1], vy[1:-1] = (x[2:] - x[:-2]) * FPS / 2.0, (y[2:] - y[:-2]) * FPS / 2.0
    fast = np.hypot(vx, vy) > TOUCH_MAX_MS
    vx[fast], vy[fast] = np.nan, np.nan

    k = TOUCH_K
    dv = np.full(n, np.nan)
    dv[k:-k] = np.hypot(vx[2 * k:] - vx[:-2 * k], vy[2 * k:] - vy[:-2 * k])
    peak_ok = np.where(np.isnan(dv), -np.inf, dv)
    local_max = peak_ok.copy()
    for shift in range(1, k + 1):
        np.maximum(local_max[shift:], peak_ok[:-shift], out=local_max[shift:])
        np.maximum(local_max[:-shift], peak_ok[shift:], out=local_max[:-shift])
    peak = (peak_ok >= TOUCH_DV_MS) & (peak_ok == local_max)

    dh = np.where(np.isnan(d_home), np.inf, d_home)
    da = np.where(np.isnan(d_away), np.inf, d_away)
    # Nearest player at the peak gets the touch, even when both sides are close: that is what a
    # tackle or an interception looks like, and those are the turnovers this is for.
    near = np.minimum(dh, da) <= TOUCH_R_M
    touch = np.zeros(n, dtype=np.float32)
    hit = peak & near
    touch[hit] = np.where(dh[hit] < da[hit], 1.0, -1.0)

    # Carry the last touch forward; a long gap starts a reset (0) until the next touch.
    event = touch.copy()
    event[np.abs(dh - da) < TOUCH_MIN_MARGIN_M] = 0.0
    reset = (gap >= TOUCH_RESET_GAP_S) & np.concatenate(([True], gap[:-1] < TOUCH_RESET_GAP_S))
    has = (event != 0) | reset
    idx = np.where(has, np.arange(n), -1)
    np.maximum.accumulate(idx, out=idx)
    last = np.where(idx >= 0, event[np.maximum(idx, 0)], 0.0)
    return touch, last.astype(np.float32)


def mean_player_speed(players):
    """Mean speed (m/s) over the tracked players of each frame. NaN where none is tracked."""
    speed = players[:, :, 2]
    ok = ~np.isnan(speed)
    count = ok.sum(axis=1)
    total = np.where(ok, speed, 0.0).sum(axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def tag_team_sign(event):
    """+1 Barcelona, -1 Athletic, 0 for N/A or no Team tag."""
    team = (event["tags"].get("Team") or "").strip().lower()
    return 1 if "barcelona" in team else -1 if ("atletic" in team or "athletic" in team) else 0


def tagging_features(n_frames, events=None):
    """tag_setpiece, tag_team, tag_restart and tag_restart_team from the Smart Tagging intervals."""
    if events is None:
        from load_smart_tagging import load
        events = load()
    signs = dict(TAG_SIGN, **EXTRA_TAG_SIGN)
    setpiece = np.zeros(n_frames, dtype=np.float32)
    vote = np.zeros(n_frames, dtype=np.float32)
    restart = np.zeros(n_frames, dtype=np.float32)
    restart_team = np.zeros(n_frames, dtype=np.float32)
    phase_starts = np.array(sorted(e["start_frame"] for e in events
                                   if signs.get((e["code"] or "").strip().lower(), 0)))
    fps = GAMES[GAME]["fps"]
    for event in events:
        a, b = max(0, event["start_frame"]), min(n_frames, event["end_frame"])
        if b <= a:
            continue
        code = (event["code"] or "").strip().lower()
        if code == SETPIECE_CODE:
            setpiece[a:b] = 1.0
            # The restart is set up from the set-piece start until play resumes, which the tagger
            # marks by starting the next attack/defence phase inside the set-piece interval.
            later = phase_starts[phase_starts > a]
            end = min(b, int(later[0]) if len(later) else b, a + int(RESTART_MAX_S * fps))
            restart[a:end] = 1.0
            team = tag_team_sign(event)
            if team:
                restart_team[end:min(n_frames, end + int(RESTART_TEAM_S * fps))] = team
        sign = signs.get(code, 0)
        team = tag_team_sign(event)
        if sign and team:
            vote[a:b] += sign * team
    return setpiece, np.sign(vote).astype(np.float32), restart, restart_team


def build_features(atd=None, events=None, ball_fill=None):
    """All features as a dict of float32 arrays over every video frame.

    ball_fill: None, or a pipeline.ball_fill method; the guessed ball inside gaps then feeds
    d_home/d_away only. Everything else, including ball_seen and gap_s, reads the real track.
    """
    atd = atd if atd is not None else load_arrays("tactical")
    players, ball, team = atd["players"], atd["ball"], atd["team"]
    n = len(ball)

    seen, out, ball_speed, gap = ball_features(ball)
    if ball_fill:
        from pipeline.ball_fill import fill_ball
        d_home, d_away = nearest_by_team(players, team, fill_ball(ball, ball_fill))
    else:
        d_home, d_away = nearest_by_team(players, team, ball)
    c_home, c_away = carrier_by_team(players, team, ball)
    touch, last_touch = touches(ball, d_home, d_away, gap)
    hold = keeper_hold(ball, players, team, seen, out)
    m_home, m_away = comove_by_team(players, team, ball)
    shape = team_shape(players, team)
    g_home, g_away = goal_distances(players, team, ball)
    player_speed = window_mean(mean_player_speed(players), PLAYER_SPEED_HALF)
    setpiece, tag_team, restart, restart_team = tagging_features(n, events)

    feats = {
        "ball_seen": seen, "d_home": d_home, "d_away": d_away,
        "c_home": c_home, "c_away": c_away, "touch": touch, "last_touch": last_touch,
        "keeper_hold": hold, "m_home": m_home, "m_away": m_away,
        "shape": shape, "g_home": g_home, "g_away": g_away,
        "ball_out": out,
        "ball_speed": ball_speed, "gap_s": gap, "player_speed": player_speed,
        "tag_setpiece": setpiece, "tag_team": tag_team,
        "tag_restart": restart, "tag_restart_team": restart_team,
    }
    return {k: np.asarray(v, dtype=np.float32) for k, v in feats.items()}
