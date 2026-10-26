"""Touch labels on DFL: who touched the ball, and exactly when. Aggregate output only.

    python tools/touch_labels.py            # build for all 7 matches, print the checks

Teacher, three sources merged (duplicates of the same team within MERGE_S kept once, the
earlier source winning):
  0 ball     DFL's raw ball is sharp (tools/ball_realism.py), so a touch shows as a peak of the
             velocity change |v(t+K) - v(t-K)| above DV_MS, with a player within REACH_M; the
             touch goes to that player's team. Only while the provider marks the ball alive.
             Finds kicks, receptions, deflections and headers alike: 83% of tagged passes have
             one within 1 s, but only about half of dribbles and tackles (the ball moves less).
  1 restart  the provider's BallStatus going dead -> alive: the restart touch, by the team that
             then has the ball. Source 0 cannot see these (the restart frame is still flagged
             dead: 0-15% of tagged throw-ins, free kicks, goal kicks and corners found).
  2 switch   the provider's BallPossession switching to the other team while alive: the new
             team's first touch (catches the dribbles and tackles source 0 misses).

Checked against two independent sources:
  events    the hand-tagged on-ball events (passes, shots, ball actions, tackles, claims and
            restarts; time-corrected per match): share with a teacher touch by the same team within
            MATCH_S, and the median offset
  flags     the provider's BallPossession switches to the other team at a touch: share of those
            switches with a teacher touch by the new team within MATCH_S

The student (the touch network) never sees this sharp ball: it gets the blurred, gapped one.
Cached per match in data/cache/dfl/<match>_touches.npz: frame, team (0 home / 1 guest), dv.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import dfl_data as dd  # noqa: E402
from pipeline.features import window_mean  # noqa: E402

# Params
K = 2               # velocity change over +/- K frames
SMOOTH = 1          # +/- frames of position smoothing first (DFL's raw ball is a little noisy)
DV_MS = 2.0         # minimum velocity change for a touch
REACH_M = 2.0       # nearest player at the peak within this
MAX_MS = 40.0       # faster than this is a tracker jump
MATCH_S = 0.3       # a touch this close to an event / flag switch counts as the same
MERGE_S = 0.2       # same-team touches from different sources this close are one touch
SOURCES = ("ball", "restart", "switch")


def teacher_touches(d):
    """(frames, teams, dv) of the touches on the raw ball, sorted by frame."""
    L, W = dd.PITCH
    ball, players, team = d["ball"], d["players"], d["team"]
    x = window_mean(ball[:, 0].astype(np.float64) * L, SMOOTH)
    y = window_mean(ball[:, 1].astype(np.float64) * W, SMOOTH)
    n = len(x)
    vx, vy = np.full(n, np.nan), np.full(n, np.nan)
    vx[1:-1], vy[1:-1] = (x[2:] - x[:-2]) * dd.FPS / 2.0, (y[2:] - y[:-2]) * dd.FPS / 2.0
    fast = np.hypot(vx, vy) > MAX_MS
    vx[fast], vy[fast] = np.nan, np.nan
    dv = np.full(n, np.nan)
    dv[K:-K] = np.hypot(vx[2 * K:] - vx[:-2 * K], vy[2 * K:] - vy[:-2 * K])
    peak = np.where(np.isnan(dv), -np.inf, dv)
    local = peak.copy()
    for s in range(1, K + 1):
        np.maximum(local[s:], peak[:-s], out=local[s:])
        np.maximum(local[:-s], peak[s:], out=local[:-s])
    cand = np.flatnonzero((peak >= DV_MS) & (peak == local) & (d["labels"] != dd.D)
                          & (d["labels"] != dd.X))
    px = players[cand, :, 0].astype(np.float64) * L
    py = players[cand, :, 1].astype(np.float64) * W
    dist = np.hypot(px - x[cand, None], py - y[cand, None])
    dist = np.where(np.isnan(dist), np.inf, dist)
    who = np.argmin(dist, axis=1)
    ok = dist[np.arange(len(cand)), who] <= REACH_M
    return cand[ok], team[who[ok]].astype(np.int8), dv[cand[ok]].astype(np.float32)


def flag_touches(lab):
    """(restart frames, teams), (switch frames, teams) from the provider's flags."""
    alive = np.isin(lab, (dd.H, dd.A))
    start = np.flatnonzero(alive[1:] & (lab[:-1] == dd.D)) + 1
    switch = np.flatnonzero(alive[1:] & alive[:-1] & (lab[1:] != lab[:-1])) + 1
    return ((start, (lab[start] == dd.A).astype(np.int8)),
            (switch, (lab[switch] == dd.A).astype(np.int8)))


def merged_touches(d):
    """(frames, teams, sources) of all teacher touches, sorted by frame."""
    f0, t0, _ = teacher_touches(d)
    (f1, t1), (f2, t2) = flag_touches(d["labels"])
    gap = int(round(MERGE_S * dd.FPS))
    frames, teams, sources = list(f0), list(t0), [0] * len(f0)
    for src, (fs, ts) in ((1, (f1, t1)), (2, (f2, t2))):
        have = np.array(sorted(zip(frames, teams)))
        for f, t in zip(fs, ts):
            lo, hi = np.searchsorted(have[:, 0], [f - gap, f + gap + 1]) if len(have) else (0, 0)
            if not np.any(have[lo:hi, 1] == t):
                frames.append(int(f))
                teams.append(int(t))
                sources.append(src)
    order = np.argsort(frames, kind="stable")
    return (np.array(frames)[order].astype(np.int64), np.array(teams)[order].astype(np.int8),
            np.array(sources)[order].astype(np.int8))


def load(game, rebuild=False):
    """Cached merged_touches(): (frames, teams, sources), moved onto the stretched clock when
    dfl_data.set_warp is on (the touches always come from the sharp, unwarped ball)."""
    k, dd.WARP = dd.WARP, 1.0
    try:
        path = os.path.join(dd.CACHE, "{}_touches.npz".format(game))
        if not rebuild and os.path.isfile(path):
            with np.load(path) as c:
                f, t, s = c["frame"], c["team"], c["source"]
        else:
            f, t, s = merged_touches(dd.load(game))
            np.savez(path, frame=f, team=t, source=s)
        if k != 1.0:
            f = dd.warp_frames(f, dd.load(game)["period"], k)
    finally:
        dd.WARP = k
    return f, t, s


def nearest_same_team(frames, teams, targets, target_teams, tol):
    """For each target, the signed offset (frames) to the nearest touch of its team within tol,
    or None."""
    out = []
    for f, t in zip(targets, target_teams):
        lo, hi = np.searchsorted(frames, [f - tol, f + tol + 1])
        idx = [i for i in range(lo, hi) if teams[i] == t]
        out.append(min((frames[i] - f for i in idx), key=abs) if idx else None)
    return out


def main():
    tol = int(round(MATCH_S * dd.FPS))
    print("{:<8} {:>9} {:>20} {:>11} {:>16} {:>10} {:>14}".format(
        "match", "touch/min", "ball/restart/switch", "events hit", "event offset", "flags hit",
        "share on team"))
    for g in dd.GAMES:
        d = dd.load(g)
        frames, teams, sources = load(g, rebuild=True)
        live_min = np.sum((d["labels"] == dd.H) | (d["labels"] == dd.A)) / dd.FPS / 60
        ev = d["events"]
        kinds = d["event_kind"]
        on_ball = np.isin(kinds, ("Play", "ShotAtGoal", "OtherBallAction", "BallClaiming",
                                  "ThrowIn", "FreeKick", "GoalKick", "CornerKick"))
        offs = nearest_same_team(frames, teams, ev[on_ball, 0], ev[on_ball, 1], tol)
        hit = [o for o in offs if o is not None]
        lab = d["labels"]
        switch = np.flatnonzero((lab[1:] != lab[:-1]) & np.isin(lab[1:], (dd.H, dd.A))
                                & np.isin(lab[:-1], (dd.H, dd.A))) + 1
        new_team = (lab[switch] == dd.A).astype(np.int8)
        foffs = nearest_same_team(frames, teams, switch, new_team, tol)
        # Touches by the team the flags say has the ball (a receiving or passing touch), vs not.
        own = np.mean(teams == (lab[frames] == dd.A))
        print("{:<8} {:>9.1f} {:>20} {:>10.0f}% {:>+9.2f} s (n={:>3}) {:>9.0f}% {:>13.0f}%".format(
            g, len(frames) / live_min, "/".join(str(int(np.sum(sources == k))) for k in range(3)),
            100 * len(hit) / max(len(offs), 1),
            np.median(hit) / dd.FPS if hit else np.nan, len(offs),
            100 * np.mean([o is not None for o in foffs]), 100 * own), flush=True)


if __name__ == '__main__':
    main()
