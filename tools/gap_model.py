"""Cut ball gaps like ours into an external match whose ball is always tracked (DFL).

    from gap_model import degrade
    ball = degrade(ball, players, seed=0)         # copy with NaN gaps
    python tools/gap_model.py                     # fit on our match, then check on DFL

Our match has the ball missing 41% of the time inside the halves, in gaps of 3 s or more (shorter
ones are filled by the provider). Gaps start in any phase of play but end when play speeds up:
81% of the time is a gap when the players are slow, 10% when they run. So the model is a two-state
chain over frames with, per band of the CURRENT mean player pace:
  - the chance per frame that a gap starts (while the ball is seen), and
  - the chance per frame that a gap ends, once it has lasted MIN_GAP_S.
Fitted on our match once and cached as aggregates only (eight rates, no positions) in
data/cache/gap_model.npz.

Pace alone does not carry over: in our match a gap usually comes with a stoppage, so the players
slow down after it starts, while a gap started at random in fast DFL play stays in fast play. So
calibrate() rescales the per-band rates on the DFL matches until the share of frames without ball
in each pace band matches ours (the target, measured on our match by fit()).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from games import DATA_DIR, GAME, half_video_blocks  # noqa: E402

# Params
PATH = os.path.join(DATA_DIR, "cache", "gap_model.npz")
FPS = 25
MIN_GAP_S = 3.0
PACE_EDGES = (0.0, 1.0, 2.0, 3.0, 1e9)   # m/s, mean player speed over +/- 1 s
PACE_HALF = FPS
CALIBRATE_ROUNDS = 8


def pace(players):
    """Mean tracked-player speed (m/s) per frame, averaged over +/- PACE_HALF frames."""
    from pipeline.features import mean_player_speed, window_mean
    return window_mean(mean_player_speed(players), PACE_HALF)


def band(values):
    return np.clip(np.searchsorted(PACE_EDGES, np.nan_to_num(values, nan=1.5), side="right") - 1,
                   0, len(PACE_EDGES) - 2)


def runs_of(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def fit():
    """Start and end chances per frame and pace band, from our match."""
    from atd_cache import load_arrays
    atd = load_arrays("tactical")
    seen = ~np.isnan(atd["ball"][:, 0])
    in_half = np.zeros(len(seen), bool)
    for _, first, last in half_video_blocks(GAME):
        in_half[first:last + 1] = True
    b = band(pace(atd["players"]))
    gaps = [(s, e) for s, e in runs_of(~seen & in_half) if e - s >= MIN_GAP_S * FPS]
    min_len = int(MIN_GAP_S * FPS)
    can_end = np.zeros(len(seen), bool)
    for s, e in gaps:
        can_end[s + min_len:e] = True
    starts, ends = np.array([s for s, _ in gaps]), np.array([e - 1 for _, e in gaps])
    start_p, end_p = [], []
    for k in range(len(PACE_EDGES) - 1):
        start_p.append(np.sum(b[starts] == k) / max(np.sum(in_half & seen & (b == k)), 1))
        end_p.append(np.sum(b[ends] == k) / max(np.sum(can_end & (b == k)), 1))
    share = [np.mean(~seen[in_half & (b == k)]) for k in range(len(PACE_EDGES) - 1)]
    out = {"start": np.array(start_p), "end": np.array(end_p), "target": np.array(share)}
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    np.savez(PATH, **out)
    return out


def model():
    if not os.path.isfile(PATH):
        return fit()
    with np.load(PATH) as c:
        return {k: c[k] for k in c.files}


def calibrate(matches, rounds=CALIBRATE_ROUNDS):
    """Rescale start/end rates until the gap share per pace band on `matches` [(ball, players)]
    matches the target; saves and returns the model."""
    m = model()
    for _ in range(rounds):
        sims = []
        for ball, players in matches:
            g = np.isnan(_degrade_with(m, ball, players, 0)[:, 0])
            b = band(pace(players))
            sims.append([np.mean(g[b == k]) for k in range(len(PACE_EDGES) - 1)])
        sim = np.clip(np.mean(sims, axis=0), 1e-3, 0.999)
        target = np.clip(m["target"], 1e-3, 0.999)
        # odds ratio between target and simulation, split between starting more and ending later
        r = (target / (1 - target)) / (sim / (1 - sim))
        m["start"] = m["start"] * np.sqrt(r)
        m["end"] = m["end"] / np.sqrt(r)
    np.savez(PATH, **m)
    return m


def degrade(ball, players, seed=0):
    """Copy of ball with gaps cut in by the two-state chain (see the module docstring)."""
    return _degrade_with(model(), ball, players, seed)


def _degrade_with(m, ball, players, seed):
    rng = np.random.default_rng(seed)
    b = band(pace(players))
    u = rng.random(len(ball))
    p_start, p_end = m["start"][b], m["end"][b]
    min_len = int(MIN_GAP_S * FPS)
    gap = np.zeros(len(ball), bool)
    t, n = 0, len(ball)
    while t < n:
        if u[t] >= p_start[t]:
            t += 1
            continue
        e = t + min_len
        while e < n and u[e] >= p_end[e]:
            e += 1
        gap[t:min(n, e + 1)] = True
        t = e + 1
    out = ball.copy()
    out[gap] = np.nan
    return out


def main():
    m = fit()
    print("fitted on our match, by pace band: gap starts per 10 min of seen time "
          + " / ".join("{:.1f}".format(r * FPS * 600) for r in m["start"])
          + " | mean extra gap length once past {:g} s ".format(MIN_GAP_S)
          + " / ".join("{:.0f}s".format(1 / max(r, 1e-9) / FPS) for r in m["end"]))
    import dfl_data as dd
    data = [dd.load(g) for g in dd.GAMES]
    m = calibrate([(d["ball"], d["players"]) for d in data])
    print("calibrated on DFL, by pace band: gap starts per 10 min "
          + " / ".join("{:.1f}".format(r * FPS * 600) for r in m["start"])
          + " | mean extra gap length " + " / ".join("{:.0f}s".format(1 / max(r, 1e-9) / FPS)
                                                    for r in m["end"]))
    shares = []
    for d in data:
        ball = degrade(d["ball"], d["players"], seed=0)
        seen = ~np.isnan(ball[:, 0])
        b = band(pace(d["players"]))
        live = d["controlled"] != dd.D
        shares.append([np.mean(~seen)] + [np.mean(~seen[b == k]) for k in range(len(PACE_EDGES) - 1)]
                      + [np.mean(~seen[live]), np.mean(~seen[~live])])
    s = 100 * np.mean(shares, axis=0)
    print("DFL degraded (mean of 7): ball missing {:.0f}% | by pace band {:.0f}% / {:.0f}% / {:.0f}% "
          "/ {:.0f}% | in live play {:.0f}%, in stoppages {:.0f}%".format(*s))
    print("our match:                ball missing 41% | by pace band "
          + " / ".join("{:.0f}%".format(100 * t) for t in m["target"]))


if __name__ == '__main__':
    main()
