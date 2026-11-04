"""SkillCorner open data (20 A-League 2024/25 matches) as an external set for short possessions.

    python tools/skillcorner_data.py            # build all, cache, print the label profiles

Source: github.com/SkillCorner/opendata (MIT, credit SkillCorner), cloned into
data/external/skillcorner/repo with `git lfs pull` for the tracking (gitignored with the rest of
data/). Not the competition match.

Broadcast tracking at 10 fps: 22 players per frame (extrapolated off camera, `is_detected` says
which), the ball with z and `is_detected`, nothing at all during replays / cut-aways. Resampled
here to 25 fps (linear between consecutive 10 fps frames) so the arrays match DFL's and ours:
  players (n, slots, 3)  x, y normalised 0-1, speed m/s (from the positions)
  pdet    (n, slots)     player detected (not extrapolated) at the nearest 10 fps frame
  ball    (n, 3)         x, y normalised, z (m); NaN where not detected
  ball_est (n, 3)        the same with SkillCorner's extrapolated ball kept
  tracked (n,)           the frame has players (False in replays / cut-aways)
  team    (slots,)       0 home, 1 away
  period  (n,)           1 / 2
  labels  (n,)           H / A / D / X from the player possessions (below)
  events  (k, 4)         frame_start, frame_end, team 0/1, is_header of every player possession
  event_start / event_end (k,)  their start_type / end_type

Labels. `dynamic_events.csv` has one `player_possession` row per player possession (from control
to release), including short contacts: interceptions, recoveries, headers into an opponent. So
  - inside a possession: its team;
  - between two possessions in live play: the earlier team (the ball is in flight from it; the
    switch comes at the next team's first touch);
  - when the earlier one has `game_interruption_after` or the later one a restart start type
    (throw-in, free kick, goal kick, corner) or `game_interruption_before`: D, from the moment the
    ball leaves the pitch (first detected ball outside the lines after the release; else the
    release itself) to the restart (the last frame before the next possession's start at which the
    ball was still or slow, i.e. was being placed; else the next possession's start);
  - before the first / after the last possession of a period, and wherever there are no players:
    X (not graded).
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from games import DATA_DIR  # noqa: E402
import possession_metrics as pm  # noqa: E402

# Params
SRC = os.path.join(DATA_DIR, "external", "skillcorner", "repo", "data", "matches")
CACHE = os.path.join(DATA_DIR, "cache", "skillcorner")
SRC_FPS = 10
FPS = 25
PITCH = (105.0, 68.0)
H, A, D, X = 0, 1, 2, 3
LETTERS = np.array(["H", "A", "D", "X"])
RESTART_STARTS = ("throw_in", "free_kick", "goal_kick", "corner", "penalty", "kick_off")
OUT_MARGIN_M = 0.3          # the ball counts as out this far beyond a line
SLOW_MS = 2.0               # ball speed below which the restart is still being set up
SPEED_HALF_W = 2            # +/- frames (25 fps) for the speed difference


def games():
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(SRC, "*"))
                  if os.path.isdir(p))


def _resample(t_src, v, t_dst):
    """Linear resampling of v (n_src, ...) at t_dst; NaN where either neighbour is NaN or the
    neighbours are not consecutive source frames."""
    i = np.searchsorted(t_src, t_dst, side="right") - 1
    i = np.clip(i, 0, len(t_src) - 2)
    w = (t_dst - t_src[i]) / (t_src[i + 1] - t_src[i])
    consecutive = (t_src[i + 1] - t_src[i]) <= 1.0 / SRC_FPS + 1e-6
    a, b = v[i], v[i + 1]
    shape = (-1,) + (1,) * (v.ndim - 1)
    w = w.reshape(shape)
    out = a + (b - a) * w
    exact = np.isclose(w, 0).reshape(-1)
    out[exact] = a[exact]
    bad = ~consecutive & ~exact
    out[bad] = np.nan
    out[(w.reshape(-1) < 0) | (w.reshape(-1) > 1)] = np.nan
    return out


def _speed(xy):
    """Speed in m/s from normalised x, y at FPS (central difference over SPEED_HALF_W)."""
    k = SPEED_HALF_W
    d = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
    dx = (xy[2 * k:, ..., 0] - xy[:-2 * k, ..., 0]) * PITCH[0]
    dy = (xy[2 * k:, ..., 1] - xy[:-2 * k, ..., 1]) * PITCH[1]
    d[k:-k] = np.hypot(dx, dy) * FPS / (2 * k)
    return d


def parse(game):
    """10 fps arrays: frames, period, players (m, slots, 2), pdet, ball (m, 3), bdet, tracked."""
    meta = json.load(open(os.path.join(SRC, game, f"{game}_match.json"), encoding="utf-8"))
    home = meta["home_team"]["id"]
    people = sorted(meta["players"], key=lambda p: (p["team_id"] != home, p["id"]))
    slot = {p["id"]: i for i, p in enumerate(people)}
    team = np.array([0 if p["team_id"] == home else 1 for p in people], dtype=np.int8)
    rows = [json.loads(line) for line in
            open(os.path.join(SRC, game, f"{game}_tracking_extrapolated.jsonl"), encoding="utf-8")]
    rows = [r for r in rows if r["period"] in (1, 2)]
    m, s = len(rows), len(people)
    frame = np.array([r["frame"] for r in rows], dtype=np.int64)
    period = np.array([r["period"] for r in rows], dtype=np.int8)
    pl = np.full((m, s, 2), np.nan, dtype=np.float32)
    pdet = np.zeros((m, s), dtype=bool)
    ball = np.full((m, 3), np.nan, dtype=np.float32)
    bdet = np.zeros(m, dtype=bool)
    for j, r in enumerate(rows):
        for p in r["player_data"]:
            i = slot.get(p["player_id"])
            if i is None:
                continue
            pl[j, i] = ((p["x"] + PITCH[0] / 2) / PITCH[0], (p["y"] + PITCH[1] / 2) / PITCH[1])
            pdet[j, i] = bool(p.get("is_detected"))
        b = r.get("ball_data") or {}
        if b.get("x") is not None:
            ball[j] = ((b["x"] + PITCH[0] / 2) / PITCH[0], (b["y"] + PITCH[1] / 2) / PITCH[1],
                       b.get("z") if b.get("z") is not None else np.nan)
            bdet[j] = bool(b.get("is_detected"))
    tracked = np.array([len(r["player_data"]) > 0 for r in rows])
    return dict(frame=frame, period=period, players=pl, pdet=pdet, ball=ball, bdet=bdet,
                tracked=tracked, team=team, home=home)


def possessions(game):
    d = pd.read_csv(os.path.join(SRC, game, f"{game}_dynamic_events.csv"), low_memory=False)
    p = d[d.event_type == "player_possession"]
    return p.sort_values(["period", "frame_start"]).reset_index(drop=True)


def _is_restart(start_type):
    return isinstance(start_type, str) and start_type.startswith(RESTART_STARTS)


def label(n, idx, period, ball, speed, poss, home):
    """H / A / D / X per output frame from the player possessions (see the module docstring).
    idx(f) maps a SkillCorner 10 fps frame number to an output index."""
    lab = np.full(n, X, dtype=np.int8)
    x_out = (np.abs(ball[:, 0] - 0.5) * PITCH[0] > PITCH[0] / 2 + OUT_MARGIN_M) | \
            (np.abs(ball[:, 1] - 0.5) * PITCH[1] > PITCH[1] / 2 + OUT_MARGIN_M)
    slow = speed < SLOW_MS
    rows = list(poss.itertuples(index=False))
    for k, r in enumerate(rows):
        a, b = idx(r.frame_start), idx(r.frame_end)
        side = H if r.team_id == home else A
        lab[a:b + 1] = side
        if k + 1 >= len(rows) or rows[k + 1].period != r.period:
            continue
        q = rows[k + 1]
        c = idx(q.frame_start)
        if c <= b + 1:
            continue
        dead = (isinstance(r.game_interruption_after, str) or
                isinstance(q.game_interruption_before, str) or _is_restart(q.start_type))
        if not dead:
            lab[b + 1:c] = side
            continue
        gap = np.arange(b + 1, c)
        out = gap[x_out[gap]]
        d0 = out[0] if len(out) else b + 1
        still = gap[slow[gap] & (gap > d0)]
        d1 = still[-1] + 1 if len(still) else c
        lab[b + 1:d0] = side
        lab[d0:d1] = D
        # d1..c: the restart pass in flight, the restart team's
        lab[d1:c] = H if q.team_id == home else A
    return lab


def build(game):
    src = parse(game)
    poss = possessions(game)
    t_src = src["frame"] / SRC_FPS
    chunks, offsets = [], {}
    for per in (1, 2):
        sel = src["period"] == per
        t = t_src[sel]
        t_out = np.arange(t[0], t[-1] + 1e-9, 1.0 / FPS)
        offsets[per] = (sum(len(c["t"]) for c in chunks), t[0])
        near = np.clip(np.round((t_out - t[0]) * SRC_FPS).astype(int), 0, sel.sum() - 1)
        pl = _resample(t, src["players"][sel], t_out)
        ball_est = _resample(t, src["ball"][sel], t_out)
        bdet = src["bdet"][sel][near] & ~np.isnan(ball_est[:, 0])
        chunks.append(dict(t=t_out, players=pl, ball_est=ball_est, bdet=bdet,
                           pdet=src["pdet"][sel][near], tracked=src["tracked"][sel][near],
                           period=np.full(len(t_out), per, dtype=np.int8)))
    out = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0] if k != "t"}
    n = len(out["period"])

    def idx(f, per=None):
        per = per or (1 if f < src["frame"][src["period"] == 2][0] else 2)
        off, t0 = offsets[per]
        return int(np.clip(off + round((f / SRC_FPS - t0) * FPS), 0, n - 1))

    speed = _speed(out["players"])
    out["players"] = np.concatenate([out["players"], speed[..., None]], axis=-1)
    ball = out["ball_est"].copy()
    ball[~out["bdet"]] = np.nan
    out["ball"] = ball
    bspeed = _speed(out["ball_est"][:, None, :2])[:, 0]
    out["labels"] = label(n, idx, out["period"], out["ball_est"], bspeed, poss, src["home"])
    out["labels"][~out["tracked"]] = X
    out["team"] = src["team"]
    out["events"] = np.array([(idx(r.frame_start), idx(r.frame_end), int(r.team_id != src["home"]),
                               int(bool(r.is_header))) for r in poss.itertuples(index=False)],
                             dtype=np.int64)
    out["event_start"] = np.array(poss.start_type.astype(str).tolist(), dtype="U40")
    out["event_end"] = np.array(poss.end_type.astype(str).tolist(), dtype="U40")
    out["tracked_frac_poss"] = np.array(np.mean(out["tracked"][out["labels"] != X]))
    del out["bdet"]
    return out


def load(game, rebuild=False):
    path = os.path.join(CACHE, f"{game}.npz")
    if not rebuild and os.path.isfile(path):
        with np.load(path, allow_pickle=False) as c:
            return {k: c[k] for k in c.files}
    out = build(game)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def profile(lab):
    """(possessions per live minute, median length s, share < 2 s, share < 5 s, D share)."""
    runs = pm.valid_runs(lab)
    lengths = np.array([(b - a) / FPS for a, b, _ in pm.possession_spells(lab, runs)])
    graded = lab != X
    live_min = np.sum(graded & (lab != D)) / FPS / 60
    return (len(lengths) / live_min, np.median(lengths), np.mean(lengths < 2),
            np.mean(lengths < 5), np.mean(lab[graded] == D))


def main():
    gs = sys.argv[1:] or games()
    row = "{:<8} {:>8.1f} {:>8.1f}s {:>6.0f}% {:>6.0f}% {:>5.0f}% {:>7.0f}% {:>6.0f}% {:>6.0f}%"
    print("{:<8} {:>8} {:>9} {:>7} {:>7} {:>6} {:>8} {:>7} {:>7}".format(
        "match", "poss/min", "median", "<2s", "<5s", "D", "graded", "ball", "plDet"))
    for g in gs:
        d = load(g, rebuild=True)
        lab = d["labels"]
        p = np.array(profile(lab)) * [1, 1, 100, 100, 100]
        g_ok = lab != X
        print(row.format(g, *p, 100 * g_ok.mean(), 100 * np.mean(~np.isnan(d["ball"][g_ok, 0])),
                         100 * d["pdet"][g_ok].mean() * len(d["team"]) / 22), flush=True)
    print("{:<8} {:>8.1f} {:>8.1f}s {:>6.0f}%".format("public", 4.2, 7.6, 12))


if __name__ == "__main__":
    main()
