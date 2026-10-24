"""Metrica Sports open sample data (3 matches) as an external validation and training set.

    python tools/metrica_data.py            # load all 3, build labels, cache, print a summary

Source: https://github.com/metrica-sports/sample-data (downloaded to data/external/metrica/,
gitignored with the rest of data/). Public open data, not the competition match.

For each game this gives what our pipeline expects for BAR-ATH:
  players (n, slots, 3)  x, y normalised 0-1, speed m/s (computed here; Metrica has none)
  ball    (n, 3)         x, y normalised, NaN where the ball is not tracked
  team    (slots,)       0 home, 1 away
  period  (n,)           1 / 2, 0 outside play
and frame labels H/A/D/X built from the event data:
  - a team has the ball from its controlled actions (pass, carry, shot, set piece; a recovery
    counts from its own frame when that team's next controlled action confirms it) until the
    other team's;
  - D from a stoppage (ball out, foul, shot out or goal) until the next controlled action (the
    restart);
  - X outside the periods.
A pass in flight stays with the passer's team until someone else's event, the convention the
competition labels appear to follow as well (to check against the public window).
"""

import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import DATA_DIR  # noqa: E402

# Params
SRC = os.path.join(DATA_DIR, "external", "metrica")
CACHE = os.path.join(DATA_DIR, "cache", "metrica")
FPS = 25
PITCH = (105.0, 68.0)
GAMES = (1, 2, 3)
H, A, D, X = 0, 1, 2, 3
LETTERS = np.array(["H", "A", "D", "X"])
# Controlled actions start or continue a possession. A RECOVERY counts (from its own frame) only when
# the same team's next controlled action confirms it; BALL LOST and unconfirmed recoveries don't.
# Calibrated on the public window: with every recovery / ball-lost event counting, Metrica had 5.8
# possessions per live minute, median 4.9 s, against the competition labels' 4.2 and 7.6 s; with
# this rule it is about 4 per live minute (see JOURNAL.md).
# Only event types all three games record: game 3 alone has CARRY, and using it gave game 3 twice
# the possessions of games 1-2 (374 vs ~200), which a model trained across games cannot reconcile.
OWN = {"PASS", "SHOT", "SET PIECE"}
CONFIRMED = {"RECOVERY"}
# A recovery followed by the same team's BALL LOST also confirms it: they had the ball, briefly.
# Possessions per live minute then 5.1 / 4.5 / 4.9 (public window: 4.2), median 6.8 / 8.6 / 7.0 s
# (public: 7.6 s).
CONFIRM_WITH = {"BALL LOST"}
SPEED_HALF = 5   # +/- frames for player speed


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def tracking_csv(game):
    """Games 1 and 2: one CSV per team, each with the ball in its last two columns."""
    sides, ball, period = [], None, None
    for name in ("Home", "Away"):
        path = os.path.join(SRC, "Sample_Game_{}_RawTrackingData_{}_Team.csv".format(game, name))
        with open(path) as handle:
            rows = list(csv.reader(handle))
        header = rows[2]
        body = np.array([[_float(v) for v in r] for r in rows[3:]], dtype=np.float64)
        cols = [i for i, h in enumerate(header) if h.startswith("Player")]
        xy = np.stack([np.stack([body[:, i], body[:, i + 1]], axis=1) for i in cols], axis=1)
        sides.append(xy)
        b = header.index("Ball")
        ball = body[:, b:b + 2]
        period = body[:, 0].astype(int)
    players = np.concatenate(sides, axis=1)
    team = np.array([0] * sides[0].shape[1] + [1] * sides[1].shape[1], dtype=np.int8)
    return players, ball, team, period


def tracking_epts(game):
    """Game 3: EPTS (the same format as the competition ATD), read through kloppy."""
    from kloppy import metrica
    ds = metrica.load_tracking_epts(
        meta_data=os.path.join(SRC, "Sample_Game_{}_metadata.xml".format(game)),
        raw_data=os.path.join(SRC, "Sample_Game_{}_tracking.txt".format(game)))
    home, away = ds.metadata.teams
    ids = [p.player_id for p in home.players] + [p.player_id for p in away.players]
    team = np.array([0] * len(home.players) + [1] * len(away.players), dtype=np.int8)
    n = len(ds.frames)
    players = np.full((n, len(ids), 2), np.nan)
    ball = np.full((n, 2), np.nan)
    period = np.zeros(n, dtype=int)
    slot = {pid: i for i, pid in enumerate(ids)}
    for k, fr in enumerate(ds.frames):
        period[k] = fr.period.id
        if fr.ball_coordinates is not None:
            ball[k] = (fr.ball_coordinates.x, fr.ball_coordinates.y)
        for player, data in fr.players_data.items():
            if data.coordinates is not None and player.player_id in slot:
                players[k, slot[player.player_id]] = (data.coordinates.x, data.coordinates.y)
    return players, ball, team, period


def player_speed(players):
    """Speed (m/s) per player from smoothed positions."""
    n = len(players)
    x, y = players[:, :, 0] * PITCH[0], players[:, :, 1] * PITCH[1]
    speed = np.full(x.shape, np.nan)
    k = SPEED_HALF
    speed[k:n - k] = np.hypot(x[2 * k:] - x[:-2 * k], y[2 * k:] - y[:-2 * k]) * FPS / (2.0 * k)
    return speed


def events(game):
    """[(frame_start, frame_end, team 0/1, type, subtype)] sorted by start frame (0-based)."""
    out = []
    if game in (1, 2):
        path = os.path.join(SRC, "Sample_Game_{}_RawEventsData.csv".format(game))
        with open(path) as handle:
            for r in csv.DictReader(handle):
                out.append((int(r["Start Frame"]) - 1, int(r["End Frame"]) - 1,
                            0 if r["Team"] == "Home" else 1, r["Type"], r["Subtype"] or ""))
    else:
        with open(os.path.join(SRC, "Sample_Game_{}_events.json".format(game))) as handle:
            data = json.load(handle)["data"]
        home_id = data[0]["team"]["id"] if data[0]["team"]["name"] == "Team A" else None
        for e in data:
            team = 0 if e["team"]["name"] == "Team A" else 1
            sub = e["subtypes"]
            if isinstance(sub, list):
                sub = "-".join(s["name"] for s in sub)
            elif isinstance(sub, dict):
                sub = sub["name"]
            out.append((int(e["start"]["frame"]) - 1, int(e["end"]["frame"] or e["start"]["frame"]) - 1,
                        team, e["type"]["name"], sub or ""))
        del home_id
    return sorted(out, key=lambda e: e[0])


def labels(evts, period):
    """H/A/D/X per frame from the events (see the module docstring)."""
    n = len(period)
    assertions = []                       # (frame, state)
    for i, (start, end, team, kind, sub) in enumerate(evts):
        if kind in OWN:
            assertions.append((start, H if team == 0 else A))
        elif kind in CONFIRMED:
            nxt = next((e for e in evts[i + 1:] if e[3] in OWN or e[3] in CONFIRM_WITH), None)
            if nxt is not None and nxt[2] == team:
                assertions.append((start, H if team == 0 else A))
        stop = None
        if kind == "BALL OUT":
            stop = max(start, end)
        elif kind == "FAULT RECEIVED":
            stop = start
        elif kind == "SHOT" and ("OUT" in sub or "GOAL" in sub) and "SAVED" not in sub:
            stop = max(start, end)
        if stop is not None:
            assertions.append((stop, D))
    assertions.sort(key=lambda a: a[0])
    codes = np.full(n, X, dtype=np.int8)
    state, j = X, 0
    for t in range(n):
        while j < len(assertions) and assertions[j][0] <= t:
            state = assertions[j][1]
            j += 1
        if period[t] > 0:
            codes[t] = state
    # Before a period's first event (players walking out), and anything still X inside a period,
    # stays X: not graded, like the competition's pre-kickoff frames.
    for p in (1, 2):
        idx = np.flatnonzero(period == p)
        if not len(idx):
            continue
        first = [e[0] for e in evts if idx[0] <= e[0] <= idx[-1]]
        if first:
            codes[idx[0]:first[0]] = X
    return codes


def load(game, rebuild=False):
    """Cached arrays + labels for one game."""
    path = os.path.join(CACHE, "game{}.npz".format(game))
    if not rebuild and os.path.isfile(path):
        with np.load(path) as c:
            return {k: c[k] for k in c.files}
    xy, ball, team, period = tracking_csv(game) if game in (1, 2) else tracking_epts(game)
    speed = player_speed(xy)
    players = np.concatenate([xy, speed[:, :, None]], axis=2).astype(np.float32)
    ball = np.concatenate([ball, np.full((len(ball), 1), np.nan)], axis=1).astype(np.float32)
    out = {"players": players, "ball": ball, "team": team, "period": period,
           "labels": labels(events(game), period)}
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, **out)
    return out


def main():
    import possession_metrics as pm
    for g in GAMES:
        d = load(g, rebuild=True)
        lab, per = d["labels"], d["period"]
        graded = lab != X
        runs = pm.valid_runs(lab)
        spells = pm.possession_spells(lab, runs)
        seen = ~np.isnan(d["ball"][:, 0])
        print("game {}: {} frames ({:.0f} min), graded {:.0f} min | H {:.0f}% A {:.0f}% D {:.0f}% | "
              "possessions {} (median {:.1f}s) | ball tracked {:.0f}% of graded | players/frame {:.1f}"
              .format(g, len(lab), len(lab) / FPS / 60, graded.sum() / FPS / 60,
                      *(100 * np.mean(lab[graded] == k) for k in (H, A, D)),
                      len(spells), np.median([(b - a) / FPS for a, b, _ in spells]),
                      100 * seen[graded].mean(),
                      np.mean(np.sum(~np.isnan(d["players"][graded, :, 0]), axis=1))))


if __name__ == '__main__':
    sys.path.insert(0, os.path.join(ROOT, "evaluation"))
    main()
