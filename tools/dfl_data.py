"""DFL / Sportec open data (IDSSE, 7 German league matches) as an external validation and training set.

    python tools/dfl_data.py            # parse all 7, cache, print the label profiles

Source: Bassek, Weber, Rein & Memmert, "An integrated dataset of spatiotemporal and event data in
elite soccer", Scientific Data (2025), CC BY 4.0. Files from figshare (the IDs kloppy uses), stored
in data/external/dfl/raw/ (gitignored with the rest of data/). Not the competition match.

    curl -L https://ndownloader.figshare.com/files/<id>    # springernature.figshare.com blocks scripts

Real optical tracking at 25 fps, ball included during stoppages. Every ball frame carries the
provider's own BallStatus (1 alive, 0 dead) and BallPossession (1 home, 2 guest): the same kind of
mechanical per-frame label the competition's ground truth is derived from (CHALLENGE.md).

For each match, the arrays our pipeline expects for BAR-ATH:
  players (n, slots, 3)  x, y normalised 0-1, speed m/s (the file's S is km/h)
  ball    (n, 3)         x, y normalised, z (m); NaN where not tracked
  team    (slots,)       0 home, 1 guest
  period  (n,)           1 / 2
  labels  (n,)           H / A / D from BallStatus and BallPossession ("flags")
  controlled (n,)        the flags with uncontrolled possessions removed (see below)
  events  (k, 3)         frame, team 0/1, controlled 0/1 of the on-ball events; event_kind (k,)
The two halves are concatenated; period says which is which.

The flags switch on any touch: 6.5-11 possessions per live minute, median 3-4.7 s, against the
public window's 4.2 and 7.6 s. The competition counts only CONTROLLED possession ("a touch without
control, a deflection... does not count"). So `controlled` keeps a flagged possession only if its
team makes a controlled action inside it (+/- EVENT_TOL_S): a pass or cross, a shot, a ball action
that is not a clearance, a won tackle with PossessionChange, a keeper claim, or a restart. Otherwise
it goes to the team before it (the team after it, for the first one of a live stretch).

Event times are hand-entered on the wall clock and offset from the tracking by a different amount
per match (-0.1 to -0.8 s), scattered by about +/- 0.6 s per event. The offset is estimated per
match from the passes (event_lag) and applied.
"""

import os
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from games import DATA_DIR  # noqa: E402
import possession_metrics as pm  # noqa: E402

# Params
SRC = os.path.join(DATA_DIR, "external", "dfl", "raw")
CACHE = os.path.join(DATA_DIR, "cache", "dfl")
FPS = 25
PITCH = (105.0, 68.0)
GAMES = ("J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9")
H, A, D, X = 0, 1, 2, 3
LETTERS = np.array(["H", "A", "D", "X"])
KMH = 3.6                  # the S attribute is km/h (checked against positions: ratio 1.00)
SECTIONS = ("firstHalf", "secondHalf")
EVENT_TOL_S = 1.0          # a controlled action this close to a possession counts for it
LAG_SEARCH = 50            # +/- frames searched for the per-match event offset
RESTARTS = ("ThrowIn", "FreeKick", "GoalKick", "CornerKick", "Penalty")
# Our ball track is time-blurred: DFL's ball smoothed with a Gaussian of this many frames (0.24 s)
# matches our velocity-change peak profile at every scale (tools/ball_realism.py, JOURNAL.md).
BLUR_SIGMA = 6

_FRAMESET = re.compile(r'<FrameSet GameSection="(\w+)"[^>]*TeamId="([^"]+)"[^>]*PersonId="([^"]+)"')
_FRAME = re.compile(r'<Frame N="(\d+)"[^>]*? X="([-\d.]+)" Y="([-\d.]+)"(?: Z="([-\d.]+)")?'
                    r'[^>]*? S="([-\d.]+)"(?:[^>]*?BallPossession="(\d)")?(?:[^>]*?BallStatus="(\d)")?')


def teams(game):
    """(home_id, guest_id) from the match information file."""
    root = ET.parse(os.path.join(SRC, "{}_meta.xml".format(game))).getroot()
    general = next(root.iter("General"))
    return general.get("HomeTeamId"), general.get("GuestTeamId")


def parse(game):
    """{(section, team_id, person_id): (N, x, y, z, s, possession, status)} per FrameSet, plus
    {period: datetime of the first ball frame}."""
    out, t0, key, rows = {}, {}, None, []
    with open(os.path.join(SRC, "{}_tracking.xml".format(game)), encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<Frame "):
                m = _FRAME.match(line)
                if m:
                    if not rows and key[1] == "BALL":
                        t0[SECTIONS.index(key[0]) + 1] = datetime.fromisoformat(
                            re.search(r'T="([^"]+)"', line).group(1))
                    rows.append(m.groups())
                continue
            m = _FRAMESET.match(line)
            if m:
                key, rows = m.groups(), []
            elif line.startswith("</FrameSet") and key is not None:
                out[key] = _columns(rows)
                key = None
    return out, t0


def _columns(rows):
    def col(i, kind=float):
        return np.array([kind(r[i]) if r[i] not in (None, "") else np.nan for r in rows])
    return (col(0, int).astype(np.int64), col(1), col(2), col(3), col(4), col(5), col(6))


def raw_events(game, home, period, t0):
    """[(frame, team 0/1, kind, controlled)] for the on-ball events, unshifted, sorted by frame."""
    starts = {p: int(np.flatnonzero(period == p)[0]) for p in (1, 2)}
    sizes = {p: int(np.sum(period == p)) for p in (1, 2)}
    out = []
    for e in ET.parse(os.path.join(SRC, "{}_event.xml".format(game))).getroot().iter("Event"):
        for ch in e:
            kind, team_id = ch.tag, ch.get("Team")
            if kind in ("Play", "ShotAtGoal", "BallClaiming") or kind in RESTARTS:
                controlled = True
            elif kind == "OtherBallAction":
                controlled = ch.get("DefensiveClearance") != "true"
            elif kind == "TacklingGame":
                team_id = ch.get("WinnerTeam")
                controlled = ch.get("PossessionChange") == "true"
            else:
                continue
            if team_id is None:
                continue
            t = datetime.fromisoformat(e.get("EventTime"))
            for p in (1, 2):
                k = int(round((t - t0[p]).total_seconds() * FPS))
                if 0 <= k < sizes[p]:
                    out.append((starts[p] + k, 0 if team_id == home else 1, kind, controlled))
    return sorted(out)


def event_lag(ball, evts):
    """Frames to add to event frames so the passes line up with the ball speeding up."""
    x, y = ball[:, 0].astype(np.float64) * PITCH[0], ball[:, 1].astype(np.float64) * PITCH[1]
    v = np.zeros(len(x))
    v[1:] = np.hypot(np.diff(x), np.diff(y)) * FPS
    rise = np.zeros(len(x))
    rise[2:-2] = v[4:] - v[:-4]
    idx = np.array([f for f, _, k, _ in evts if k == "Play"])
    lags = np.arange(-LAG_SEARCH, LAG_SEARCH + 1)
    score = [np.nanmean(rise[np.clip(idx + lag, 0, len(rise) - 1)]) for lag in lags]
    return int(lags[int(np.argmax(score))])


def controlled_labels(flags, evts):
    """flags with possessions that have no controlled action of their team handed to a neighbour."""
    n = len(flags)
    tol = int(EVENT_TOL_S * FPS)
    near = {}
    for team in (0, 1):
        mark = np.zeros(n + 1, np.int64)
        for f, t, _, controlled in evts:
            if controlled and t == team:
                mark[max(0, f - tol)] += 1
                mark[min(n, f + tol + 1)] -= 1
        near[team] = np.cumsum(mark)[:n] > 0
    out = flags.copy()
    runs = pm.valid_runs(flags)
    for _ in range(10):
        changed = False
        segs = pm.segments(out, runs)
        for i, (a, b, k) in enumerate(segs):
            if k not in (H, A) or near[0 if k == H else 1][a:b].any():
                continue
            before = segs[i - 1][2] if i > 0 and segs[i - 1][1] == a else D
            after = segs[i + 1][2] if i + 1 < len(segs) and segs[i + 1][0] == b else D
            new = before if before in (H, A) else after if after in (H, A) else k
            if new != k:
                out[a:b] = new
                changed = True
        if not changed:
            break
    return out


def build(game):
    """Arrays, flag labels, events and controlled labels for one match."""
    home, guest = teams(game)
    sets, t0 = parse(game)
    people = sorted({(t, p) for (_, t, p) in sets if t != "BALL"},
                    key=lambda tp: (tp[0] != home, tp[1]))
    slot = {tp: i for i, tp in enumerate(people)}
    team = np.array([0 if t == home else 1 for t, _ in people], dtype=np.int8)

    chunks = []
    for period, section in enumerate(SECTIONS, start=1):
        ball = next(v for (s, t, _), v in sets.items() if s == section and t == "BALL")
        first, last = int(ball[0].min()), int(ball[0].max())
        n = last - first + 1
        players = np.full((n, len(people), 3), np.nan, dtype=np.float32)
        for (s, t, p), (N, x, y, _, sp, _, _) in sets.items():
            if s != section or t == "BALL":
                continue
            i = N - first
            ok = (i >= 0) & (i < n)
            players[i[ok], slot[(t, p)], 0] = (x[ok] + PITCH[0] / 2) / PITCH[0]
            players[i[ok], slot[(t, p)], 1] = (y[ok] + PITCH[1] / 2) / PITCH[1]
            players[i[ok], slot[(t, p)], 2] = sp[ok] / KMH
        N, x, y, z, _, poss, status = ball
        i = N - first
        b = np.full((n, 3), np.nan, dtype=np.float32)
        b[i, 0] = (x + PITCH[0] / 2) / PITCH[0]
        b[i, 1] = (y + PITCH[1] / 2) / PITCH[1]
        b[i, 2] = z
        lab = np.full(n, X, dtype=np.int8)
        lab[i] = np.where(status == 0, D, np.where(poss == 1, H, np.where(poss == 2, A, X)))
        chunks.append((players, b, np.full(n, period, dtype=np.int8), lab))
    out = {"players": np.concatenate([c[0] for c in chunks]),
           "ball": np.concatenate([c[1] for c in chunks]),
           "period": np.concatenate([c[2] for c in chunks]),
           "labels": np.concatenate([c[3] for c in chunks]),
           "team": team}
    evts = raw_events(game, home, out["period"], t0)
    lag = event_lag(out["ball"], evts)
    evts = [(f + lag, t, k, c) for f, t, k, c in evts]
    out["controlled"] = controlled_labels(out["labels"], evts)
    out["event_lag"] = np.array(lag)
    out["events"] = np.array([(f, t, int(c)) for f, t, _, c in evts], dtype=np.int64)
    out["event_kind"] = np.array([k for _, _, k, _ in evts])
    return out


def blur_ball(ball, sigma=BLUR_SIGMA):
    """Copy of ball with x, y Gaussian-smoothed over sigma frames, gaps left as they are."""
    if sigma <= 0:
        return ball
    r = int(3 * sigma)
    w = np.exp(-0.5 * (np.arange(-r, r + 1) / float(sigma)) ** 2)
    w /= w.sum()
    out = ball.copy()
    for c in (0, 1):
        v = ball[:, c].astype(np.float64)
        ok = ~np.isnan(v)
        num = np.convolve(np.where(ok, v, 0.0), w, "same")
        den = np.convolve(ok.astype(np.float64), w, "same")
        smooth = np.where(den > 0.5, num / np.maximum(den, 1e-9), np.nan)
        smooth[~ok] = np.nan
        out[:, c] = smooth
    return out


def load(game, rebuild=False):
    """Cached build()."""
    path = os.path.join(CACHE, "{}.npz".format(game))
    if not rebuild and os.path.isfile(path):
        with np.load(path) as c:
            return {k: c[k] for k in c.files}
    out = build(game)
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, **out)
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
    games = sys.argv[1:] or GAMES
    row = "{:<8} {:<11} {:>8.1f} {:>8.1f}s {:>6.0f}% {:>6.0f}% {:>5.0f}%"
    print("{:<8} {:<11} {:>8} {:>9} {:>7} {:>7} {:>6}   event lag".format(
        "match", "labels", "poss/min", "median", "<2s", "<5s", "D"))
    for g in games:
        d = load(g, rebuild=True)
        for name in ("labels", "controlled"):
            p = np.array(profile(d[name])) * [1, 1, 100, 100, 100]
            line = row.format(g if name == "labels" else "", "flags" if name == "labels"
                              else "controlled", *p)
            if name == "labels":
                line += "   {:+.2f} s".format(int(d["event_lag"]) / FPS)
            print(line, flush=True)
    print("{:<8} {:<11} {:>8.1f} {:>8.1f}s {:>6.0f}%".format("public", "competition", 4.2, 7.6, 12))


if __name__ == '__main__':
    main()
