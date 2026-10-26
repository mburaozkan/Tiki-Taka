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
# Dead-ball motion (animate_dead_ball), calibrated to our ball in stoppages (JOURNAL.md).
LIE_SHARE = (0.1, 0.5)      # share of a dead stretch the ball lies where play stopped
FLIGHT_MS = (6.0, 15.0)     # speed of the passes / throws while the restart is set up
REST_S = (0.3, 1.5)         # time the ball stays with a player between passes
OFFSET_M = (0.5, 3.0)       # how far from that player it rests
N_HELPERS = 4               # players nearest the restart spot who pass it around
BLEND_S = 0.5               # glide into the restart position over the last half second
MIN_DEAD_S = 2.0

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


def animate_dead_ball(d, seed=0, lie=LIE_SHARE, flight_ms=FLIGHT_MS, rest_s=REST_S,
                      offset_m=OFFSET_M):
    """Copy of the ball with every dead stretch given motion like ours.

    DFL holds the ball still while it is dead (speed 0.0, still 98% of dead frames); ours is
    picked up, passed around and carried to the restart (speed 2.4 m/s, still 50%, faster than
    8 m/s 13-19%, within 1.5 m of a player about 30%, nearest player about 3 m). For each dead
    stretch [s, e):
      lying     the ball stays where play stopped, for a share drawn from `lie` of the stretch
      passed    then it moves between the N_HELPERS players nearest the restart spot: a flight at
                a speed drawn from `flight_ms` to a player, a rest of `rest_s` seconds next to him
                (offset drawn from `offset_m`), the next player, ...
      restart   the last rest is with the taker (the restarting team's player nearest the
                restart spot), blending into the restart position over the last BLEND_S
    Stretches shorter than MIN_DEAD_S are interpolated straight from stop to restart.
    """
    rng = np.random.default_rng(seed)
    L, W = PITCH
    lab, players, team = d["labels"], d["players"], d["team"]
    x = d["ball"][:, 0].astype(np.float64) * L
    y = d["ball"][:, 1].astype(np.float64) * W
    px, py = players[:, :, 0].astype(np.float64) * L, players[:, :, 1].astype(np.float64) * W
    dead = lab == D
    edges = np.diff(np.concatenate(([0], dead.astype(np.int8), [0])))
    blend = int(BLEND_S * FPS)
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if s == 0 or e >= len(lab) or lab[e] not in (H, A):
            continue
        x0, y0, x1, y1 = x[s], y[s], x[e], y[e]
        n = e - s
        if n < MIN_DEAD_S * FPS or np.isnan(x0 + y0 + x1 + y1):
            frac = np.arange(1, n + 1) / float(n + 1)
            x[s:e], y[s:e] = x0 + frac * (x1 - x0), y0 + frac * (y1 - y0)
            continue
        side = 0 if lab[e] == H else 1
        cols = np.flatnonzero(team == side)
        dist_end = np.hypot(px[e, cols] - x1, py[e, cols] - y1)
        if np.all(np.isnan(dist_end)):
            continue
        taker = cols[int(np.nanargmin(dist_end))]
        near_spot = np.hypot(px[e] - x1, py[e] - y1)
        helpers = [int(i) for i in np.argsort(np.where(np.isnan(near_spot), np.inf, near_spot))
                   [:N_HELPERS] if np.isfinite(near_spot[i])]

        def track(j):
            tx, ty = px[s:e, j].copy(), py[s:e, j].copy()
            ok = ~np.isnan(tx)
            idx = np.arange(n)
            if not ok.any():
                return np.full(n, x1), np.full(n, y1)
            return np.interp(idx, idx[ok], tx[ok]), np.interp(idx, idx[ok], ty[ok])

        bx, by = np.full(n, x0), np.full(n, y0)
        t = int(rng.uniform(*lie) * n)
        cur_x, cur_y = x0, y0
        while t < n:
            last = t + int(rest_s[1] * FPS) >= n - blend
            j = taker if last else int(rng.choice(helpers))
            tx, ty = track(j)
            ang = rng.uniform(0, 2 * np.pi)
            off = rng.uniform(*offset_m)
            gx, gy = tx + off * np.cos(ang), ty + off * np.sin(ang)
            fly = max(1, int(np.hypot(gx[t] - cur_x, gy[t] - cur_y) / rng.uniform(*flight_ms) * FPS))
            a, b = t, min(n, t + fly)
            frac = (np.arange(a, b) - a + 1) / float(b - a + 1)
            bx[a:b] = cur_x + frac * (gx[a:b] - cur_x)
            by[a:b] = cur_y + frac * (gy[a:b] - cur_y)
            stay = n if last else min(n, b + int(rng.uniform(*rest_s) * FPS))
            bx[b:stay], by[b:stay] = gx[b:stay], gy[b:stay]
            t = stay
            cur_x, cur_y = bx[t - 1], by[t - 1]
        w = np.clip((np.arange(n) - (n - blend)) / float(max(blend, 1)), 0, 1)
        x[s:e] = (1 - w) * bx + w * x1
        y[s:e] = (1 - w) * by + w * y1
    out = d["ball"].copy()
    out[:, 0], out[:, 1] = x / L, y / W
    return out


def prepare_ball(d, seed, animate=True, blur=True, gaps=True):
    """DFL's ball made to look like ours: dead ball animated, blurred, our gap pattern cut in."""
    ball = animate_dead_ball(d, seed) if animate else d["ball"]
    if blur:
        ball = blur_ball(ball)
    if gaps:
        from gap_model import degrade
        ball = degrade(ball, d["players"], seed=seed)
    return ball


# Time warp (set_warp): every match is resampled onto a clock stretched by WARP, so play runs
# WARP times slower (our live play is slower than DFL's; JOURNAL.md 2026-10-25). 1.0 = off.
WARP = 1.0


def set_warp(k):
    """Turn the time warp on for every later load() / touch_labels.load(); returns the suffix
    that warped caches and model files carry ('' when off)."""
    global WARP
    WARP = float(k)
    return suffix()


def suffix():
    return "" if WARP == 1.0 else "_w{:g}".format(WARP)


def _warp_index(period, k):
    """For a stretched clock: per new frame the (fractional) old frame, the new period array, and
    the new start frame of each period."""
    t_all, per_all, starts = [], [], {}
    pos = 0
    for p in (1, 2):
        idx = np.flatnonzero(period == p)
        if not len(idx):
            continue
        n = len(idx)
        m = int((n - 1) * k) + 1
        t_all.append(idx[0] + np.arange(m) / k)
        per_all.append(np.full(m, p, dtype=np.int8))
        starts[p] = (idx[0], pos)
        pos += m
    return np.concatenate(t_all), np.concatenate(per_all), starts


def warp_frames(frames, period, k):
    """Old frame numbers -> frame numbers on the stretched clock."""
    frames = np.asarray(frames)
    _, _, starts = _warp_index(period, k)
    out = frames.astype(np.float64).copy()
    for p, (old0, new0) in starts.items():
        n = np.sum(period == p)
        sel = (frames >= old0) & (frames < old0 + n)
        out[sel] = new0 + np.round((frames[sel] - old0) * k)
    return out.astype(np.int64)


def warp(d, k):
    """d resampled onto a clock stretched by k: positions interpolated linearly (NaN where either
    neighbour is missing), speeds divided by k, labels and events moved to the nearest frame."""
    t, period, _ = _warp_index(d["period"], k)
    i0 = np.floor(t).astype(np.int64)
    i1 = np.minimum(i0 + 1, len(d["period"]) - 1)
    w = (t - i0).astype(np.float32)

    def lerp(a):
        wb = w.reshape((-1,) + (1,) * (a.ndim - 1))
        return ((1 - wb) * a[i0] + wb * a[i1]).astype(a.dtype)

    near = np.round(t).astype(np.int64)
    near = np.minimum(near, len(d["period"]) - 1)
    out = dict(d)
    out["players"] = lerp(d["players"])
    out["players"][:, :, 2] /= k
    out["ball"] = lerp(d["ball"])
    out["period"] = period
    for key in ("labels", "controlled"):
        out[key] = d[key][near]
    ev = d["events"].copy()
    ev[:, 0] = warp_frames(ev[:, 0], d["period"], k)
    out["events"] = ev
    return out


def load(game, rebuild=False):
    """Cached build() (time-warped when set_warp is on)."""
    path = os.path.join(CACHE, "{}.npz".format(game))
    if not rebuild and os.path.isfile(path):
        with np.load(path) as c:
            out = {k: c[k] for k in c.files}
    else:
        out = build(game)
        os.makedirs(CACHE, exist_ok=True)
        np.savez(path, **out)
    if WARP == 1.0:
        return out
    key = (game, WARP)
    if key not in _WARPED:
        _WARPED[key] = warp(out, WARP)
    return _WARPED[key]


_WARPED = {}


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
