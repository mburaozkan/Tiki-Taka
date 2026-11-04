"""PFF FC World Cup 2022 (64 matches) as an external set for short possessions.

    python tools/pff_data.py [game ...] [--jobs N]   # build, cache, print the label profiles

Source: PFF FC's free World Cup 2022 release (event data, Sportlogiq broadcast tracking, metadata,
rosters), stored in data/external/pff/ (gitignored with the rest of data/). Not the competition
match.

Tracking at the video's 29.97 fps (`fps` in the metadata), 11 + 11 players identified by shirt
number (VISIBLE or ESTIMATED off screen), the raw ball (x, y, z, VISIBLE / ESTIMATED; missing in
about a quarter of frames). The `ballsSmoothed` track is NOT used: PFF places it on the carrier
from the event data, so it would leak the labels. Resampled to 25 fps (linear between consecutive
source frames) into DFL's arrays:
  players (n, slots, 3)  x, y normalised 0-1, speed m/s (from the positions)
  pvis    (n, slots)     player VISIBLE (not estimated) at the nearest source frame
  ball    (n, 3)         raw ball, normalised x, y and z (m); NaN where missing
  bvis    (n,)           ball VISIBLE at the nearest source frame
  team    (slots,)       0 home, 1 away
  period  (n,)           1 / 2 (extra time 3 / 4 kept)
  labels  (n,)           H / A / D / X from PFF's sequences (below)
  events  (k, 4)         frame_start, frame_end, team 0/1, sequence of every OTB game event
  event_kind (k,)        its possession event type (PA, CR, CL, CH, RE, SH, BC, TC, IT, ...)
  event_t0 (k,)          its start time in video seconds (the event file's startTime)

Labels. PFF's `sequence` is "an uninterrupted possession by one team". Two versions exist: the
tracking files' game events (sequence + `home_ball`, the original) and a `sequence` column added to
the event files in May 2025 (coarser: 295 vs 311 in match 10502, a third of them holding OTB events
of both teams; numbering differs). The tracking version is used; the event file gives only the set
piece type and the possession event kind, matched by start time.
  - from a sequence's first OTB start to its last OTB end: its team (other-team touches inside it,
    e.g. a lost duel, don't break it);
  - between two sequences: the earlier team until the ball goes out (OUT game event), or until the
    later sequence starts if it doesn't; D from the OUT event (or, with no OUT, the earlier
    sequence's end) to the restart when the later sequence starts with a set piece (throw-in,
    free kick, goal kick, corner, penalty, kick-off, drop ball) or an OUT came in between;
  - before the kick-off of a period, after its END event, and in VID (video missing) stretches: X.
"""

import bz2
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from games import DATA_DIR  # noqa: E402
import possession_metrics as pm  # noqa: E402

# Params
SRC = os.path.join(DATA_DIR, "external", "pff", "FIFA World Cup 2022")
CACHE = os.path.join(DATA_DIR, "cache", "pff")
FPS = 25
H, A, D, X = 0, 1, 2, 3
LETTERS = np.array(["H", "A", "D", "X"])
SET_PIECES = ("C", "D", "F", "G", "K", "P", "T")   # O = open play
SPEED_HALF_W = 2


def games():
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(SRC, "Metadata", "*.json"))
                  if os.path.isfile(os.path.join(SRC, "Tracking Data",
                                                 os.path.basename(p)[:-5] + ".jsonl.bz2")))


def _meta(game):
    m = json.load(open(os.path.join(SRC, "Metadata", f"{game}.json"), encoding="utf-8"))
    m = m[0] if isinstance(m, list) else m
    pitch = (m.get("stadium") or {}).get("pitches") or [{}]
    length = float(pitch[0].get("length") or 105.0)
    width = float(pitch[0].get("width") or 68.0)
    return m, (length, width), float(m.get("fps") or 29.97)


def parse_tracking(game, pitch):
    """Source-rate arrays: time (video s), period, players, pvis, ball, bvis, slots, home_ball
    per sequence."""
    L, W = pitch
    rows, seen = [], set()
    for line in bz2.open(os.path.join(SRC, "Tracking Data", f"{game}.jsonl.bz2"), "rt",
                         encoding="utf-8"):
        r = json.loads(line)
        if r["frameNum"] in seen or r.get("period") is None:
            continue
        seen.add(r["frameNum"])
        rows.append(r)
    rows.sort(key=lambda r: r["frameNum"])
    shirts = sorted({(side, str(p["jerseyNum"])) for r in rows
                     for side, key in ((0, "homePlayers"), (1, "awayPlayers"))
                     for p in (r.get(key) or [])},
                    key=lambda s: (s[0], int(s[1]) if s[1].isdigit() else 999, s[1]))
    slot = {s: i for i, s in enumerate(shirts)}
    m, s = len(rows), len(shirts)
    t = np.array([r["videoTimeMs"] / 1000.0 for r in rows])
    period = np.array([r["period"] for r in rows], dtype=np.int8)
    pl = np.full((m, s, 2), np.nan, dtype=np.float32)
    pvis = np.zeros((m, s), dtype=bool)
    ball = np.full((m, 3), np.nan, dtype=np.float32)
    bvis = np.zeros(m, dtype=bool)
    home_ball, otb_seq = {}, {}
    for j, r in enumerate(rows):
        for side, key in ((0, "homePlayers"), (1, "awayPlayers")):
            for p in r.get(key) or []:
                i = slot[(side, str(p["jerseyNum"]))]
                if not np.isnan(pl[j, i, 0]):
                    continue          # a duplicated shirt number in this frame: keep the first
                pl[j, i] = ((p["x"] + L / 2) / L, (p["y"] + W / 2) / W)
                pvis[j, i] = p.get("visibility") == "VISIBLE"
        b = (r.get("balls") or [None])[0]
        if b and b.get("x") is not None:
            ball[j] = ((b["x"] + L / 2) / L, (b["y"] + W / 2) / W,
                       b["z"] if b.get("z") is not None else np.nan)
            bvis[j] = b.get("visibility") == "VISIBLE"
        g = r.get("game_event") or {}
        if g.get("sequence") is not None and g.get("home_ball") is not None:
            home_ball.setdefault(int(g["sequence"]), bool(g["home_ball"]))
            if g.get("start_time") is not None:
                otb_seq.setdefault(round(g["start_time"], 3), int(g["sequence"]))
    team = np.array([sd for sd, _ in shirts], dtype=np.int8)
    return dict(t=t, period=period, players=pl, pvis=pvis, ball=ball, bvis=bvis, team=team,
                home_ball=home_ball, otb_seq=otb_seq)


def parse_events(game):
    """OTB / kick-off events with sequence, and the OUT / END / VID markers (video seconds)."""
    ev = json.load(open(os.path.join(SRC, "Event Data", f"{game}.json"), encoding="utf-8"))
    otb, outs, ends, vids, kickoffs = [], [], [], [], []
    for r in ev:
        g = r.get("gameEvents") or {}
        kind = g.get("gameEventType")
        t0 = r.get("startTime")
        if t0 is None:
            continue
        t1 = r.get("endTime") if r.get("endTime") is not None else t0
        if kind in ("OTB", "FIRSTKICKOFF", "SECONDKICKOFF") and r.get("sequence") is not None:
            pe = r.get("possessionEvents") or {}
            otb.append((t0, t1, int(not g.get("homeTeam")), int(r["sequence"]),
                        pe.get("possessionEventType") or "", g.get("setpieceType") or "O",
                        g.get("period")))
            if kind != "OTB":
                kickoffs.append(t0)
        elif kind == "OUT":
            outs.append(t0)
        elif kind == "END":
            ends.append(t0)
        elif kind == "VID":
            vids.append((t0, t1))
    otb.sort()
    return otb, np.array(sorted(outs)), sorted(ends), vids, sorted(kickoffs)


def label_times(t, otb, outs, ends, vids, kickoffs, home_ball):
    """H / A / D / X at the video times t (see the module docstring)."""
    lab = np.full(len(t), X, dtype=np.int8)
    seqs = {}
    for t0, t1, side, seq, _, sp, per in otb:
        if seq not in seqs:
            seqs[seq] = [t0, t1, side, sp, per]
        else:
            seqs[seq][1] = max(seqs[seq][1], t1)
    order = sorted(seqs.items(), key=lambda kv: kv[1][0])

    def team_of(seq, fallback):
        hb = home_ball.get(seq)
        return fallback if hb is None else (H if hb else A)

    def span(a, b, v):
        lab[(t >= a) & (t < b)] = v

    for k, (seq, (t0, t1, side, sp, per)) in enumerate(order):
        me = team_of(seq, side)
        span(t0, t1 + 1e-6, me)
        if k + 1 >= len(order):
            continue
        nseq, (n0, _, nside, nsp, nper) = order[k + 1]
        if nper != per or n0 <= t1:
            continue
        out_in = outs[(outs > t1) & (outs < n0)]
        if len(out_in):
            span(t1, out_in[0], me)
            span(out_in[0], n0, D)
        elif nsp in SET_PIECES:
            span(t1, n0, D)
        else:
            span(t1, n0, me)
    # outside play: before each kick-off back to the previous END, after the last END
    bounds = sorted(kickoffs)
    for i, ko in enumerate(bounds):
        prev_end = max([e for e in ends if e < ko], default=-np.inf)
        lab[(t > prev_end) & (t < ko)] = X
    if ends:
        lab[t >= max(ends)] = X
    for a, b in vids:
        lab[(t >= a) & (t <= b)] = X
    return lab


def _resample(t_src, v, t_dst, max_step):
    i = np.clip(np.searchsorted(t_src, t_dst, side="right") - 1, 0, len(t_src) - 2)
    dt = t_src[i + 1] - t_src[i]
    w = ((t_dst - t_src[i]) / dt).reshape((-1,) + (1,) * (v.ndim - 1))
    out = v[i] + (v[i + 1] - v[i]) * w
    bad = (dt > max_step) | (w.reshape(-1) < -1e-9) | (w.reshape(-1) > 1 + 1e-9)
    out[bad] = np.nan
    return out


def _speed(xy, pitch):
    k = SPEED_HALF_W
    d = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
    dx = (xy[2 * k:, ..., 0] - xy[:-2 * k, ..., 0]) * pitch[0]
    dy = (xy[2 * k:, ..., 1] - xy[:-2 * k, ..., 1]) * pitch[1]
    d[k:-k] = np.hypot(dx, dy) * FPS / (2 * k)
    return d


def build(game):
    meta, pitch, src_fps = _meta(game)
    src = parse_tracking(game, pitch)
    otb, outs, ends, vids, kickoffs = parse_events(game)
    # the tracking files' sequence numbers (see the module docstring); unmatched events dropped
    n_ev = len(otb)
    otb = [(a, b, side, src["otb_seq"][round(a, 3)], k, sp, per)
           for a, b, side, _, k, sp, per in otb if round(a, 3) in src["otb_seq"]]
    chunks = []
    for per in sorted(set(src["period"].tolist())):
        sel = src["period"] == per
        t = src["t"][sel]
        t_out = np.arange(t[0], t[-1] + 1e-9, 1.0 / FPS)
        near = np.clip(np.searchsorted(t, t_out), 0, len(t) - 1)
        step = 1.5 / src_fps
        chunks.append(dict(
            t=t_out, period=np.full(len(t_out), per, dtype=np.int8),
            players=_resample(t, src["players"][sel], t_out, step),
            pvis=src["pvis"][sel][near],
            ball=_resample(t, src["ball"][sel], t_out, step),
            bvis=src["bvis"][sel][near]))
    out = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    t = out.pop("t")
    speed = _speed(out["players"], pitch)
    out["players"] = np.concatenate([out["players"], speed[..., None]], axis=-1)
    out["team"] = src["team"]
    out["labels"] = label_times(t, otb, outs, ends, vids, kickoffs, src["home_ball"])
    idx = lambda tt: int(np.clip(np.searchsorted(t, tt), 0, len(t) - 1))  # noqa: E731
    out["events"] = np.array([(idx(a), idx(b), side, seq) for a, b, side, seq, *_ in otb],
                             dtype=np.int64)
    out["event_kind"] = np.array([e[4] for e in otb], dtype="U4")
    out["event_t0"] = np.array([e[0] for e in otb])         # video seconds (event file key)
    out["home_left_p1"] = np.array(bool(meta.get("homeTeamStartLeft")))
    out["time"] = t
    out["events_matched"] = np.array([len(otb), n_ev])
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


def _row(game):
    d = load(game, rebuild=True)
    lab = d["labels"]
    g = lab != X
    p = np.array(profile(lab)) * [1, 1, 100, 100, 100]
    return (game, *p, 100 * g.mean(), 100 * np.mean(~np.isnan(d["ball"][g, 0])),
            100 * np.mean(d["bvis"][g]), 100 * np.mean(d["pvis"][g].sum(1) / 22))


def main():
    args = sys.argv[1:]
    jobs = 1
    if "--jobs" in args:
        i = args.index("--jobs")
        jobs = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    gs = args or games()
    print("{:<6} {:>8} {:>9} {:>6} {:>6} {:>5} {:>7} {:>6} {:>6} {:>6}".format(
        "match", "poss/min", "median", "<2s", "<5s", "D", "graded", "ball", "bVis", "plVis"))
    row = "{:<6} {:>8.1f} {:>8.1f}s {:>5.0f}% {:>5.0f}% {:>4.0f}% {:>6.0f}% {:>5.0f}% {:>5.0f}% {:>5.0f}%"
    rows = []
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as ex:
            for r in ex.map(_row, gs):
                print(row.format(*r), flush=True)
                rows.append(r)
    else:
        for g in gs:
            r = _row(g)
            print(row.format(*r), flush=True)
            rows.append(r)
    if len(rows) > 1:
        a = np.array([r[1:] for r in rows])
        print(row.format("median", *np.median(a, axis=0)))
    print("{:<6} {:>8.1f} {:>8.1f}s {:>5.0f}%".format("public", 4.2, 7.6, 12))


if __name__ == "__main__":
    main()
