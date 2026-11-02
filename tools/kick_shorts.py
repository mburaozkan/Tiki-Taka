"""How often does a long kick end in a short possession for the other team? (DFL, aggregates only)

A long kick = an on-ball event (Play / GoalKick / FreeKick) of team T whose next on-ball event
is at least MIN_M metres away and within MAX_FLIGHT_S. "Deep" = taken within DEEP_M of T's own
goal line (keeper kicks, goal kicks, deep clearances). Outcome from the controlled labels (the
benchmark's truth): the first possession of the other team O that starts within WINDOW_S after
the kick, its length, and what follows it.

Prints counts and shares only.

    python tools/kick_shorts.py              (controlled labels, the benchmark truth)
    KS_LABELS=labels python tools/kick_shorts.py   (raw provider flags)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dfl_data  # noqa: E402

FPS = dfl_data.FPS
H, A, D, X = 0, 1, 2, 3
KICKS = ("Play", "GoalKick", "FreeKick")
MIN_M = 35.0
MAX_FLIGHT_S = 5.0
DEEP_M = 25.0
WINDOW_S = 6.0
SHORT_S = 2.0


def runs(lab):
    """[(start, end, label)] of constant label (X kept as its own run)."""
    cut = np.flatnonzero(np.diff(lab)) + 1
    s = np.r_[0, cut]
    e = np.r_[cut, len(lab)]
    return [(int(a), int(b), int(lab[a])) for a, b in zip(s, e)]


def main():
    tot = {}
    short_all = 0
    short_after = {"deep": 0, "any": 0}

    def add(key, outcome):
        tot.setdefault(key, {}).setdefault(outcome, 0)
        tot[key][outcome] += 1

    for game in dfl_data.GAMES:
        d = dfl_data.load(game)
        lab = d[os.environ.get("KS_LABELS", "controlled")].astype(np.int64)
        ball = d["ball"]
        pl = d["players"]
        team = d["team"]
        ev = d["events"]
        kinds = d["event_kind"]
        segs = runs(lab)
        starts = np.array([a for a, _, _ in segs])
        n = len(lab)
        short_starts = {}  # start frame -> (team, len)
        for i, (a, b, k) in enumerate(segs):
            if k in (H, A) and (b - a) < SHORT_S * FPS:
                short_all += 1
                short_starts[a] = k
        hit = {"deep": set(), "any": set()}
        for j in range(len(ev) - 1):
            f, t, _ = ev[j]
            if kinds[j] not in KICKS or not (0 <= f < n):
                continue
            g = ev[j + 1][0]
            if g <= f or g - f > MAX_FLIGHT_S * FPS or g >= n:
                continue
            p, q = ball[f], ball[g]
            if np.isnan(p[:2]).any() or np.isnan(q[:2]).any():
                continue
            dist = np.hypot((q[0] - p[0]) * 105.0, (q[1] - p[1]) * 68.0)
            if dist < MIN_M:
                continue
            xs = pl[f, team == t, 0]
            if np.all(np.isnan(xs)):
                continue
            own_low = np.nanmean(xs) < 0.5   # T defends x = 0 in this half (centroid)
            to_goal = p[0] * 105.0 if own_low else (1 - p[0]) * 105.0
            deep = to_goal < DEEP_M
            o = A if t == 0 else H
            tk = H if t == 0 else A
            # first possession of O starting in [f, f + WINDOW_S]
            i0 = int(np.searchsorted(starts, f))
            outcome = "O none (T keeps / dead)"
            for i in range(i0, len(segs)):
                a, b, k = segs[i]
                if a > f + WINDOW_S * FPS:
                    break
                if k != o:
                    continue
                L = (b - a) / FPS
                nxt = segs[i + 1][2] if i + 1 < len(segs) else D
                if L < SHORT_S:
                    outcome = "O short, then T" if nxt == tk else "O short, then D/other"
                    for key in (["deep", "any"] if deep else ["any"]):
                        hit[key].add(a)
                else:
                    outcome = "O wins (>= 2 s)"
                break
            add("deep" if deep else "not deep", outcome)
            add("all long", outcome)
        for key in hit:
            short_after[key] += len(hit[key])

    print("long kicks: next event >= {:.0f} m away within {:.0f} s; deep = within {:.0f} m of own "
          "goal line; outcome window {:.0f} s; short < {:.0f} s (7 DFL matches, labels: {})"
          .format(MIN_M, MAX_FLIGHT_S, DEEP_M, WINDOW_S, SHORT_S, os.environ.get("KS_LABELS", "controlled")))
    order = ["O none (T keeps / dead)", "O short, then T", "O short, then D/other", "O wins (>= 2 s)"]
    print("{:<10} {:>6}   ".format("kicks", "n") + "  ".join("{:>24}".format(o) for o in order))
    for key in ("deep", "not deep", "all long"):
        c = tot.get(key, {})
        n = sum(c.values())
        print("{:<10} {:>6}   ".format(key, n) + "  ".join(
            "{:>16} ({:4.1f}%)".format(c.get(o, 0), 100 * c.get(o, 0) / max(n, 1)) for o in order))
    print()
    print("short possessions (< {:.0f} s) in all: {}".format(SHORT_S, short_all))
    print("  of them right after a long kick of the other team: {} ({:.1f}%)".format(
        short_after["any"], 100 * short_after["any"] / max(short_all, 1)))
    print("  of them right after a DEEP long kick:              {} ({:.1f}%)".format(
        short_after["deep"], 100 * short_after["deep"] / max(short_all, 1)))


if __name__ == "__main__":
    main()
