"""The #10 / #12 pattern in Metrica's 3 games, from the hand-tagged events (aggregates only).

Long kick = a PASS / SET PIECE of team T whose next event starts >= MIN_M metres away (tracked
ball) within MAX_FLIGHT_S; deep = taken within DEEP_M of T's own goal line. Pattern = the first
event after the kick (other than T's own) is the other team O's, and T has an event again within
BACK_S of it (O's contact is a short spell). Prints counts and O's first event types only.

    python tools/kick_shorts_metrica.py
"""
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import metrica_data as md  # noqa: E402

FPS = md.FPS
KICKS = ("PASS", "SET PIECE")
MIN_M = 35.0
MAX_FLIGHT_S = 5.0
DEEP_M = 25.0
BACK_S = 2.0


def main():
    tot = Counter()
    kinds = Counter()
    for game in md.GAMES:
        d = md.load(game)
        ev = md.events(game)
        ball, pl, team = d["ball"], d["players"], d["team"]
        n = len(ball)
        for j in range(len(ev) - 1):
            f, _, t, kind, sub = ev[j]
            if kind not in KICKS or not (0 <= f < n):
                continue
            g = ev[j + 1][0]
            if g <= f or g - f > MAX_FLIGHT_S * FPS or g >= n:
                continue
            p, q = ball[f], ball[g]
            if np.isnan(p[:2]).any() or np.isnan(q[:2]).any():
                continue
            if np.hypot((q[0] - p[0]) * 105.0, (q[1] - p[1]) * 68.0) < MIN_M:
                continue
            xs = pl[f, team == t, 0]
            if np.all(np.isnan(xs)):
                continue
            own_low = np.nanmean(xs) < 0.5
            deep = (p[0] if own_low else 1 - p[0]) * 105.0 < DEEP_M
            key = "deep" if deep else "not deep"
            tot[(key, "kicks")] += 1
            # first event after the kick by the other team, skipping T's own
            k = j + 1
            while k < len(ev) and ev[k][2] == t and ev[k][0] - f < MAX_FLIGHT_S * FPS:
                k += 1
            if k >= len(ev) or ev[k][2] == t or ev[k][0] - f > MAX_FLIGHT_S * FPS:
                tot[(key, "O no contact")] += 1
                continue
            fo = ev[k][0]
            back = any(e[2] == t and 0 < e[0] - fo <= BACK_S * FPS for e in ev[k + 1:k + 8])
            if back:
                tot[(key, "O contact, T back < 2 s")] += 1
                kinds[(ev[k][3] + " " + ev[k][4]).strip()] += 1
            else:
                tot[(key, "O contact, O keeps")] += 1
    cols = ["kicks", "O no contact", "O contact, T back < 2 s", "O contact, O keeps"]
    print("Metrica 3 games, long kick: next event >= {:.0f} m within {:.0f} s; deep < {:.0f} m "
          "from own goal line".format(MIN_M, MAX_FLIGHT_S, DEEP_M))
    print("{:<10}".format("") + "".join("{:>26}".format(c) for c in cols))
    for key in ("deep", "not deep"):
        print("{:<10}".format(key) + "".join("{:>26}".format(tot[(key, c)]) for c in cols))
    print("O's contact type in the pattern:", ", ".join(
        "{} {}".format(k, v) for k, v in kinds.most_common()))


if __name__ == "__main__":
    main()
