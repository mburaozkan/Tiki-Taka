"""The #10 / #12 pattern in SkillCorner's 10 open matches, from player_possession events
(aggregates only).

Long kick = a player possession of team T ending in a pass or clearance whose next player
possession starts >= MIN_M metres away within MAX_FLIGHT_S; deep = kicked within DEEP_M of T's own
goal line. Pattern = the next possession is the other team O's, O's spell (consecutive O
possessions) lasts < SHORT_S, and T has the ball again within BACK_S of O's spell ending.
Also prints the base rate of short team spells and how O's contact looks (header, start / end
type).

    python tools/kick_shorts_skillcorner.py
"""
import glob
import os
from collections import Counter

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "external",
                    "skillcorner", "repo", "data", "matches")
MIN_M = 35.0
MAX_FLIGHT_S = 5.0
DEEP_M = 25.0
SHORT_S = 2.0
BACK_S = 2.0
HALF_L = 52.5


def possessions(path):
    d = pd.read_csv(path, low_memory=False)
    p = d[d.event_type == "player_possession"].copy()
    p["t0"] = p.frame_start / 10.0
    p["t1"] = p.frame_end / 10.0
    return p.sort_values(["period", "frame_start"]).reset_index(drop=True)


def spells(p):
    """Team spells: runs of consecutive possessions by one team, broken by a period change."""
    out = []
    for _, r in p.iterrows():
        if out and out[-1]["team"] == r.team_id and out[-1]["period"] == r.period:
            out[-1]["t1"] = r.t1
            out[-1]["n"] += 1
        else:
            out.append(dict(team=r.team_id, period=r.period, t0=r.t0, t1=r.t1, n=1))
    return out


def main():
    tot = Counter()
    contact = Counter()
    short_spells = Counter()
    files = sorted(glob.glob(os.path.join(ROOT, "*", "*_dynamic_events.csv")))
    for f in files:
        p = possessions(f)
        sp = spells(p)
        # base rate: short team spells between two spells of the other team
        for a, b, c in zip(sp, sp[1:], sp[2:]):
            if a["period"] == b["period"] == c["period"]:
                tot["spells"] += 1
                if b["t1"] - b["t0"] < SHORT_S and a["team"] == c["team"]:
                    short_spells["sandwiched < 2 s"] += 1
                    short_spells[f"  with {min(b['n'], 3)}{'+' if b['n'] >= 3 else ''} poss"] += 1
        for i in range(len(p) - 1):
            r, q = p.iloc[i], p.iloc[i + 1]
            if r.end_type not in ("pass", "clearance") or q.period != r.period:
                continue
            if q.t0 - r.t1 > MAX_FLIGHT_S or q.t0 < r.t1:
                continue
            if np.hypot(q.x_start - r.x_end, q.y_start - r.y_end) < MIN_M:
                continue
            own_x = -HALF_L if r.attacking_side == "left_to_right" else HALF_L
            deep = abs(r.x_end - own_x) < DEEP_M
            key = "deep" if deep else "not deep"
            tot[(key, "kicks")] += 1
            if q.team_id == r.team_id:
                tot[(key, "T keeps")] += 1
                continue
            j = i + 1
            while j + 1 < len(p) and p.iloc[j + 1].team_id == q.team_id \
                    and p.iloc[j + 1].period == q.period:
                j += 1
            o_end = p.iloc[j].t1
            nxt = p.iloc[j + 1] if j + 1 < len(p) else None
            back = (nxt is not None and nxt.period == q.period and nxt.team_id == r.team_id
                    and nxt.t0 - o_end <= BACK_S)
            if o_end - q.t0 < SHORT_S and back:
                tot[(key, "O short, T back")] += 1
                if deep:
                    contact[f"header={bool(q.is_header)}"] += 1
                    contact[f"start={q.start_type}"] += 1
                    contact[f"end={q.end_type}"] += 1
                    contact[f"O poss={j - i}"] += 1
            else:
                tot[(key, "O keeps")] += 1
    print(f"{len(files)} matches")
    for key in ("deep", "not deep"):
        n = tot[(key, "kicks")]
        parts = [f"{k}={tot[(key, k)]}" for k in ("T keeps", "O short, T back", "O keeps")]
        print(f"{key:9s} kicks={n}  " + "  ".join(parts))
    print("team spells", tot["spells"], dict(short_spells))
    print("deep pattern, O's contact:", dict(sorted(contact.items())))


if __name__ == "__main__":
    main()
