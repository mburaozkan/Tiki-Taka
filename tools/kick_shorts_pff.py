"""The #10 / #12 pattern in PFF FC's World Cup 2022 event data (64 matches, aggregates only).

PFF game events of type OTB ("on the ball") mark each time a player has the ball, from first touch
to release, with team, start / end time, the possession event (PA pass, CR cross, CL clearance,
CH challenge, RE rebound, SH shot, BC carry, TC touch) and the ball position. `sequence` is PFF's
own team possession ("an uninterrupted possession by one team").

Long kick = an OTB event ending in a pass, cross or clearance whose next OTB event is >= MIN_M
metres away within MAX_FLIGHT_S; deep = kicked within DEEP_M of the kicker's own goal line.
Pattern = the next OTB is the other team O's, O's run of OTB events lasts < SHORT_S, and the
kicker's team has an OTB again within BACK_S. For each case: does PFF open a new sequence for
O's contact (i.e. count it as a team possession), and what the contact is.

    python tools/kick_shorts_pff.py
"""
import glob
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import DATA_DIR  # noqa: E402

SRC = os.path.join(DATA_DIR, "external", "pff", "FIFA World Cup 2022")
MIN_M = 35.0
MAX_FLIGHT_S = 5.0
DEEP_M = 25.0
SHORT_S = 2.0
BACK_S = 2.0
KICKS = ("PA", "CR", "CL")


def otb_events(game):
    """[(period, t0, t1, team_is_home, sequence, ball_xy, pe_type, body, setpiece)] in time order."""
    ev = json.load(open(os.path.join(SRC, "Event Data", f"{game}.json"), encoding="utf-8"))
    out = []
    for r in ev:
        g = r.get("gameEvents") or {}
        if g.get("gameEventType") not in ("OTB", "FIRSTKICKOFF", "SECONDKICKOFF") \
                or g.get("homeTeam") is None or not r.get("ball"):
            continue
        b = r["ball"][0]
        if b.get("x") is None:
            continue
        pe = r.get("possessionEvents") or {}
        out.append((g["period"], r["startTime"], r["endTime"] or r["startTime"], bool(g["homeTeam"]),
                    r.get("sequence"), (b["x"], b["y"]), pe.get("possessionEventType"),
                    pe.get("bodyType"), g.get("setpieceType")))
    out.sort(key=lambda e: (e[0], e[1]))
    return out


def main():
    meta_files = sorted(glob.glob(os.path.join(SRC, "Metadata", "*.json")))
    tot, contact, spells = Counter(), Counter(), Counter()
    n_games = 0
    for mf in meta_files:
        game = os.path.splitext(os.path.basename(mf))[0]
        if not os.path.isfile(os.path.join(SRC, "Event Data", f"{game}.json")):
            continue
        meta = json.load(open(mf, encoding="utf-8"))
        meta = meta[0] if isinstance(meta, list) else meta
        length = float((meta.get("stadium") or {}).get("pitches", [{}])[0].get("length") or 105.0) \
            if isinstance((meta.get("stadium") or {}).get("pitches"), list) else 105.0
        home_left = bool(meta.get("homeTeamStartLeft"))
        e = otb_events(game)
        n_games += 1
        # base rate: short team runs of OTB events sandwiched by the other team; PFF sequences
        runs = []
        for x in e:
            if runs and runs[-1][0] == x[0] and runs[-1][1] == x[3]:
                runs[-1][3] = x[2]
                runs[-1][4].add(x[4])
            else:
                runs.append([x[0], x[3], x[1], x[2], {x[4]}])
        for a, b, c in zip(runs, runs[1:], runs[2:]):
            if a[0] == b[0] == c[0]:
                spells["team runs"] += 1
                if a[1] == c[1] and b[3] - b[2] < SHORT_S:
                    spells["short sandwiched"] += 1
                    spells["  own PFF sequence"] += int(not (b[4] & a[4]) and not (b[4] & c[4]))
        for i in range(len(e) - 1):
            k, q = e[i], e[i + 1]
            if k[6] not in KICKS or q[0] != k[0] or not (0 <= q[1] - k[2] <= MAX_FLIGHT_S):
                continue
            if np.hypot(q[5][0] - k[5][0], q[5][1] - k[5][1]) < MIN_M:
                continue
            # home attacks left to right (own goal at -L/2) in period 1 if home_left
            home_own_left = home_left if k[0] in (1, 3) else not home_left
            own_x = -length / 2 if (k[3] == home_own_left) else length / 2
            key = "deep" if abs(k[5][0] - own_x) < DEEP_M else "not deep"
            tot[(key, "kicks")] += 1
            if q[3] == k[3]:
                tot[(key, "T keeps")] += 1
                continue
            j = i + 1
            while j + 1 < len(e) and e[j + 1][3] == q[3] and e[j + 1][0] == q[0]:
                j += 1
            o_end = e[j][2]
            nxt = e[j + 1] if j + 1 < len(e) else None
            back = nxt is not None and nxt[0] == q[0] and nxt[3] == k[3] and nxt[1] - o_end <= BACK_S
            if o_end - q[1] < SHORT_S and back:
                tot[(key, "O short, T back")] += 1
                own_seq = q[4] != k[4] and q[4] != nxt[4]
                tot[(key, "  own PFF sequence")] += int(own_seq)
                if key == "deep":
                    contact[f"type={q[6]}"] += 1
                    contact[f"body={q[7]}"] += 1
                    contact[f"O events={j - i}"] += 1
            else:
                tot[(key, "O keeps")] += 1
    print(f"{n_games} matches")
    for key in ("deep", "not deep"):
        parts = [f"{k.strip()}={tot[(key, k)]}" for k in
                 ("kicks", "T keeps", "O short, T back", "  own PFF sequence", "O keeps")]
        print(f"{key:9s} " + "  ".join(parts))
    print(dict(spells))
    print("deep pattern, O's contact:", dict(sorted(contact.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
