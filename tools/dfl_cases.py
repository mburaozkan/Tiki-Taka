"""Case studies of turnovers the change net misses, on DFL open data only (CC BY 4.0).
Never run on our match: it loads only dfl_data.GAMES.

    python tools/dfl_cases.py kinds [--tag c1dlbd]          # aggregate: events around found / missed
    python tools/dfl_cases.py show [--tag c1dlbd] [--n 30] [--found 10] [--seed 0]

A true turnover is "missed" when no predicted turnover lies within 2 s. kinds: the DFL events within
+-1 s of the turnover (kind, by the new / old team, controlled), share and recall per kind.
show: per case a timeline from -2 s to +2 s every 0.2 s: true / predicted state, ball (x toward
the side the new team attacks, y, height, speed, seen by the net), the 3 nearest players of the new
team (N) and the old team (O): distance, speed, approach speed to the ball, player number (stable
within the case), the net's P(turnover to the new team), P(new team); and the events nearby.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
import change_net as C  # noqa: E402
import dfl_data as dd  # noqa: E402
import miss_profile as M  # noqa: E402
from pipeline.decode import H, A, D  # noqa: E402

FPS = 25
CODE = {H: "H", A: "A", D: "D"}


def cases(tag):
    import cn_dfl
    from train_learned_dfl import test_seed
    out = []
    for g in dd.GAMES:
        with np.load(cn_dfl.path(g)) as z:
            lab, blk = z["lab"], [tuple(b) for b in z["blk"]]
        pr = cn_dfl.heldout(tag, g)
        dec = C.viterbi(pr, blk, 0.5, -3.0)
        runs = pm.valid_runs(lab)
        filled = pm.fill_abstentions(dec, runs)
        tc = [f for f, k in M.changes(lab, runs) if k == 0]
        pc = np.array([f for f, k in M.changes(filled, runs) if k == 0])
        d = dd.load(g)
        ball_net = dd.prepare_ball(d, test_seed(g))
        for t in tc:
            off = np.min(np.abs(pc - t)) if len(pc) else 9999
            out.append({"g": g, "t": t, "missed": off > 2 * FPS, "lab": lab, "dec": filled, "pr": pr,
                        "d": d, "seen": np.isfinite(ball_net[:, 0])})
    return out


def events_near(d, t, new, w=FPS):
    ev, kinds = d["events"], d["event_kind"]
    sel = np.flatnonzero(np.abs(ev[:, 0] - t) <= w)
    team_code = 0 if new == H else 1
    return [(int(ev[i, 0] - t), "N" if ev[i, 1] == team_code else "O", str(kinds[i]), int(ev[i, 2]))
            for i in sel]


def kinds_cmd(args):
    cs = cases(args.tag)
    rows = []
    for c in cs:
        new = c["lab"][c["t"]]
        evs = events_near(c["d"], c["t"], new)
        key = ",".join(sorted({"{}:{}{}".format(s, k, "" if ctl else "(unctl)") for _, s, k, ctl in evs})) or "none"
        rows.append((key, c["missed"]))
    from collections import Counter
    tot = Counter(k for k, _ in rows)
    mis = Counter(k for k, m in rows if m)
    nm = sum(m for _, m in rows)
    print("{} true turnovers on DFL, {} missed (no predicted turnover within 2 s, {:.0%})".format(
        len(rows), nm, nm / len(rows)))
    print("events within +-1 s (N = new team, O = old team): share of turnovers, missed rate, share of misses")
    for k, n in tot.most_common(25):
        print("  {:<62} {:>5.1%} missed {:>5.1%} | {:>5.1%}".format(k[:62], n / len(rows), mis[k] / n, mis[k] / nm))


def show_cmd(args):
    cs = cases(args.tag)
    rng = np.random.default_rng(args.seed)
    mi = [c for c in cs if c["missed"]]
    fo = [c for c in cs if not c["missed"]]
    pick = [mi[i] for i in rng.choice(len(mi), args.n, replace=False)] + \
           [fo[i] for i in rng.choice(len(fo), args.found, replace=False)]
    from pipeline import features
    for ci, c in enumerate(pick):
        t, lab, d = c["t"], c["lab"], c["d"]
        new, old = lab[t], lab[t - 1]
        L, W = dd.PITCH
        P = (d["players"][:, :, :2] - 0.5) * np.array(dd.PITCH)
        B = (d["ball"][:, :2] - 0.5) * np.array(dd.PITCH)
        bz = d["ball"][:, 2]
        team = d["team"]
        nt = 0 if new == H else 1
        # attack direction of the new team: away from its own keeper side (mean x of its players
        # vs the other team's over the case is a rough proxy)
        lo, hi = max(t - 2 * FPS, 0), min(t + 2 * FPS + 1, len(lab))
        sgn = 1.0 if np.nanmean(P[lo:hi, team == nt, 0]) < np.nanmean(P[lo:hi, team != nt, 0]) else -1.0
        ids = {}
        print("\n=== case {} ({}) {} -> {} at {} ===".format(ci + 1, "MISSED" if c["missed"] else "found",
                                                         CODE[old], CODE[new], c["g"]))
        print("  events: " + "; ".join("{:+.1f}s {} {}{}".format(o / FPS, s, k, "" if ctl else " (unctl)")
                                     for o, s, k, ctl in events_near(d, t, new, 2 * FPS)))
        print("  {:>5} {:>2} {:>2} | {:>6} {:>5} {:>4} {:>4} {:>2} | {:<34} | {:<34} | {:>5} {:>5}".format(
            "t", "tr", "pr", "ball x", "y", "z", "spd", "sn", "new team: id dist spd tow", "old team: id dist spd tow",
            "Pturn", "Pnew"))
        for tt in range(lo, hi, 5):
            vb = (B[min(tt + 2, len(B) - 1)] - B[max(tt - 2, 0)]) * FPS / 4.0
            line = "  {:>+5.1f} {:>2} {:>2} | {:>6.1f} {:>5.1f} {:>4.1f} {:>4.1f} {:>2} |".format(
                (tt - t) / FPS, CODE.get(lab[tt], "X"), CODE.get(c["dec"][tt], "X"), B[tt, 0] * sgn, B[tt, 1],
                bz[tt] if np.isfinite(bz[tt]) else -1, np.hypot(*vb), "y" if c["seen"][tt] else "-")
            for side in (nt, 1 - nt):
                sel = np.flatnonzero(team == side)
                dist = np.hypot(*(P[tt, sel] - B[tt]).T)
                dist = np.where(np.isnan(dist), np.inf, dist)
                o = sel[np.argsort(dist)[:3]]
                parts = []
                for j in o:
                    dj = np.hypot(*(P[tt, j] - B[tt]))
                    v = (P[min(tt + 2, len(P) - 1), j] - P[max(tt - 2, 0), j]) * FPS / 4.0
                    tow = -np.dot(v, P[tt, j] - B[tt]) / max(dj, 0.3)
                    key = ids.setdefault(j, ("n" if side == nt else "o") + str(len([1 for k in ids if team[k] == side]) + 1))
                    parts.append("{}:{:.1f}/{:.1f}/{:+.1f}".format(key, dj, np.hypot(*v), tow))
                line += " {:<34} |".format(" ".join(parts))
            pt = c["pr"][tt, 0 if new == H else 1]
            pn = c["pr"][tt, C.N_TGT + (0 if new == H else 1)]
            line += " {:>5.2f} {:>5.2f}".format(pt, pn)
            print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("kinds", "show"))
    ap.add_argument("--tag", default="c1dlbd")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--found", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    {"kinds": kinds_cmd, "show": show_cmd}[args.cmd](args)


if __name__ == "__main__":
    main()
