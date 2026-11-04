"""Landing rule for long kicks (user's idea): at the landing, the team that closes on the ball gets
it. Hand rule on top of v31. Aggregate only.

    python tools/landing_rule.py                    # grid on SkillCorner + PFF (cached v31 runs)
    python tools/landing_rule.py --ours MODE MARGIN DEEP KEEP --name N   # our match, public check

Kick: a touch of team T (v31's touch picks, f["touch"]) while v31 says T, after which the ball is
seen >= MIN_M from the kick point within MAX_FLIGHT_S. Landing L: the first peak of the touch
network's P(touch) (either team) >= LAND_P at a frame where the ball is >= MIN_M from the kick
point, within MAX_FLIGHT_S; the landing point is the ball there (nearest seen frame within +/- 5).
DEEP: only kicks from within DEEP_M of T's own goal line (keeper kicks, as #10 / #12).

Who wins at L (O = the other team):
  dist   O's nearest player to the landing point is closer than T's by MARGIN m
  close  the same on "arrival" distance: distance at L minus CLOSE_S x closing speed (how fast
         that player's distance to the landing point fell over the last CLOSE_S)
  touch  max P(O touch) over L +/- TOUCH_WIN_S beats max P(T touch) by MARGIN / 10
  both   dist and touch agree
If O wins and v31 has T at L: O gets [L, next T touch) when that touch comes within KEEP s
(no T touch in time: nothing changes); KEEP 0 = a turnover, O until v31's next change.

Per setting: mean spellF1, delta vs v31, matches up / down, firings, how often O really has the
ball at L + 0.2 s (the rule's precision), short spells caught.
"""

import argparse
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
import dfl_data as dd  # noqa: E402
import run_v31_external as rx  # noqa: E402
import eval_touch_run as tr  # noqa: E402
from pipeline.decode import H, A, D, X  # noqa: E402

# Params
FPS = 25
MIN_M = 35.0
MAX_FLIGHT_S = 5.0
LAND_P = 0.3
DEEP_M = 25.0
CLOSE_S = 0.5
TOUCH_WIN_S = 0.4
MODES = ("dist", "close", "touch", "both")
MARGINS = (0.0, 1.0, 2.0, 4.0)
DEEPS = (True, False)
KEEPS = (2.0, 3.0, 5.0, 0.0)    # 0: turnover, O until v31's next change (at most TURN_MAX_S)
TURN_MAX_S = 10.0


def _pos(ball, i, pitch, r=5):
    for o in sorted(range(-r, r + 1), key=abs):
        j = i + o
        if 0 <= j < len(ball) and not np.isnan(ball[j, 0]):
            return ball[j, :2] * pitch
    return None


def keeper_sides(players, team, period):
    """{(team, period): goal x in metres-normalised units (0 or 1)} from each team's extreme player."""
    out = {}
    for per in np.unique(period):
        sel = period == per
        for t in (0, 1):
            xs = np.nanmean(players[sel][:, team == t, 0], axis=0)
            xs = xs[~np.isnan(xs)]
            out[(t, int(per))] = None if not len(xs) else (1.0 if xs[np.argmax(np.abs(xs - 0.5))]
                                                           > 0.5 else 0.0)
    return out


def landings(players, team, ball, period, f, codes, pitch):
    """[(kick frame, landing frame, T, deep, dT, dO, arrT, arrO, pT, pO)] for every long kick."""
    touch = np.asarray(f["touch"])
    ptouch = np.stack([f["touch_p_home"], f["touch_p_away"]], axis=1).astype(np.float64)
    score = ptouch.sum(1)
    peak = score.copy()
    for s in range(1, 4):
        np.maximum(peak[s:], score[:-s], out=peak[s:])
        np.maximum(peak[:-s], score[s:], out=peak[:-s])
    cand = np.flatnonzero((score >= LAND_P) & (score == peak))
    goals = keeper_sides(players, team, period)
    L, W = pitch
    pxy = np.stack([players[:, :, 0] * L, players[:, :, 1] * W], axis=-1)
    out = []
    win, cw = int(TOUCH_WIN_S * FPS), int(CLOSE_S * FPS)
    for k in np.flatnonzero(touch != 0):
        T = H if touch[k] > 0 else A
        if codes[k] != T:
            continue
        p0 = _pos(ball, k, np.array(pitch), r=2)
        if p0 is None:
            continue
        lo, hi = np.searchsorted(cand, [k + 3, k + int(MAX_FLIGHT_S * FPS) + 1])
        land = None
        for c in cand[lo:hi]:                 # the FIRST touch peak after the kick must be far
            if period[c] != period[k]:
                break
            q = _pos(ball, c, np.array(pitch))
            if q is None:
                continue
            if np.hypot(*(q - p0)) >= MIN_M:
                land, pl = c, q
            break
        if land is None:
            continue
        g = goals.get((T, int(period[k])))
        deep = g is not None and abs(p0[0] - g * L) < DEEP_M
        d = np.hypot(pxy[land, :, 0] - pl[0], pxy[land, :, 1] - pl[1])
        prev = max(land - cw, 0)
        d0 = np.hypot(pxy[prev, :, 0] - pl[0], pxy[prev, :, 1] - pl[1])
        arr = d - np.nan_to_num(d0 - d, nan=0.0)          # distance minus CLOSE_S x closing speed
        row = [k, land, T, deep]
        for vals in (d, arr):
            for t in (T, 1 - T):
                v = vals[team == t]
                row.append(np.nanmin(v) if np.any(~np.isnan(v)) else np.inf)
        a, b = max(land - win, 0), land + win + 1
        row += [ptouch[a:b, T].max(), ptouch[a:b, 1 - T].max()]
        out.append(tuple(row))
    return out


def o_wins(row, mode, margin):
    _, _, _, _, dT, dO, aT, aO, pT, pO = row
    by_dist = dO + margin < dT
    if mode == "dist":
        return by_dist
    if mode == "close":
        return aO + margin < aT
    by_touch = pO > pT + margin / 10.0
    return by_touch if mode == "touch" else (by_touch and by_dist)


def apply(codes, f, rows, mode, margin, deep_only, keep):
    """(new codes, [landing frames where the rule fired])."""
    touch = np.asarray(f["touch"])
    out, fired = codes.copy(), []
    for row in rows:
        k, land, T, deep = row[:4]
        if (deep_only and not deep) or not o_wins(row, mode, margin) or codes[land] != T:
            continue
        O = 1 - T
        if keep == 0:
            seg = codes[land:land + int(TURN_MAX_S * FPS)]
            ch = np.flatnonzero(seg != T)
            out[land:land + (ch[0] if len(ch) else len(seg))] = O
            fired.append((land, O))
            continue
        nxt = np.flatnonzero(touch[land + 1:land + 1 + int(keep * FPS)] == (1 if T == H else -1))
        if not len(nxt):
            continue
        out[land:land + 1 + nxt[0]] = O
        fired.append((land, O))
    return out, fired


def caught_n(lab, codes):
    short = rx.short_spells(lab)
    sp = pm.possession_spells(codes, pm.valid_runs(lab))
    return len(short), sum(rx.caught(s, sp) for s in short)


def external():
    import short_head as sh
    data = sh.load_all()
    grid = list(itertools.product(MODES, MARGINS, DEEPS, KEEPS))
    base, res, fires, hits, cat = {}, {s: {} for s in grid}, {s: 0 for s in grid}, \
        {s: 0 for s in grid}, {s: 0 for s in grid}
    cat0 = [0, 0]
    n_kicks = [0, 0]
    for key in sorted(data):
        name, g = key
        E = data[key]
        d, pitch, ball = rx.load_set(name, g)
        ball = dd.blur_ball(ball)
        f = {k[2:]: v for k, v in E.items() if k.startswith("f_")}
        codes, lab = E["v31"], E["labels"]
        rows = landings(d["players"], d["team"], ball, d["period"], f, codes, pitch)
        n_kicks[0] += len(rows)
        n_kicks[1] += sum(r[3] for r in rows)
        base[key] = tr.score(lab, codes)
        n0, c0 = caught_n(lab, codes)
        cat0[0] += n0
        cat0[1] += c0
        for s in grid:
            new, fired = apply(codes, f, rows, *s)
            res[s][key] = tr.score(lab, new) if fired else base[key]
            fires[s] += len(fired)
            hits[s] += sum(lab[min(land + 5, len(lab) - 1)] == O for land, O in fired)
            cat[s] += caught_n(lab, new)[1] if fired else c0
    keys = sorted(data)
    b = np.array([base[k] for k in keys])
    print("{} matches, long kicks {} (deep {}); v31 {:.4f}, short caught {} / {}".format(
        len(keys), n_kicks[0], n_kicks[1], b.mean(), cat0[1], cat0[0]))
    ranked = sorted(grid, key=lambda s: -np.mean([res[s][k] for k in keys]))
    for s in ranked:
        v = np.array([res[s][k] for k in keys])
        print("{:<5} margin {:<3g} deep {:<5} keep {:<3g} | {:.4f} {:+.4f} | {}/{} | fired {} "
              "precision {:.2f} | caught {}".format(
                  *s, v.mean(), v.mean() - b.mean(), int(np.sum(v > b + 1e-9)),
                  int(np.sum(v < b - 1e-9)), fires[s], hits[s] / max(fires[s], 1), cat[s]))


def ours(mode, margin, deep_only, keep, name):
    import ours_chain as oc
    from atd_cache import load_arrays
    from pipeline.features import LENGTH, WIDTH
    codes, E = oc.ours_v31()
    atd = load_arrays()
    f = {k[2:]: v for k, v in E.items() if k.startswith("f_")}
    period = np.zeros(len(codes), dtype=np.int8)
    for i, (a, b) in enumerate(E["blk"], start=1):
        period[a:b] = i
    rows = landings(atd["players"], atd["team"], dd.blur_ball(atd["ball"]), period, f, codes,
                    (LENGTH, WIDTH))
    print("long kicks {} (deep {})".format(len(rows), sum(r[3] for r in rows)))
    print(oc.write_and_score(codes, "v31_check"))
    settings = [(mode, margin, deep_only, keep)] if mode != "all" else \
        list(itertools.product(MODES, MARGINS, DEEPS, KEEPS))
    from report import PUBLIC_GT
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    win = gt != X
    for s in settings:
        new, fired = apply(codes, f, rows, *s)
        n_win = sum(win[land] for land, _ in fired)
        tag = name if mode != "all" else "{}_{}_{:g}_{}_{:g}".format(name, *s)
        print(oc.write_and_score(new, tag), " fired {} (in public window {})".format(len(fired), n_win))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ours", nargs=4, metavar=("MODE", "MARGIN", "DEEP", "KEEP"),
                        help="MODE 'all' runs the whole grid")
    parser.add_argument("--name", default="v31_land")
    args = parser.parse_args()
    if args.ours:
        m, mg, dp, kp = args.ours
        ours(m, float(mg), dp.lower() in ("1", "true", "yes"), float(kp), args.name)
    else:
        external()


if __name__ == "__main__":
    main()
