"""A ball in the gaps: where the ball is while it is unseen, from both ends of the gap and the players.
Aggregate only.

    python tools/ball_fill.py eval            # DFL: ball error in the gaps vs the true ball

We label offline, so both ends of every gap are known. Fillers:
  linear   straight line between the last seen and the next seen ball;
  carrier  (the user's rule) the player nearest the ball when it disappears (within 3 m) keeps it,
           0.5 m in front of him along his running direction; if another player has it when it
           comes back, the ball leaves the first one at the earliest of: an opponent within 1.5 m
           (duel), the carrier stopping or turning sharply, the last moment a 15 m/s pass still
           reaches the second player in time; then it travels straight to him. Both ends are pinned
           to the seen ball (the residual at each end is blended out linearly over the gap).
eval: DFL, ball made like ours (dead ball animated, blurred, our gap pattern), truth = the same ball
without the gaps. Error in metres inside the gaps (median, mean, within 2 / 5 m) and whether the
nearest player's team is the same as with the true ball, by gap length.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

FPS = 25
CARRY_M = 3.0
DUEL_M = 1.5
PASS_MS = 15.0
FRONT_M = 0.5


def gap_runs(ball, blk):
    """[(a, b)] runs of unseen ball inside each block."""
    out = []
    miss = np.isnan(ball[:, 0])
    for s, e in blk:
        m = miss[s:e]
        if not m.any():
            continue
        d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            out.append((s + a, s + b, s, e))
    return out


def to_m(xy, pitch):
    return (xy - 0.5) * np.array(pitch)


def pin_ends(path, b0, b1):
    """shift path (m, 2) so it starts next to b0 and ends next to b1 (linear blend of residuals)."""
    m = len(path)
    s = (np.arange(m) + 1.0) / (m + 1.0)
    r0 = b0 - path[0] if b0 is not None else np.zeros(2)
    r1 = b1 - path[-1] if b1 is not None else np.zeros(2)
    return path + (1 - s)[:, None] * r0 + s[:, None] * r1


def fill_linear(ball, blk):
    out = ball[:, :2].copy()
    for a, b, s, e in gap_runs(ball, blk):
        b0 = out[a - 1] if a > s else None
        b1 = out[b] if b < e else None
        if b0 is None and b1 is None:
            continue
        if b0 is None:
            out[a:b] = b1
        elif b1 is None:
            out[a:b] = b0
        else:
            w = (np.arange(a, b) - (a - 1)) / float(b - a + 1)
            out[a:b] = b0 + w[:, None] * (b1 - b0)
    return out


def fill_carrier(ball, players, team, blk, pitch):
    """ball (n, >=2) normalised with NaN gaps -> (n, 2) normalised, gaps filled."""
    P = to_m(players[:, :, :2], pitch)                     # (n, p, 2)
    V = np.full_like(P, np.nan)
    V[5:] = (P[5:] - P[:-5]) * FPS / 5.0
    V[:5] = V[5]
    B = to_m(ball[:, :2], pitch)
    out = B.copy()
    lin = to_m(fill_linear(ball, blk), pitch)
    for a, b, s, e in gap_runs(ball, blk):
        b0 = B[a - 1] if a > s else None
        b1 = B[b] if b < e else None
        if b0 is None or b1 is None:
            out[a:b] = lin[a:b]
            continue

        def holder(t, bp):
            d = np.hypot(*(P[t] - bp).T)
            d = np.where(np.isnan(d), np.inf, d)
            j = int(np.argmin(d))
            return j if d[j] <= CARRY_M else None
        ps, pe = holder(a - 1, b0), holder(b, b1)
        if ps is None:
            out[a:b] = lin[a:b]
            continue
        T = np.arange(a, b)
        pp = P[T, ps]
        vv = V[T, ps]
        if np.isnan(pp).any():
            out[a:b] = lin[a:b]
            continue
        spd = np.hypot(vv[:, 0], vv[:, 1])
        unit = np.where(spd[:, None] > 1.0, vv / np.maximum(spd, 1e-6)[:, None], 0.0)
        carry = pp + FRONT_M * np.nan_to_num(unit)
        if pe == ps:
            out[a:b] = pin_ends(carry, b0, b1)
            continue
        # when does the ball leave the carrier?
        dest = P[b, pe] if pe is not None else b1
        fly = np.hypot(*(dest - pp).T) / PASS_MS * FPS              # frames of flight from t
        ok = np.flatnonzero(T + fly <= b)
        th = int(ok[-1]) if len(ok) else 0
        cand = [th]
        opp = np.flatnonzero(team != team[ps])
        if len(opp):
            dopp = np.hypot(*(P[T][:, opp] - pp[:, None]).transpose(2, 0, 1))
            hit = np.flatnonzero(np.nanmin(np.where(np.isnan(dopp), np.inf, dopp), axis=1) < DUEL_M)
            if len(hit):
                cand.append(int(hit[0]))
        if len(spd) > 1:
            ang = np.abs(np.arctan2(vv[1:, 1], vv[1:, 0]) - np.arctan2(vv[:-1, 1], vv[:-1, 0]))
            ang = np.minimum(ang, 2 * np.pi - ang)
            turn = np.flatnonzero(((spd[1:] < 1.0) & (spd[:-1] >= 2.0)) | ((ang > 1.2) & (spd[1:] > 2.0)))
            if len(turn):
                cand.append(int(turn[0]) + 1)
        th = max(min(cand), 0)
        path = carry.copy()
        start = carry[th]
        k = b - (a + th)
        w = (np.arange(k) + 1.0) / (k + 1.0)
        path[th:] = start + w[:, None] * (dest - start)
        out[a:b] = pin_ends(path, b0, b1)
    return out / np.array(pitch) + 0.5


def dfl_match(g):
    import dfl_data as dd
    from train_learned_dfl import test_seed
    d = dd.load(g)
    seed = test_seed(g)
    ball = dd.prepare_ball(d, seed)
    truth = dd.prepare_ball(d, seed, gaps=False)
    import cn_dfl
    with np.load(cn_dfl.path(g)) as z:
        blk = [tuple(b) for b in z["blk"]]
    return d, ball, truth, blk


def eval_one(g):
    import dfl_data as dd
    d, ball, truth, blk = dfl_match(g)
    gap = np.isnan(ball[:, 0]) & np.isfinite(truth[:, 0])
    inblk = np.zeros(len(ball), bool)
    for s, e in blk:
        inblk[s:e] = True
    gap &= inblk
    glen = np.zeros(len(ball))
    for a, b, _, _ in gap_runs(ball, blk):
        glen[a:b] = (b - a) / FPS
    fills = {"linear": fill_linear(ball, blk), "carrier": fill_carrier(ball, d["players"], d["team"], blk, dd.PITCH)}
    P = to_m(d["players"][:, :, :2], dd.PITCH)

    def near_team(xy):
        dd_ = np.hypot(*(P - to_m(xy, dd.PITCH)[:, None]).transpose(2, 0, 1))
        dd_ = np.where(np.isnan(dd_), np.inf, dd_)
        return d["team"][np.argmin(dd_, axis=1)]
    tt = near_team(truth[:, :2])
    import cn_dfl
    with np.load(cn_dfl.path(g)) as z:
        live = z["lab"] <= 1
    res = {}
    for nm, f in fills.items():
        err = np.hypot(*((to_m(f, dd.PITCH) - to_m(truth[:, :2], dd.PITCH))[gap]).T)
        same = near_team(f)[gap] == tt[gap]
        res[nm] = (err, same, glen[gap], live[gap])
    return res


def eval_cmd(args):
    import dfl_data as dd
    from multiprocessing import Pool
    with Pool(7) as p:
        outs = p.map(eval_one, dd.GAMES)
    print("DFL, 7 matches, ball in the gaps vs the true ball")
    bins = [(0, 5), (5, 10), (10, 30), (30, 1e9)]
    print("{:<8} {:>7} {:>7} {:>6} {:>6} {:>10} | team right by gap < 5 / 5-10 / 10-30 / >= 30 s | live frames: median, team right".format(
        "filler", "median", "mean", "<2 m", "<5 m", "team right"))
    for nm in outs[0]:
        err = np.concatenate([o[nm][0] for o in outs])
        same = np.concatenate([o[nm][1] for o in outs])
        gl = np.concatenate([o[nm][2] for o in outs])
        lv = np.concatenate([o[nm][3] for o in outs])
        print("{:<8} {:>6.1f}m {:>6.1f}m {:>5.0%} {:>5.0%} {:>9.1%} | ".format(
            nm, np.median(err), err.mean(), np.mean(err < 2), np.mean(err < 5), same.mean())
            + " / ".join("{:.1%}".format(same[(gl >= lo) & (gl < hi)].mean()) for lo, hi in bins)
            + " | {:.1f}m, {:.1%}".format(np.median(err[lv]), same[lv].mean()))
    gl = np.concatenate([o["linear"][2] for o in outs])
    lv = np.concatenate([o["linear"][3] for o in outs])
    print("live share of gap frames {:.0%}".format(lv.mean()))
    print("gap frames {} ({} per match); share by gap length: ".format(
        len(gl), len(gl) // 7) + " / ".join("{:.0%}".format(np.mean((gl >= lo) & (gl < hi))) for lo, hi in bins))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("eval",))
    args = ap.parse_args()
    {"eval": eval_cmd}[args.cmd](args)


if __name__ == "__main__":
    main()
