"""Change-net inputs with a ball in the gaps (tools/ball_net.py). Aggregate only.

    python tools/cn_ballfill.py prep [--src net|carrier] [--out xb]

Rewrites only the gap frames of the ball-derived inputs (ball position / velocity / speed /
acceleration, and per team the nearest distances, counts within 2 / 5 / 10 m, nearest player's speed
and approach) from the filled ball; `seen` stays 0 there, so the net knows the ball is estimated.
Also adds nothing else: same channels, same order. DFL uses the ball net that never saw that match
(ball_net_<game>), SkillCorner / PFF the one trained on all 7 DFL matches (ball_net_all).
Writes data/cache/change_net/<out>/<name>_<game>.npy (float16). Prints the agreement of the rebuilt
`seen` with the stored one and the share of frames changed.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import short_net as S  # noqa: E402
import change_net as C  # noqa: E402
import ball_fill as BF  # noqa: E402

FPS = 25


def geom(players, team, ball, pitch, sign_home):
    """the ball-derived channels of short_net.frame_inputs, as a dict of (n,) arrays."""
    from pipeline import features
    L, W = pitch
    n = len(ball)
    bx = (ball[:, 0] - 0.5) * L
    by = (ball[:, 1] - 0.5) * W
    vx = np.gradient(bx) * FPS
    vy = np.gradient(by) * FPS
    spd = np.hypot(vx, vy)
    acc = np.abs(np.gradient(spd)) * FPS
    out = {"bx": np.nan_to_num(bx * sign_home), "by": np.nan_to_num(by),
           "bvx": np.nan_to_num(vx * sign_home), "bvy": np.nan_to_num(vy),
           "bspd": np.nan_to_num(np.clip(spd, 0, 40)), "bacc": np.nan_to_num(np.clip(acc, 0, 300))}
    px = (players[:, :, 0] - 0.5) * L
    py = (players[:, :, 1] - 0.5) * W
    pvx = np.zeros_like(px)
    pvy = np.zeros_like(py)
    pvx[5:], pvy[5:] = (px[5:] - px[:-5]) * FPS / 5.0, (py[5:] - py[:-5]) * FPS / 5.0
    dx, dy = px - bx[:, None], py - by[:, None]
    dist = np.hypot(dx, dy)
    for tag, tid in (("h_", features.HOME), ("a_", features.AWAY)):
        sel = team == tid
        ds = np.where(np.isnan(dist[:, sel]), np.inf, dist[:, sel])
        srt = np.sort(ds, axis=1)
        for r in range(3):
            out[tag + "d%d" % (r + 1)] = np.clip(np.where(np.isinf(srt[:, r]), 40, srt[:, r]), 0, 40)
        for m in (2, 5, 10):
            out[tag + "n%d" % m] = np.sum(ds < m, axis=1).astype(np.float32)
        j = np.argmin(ds, axis=1)
        rows = np.arange(n)
        vxj, vyj = pvx[:, sel][rows, j], pvy[:, sel][rows, j]
        dxj, dyj = dx[:, sel][rows, j], dy[:, sel][rows, j]
        dj = np.maximum(np.hypot(dxj, dyj), 0.3)
        out[tag + "nspd"] = np.clip(np.nan_to_num(np.hypot(vxj, vyj)), 0, 12)
        out[tag + "ntow"] = np.clip(np.nan_to_num(-(vxj * dxj + vyj * dyj) / dj), -12, 12)
    return out


def one(job):
    name, g, src, outdir = job
    path = os.path.join(C.OUT, outdir, "{}_{}.npy".format(name, g))
    if os.path.isfile(path):
        return None
    import dfl_data as dd
    import turnover_cls as T
    import ball_net as N
    if name == "dfl":
        import cn_dfl
        from train_learned_dfl import test_seed
        d = dd.load(g)
        with np.load(cn_dfl.path(g)) as z:
            x = z["x"].astype(np.float32)
            blk = [tuple(b) for b in z["blk"]]
        period = np.zeros(len(x), np.int8)
        for i, (a, b) in enumerate(blk, start=1):
            period[a:b] = i
        ball = dd.prepare_ball(d, test_seed(g))
        dd_ = {"players": d["players"], "team": d["team"], "period": period}
        pitch, tag = dd.PITCH, g
    else:
        import run_v31_external as rx
        c, _ = T.load_cache(name, g)
        dd_, pitch, ball = rx.load_set(name, g)
        ball = dd.blur_ball(ball)
        period = c["period"]
        x = np.load(os.path.join(S.OUT, "in_{}_{}.npy".format(name, g))).astype(np.float32)
        blk = [(a, b) for a, b in __import__("run_v31_external").blocks(period)]
        tag = "all"
    names = S.channel_names()
    seen_x = x[:, names.index("seen")] > 0.5
    seen_b = np.isfinite(ball[:, 0])
    if src == "net":
        filled, _ = N.fill(N.load(tag), ball, dd_["players"], dd_["team"], blk, pitch)
    else:
        filled = BF.fill_carrier(ball, dd_["players"], dd_["team"], blk, pitch)
    sign = T.attack_sign(dd_, period)
    gm = geom(dd_["players"], dd_["team"], filled, pitch, sign)
    gap = ~seen_b & np.isfinite(filled[:, 0])
    for k, v in gm.items():
        x[gap, names.index(k)] = v[gap]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, x.astype(np.float16))
    return name, float(np.mean(seen_x == seen_b)), float(gap.mean())


def prep_cmd(args):
    import turnover_cls as T
    import dfl_data as dd
    jobs = [(n, g, args.src, args.out) for n, g, _ in T.ext_list()] + [("dfl", g, args.src, args.out) for g in dd.GAMES]
    res = [r for r in map(one, jobs) if r is not None]
    for nm in ("skillcorner", "pff", "dfl"):
        r = [x for x in res if x[0] == nm]
        if r:
            print("{:<12} {} matches: seen agrees {:.3%}, frames filled {:.1%}".format(
                nm, len(r), np.mean([x[1] for x in r]), np.mean([x[2] for x in r])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("prep",))
    ap.add_argument("--src", default="net")
    ap.add_argument("--out", default="xb")
    args = ap.parse_args()
    prep_cmd(args)


if __name__ == "__main__":
    main()
