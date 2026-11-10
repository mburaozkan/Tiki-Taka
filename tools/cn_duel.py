"""Duel inputs for the change net: who runs with the ball, and for how long. Aggregate only.

    python tools/cn_duel.py prep [--src xb] [--out xbd]
    python tools/cn_duel.py prep --src base --out xd     # on the original ball (c1dld, 2026-11-10)

Two channels per team, appended to the team block (short_net SN_EXTRA=ctl,keep):
  ctl   |velocity of the team's nearest player - ball velocity| (m/s, 0-15; centred 5-frame
        differences): the duel winner moves with the ball, the loser does not;
  keep  seconds the same player has been the team's nearest to the ball within 3 m (0-3 s).
The ball is the one in the source inputs (src, e.g. xb: ball net in the gaps), read back from the
bx / by channels. Writes data/cache/change_net/<out>/<name>_<game>.npy.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
os.environ["SN_EXTRA"] = "ctl,keep"
import short_net as S  # noqa: E402
import change_net as C  # noqa: E402

FPS = 25
EXTRA = ["ctl", "keep"]


def cdiff(v):
    out = np.full_like(v, np.nan)
    out[2:-2] = (v[4:] - v[:-4]) * FPS / 4.0
    return out


def duel_channels(players, team, bx, by, pitch):
    from pipeline import features
    L, W = pitch
    px = (players[:, :, 0] - 0.5) * L
    py = (players[:, :, 1] - 0.5) * W
    pvx, pvy = cdiff(px), cdiff(py)
    bvx, bvy = cdiff(bx), cdiff(by)
    n = len(bx)
    out = {}
    for tag, tid in (("h_", features.HOME), ("a_", features.AWAY)):
        sel = np.flatnonzero(team == tid)
        ds = np.hypot(px[:, sel] - bx[:, None], py[:, sel] - by[:, None])
        ds = np.where(np.isnan(ds), np.inf, ds)
        j = np.argmin(ds, axis=1)
        dmin = ds[np.arange(n), j]
        r = np.arange(n)
        rel = np.hypot(pvx[:, sel][r, j] - bvx, pvy[:, sel][r, j] - bvy)
        out[tag + "ctl"] = np.clip(np.nan_to_num(rel, nan=15.0), 0, 15).astype(np.float32)
        keep = np.zeros(n, np.float32)
        run = 0
        for t in range(n):
            if dmin[t] <= 3.0 and np.isfinite(bx[t]):
                run = run + 1 if t > 0 and j[t] == j[t - 1] and keep[t - 1] > 0 else 1
            else:
                run = 0
            keep[t] = min(run / FPS, 3.0)
        out[tag + "keep"] = keep
    return out


def one(job):
    name, g, src, outdir = job
    path = os.path.join(C.OUT, outdir, "{}_{}.npy".format(name, g))
    if os.path.isfile(path):
        return None
    import dfl_data as dd
    import turnover_cls as T
    if src == "base":                      # the change net's default inputs (no ball fill)
        if name == "dfl":
            import cn_dfl
            with np.load(cn_dfl.path(g)) as z:
                x = z["x"].astype(np.float32)
        else:
            x = np.load(os.path.join(S.OUT, "in_{}_{}.npy".format(name, g))).astype(np.float32)
    else:
        x = np.load(os.path.join(C.OUT, src, "{}_{}.npy".format(name, g))).astype(np.float32)
    old = S.SHARED + ["h_" + k for k in S.BASE_TEAM] + ["a_" + k for k in S.BASE_TEAM]
    if name == "dfl":
        import cn_dfl
        d = dd.load(g)
        with np.load(cn_dfl.path(g)) as z:
            blk = [tuple(b) for b in z["blk"]]
        period = np.zeros(len(x), np.int8)
        for i, (a, b) in enumerate(blk, start=1):
            period[a:b] = i
        dd_ = {"players": d["players"], "team": d["team"], "period": period}
        pitch = dd.PITCH
    else:
        import run_v31_external as rx
        c, _ = T.load_cache(name, g)
        dd_, pitch, _ = rx.load_set(name, g)
        period = c["period"]
    sign = T.attack_sign(dd_, period)
    bx = x[:, old.index("bx")] * sign
    by = x[:, old.index("by")]
    unseen_all = (x[:, old.index("seen")] < 0.5) & (bx == 0) & (by == 0)
    bx, by = np.where(unseen_all, np.nan, bx), np.where(unseen_all, np.nan, by)
    ex = duel_channels(dd_["players"], dd_["team"], bx, by, pitch)
    cols = []
    for nm in S.channel_names():
        cols.append(ex[nm] if nm[2:] in EXTRA and nm[:2] in ("h_", "a_") else x[:, old.index(nm)])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.stack(cols, axis=1).astype(np.float16))
    return name, float(np.mean(ex["h_keep"] > 0)), float(np.median(ex["h_ctl"]))


def main():
    import turnover_cls as T
    import dfl_data as dd
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("prep",))
    ap.add_argument("--src", default="xb")
    ap.add_argument("--out", default="xbd")
    args = ap.parse_args()
    jobs = [(n, g, args.src, args.out) for n, g, _ in T.ext_list()] + [("dfl", g, args.src, args.out) for g in dd.GAMES]
    with Pool(4) as p:
        res = [r for r in p.map(one, jobs) if r is not None]
    for nm in ("skillcorner", "pff", "dfl"):
        r = [x for x in res if x[0] == nm]
        if r:
            print("{:<12} {} matches: home keep > 0 {:.0%}, median home ctl {:.1f} m/s".format(
                nm, len(r), np.mean([x[1] for x in r]), np.mean([x[2] for x in r])))


if __name__ == "__main__":
    main()
