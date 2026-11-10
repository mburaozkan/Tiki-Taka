"""Loose-ball memory inputs for the change net: who controlled the ball last, and how long ago.
Aggregate only.

    python tools/cn_loose.py prep [--src xd] [--out xdl]
    $env:CN_X="xdl"; $env:SN_EXTRA="ctl,keep,since,last"; $env:CN_DFL="5"
    python tools/change_net.py train --tag c1dldl --folds 0 1 2 3 all

Why (2026-11-10, DFL timelines): the "other team" misses are loose and airborne balls (clearances,
long balls, throw-ins rolling past a defender, contested headers); the truth keeps possession with
the team that last controlled the ball until the other team controls it, we hand it over early.
Control = cn_duel's keep (the same player nearest within 3 m) >= CTL_S. Two channels per team,
appended after cn_duel's ctl / keep (source src must already hold them, e.g. xd):
  since  seconds since the team last had control (0 while it has it, capped at 10; 10 before any)
  last   +1 if this team had the latest control of either team, else 0
Memory runs through gaps and dead ball within a period and resets at each period start.
Writes data/cache/change_net/<out>/<name>_<game>.npy.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
SRC_EXTRA = ["ctl", "keep"]
EXTRA = ["since", "last"]
os.environ["SN_EXTRA"] = ",".join(SRC_EXTRA + EXTRA)
import short_net as S  # noqa: E402
import change_net as C  # noqa: E402

FPS = 25
CTL_S = 0.2
CAP = 10.0


def loose_channels(hkeep, akeep, starts):
    n = len(hkeep)
    out = {k: np.zeros(n, np.float32) for k in ("h_since", "a_since", "h_last", "a_last")}
    hs = as_ = CAP
    last = -1
    st = set(int(s) for s in starts)
    for t in range(n):
        if t in st:
            hs = as_ = CAP
            last = -1
        hc, ac = hkeep[t] >= CTL_S, akeep[t] >= CTL_S
        hs = 0.0 if hc else min(hs + 1.0 / FPS, CAP)
        as_ = 0.0 if ac else min(as_ + 1.0 / FPS, CAP)
        if hc and not ac:
            last = 0
        elif ac and not hc:
            last = 1
        out["h_since"][t], out["a_since"][t] = hs, as_
        out["h_last"][t], out["a_last"][t] = last == 0, last == 1
    return out


def one(job):
    name, g, src, outdir = job
    path = os.path.join(C.OUT, outdir, "{}_{}.npy".format(name, g))
    if os.path.isfile(path):
        return None
    x = np.load(os.path.join(C.OUT, src, "{}_{}.npy".format(name, g))).astype(np.float32)
    old = S.SHARED + ["h_" + k for k in S.BASE_TEAM + SRC_EXTRA] + ["a_" + k for k in S.BASE_TEAM + SRC_EXTRA]
    assert x.shape[1] == len(old), (name, g, x.shape)
    if name == "dfl":
        import cn_dfl
        with np.load(cn_dfl.path(g)) as z:
            starts = [int(b[0]) for b in z["blk"]]
    else:
        import turnover_cls as T
        c, _ = T.load_cache(name, g)
        p = np.asarray(c["period"])
        starts = list(np.flatnonzero(np.diff(np.concatenate(([-1], p))) != 0))
    ex = loose_channels(x[:, old.index("h_keep")], x[:, old.index("a_keep")], starts)
    cols = [ex[nm] if nm[2:] in EXTRA and nm[:2] in ("h_", "a_") else x[:, old.index(nm)] for nm in S.channel_names()]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.stack(cols, axis=1).astype(np.float16))
    return name, float(np.mean(ex["h_since"] < 1.0)), float(np.mean(ex["h_last"] + ex["a_last"]))


def main():
    import turnover_cls as T
    import dfl_data as dd
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("prep",))
    ap.add_argument("--src", default="xd")
    ap.add_argument("--out", default="xdl")
    args = ap.parse_args()
    jobs = [(n, g, args.src, args.out) for n, g, _ in T.ext_list()] + [("dfl", g, args.src, args.out) for g in dd.GAMES]
    with Pool(3) as p:
        res = [r for r in p.map(one, jobs, chunksize=1) if r is not None]
    for nm in ("skillcorner", "pff", "dfl"):
        r = [x for x in res if x[0] == nm]
        if r:
            print("{:<12} {} matches: home controlled within 1 s {:.0%}, a last-control team known {:.0%}".format(
                nm, len(r), np.mean([x[1] for x in r]), np.mean([x[2] for x in r])))


if __name__ == "__main__":
    main()
