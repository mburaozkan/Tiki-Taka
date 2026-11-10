"""Ball-free player-motion inputs for the change net. Aggregate only.

    python tools/cn_motion.py prep [--out xm]
    $env:CN_X="xm"; $env:SN_EXTRA="mspd,top3,run4"; $env:CN_DFL="5"
    python tools/change_net.py train --tag c1dlm --folds 0 1 2 3

Why: when the ball is unseen every player input of the net is measured from the ball (nearest
distances, counts, approach speed) and goes blank, so in a ball gap the net sees only the team
centroid / spread. Live play in a gap is then called D: live frames with the ball unseen are called D
33% (DFL) / 41% (SkillCorner) / 51% (PFF) of the time, vs 3-5% with the ball seen (2026-11-10).
Three channels per team, appended to the team block (short_net SN_EXTRA), none uses the ball:
  mspd  mean player speed (m/s, 0.2 s differences, 0-12)
  top3  mean speed of the team's three fastest players (0-12)
  run4  players faster than 4 m/s (count)
Source inputs: the change net's default ones (short_net in_*.npy, cn_dfl dfl_*.npz).
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
EXTRA = ["mspd", "top3", "run4"]
os.environ["SN_EXTRA"] = ",".join(EXTRA)
import short_net as S  # noqa: E402
import change_net as C  # noqa: E402

FPS = 25


def motion_channels(players, team, pitch):
    from pipeline import features
    L, W = pitch
    px = (players[:, :, 0] - 0.5) * L
    py = (players[:, :, 1] - 0.5) * W
    spd = np.full(px.shape, np.nan, np.float32)
    spd[5:] = np.hypot(px[5:] - px[:-5], py[5:] - py[:-5]) * FPS / 5.0
    spd[:5] = spd[5:6]
    spd = np.clip(spd, 0, 12)
    out = {}
    for tag, tid in (("h_", features.HOME), ("a_", features.AWAY)):
        s = spd[:, team == tid]
        ok = np.isfinite(s)
        n = np.maximum(ok.sum(axis=1), 1)
        out[tag + "mspd"] = (np.where(ok, s, 0).sum(axis=1) / n).astype(np.float32)
        srt = -np.sort(-np.where(ok, s, -1), axis=1)[:, :3]
        out[tag + "top3"] = np.clip(srt, 0, None).mean(axis=1).astype(np.float32)
        out[tag + "run4"] = np.sum(np.where(ok, s, 0) > 4.0, axis=1).astype(np.float32)
    return out


def one(job):
    name, g, outdir = job
    path = os.path.join(C.OUT, outdir, "{}_{}.npy".format(name, g))
    if os.path.isfile(path):
        return None
    import dfl_data as dd
    old = S.SHARED + ["h_" + k for k in S.BASE_TEAM] + ["a_" + k for k in S.BASE_TEAM]
    if name == "dfl":
        import cn_dfl
        with np.load(cn_dfl.path(g)) as z:
            x = z["x"]
        d = dd.load(g)
        players, team, pitch = d["players"], d["team"], dd.PITCH
    else:
        import run_v31_external as rx
        x = np.load(os.path.join(S.OUT, "in_{}_{}.npy".format(name, g)))
        dd_, pitch, _ = rx.load_set(name, g)
        players, team = dd_["players"], dd_["team"]
    assert x.shape[1] == len(old) and len(players) == len(x), (name, g, x.shape, len(players))
    ex = motion_channels(players, team, pitch)
    cols = [ex[nm] if nm[2:] in EXTRA and nm[:2] in ("h_", "a_") else x[:, old.index(nm)] for nm in S.channel_names()]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.stack(cols, axis=1).astype(np.float16))
    seen = np.asarray(x[:, old.index("seen")], np.float32) > 0.5
    return name, float(np.mean(ex["h_mspd"][seen])), float(np.mean(ex["h_mspd"][~seen])) if (~seen).any() else np.nan


def main():
    import turnover_cls as T
    import dfl_data as dd
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("prep",))
    ap.add_argument("--out", default="xm")
    args = ap.parse_args()
    jobs = [(n, g, args.out) for n, g, _ in T.ext_list()] + [("dfl", g, args.out) for g in dd.GAMES]
    with Pool(4) as p:
        res = [r for r in p.map(one, jobs, chunksize=1) if r is not None]
    for nm in ("skillcorner", "pff", "dfl"):
        r = [x for x in res if x[0] == nm]
        if r:
            print("{:<12} {} matches: home mean speed, ball seen {:.2f} m/s, ball unseen {:.2f} m/s".format(
                nm, len(r), np.mean([x[1] for x in r]), np.nanmean([x[2] for x in r])))


if __name__ == "__main__":
    main()
