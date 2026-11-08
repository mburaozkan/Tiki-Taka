"""YOLO ball vs tracking ball at the public possession starts. Flags and peak sizes only.

    $env:MTAH_VARIANT="_kt"; python tools/yolo_contacts.py

Per true possession start (+-0.5 s; numbered as in spell_table): the largest velocity change
(m/s, +-3 frames) of the raw YOLO ball (full-match pass, mapped through the homography), of the
cleaned YOLO ball (yolo_ball.py clean) and of the tracking ball; at the raw YOLO peak, whether a
player is within 2 m and whose (new / old team). Also the typical raw YOLO velocity change on all
public live frames (median, 90th percentile), to judge the size of a peak.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "starter")]
import possession_metrics as pm  # noqa: E402
from report import PUBLIC_GT  # noqa: E402

FPS = 25


def dv_of(xy, k=3):
    v = np.gradient(xy, axis=0) * FPS
    out = np.full(len(xy), np.nan)
    out[k:-k] = np.hypot(*(v[2 * k:] - v[:-2 * k]).T)
    return out


def main():
    from load_homography import load as load_h
    from atd_cache import load_arrays
    from games import GAMES, GAME
    L, W = GAMES[GAME]["pitch"]
    atd = load_arrays("tactical")
    n = len(atd["ball"])
    z = np.load(os.path.join(ROOT, "data", "cache", "yolo", "public_bambezius.npz"))
    det = np.full((n, 3), np.nan)
    a0 = int(z["a"])
    m = min(n - a0, len(z["det"]))
    det[a0:a0 + m] = z["det"][:m]
    Hm, _, valid = load_h()
    Hm = Hm[:n].astype(np.float64)
    ok = (valid[:n] > 0) & np.isfinite(det[:, 0])
    Hi = np.full_like(Hm, np.nan)
    Hi[ok] = np.linalg.inv(Hm[ok])
    v = np.einsum("nij,nj->ni", Hi, np.column_stack([det[:, 0], det[:, 1], np.ones(n)]))
    raw = v[:, :2] / v[:, 2:3] * np.array([L, W])
    raw[~ok] = np.nan
    clean = np.load(os.path.join(ROOT, "data", "cache", "yolo", "ball_clean_bambezius.npy")).astype(np.float64) * np.array([L, W])
    trk = atd["ball"][:, :2].astype(np.float64) * np.array([L, W])
    P = atd["players"][:, :, :2].astype(np.float64) * np.array([L, W])
    team = np.asarray(atd["team"])
    from atd_cache import HOME
    dr, dc, dt = dv_of(raw), dv_of(clean), dv_of(trk)
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    runs = pm.valid_runs(gt)
    live = np.isin(gt, (0, 1))
    print("public live frames: raw YOLO velocity change median {:.1f}, p90 {:.1f} m/s; tracking median {:.1f}, p90 {:.1f}".format(
        np.nanmedian(dr[live]), np.nanpercentile(dr[live], 90), np.nanmedian(dt[live]), np.nanpercentile(dt[live], 90)))
    print("{:<4} {:<8} | {:>9} {:>11} {:>9} | at the raw YOLO peak: player within 2 m".format(
        "#", "context", "YOLO raw", "YOLO clean", "tracking"))
    for i, (a, b, k) in enumerate(pm.possession_spells(gt, runs)):
        w = np.arange(max(a - 12, 0), a + 13)
        prev = gt[a - 1] if a > 0 else -1
        ctx = {0: "H", 1: "A", 2: "D"}.get(int(prev), "|") + ">" + "HA"[k]
        jr = w[np.nanargmax(np.nan_to_num(dr[w], nan=-1))]
        who = "-"
        if np.isfinite(raw[jr, 0]):
            d = np.hypot(*(P[jr] - raw[jr]).T)
            d = np.where(np.isnan(d), np.inf, d)
            j = int(np.argmin(d))
            if d[j] <= 2:
                side = 0 if team[j] == HOME else 1
                who = "new team" if side == k else "old team"
        print("#{:<3} {:<8} | {:>9.1f} {:>11.1f} {:>9.1f} | {}".format(
            i + 1, ctx, np.nanmax(np.nan_to_num(dr[w], nan=0)), np.nanmax(np.nan_to_num(dc[w], nan=0)),
            np.nanmax(np.nan_to_num(dt[w], nan=0)), who))


if __name__ == "__main__" and (len(sys.argv) < 2 or sys.argv[1] != "local"):
    main()


def local_cmd():
    """windows +-0.1 / +-0.5 s, the last peak before each start and its local rank; false-alarm rate."""
    from atd_cache import load_arrays
    from games import GAMES, GAME
    from scipy.ndimage import maximum_filter1d
    L, W = GAMES[GAME]["pitch"]
    clean = np.load(os.path.join(ROOT, "data", "cache", "yolo", "ball_clean_bambezius.npy")).astype(np.float64) * np.array([L, W])
    trk = load_arrays("tactical")["ball"][:, :2].astype(np.float64) * np.array([L, W])
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    runs = pm.valid_runs(gt)
    sp = pm.possession_spells(gt, runs)
    live = np.isin(gt, (0, 1))
    for nm, xy in (("cleaned YOLO", clean), ("tracking", trk)):
        d = np.nan_to_num(dv_of(xy))
        pk = np.flatnonzero((d >= 2) & (d == maximum_filter1d(d, 7)))
        print("\n{}: per true start".format(nm))
        print("{:<4} {:<5} | {:>7} {:>7} | last peak before (<= 1 s): height, seconds before, rank in +-5 s".format(
            "#", "ctx", "+-0.1s", "+-0.5s"))
        hits = []
        for i, (a, b, k) in enumerate(sp):
            prev = gt[a - 1] if a > 0 else -1
            ctx = {0: "H", 1: "A", 2: "D"}.get(int(prev), "|") + ">" + "HA"[k]
            w1, w5 = d[max(a - 3, 0):a + 4], d[max(a - 12, 0):a + 13]
            before = pk[(pk <= a + 3) & (pk >= a - FPS)]
            if len(before):
                p = before[-1]
                around = pk[np.abs(pk - p) <= 5 * FPS]
                rank = 1 + int(np.sum(d[around] > d[p]))
                txt = "{:5.1f} m/s {:4.2f} s  rank {} of {}".format(d[p], (a - p) / FPS, rank, len(around))
                hits.append(rank)
            else:
                txt = "none"
            print("#{:<3} {:<5} | {:>7.1f} {:>7.1f} | {}".format(i + 1, ctx, w1.max(), w5.max(), txt))
        # rule: a peak that is the highest within +-2 s and >= 6 m/s; how many in live play, how many at starts
        loc = pk[(d[pk] >= 6) & (d[pk] == maximum_filter1d(d, 101)[pk]) & live[pk]]
        at_start = [p for p in loc if any(-FPS <= a - p <= 3 for a, b, k in sp)]
        caught = sum(1 for a, b, k in sp if np.any((loc <= a + 3) & (loc >= a - FPS)))
        mins = live.sum() / FPS / 60
        print("rule 'highest within +-2 s and >= 6 m/s': {} peaks in {:.1f} live min ({:.1f}/min); "
              "{} of them just before a start; starts caught {} of {}".format(
                  len(loc), mins, len(loc) / mins, len(at_start), caught, len(sp)))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "local":
    local_cmd()
