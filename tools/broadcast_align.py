"""Align the broadcast ATD to the tactical timeline by player formations. Aggregate output only.

    python tools/broadcast_align.py          # align, cache, print the checks

The two recordings are not frame-aligned (cuts, replays, its own frame count), but both ATDs give
positions on the same 0-1 pitch. Score of a (broadcast frame, tactical frame) pair: for every
player the broadcast sees, the distance (m) to the nearest tactical player of the same team; the
median over them (at least MIN_PLAYERS). Then, per usable broadcast stretch (MIN_TRACKED players
and a valid homography):
  1. search the whole tactical match (stride COARSE) for the stretch's middle frame, refine to the
     frame: the stretch's offset;
  2. every frame of the stretch at that offset, +/- REFINE frames: accepted if the score is under
     ACCEPT_M (slow-motion replays drift off and drop out).
If several broadcast frames land on one tactical frame (a replay of a matched moment), the best
score wins. Cached in data/cache/broadcast_align.npz: for every tactical frame, the broadcast
frame matched to it (-1 none), the score, and the broadcast ball (x, y normalised).

Checks printed: orientation test; share of the broadcast aligned; where both recordings have the
ball, how far apart (m); how many tactical ball gaps (inside the halves) the broadcast fills.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, DATA_DIR, half_video_blocks  # noqa: E402
from atd_cache import load_arrays  # noqa: E402

# Params
OUT = os.path.join(DATA_DIR, "cache", "broadcast_align.npz")
MIN_PLAYERS = 6
MIN_TRACKED = 10
COARSE = 5
REFINE = 2
ACCEPT_M = 1.0
SEARCH_ACCEPT_M = 1.5       # a stretch's middle frame must match at least this well
CHUNK = 4000


def team_points(players, team, frame, side, L, W):
    p = players[frame, team == side, :2].astype(np.float64)
    p = p[~np.isnan(p[:, 0])]
    return p * [L, W]


def scores_against(bpts, tac, tteam, frames, L, W):
    """Median nearest-same-team distance (m) of broadcast points to each tactical frame."""
    out = np.full(len(frames), np.inf)
    for c in range(0, len(frames), CHUNK):
        fr = frames[c:c + CHUNK]
        ds = []
        for side in (0, 1):
            b = bpts[side]
            if not len(b):
                continue
            t = tac[fr][:, tteam == side, :2].astype(np.float64) * [L, W]     # (k, s, 2)
            d = np.hypot(t[:, None, :, 0] - b[None, :, None, 0], t[:, None, :, 1] - b[None, :, None, 1])
            d = np.where(np.isnan(d), np.inf, d).min(axis=2)                    # (k, m)
            ds.append(d)
        if ds:
            out[c:c + CHUNK] = np.median(np.concatenate(ds, axis=1), axis=1)
    return out


def align(variant=(False, False, False), stretches_limit=None, verbose=True):
    L, W = GAMES[GAME]["pitch"]
    tac, bro = load_arrays("tactical"), load_arrays("broadcast")
    from load_homography import load
    _, _, bvalid = load(source="broadcast")
    tp, tt = tac["players"], tac["team"]
    bp, bt, bball = bro["players"].copy(), bro["team"].copy(), bro["ball"].copy()
    flip_x, flip_y, swap = variant
    if flip_x:
        bp[:, :, 0], bball[:, 0] = 1 - bp[:, :, 0], 1 - bball[:, 0]
    if flip_y:
        bp[:, :, 1], bball[:, 1] = 1 - bp[:, :, 1], 1 - bball[:, 1]
    if swap:
        bt = 1 - bt
    nb, nt = len(bp), len(tp)
    tracked = (~np.isnan(bp[:, :, 0])).sum(1)
    usable = (tracked >= MIN_TRACKED) & (bvalid[:nb] > 0)
    edges = np.diff(np.concatenate(([0], usable.astype(np.int8), [0])))
    stretches = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    if stretches_limit:
        stretches = stretches[::max(1, len(stretches) // stretches_limit)][:stretches_limit]
    coarse = np.arange(0, nt, COARSE)
    match = np.full(nt, -1, dtype=np.int64)
    score = np.full(nt, np.inf)
    mids = []
    for a, b in stretches:
        mid = (a + b) // 2
        pts = [team_points(bp, bt, mid, s, L, W) for s in (0, 1)]
        if len(pts[0]) + len(pts[1]) < MIN_PLAYERS:
            continue
        sc = scores_against(pts, tp, tt, coarse, L, W)
        j = coarse[int(np.argmin(sc))]
        fine = np.arange(max(0, j - COARSE), min(nt, j + COARSE + 1))
        sf = scores_against(pts, tp, tt, fine, L, W)
        j = fine[int(np.argmin(sf))]
        mids.append(sf.min())
        if sf.min() > SEARCH_ACCEPT_M:
            continue
        off = j - mid
        for i in range(a, b):
            pts_i = [team_points(bp, bt, i, s, L, W) for s in (0, 1)]
            if len(pts_i[0]) + len(pts_i[1]) < MIN_PLAYERS:
                continue
            cand = np.arange(i + off - REFINE, i + off + REFINE + 1)
            cand = cand[(cand >= 0) & (cand < nt)]
            if not len(cand):
                continue
            s_i = scores_against(pts_i, tp, tt, cand, L, W)
            k = int(np.argmin(s_i))
            if s_i[k] < ACCEPT_M and s_i[k] < score[cand[k]]:
                match[cand[k]], score[cand[k]] = i, s_i[k]
    return match, score, bball, np.array(mids), len(stretches)


def main():
    L, W = GAMES[GAME]["pitch"]
    print("orientation test (12 stretches each): median middle-frame score (m)")
    best = None
    for variant in [(False, False, False), (True, True, False), (False, False, True),
                    (True, True, True), (True, False, False), (False, True, False)]:
        _, _, _, mids, _ = align(variant, stretches_limit=12, verbose=False)
        med = np.median(mids) if len(mids) else np.inf
        print("  flip x {:d} flip y {:d} swap teams {:d}: {:.2f} m".format(*variant, med), flush=True)
        if best is None or med < best[0]:
            best = (med, variant)
    variant = best[1]
    print("using flip x {:d} flip y {:d} swap teams {:d}".format(*variant), flush=True)
    match, score, bball, mids, n_str = align(variant)
    np.savez(OUT, match=match, score=score, ball_x=np.where(match >= 0, bball[np.maximum(match, 0), 0], np.nan),
             ball_y=np.where(match >= 0, bball[np.maximum(match, 0), 1], np.nan), variant=np.array(variant))

    tac = load_arrays("tactical")
    tball = tac["ball"]
    in_half = np.zeros(len(tball), bool)
    for _, first, last in half_video_blocks(GAME):
        in_half[first:last + 1] = True
    matched = match >= 0
    bx = np.where(matched, bball[np.maximum(match, 0), 0], np.nan)
    by = np.where(matched, bball[np.maximum(match, 0), 1], np.nan)
    b_seen = ~np.isnan(bx)
    t_seen = ~np.isnan(tball[:, 0])
    both = matched & b_seen & t_seen & in_half
    dist = np.hypot((bx - tball[:, 0]) * L, (by - tball[:, 1]) * W)[both]
    print("\nstretches: {} | middle frames matched under {} m: {:.0f}%".format(
        n_str, SEARCH_ACCEPT_M, 100 * np.mean(mids <= SEARCH_ACCEPT_M)))
    print("tactical frames (in the halves) with an aligned broadcast frame: {:.0f}% | median "
          "formation score {:.2f} m".format(100 * np.mean(matched[in_half]),
                                           np.median(score[matched & in_half])))
    print("both recordings have the ball on {:.0f}% of aligned frames: ball distance median "
          "{:.2f} m, p75 {:.2f} m, p90 {:.2f} m, within 1 m {:.0f}%".format(
              100 * both.sum() / max((matched & in_half).sum(), 1), np.median(dist),
              *np.percentile(dist, [75, 90]), 100 * np.mean(dist <= 1)))
    gap = in_half & ~t_seen
    fill = gap & matched & b_seen
    print("tactical ball gaps inside the halves: {:.0f} s | aligned there: {:.0f}% | broadcast "
          "has the ball there: {:.0f}% of all gap time".format(
              gap.sum() / 25, 100 * np.mean(matched[gap]), 100 * fill.sum() / gap.sum()))


if __name__ == '__main__':
    main()
