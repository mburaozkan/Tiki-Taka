"""Is the impostor keeper at one goal the other keeper's slot, swapped across the pitch?

    python tools/keeper_swap.py 9 13 [--pad 5]

User's observation in the viewer (#9-#13, 2026-11-03): Barcelona's keeper ("H keeper") vanishes
during #11 and only comes back after #13, while in the same stretch Barcelona-coloured slots are
ignored (grey crosses) at Athletic's goal. Hypothesis: the tracker hands Barcelona keeper's slot
to Athletic's keeper (an identity swap across a gap), so one swap explains both symptoms.

Window and time axis as tools/possession_view.py (true possessions #first..#last, +/- pad; seconds
from the window start). For each team, aggregates only: the keeper's visible share, every
"keeper lost" stretch and what the slot that was the keeper does next, and every impostor stretch
(keeper_status's ignored slots at that goal) and what that slot did before it. Only seconds,
metres and yes/no are printed: no frames, positions or slot ids.
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
from report import PUBLIC_GT, FPS  # noqa: E402
from pipeline import track_fix as tf  # noqa: E402

NAMES = ("Barcelona", "Athletic")
NEAR_M = 25.0      # "at a goal": within this of the goal centre
LOOK_S = 60.0      # how far to look before / after a stretch for the same slot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("first", type=int)
    ap.add_argument("last", type=int)
    ap.add_argument("--pad", type=float, default=5.0)
    args = ap.parse_args()

    from games import GAME, GAMES, half_video_blocks
    from atd_cache import load_arrays
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    runs = pm.valid_runs(gt)
    spells = pm.possession_spells(gt, runs)
    a = max(spells[args.first - 1][0] - int(args.pad * FPS), 0)
    b = min(spells[args.last - 1][1] + int(args.pad * FPS), len(gt))
    run = next(r for r in runs if r[0] <= spells[args.first - 1][0] < r[1])
    a, b = max(a, run[0]), min(b, run[1])

    raw = load_arrays(raw=True)
    players, team = raw["players"], np.asarray(raw["team"])
    length, width = GAMES[GAME]["pitch"]
    blk = [(x, y + 1) for _, x, y in half_video_blocks(GAME)]
    h0, h1 = next((x, y) for x, y in blk if x <= a < y)
    visible, ignored = tf.keeper_status(players, team, blk, (length, width))

    px = players[:, :, 0].astype(np.float64) * length
    py = players[:, :, 1].astype(np.float64) * width
    tracked = ~np.isnan(px)
    goal = tf._goals(px, tracked, team, blk, length)
    n, k = px.shape
    rows = np.arange(n)
    with np.errstate(invalid="ignore"):
        dist = [np.hypot(px - goal[:, s][:, None], py - width / 2.0) for s in (0, 1)]
        depth = [np.where(tracked, np.abs(px - goal[:, s][:, None]), np.inf) for s in (0, 1)]
    keeper_slot = [np.argmin(np.where((team == s)[None, :], depth[s], np.inf), axis=1)
                   for s in (0, 1)]
    look = int(LOOK_S * FPS)

    def rel(f):
        return (f - a) / float(FPS)

    def piece(i, f):
        """(start, end) of slot i's tracked run containing frame f."""
        s = f
        while s > h0 and tracked[s - 1, i]:
            s -= 1
        e = f
        while e < h1 and tracked[e, i]:
            e += 1
        return s, e

    def is_keeper(i, f, side):
        return bool(visible[f, side]) and keeper_slot[side][f] == i

    print("window #{}-#{}: {:.1f} s (seconds below are from the window start, as in the "
          "viewer)".format(args.first, args.last, (b - a) / float(FPS)))
    for side in (0, 1):
        other = 1 - side
        name, oname = NAMES[side], NAMES[other]
        vis = visible[a:b, side]
        print("\n== {}'s keeper (at {}'s own goal)".format(name, name))
        print("  visible {:.0f}% of the window".format(100 * vis.mean()))

        # 1. every "keeper lost" stretch: what the slot that was the keeper does next
        for s, e in tf._runs(~vis):
            s, e = s + a, e + a
            if s == a:
                continue                          # lost before the window: no keeper to follow
            i = keeper_slot[side][s - 1]
            ps, pe = piece(i, s - 1)
            print("  lost at {:+.1f} s for {:.1f} s:".format(rel(s), (e - s) / float(FPS)))
            if pe > s:
                print("    the slot that was the keeper is still tracked, {:.0f} m from its own "
                      "goal; another slot became the deepest or the line gap closed".format(
                          dist[side][s, i]))
            else:
                print("    the slot that was the keeper stops being tracked {:.1f} s before".format(
                    (s - pe) / float(FPS)))
            nxt = np.flatnonzero(tracked[pe:min(h1, pe + look), i])
            if len(nxt):
                f = pe + nxt[0]
                print("    it reappears {:.1f} s after its track ended, {:.0f} m from {}'s goal, "
                      "{:.0f} m from {}'s goal{}".format(
                          (f - pe) / float(FPS), dist[side][f, i], name, dist[other][f, i], oname,
                          " (the other end of the pitch)" if dist[other][f, i] <= NEAR_M else ""))
                cases = ignored[f:min(h1, f + look), i]
                if cases.any():
                    print("    it is then ignored as an impostor at {}'s goal for {:.1f} s".format(
                        oname, cases.sum() / float(FPS)))
            else:
                print("    it does not reappear within {:.0f} s".format(LOOK_S))
            back = np.flatnonzero(visible[e:min(h1, e + 1), side])
            if e < b and len(back):
                j = keeper_slot[side][e]
                print("    when the keeper is visible again ({:+.1f} s) it is {} slot".format(
                    rel(e), "the same" if j == i else "a different"))

        # 2. every impostor stretch at this side's goal: what that slot did before it
        imp = ignored[a:b] & (team != side)[None, :] & (dist[side][a:b] <= tf.EXTEND_M)
        print("\n== impostors at {}'s goal ({}-coloured slots behind {}'s line)".format(
            name, oname, name))
        if not imp.any():
            print("  none in the window")
        for i in np.flatnonzero(imp.any(axis=0)):
            for s, e in tf._runs(imp[:, i]):
                s, e = s + a, e + a
                ps, pe = piece(i, s)
                print("  impostor from {:+.1f} s for {:.1f} s (its tracked piece started {:.1f} s "
                      "before)".format(rel(s), (e - s) / float(FPS), (s - ps) / float(FPS)))
                prev = np.flatnonzero(tracked[max(h0, ps - look):ps, i])
                if len(prev):
                    f = max(h0, ps - look) + prev[-1]
                    print("    before that the same slot was last seen {:.1f} s earlier, {:.0f} m "
                          "from {}'s goal, {:.0f} m from {}'s goal; {}'s visible keeper then: "
                          "{}".format((ps - f) / float(FPS), dist[side][f, i], name,
                                      dist[other][f, i], oname, oname,
                                      "yes" if is_keeper(i, f, other) else "no"))
                else:
                    print("    the slot is not tracked in the {:.0f} s before".format(LOOK_S))
                with np.errstate(invalid="ignore"):
                    ghost = tf._still(px[s:e], py[s:e], int(tf.STILL_S * FPS))
                own = keeper_slot[side][s:e]
                print("    meanwhile {}'s own deepest slot is still (a ghost?) {:.0f}% of the time; "
                      "{}'s keeper visible {:.0f}% of it".format(
                          name, 100 * np.mean(ghost[np.arange(e - s), own]), oname,
                          100 * visible[s:e, other].mean()))


if __name__ == '__main__':
    main()
