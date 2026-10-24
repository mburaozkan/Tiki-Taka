"""How do the Smart Tagging intervals line up with the possession labels? Aggregate answer only.

    python tools/tag_alignment.py [submission.csv ...]

Per code: how many intervals, their median length, and inside the public window the share of true
H/A/D frames and how often the code's team reading (attacking = tagged team has it, defending =
the other one) agrees with the true team. Then, for SET PIECES, the true label mix by position
relative to the interval (before / start / middle / end / after), whether the tagged team is the
one that gets the ball after it, and the same position profile for any submission over the full
match. Shares and counts only; no frame numbers, no clock times.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "starter"))
import possession_metrics as pm  # noqa: E402
from report import PUBLIC_GT, FPS  # noqa: E402
from pipeline.features import TAG_SIGN, SETPIECE_CODE  # noqa: E402

NAMES = {pm.H: "H", pm.A: "A", pm.D: "D"}
# Position bins around a set-piece interval: (name, where) with where in interval-relative terms.
BINS = [("-10..-3s", ("before", -10, -3)), ("-3..0s", ("before", -3, 0)),
        ("first 25%", ("inside", 0.0, 0.25)), ("25-75%", ("inside", 0.25, 0.75)),
        ("last 25%", ("inside", 0.75, 1.0)), ("+0..3s", ("after", 0, 3)),
        ("+3..10s", ("after", 3, 10))]


def tag_team(event):
    """+1 Barcelona, -1 Athletic, 0 N/A or missing."""
    team = (event["tags"].get("Team") or "").strip().lower()
    return 1 if "barcelona" in team else -1 if ("atletic" in team or "athletic" in team) else 0


def mix(codes):
    """'H 10% A 5% D 85% (n s)' over the given labels (X ignored)."""
    codes = codes[codes != pm.X]
    if not len(codes):
        return "{:<24}".format("-")
    return "{:<24}".format(" ".join("{} {:3.0f}%".format(NAMES[k], 100 * np.mean(codes == k))
                                    for k in NAMES) + " ({:.0f}s)".format(len(codes) / FPS))


def bin_frames(event, where, n):
    kind, lo, hi = where
    a, b = event["start_frame"], event["end_frame"]
    if kind == "before":
        r = (a + int(lo * FPS), a + int(hi * FPS))
    elif kind == "after":
        r = (b + int(lo * FPS), b + int(hi * FPS))
    else:
        r = (a + int(lo * (b - a)), a + int(hi * (b - a)))
    return max(0, r[0]), min(n, r[1])


def profile(events, labels, keep=lambda e: True):
    """Label mix per position bin, pooled over the set-piece intervals passing keep()."""
    n = len(labels)
    for name, where in BINS:
        parts = [labels[slice(*bin_frames(e, where, n))] for e in events if keep(e)]
        pooled = np.concatenate(parts) if parts else np.array([], dtype=labels.dtype)
        yield name, mix(pooled)


def main():
    from load_smart_tagging import load
    events = load()
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    graded = np.flatnonzero(gt != pm.X)
    w0, w1 = graded[0], graded[-1] + 1
    in_window = [e for e in events if e["start_frame"] < w1 and w0 < e["end_frame"]]

    print("PER CODE   n (full) / median length / in public window: truth mix, team agreement")
    for code in sorted({(e["code"] or "").strip().upper() for e in events}):
        ev = [e for e in events if (e["code"] or "").strip().upper() == code]
        lengths = [(e["end_frame"] - e["start_frame"]) / FPS for e in ev]
        win = [e for e in in_window if (e["code"] or "").strip().upper() == code]
        frames = np.concatenate([gt[max(e["start_frame"], w0):min(e["end_frame"], w1)]
                                 for e in win]) if win else np.array([], dtype=gt.dtype)
        agree = ""
        sign = TAG_SIGN.get(code.lower(), 0)
        if sign and win:
            hit = tot = 0
            for e in win:
                team = tag_team(e) * sign
                seg = gt[max(e["start_frame"], w0):min(e["end_frame"], w1)]
                alive = seg[(seg == pm.H) | (seg == pm.A)]
                if team:
                    hit += np.sum(alive == (pm.H if team > 0 else pm.A))
                    tot += len(alive)
            agree = "team right {:3.0f}% of alive".format(100.0 * hit / tot) if tot else ""
        print("  {:<30} {:3d}  {:5.1f}s   win {:2d}  {}  {}".format(
            code, len(ev), np.median(lengths), len(win), mix(frames), agree))

    sp = [e for e in events if (e["code"] or "").strip().lower() == SETPIECE_CODE]
    sp_win = [e for e in sp if w0 <= e["start_frame"] and e["end_frame"] <= w1]
    print("\nSET PIECES: {} in the match, {} fully inside the public window".format(
        len(sp), len(sp_win)))
    types = {}
    for e in sp:
        key = (e["tags"].get("Type") or "-", tag_team(e) != 0)
        types[key] = types.get(key, 0) + 1
    print("  by Type (team tagged?): " + "  ".join(
        "{} {}: {}".format(t, "team" if has else "N/A", c) for (t, has), c in sorted(types.items())))

    # What follows a set piece in the tags: the next phase with a team reading, and its delay.
    phase = [e for e in events if TAG_SIGN.get((e["code"] or "").strip().lower(), 0)]
    delays, same = [], []
    for e in sp:
        nxt = [p for p in phase if p["start_frame"] >= e["start_frame"] - FPS]
        if not nxt:
            continue
        p = min(nxt, key=lambda q: q["start_frame"])
        delays.append((p["start_frame"] - e["end_frame"]) / FPS)
        if tag_team(e) and tag_team(p):
            same.append(tag_team(e) == tag_team(p) * TAG_SIGN[(p["code"] or "").strip().lower()])
    q = np.percentile(delays, (10, 50, 90))
    print("  next team phase starts (s after set-piece END): p10 {:+.1f}  median {:+.1f}  "
          "p90 {:+.1f}".format(*q))
    if same:
        print("  that phase gives the ball to the set-piece team: {}/{}".format(sum(same),
                                                                                len(same)))

    print("\n  truth mix around the set pieces in the public window:")
    for name, text in profile(sp_win, gt):
        print("    {:<10} {}".format(name, text))

    if sp_win:
        hit = tot = 0
        for e in sp_win:
            team = tag_team(e)
            after = gt[e["start_frame"]:min(len(gt), e["end_frame"] + 10 * FPS)]
            alive = after[(after == pm.H) | (after == pm.A)]
            if team and len(alive):
                tot += 1
                hit += alive[0] == (pm.H if team > 0 else pm.A)
        print("  first possession during/after it belongs to the tagged team: {}/{}".format(
            hit, tot))

    for path in sys.argv[1:]:
        pred = pm.label_codes(pm.read_label_csv(path))
        print("\n  {} around ALL {} set pieces (full match, prediction not truth):".format(
            os.path.basename(path), len(sp)))
        for name, text in profile(sp, pred):
            print("    {:<10} {}".format(name, text))


if __name__ == '__main__':
    main()
