"""Why is each true change in the public window detected early or late? Relative timings only.

    python tools/boundary_diag.py evaluation/results/v10.csv [--set key=value ...]

For every true change of label (H>A, A>D, D>H, ...) in the public window, numbered by the true
possession it starts or ends (#k, as in tools/spell_table.py):
  pred      predicted change into the new label, nearest the true one (s, + = late; '-' = none
            within the search window)
  evidence  when the per-frame scores first favour the new label over the old one, held for
            EVIDENCE_HOLD_S (s, relative to the true change)
  touch     first detected touch by the new team (H/A changes only), relative to the true change
  ball      share of the +/-1 s around the true change with the ball seen
  cause     evidence late or missing -> a feature problem; evidence on time but pred late -> the
            decoder (switch penalty) holds on; both on time -> fine
Then counts per cause and per kind of change. No frame numbers, clock times or tracking values.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from report import PUBLIC_GT, FPS  # noqa: E402
from pipeline.features import build_features  # noqa: E402
from pipeline.decode import emissions  # noqa: E402
from pipeline.run import parse_overrides  # noqa: E402

# Params
NAMES = {pm.H: "H", pm.A: "A", pm.D: "D"}
SEARCH_S = 15.0          # look this far either side for the predicted change / evidence
EVIDENCE_HOLD_S = 1.0    # the new label must lead for this long to count as evidence
ON_TIME_S = 1.0          # within this is on time


def first_hold(mask, start, stop, hold):
    """First index in [start, stop) where mask stays True for `hold` frames, or None."""
    run = 0
    for i in range(max(0, start), min(len(mask), stop)):
        run = run + 1 if mask[i] else 0
        if run >= hold:
            return i - hold + 1
    return None


def fmt(x):
    return "    -" if x is None else "{:+5.1f}".format(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submission")
    parser.add_argument("--set", nargs="*", metavar="KEY=VALUE")
    args = parser.parse_args()
    p = parse_overrides(args.set)

    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    pred = pm.label_codes(pm.read_label_csv(args.submission))
    runs = pm.valid_runs(gt)
    segs = pm.segments(gt, runs)
    spells = pm.possession_spells(gt, runs)
    number = {s[0]: i + 1 for i, s in enumerate(spells)}
    number_end = {s[1]: i + 1 for i, s in enumerate(spells)}

    f = build_features()
    scores = emissions(f, p)
    search = int(SEARCH_S * FPS)
    hold = int(EVIDENCE_HOLD_S * FPS)

    rows, causes = [], {}
    for prev, nxt in zip(segs[:-1], segs[1:]):
        if prev[1] != nxt[0]:
            continue
        t, old, new = nxt[0], prev[2], nxt[2]
        lo, hi = t - search, t + search
        # Predicted change into `new`, nearest t.
        changes = [i for i in range(max(1, lo), min(len(pred), hi))
                   if pred[i] == new and pred[i - 1] != new]
        p_off = (min(changes, key=lambda i: abs(i - t)) - t) / FPS if changes else None
        # Evidence: new label's score above the old one's, held.
        # The start of the held run of "new leads old" nearest the true change (a run that began
        # before it counts from where it began).
        lead = scores[:, new] > scores[:, old]
        starts = []
        i = max(1, lo)
        while i < min(len(lead), hi):
            if lead[i] and not lead[i - 1]:
                j = i
                while j < len(lead) and lead[j]:
                    j += 1
                if j - i >= hold:
                    starts.append(i)
                i = j
            else:
                i += 1
        if lead[t] and lead[t - 1]:                       # already leading at the change
            j = t
            while j > 0 and lead[j - 1]:
                j -= 1
            starts.append(j)
        e_idx = min(starts, key=lambda s: abs(s - t)) if starts else None
        e_off = None if e_idx is None else (e_idx - t) / FPS
        touch_off = None
        if new in (pm.H, pm.A):
            side = 1.0 if new == pm.H else -1.0
            hits = np.flatnonzero(f["touch"][max(0, t - 3 * FPS):hi] == side)
            touch_off = None if not len(hits) else (hits[0] + max(0, t - 3 * FPS) - t) / FPS
        ball = np.mean(f["ball_seen"][t - FPS:t + FPS] > 0)

        late_pred = p_off is None or abs(p_off) > ON_TIME_S
        late_ev = e_off is None or abs(e_off) > ON_TIME_S
        if not late_pred:
            cause = "on time"
        elif late_ev:
            cause = "evidence " + ("missing" if e_off is None else
                                   "late" if e_off > 0 else "early")
        else:
            cause = "decoder holds on" if (p_off is None or p_off > 0) else "decoder early"
        causes[cause] = causes.get(cause, 0) + 1
        where = []
        if t in number_end:
            where.append("end #{}".format(number_end[t]))
        if t in number:
            where.append("start #{}".format(number[t]))
        rows.append((" / ".join(where) or "-", "{}>{}".format(NAMES[old], NAMES[new]), p_off,
                     e_off, touch_off, ball, cause))

    print("{:<20} {:<5} {:>6} {:>9} {:>6} {:>5}  {}".format(
        "true change", "kind", "pred", "evidence", "touch", "ball", "cause"))
    for where, kind, p_off, e_off, t_off, ball, cause in rows:
        print("{:<20} {:<5} {:>6} {:>9} {:>6} {:>4.0f}%  {}".format(
            where, kind, fmt(p_off), fmt(e_off), fmt(t_off), 100 * ball, cause))
    print("\ncauses: " + "   ".join("{} {}".format(k, v) for k, v in sorted(causes.items())))
    kinds = {}
    for _, kind, p_off, _, _, _, cause in rows:
        group = "turnover" if "D" not in kind else "stoppage" if kind.endswith("D") else "restart"
        k = kinds.setdefault(group, [0, 0])
        k[0] += 1
        k[1] += cause == "on time"
    print("by kind:  " + "   ".join("{} {}/{} on time".format(g, v[1], v[0])
                                    for g, v in kinds.items()))


if __name__ == '__main__':
    main()
