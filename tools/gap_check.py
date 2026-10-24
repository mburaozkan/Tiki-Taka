"""Is "ball missing" in the ATD really a dead-ball signal? Aggregate answer only.

Compares the ATD's ball-missing stretches with the public ground truth's dead-ball stretches,
inside the public 5-minute window. Prints rates and offset summaries, never frame numbers.

    python tools/gap_check.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from atd_cache import load_arrays  # noqa: E402
from report import PUBLIC_GT, FPS  # noqa: E402


def spans(mask):
    """[(start, stop)] half-open True runs."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def summary(values):
    """n, median, min, max of signed frame offsets, with seconds."""
    if not len(values):
        return "n=0"
    v = np.asarray(values, dtype=float)
    return "n={}  median {:+.0f} fr ({:+.1f}s)  min {:+.0f}  max {:+.0f}".format(
        len(v), np.median(v), np.median(v) / FPS, v.min(), v.max())


def main():
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    graded = gt != pm.X
    first, last = np.flatnonzero(graded)[[0, -1]]
    sl = slice(first, last + 1)

    ball = load_arrays("tactical")["ball"]
    missing = np.isnan(ball[sl, 0])
    truth = gt[sl]
    dead, alive = truth == pm.D, truth != pm.D

    print("PUBLIC WINDOW  {} frames".format(len(truth)))
    print("  ball missing on GT-dead frames   {:5.1f}%".format(100.0 * missing[dead].mean()))
    print("  ball missing on GT-alive frames  {:5.1f}%".format(100.0 * missing[alive].mean()))
    print("  share of missing frames that are GT-dead  {:5.1f}%".format(
        100.0 * dead[missing].mean() if missing.any() else 0.0))

    dead_spans, gap_spans = spans(dead), spans(missing)
    print("\n  GT dead stretches {}   ball gaps {}".format(len(dead_spans), len(gap_spans)))

    # For each GT dead stretch, the overlapping gap with the largest overlap: how far its edges
    # sit from the truth's edges. Positive = the gap starts/ends later than the dead ball.
    start_off, end_off, unmatched = [], [], 0
    for ds, de in dead_spans:
        best = max(gap_spans, key=lambda g: min(g[1], de) - max(g[0], ds), default=None)
        if best is None or min(best[1], de) <= max(best[0], ds):
            unmatched += 1
            continue
        start_off.append(best[0] - ds)
        end_off.append(best[1] - de)
    print("  dead stretches with no overlapping gap: {}".format(unmatched))
    print("  gap start - dead start   {}".format(summary(start_off)))
    print("  gap end   - dead end     {}".format(summary(end_off)))

    inside_alive = [(a, b) for a, b in gap_spans if not dead[a:b].any()]
    lengths = [(b - a) / float(FPS) for a, b in inside_alive]
    print("  gaps entirely inside live play: {}   lengths (s) median {:.1f} max {:.1f}".format(
        len(inside_alive), np.median(lengths) if lengths else 0, max(lengths) if lengths else 0))


if __name__ == '__main__':
    main()
