"""Where do two submissions differ over the full match? For a local video check.

    python tools/diff_spots.py evaluation/results/v31.csv evaluation/results/v31_kt.csv

Prints a summary line (counts and seconds: safe to share) and then, for every stretch of at least
MIN_S seconds where the labels differ, its length, A's and B's labels, whether it is inside the
public window, and the render_labels.py command that cuts a clip around it. The commands carry
video frame numbers: run them locally, do not paste them anywhere.
"""

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

MIN_S = 1.0
PAD_S = 8.0
NAME = {pm.H: "H", pm.A: "A", pm.D: "D", pm.X: "X"}


def main():
    path_a, path_b = sys.argv[1], sys.argv[2]
    fa = pm.read_label_csv(path_a)
    frames = fa["frame"].values
    la = pm.label_codes(fa)
    lb = pm.label_codes(pm.read_label_csv(path_b))
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    pub = np.zeros(len(gt), dtype=bool)
    for s, e in pm.valid_runs(gt):
        pub[s:e] = True
    diff = la != lb
    edges = np.diff(np.concatenate(([0], diff.astype(np.int8), [0])))
    runs = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    long_runs = [(s, e) for s, e in runs if e - s >= MIN_S * FPS]
    print("{} vs {}: {:.1f} s differ in {} stretches; {} of at least {:.0f} s ({:.1f} s), "
          "{} of them in the public window".format(
              os.path.basename(path_a), os.path.basename(path_b), diff.sum() / FPS, len(runs),
              len(long_runs), MIN_S, sum(e - s for s, e in long_runs) / FPS,
              sum(pub[s:e].any() for s, e in long_runs)))
    print("\nlocal only, do not paste below this line:")
    for s, e in long_runs:
        def share(lab):
            vals, counts = np.unique(lab[s:e], return_counts=True)
            return "/".join("{} {:.0f}%".format(NAME.get(v, "?"), 100.0 * c / (e - s))
                            for v, c in zip(vals, counts))
        start = max(s - int(PAD_S * FPS), 0)
        frame = int(frames[start])
        print("  {:5.1f} s  A: {:12s} B: {:12s}{}  ->  python tools/render_labels.py {} --compare {} "
              "--start {} --minutes {:.2f}".format(
                  (e - s) / FPS, share(la), share(lb), "  (public)" if pub[s:e].any() else "",
                  path_a, path_b, frame, (e - s) / FPS / 60.0 + 2 * PAD_S / 60.0))


if __name__ == '__main__':
    main()
