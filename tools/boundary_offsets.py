"""Boundary timing of v31 on the external sets: are our changes early or late? Aggregate only.

    python tools/boundary_offsets.py                  # offsets per change kind, SkillCorner + PFF
    python tools/boundary_offsets.py --shift          # + spellF1 with each kind shifted by a constant
    python tools/boundary_offsets.py --public         # also the public window (submission file)

Reads the cached intermediates of tools/run_v31_external.py --save (data/cache/ext_v31/), so no
features are rebuilt. Change kinds (the first frame of the new label):
  team  = H -> A or A -> H (direct change of possession)
  start = D -> H / A (restart)
  stop  = H / A -> D (stoppage)
For every true change, the nearest predicted change of the same kind (same from / to labels)
within +/- WINDOW_S; offset = predicted - true, in seconds (> 0: we are late). Prints per set and
kind: n, share matched, median, mean, quartiles, share within 0.2 / 0.5 / 1 s.
--shift: every predicted change of one kind is moved by a constant (kept inside its neighbours),
then spellF1 is recomputed per match; mean delta vs v31 and matches up / down, per set.
"""

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from pipeline.decode import H, A, D, X  # noqa: E402

# Params
FPS = 25
WINDOW_S = 5.0
SHIFTS_S = (-1.0, -0.6, -0.4, -0.2, -0.1, 0.1, 0.2, 0.4)
INTER = os.path.join(ROOT, "data", "cache", "ext_v31")
KINDS = ("team", "start", "stop")


def kind(a, b):
    if a in (H, A) and b in (H, A):
        return "team"
    if a == D and b in (H, A):
        return "start"
    if a in (H, A) and b == D:
        return "stop"
    return None


def changes(codes, runs):
    """{kind: [(frame, from, to)]} of codes (X filled as the scorer does)."""
    out = {k: [] for k in KINDS}
    for fr, a, b in pm.transitions(pm.fill_abstentions(codes, runs), runs):
        k = kind(a, b)
        if k:
            out[k].append((fr, a, b))
    return out


def offsets(lab, pred):
    """{kind: (offsets in s of matched true changes, n true changes)}."""
    runs = pm.valid_runs(lab)
    t, p = changes(lab, runs), changes(pred, runs)
    out = {}
    for k in KINDS:
        offs = []
        for fr, a, b in t[k]:
            cand = [q - fr for q, c, e in p[k] if c == a and e == b and abs(q - fr) <= WINDOW_S * FPS]
            if cand:
                offs.append(min(cand, key=abs) / FPS)
        out[k] = (offs, len(t[k]))
    return out


def shift(codes, runs, k, frames):
    """codes with every change of kind k moved by frames (kept strictly inside its neighbours)."""
    out = pm.fill_abstentions(codes, runs)
    res = out.copy()
    for start, stop in runs:
        seg = pm.segments(out, [(start, stop)])
        bounds = [s[0] for s in seg[1:]]
        new = list(bounds)
        for i, b in enumerate(bounds):
            if kind(seg[i][2], seg[i + 1][2]) != k:
                continue
            lo = (new[i - 1] if i else start) + 1
            hi = (bounds[i + 1] if i + 1 < len(bounds) else stop) - 1
            new[i] = int(min(max(b + frames, lo), hi))
        edges = [start] + new + [stop]
        for i, s in enumerate(seg):
            res[edges[i]:edges[i + 1]] = s[2]
    return res


def summary(name, offs, n):
    o = np.asarray(offs)
    if not len(o):
        return "{:<22} n {:>6}  matched 0".format(name, n)
    q1, med, q3 = np.percentile(o, (25, 50, 75))
    return ("{:<22} n {:>6}  matched {:>4.0%}  median {:+.2f}  mean {:+.2f}  IQR [{:+.2f}, {:+.2f}]"
            "  |off| <= 0.2 {:>4.0%}  0.5 {:>4.0%}  1 {:>4.0%}".format(
                name, n, len(o) / max(n, 1), med, o.mean(), q1, q3, np.mean(np.abs(o) <= 0.2),
                np.mean(np.abs(o) <= 0.5), np.mean(np.abs(o) <= 1.0)))


def load_set(name):
    for path in sorted(glob.glob(os.path.join(INTER, "v31_{}_*.npz".format(name)))):
        with np.load(path) as c:
            yield c["labels"], c["v31"]


def score(lab, codes):
    return pm.evaluate(lab, codes, bootstrap_n=0)["spell_f1_mean"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shift", action="store_true")
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--sub", default=os.path.join(ROOT, "evaluation", "results", "v31_kt.csv"))
    args = parser.parse_args()
    for name in ("skillcorner", "pff"):
        acc = {k: ([], 0) for k in KINDS}
        data = list(load_set(name))
        for lab, pred in data:
            for k, (o, n) in offsets(lab, pred).items():
                acc[k] = (acc[k][0] + o, acc[k][1] + n)
        print("{} ({} matches)".format(name, len(data)))
        for k in KINDS:
            print("  " + summary(k, *acc[k]))
        if args.shift:
            base = np.array([score(lab, pred) for lab, pred in data])
            print("  v31 spellF1 {:.4f}".format(base.mean()))
            for k in KINDS:
                for s in SHIFTS_S:
                    fr = int(round(s * FPS))
                    sc = np.array([score(lab, shift(pred, pm.valid_runs(lab), k, fr))
                                   for lab, pred in data])
                    d = sc - base
                    print("  shift {:<5} {:+.1f} s: {:.4f} ({:+.4f}, {}/{})".format(
                        k, s, sc.mean(), d.mean(), int(np.sum(d > 1e-9)), int(np.sum(d < -1e-9))),
                        flush=True)
    if args.public:
        from report import PUBLIC_GT
        gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
        sub = pm.label_codes(pm.read_label_csv(args.sub))
        print("public window ({})".format(os.path.basename(args.sub)))
        for k, (o, n) in offsets(gt, sub).items():
            print("  " + summary(k, o, n))


if __name__ == "__main__":
    main()
