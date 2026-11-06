"""Dead-ball errors of v31 on SkillCorner / PFF (real broadcast stoppages). Aggregate only.

    python tools/dead_external.py              # SkillCorner + PFF, from data/cache/ext_v31
    python tools/dead_external.py --public     # also the public window (submission file)

As tools/dead_overcall.py (DFL), but on the cached v31 predictions of tools/run_v31_external.py:
true-live frames called D are split by the called D stretch they sit in: early (before the
true stoppage it overlaps), late (after it: restart called late), phantom (overlaps no true
stoppage). The other way: true D called live, split into edge (the true stoppage overlaps some
called D: we start / end it at a different time) and missed (a true stoppage >= MIN_STOP_S with
no called D at all). Oracle = spellF1 with the frames of one kind set to the truth (upper bound
of a fix). Early / late / edge depend on where a provider puts the D edges (its OUT tag to the
restart touch); phantom and missed don't, so those are the comparable ones.
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
from dead_overcall import classify, runs  # noqa: E402
from pipeline.decode import H, A, D, X  # noqa: E402

# Params
FPS = 25
MIN_STOP_S = 2.0
INTER = os.path.join(ROOT, "data", "cache", "ext_v31")
KINDS = ("early", "late", "phantom", "edge", "missed")


def under(lab, pred):
    """Per frame: 0 none, 4 edge, 5 missed (true D called live); missed stoppages, true stoppages."""
    kind = np.zeros(len(lab), np.int8)
    n_stop, n_missed = 0, 0
    for a, b in runs(lab == D):
        live = np.isin(pred[a:b], (H, A))
        long_ = (b - a) / FPS >= MIN_STOP_S
        n_stop += long_
        if not live.any():
            continue
        if (pred[a:b] == D).any():
            kind[a:b][live] = 4
        elif long_:
            kind[a:b][live] = 5
            n_missed += 1
        else:
            kind[a:b][live] = 4
    return kind, n_stop, n_missed


def score(lab, codes):
    return pm.evaluate(lab, codes, bootstrap_n=0)["spell_f1_mean"]


def analyse(pairs, oracle=True):
    st = {"live": 0, "dead": 0, "min": 0.0, "n": np.zeros(6), "lens": [], "stops": 0, "missed": 0,
          "score": [], "oracle": {k: [] for k in KINDS}, "pd": 0, "td": 0, "graded": 0}
    for lab, pred in pairs:
        pred = pm.fill_abstentions(pred, pm.valid_runs(lab))
        over, lens = classify(lab, pred)
        und, n_stop, n_missed = under(lab, pred)
        kind = np.where(und > 0, und, over)
        g = lab != X
        st["live"] += int(np.sum(np.isin(lab, (H, A))))
        st["dead"] += int(np.sum(lab == D))
        st["graded"] += int(g.sum())
        st["pd"] += int(np.sum(g & (pred == D)))
        st["min"] += g.sum() / FPS / 60
        st["n"] += np.bincount(kind, minlength=6)
        st["lens"] += lens
        st["stops"] += n_stop
        st["missed"] += n_missed
        if oracle:
            st["score"].append(score(lab, pred))
            for k, name in enumerate(KINDS, start=1):
                st["oracle"][name].append(score(lab, np.where(kind == k, lab, pred).astype(np.int8)))
    return st


def report(name, st):
    print("{}: graded D truth {:.1f}%, called {:.1f}%".format(
        name, 100 * st["dead"] / st["graded"], 100 * st["pd"] / st["graded"]))
    base = np.mean(st["score"]) if st["score"] else None
    for k, kname in enumerate(KINDS, start=1):
        denom = st["live"] if k <= 3 else st["dead"]
        line = "  {:<8} {:5.1f}% of true {} frames".format(kname, 100 * st["n"][k] / max(denom, 1),
                                                            "live" if k <= 3 else "D")
        if base is not None:
            o = np.mean(st["oracle"][kname])
            line += " | oracle {:.4f} ({:+.4f})".format(o, o - base)
        print(line)
    lens = np.array(st["lens"])
    per90 = 90 / max(st["min"], 1e-9)
    if len(lens):
        print("  phantom stretches {:.1f} per 90 min, median {:.1f} s".format(len(lens) * per90,
                                                                           np.median(lens)))
    print("  true stoppages >= {:g} s: {:.1f} per 90 min, missed entirely {} ({:.1f}%)".format(
        MIN_STOP_S, st["stops"] * per90, st["missed"], 100 * st["missed"] / max(st["stops"], 1)))
    if base is not None:
        print("  v31 spellF1 {:.4f}".format(base))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--sub", default=os.path.join(ROOT, "evaluation", "results", "v31_kt.csv"))
    args = parser.parse_args()
    for name in ("skillcorner", "pff"):
        pairs = []
        for path in sorted(glob.glob(os.path.join(INTER, "v31_{}_*.npz".format(name)))):
            with np.load(path) as c:
                pairs.append((c["labels"], c["v31"]))
        report("{} ({} matches)".format(name, len(pairs)), analyse(pairs))
    if args.public:
        from report import PUBLIC_GT
        gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
        sub = pm.label_codes(pm.read_label_csv(args.sub))
        report("public window ({})".format(os.path.basename(args.sub)), analyse([(gt, sub)]))


if __name__ == "__main__":
    main()
