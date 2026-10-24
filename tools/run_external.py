"""Run our pipeline on the Metrica matches and score it against their event-derived labels.

    python tools/run_external.py                          # current defaults (PARAMS)
    python tools/run_external.py --name v10 --set w_comove=0
    python tools/run_external.py --name v5 --set switch=20 w_keeper=0 gap_gate=0 w_comove=0 \
        --touch TOUCH_DV_MS=3.0 TOUCH_R_M=2.0

The same features and decoder as pipeline/run.py; no Smart Tagging (Metrica has none). Scored with
the competition's own evaluate() over each full match. Writes data/external/results/<name>_game<g>.csv
and the labels as gt_game<g>.csv, so tools that take --gt can read them.
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
import metrica_data as md  # noqa: E402
import pipeline.features as features  # noqa: E402
from pipeline.decode import emissions, decode_block  # noqa: E402
from pipeline.run import parse_overrides  # noqa: E402
from games import DATA_DIR  # noqa: E402

# Params
OUT = os.path.join(DATA_DIR, "external", "results")


def write(path, codes):
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, md.LETTERS[c]) for i, c in enumerate(codes))


def predict(d, p):
    """Label codes for one game: decode each period separately, X outside."""
    features.LENGTH, features.WIDTH = md.PITCH
    f = features.build_features({"players": d["players"], "ball": d["ball"], "team": d["team"]},
                                events=[])
    scores = emissions(f, p)
    codes = np.full(len(scores), md.X, dtype=np.int8)
    for per in (1, 2):
        idx = np.flatnonzero(d["period"] == per)
        if len(idx):
            a, b = idx[0], idx[-1] + 1
            codes[a:b] = decode_block(scores, f, p, a, b)
    return codes


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="current")
    parser.add_argument("--set", nargs="*", metavar="KEY=VALUE")
    parser.add_argument("--touch", nargs="*", metavar="KEY=VALUE")
    parser.add_argument("--games", nargs="*", type=int, default=list(md.GAMES))
    args = parser.parse_args()
    for pair in args.touch or []:
        key, _, value = pair.partition("=")
        setattr(features, key, float(value))
    p = parse_overrides(args.set)

    os.makedirs(OUT, exist_ok=True)
    rows = []
    for g in args.games:
        d = md.load(g)
        gt = d["labels"]
        pred = predict(d, p)
        write(os.path.join(OUT, "gt_game{}.csv".format(g)), gt)
        write(os.path.join(OUT, "{}_game{}.csv".format(args.name, g)), pred)
        r = pm.evaluate(gt, pred)
        runs = pm.valid_runs(gt)
        filled = pm.fill_abstentions(pred, runs)
        graded = gt != md.X
        dead = 100 * np.mean(filled[graded] == md.D)
        rows.append((r["spell_f1_mean"], r["macro_f1"]))
        print("{:<14} game {}  spellF1 {:.3f}  macroF1 {:.3f}  | poss pred {:4d} / true {:4d}  "
              "D pred {:4.1f}% / true {:4.1f}%".format(
                  args.name, g, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"],
                  r["n_gt_spells"], dead, 100 * np.mean(gt[graded] == md.D)))
    s, m = np.mean(rows, axis=0)
    print("{:<14} MEAN    spellF1 {:.3f}  macroF1 {:.3f}".format(args.name, s, m))


if __name__ == '__main__':
    main()
