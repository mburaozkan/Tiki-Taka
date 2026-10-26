"""Does a model trained on DFL transfer to our match? Aggregate numbers only.

    python tools/transfer_check.py evaluation/results/v21.csv evaluation/results/v23*.csv

Per submission, one line:
  full D      share of graded frames called D over the full match (plausible: 37-50%)
  poss        possessions over the full match (plausible: ~230-300)
  pub D->live share of true public D frames called H or A (v20: low; v21_blend: high)
  pub live->D share of true public live frames called D
  pub seen live->D  the same on frames with the ATD ball seen (the learned model's failure mode)
  pub spF1 / mF1
"""

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from atd_cache import load_arrays  # noqa: E402
from report import PUBLIC_GT, half_runs  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submissions", nargs="+")
    args = parser.parse_args()
    paths = sorted({os.path.normpath(p) for s in args.submissions for p in glob.glob(s)})

    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    runs = pm.valid_runs(gt)
    seen = ~np.isnan(load_arrays("tactical")["ball"][:, 0])
    graded_full = np.zeros(len(gt), bool)
    for _, a, b in half_runs():
        graded_full[a:b] = True
    g = gt != pm.X
    t_dead, t_live = g & (gt == pm.D), g & ((gt == pm.H) | (gt == pm.A))

    print("{:<26} {:>6} {:>5} {:>10} {:>10} {:>15} {:>8} {:>7}".format(
        "submission", "full D", "poss", "pub D>live", "pub live>D", "pub seen live>D",
        "pub spF1", "mF1"))
    for path in paths:
        raw = pm.label_codes(pm.read_label_csv(path))
        full_runs = [(a, b) for _, a, b in half_runs()]
        filled = pm.fill_abstentions(raw, full_runs)
        n_poss = len(pm.possession_spells(filled, full_runs))
        full_d = np.mean(filled[graded_full] == pm.D)
        pred = pm.fill_abstentions(raw, runs)
        live_p = (pred == pm.H) | (pred == pm.A)
        r = pm.evaluate(gt, raw, bootstrap_n=0)
        print("{:<26} {:>5.1f}% {:>5} {:>9.1f}% {:>9.1f}% {:>14.1f}% {:>8.3f} {:>7.3f}".format(
            os.path.splitext(os.path.basename(path))[0][:26], 100 * full_d, n_poss,
            100 * np.mean(live_p[t_dead]), 100 * np.mean(~live_p[t_live]),
            100 * np.mean(~live_p[t_live & seen]), r["spell_f1_mean"], r["macro_f1"]))


if __name__ == '__main__':
    main()
