"""Second-stage inputs and targets for the external matches (SkillCorner / PFF). Aggregate only.

    python tools/ext_inputs.py [--jobs 4]

For every match cached by tools/run_v31_external.py --save (data/cache/ext_v31/v31_<set>_<g>.npz):
players-only columns (in_domain_team.player_columns, as the chain computes them), then at BIN 2
(the deployed refine_v29 / dead_d1n bin) the team inputs (refine.inputs, plain + flipped), the
dead-ball inputs without the first stage's P(dead) (refine.dead_inputs use_pd=False, as d1n),
and the targets (refine.targets / dead_targets, plain + flipped). Written to
data/cache/ext_v31/x_<set>_<g>.npz. Prints counts only.
"""

import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

INTER = os.path.join(ROOT, "data", "cache", "ext_v31")
BIN = 2


def build(path):
    import in_domain_team as idt
    import run_v31_external as rx
    from pipeline import refine
    refine.BIN = BIN
    base = os.path.basename(path)[:-4]                 # v31_<set>_<g>
    _, name, game = base.split("_", 2)
    out = os.path.join(INTER, "x_{}_{}.npz".format(name, game))
    if os.path.isfile(out):
        return 0
    with np.load(path) as z:
        c = {k: z[k] for k in z.files}
    f = {k[2:]: v for k, v in c.items() if k.startswith("f_")}
    d, pitch, _ = rx.load_set(name, game)
    cols = idt.player_columns(d["players"], d["team"], d["period"].astype(np.int8), pitch)
    p1, hand, lab = c["p1"], c["hand"], c["labels"]
    res = {}
    for flip in (False, True):
        s = "_f" if flip else ""
        res["team" + s] = refine.inputs(p1, hand, f, cols, flip)
        res["dead" + s] = refine.dead_inputs(p1, hand, f, cols, flip, False)
        res["ty" + s], res["tv" + s] = refine.targets(lab, flip)
        res["dy" + s], res["dv" + s] = refine.dead_targets(lab, flip)
    np.savez_compressed(out, **res)
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    paths = sorted(glob.glob(os.path.join(INTER, "v31_skillcorner_*.npz"))) + \
        sorted(glob.glob(os.path.join(INTER, "v31_pff_*.npz")))
    with Pool(args.jobs) as pool:
        done = 0
        for i, r in enumerate(pool.imap_unordered(build, paths), start=1):
            done += r
            if i % 10 == 0:
                print("  {} / {}".format(i, len(paths)), flush=True)
    print("built {} new, {} matches total".format(done, len(paths)))


if __name__ == "__main__":
    main()
