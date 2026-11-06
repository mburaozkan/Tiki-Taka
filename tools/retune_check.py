"""Check re-tuned chain settings (tools/retune_chain.py knobs) on DFL and our match. Aggregate only.

    python tools/retune_check.py "vote_w=0" "switch=24 vote_w=1"      # v31 is always row one
    python tools/retune_check.py --write v32 "vote_w=0"               # also writes <name>.csv

DFL: the 7 matches with held-out inputs (as tools/eval_dead_v30.py: v25 inputs, refine_b2,
dead_d1n, add-only head, ball-out protected gate), v31's chain with the given knobs; mean
spellF1, delta vs v31, matches up / down. Our match: v31's chain rebuilt in process
(tools/ours_chain.py; set MTAH_VARIANT=_kt for the submission's tracking), then the knobs:
public spellF1 / macroF1, possessions, D share, frames that differ from the reference.
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
import dfl_data as dd  # noqa: E402
import eval_touch_run as tr  # noqa: E402
import eval_dead_v30 as ev  # noqa: E402
import retune_chain as rc  # noqa: E402


def dfl_contexts():
    from eval_mbr import v25_inputs
    from eval_refine import stage1
    from eval_dead_refine import constrain
    from train_learned_dfl import match_features, test_seed, blocks
    from pipeline import refine
    data = v25_inputs()
    with np.load(os.path.join(dd.CACHE, "refine_{}.npz".format(tr.TEAM_TAG))) as c:
        team = {k: c[k] for k in c.files}
    with np.load(os.path.join(dd.CACHE, "dead_{}.npz".format(ev.DEAD_TAG))) as c:
        pdead = {g: np.mean([c["{}__{}".format(g, s)] for s in range(ev.DEAD_SEEDS)], axis=0)
                 for g in dd.GAMES}
    out = []
    for g in dd.GAMES:
        d = dd.load(g)
        f = match_features(g, test_seed(g))
        blk = blocks(d["period"])
        lab, hand, prob = data[g]
        p1 = stage1(prob, f)
        p_ref = np.mean([team["{}__{}".format(g, s)] for s in range(tr.TEAM_SEEDS)], axis=0)
        h2 = constrain(hand, refine.decode_dead(hand, pdead[g], blk, *ev.SETTING), ev.MODE)
        protect = np.asarray(f["ball_out"]) > 0
        h2 = ev.remove_gate(h2, pdead[g], blk, rc.V31["gate_t"], 0.0, dd.FPS, protect)
        c = {"p1": p1, "p_ref": p_ref, "vote": tr.run_vote(f, ev.VOTE), "hand2": h2, "hand": hand,
             "pdead": pdead[g]}
        out.append((lab, c, f, blk))
        print("  DFL", g, flush=True)
    return out


def run(c, f, blk, s):
    h = None
    if (s["gate_t"], s["gate_max_s"]) != (rc.V31["gate_t"], rc.V31["gate_max_s"]):
        h = rc.gated_hand(c, f, blk, s["gate_t"], s["gate_max_s"])
    return rc.chain(c, f, blk, s, h)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("settings", nargs="+")
    parser.add_argument("--write", help="write the LAST setting on our match as <name>.csv")
    parser.add_argument("--skip-dfl", action="store_true")
    args = parser.parse_args()
    settings = [("v31", dict(rc.V31))] + [(t, rc.parse(t)) for t in args.settings]
    if not args.skip_dfl:
        ctx = dfl_contexts()
        res = np.array([[tr.score(lab, run(c, f, blk, s)) for lab, c, f, blk in ctx]
                        for _, s in settings])
        for (name, _), r in zip(settings, res):
            d = r - res[0]
            print("DFL {:<40} {} | {:.4f} | {:+.4f} | {}/{}".format(
                name, " ".join("{:.3f}".format(x) for x in r), r.mean(), d.mean(),
                int(np.sum(d > 1e-9)), int(np.sum(d < -1e-9))), flush=True)
    import ours_chain as oc
    from report import brief_line, PUBLIC_GT
    _, ex = oc.ours_v31()
    f = {k[2:]: v for k, v in ex.items() if k.startswith("f_")}
    blk = [tuple(int(x) for x in b) for b in ex["blk"]]
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    inside = np.zeros(len(gt), bool)
    for a, b in blk:
        inside[a:b] = True
    ref = None
    codes = None
    for name, s in settings:
        codes = run(ex, f, blk, s)
        if ref is None:
            ref = codes
        print("ours " + brief_line(name[:28], codes, gt) + "  differ {:.2f}%".format(
            100 * np.mean(codes[inside] != ref[inside])), flush=True)
    if args.write:
        from games import GAME, GAMES
        from pipeline.run import write_submission, RESULTS_DIR
        path = write_submission(os.path.join(RESULTS_DIR, args.write + ".csv"), codes)
        reason = pm.validate_submission(pm.read_label_csv(path), GAMES[GAME]["n_video_frames"])
        print("wrote {} ({})".format(os.path.relpath(path, ROOT), reason or "valid"))


if __name__ == "__main__":
    main()
