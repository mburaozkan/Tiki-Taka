"""v31's chain on our match in one process, with a swappable touch network. Aggregate only.

    python tools/ours_chain.py                       # rebuild v31, check it equals v31.csv
    python tools/ours_chain.py --touch PATH --name N # v31 with another touch network -> N.csv
    python tools/ours_chain.py --dead models/dead_ext_d1 --name v31_d1   # the working base (with
                                                     # MTAH_VARIANT=_kt -> v31_d1_kt.csv)

Same steps as the committed rebuild (tools/in_domain_team.py --ours --source dfl+decoded
--weights 0.5, tools/tune_refine.py --ours, tools/eval_dead_v30.py --ours 0.5 0.5 25 --gate 0.1 0
--protect), without writing their caches or result files: features with the touch network,
hand path, gbm_dfl + gru_dfl blend, models/po_team_v25 players-only vote (w 0.5), stage 1,
refine_v29 x5, touch-run vote, dead head d1n x3, protected remove gate, backdating.
ours_v31() also returns the intermediates (for the short-possession head).
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
import in_domain_team as idt  # noqa: E402
import eval_touch_run as tr  # noqa: E402
import eval_dead_v30 as ev  # noqa: E402
from eval_refine import stage1  # noqa: E402
from eval_dead_refine import constrain  # noqa: E402
from pipeline import learned, refine, touch_net  # noqa: E402

GATE = (0.1, 0.0)


class Args:
    protect, hold = True, 0.0


def ours_v31(touch_path=None, dead_prefix=None):
    """(codes, extras) of v31 on our match with the given touch network (default: deployed).
    dead_prefix: dead-ball head models <prefix>_s*.pt (no P(dead) input, e.g. models/dead_ext_d1,
    tools/train_ext_heads.py) in place of dead_d1n."""
    import lightgbm as lgb
    from games import GAME, half_video_blocks
    from atd_cache import load_arrays
    from pipeline.features import build_features, LENGTH, WIDTH
    from pipeline.sequence import predict_gru
    from pipeline.decode import decode
    from pipeline import run_learned as rl
    tm = touch_net.load(touch_path or os.path.join(ROOT, touch_net.MODEL_PATH))
    f = build_features(events=[], touch_model=tm)
    trained = rl.models("dfl", ("gbm", "gru"))
    prob = 0.5 * learned.predict_proba(trained["gbm"], f) + 0.5 * predict_gru(trained["gru"], f)
    hand = decode(build_features(touch_model=tm))
    blk = [(a, b + 1) for _, a, b in half_video_blocks(GAME)]
    groups = np.zeros(len(hand), dtype=np.int8)
    for i, (a, b) in enumerate(blk, start=1):
        groups[a:b] = i
    team, conf, weak = idt.pseudo_labels(f, hand)
    atd = load_arrays()
    cols = idt.player_columns(atd["players"], atd["team"], groups, (LENGTH, WIDTH))
    model = lgb.Booster(model_file=os.path.join(ROOT, "models", "po_team_v25.txt"))
    prob = idt.combine(prob, idt.predict_po(model, cols), weak, 0.5)
    n = len(hand)
    p1 = stage1(prob, f)
    bundles = [refine.load(os.path.join(ROOT, "models", "refine_v29_s{}.pt".format(s)))
               for s in range(5)]
    refine.BIN = bundles[0][0].cfg["BIN"]
    xs = (refine.inputs(p1, hand, f, cols), refine.inputs(p1, hand, f, cols, True))
    p_ref = np.mean([refine.predict(b, xs[0], xs[1], n) for b in bundles], axis=0)
    vote = tr.run_vote(f, ev.VOTE)
    if dead_prefix:
        import glob
        paths = sorted(glob.glob(os.path.join(ROOT, dead_prefix + "_s*.pt")))
        if not paths:
            raise SystemExit("no models " + dead_prefix + "_s*.pt")
        dead = [refine.load(p) for p in paths]
    else:
        dead = [refine.load(os.path.join(ROOT, "models", "dead_{}_s{}.pt".format(ev.DEAD_TAG, k)))
                for k in range(ev.DEAD_SEEDS)]
    refine.BIN = dead[0][0].cfg["BIN"]
    xd = (refine.dead_inputs(p1, hand, f, cols, False, False),
          refine.dead_inputs(p1, hand, f, cols, True, False))
    pdead = np.mean([refine.predict(b, xd[0], xd[1], n, symmetric=True) for b in dead], axis=0)
    hand2 = constrain(hand, refine.decode_dead(hand, pdead, blk, *ev.SETTING), ev.MODE)
    hand2, protect = ev.out_rule(hand2, f, blk, Args)
    hand2 = ev.remove_gate(hand2, pdead, blk, GATE[0], GATE[1], dd.FPS, protect)
    codes = ev.chain(hand2, p1, p_ref, f, blk, vote)
    extras = {"f_" + k: v for k, v in f.items()}
    extras.update(v31=codes, hand=hand, hand2=hand2, p1=p1, p_ref=p_ref,
                  vote=np.asarray(vote, dtype=np.float32), pdead=pdead,
                  ball_z=dd.blur_ball(atd["ball"])[:, 2] if atd["ball"].shape[1] > 2 else
                  np.full(n, np.nan, np.float32), blk=np.array(blk))
    return codes, extras


def write_and_score(codes, name):
    """Write evaluation/results/<name>.csv; return its public line (aggregate)."""
    from games import GAME, GAMES
    from pipeline.run import write_submission, RESULTS_DIR
    from report import brief_line, PUBLIC_GT
    path = write_submission(os.path.join(RESULTS_DIR, name + ".csv"), codes)
    reason = pm.validate_submission(pm.read_label_csv(path), GAMES[GAME]["n_video_frames"])
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    return brief_line(name, codes, gt) + ("" if not reason else "  INVALID " + reason)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--touch")
    parser.add_argument("--name")
    parser.add_argument("--dead", help="dead-ball head prefix, e.g. models/dead_ext_d1")
    parser.add_argument("--pick", type=float, help="touch pick threshold (default touch_net.PICK_THRESHOLD)")
    args = parser.parse_args()
    if args.pick:
        touch_net.PICK_THRESHOLD = args.pick
    codes, _ = ours_v31(args.touch, args.dead)
    from pipeline.run import RESULTS_DIR
    ref = pm.label_codes(pm.read_label_csv(os.path.join(RESULTS_DIR, "v31.csv")))
    print("frames differing from v31.csv: {:.3f}%".format(100 * np.mean(codes != ref)))
    if args.name:
        print(write_and_score(codes, args.name))


if __name__ == "__main__":
    main()
