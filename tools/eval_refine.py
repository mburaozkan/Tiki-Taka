"""Second-stage bidirectional team model (pipeline/refine.py) against v27 on DFL. Aggregate only.

    python tools/eval_refine.py

Leave one match out: the refiner is trained on the other six matches' first-stage team
probabilities (v27's: the held-out v25 inputs plus the touch-margin vote), features and true
labels, then re-decides the held-out match's team. Caveat: the first-stage probabilities of the
training matches come from models that saw the held-out match (mild stacking leak).

Decoders: "replace" (the refined P(home) alone) at switch 10 / 20 / 30, and "blend" (v27 log-odds
plus alpha times the refined log-odds) at switch 10, each followed by v27's restart handover.
Reports spellF1, delta vs v27, folds won / lost, turnover timing and false switches (as
tools/eval_reaction.py). Held-out refined probabilities: data/cache/dfl/refine_probs.npz.
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
import dfl_data as dd  # noqa: E402
import in_domain_team as idt  # noqa: E402
from eval_mbr import v25_inputs  # noqa: E402
from eval_reaction import v27_codes, timing  # noqa: E402
from train_learned_dfl import match_features, test_seed, blocks  # noqa: E402
from pipeline import learned, fixes, refine  # noqa: E402

# Params
PROBS = os.path.join(dd.CACHE, "refine_probs.npz")
SWITCHES = (10.0, 20.0, 30.0)
ALPHAS = (0.5, 1.0)


def stage1(prob, f):
    return fixes.add_vote(prob, 0.5 * fixes.margin_vote(f, vote_s=2.0), 1.0)


def main():
    import argparse
    global PROBS
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-reaction", action="store_true",
                        help="ablation: the refiner without the team-reaction inputs")
    parser.add_argument("--ours", type=float, metavar="ALPHA",
                        help="train on all 7, save models/refine_v28.pt, write v28.csv (blend alpha)")
    parser.add_argument("--skip-dfl", action="store_true")
    args = parser.parse_args()
    if args.no_reaction:
        refine.USE_REACTION = False
        PROBS = PROBS.replace(".npz", "_noreact.npz")
    data = v25_inputs()
    ctx, ex = {}, {}
    for g in dd.GAMES:
        d = dd.load(g)
        f = match_features(g, test_seed(g))
        cols = idt.player_cols_cached(g, d)
        lab, hand, prob = data[g]
        p1 = stage1(prob, f)
        ctx[g] = (f, blocks(d["period"]), p1)
        ex[g] = []
        for flip in (False, True):
            y, v = refine.targets(lab, flip)
            ex[g].append((refine.inputs(p1, hand, f, cols, flip), y, v))
        print("inputs", g, ex[g][0][0].shape, flush=True)

    if args.ours is not None:
        ours(ex, args.ours)
    if args.skip_dfl:
        return
    refined = {}
    if os.path.isfile(PROBS):
        with np.load(PROBS) as c:
            refined = {g: c[g] for g in c.files}
    for g in dd.GAMES:
        if g in refined:
            continue
        print("held out", g, flush=True)
        bundle = refine.train([e for o in dd.GAMES if o != g for e in ex[o]], log=lambda *_: None)
        refined[g] = refine.predict(bundle, ex[g][0][0], ex[g][1][0], len(data[g][0]))
        np.savez(PROBS, **refined)

    def summary(preds):
        s, n_true, offs, false, longf = [], 0, [], 0, 0
        for g in dd.GAMES:
            lab = data[g][0]
            s.append(pm.evaluate(lab, preds[g], bootstrap_n=0)["spell_f1_mean"])
            nt, o, fa, lf = timing(lab, preds[g])
            n_true, offs, false, longf = n_true + nt, offs + o, false + fa, longf + lf
        offs = np.array(offs)
        return s, "found {:.0f}%  median {:+.2f}s  within 1 s {:.0f}%  false/h in long {:.1f}".format(
            100.0 * len(offs) / max(n_true, 1), np.median(offs), 100.0 * np.mean(np.abs(offs) <= 1),
            false / (longf / dd.FPS / 3600.0))

    # team accuracy on true live frames
    acc1, acc2 = [], []
    for g in dd.GAMES:
        lab = data[g][0]
        live = np.isin(lab, (0, 1))
        p1 = ctx[g][2]
        acc1.append(np.mean((p1[live, 0] >= p1[live, 1]) == (lab[live] == 0)))
        acc2.append(np.mean((refined[g][live] >= 0.5) == (lab[live] == 0)))
    print("\nper-frame team accuracy on live frames: first stage {:.1f}%, refined {:.1f}%".format(
        100 * np.mean(acc1), 100 * np.mean(acc2)))

    ref, ref_t = summary({g: v27_codes(data[g][1], data[g][2], ctx[g][0], ctx[g][1])
                          for g in dd.GAMES})
    print("v27 reference        {} | {:.3f} | {}".format(
        " ".join("{:.3f}".format(x) for x in ref), np.mean(ref), ref_t), flush=True)

    def show(name, preds):
        s, t = summary(preds)
        print("{:<20} {} | {:.3f} | {:+.4f} | {}/{} | {}".format(
            name, " ".join("{:.3f}".format(x) for x in s), np.mean(s), np.mean(s) - np.mean(ref),
            sum(x > y + 1e-9 for x, y in zip(s, ref)), sum(x < y - 1e-9 for x, y in zip(s, ref)), t),
            flush=True)

    for sw in SWITCHES:
        preds = {}
        for g in dd.GAMES:
            f, blk, _ = ctx[g]
            p = refined[g]
            pr = np.stack([p, 1 - p, np.zeros_like(p)], axis=1)
            codes = learned.decode_hybrid(data[g][1], pr, blk, sw)
            preds[g] = fixes.restart_handover(codes, f, max_s=3.0, slow_ms=2.0)
        show("replace sw {:g}".format(sw), preds)
    for alpha in ALPHAS:
        preds = {}
        for g in dd.GAMES:
            f, blk, p1 = ctx[g]
            q = np.clip(refined[g], 1e-4, 1 - 1e-4)
            pr = fixes.add_vote(p1, alpha * np.log(q / (1 - q)), 1.0)
            codes = learned.decode_hybrid(data[g][1], pr, blk, 10.0)
            preds[g] = fixes.restart_handover(codes, f, max_s=3.0, slow_ms=2.0)
        show("blend alpha {:g}".format(alpha), preds)


def ours(ex, alpha):
    """Refiner trained on all 7 DFL matches, on top of v27 on our match -> v28.csv."""
    from games import GAME, GAMES, half_video_blocks
    from atd_cache import load_arrays
    from pipeline.features import build_features, LENGTH, WIDTH
    from pipeline import touch_net
    from pipeline.run import write_submission, RESULTS_DIR
    from report import brief_line, PUBLIC_GT
    path = os.path.join(ROOT, "models", "refine_v28.pt")
    if os.path.isfile(path):
        bundle = refine.load(path)
    else:
        bundle = refine.train([e for g in dd.GAMES for e in ex[g]], log=lambda *_: None)
        refine.save(bundle, path)
    with np.load(os.path.join(dd.DATA_DIR, "cache", "v25_ours_prob.npz")) as c:
        hand, prob = c["hand"], c["prob"]
    f = build_features(events=[], touch_model=touch_net.load(os.path.join(ROOT, touch_net.MODEL_PATH)))
    blk = [(a, b + 1) for _, a, b in half_video_blocks(GAME)]
    groups = np.zeros(len(hand), dtype=np.int8)
    for i, (a, b) in enumerate(blk, start=1):
        groups[a:b] = i
    atd = load_arrays()
    cols = idt.player_columns(atd["players"], atd["team"], groups, (LENGTH, WIDTH))
    p1 = stage1(prob, f)
    p = refine.predict(bundle, refine.inputs(p1, hand, f, cols), refine.inputs(p1, hand, f, cols, True),
                       len(hand))
    q = np.clip(p, 1e-4, 1 - 1e-4)
    codes = learned.decode_hybrid(hand, fixes.add_vote(p1, alpha * np.log(q / (1 - q)), 1.0), blk, 10.0)
    codes = fixes.restart_handover(codes, f, max_s=3.0, slow_ms=2.0)
    v27 = learned.decode_hybrid(hand, p1, blk, 10.0)
    v27 = fixes.restart_handover(v27, f, max_s=3.0, slow_ms=2.0)
    out = write_submission(os.path.join(RESULTS_DIR, "v28.csv"), codes)
    reason = pm.validate_submission(pm.read_label_csv(out), GAMES[GAME]["n_video_frames"])
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    live = np.isin(hand, (0, 1)) & (groups > 0)
    print("v28 (alpha {:g}) -> {} ({}); team differs from v27 on {:.1f}% of live frames".format(
        alpha, os.path.relpath(out, ROOT), reason or "valid", 100 * np.mean(codes[live] != v27[live])))
    print("  " + brief_line("v28", codes, gt))


if __name__ == '__main__':
    main()
