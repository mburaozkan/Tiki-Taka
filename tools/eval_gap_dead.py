"""Missing ball as dead-ball evidence: w_gap and the players-easing-off rule, on top of v29.

    python tools/eval_gap_dead.py

Only the hand path (D vs live) changes: it is decoded again with each setting (pipeline/decode.py
PARAMS w_gap, w_ease, ease_gate), and the team is decoded as v29 on top (tools/tune_refine.decode,
alpha 0.5, switch 20, restart handover).
  DFL (blur + our gaps, 7 matches): the first stage and the second-stage team probabilities are
      v29's held-out ones (refine_b2.npz), kept fixed; per setting the mean spellF1, delta vs v29,
      folds won / lost, and nested (the setting picked on the other six).
  Our match: the second stage is run again on each new hand path (its live flag is an input), the
      first stage kept; public spF1 / mF1, full-match D share and possessions, and the share of
      the true stoppage after public #5 (the foul) called D.
Aggregate only.
"""

import itertools
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
from eval_refine import stage1  # noqa: E402
from tune_refine import decode  # noqa: E402
from train_learned_dfl import match_features, test_seed, blocks  # noqa: E402
from pipeline import refine  # noqa: E402
from pipeline.decode import PARAMS, emissions, decode_block, X, D  # noqa: E402

# Params
TEAM_TAG, TEAM_SEEDS, ALPHA, SWITCH = "b2", 5, 0.5, 20.0
W_GAP = (0.4, 0.6, 0.8)
W_EASE = (0.0, 0.5, 1.0)
EASE_GATE = (0.0, 1.0)


def hand_path(f, blk, p):
    sc = emissions(f, p)
    hand = np.full(len(f["ball_seen"]), X, dtype=np.int8)
    for a, b in blk:
        hand[a:b] = decode_block(sc, f, p, a, b)
    return hand


def score(lab, codes):
    return pm.evaluate(lab, codes, bootstrap_n=0)["spell_f1_mean"]


def main():
    refine.BIN = 2
    settings = [dict(PARAMS, w_gap=w, w_ease=e, ease_gate=gt)
                for w, e, gt in itertools.product(W_GAP, W_EASE, EASE_GATE) if e > 0 or not gt]
    names = ["w_gap {:g} ease {:g} gate {:g}".format(p["w_gap"], p["w_ease"], p["ease_gate"])
             for p in settings]

    data = v25_inputs()
    with np.load(os.path.join(dd.CACHE, "refine_{}.npz".format(TEAM_TAG))) as c:
        team = {k: c[k] for k in c.files}
    res = {k: [] for k in names}
    ref, same = [], []
    for g in dd.GAMES:
        d = dd.load(g)
        f = match_features(g, test_seed(g))
        blk = blocks(d["period"])
        lab, hand, prob = data[g]
        p1 = stage1(prob, f)
        p_ref = np.mean([team["{}__{}".format(g, s)] for s in range(TEAM_SEEDS)], axis=0)
        ref.append(score(lab, decode(hand, p1, p_ref, f, blk, ALPHA, SWITCH)))
        same.append(np.mean(hand_path(f, blk, PARAMS) == hand))
        for name, p in zip(names, settings):
            res[name].append(score(lab, decode(hand_path(f, blk, p), p1, p_ref, f, blk, ALPHA, SWITCH)))
        print("  DFL", g, flush=True)
    ref = np.array(ref)
    print("check: default PARAMS rebuild the cached hand path on {:.2f}% of frames".format(
        100 * np.mean(same)))

    ours = ours_scores(settings)
    print("\nv29 DFL {} | {:.3f}".format(" ".join("{:.3f}".format(x) for x in ref), ref.mean()))
    print("{:<30} {:>6} {:>8} {:>6} | {:>7} {:>6} {:>6} {:>5} {:>8}".format(
        "setting", "DFL", "delta", "folds", "pub spF1", "mF1", "D", "poss", "foul D"))
    for name in names:
        s = np.array(res[name])
        o = ours[name]
        print("{:<30} {:.3f} {:+.4f} {:>3}/{:<2} | {:>8.3f} {:.3f} {:5.1f}% {:>5} {:>7.0f}%".format(
            name, s.mean(), s.mean() - ref.mean(), int(np.sum(s > ref + 1e-9)),
            int(np.sum(s < ref - 1e-9)), o["spf1"], o["mf1"], o["D"], o["poss"], o["foul"]))
    table = {k: np.array(v) for k, v in res.items()}
    nested = np.array([table[max(table, key=lambda k: np.delete(table[k], i).mean())][i]
                       for i in range(len(ref))])
    print("nested {:.3f} | {:+.4f} | {}/{}".format(nested.mean(), nested.mean() - ref.mean(),
                                                  int(np.sum(nested > ref + 1e-9)),
                                                  int(np.sum(nested < ref - 1e-9))))


def ours_scores(settings):
    """{name: public spF1, mF1, full D %, possessions, % of the foul stoppage called D}."""
    from games import GAME, half_video_blocks
    from atd_cache import load_arrays
    from pipeline.features import build_features, LENGTH, WIDTH
    from pipeline import touch_net
    from pipeline.decode import decode as hand_decode
    from report import PUBLIC_GT
    tm = touch_net.load(os.path.join(ROOT, touch_net.MODEL_PATH))
    f_tag = build_features(touch_model=tm)                     # the hand path's features (tags)
    f = build_features(events=[], touch_model=tm)
    with np.load(os.path.join(dd.DATA_DIR, "cache", "v25_ours_prob.npz")) as c:
        prob = c["prob"]
    blk = [(a, b + 1) for _, a, b in half_video_blocks(GAME)]
    groups = np.zeros(len(prob), dtype=np.int8)
    for i, (a, b) in enumerate(blk, start=1):
        groups[a:b] = i
    atd = load_arrays()
    cols = idt.player_columns(atd["players"], atd["team"], groups, (LENGTH, WIDTH))
    bundles = [refine.load(os.path.join(ROOT, "models", "refine_v29_s{}.pt".format(s)))
               for s in range(TEAM_SEEDS)]
    p1 = stage1(prob, f)
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    runs = pm.valid_runs(gt)
    foul = [s for s in pm.segments(gt, runs) if s[2] == D][0]
    graded = groups > 0
    out = {}
    for p in settings:
        name = "w_gap {:g} ease {:g} gate {:g}".format(p["w_gap"], p["w_ease"], p["ease_gate"])
        hand = hand_decode(f_tag, p)
        xs = (refine.inputs(p1, hand, f, cols), refine.inputs(p1, hand, f, cols, True))
        p_ref = np.mean([refine.predict(b, xs[0], xs[1], len(hand)) for b in bundles], axis=0)
        codes = decode(hand, p1, p_ref, f, blk, ALPHA, SWITCH)
        ev = pm.evaluate(gt, codes, bootstrap_n=0)
        poss = sum(len(pm.possession_spells(codes, [(a, b)])) for a, b in blk)
        out[name] = {"spf1": ev["spell_f1_mean"], "mf1": ev["macro_f1"],
                     "D": 100 * np.mean(codes[graded] == D), "poss": poss,
                     "foul": 100 * np.mean(codes[foul[0]:foul[1]] == D)}
        print("  ours", name, flush=True)
    return out


if __name__ == '__main__':
    main()
