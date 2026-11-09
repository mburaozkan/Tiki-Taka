"""MBR decoding (pipeline/mbr.py) against the v25 decoder on realistic DFL. Aggregate only.

    python tools/eval_mbr.py              # grid of tau x sample switch, 7 matches
    python tools/eval_mbr.py --ours       # also write evaluation/results/v26.csv with the chosen setting

Per DFL match, the v25 pipeline held out: hand path for D vs live, the held-out learned blend
(learned_probs_anim.npz) plus the players-only vote trained on the other six DFL matches and the
match's own pseudo-labels (in_domain_team, source dfl+decoded, weak gate, w 0.5); reference =
decode_hybrid at switch 10 (as v25). Inputs are cached in data/cache/dfl/v25_inputs.npz.

For each (tau, sample switch): spellF1 per match, mean, delta vs the reference, folds won / lost,
and which candidate the MBR picked. "nested" chooses the setting on the other six matches for each
held-out match (an honest estimate of choosing the setting this way).
"""

import argparse
import itertools
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
import dfl_data as dd  # noqa: E402
import in_domain_team as idt  # noqa: E402
from train_learned_dfl import match_features, test_seed, blocks  # noqa: E402
from pipeline import learned  # noqa: E402
from pipeline.mbr import mbr_hybrid  # noqa: E402
from pipeline.decode import PARAMS, emissions, decode_block, X, H, A  # noqa: E402

# Params
TAUS = (0.02, 0.05, 0.1, 0.2)
SAMPLE_SWITCHES = (5.0, 10.0, 20.0)
W = 0.5
REF_SWITCH = 10.0
CACHE = os.path.join(dd.CACHE, "v25_inputs.npz")


def v25_inputs():
    """{match: (labels, hand codes, combined probabilities)}; cached."""
    if os.path.isfile(CACHE):
        with np.load(CACHE) as c:
            return {g: (c[g + "__lab"], c[g + "__hand"], c[g + "__prob"]) for g in dd.GAMES}
    base = {}
    with np.load(idt.PROBS) as c:
        for k in c.files:
            name, g = k.split("__")
            base.setdefault(name, {})[g] = c[k]
    info, out = {}, {}
    for g in dd.GAMES:
        d = dd.load(g)
        f = match_features(g, test_seed(g))
        lab, blk = d["controlled"], blocks(d["period"])
        sc = emissions(f, PARAMS)
        hand = np.full(len(lab), X, dtype=np.int8)
        for a, b in blk:
            hand[a:b] = decode_block(sc, f, PARAMS, a, b)
        blend = (base["gbm"][g] + base["gru"][g]) / 2
        team, conf, weak = idt.pseudo_labels(f, hand)
        dteam, dmask = idt.decoded_pseudo(learned.decode_hybrid(hand, base["gbm"][g], blk, 20.0),
                                          base["gbm"][g])
        info[g] = dict(lab=lab, hand=hand, blend=blend, team=team, conf=conf, weak=weak,
                       dteam=dteam, dmask=dmask, cols=idt.player_cols_cached(g, d))
    for g in dd.GAMES:
        I = info[g]
        sets = [(info[o]["cols"], info[o]["lab"], np.isin(info[o]["lab"], (H, A)))
                for o in dd.GAMES if o != g]
        sets.append((I["cols"], np.where(I["conf"], I["team"], I["dteam"]), I["conf"] | I["dmask"]))
        p_po = idt.predict_po(idt.train_po_multi(sets), I["cols"])
        out[g] = (I["lab"], I["hand"], idt.combine(I["blend"], p_po, I["weak"], W))
        print("  inputs", g, flush=True)
    np.savez(CACHE, **{"{}__{}".format(g, k): v for g, t in out.items()
                       for k, v in zip(("lab", "hand", "prob"), t)})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ours", nargs=2, type=float, metavar=("TAU", "SWITCH"),
                        help="write evaluation/results/v26.csv with this setting")
    parser.add_argument("--skip-dfl", action="store_true")
    args = parser.parse_args()
    if not args.skip_dfl:
        dfl()
    if args.ours:
        ours(*args.ours)


def dfl():
    print("building v25 inputs on DFL ...", flush=True)
    data = v25_inputs()
    blk = {g: blocks(dd.load(g)["period"]) for g in dd.GAMES}
    ref = []
    for g in dd.GAMES:
        lab, hand, prob = data[g]
        ref.append(pm.evaluate(lab, learned.decode_hybrid(hand, prob, blk[g], REF_SWITCH),
                               bootstrap_n=0)["spell_f1_mean"])
    print("\nreference (v25 decoder, switch {:g}): {} | {:.3f}".format(
        REF_SWITCH, " ".join("{:.3f}".format(x) for x in ref), np.mean(ref)), flush=True)
    res = {}
    for tau, sw in itertools.product(TAUS, SAMPLE_SWITCHES):
        scores, picks = [], Counter()
        for g in dd.GAMES:
            lab, hand, prob = data[g]
            codes, pk = mbr_hybrid(hand, prob, blk[g], tau, sw, return_info=True)
            scores.append(pm.evaluate(lab, codes, bootstrap_n=0)["spell_f1_mean"])
            picks.update(pk)
        res[(tau, sw)] = scores
        won = sum(x > y + 1e-9 for x, y in zip(scores, ref))
        lost = sum(x < y - 1e-9 for x, y in zip(scores, ref))
        print("tau {:<5g} switch {:<4g} {} | {:.3f} | {:+.3f} | {}/{} | picks {}".format(
            tau, sw, " ".join("{:.3f}".format(x) for x in scores), np.mean(scores),
            np.mean(scores) - np.mean(ref), won, lost, dict(picks)), flush=True)
    nested = []
    for i in range(len(dd.GAMES)):
        best = max(res, key=lambda k: np.mean([s for j, s in enumerate(res[k]) if j != i]))
        nested.append(res[best][i])
    print("nested (setting chosen on the other six): {} | {:.3f} | {:+.3f} | {}/{}".format(
        " ".join("{:.3f}".format(x) for x in nested), np.mean(nested), np.mean(nested) - np.mean(ref),
        sum(x > y + 1e-9 for x, y in zip(nested, ref)), sum(x < y - 1e-9 for x, y in zip(nested, ref))))


def ours(tau, sw):
    """v25's inputs on our match, decoded by MBR -> evaluation/results/v26.csv."""
    from games import GAME, GAMES, half_video_blocks
    from pipeline.run import write_submission, RESULTS_DIR
    import possession_metrics as pm2
    v25 = pm2.label_codes(pm2.read_label_csv(os.path.join(RESULTS_DIR, "v25_w0.5.csv")))
    path = dd.ours_prob_path()
    if not os.path.isfile(path):
        sys.exit("run tools/in_domain_team.py --ours --skip-dfl --source dfl+decoded --weights 0.5 "
                 "--save-prob first")
    with np.load(path) as c:
        hand, prob = c["hand"], c["prob"]
    blks = [(a, b + 1) for _, a, b in half_video_blocks(GAME)]
    check = learned.decode_hybrid(hand, prob, blks, REF_SWITCH)
    print("v25 rebuilt from the saved inputs: {}".format(
        "identical" if np.array_equal(check, v25) else "DIFFERENT"))
    codes, picks = mbr_hybrid(hand, prob, blks, tau, sw, return_info=True)
    out = write_submission(os.path.join(RESULTS_DIR, "v26.csv"), codes)
    reason = pm2.validate_submission(pm2.read_label_csv(out), GAMES[GAME]["n_video_frames"])
    print("v26 (tau {:g}, switch {:g}, picks {}) -> {} ({})".format(
        tau, sw, picks, os.path.relpath(out, ROOT), reason or "valid"))


if __name__ == '__main__':
    main()
