"""Re-tune v31's decoder and post-step settings on SkillCorner + PFF together. Aggregate only.

    python tools/retune_chain.py team        # team decoder: switch x alpha x run-vote weight
    python tools/retune_chain.py post --base "switch=20 alpha=0.5"   # handover x backdating
    python tools/retune_chain.py gate --base "..."                    # remove-gate T
    python tools/retune_chain.py one --base "..."                     # a single setting

Every setting was tuned on DFL one step at a time, before the dead-ball head and the gate existed.
This re-runs v31's chain after the dead-ball step (cached by tools/run_v31_external.py --save,
data/cache/ext_v31/) with other settings, on all 84 external matches (0.4 s per match and
setting, worker processes over matches). v31's chain, with the knobs:
  p = add_vote(p1, alpha * logit(p_ref) + vote_w / 2 * vote)    (cached vote = 2 x run_vote)
  codes = restart_handover(decode_hybrid(hand2, p, switch), ho_max_s, ho_slow)
  codes = backdate(codes, bd_p_max, bd_max_s, bd_loose)
  gate: hand2 rebuilt from hand + pdead (add-only head, ball-out protected remove gate T, only
  stretches <= gate_max_s if > 0).
Prints the best settings (mean spellF1 over the 84 matches; SkillCorner and PFF separately),
delta vs v31 and matches up / down, and a nested 4-fold result (setting chosen on 3 folds of
run_v31_external.fold_of, scored on the 4th). Per-match scores go to data/cache/retune_<grid>.npz.
"""

import argparse
import glob
import itertools
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

# Params
V31 = {"switch": 20.0, "alpha": 0.5, "vote_w": 2.0, "ho_max_s": 3.0, "ho_slow": 2.0,
       "bd_p_max": 0.7, "bd_max_s": 9.0, "bd_loose": 1.5, "gate_t": 0.1, "gate_max_s": 0.0}
GRIDS = {
    "team": {"switch": (10.0, 14.0, 17.0, 20.0, 24.0, 28.0, 35.0), "alpha": (0.3, 0.5, 0.7, 1.0),
             "vote_w": (0.0, 1.0, 2.0, 3.0)},
    "post": {"ho_max_s": (0.0, 2.0, 3.0, 4.0), "ho_slow": (1.5, 2.0, 2.5),
             "bd_p_max": (0.0, 0.6, 0.7, 0.8, 0.9), "bd_max_s": (6.0, 9.0, 12.0),
             "bd_loose": (1.0, 1.5, 2.5)},
    "gate": {"gate_t": (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6), "gate_max_s": (0.0, 6.0, 8.0, 12.0)},
    "one": {},
}
INTER = os.path.join(ROOT, "data", "cache", "ext_v31")
WORKERS = 12


def chain(c, f, blk, s, hand2=None):
    import eval_touch_run as tr
    from pipeline import fixes, learned
    q = np.clip(c["p_ref"], 1e-4, 1 - 1e-4)
    p = fixes.add_vote(c["p1"], s["alpha"] * np.log(q / (1 - q)) + s["vote_w"] / 2.0 * c["vote"], 1.0)
    h = c["hand2"] if hand2 is None else hand2
    codes = learned.decode_hybrid(h, p, blk, s["switch"])
    if s["ho_max_s"] > 0:
        codes = fixes.restart_handover(codes, f, max_s=s["ho_max_s"], slow_ms=s["ho_slow"])
    if s["bd_p_max"] > 0:
        codes = fixes.backdate(codes, f, p_max=s["bd_p_max"], max_s=s["bd_max_s"],
                               loose_m=s["bd_loose"])
    return codes


def gated_hand(c, f, blk, t, max_s):
    import dfl_data as dd
    import eval_dead_v30 as ev
    from eval_dead_refine import constrain
    from pipeline import refine
    hand = c["hand"]
    h2 = constrain(hand, refine.decode_dead(hand, c["pdead"], blk, *ev.SETTING), ev.MODE)
    protect = np.asarray(f["ball_out"]) > 0
    return ev.remove_gate(h2, c["pdead"], blk, t, max_s, dd.FPS, protect) if t > 0 else h2


def run_match(job):
    path, settings = job
    import eval_touch_run as tr
    import run_v31_external as rx
    with np.load(path) as z:
        c = {k: z[k] for k in z.files}
    f = {k[2:]: v for k, v in c.items() if k.startswith("f_")}
    blk = rx.blocks(c["period"])
    hands = {}
    out = []
    for s in settings:
        h = None
        g = (s["gate_t"], s["gate_max_s"])
        if g != (V31["gate_t"], V31["gate_max_s"]):
            if g not in hands:
                hands[g] = gated_hand(c, f, blk, *g)
            h = hands[g]
        out.append(tr.score(c["labels"], chain(c, f, blk, s, h)))
    return out


def parse(text):
    s = dict(V31)
    for item in (text or "").split():
        k, v = item.split("=")
        if k not in s:
            raise SystemExit("unknown knob " + k)
        s[k] = float(v)
    return s


def label(s, keys):
    return " ".join("{}={:g}".format(k, s[k]) for k in keys) or "base"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("grid", choices=list(GRIDS))
    parser.add_argument("--base", help='knobs other than the grid\'s, e.g. "switch=17 alpha=0.7"')
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()
    import run_v31_external as rx
    base = parse(args.base)
    keys = list(GRIDS[args.grid])
    settings = [dict(base, **dict(zip(keys, v))) for v in itertools.product(*GRIDS[args.grid].values())]
    settings = [dict(V31)] + [s for s in settings if s != V31]
    paths = sorted(glob.glob(os.path.join(INTER, "v31_skillcorner_*.npz"))) + \
        sorted(glob.glob(os.path.join(INTER, "v31_pff_*.npz")))
    sets = np.array(["skillcorner" if "_skillcorner_" in p else "pff" for p in paths])
    games = [os.path.basename(p)[:-4].split("_", 2)[2] for p in paths]
    folds = np.array([rx.fold_of(st, g) for st, g in zip(sets, games)])
    print("{} settings x {} matches".format(len(settings), len(paths)), flush=True)
    with Pool(WORKERS) as pool:
        res = np.array(pool.map(run_match, [(p, settings) for p in paths]))   # (matches, settings)
    np.savez(os.path.join(ROOT, "data", "cache", "retune_{}.npz".format(args.grid)), scores=res,
             sets=sets, folds=folds, labels=np.array([label(s, V31) for s in settings]))
    ref = res[:, 0]
    sc, pf = sets == "skillcorner", sets == "pff"
    print("v31 reference: all {:.4f} | SkillCorner {:.4f} | PFF {:.4f}".format(
        ref.mean(), ref[sc].mean(), ref[pf].mean()))
    order = np.argsort(-res.mean(axis=0))
    print("{:<52} {:>7} {:>8} {:>7} {:>7} {:>8}".format("setting", "all", "delta", "up/dn", "SC", "PFF"))
    for j in order[:args.top]:
        d = res[:, j] - ref
        print("{:<52} {:.4f} {:+.4f} {:>3}/{:<3} {:.4f} {:.4f}".format(
            label(settings[j], keys) if j else "v31", res[:, j].mean(), d.mean(),
            int(np.sum(d > 1e-9)), int(np.sum(d < -1e-9)), res[sc, j].mean(), res[pf, j].mean()))
    nested = np.zeros(len(paths))
    picks = []
    for k in range(4):
        tr_, te = folds != k, folds == k
        j = int(np.argmax(res[tr_].mean(axis=0)))
        picks.append(label(settings[j], keys) if j else "v31")
        nested[te] = res[te, j]
    d = nested - ref
    print("nested 4-fold: {:.4f} ({:+.4f}, {}/{} up/down; SC {:+.4f}, PFF {:+.4f})".format(
        nested.mean(), d.mean(), int(np.sum(d > 1e-9)), int(np.sum(d < -1e-9)), d[sc].mean(),
        d[pf].mean()))
    print("fold picks: " + " | ".join(picks))


if __name__ == "__main__":
    main()
