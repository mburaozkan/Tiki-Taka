"""The turnover network (touch_net trained on possession changes, train_touch.py --target
turnover) as cheaper team switches under the submission v30 (v29 + touch-run vote + backdating).
Aggregate only.

    python tools/eval_turnover_switch.py
    python tools/eval_turnover_switch.py --ours "w=10 p0=0.3 half_s=0.2" --name v30_to
    python tools/eval_turnover_switch.py --net short_net     # starts of possessions < 3 s

Where the network says the ball changes to team T, the H <-> A switch into T costs
switch - w x clip((peak P - p0) / (1 - p0), 0, 1) (peak over +/- half_s, at least 1 left), so a
short possession can survive the Viterbi; elsewhere v30 is unchanged. DFL as tools/eval_backdate.py
(held-out turnover net per match, v29's held-out second stage refine_b2): mean spellF1, delta vs
v30, folds won / lost, nested, and the catch rate (IoU >= 0.5) by true possession length.
"""

import argparse
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
import eval_touch_run as tr  # noqa: E402
from eval_backdate import SUBMISSION  # noqa: E402
from eval_mbr import v25_inputs  # noqa: E402
from eval_refine import stage1  # noqa: E402
from train_learned_dfl import match_features, test_seed, blocks  # noqa: E402
from pipeline import learned, fixes, touch_net  # noqa: E402

# Params
MODEL_DIR = os.path.join(dd.DATA_DIR, "cache", "models")
BACKDATE = {"p_max": 0.7, "max_s": 9.0, "loose_m": 1.5}
GRID = {"w": (5.0, 10.0, 15.0, 19.0), "p0": (0.0, 0.3, 0.5), "half_s": (0.2, 0.5)}
LENGTHS = ((0, 1), (1, 2), (2, 5), (5, np.inf))
NET = "turnover_net"    # --net short_net: train_touch.py --target short


def turnover_prob(players, ball, team, pitch, model):
    """(n, 2) P(possession changes to home / away) per frame."""
    path = len(model[1]) > touch_net.n_base_inputs()
    return touch_net.predict(model, touch_net.inputs(players, ball, team, pitch, path=path))[:, 1:]


def discount(turn, w, p0, half_s, switch=tr.SWITCH):
    """(n, 2): [:, 0] off the switch into H, [:, 1] into A."""
    k = int(round(half_s * dd.FPS))
    out = np.zeros(turn.shape, dtype=np.float64)
    for col in (0, 1):
        p = turn[:, col].astype(np.float64)
        peak = p.copy()
        for s in range(1, k + 1):
            np.maximum(peak[s:], p[:-s], out=peak[s:])
            np.maximum(peak[:-s], p[s:], out=peak[:-s])
        out[:, col] = np.minimum(w * np.clip((peak - p0) / (1.0 - p0), 0, 1), switch - 1.0)
    return out


def v30(hand, prob, p_ref, f, blk, disc=None):
    """The submission's chain (eval_touch_run.decode + backdating), with an optional discount."""
    q = np.clip(p_ref, 1e-4, 1 - 1e-4)
    vote = tr.run_vote(f, SUBMISSION)
    p = fixes.add_vote(stage1(prob, f), tr.ALPHA * np.log(q / (1 - q)) + vote, 1.0)
    codes = learned.decode_hybrid(hand, p, blk, tr.SWITCH, discount=disc)
    codes = fixes.restart_handover(codes, f, max_s=3.0, slow_ms=2.0)
    return fixes.backdate(codes, f, **BACKDATE)


def by_length(lab, codes):
    """{(lo, hi): [best IoU of one predicted same-team spell, per true possession]}."""
    out = {b: [] for b in LENGTHS}
    for sa, sb, k in pm.possession_spells(lab, pm.valid_runs(lab)):
        seg = (codes[sa:sb] == k).astype(np.int8)
        e = np.flatnonzero(np.diff(np.concatenate(([0], seg, [0]))))
        best = 0.0
        for ra, rb in zip(e[::2], e[1::2]):
            pa, pb = sa + ra, sa + rb
            while pa > 0 and codes[pa - 1] == k:
                pa -= 1
            while pb < len(codes) and codes[pb] == k:
                pb += 1
            best = max(best, (rb - ra) / float(max(pb, sb) - min(pa, sa)))
        length = (sb - sa) / float(dd.FPS)
        out[next(b for b in LENGTHS if b[0] <= length < b[1])].append(best)
    return out


def caught(rows):
    return "  ".join("{:g}-{:g}s {:.2f}".format(b[0], b[1], np.mean(np.concatenate(
        [m[b] for m in rows]) >= 0.5)) for b in LENGTHS)


def dfl():
    data = v25_inputs()
    with np.load(os.path.join(dd.CACHE, "refine_{}.npz".format(tr.TEAM_TAG))) as c:
        team = {k: c[k] for k in c.files}
    settings = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    res = [[] for _ in settings]
    ref, lens = [], {}
    for g in dd.GAMES:
        d = dd.load(g)
        f = match_features(g, test_seed(g))
        cache = os.path.join(dd.CACHE, "{}_{}_{}.npy".format(
            "turn" if NET == "turnover_net" else NET, g, test_seed(g)))
        if os.path.isfile(cache):
            turn = np.load(cache)
        else:
            model = touch_net.load(os.path.join(MODEL_DIR, "{}_{}.pt".format(NET, g)))
            turn = turnover_prob(d["players"], dd.prepare_ball(d, test_seed(g)), d["team"],
                                 dd.PITCH, model)
            np.save(cache, turn)
        blk = blocks(d["period"])
        lab, hand, prob = data[g]
        p_ref = np.mean([team["{}__{}".format(g, s)] for s in range(tr.TEAM_SEEDS)], axis=0)
        base = v30(hand, prob, p_ref, f, blk)
        ref.append(tr.score(lab, base))
        lens.setdefault("v30", []).append(by_length(lab, base))
        for k, kw in enumerate(settings):
            codes = v30(hand, prob, p_ref, f, blk, discount(turn, **kw))
            res[k].append(tr.score(lab, codes))
            lens.setdefault(k, []).append(by_length(lab, codes))
        print("  DFL", g, flush=True)
    ref = np.array(ref)
    res = [np.array(r) for r in res]
    name = [" ".join("{}={:g}".format(a, b) for a, b in s.items()) for s in settings]
    print("v30 reference {} | {:.4f}".format(" ".join("{:.3f}".format(x) for x in ref), ref.mean()))
    order = sorted(range(len(settings)), key=lambda k: -res[k].mean())
    for k in order:
        s = res[k]
        print("{:<24} {} | {:.4f} | {:+.4f} | {}/{}".format(
            name[k], " ".join("{:.3f}".format(x) for x in s), s.mean(), s.mean() - ref.mean(),
            int(np.sum(s > ref + 1e-9)), int(np.sum(s < ref - 1e-9))))
    nested = np.array([res[max(range(len(settings)), key=lambda k: np.delete(res[k], i).mean())][i]
                       for i in range(len(dd.GAMES))])
    print("nested {} | {:.4f} | {:+.4f} | {}/{}".format(
        " ".join("{:.3f}".format(x) for x in nested), nested.mean(), nested.mean() - ref.mean(),
        int(np.sum(nested > ref + 1e-9)), int(np.sum(nested < ref - 1e-9))))
    print("\ncaught (IoU >= 0.5) by true possession length, all 7 matches")
    print("  {:<24} {}".format("v30", caught(lens["v30"])))
    strongest = max(range(len(settings)), key=lambda k: (settings[k]["w"], -settings[k]["p0"],
                                                         settings[k]["half_s"]))
    for k in (order[0], strongest):
        print("  {:<24} {}".format(name[k], caught(lens[k])))


def ours(text, name):
    from games import GAME, GAMES, half_video_blocks
    from pipeline.run import write_submission, RESULTS_DIR
    from report import brief_line, PUBLIC_GT
    from possession_view import signals
    kw = {k: float(v) for k, v in (t.split("=") for t in text.split())}
    s = signals()
    with np.load(os.path.join(dd.DATA_DIR, "cache", "v25_ours_prob{}.npz".format(
            os.environ.get("MTAH_VARIANT", "")))) as c:
        prob = c["prob"]
    hand, f = s["hand"], s["f"]
    blk = [(a, b + 1) for _, a, b in half_video_blocks(GAME)]
    model = touch_net.load(os.path.join(MODEL_DIR, "{}_all.pt".format(NET)))
    turn = turnover_prob(s["players"], s["ball"], s["team"], s["pitch"], model)
    base = v30(hand, prob, s["p_ref"], f, blk)
    codes = v30(hand, prob, s["p_ref"], f, blk, discount(turn, **kw))
    path = write_submission(os.path.join(RESULTS_DIR, name + ".csv"), codes)
    reason = pm.validate_submission(pm.read_label_csv(path), GAMES[GAME]["n_video_frames"])
    live = np.isin(hand, (0, 1))
    print("{} ({}) -> {} ({}); differs from v30 on {:.2f}% of live frames".format(
        name, text, os.path.relpath(path, ROOT), reason or "valid",
        100 * np.mean(codes[live] != base[live])))
    print("  " + brief_line(name, codes, pm.label_codes(pm.read_label_csv(PUBLIC_GT))))


def main():
    global NET
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ours", help='"w=.. p0=.. half_s=.."')
    parser.add_argument("--name", default="v30_to")
    parser.add_argument("--net", default=NET, choices=("turnover_net", "short_net"))
    args = parser.parse_args()
    NET = args.net
    if args.ours:
        ours(args.ours, args.name)
    else:
        dfl()


if __name__ == "__main__":
    main()
