"""v31's chain, unchanged and untrained on the target, on SkillCorner / PFF matches. Aggregate only.

    python tools/run_v31_external.py skillcorner [game ...]
    python tools/run_v31_external.py pff [game ...] [--limit N]

Per match, exactly the deployed recipe of our match (tools/in_domain_team.ours + possession_view.
signals + tools/eval_dead_v30.ours, v31 settings):
  features with the deployed touch network (events=[]), hand path (emissions / decode_block as on
  DFL), learned blend of models/gbm_dfl + gru_dfl, players-only team model trained on the 7 DFL
  matches' true live frames plus THIS match's strong and decoded pseudo-labels (no truth: the
  same unsupervised step our match gets), stage 1, refine_v29 x5, touch-run vote, dead-ball head
  d1n x3 (add-only, 0.5 / 0.5 / 25), ball-out protected remove gate T 0.1, backdating.

Ball: the set's own broadcast ball (SkillCorner detected only, PFF raw VISIBLE + ESTIMATED; both
already have broadcast gaps and moving dead balls), blurred like DFL (dfl_data.blur_ball); no DFL
animation, no extra gaps.

Prints per match: spellF1 of v25 (hand + team vote) and v31, and how v31 does on the short truth
spells (< 2 s, between two spells of the other team, no D in between): caught = a predicted spell
of that team with IoU >= 0.5; also for the subset that follows a deep long kick (event
positions, >= 35 m within 5 s, kicked within 25 m of the kicker's goal). Results are appended to
data/external/results/v31_<set>.csv.
"""

import argparse
import os
import sys
import time

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
from pipeline import features, learned, refine, touch_net  # noqa: E402
from pipeline.decode import PARAMS, emissions, decode_block, H, A, D, X  # noqa: E402
from pipeline.sequence import predict_gru  # noqa: E402

# Params
W_PO = 0.5                       # v25's players-only weight
V25_SWITCH = 10.0
REFINE_TAG, REFINE_SEEDS = "v29", 5
GATE = (0.1, 0.0)                # v31: T, max_s
SHORT_S = 2.0
IOU = 0.5
MIN_M, MAX_FLIGHT_S, DEEP_M = 35.0, 5.0, 25.0
OUT = os.path.join(dd.DATA_DIR, "external", "results")


def blocks(period):
    """[(start, stop)] per period present (extra time 3 / 4 included, unlike DFL's helper)."""
    out = []
    for per in np.unique(period):
        idx = np.flatnonzero(period == per)
        out.append((idx[0], idx[-1] + 1))
    return out


class Args:                     # what eval_dead_v30.out_rule reads
    protect, hold = True, 0.0


_CACHE = {}


def deployed():
    if not _CACHE:
        from pipeline import run_learned as rl
        _CACHE["touch"] = touch_net.load(os.path.join(ROOT, touch_net.MODEL_PATH))
        _CACHE["learned"] = rl.models("dfl", ("gbm", "gru"))
        _CACHE["refine"] = [refine.load(os.path.join(ROOT, "models", "refine_{}_s{}.pt".format(
            REFINE_TAG, s))) for s in range(REFINE_SEEDS)]
        _CACHE["dead"] = [refine.load(os.path.join(ROOT, "models", "dead_{}_s{}.pt".format(
            ev.DEAD_TAG, k))) for k in range(ev.DEAD_SEEDS)]
        _CACHE["dfl_sets"] = []
        for g in dd.GAMES:
            d = dd.load(g)
            lab = d["controlled"]
            _CACHE["dfl_sets"].append((idt.player_cols_cached(g, d), lab, np.isin(lab, (H, A))))
    return _CACHE


def load_set(name, game):
    if name == "skillcorner":
        import skillcorner_data as sc
        d = sc.load(game)
        return d, (105.0, 68.0), d["ball"]
    import pff_data as pf
    d = pf.load(game)
    return d, pf._meta(game)[1], d["ball"]


FOLDS = 4
INTER = os.path.join(dd.DATA_DIR, "cache", "ext_v31")


def fold_of(name, game):
    """Fixed fold (0..FOLDS-1) of an external match: every 4th match of each set, in sorted order."""
    return games_of(name).index(game) % FOLDS


def chain(d, pitch, ball, touch_model=None, extras=None):
    """(v25, v31); if extras is a dict, the intermediates are put in it."""
    m = deployed()
    n = len(ball)
    tm = touch_model if touch_model is not None else m["touch"]
    ball = dd.blur_ball(ball)
    length, width = features.LENGTH, features.WIDTH
    features.LENGTH, features.WIDTH = pitch
    try:
        f = features.build_features({"players": d["players"], "ball": ball, "team": d["team"]},
                                    events=[], touch_model=tm)
        blk = blocks(d["period"])
        sc = emissions(f, PARAMS)
        hand = np.full(n, X, dtype=np.int8)
        for a, b in blk:
            hand[a:b] = decode_block(sc, f, PARAMS, a, b)
        prob = 0.5 * learned.predict_proba(m["learned"]["gbm"], f) + \
            0.5 * predict_gru(m["learned"]["gru"], f)
        groups = d["period"].astype(np.int8)
        cols = idt.player_columns(d["players"], d["team"], groups, pitch)
        team, conf, weak = idt.pseudo_labels(f, hand)
        dteam, dmask = idt.decoded_pseudo(learned.decode_hybrid(hand, prob, blk, V25_SWITCH), prob)
        merged = np.where(conf, team, dteam)
        po = idt.train_po_multi([(cols, merged, conf | dmask)] + m["dfl_sets"])
        prob = idt.combine(prob, idt.predict_po(po, cols), weak, W_PO)
        v25 = learned.decode_hybrid(hand, prob, blk, V25_SWITCH)
        p1 = stage1(prob, f)
        refine.BIN = m["refine"][0][0].cfg["BIN"]
        xs = (refine.inputs(p1, hand, f, cols), refine.inputs(p1, hand, f, cols, True))
        p_ref = np.mean([refine.predict(b, xs[0], xs[1], n) for b in m["refine"]], axis=0)
        vote = tr.run_vote(f, ev.VOTE)
        refine.BIN = m["dead"][0][0].cfg["BIN"]
        xd = (refine.dead_inputs(p1, hand, f, cols, False, False),
              refine.dead_inputs(p1, hand, f, cols, True, False))
        pdead = np.mean([refine.predict(b, xd[0], xd[1], n, symmetric=True) for b in m["dead"]],
                        axis=0)
        hand2 = constrain(hand, refine.decode_dead(hand, pdead, blk, *ev.SETTING), ev.MODE)
        hand2, protect = ev.out_rule(hand2, f, blk, Args)
        hand2 = ev.remove_gate(hand2, pdead, blk, GATE[0], GATE[1], dd.FPS, protect)
        v31 = ev.chain(hand2, p1, p_ref, f, blk, vote)
        if extras is not None:
            extras.update({"f_" + k: v for k, v in f.items()})
            extras.update(v25=v25, v31=v31, hand=hand, hand2=hand2, p1=p1, p_ref=p_ref,
                          vote=np.asarray(vote, dtype=np.float32), pdead=pdead, ball_z=ball[:, 2],
                          labels=d["labels"], period=d["period"])
    finally:
        features.LENGTH, features.WIDTH = length, width
    return v25, v31


def short_spells(lab):
    """[(a, b, code)] truth spells < SHORT_S between two adjacent spells of the other team."""
    runs = pm.valid_runs(lab)
    sp = pm.possession_spells(lab, runs)
    out = []
    for p, s, q in zip(sp, sp[1:], sp[2:]):
        if p[2] == q[2] != s[2] and p[1] == s[0] and s[1] == q[0] and \
                (s[1] - s[0]) / dd.FPS < SHORT_S:
            out.append(s)
    return out


def caught(spell, pred_spells):
    a, b, k = spell
    for c, e, j in pred_spells:
        if j == k and e > a and c < b:
            if (min(b, e) - max(a, c)) / float(max(b, e) - min(a, c)) >= IOU:
                return True
    return False


def keeper_side(d, team_id, per):
    """+1 if team team_id's goal is at x = 1 in period per (its most extreme player), else -1."""
    sel = d["period"] == per
    xs = np.nanmean(d["players"][sel][:, d["team"] == team_id, 0], axis=0)
    xs = xs[~np.isnan(xs)]
    if not len(xs):
        return 0
    k = xs[np.argmax(np.abs(xs - 0.5))]
    return 1 if k > 0.5 else -1


def deep_kick_spells(d, pitch, spells, name):
    """The short spells whose first frame is the landing of a deep long kick of the other team."""
    ev_ = d["events"]
    ball = d["ball_est"] if "ball_est" in d else d["ball"]
    kinds = d["event_end"] if name == "skillcorner" else d["event_kind"]
    kick_ok = (lambda k: k in ("pass", "clearance")) if name == "skillcorner" else \
        (lambda k: k in ("PA", "CR", "CL"))

    def pos(fr):
        for o in (0, -1, 1, -2, 2, -3, 3):
            i = fr + o
            if 0 <= i < len(ball) and not np.isnan(ball[i, 0]):
                return ball[i, :2] * pitch
        return None

    sides = {(t, int(p)): keeper_side(d, t, p) for t in (0, 1) for p in np.unique(d["period"])}
    landings = set()
    for i in range(len(ev_) - 1):
        a, b, t = ev_[i][:3]
        c, _, u = ev_[i + 1][:3]
        if not kick_ok(str(kinds[i])) or u == t or not (0 <= c - b <= MAX_FLIGHT_S * dd.FPS):
            continue
        p, q = pos(int(b)), pos(int(c))
        if p is None or q is None or np.hypot(*(q - p)) < MIN_M:
            continue
        side = sides[(int(t), int(d["period"][b]))]
        goal_x = pitch[0] if side > 0 else 0.0
        if side and abs(p[0] - goal_x) < DEEP_M:
            landings.add(int(c))
    out = []
    for s in spells:
        if any(s[0] - 3 * dd.FPS // 5 <= f0 <= s[1] for f0 in landings):
            out.append(s)
    return out


def games_of(name):
    if name == "skillcorner":
        import skillcorner_data as sc
        return sc.games()
    import pff_data as pf
    return pf.games()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("set", choices=("skillcorner", "pff"))
    parser.add_argument("games", nargs="*")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save", action="store_true", help="cache intermediates in data/cache/ext_v31")
    parser.add_argument("--touch-dir", help="per-fold touch networks <dir>/touch_f<k>.pt (the "
                        "fold that holds the match out)")
    parser.add_argument("--tag", default="v31", help="results file stem")
    parser.add_argument("--pick", type=float, help="touch pick threshold (default touch_net.PICK_THRESHOLD)")
    args = parser.parse_args()
    if args.pick:
        touch_net.PICK_THRESHOLD = args.pick
    gs = args.games or games_of(args.set)
    if args.limit:
        gs = gs[:args.limit]
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "{}_{}.csv".format(args.tag, args.set))
    done = set()
    if os.path.isfile(path):
        done = {line.split(",")[0] for line in open(path)}
    else:
        open(path, "w").write("game,v25,v31,short_n,short_v25,short_v31,deep_n,deep_v25,deep_v31\n")
    for g in gs:
        if g in done:
            continue
        t0 = time.time()
        d, pitch, ball = load_set(args.set, g)
        tm = None
        if args.touch_dir:
            tm = touch_net.load(os.path.join(args.touch_dir, "touch_f{}.pt".format(
                fold_of(args.set, g))))
        extras = {} if args.save else None
        v25, v31 = chain(d, pitch, ball, tm, extras)
        if args.save:
            os.makedirs(INTER, exist_ok=True)
            np.savez_compressed(os.path.join(INTER, "{}_{}_{}.npz".format(args.tag, args.set, g)),
                                **extras)
        lab = d["labels"]
        s25, s31 = tr.score(lab, v25), tr.score(lab, v31)
        short = short_spells(lab)
        deep = deep_kick_spells(d, pitch, short, args.set)
        runs = pm.valid_runs(lab)
        p25, p31 = pm.possession_spells(v25, runs), pm.possession_spells(v31, runs)
        row = [g, s25, s31, len(short), sum(caught(s, p25) for s in short),
               sum(caught(s, p31) for s in short), len(deep), sum(caught(s, p25) for s in deep),
               sum(caught(s, p31) for s in deep)]
        open(path, "a").write(",".join(str(x) if isinstance(x, (int, str)) else "{:.4f}".format(x)
                                       for x in row) + "\n")
        print("{} v25 {:.3f} v31 {:.3f} | short {} caught {} / {} | deep-kick {} caught {} / {} "
              "| {:.0f} s".format(g, s25, s31, row[3], row[4], row[5], row[6], row[7], row[8],
                                  time.time() - t0), flush=True)
    import pandas as pd
    r = pd.read_csv(path)
    r = r[r.game.astype(str).isin([str(x) for x in gs])]
    print("{} matches: v25 {:.4f}, v31 {:.4f} ({:+.4f}, {}/{} up/down) | short {} caught v25 {} "
          "v31 {} | deep-kick {} caught v25 {} v31 {}".format(
              len(r), r.v25.mean(), r.v31.mean(), (r.v31 - r.v25).mean(), int((r.v31 > r.v25).sum()),
              int((r.v31 < r.v25).sum()), r.short_n.sum(), r.short_v25.sum(), r.short_v31.sum(),
              r.deep_n.sum(), r.deep_v25.sum(), r.deep_v31.sum()))


if __name__ == "__main__":
    main()
