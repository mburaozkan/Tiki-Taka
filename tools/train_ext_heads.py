"""Second-stage heads (team refiner, dead-ball head) retrained with SkillCorner + PFF. Aggregate only.

    python tools/train_ext_heads.py train team --tag t1 [--seeds 2] [--no-dfl]
    python tools/train_ext_heads.py train dead --tag d1
    python tools/train_ext_heads.py eval --team t1 --dead d1 [--ours]   # either or both

Why: refine_v29 (team) and dead_d1n were trained on DFL only. DFL's truth comes from our
controlled-possession filter, which merges brief contacts the organisers count as possessions
(#10 / #12), and DFL's dead ball is animated by us. SkillCorner / PFF label short possessions
as possessions and have real broadcast stoppages.

train: same architecture and inputs as the deployed models (pipeline/refine.py, BIN 2, GRU 64 x 2;
team = refine.inputs, dead = refine.dead_inputs without P(dead), as d1n), from
tools/ext_inputs.py (external) and the DFL inputs of tools/eval_dead_refine.py.
  folds  : for each of the 4 fixed external folds, DFL + the other 3 folds -> held-out P on the fold
  dflout : the 84 external matches only -> P on the 7 DFL matches (a provider never seen)
  dfllomo: per DFL match, the other 6 DFL + all 84 external -> P on it (keys dfl_<g>, use its own tag)
  all    : DFL + all 84 -> models for our match (data/cache/models/ext_heads/<tag>_s<k>.pt)
Held-out probabilities go to data/cache/ext_heads_<tag>.npz (keys <set>_<game>__<seed>, dfl_<g>__<seed>).
eval: v31's chain with the new team P (in place of refine_v29's p_ref) and / or the new P(dead) (in
place of d1n: add-only head, ball-out protected gate T 0.1 rebuilt), spellF1 vs v31 on SkillCorner +
PFF (held-out folds), DFL (dflout) and, with --ours, our match (public; MTAH_VARIANT respected),
plus short spells caught (< 2 s between two of the other team, IoU >= 0.5).
"""

import argparse
import glob
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

# Params
BIN = 2
FOLDS = 4
INTER = os.path.join(ROOT, "data", "cache", "ext_v31")
MODELS = os.path.join(ROOT, "data", "cache", "models", "ext_heads")


def ext_list():
    import run_v31_external as rx
    out = []
    for name in ("skillcorner", "pff"):
        for p in sorted(glob.glob(os.path.join(INTER, "x_{}_*.npz".format(name)))):
            g = os.path.basename(p)[:-4].split("_", 2)[2]
            out.append((name, g, rx.fold_of(name, g)))
    return out


def ext_examples(kind, name, g):
    with np.load(os.path.join(INTER, "x_{}_{}.npz".format(name, g))) as z:
        xk, yk, vk = ("team", "ty", "tv") if kind == "team" else ("dead", "dy", "dv")
        return [(z[xk], z[yk], z[vk]), (z[xk + "_f"], z[yk + "_f"], z[vk + "_f"])]


def ext_x(kind, name, g):
    with np.load(os.path.join(INTER, "x_{}_{}.npz".format(name, g))) as z:
        return z[kind], z[kind + "_f"]


_DFL = {}


def dfl_data():
    """{g: (lab, hand, p1, f, blk, cols)} from the held-out first stage (as eval_dead_refine)."""
    if not _DFL:
        import in_domain_team as idt
        from eval_mbr import v25_inputs
        from eval_refine import stage1
        from train_learned_dfl import match_features, test_seed, blocks
        data = v25_inputs()
        for g in dd.GAMES:
            d = dd.load(g)
            f = match_features(g, test_seed(g))
            lab, hand, prob = data[g]
            _DFL[g] = (lab, hand, stage1(prob, f), f, blocks(d["period"]),
                       idt.player_cols_cached(g, d), prob)
            print("  DFL", g, flush=True)
    return _DFL


def dfl_examples(kind, g):
    from pipeline import refine
    lab, hand, p1, f, _, cols, _ = dfl_data()[g]
    out = []
    for flip in (False, True):
        if kind == "team":
            out.append((refine.inputs(p1, hand, f, cols, flip),) + refine.targets(lab, flip))
        else:
            out.append((refine.dead_inputs(p1, hand, f, cols, flip, False),) +
                       refine.dead_targets(lab, flip))
    return out


def predict(bundle, kind, x, xf, n):
    from pipeline import refine
    return refine.predict(bundle, x, xf, n, symmetric=(kind == "dead")).astype(np.float32)


def train_cmd(args):
    from pipeline import refine
    refine.BIN = BIN
    cache = os.path.join(ROOT, "data", "cache", "ext_heads_{}.npz".format(args.tag))
    probs = {}
    if os.path.isfile(cache):
        with np.load(cache) as c:
            probs = {k: c[k] for k in c.files}
    ext = ext_list()
    n_of = {}
    for name, g, _ in ext:
        with np.load(os.path.join(INTER, "v31_{}_{}.npz".format(name, g))) as z:
            n_of[(name, g)] = len(z["labels"])
    ex_ext = {(name, g): ext_examples(args.kind, name, g) for name, g, _ in ext}
    ex_dfl = {} if args.no_dfl else {g: dfl_examples(args.kind, g) for g in dd.GAMES}
    print("{} examples: {} external matches, {} DFL".format(args.kind, len(ex_ext), len(ex_dfl)),
          flush=True)
    jobs = []
    if "folds" in args.modes:
        jobs += [("fold", k) for k in range(FOLDS)]
    if "dflout" in args.modes:
        jobs.append(("dflout", None))
    if "dfllomo" in args.modes:
        jobs += [("dfllomo", g) for g in dd.GAMES]
    if "all" in args.modes:
        jobs.append(("all", None))
    for mode, k in jobs:
        for seed in range(args.seeds):
            if mode == "fold":
                keys = ["{}_{}__{}".format(n, g, seed) for n, g, fo in ext if fo == k]
            elif mode == "dflout":
                keys = ["dfl_{}__{}".format(g, seed) for g in dd.GAMES]
            elif mode == "dfllomo":
                keys = ["dfl_{}__{}".format(k, seed)]
            else:
                keys = []
                path = os.path.join(MODELS, "{}_s{}.pt".format(args.tag, seed))
                if os.path.isfile(path):
                    continue
            if keys and all(key in probs for key in keys):
                continue
            t0 = time.time()
            train = [e for (n, g, fo) in ext if mode == "all" or (mode == "fold" and fo != k)
                     or mode in ("dflout", "dfllomo") for e in ex_ext[(n, g)]]
            if mode != "dflout":
                train += [e for g in ex_dfl if not (mode == "dfllomo" and g == k)
                          for e in ex_dfl[g]]
            bundle = refine.train(train, steps=args.steps, seed=seed, log=lambda *_: None)
            if mode == "fold":
                for n, g, fo in ext:
                    if fo == k:
                        x, xf = ext_x(args.kind, n, g)
                        probs["{}_{}__{}".format(n, g, seed)] = predict(bundle, args.kind, x, xf,
                                                                        n_of[(n, g)])
            elif mode == "dflout":
                for g in dd.GAMES:
                    e = dfl_examples(args.kind, g)
                    probs["dfl_{}__{}".format(g, seed)] = predict(bundle, args.kind, e[0][0],
                                                                  e[1][0], len(dfl_data()[g][0]))
            elif mode == "dfllomo":
                e = ex_dfl[k]
                probs["dfl_{}__{}".format(k, seed)] = predict(bundle, args.kind, e[0][0], e[1][0],
                                                              len(dfl_data()[k][0]))
            else:
                os.makedirs(MODELS, exist_ok=True)
                refine.save(bundle, path)
            np.savez(cache, **probs)
            print("  {} {} seed {}: {} stretches, {:.0f} s".format(
                mode, "" if k is None else k, seed, len(train), time.time() - t0), flush=True)


def seeds_mean(probs, prefix):
    ks = sorted(k for k in probs if k.startswith(prefix + "__"))
    return np.mean([probs[k] for k in ks], axis=0) if ks else None


def load_probs(tag):
    if not tag:
        return None
    with np.load(os.path.join(ROOT, "data", "cache", "ext_heads_{}.npz".format(tag))) as c:
        return {k: c[k] for k in c.files}


def rebuild_hand2(hand, pdead, f, blk):
    import eval_dead_v30 as ev
    from eval_dead_refine import constrain
    from pipeline import refine
    h2 = constrain(hand, refine.decode_dead(hand, pdead, blk, *ev.SETTING), ev.MODE)
    protect = np.asarray(f["ball_out"]) > 0
    return ev.remove_gate(h2, pdead, blk, 0.1, 0.0, dd.FPS, protect)


def short_caught(lab, codes):
    import run_v31_external as rx
    short = rx.short_spells(lab)
    pred = pm.possession_spells(codes, pm.valid_runs(lab))
    return len(short), sum(rx.caught(s, pred) for s in short)


def line(name, new, ref, extra=""):
    new, ref = np.asarray(new), np.asarray(ref)
    d = new - ref
    return "{:<14} v31 {:.4f} -> {:.4f} ({:+.4f}, {}/{}){}".format(
        name, ref.mean(), new.mean(), d.mean(), int(np.sum(d > 1e-9)), int(np.sum(d < -1e-9)), extra)


def eval_cmd(args):
    import eval_dead_v30 as ev
    import eval_touch_run as tr
    import run_v31_external as rx
    team, dead = load_probs(args.team), load_probs(args.dead)
    for name in ("skillcorner", "pff"):
        ref, new, sh = [], [], np.zeros(3, int)
        for n, g, _ in ext_list():
            if n != name:
                continue
            pt = seeds_mean(team, "{}_{}".format(n, g)) if team else None
            pd_ = seeds_mean(dead, "{}_{}".format(n, g)) if dead else None
            if (team and pt is None) or (dead and pd_ is None):
                continue
            with np.load(os.path.join(INTER, "v31_{}_{}.npz".format(n, g))) as z:
                c = {k: z[k] for k in z.files}
            f = {k[2:]: v for k, v in c.items() if k.startswith("f_")}
            blk = rx.blocks(c["period"])
            if pd_ is not None and args.blend < 1:
                pd_ = (1 - args.blend) * c["pdead"] + args.blend * pd_
            h2 = c["hand2"] if pd_ is None else rebuild_hand2(c["hand"], pd_, f, blk)
            codes = ev.chain(h2, c["p1"], c["p_ref"] if pt is None else pt, f, blk, c["vote"])
            ref.append(tr.score(c["labels"], c["v31"]))
            new.append(tr.score(c["labels"], codes))
            a, b = short_caught(c["labels"], c["v31"])
            _, b2 = short_caught(c["labels"], codes)
            sh += (a, b, b2)
        print(line(name, new, ref, " | short caught {} -> {} of {}".format(sh[1], sh[2], sh[0])),
              flush=True)
    if not args.skip_dfl and ((team and any(k.startswith("dfl_") for k in team)) or
                              (dead and any(k.startswith("dfl_") for k in dead))):
        with np.load(os.path.join(dd.CACHE, "refine_{}.npz".format(tr.TEAM_TAG))) as cc:
            b2 = {k: cc[k] for k in cc.files}
        with np.load(os.path.join(dd.CACHE, "dead_{}.npz".format(ev.DEAD_TAG))) as cc:
            d1n = {k: cc[k] for k in cc.files}
        ref, new, sh = [], [], np.zeros(3, int)
        for g in dd.GAMES:
            lab, hand, p1, f, blk, cols, _ = dfl_data()[g]
            vote = tr.run_vote(f, ev.VOTE)
            p_ref = np.mean([b2["{}__{}".format(g, s)] for s in range(tr.TEAM_SEEDS)], axis=0)
            pdead = np.mean([d1n["{}__{}".format(g, s)] for s in range(ev.DEAD_SEEDS)], axis=0)
            base = ev.chain(rebuild_hand2(hand, pdead, f, blk), p1, p_ref, f, blk, vote)
            pt = seeds_mean(team, "dfl_" + g) if team else None
            pd_ = seeds_mean(dead, "dfl_" + g) if dead else None
            if pd_ is not None and args.blend < 1:
                pd_ = (1 - args.blend) * pdead + args.blend * pd_
            codes = ev.chain(rebuild_hand2(hand, pdead if pd_ is None else pd_, f, blk), p1,
                             p_ref if pt is None else pt, f, blk, vote)
            ref.append(tr.score(lab, base))
            new.append(tr.score(lab, codes))
            a, b = short_caught(lab, base)
            _, c2 = short_caught(lab, codes)
            sh += (a, b, c2)
        print(line("DFL", new, ref, " | short caught {} -> {} of {}".format(sh[1], sh[2],
                                                                                     sh[0])))
    if args.ours:
        ours(args)


def ours(args):
    import eval_dead_v30 as ev
    import ours_chain as oc
    from pipeline import refine
    from report import brief_line, PUBLIC_GT
    _, ex = oc.ours_v31()
    f = {k[2:]: v for k, v in ex.items() if k.startswith("f_")}
    blk = [tuple(int(x) for x in b) for b in ex["blk"]]
    hand, p1, n = ex["hand"], ex["p1"], len(ex["hand"])
    from games import GAME, half_video_blocks  # noqa: F401
    from atd_cache import load_arrays
    from pipeline.features import LENGTH, WIDTH
    import in_domain_team as idt
    groups = np.zeros(n, dtype=np.int8)
    for i, (a, b) in enumerate(blk, start=1):
        groups[a:b] = i
    atd = load_arrays()
    cols = idt.player_columns(atd["players"], atd["team"], groups, (LENGTH, WIDTH))
    refine.BIN = BIN
    p_ref, pdead, hand2 = ex["p_ref"], ex["pdead"], ex["hand2"]
    if args.team:
        bs = [refine.load(p) for p in sorted(glob.glob(os.path.join(MODELS, args.team + "_s*.pt")))]
        if not bs:
            raise SystemExit("no models {}_s*.pt (train with --modes all)".format(args.team))
        x = (refine.inputs(p1, hand, f, cols), refine.inputs(p1, hand, f, cols, True))
        p_ref = np.mean([predict(b, "team", x[0], x[1], n) for b in bs], axis=0)
    if args.dead:
        bs = [refine.load(p) for p in sorted(glob.glob(os.path.join(MODELS, args.dead + "_s*.pt")))]
        if not bs:
            raise SystemExit("no models {}_s*.pt (train with --modes all)".format(args.dead))
        x = (refine.dead_inputs(p1, hand, f, cols, False, False),
             refine.dead_inputs(p1, hand, f, cols, True, False))
        pdead = np.mean([predict(b, "dead", x[0], x[1], n) for b in bs], axis=0)
        pdead = (1 - args.blend) * ex["pdead"] + args.blend * pdead
        hand2 = rebuild_hand2(hand, pdead, f, blk)
    codes = ev.chain(hand2, p1, p_ref, f, blk, ex["vote"])
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    inside = np.zeros(n, bool)
    for a, b in blk:
        inside[a:b] = True
    print("ours " + brief_line("v31 (variant)", ex["v31"], gt))
    print("ours " + brief_line("new heads", codes, gt) + "  differ {:.2f}%".format(
        100 * np.mean(codes[inside] != ex["v31"][inside])))
    if args.write:
        from games import GAMES
        from pipeline.run import write_submission, RESULTS_DIR
        path = write_submission(os.path.join(RESULTS_DIR, args.write + ".csv"), codes)
        reason = pm.validate_submission(pm.read_label_csv(path), GAMES[GAME]["n_video_frames"])
        print("wrote {} ({})".format(os.path.relpath(path, ROOT), reason or "valid"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("kind", choices=("team", "dead"))
    t.add_argument("--tag", required=True)
    t.add_argument("--seeds", type=int, default=2)
    t.add_argument("--steps", type=int, default=1500)
    t.add_argument("--no-dfl", action="store_true")
    t.add_argument("--modes", nargs="+", default=["folds", "dflout", "all"],
                   choices=("folds", "dflout", "dfllomo", "all"))
    e = sub.add_parser("eval")
    e.add_argument("--team")
    e.add_argument("--dead")
    e.add_argument("--ours", action="store_true")
    e.add_argument("--skip-dfl", action="store_true")
    e.add_argument("--write")
    e.add_argument("--blend", type=float, default=1.0,
                   help="P(dead) = (1 - w) x deployed d1n + w x new head (default 1: new only)")
    args = parser.parse_args()
    if args.cmd == "train":
        train_cmd(args)
    else:
        eval_cmd(args)


if __name__ == "__main__":
    main()
