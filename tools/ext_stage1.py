"""First stage (GBM + GRU blend) retrained with SkillCorner + PFF; the chain re-run on top. Aggregate only.

    python tools/ext_stage1.py train --tag s1 [--short-weight 3] [--kinds gbm gru] [--modes folds all]
    python tools/ext_stage1.py eval --tag s1 [--jobs 4] [--refine TAG]
    python tools/ext_stage1.py ours --tag s1 [--refine TAG] [--write NAME]
    python tools/ext_stage1.py cols [--jobs 3]       # cache the players-only columns first
    python tools/ext_stage1.py oldgru --source s1 --tag s1o   # new GBM + the deployed DFL GRU
    python tools/ext_stage1.py train --tag tx5s1 --src tx5 --no-dfl --gbm xgb   # on features rebuilt with the
        # held-out external touch networks (run_v31_external.py --touch-dir ... --tag tx5 --save)
    python tools/ext_stage1.py ours --tag tx5s1 --touch data/cache/models/touch_ext/touch_all.pt --pick 0.5
GRU training reads float16 inputs from disk (data/cache/dfl/gru_prep_*, ext_v31/gru_*), so it
never holds the feature dicts of all matches at once.

The deployed first stage (models/gbm_dfl + gru_dfl) is trained on DFL only, whose truth merges the
brief contacts the organisers count. Here: same features (learned.matrix), same GBM / GRU, trained
on DFL (run_learned.training("dfl"), every gap pattern) + the external matches.
  folds : for each of the 4 fixed external folds, DFL + the other 3 folds -> held-out P on the fold
  all   : DFL + all 84 -> data/cache/models/ext_stage1/<tag>_gbm.txt, <tag>_gru.pt (our match)
--short-weight w: frames inside true H / A spells shorter than SHORT_S get weight w (GBM rows; the
GRU sees those stretches w times as often as crop starts), the small-touch lever.
Held-out P: data/cache/ext_stage1_<tag>.npz (<set>_<game>__gbm / __gru).

eval: per held-out external match the chain from the first stage on, as tools/run_v31_external.py
(players-only vote with this match's pseudo-labels, stage 1, refine_v29 x5 or --refine TAG from
train_ext_heads, touch-run vote, dead head d1 held-out + protected gate, backdating), against the
working base (same chain with the cached v31 first stage and d1). Prints spellF1, short spells
caught (< 2 s between two of the other team, IoU >= 0.5), and the first stage's mean P(true team)
on frames of short spells and on all live frames.
"""

import argparse
import glob
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from pipeline.decode import H, A, D, X  # noqa: E402

# Params
FOLDS = 4
EXT_STEP = 6                     # GBM rows: every 6th frame of an external match (DFL: 3)
DFL_STEP = 3
SHORT_S = 3.0
DEAD_TAG = "d1"                  # the base's dead-ball head (train_ext_heads, held-out per fold)
INTER = os.path.join(ROOT, "data", "cache", "ext_v31")
MODELS = os.path.join(ROOT, "data", "cache", "models", "ext_stage1")
SRC = "v31"                      # feature source: cached run_v31_external.py --save tag (--src)


def ext_list():
    import run_v31_external as rx
    out = []
    for name in ("skillcorner", "pff"):
        for p in sorted(glob.glob(os.path.join(INTER, "v31_{}_*.npz".format(name)))):
            g = os.path.basename(p)[:-4].split("_", 2)[2]
            out.append((name, g, rx.fold_of(name, g)))
    return out


def load_ext(name, g, keys=None, src=None):
    with np.load(os.path.join(INTER, "{}_{}_{}.npz".format(src or SRC, name, g))) as z:
        c = {k: z[k] for k in z.files if keys is None or k in keys or k.startswith("f_")}
    f = {k[2:]: v for k, v in c.items() if k.startswith("f_")}
    return c, f


def short_mask(lab, short_s=SHORT_S, fps=25):
    """Frames inside true H / A spells shorter than short_s."""
    m = np.zeros(len(lab), bool)
    for a, b, k in pm.possession_spells(lab, pm.valid_runs(lab)):
        if (b - a) / fps < short_s:
            m[a:b] = True
    return m


def gbm_rows(f, lab, step, w_short):
    from pipeline import learned
    x, y = learned.training_rows(f, lab, step=step)
    if w_short == 1:
        return x, y, np.ones(len(y), np.float32)
    sm = short_mask(lab)
    keep = np.flatnonzero(lab != X)[::step]
    w = np.where(sm[keep], w_short, 1.0).astype(np.float32)
    return x, y, np.concatenate([w, w])                 # plain + flipped copy, same frames


def train_gbm_w(rows, rounds=400, seed=0):
    import lightgbm as lgb
    params = {"objective": "multiclass", "num_class": 3, "learning_rate": 0.05,
              "num_leaves": 63, "min_data_in_leaf": 200, "feature_fraction": 0.8,
              "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "seed": seed,
              "verbose": -1}
    xs = np.concatenate([r[0] for r in rows])
    ys = np.concatenate([r[1] for r in rows])
    ws = np.concatenate([r[2] for r in rows])
    return lgb.train(params, lgb.Dataset(xs, ys, weight=ws), num_boost_round=rounds)


class XGB:
    """XGBoost booster with LightGBM's predict(matrix) -> (n, 3) interface (learned.predict_proba)."""

    def __init__(self, booster):
        self.booster = booster

    def predict(self, x):
        return self.booster.inplace_predict(np.asarray(x, dtype=np.float32))

    def save_model(self, path):
        self.booster.save_model(path)


def load_xgb(path):
    import xgboost as xgb
    b = xgb.Booster()
    b.load_model(path)
    b.set_param({"device": "cuda"})
    return XGB(b)


def train_xgb_w(rows, rounds=400, seed=0):
    """GPU XGBoost with the LightGBM settings translated (63 leaves, lr 0.05, 0.8 / 0.8, l2 1)."""
    import xgboost as xgb
    xs = np.concatenate([r[0] for r in rows])
    ys = np.concatenate([r[1] for r in rows])
    ws = np.concatenate([r[2] for r in rows])
    dm = xgb.QuantileDMatrix(xs, ys, weight=ws)
    del xs
    params = {"objective": "multi:softprob", "num_class": 3, "eta": 0.05, "device": "cuda",
              "tree_method": "hist", "grow_policy": "lossguide", "max_leaves": 63, "max_depth": 0,
              "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.8, "lambda": 1.0,
              "seed": seed}
    return XGB(xgb.train(params, dm, num_boost_round=rounds))


def _prep_save(path, f, lab):
    from pipeline.sequence import _prepare
    (x0, y0), (x1, y1) = _prepare(f, lab)
    np.savez(path, x0=x0.astype(np.float16), x1=x1.astype(np.float16), y0=y0, y1=y1)


def gru_prep_paths(no_dfl=False):
    """GRU inputs (sequence._prepare, plain + flipped) as float16 on disk, built once:
    {("dfl", i): path} for the 21 DFL training versions, {(set, game): path} for the external."""
    import dfl_data as dd
    from train_learned_dfl import match_features, test_seed, TRAIN_VERSIONS
    out = {}
    i = 0
    for g in ([] if no_dfl else dd.GAMES):
        lab = dd.load(g)["controlled"]
        for seed in [test_seed(g)] + [test_seed(g) + 1 + v for v in range(TRAIN_VERSIONS)]:
            path = os.path.join(ROOT, "data", "cache", "dfl", "gru_prep_{}_{}.npz".format(g, seed))
            if not os.path.isfile(path):
                _prep_save(path, match_features(g, seed), lab)
            out[("dfl", i)] = path
            i += 1
    for n, g, _ in ext_list():
        path = os.path.join(INTER, ("gru_{}_{}.npz" if SRC == "v31" else
                                    "gru_" + SRC + "_{}_{}.npz").format(n, g))
        if not os.path.isfile(path):
            c, f = load_ext(n, g, ("labels",))
            _prep_save(path, f, c["labels"])
        out[(n, g)] = path
    return out


def train_gru_lean(paths, w_short_of, epochs=None, seed=0):
    """pipeline.sequence.train_gru on float16 matrices from disk, normalised per batch.
    w_short_of(key) > 1 adds copies whose labels are kept only within 1 s of short spells."""
    import torch
    from pipeline import sequence as sq
    epochs = epochs or sq.EPOCHS
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seqs = []
    s1 = s2 = cnt = None
    for key, path in paths:
        with np.load(path) as z:
            for x, y in ((z["x0"], z["y0"]), (z["x1"], z["y1"])):
                xf = x.astype(np.float64)
                ok = ~np.isnan(xf)
                if s1 is None:
                    s1, s2, cnt = np.zeros(x.shape[1]), np.zeros(x.shape[1]), np.zeros(x.shape[1])
                s1 += np.where(ok, xf, 0).sum(0)
                s2 += np.where(ok, xf * xf, 0).sum(0)
                cnt += ok.sum(0)
                del xf, ok
                seqs.append((x, y))
                w = w_short_of(key)
                if w > 1:
                    sm = short_mask(np.repeat(y, sq.STEP), fps=25)[::sq.STEP]
                    grow = np.convolve(sm.astype(float), np.ones(2 * 25 // sq.STEP + 1),
                                       mode="same") > 0
                    y2 = np.where(grow, y, X).astype(y.dtype)
                    seqs += [(x, y2)] * int(round(w - 1))
    mean = (s1 / np.maximum(cnt, 1)).astype(np.float32)
    var = s2 / np.maximum(cnt, 1) - mean.astype(np.float64) ** 2
    std = (np.sqrt(np.maximum(var, 0)) + 1e-6).astype(np.float32)

    def norm(x):
        return np.nan_to_num((x.astype(np.float32) - mean) / std)
    cut = [int(len(x) * (1 - sq.VAL_SHARE)) for x, _ in seqs]
    model = sq._net(len(mean)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=sq.LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=sq.LR,
                                                total_steps=epochs * sq.EPOCH_STEPS)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=X)
    crop = sq.CROP // sq.STEP
    val_idx = sorted(set(range(0, len(seqs), max(1, len(seqs) // 40))))   # ~40 stretches
    best = None
    for ep in range(epochs):
        model.train()
        for _ in range(sq.EPOCH_STEPS):
            xb, yb = [], []
            for _ in range(sq.BATCH):
                k = rng.integers(len(seqs))
                x, y = seqs[k]
                st = rng.integers(0, cut[k] - crop)
                xb.append(norm(x[st:st + crop]))
                yb.append(y[st:st + crop])
            xt = torch.tensor(np.stack(xb), dtype=torch.float32, device=dev)
            xt = xt + sq.NOISE * torch.randn_like(xt)
            yt = torch.tensor(np.stack(yb), dtype=torch.long, device=dev)
            loss = loss_fn(model(xt).reshape(-1, 3), yt.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for k in val_idx:
                x, y = seqs[k]
                logits = sq._windowed(model, norm(x[cut[k]:]), dev)
                yt = torch.tensor(y[cut[k]:], dtype=torch.long, device=dev)
                m = int((yt != X).sum())
                if m:
                    tot += float(loss_fn(logits, yt)) * m
                    n += m
        vloss = tot / max(n, 1)
        if best is None or vloss < best[0]:
            best = (vloss, {k: v.detach().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    return model, mean, std


def train_cmd(args):
    import torch
    from pipeline import learned, run_learned as rl
    from pipeline.sequence import predict_gru
    cache = os.path.join(ROOT, "data", "cache", "ext_stage1_{}.npz".format(args.tag))
    probs = {}
    if os.path.isfile(cache):
        with np.load(cache) as c:
            probs = {k: c[k] for k in c.files}
    ext = ext_list()
    print("external matches {}".format(len(ext)), flush=True)
    jobs = [("fold", k) for k in range(FOLDS)] if "folds" in args.modes else []
    if "all" in args.modes:
        jobs.append(("all", None))
    os.makedirs(MODELS, exist_ok=True)
    if "gbm" in args.kinds:
        t0 = time.time()
        dfl_rows = [] if args.no_dfl else [gbm_rows(f, lab, DFL_STEP, 1.0)
                                           for f, lab in rl.training("dfl")]
        ext_rows = {}
        for n, g, _ in ext:
            c, f = load_ext(n, g, ("labels",))
            ext_rows[(n, g)] = gbm_rows(f, c["labels"], EXT_STEP, args.short_weight)
        print("GBM rows built in {:.0f} s".format(time.time() - t0), flush=True)
        for mode, k in jobs:
            path = os.path.join(MODELS, "{}_gbm.{}".format(args.tag,
                                                          "json" if args.gbm == "xgb" else "txt"))
            held = [(n, g) for n, g, fo in ext if mode == "fold" and fo == k]
            if mode == "fold" and all("{}_{}__gbm".format(n, g) in probs for n, g in held):
                continue
            if mode == "all" and os.path.isfile(path):
                continue
            t0 = time.time()
            rows = dfl_rows + [ext_rows[(n, g)] for n, g, fo in ext if mode == "all" or fo != k]
            model = train_xgb_w(rows) if args.gbm == "xgb" else train_gbm_w(rows)
            if mode == "all":
                model.save_model(path)
            for n, g in held:
                _, f = load_ext(n, g, ())
                probs["{}_{}__gbm".format(n, g)] = learned.predict_proba(model, f).astype(np.float16)
            np.savez(cache, **probs)
            print("  GBM {} {}: {:.0f} s".format(mode, "" if k is None else k, time.time() - t0),
                  flush=True)
        del dfl_rows, ext_rows
    if "gru" in args.kinds:
        t0 = time.time()
        prep = gru_prep_paths(args.no_dfl)
        print("GRU inputs ready ({:.0f} s)".format(time.time() - t0), flush=True)
        folds = {(n, g): fo for n, g, fo in ext}
        for mode, k in jobs:
            path = os.path.join(MODELS, "{}_gru.pt".format(args.tag))
            held = [(n, g) for n, g, fo in ext if mode == "fold" and fo == k]
            if mode == "fold" and all("{}_{}__gru".format(n, g) in probs for n, g in held):
                continue
            if mode == "all" and os.path.isfile(path):
                continue
            t0 = time.time()
            use = [(key, p) for key, p in prep.items()
                   if (key[0] == "dfl" and not args.no_dfl) or
                   (key[0] != "dfl" and (mode == "all" or folds[key] != k))]
            model, mean, std = train_gru_lean(
                use, lambda key: 1.0 if key[0] == "dfl" else args.short_weight)
            if mode == "all":
                torch.save({"state": model.state_dict(), "mean": mean, "std": std,
                            "n_in": len(mean)}, path)
            for n, g in held:
                _, f = load_ext(n, g, ())
                probs["{}_{}__gru".format(n, g)] = predict_gru((model, mean, std), f).astype(np.float16)
            np.savez(cache, **probs)
            print("  GRU {} {}: {:.0f} s".format(mode, "" if k is None else k, time.time() - t0),
                  flush=True)


def oldgru_cmd(args):
    """The deployed DFL-only GRU on every external match, next to the GBM of --source: tag --tag."""
    from pipeline import run_learned as rl
    from pipeline.sequence import predict_gru
    gru = rl.models("dfl", ("gru",))["gru"]
    with np.load(os.path.join(ROOT, "data", "cache", "ext_stage1_{}.npz".format(args.source))) as c:
        probs = {k: c[k] for k in c.files if k.endswith("__gbm")}
    for n, g, _ in ext_list():
        _, f = load_ext(n, g, ())
        probs["{}_{}__gru".format(n, g)] = predict_gru(gru, f).astype(np.float16)
    np.savez(os.path.join(ROOT, "data", "cache", "ext_stage1_{}.npz".format(args.tag)), **probs)
    print("old GRU predicted on {} matches -> tag {}".format(len(ext_list()), args.tag))


def cols_for(name, g):
    """Players-only columns of an external match, cached in data/cache/ext_v31/cols_<set>_<g>.npz."""
    import in_domain_team as idt
    import run_v31_external as rx
    path = os.path.join(INTER, "cols_{}_{}.npz".format(name, g))
    if os.path.isfile(path):
        with np.load(path) as z:
            return {k: z[k].astype(np.float32) for k in z.files}
    d, pitch, _ = rx.load_set(name, g)
    cols = idt.player_columns(d["players"], d["team"], d["period"].astype(np.int8), pitch)
    np.savez(path, **{k: v.astype(np.float16) for k, v in cols.items()})
    return {k: v.astype(np.float32) for k, v in cols.items()}


def chain_from_prob(prob, c, f, cols, refine_bundles, pdead):
    """v31's chain from the first-stage blend on (as run_v31_external.chain), with pdead given."""
    import eval_dead_v30 as ev
    import in_domain_team as idt
    import run_v31_external as rx
    import train_ext_heads as th
    from eval_refine import stage1
    from pipeline import learned, refine
    m = rx.deployed()
    hand = c["hand"]
    blk = rx.blocks(c["period"])
    team, conf, weak = idt.pseudo_labels(f, hand)
    dteam, dmask = idt.decoded_pseudo(learned.decode_hybrid(hand, prob, blk, rx.V25_SWITCH), prob)
    merged = np.where(conf, team, dteam)
    po = idt.train_po_multi([(cols, merged, conf | dmask)] + m["dfl_sets"])
    prob = idt.combine(prob, idt.predict_po(po, cols), weak, rx.W_PO)
    p1 = stage1(prob, f)
    refine.BIN = refine_bundles[0][0].cfg["BIN"]
    n = len(hand)
    xs = (refine.inputs(p1, hand, f, cols), refine.inputs(p1, hand, f, cols, True))
    p_ref = np.mean([refine.predict(b, xs[0], xs[1], n) for b in refine_bundles], axis=0)
    h2 = th.rebuild_hand2(hand, pdead, f, blk)
    return ev.chain(h2, p1, p_ref, f, blk, c["vote"]), p1


def p_true(p1, lab, mask):
    """Mean P(true team) of the first stage on mask & true live frames (H / A renormalised)."""
    m = mask & np.isin(lab, (H, A))
    if not m.any():
        return np.nan, 0
    ph = np.clip(p1[m, H], 1e-6, 1)
    pa = np.clip(p1[m, A], 1e-6, 1)
    pt = np.where(lab[m] == H, ph, pa) / (ph + pa)
    return float(pt.mean()), int(m.sum())


def length_caught(lab, codes, classes=((0, 2), (2, 5))):
    """Per length class (s): (true possessions, caught = a predicted spell of that team with
    IoU >= 0.5)."""
    import run_v31_external as rx
    runs = pm.valid_runs(lab)
    truth = pm.possession_spells(lab, runs)
    pred = pm.possession_spells(pm.fill_abstentions(codes, runs), runs)
    out = []
    for lo, hi in classes:
        sel = [t for t in truth if lo <= (t[1] - t[0]) / 25.0 < hi]
        out.append((len(sel), sum(rx.caught(t, pred) for t in sel)))
    return out


_W = {}


def eval_match(job):
    name, g, tag, refine_tag, src = job
    global SRC
    SRC = src
    import eval_dead_v30 as ev
    import eval_touch_run as tr
    import run_v31_external as rx
    import train_ext_heads as th
    from pipeline import refine
    if "refine" not in _W:
        if refine_tag:
            _W["refine"] = [refine.load(p) for p in sorted(glob.glob(os.path.join(
                th.MODELS, refine_tag + "_s*.pt")))]
        else:
            _W["refine"] = [refine.load(os.path.join(ROOT, "models", "refine_v29_s{}.pt".format(s)))
                            for s in range(5)]
        with np.load(os.path.join(ROOT, "data", "cache", "ext_heads_{}.npz".format(DEAD_TAG))) as z:
            _W["dead"] = {k: z[k] for k in z.files}
        with np.load(os.path.join(ROOT, "data", "cache", "ext_stage1_{}.npz".format(tag))) as z:
            _W["s1"] = {k: z[k] for k in z.files}
        if refine_tag:
            with np.load(os.path.join(ROOT, "data", "cache", "ext_heads_{}.npz".format(
                    refine_tag))) as z:
                _W["ref_p"] = {k: z[k] for k in z.files}
    key = "{}_{}".format(name, g)
    if key + "__gbm" not in _W["s1"] or key + "__gru" not in _W["s1"]:
        return None
    c, f = load_ext(name, g, src="v31")
    lab = c["labels"]
    blk = rx.blocks(c["period"])
    pdead = th.seeds_mean(_W["dead"], key)
    base = ev.chain(th.rebuild_hand2(c["hand"], pdead, f, blk), c["p1"], c["p_ref"], f, blk,
                    c["vote"])
    base_p1 = c["p1"]
    if SRC != "v31":
        c, f = load_ext(name, g)
    prob = 0.5 * _W["s1"][key + "__gbm"].astype(np.float64) + \
        0.5 * _W["s1"][key + "__gru"].astype(np.float64)
    cols = cols_for(name, g)
    codes, p1 = chain_from_prob(prob, c, f, cols, _W["refine"], pdead)
    sm = short_mask(lab, 2.0)
    n_sh, b_sh = th.short_caught(lab, base)
    _, c_sh = th.short_caught(lab, codes)
    lc = (length_caught(lab, base), length_caught(lab, codes))
    return (name, tr.score(lab, base), tr.score(lab, codes), n_sh, b_sh, c_sh,
            p_true(base_p1, lab, sm), p_true(p1, lab, sm),
            p_true(base_p1, lab, np.ones(len(lab), bool)), p_true(p1, lab, np.ones(len(lab), bool)),
            lc)


def _cols_job(j):
    cols_for(j[0], j[1])
    return 1


def cols_cmd(args):
    with Pool(args.jobs) as pool:
        print("players-only columns cached for {} matches".format(
            sum(pool.map(_cols_job, ext_list()))))


def eval_cmd(args):
    jobs = [(n, g, args.tag, args.refine, SRC) for n, g, _ in ext_list()
            if os.path.isfile(os.path.join(INTER, "{}_{}_{}.npz".format(SRC, n, g)))]
    with Pool(args.jobs) as pool:
        res = [r for r in pool.map(eval_match, jobs) if r is not None]
    for name in ("skillcorner", "pff"):
        r = [x for x in res if x[0] == name]
        if not r:
            continue
        b = np.array([x[1] for x in r])
        s = np.array([x[2] for x in r])
        d = s - b
        sh = np.sum([x[3:6] for x in r], axis=0)

        def wmean(i, j):
            v = np.array([x[i][0] for x in r])
            w = np.array([x[i][1] for x in r])
            ok = ~np.isnan(v)
            return np.sum(v[ok] * w[ok]) / max(np.sum(w[ok]), 1)
        print("{:<12} ({} matches) base {:.4f} -> {:.4f} ({:+.4f}, {}/{}) | short caught {} -> {} of "
              "{} | P(true team) short {:.3f} -> {:.3f}, all live {:.3f} -> {:.3f}".format(
                  name, len(r), b.mean(), s.mean(), d.mean(), int(np.sum(d > 1e-9)),
                  int(np.sum(d < -1e-9)), sh[1], sh[2], sh[0], wmean(6, None), wmean(7, None),
                  wmean(8, None), wmean(9, None)), flush=True)
        lc = np.sum([[[c[0], c[1]] for c in x[10][0]] + [[c[0], c[1]] for c in x[10][1]]
                     for x in r], axis=0)            # rows: base <2, base 2-5, new <2, new 2-5
        print("{:<12} caught (IoU >= 0.5): < 2 s {} -> {} of {} ({:.0%} -> {:.0%}) | 2-5 s {} -> {} of {} "
              "({:.0%} -> {:.0%})".format(name, lc[0][1], lc[2][1], lc[0][0], lc[0][1] / max(lc[0][0], 1),
                                         lc[2][1] / max(lc[0][0], 1), lc[1][1], lc[3][1], lc[1][0],
                                         lc[1][1] / max(lc[1][0], 1), lc[3][1] / max(lc[1][0], 1)),
              flush=True)


def ours_cmd(args):
    import torch
    import lightgbm as lgb
    import ours_chain as oc
    from pipeline.sequence import _net
    from report import brief_line, PUBLIC_GT
    xgb_path = os.path.join(MODELS, "{}_gbm.json".format(args.tag))
    gbm = load_xgb(xgb_path) if os.path.isfile(xgb_path) else \
        lgb.Booster(model_file=os.path.join(MODELS, "{}_gbm.txt".format(args.tag)))
    saved = torch.load(os.path.join(MODELS, "{}_gru.pt".format(args.tag)), weights_only=False,
                       map_location=None if torch.cuda.is_available() else "cpu")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = _net(saved["n_in"]).to(dev)
    net.load_state_dict(saved["state"])
    learned_models = {"gbm": gbm, "gru": (net, saved["mean"], saved["std"])}
    refine_prefix = None
    if args.refine:
        import train_ext_heads as th
        refine_prefix = os.path.relpath(os.path.join(th.MODELS, args.refine), ROOT)
    touch_path = None
    if args.touch:
        from pipeline import touch_net
        touch_path = os.path.join(ROOT, args.touch)
        if args.pick:
            touch_net.PICK_THRESHOLD = args.pick
    codes, ex = oc.ours_v31(touch_path, dead_prefix="models/dead_ext_d1",
                            learned_models=learned_models, refine_prefix=refine_prefix)
    if args.touch:
        from pipeline import touch_net
        touch_net.PICK_THRESHOLD = 0.7
    base, _ = oc.ours_v31(dead_prefix="models/dead_ext_d1")
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    blk = [tuple(int(x) for x in b) for b in ex["blk"]]
    inside = np.zeros(len(gt), bool)
    for a, b in blk:
        inside[a:b] = True
    print("ours " + brief_line("base (v31_d1)", base, gt))
    print("ours " + brief_line(args.tag, codes, gt) + "  differ {:.2f}%".format(
        100 * np.mean(codes[inside] != base[inside])))
    if args.write:
        print(oc.write_and_score(codes, args.write))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--tag", required=True)
    t.add_argument("--short-weight", type=float, default=1.0)
    t.add_argument("--kinds", nargs="+", default=["gbm", "gru"], choices=("gbm", "gru"))
    t.add_argument("--modes", nargs="+", default=["folds", "all"], choices=("folds", "all"))
    t.add_argument("--no-dfl", action="store_true", help="external matches only")
    t.add_argument("--gbm", choices=("lgb", "xgb"), default="lgb",
                   help="xgb: XGBoost on the GPU (same settings; ~10x faster here)")
    e = sub.add_parser("eval")
    e.add_argument("--tag", required=True)
    e.add_argument("--jobs", type=int, default=4)
    e.add_argument("--refine", help="team refiner tag from train_ext_heads (default refine_v29)")
    og = sub.add_parser("oldgru")
    og.add_argument("--source", required=True)
    og.add_argument("--tag", required=True)
    cl = sub.add_parser("cols")
    cl.add_argument("--jobs", type=int, default=3)
    o = sub.add_parser("ours")
    o.add_argument("--tag", required=True)
    o.add_argument("--refine")
    o.add_argument("--write")
    o.add_argument("--touch", help="touch network for our match, e.g. data/cache/models/touch_ext/touch_all.pt")
    o.add_argument("--pick", type=float, help="touch pick threshold with --touch")
    for q in (t, e, o):
        q.add_argument("--src", default="v31", help="feature source tag (run_v31_external --save)")
    args = parser.parse_args()
    global SRC
    SRC = getattr(args, "src", "v31")
    {"train": train_cmd, "eval": eval_cmd, "ours": ours_cmd, "cols": cols_cmd, "oldgru": oldgru_cmd}[args.cmd](args)


if __name__ == "__main__":
    main()
