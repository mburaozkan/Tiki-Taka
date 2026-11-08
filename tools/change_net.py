"""Dense change network: every change of state, by type, per frame, on all possessions. Aggregate only.

    python tools/change_net.py train [--tag c1] [--folds 0 1 2 3 all] [--epochs 12] [--seed 0]
    python tools/change_net.py eval [--tag c1]

The challenge's question is when possession changes hands. This network answers it directly instead
of labelling frames and patching short spells afterwards.
Inputs: the short net's per-frame inputs (tools/short_net.py prep, data/cache/short_net/in_*.npy:
ball, both teams' geometry, touch-network P, first-stage P, refiner, vote, hand D, dead head).
Targets per frame (Gaussian, sigma 2 frames, at the first frame of the new state): turnover to H
(A -> H), turnover to A (H -> A), stoppage (H / A -> D), restart to H (D -> H), restart to A
(D -> A); and the frame class (H / A / D) as a second head. Ungraded frames masked. Same model as
the short net (dilated convolutions + bidirectional GRU), random 20 s crops, H/A swap
augmentation, 4 fixed folds + one model on all 84.
eval: decode = Viterbi over H / A / D with emissions w * log P(class) and, for each change i -> j at
frame t, the log of k * P(change i -> j at t) (+ bias); staying is free. Grid of (w, bias); spellF1
vs the best candidate (v31_d1 + e3 short net + check, tools/miss_profile.py) per match, up / down,
nested 4-fold; change recall / precision within 0.2 / 0.5 / 1 / 2 s; short catches.
"""

import argparse
import itertools
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("TC_ROWS", "v2")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
import short_net as S  # noqa: E402
from pipeline.decode import H, A, D  # noqa: E402

FPS = 25
SIGMA = 2.0
CROP = 500
N_TGT = 5                 # to H, to A, stop, restart H, restart A
SWAP_T = [1, 0, 2, 4, 3]
OUT = os.path.join(ROOT, "data", "cache", "change_net")
MODELS = os.path.join(OUT, "models")
HID = int(os.environ.get("CN_HID", "128"))
GRID = {"w": (0.5, 1.0, 2.0), "bias": (-3.0, -1.5, 0.0)}
K = 0.2                   # a sigma-2 Gaussian spreads one change over ~5 frames


def targets_from_cls(cls):
    """(n, 5) float32 change targets from the frame class (0 H, 1 A, 2 D, -1 ungraded)."""
    n = len(cls)
    y = np.zeros((n, N_TGT), dtype=np.float32)
    k = int(3 * SIGMA)
    g = np.exp(-0.5 * (np.arange(-k, k + 1) / SIGMA) ** 2).astype(np.float32)
    p, q = cls[:-1], cls[1:]
    for t in np.flatnonzero((p != q) & (p >= 0) & (q >= 0)) + 1:
        a, b = cls[t - 1], cls[t]
        col = {(1, 0): 0, (0, 1): 1, (0, 2): 2, (1, 2): 2, (2, 0): 3, (2, 1): 4}[(int(a), int(b))]
        lo, hi = max(t - k, 0), min(t + k + 1, n)
        y[lo:hi, col] = np.maximum(y[lo:hi, col], g[lo - t + k:hi - t + k])
    return y


TOL = int(os.environ.get("CN_TOL", "0"))         # > 0: forgiving change target, +-TOL frames
DFL_W = float(os.environ.get("CN_DFL", "0"))     # > 0: DFL's 7 matches join training, sampled x this


X_DIR = os.environ.get("CN_X", "")             # e.g. xb: inputs with a ball in the gaps (cn_ballfill.py)


def load_x(name, g):
    if X_DIR:
        return S.drop_channels(np.load(os.path.join(OUT, X_DIR, "{}_{}.npy".format(name, g))))
    if name == "dfl":
        with np.load(os.path.join(OUT, "dfl_{}.npz".format(g))) as z:
            return S.drop_channels(z["x"])
    return S.drop_channels(np.load(os.path.join(S.OUT, "in_{}_{}.npy".format(name, g))))


def load_cls(name, g):
    if name == "dfl":
        with np.load(os.path.join(OUT, "dfl_{}.npz".format(g))) as z:
            lab = z["lab"]
        return np.where(lab <= 2, lab, -1).astype(np.int8)
    return np.load(os.path.join(S.OUT, "c_{}_{}.npy".format(name, g)))


def train_list():
    import turnover_cls as T
    import dfl_data as dd
    ext = list(T.ext_list())
    if DFL_W > 0:
        ext += [("dfl", g, i % 4) for i, g in enumerate(dd.GAMES)]
    return ext


def load_data(keys):
    xs, ys, cs = [], [], []
    for name, g in keys:
        xs.append(load_x(name, g))
        c = load_cls(name, g)
        cs.append(c)
        ys.append(targets_from_cls(c))
    return xs, ys, cs


def train_one(keys, epochs, seed=0, log=""):
    import torch
    S.N_TGT = N_TGT                     # make_model / predict read the head size from short_net
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    xs, ys, cs = load_data(keys)
    mu, sd = S.norm_stats(xs)
    perm, sign = S.swap_index()
    dev = "cuda"
    model = S.make_model(xs[0].shape[1], HID).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = sum(len(x) for x in xs) // (CROP * 32)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * steps)
    pos_w = torch.tensor([20.0] * N_TGT, device=dev)
    lens = np.array([len(x) for x in xs], dtype=np.float64)
    lens *= np.array([DFL_W if n == "dfl" else 1.0 for n, _ in keys])
    pm_ = lens / lens.sum()
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for _ in range(steps):
            bx, by, bc = [], [], []
            for _ in range(32):
                m = rng.choice(len(xs), p=pm_)
                a = int(rng.integers(0, len(xs[m]) - CROP))
                xb = (xs[m][a:a + CROP].astype(np.float32) - mu) / sd
                yb = ys[m][a:a + CROP]
                cb = cs[m][a:a + CROP].astype(np.int64)
                if rng.random() < 0.5:
                    xb = ((xb * sd + mu)[:, perm] * sign - mu) / sd
                    yb = yb[:, SWAP_T]
                    cb = np.where(cb == 0, 1, np.where(cb == 1, 0, cb))
                bx.append(xb)
                by.append(yb)
                bc.append(cb)
            xb = torch.from_numpy(np.stack(bx)).to(dev)
            yb = torch.from_numpy(np.stack(by)).to(dev)
            cb = torch.from_numpy(np.stack(bc)).to(dev)
            out = model(xb)
            mask = (cb >= 0).float()
            if TOL > 0:
                # forgiving target: a peak anywhere within +-TOL frames of a change counts
                lg = out[..., :N_TGT].transpose(1, 2)                       # (b, 5, T)
                yt = yb.transpose(1, 2)
                ctr = (yt >= 0.999).float()
                k = 2 * TOL + 1
                pooled = torch.nn.functional.max_pool1d(lg, k, 1, TOL)
                near = torch.nn.functional.max_pool1d(ctr, k, 1, TOL)
                lp = torch.nn.functional.binary_cross_entropy_with_logits(pooled, torch.ones_like(pooled),
                                                                         reduction="none") * ctr * 20.0
                ln = torch.nn.functional.binary_cross_entropy_with_logits(lg, torch.zeros_like(lg),
                                                                         reduction="none") * (1 - near)
                bce = (lp + ln).transpose(1, 2).mean(-1)
            else:
                bce = torch.nn.functional.binary_cross_entropy_with_logits(
                    out[..., :N_TGT], yb, pos_weight=pos_w, reduction="none").mean(-1)
            ce = torch.nn.functional.cross_entropy(out[..., N_TGT:].reshape(-1, 3),
                                                   cb.clamp(min=0).reshape(-1),
                                                   reduction="none").reshape(cb.shape)
            loss = ((bce + 0.3 * ce) * mask).sum() / mask.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
        print("  {} epoch {} loss {:.4f} ({:.0f} s)".format(log, ep + 1, tot / steps, time.time() - t0),
              flush=True)
    return model, mu, sd


def train_cmd(args):
    import torch
    import turnover_cls as T
    S.N_TGT = N_TGT
    ext = train_list()
    os.makedirs(MODELS, exist_ok=True)
    pred_path = os.path.join(OUT, "pred_{}.npz".format(args.tag))
    preds = dict(np.load(pred_path)) if os.path.isfile(pred_path) else {}
    for fold in args.folds:
        if fold == "all":
            model, mu, sd = train_one([(n, g) for n, g, _ in ext], args.epochs, 100 * args.seed, "all")
            torch.save({"state": model.state_dict(), "mu": mu, "sd": sd, "n_in": len(mu), "hid": HID},
                       os.path.join(MODELS, "{}_all.pt".format(args.tag)))
            continue
        if fold.startswith("hold:"):                 # leave one provider out
            prov = fold[5:]
            model, mu, sd = train_one([(n, g) for n, g, _ in ext if n != prov], args.epochs,
                                      7 + 100 * args.seed, "without " + prov)
            for n, g, _ in ext:
                if n == prov:
                    preds["{}_{}".format(n, g)] = S.predict(model, mu, sd, load_x(n, g)).astype(np.float16)
            np.savez(pred_path, **preds)
            print("{} held out, {} matches predicted".format(prov, len(preds)), flush=True)
            continue
        k = int(fold)
        model, mu, sd = train_one([(n, g) for n, g, f in ext if f != k], args.epochs,
                                  k + 100 * args.seed, "fold %d" % k)
        for n, g, f in ext:
            if f == k:
                preds["{}_{}".format(n, g)] = S.predict(model, mu, sd, load_x(n, g)).astype(np.float16)
        np.savez(pred_path, **preds)
        print("fold {} done, {} matches predicted".format(k, len(preds)), flush=True)


# ---------------------------------------------------------------- decode / eval

def viterbi(pr, blk, w, bias, dmask=None, tbias=None):
    """codes (H / A / D) from the change net's output (n, 8): 5 change heads + class P.
    dmask (n,) bool: if given, D exactly there (the team decided by the net elsewhere)."""
    n = len(pr)
    em = w * np.log(np.clip(pr[:, N_TGT:], 1e-6, 1.0))              # (n, 3) H, A, D
    if dmask is not None:
        em[dmask, :2] = -1e4
        em[~dmask, 2] = -1e4
        pr = pr.copy()
        pr[:, 2:N_TGT] = 0.5 / K                                     # stop / restart free: D is fixed
    ch = np.log(np.clip(K * pr[:, :N_TGT], 1e-8, 0.5)) + bias
    if tbias is not None:                                            # own bias for turnovers
        ch[:, :2] += tbias - bias
    # trans[t, i, j]: log weight of being in i at t-1 and j at t
    tr = np.zeros((n, 3, 3), dtype=np.float64)
    tr[:, 1, 0] = ch[:, 0]
    tr[:, 0, 1] = ch[:, 1]
    tr[:, 0, 2] = ch[:, 2]
    tr[:, 1, 2] = ch[:, 2]
    tr[:, 2, 0] = ch[:, 3]
    tr[:, 2, 1] = ch[:, 4]
    codes = np.full(n, D, dtype=np.int8)
    lab = np.array([H, A, D], dtype=np.int8)
    for a, b in blk:
        m = b - a
        back = np.zeros((m, 3), dtype=np.int8)
        v = em[a].astype(np.float64).copy()
        for t in range(1, m):
            s = v[:, None] + tr[a + t]
            j = s.argmax(axis=0)
            back[t] = j
            v = s[j, (0, 1, 2)] + em[a + t]
        st = int(v.argmax())
        out = np.empty(m, dtype=np.int8)
        for t in range(m - 1, -1, -1):
            out[t] = st
            st = back[t, st]
        codes[a:b] = lab[out]
    return codes


def eval_match(job):
    name, g, tag, settings = job
    import run_v31_external as rx
    import miss_profile as M
    import eval_touch_run as tr
    from decoder_sweep import stats
    key = "{}_{}".format(name, g)
    with np.load(os.path.join(OUT, "pred_{}.npz".format(tag))) as z:
        if key not in z.files:
            return None
        pr = z[key].astype(np.float32)
    c, f, lab, base, cand, _ = M.candidate((name, g, "e3", 0.6, "k1", 0.3))
    blk = rx.blocks(c["period"])
    runs = pm.valid_runs(lab)
    res, bnd = [stats(lab, cand)], [M.boundaries(lab, pm.fill_abstentions(cand, runs), runs)]
    dm = pm.fill_abstentions(cand, runs) == D
    for st in settings:
        out = viterbi(pr, blk, st["w"], st["bias"], dm if st.get("dfix") else None)
        res.append(stats(lab, out))
        bnd.append(M.boundaries(lab, pm.fill_abstentions(out, runs), runs))
    _ = tr
    return name, g, np.array(res, float), bnd


def eval_cmd(args):
    import run_v31_external as rx
    import turnover_cls as T
    settings = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    if args.dfix:
        settings = [dict(s, dfix=1) for s in settings]
    jobs = [(n, g, args.tag, settings) for n, g, _ in T.ext_list()]
    with Pool(6) as pool:
        res = [r for r in pool.map(eval_match, jobs, chunksize=1) if r is not None]
    names = np.array([r[0] for r in res])
    folds = np.array([rx.fold_of(r[0], r[1]) for r in res])
    arr = np.stack([r[2] for r in res])
    gain = arr[:, 1:, 0] - arr[:, :1, 0]
    labels = ["candidate e3k"] + ["viterbi w {:g} bias {:g}{}".format(
        s["w"], s["bias"], ", D fixed" if s.get("dfix") else "") for s in settings]
    print("change net {}: {} matches".format(args.tag, len(res)))
    print("{:<26} {:>17} {:>17} | {:>10} {:>10} | {:>9} {:>9} | turnover recall 0.5 / 1 s".format(
        "setting", "SkillCorner", "PFF", "caught <2s", "2-5s", "pred/true", "precision"))
    for j, lb in enumerate(labels):
        parts = []
        for nm in ("skillcorner", "pff"):
            v = arr[names == nm, j, 0]
            d = v - arr[names == nm, 0, 0]
            parts.append("{:.4f} {:>3}/{:<3}".format(v.mean(), int(np.sum(d > 1e-9)), int(np.sum(d < -1e-9))))
        t = arr[:, j, 1:].sum(axis=0)
        bt = [r[3][j][0] for r in res]
        nt = sum(x[0] for x in bt)
        ht = np.sum([x[1] for x in bt], axis=0)
        print("{:<26} {:>17} {:>17} | {:>9.1%} {:>9.1%}  | {:>9.2f} {:>9.1%} | {:.0%} / {:.0%}".format(
            lb, parts[0], parts[1], t[1] / t[0], t[3] / t[2], t[5] / t[4], t[6] / t[5],
            ht[1] / nt, ht[2] / nt))
    nested = np.zeros(len(arr))
    for k in range(T.FOLDS):
        j = int(np.argmax(gain[folds != k].mean(axis=0)))
        nested[folds == k] = gain[folds == k, j]
    print("nested 4-fold vs candidate: SkillCorner {:+.4f}, PFF {:+.4f} ({}/{})".format(
        nested[names == "skillcorner"].mean(), nested[names == "pff"].mean(),
        int(np.sum(nested > 1e-9)), int(np.sum(nested < -1e-9))))
    j = 1 + int(np.argmax(gain.mean(axis=0)))
    print("\nchanges, recall / precision within 0.2 / 0.5 / 1 / 2 s (both sets): candidate vs best "
          "setting ({})".format(labels[j]))
    for ty, tn in enumerate(("turnover", "stoppage", "restart")):
        for jj in (0, j):
            bt = [r[3][jj][ty] for r in res]
            nt, ht = sum(x[0] for x in bt), np.sum([x[1] for x in bt], axis=0)
            npr, hp = sum(x[2] for x in bt), np.sum([x[3] for x in bt], axis=0)
            print("  {:<9} {:<14} true {:>5} pred {:>5} | recall {} | precision {}".format(
                tn, "candidate" if jj == 0 else "change net", nt, npr,
                " ".join("{:.0%}".format(h / max(nt, 1)) for h in ht),
                " ".join("{:.0%}".format(h / max(npr, 1)) for h in hp)))


# ---------------------------------------------------------------- hybrid: the chain stays, the net edits it

HYB = {"s": (0.0, 0.3, 0.6, 0.9), "b": (0.0, 0.5, 1.0)}


def hyb_inputs(pr, s, b, switch=20.0):
    """(disc (n, 2), boost (n,)): the turnover heads lower the switch cost into H / into A by
    s x switch x peak (max over +/- 3 frames, capped at switch - 1); the class head adds
    b x log(P(H) / P(A)) home log-odds."""
    from scipy.ndimage import maximum_filter1d
    disc = None
    if s > 0:
        hd = maximum_filter1d(pr[:, :2].astype(np.float64), size=7, axis=0)
        disc = np.minimum(s * switch * hd, switch - 1.0)
    boost = None
    if b > 0:
        boost = b * np.log(np.clip(pr[:, N_TGT], 1e-4, 1) / np.clip(pr[:, N_TGT + 1], 1e-4, 1))
    return disc, boost


def hyb_match(job):
    name, g, tag, settings = job
    import miss_profile as M
    from decoder_sweep import stats
    key = "{}_{}".format(name, g)
    with np.load(os.path.join(OUT, "pred_{}.npz".format(tag))) as z:
        pr = z[key].astype(np.float32)
    cj = (name, g, "e3", 0.6, "k1", 0.3)
    res = []
    for st in settings:
        disc, boost = hyb_inputs(pr, st["s"], st["b"])
        _, _, lab, _, out, _ = M.candidate(cj, disc, boost)
        res.append(stats(lab, out))
    return name, g, np.array(res, float)


def hyb_cmd(args):
    import run_v31_external as rx
    import turnover_cls as T
    settings = [dict(zip(HYB, v)) for v in itertools.product(*HYB.values())]
    jobs = [(n, g, args.tag, settings) for n, g, _ in T.ext_list()]
    with Pool(6) as pool:
        res = pool.map(hyb_match, jobs, chunksize=1)
    names = np.array([r[0] for r in res])
    folds = np.array([rx.fold_of(r[0], r[1]) for r in res])
    arr = np.stack([r[2] for r in res])
    gain = arr[:, 1:, 0] - arr[:, :1, 0]
    print("hybrid: candidate e3k with the change net {} as switch discount (s) and team boost (b)".format(args.tag))
    for j, st in enumerate(settings):
        parts = []
        for nm in ("skillcorner", "pff"):
            v = arr[names == nm, j, 0]
            d = v - arr[names == nm, 0, 0]
            parts.append("{:.4f} {:>3}/{:<3}".format(v.mean(), int(np.sum(d > 1e-9)), int(np.sum(d < -1e-9))))
        t = arr[:, j, 1:].sum(axis=0)
        print("  s {:.1f} b {:.1f}  SC {}  PFF {} | caught <2s {:.1%} 2-5s {:.1%} | pred/true {:.2f} prec {:.1%}".format(
            st["s"], st["b"], parts[0], parts[1], t[1] / t[0], t[3] / t[2], t[5] / t[4], t[6] / t[5]))
    nested = np.zeros(len(arr))
    for k in range(T.FOLDS):
        j = int(np.argmax(gain[folds != k].mean(axis=0)))
        nested[folds == k] = gain[folds == k, j]
    print("nested 4-fold vs candidate: SkillCorner {:+.4f}, PFF {:+.4f} ({}/{})".format(
        nested[names == "skillcorner"].mean(), nested[names == "pff"].mean(),
        int(np.sum(nested > 1e-9)), int(np.sum(nested < -1e-9))))


# ---------------------------------------------------------------- mix: the decoder sees the base and the net

MIX = {"alpha": (0.3, 0.5, 0.7), "beta": (0.0, 0.5, 1.0), "dfix": (0, 1)}
if os.environ.get("CN_MIX"):
    MIX = {"alpha": tuple(float(v) for v in os.environ["CN_MIX"].split(",")), "beta": (0.0,), "dfix": (0,)}


def mix_pr(pr, base, blk, alpha, beta):
    """pr with the class head blended with the base's codes (weight alpha) and the base's own
    changes added to the change heads (max with beta x a sigma-2 bump at each base change)."""
    bf = pm.fill_abstentions(base, [tuple(b) for b in blk])
    cls = np.full(len(bf), -1, np.int8)
    cls[bf == H], cls[bf == A], cls[bf == D] = 0, 1, 2
    inb = np.zeros(len(bf), bool)
    for a, b in blk:
        inb[a:b] = True
    cls[~inb] = -1
    q = pr.copy()
    oh = np.zeros((len(bf), 3), np.float32)
    oh[cls >= 0, cls[cls >= 0]] = 1.0
    oh = 0.9 * oh + 0.1 / 3
    q[:, N_TGT:] = np.where((cls >= 0)[:, None], (1 - alpha) * pr[:, N_TGT:] + alpha * oh, pr[:, N_TGT:])
    if beta > 0:
        q[:, :N_TGT] = np.maximum(pr[:, :N_TGT], beta * targets_from_cls(cls))
    return q, bf == D


def mix_settings():
    return [dict(zip(MIX, v)) for v in itertools.product(*MIX.values())]


def mix_label(st):
    return "alpha {:.1f} beta {:.1f}{}".format(st["alpha"], st["beta"], " dfix" if st["dfix"] else "")


def mix_match(job):
    name, g, tag, settings, w, bias = job
    import run_v31_external as rx
    import miss_profile as M
    from decoder_sweep import stats
    key = "{}_{}".format(name, g)
    with np.load(os.path.join(OUT, "pred_{}.npz".format(tag))) as z:
        pr = z[key].astype(np.float32)
    c, f, lab, base, cand, _ = M.candidate((name, g, "e3", 0.6, "k1", 0.3))
    blk = rx.blocks(c["period"])
    res = [stats(lab, cand), stats(lab, base), stats(lab, viterbi(pr, blk, w, bias))]
    for st in settings:
        q, dm = mix_pr(pr, base, blk, st["alpha"], st["beta"])
        res.append(stats(lab, viterbi(q, blk, w, bias, dm if st["dfix"] else None)))
    return name, g, np.array(res, float)


def mix_cmd(args):
    import run_v31_external as rx
    import turnover_cls as T
    settings = mix_settings()
    if args.ours:
        import ours_chain as oc
        z = np.load(os.path.join(OUT, "ours_{}.npz".format(args.tag)))
        pr, base, blk = z["pr"].astype(np.float32), z["base"], [tuple(b) for b in z["blk"]]
        print("  " + str(oc.write_and_score(base, "v31_d1_basecheck")))
        print("  " + str(oc.write_and_score(viterbi(pr, blk, args.w, args.bias),
                                           "cn_{}_alone".format(args.tag))))
        for st in settings:
            q, dm = mix_pr(pr, base, blk, st["alpha"], st["beta"])
            out = viterbi(q, blk, args.w, args.bias, dm if st["dfix"] else None)
            nm = "cn_{}_a{:g}b{:g}{}".format(args.tag, st["alpha"], st["beta"],
                                             "d" if st["dfix"] else "").replace(".", "")
            print("  " + str(oc.write_and_score(out, nm)) + "   <- " + mix_label(st))
        return
    jobs = [(n, g, args.tag, settings, args.w, args.bias) for n, g, _ in T.ext_list()]
    with Pool(6) as pool:
        res = pool.map(mix_match, jobs, chunksize=1)
    names = np.array([r[0] for r in res])
    folds = np.array([rx.fold_of(r[0], r[1]) for r in res])
    arr = np.stack([r[2] for r in res])
    labels = ["candidate e3k", "base v31_d1", "net alone"] + [mix_label(s) for s in settings]
    print("mix, change net {} (w {:g}, bias {:g}); up/down vs candidate e3k".format(args.tag, args.w, args.bias))
    for j, lb in enumerate(labels):
        parts = []
        for nm in ("skillcorner", "pff"):
            v = arr[names == nm, j, 0]
            d = v - arr[names == nm, 0, 0]
            parts.append("{:.4f} {:>3}/{:<3}".format(v.mean(), int(np.sum(d > 1e-9)), int(np.sum(d < -1e-9))))
        t = arr[:, j, 1:].sum(axis=0)
        print("  {:<24} SC {}  PFF {} | caught <2s {:.1%} 2-5s {:.1%} | pred/true {:.2f} prec {:.1%}".format(
            lb, parts[0], parts[1], t[1] / t[0], t[3] / t[2], t[5] / t[4], t[6] / t[5]))
    gain = arr[:, 3:, 0] - arr[:, :1, 0]
    nested = np.zeros(len(arr))
    for k in range(T.FOLDS):
        j = int(np.argmax(gain[folds != k].mean(axis=0)))
        nested[folds == k] = gain[folds == k, j]
    print("nested 4-fold (mix settings) vs candidate: SkillCorner {:+.4f}, PFF {:+.4f} ({}/{})".format(
        nested[names == "skillcorner"].mean(), nested[names == "pff"].mean(),
        int(np.sum(nested > 1e-9)), int(np.sum(nested < -1e-9))))


def ours_setup(tag):
    """(pr, base, blk, c, ex, x) on our match (MTAH_VARIANT=_kt); pr / x cached in OUT/ours_<tag>.npz."""
    import torch
    import dfl_data as dd
    import ours_chain as oc
    import turnover_cls as T
    from atd_cache import load_arrays
    from pipeline.features import LENGTH, WIDTH
    S.N_TGT = N_TGT
    codes, ex = oc.ours_v31(dead_prefix=os.path.join("models", "dead_ext_d1"))
    f = {k[2:]: v for k, v in ex.items() if k.startswith("f_")}
    blk = [tuple(b) for b in ex["blk"]]
    n = len(codes)
    period = np.zeros(n, dtype=np.int8)
    for i, (a, b) in enumerate(blk, start=1):
        period[a:b] = i
    c = dict(ex, period=period)
    base = T.base_chain(c, f, blk, ex["pdead"])
    atd = load_arrays()
    d = {"players": atd["players"], "team": atd["team"], "period": period}
    sign = T.attack_sign(d, period)
    x = S.frame_inputs(c, f, d, dd.blur_ball(atd["ball"]), (LENGTH, WIDTH), ex["pdead"], sign)
    ck = torch.load(os.path.join(MODELS, "{}_all.pt".format(tag)), weights_only=False)
    model = S.make_model(ck["n_in"], ck.get("hid", 128)).cuda()
    model.load_state_dict(ck["state"])
    pr = S.predict(model, ck["mu"], ck["sd"], S.drop_channels(x))
    np.savez(os.path.join(OUT, "ours_{}.npz".format(tag)), pr=pr, x=x.astype(np.float16), base=base,
             blk=np.array(blk))
    return pr, base, blk, c, ex, x


def ours_cmd(args):
    """Our match (MTAH_VARIANT=_kt): the all-84 model, decoded alone and with v31_d1's D fixed."""
    import ours_chain as oc
    import turnover_cls as T
    from report import PUBLIC_GT
    pr, base, blk, c, ex, x = ours_setup(args.tag)
    f = {k[2:]: v for k, v in ex.items() if k.startswith("f_")}
    nb = len(pm.possession_spells(base, pm.valid_runs(base)))
    print("v31_d1 base: {} possessions, D {:.1%}".format(nb, np.mean(base == D)))
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    tc = [t for t, _ in M_changes(gt)]
    print("public true changes {}; change-head peak within +-0.5 s (median): {:.2f}".format(
        len(tc), float(np.median([pr[max(t - 12, 0):t + 13, :N_TGT].max() for t in tc]))))
    if args.hyb:
        for b in (0.5, 1.0, 1.5):
            _, boost = hyb_inputs(pr, 0.0, b)
            out = T.base_chain(c, f, blk, ex["pdead"], None, boost)
            sp = pm.possession_spells(out, pm.valid_runs(out))
            nm = "{}_{}_hb{:g}".format(args.name, args.tag, b).replace(".", "")
            print("hybrid b {:g}: {} possessions, frames changed vs base {:.2%}".format(
                b, len(sp), np.mean(out != base)))
            print("  " + str(oc.write_and_score(out, nm)))
        print("  " + str(oc.write_and_score(base, "v31_d1_basecheck")))
        return
    for w, bias in ((0.5, -3.0), (1.0, -3.0), (1.0, -1.5)):
        for dfix in (0, 1):
            out = viterbi(pr, blk, w, bias, (base == D) if dfix else None)
            sp = pm.possession_spells(out, pm.valid_runs(out))
            nm = "{}_{}_w{:g}b{:g}{}".format(args.name, args.tag, w, -bias, "_dfix" if dfix else "").replace(".", "")
            print("w {:g} bias {:g}{}: {} possessions, D {:.1%}, frames changed vs base {:.1%}".format(
                w, bias, ", D fixed" if dfix else "", len(sp), np.mean(out == D), np.mean(out != base)))
            print("  " + str(oc.write_and_score(out, nm)))


def M_changes(codes):
    import miss_profile as M
    return M.changes(codes, pm.valid_runs(codes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("train", "eval", "ours", "hyb", "mix"))
    ap.add_argument("--name", default="v31_d1_cn")
    ap.add_argument("--tag", default="c1")
    ap.add_argument("--folds", nargs="+", default=["0", "1", "2", "3"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ours", action="store_true", help="mix: on our match (needs ours_<tag>.npz)")
    ap.add_argument("--w", type=float, default=1.0)
    ap.add_argument("--bias", type=float, default=-3.0)
    ap.add_argument("--hyb", action="store_true", help="ours: v31_d1 + the class head as team boost")
    ap.add_argument("--dfix", action="store_true", help="eval: D frames fixed to the candidate's")
    args = ap.parse_args()
    {"train": train_cmd, "eval": eval_cmd, "ours": ours_cmd, "hyb": hyb_cmd, "mix": mix_cmd}[args.cmd](args)


if __name__ == "__main__":
    main()
