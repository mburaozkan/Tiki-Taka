"""Short-possession detector: a frame-level network that does not depend on touch peaks. Aggregate only.

    python tools/short_net.py prep [--jobs 6]       # per-frame inputs + targets per external match
    python tools/short_net.py train [--tag s1] [--folds 0 1 2 3 all] [--epochs 12]
    python tools/short_net.py eval [--tag s1]        # spells from P(inside), pasted over v31_d1

Inputs per frame (25 fps; home-frame ball, each team in its own attacking direction, so a home /
away swap is a channel permutation): ball x (toward home's attack), y, z, velocity, acceleration,
speed, seen, gap; per team: 3 nearest players' distances, players within 2 / 5 / 10 m, the nearest
player's speed and run toward the ball, team centroid x and spread, touch-network P (continuous,
no peak picking), distance / carrier / co-movement features, first-stage P, refiner log-odds, run
vote; shared: hand D, dead-ball head P, ball out, keeper hold.
Targets per frame: inside a true possession < SHORT_S of H / of A, its start / end (Gaussian,
sigma 2 frames), and the possession class (H / A / D) as a helper; ungraded frames masked.
Model: input projection, dilated 1D convolutions (1, 2, 4, 8, 16), 2-layer bidirectional GRU,
heads. Random 20 s crops, H/A swap augmentation, AdamW, GPU. 4 fixed folds (each external match
scored by the model that never saw it) + one model on all 84 for our match.
eval: per team, hysteresis runs of P(inside) (>= hi, extended while >= hi - 0.15), 0.2-5 s long,
written over the base's live frames. spellF1 (SkillCorner, PFF, up / down), caught < 2 s / 2-5 s
(IoU >= 0.5, of all true possessions in that class), predicted / true possessions, precision; and
frame-level AUC of P(inside) on true short frames.
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
FPS = 25
SHORT_S = 5.0
SIGMA = 2.0
CROP = 500                     # 20 s
OUT = os.path.join(ROOT, "data", "cache", "short_net")
MODELS = os.path.join(OUT, "models")
DEAD_TAG = "d1"
DROP = [c for c in os.environ.get("SN_DROP", "").split(",") if c]   # channels zeroed (--drop)
N_TGT = 6                      # inside H, inside A, start H, start A, end H, end A
SHARED = ["bx", "by", "bz", "bvx", "bvy", "bspd", "bacc", "seen", "gap", "handD", "dead",
          "out", "hold"]
NEG_ON_SWAP = {"bx", "bvx"}     # home-frame ball x components flip on an H/A swap
TEAM = ["d1", "d2", "d3", "n2", "n5", "n10", "nspd", "ntow", "cx", "spread", "touch", "fd",
        "fc", "fm", "p1", "ref", "vote"]


def channel_names():
    return SHARED + ["h_" + k for k in TEAM] + ["a_" + k for k in TEAM]


def swap_index():
    """(perm, sign): x_swapped = x[..., perm] * sign."""
    names = channel_names()
    perm, sign = [], []
    for nm in names:
        if nm.startswith("h_"):
            perm.append(names.index("a_" + nm[2:]))
        elif nm.startswith("a_"):
            perm.append(names.index("h_" + nm[2:]))
        else:
            perm.append(names.index(nm))
        sign.append(-1.0 if nm in NEG_ON_SWAP else 1.0)
    return np.array(perm), np.array(sign, dtype=np.float32)


def targets(lab):
    """(n, 6) float32 targets and (n,) class (0 H, 1 A, 2 D, -1 ungraded)."""
    n = len(lab)
    y = np.zeros((n, N_TGT), dtype=np.float32)
    k = int(3 * SIGMA)
    g = np.exp(-0.5 * (np.arange(-k, k + 1) / SIGMA) ** 2).astype(np.float32)
    for a, b, team in pm.possession_spells(lab, pm.valid_runs(lab)):
        if (b - a) / FPS >= SHORT_S:
            continue
        s = 0 if team == H else 1
        y[a:b, s] = 1.0
        for col, t in ((2 + s, a), (4 + s, b - 1)):
            lo, hi = max(t - k, 0), min(t + k + 1, n)
            y[lo:hi, col] = np.maximum(y[lo:hi, col], g[lo - t + k:hi - t + k])
    cls = np.full(n, -1, dtype=np.int8)
    cls[lab == H], cls[lab == A], cls[lab == D] = 0, 1, 2
    return y, cls


def frame_inputs(c, f, d, ball, pitch, pdead, sign_home):
    """(n, C) float32 per-frame inputs (see channel_names)."""
    from pipeline import features
    L, W = pitch
    n = len(ball)
    players, team = d["players"], d["team"]
    bx = (ball[:, 0] - 0.5) * L
    by = (ball[:, 1] - 0.5) * W
    bz = np.nan_to_num(ball[:, 2] if ball.shape[1] > 2 else np.zeros(n))
    vx = np.gradient(bx) * FPS
    vy = np.gradient(by) * FPS
    spd = np.hypot(vx, vy)
    acc = np.abs(np.gradient(spd)) * FPS
    seen = np.isfinite(bx).astype(np.float32)
    out = {"bx": bx * sign_home, "by": by, "bz": bz, "bvx": vx * sign_home, "bvy": vy,
           "bspd": np.clip(spd, 0, 40), "bacc": np.clip(acc, 0, 300), "seen": seen,
           "gap": np.clip(np.nan_to_num(f["gap_s"]), 0, 10), "handD": (c["hand"] == D) * 1.0,
           "dead": pdead, "out": f["ball_out"], "hold": f["keeper_hold"]}
    px = (players[:, :, 0] - 0.5) * L
    py = (players[:, :, 1] - 0.5) * W
    pvx = np.zeros_like(px)
    pvy = np.zeros_like(py)
    pvx[5:], pvy[5:] = (px[5:] - px[:-5]) * FPS / 5.0, (py[5:] - py[:-5]) * FPS / 5.0
    dx, dy = px - bx[:, None], py - by[:, None]
    dist = np.hypot(dx, dy)
    ph1 = c["p1"][:, H] / np.maximum(c["p1"][:, H] + c["p1"][:, A], 1e-6)
    q = np.clip(c["p_ref"], 1e-4, 1 - 1e-4)
    lq = np.log(q / (1 - q))
    for tag, tid, sgn, pk in (("h_", features.HOME, 1.0, "home"), ("a_", features.AWAY, -1.0, "away")):
        sel = team == tid
        ds = np.where(np.isnan(dist[:, sel]), np.inf, dist[:, sel])
        srt = np.sort(ds, axis=1)
        for r in range(3):
            out[tag + "d%d" % (r + 1)] = np.clip(np.where(np.isinf(srt[:, r]), 40, srt[:, r]), 0, 40)
        for m in (2, 5, 10):
            out[tag + "n%d" % m] = np.sum(ds < m, axis=1).astype(np.float32)
        j = np.argmin(ds, axis=1)
        rows = np.arange(n)
        vxj, vyj = pvx[:, sel][rows, j], pvy[:, sel][rows, j]
        dxj, dyj = dx[:, sel][rows, j], dy[:, sel][rows, j]
        dj = np.maximum(np.hypot(dxj, dyj), 0.3)
        out[tag + "nspd"] = np.clip(np.nan_to_num(np.hypot(vxj, vyj)), 0, 12)
        out[tag + "ntow"] = np.clip(np.nan_to_num(-(vxj * dxj + vyj * dyj) / dj), -12, 12)
        tx = px[:, sel] * (sign_home * sgn)[:, None]
        out[tag + "cx"] = np.nan_to_num(np.nanmean(np.where(np.isfinite(tx), tx, np.nan), axis=1))
        out[tag + "spread"] = np.nan_to_num(np.nanstd(np.where(np.isfinite(tx), tx, np.nan), axis=1))
        out[tag + "touch"] = f["touch_p_" + pk]
        out[tag + "fd"] = np.clip(np.nan_to_num(f["d_" + pk], nan=40), 0, 40)
        out[tag + "fc"] = np.nan_to_num(f["c_" + pk])
        out[tag + "fm"] = np.nan_to_num(f["m_" + pk])
        out[tag + "p1"] = ph1 if tid == features.HOME else 1 - ph1
        out[tag + "ref"] = lq * sgn
        out[tag + "vote"] = c["vote"] * sgn
    for k in ("bx", "by", "bvx", "bvy", "bspd", "bacc"):
        out[k] = np.nan_to_num(out[k])
    return np.stack([np.asarray(out[k], dtype=np.float32) for k in channel_names()], axis=1)


def prep_match(job):
    name, g, fold = job
    path = os.path.join(OUT, "in_{}_{}.npy".format(name, g))
    if os.path.isfile(path):
        return name, g, fold
    import dfl_data as dd
    import run_v31_external as rx
    import turnover_cls as T
    import pair_cls as P
    c, f = T.load_cache(name, g)
    pdead = P.pdead_of(name, g)
    d, pitch, ball = rx.load_set(name, g)
    sign = T.attack_sign(d, c["period"])
    x = frame_inputs(c, f, d, dd.blur_ball(ball), pitch, pdead, sign)
    y, cls = targets(c["labels"])
    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, "y_{}_{}.npy".format(name, g)), y)
    np.save(os.path.join(OUT, "c_{}_{}.npy".format(name, g)), cls)
    np.save(path, x.astype(np.float16))
    return name, g, fold


def prep_cmd(args):
    import turnover_cls as T
    jobs = T.ext_list()
    with Pool(args.jobs) as pool:
        done = pool.map(prep_match, jobs, chunksize=1)
    xs = [np.load(os.path.join(OUT, "in_{}_{}.npy".format(n, g)), mmap_mode="r") for n, g, _ in done]
    ys = [np.load(os.path.join(OUT, "y_{}_{}.npy".format(n, g)), mmap_mode="r") for n, g, _ in done]
    frames = sum(len(x) for x in xs)
    inside = sum(float(y[:, :2].sum()) for y in ys)
    print("{} matches, {} frames, {} channels; frames inside a short possession {:.1%}; any NaN: {}"
          .format(len(xs), frames, xs[0].shape[1], inside / frames,
                  any(np.isnan(np.asarray(x[::50], np.float32)).any() for x in xs)))


# ---------------------------------------------------------------- model

def make_model(n_in, hid=128):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(n_in, hid), nn.GELU())
            self.convs = nn.ModuleList([nn.Conv1d(hid, hid, 5, padding=2 * dl, dilation=dl)
                                        for dl in (1, 2, 4, 8, 16)])
            self.norms = nn.ModuleList([nn.LayerNorm(hid) for _ in range(5)])
            self.gru = nn.GRU(hid, hid, num_layers=2, batch_first=True, bidirectional=True,
                              dropout=0.1)
            self.head = nn.Linear(2 * hid, N_TGT + 3)

        def forward(self, x):
            h = self.inp(x)
            for conv, norm in zip(self.convs, self.norms):
                h = norm(h + torch.nn.functional.gelu(conv(h.transpose(1, 2)).transpose(1, 2)))
            h, _ = self.gru(h)
            return self.head(h)
    return Net()


def drop_channels(x):
    """x with the DROP channels set to 0 (constant: the network can't use them)."""
    if DROP:
        x = np.array(x, copy=True)
        x[:, [channel_names().index(c) for c in DROP]] = 0
    return x


def load_data(keys):
    xs, ys, cs = [], [], []
    for name, g in keys:
        xs.append(drop_channels(np.load(os.path.join(OUT, "in_{}_{}.npy".format(name, g)))))
        ys.append(np.load(os.path.join(OUT, "y_{}_{}.npy".format(name, g))))
        cs.append(np.load(os.path.join(OUT, "c_{}_{}.npy".format(name, g))))
    return xs, ys, cs


def norm_stats(xs):
    s = np.concatenate([x[::25].astype(np.float32) for x in xs])
    mu, sd = s.mean(axis=0), s.std(axis=0) + 1e-3
    return mu.astype(np.float32), sd.astype(np.float32)


HID = int(os.environ.get("SN_HID", "128"))


def train_one(keys, epochs, seed=0, log=""):
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    xs, ys, cs = load_data(keys)
    mu, sd = norm_stats(xs)
    perm, sign = swap_index()
    dev = "cuda"
    model = make_model(xs[0].shape[1], HID).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    total_frames = sum(len(x) for x in xs)
    steps_per_epoch = total_frames // (CROP * 32)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * steps_per_epoch)
    pos_w = torch.tensor([4.0, 4.0, 20.0, 20.0, 20.0, 20.0], device=dev)
    lens = np.array([len(x) for x in xs], dtype=np.float64)
    pm_ = lens / lens.sum()
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for _ in range(steps_per_epoch):
            bx, by, bc = [], [], []
            for _ in range(32):
                m = rng.choice(len(xs), p=pm_)
                a = int(rng.integers(0, len(xs[m]) - CROP))
                xb = (xs[m][a:a + CROP].astype(np.float32) - mu) / sd
                yb = ys[m][a:a + CROP]
                cb = cs[m][a:a + CROP].astype(np.int64)
                if rng.random() < 0.5:          # H / A swap
                    xb = ((xb * sd + mu)[:, perm] * sign - mu) / sd
                    yb = yb[:, [1, 0, 3, 2, 5, 4]]
                    cb = np.where(cb == 0, 1, np.where(cb == 1, 0, cb))
                bx.append(xb)
                by.append(yb)
                bc.append(cb)
            xb = torch.from_numpy(np.stack(bx)).to(dev)
            yb = torch.from_numpy(np.stack(by)).to(dev)
            cb = torch.from_numpy(np.stack(bc)).to(dev)
            out = model(xb)
            mask = (cb >= 0).float()
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
        print("  {} epoch {} loss {:.4f} ({:.0f} s)".format(log, ep + 1, tot / steps_per_epoch,
                                                            time.time() - t0), flush=True)
    return model, mu, sd


def predict(model, mu, sd, x):
    """(n, 9) sigmoid of the 6 targets + softmax class, whole match in overlapping chunks."""
    import torch
    model.eval()
    n = len(x)
    out = np.zeros((n, N_TGT + 3), dtype=np.float32)
    chunk, pad = 3000, 250
    with torch.no_grad():
        for a in range(0, n, chunk):
            lo, hi = max(a - pad, 0), min(a + chunk + pad, n)
            xb = torch.from_numpy(((x[lo:hi].astype(np.float32) - mu) / sd)[None]).cuda()
            o = model(xb)[0].float().cpu().numpy()
            o = o[a - lo:a - lo + min(chunk, n - a)]
            out[a:a + len(o), :N_TGT] = 1 / (1 + np.exp(-o[:, :N_TGT]))
            e = np.exp(o[:, N_TGT:] - o[:, N_TGT:].max(axis=1, keepdims=True))
            out[a:a + len(o), N_TGT:] = e / e.sum(axis=1, keepdims=True)
    return out


def train_cmd(args):
    import torch
    import turnover_cls as T
    ext = T.ext_list()
    os.makedirs(MODELS, exist_ok=True)
    pred_path = os.path.join(OUT, "pred_{}.npz".format(args.tag))
    preds = dict(np.load(pred_path)) if os.path.isfile(pred_path) else {}
    for fold in args.folds:
        if fold == "all":
            model, mu, sd = train_one([(n, g) for n, g, _ in ext], args.epochs,
                                      seed=100 * args.seed, log="all")
            torch.save({"state": model.state_dict(), "mu": mu, "sd": sd,
                        "n_in": len(mu), "hid": HID}, os.path.join(MODELS, "{}_all.pt".format(args.tag)))
            continue
        k = int(fold)
        model, mu, sd = train_one([(n, g) for n, g, f in ext if f != k], args.epochs,
                                  seed=k + 100 * args.seed, log="fold %d" % k)
        for n, g, f in ext:
            if f == k:
                x = drop_channels(np.load(os.path.join(OUT, "in_{}_{}.npy".format(n, g))))
                preds["{}_{}".format(n, g)] = predict(model, mu, sd, x).astype(np.float16)
        np.savez(pred_path, **preds)
        print("fold {} done, {} matches predicted".format(k, len(preds)), flush=True)


# ---------------------------------------------------------------- eval

def spells_from(q, hi, lo, min_s=0.2, max_s=SHORT_S):
    """[(a, b)] hysteresis runs of q: start where q >= hi, extend while q >= lo."""
    on = q >= lo
    e = np.flatnonzero(np.diff(np.concatenate(([0], on.astype(np.int8), [0]))))
    out = []
    for a, b in zip(e[::2], e[1::2]):
        if q[a:b].max() >= hi and min_s * FPS <= b - a <= max_s * FPS:
            out.append((a, b))
    return out


def paste_spells(base, pr, hi, lo, over_d=False):
    out = base.copy()
    ok = (H, A, D) if over_d else (H, A)
    cand = []
    for s, team in ((0, H), (1, A)):
        for a, b in spells_from(pr[:, s].astype(np.float32), hi, lo):
            cand.append((float(pr[a:b, s].mean()), a, b, team))
    for _, a, b, team in sorted(cand):            # stronger spells written last
        seg = out[a:b]
        out[a:b] = np.where(np.isin(seg, ok), team, seg)
    return out


EVAL = [(0.3, 0.15), (0.4, 0.25), (0.5, 0.35), (0.6, 0.45), (0.7, 0.55), (0.8, 0.65)]


def eval_match(job):
    name, g, tag = job
    import run_v31_external as rx
    import turnover_cls as T
    import pair_cls as P
    from decoder_sweep import stats
    with np.load(os.path.join(OUT, "pred_{}.npz".format(tag))) as z:
        key = "{}_{}".format(name, g)
        if key not in z.files:
            return None
        pr = z[key].astype(np.float32)
    c, f = T.load_cache(name, g)
    pdead = P.pdead_of(name, g)
    blk = rx.blocks(c["period"])
    lab = c["labels"]
    base = T.base_chain(c, f, blk, pdead)
    res = [stats(lab, base)]
    for hi, lo in EVAL:
        res.append(stats(lab, paste_spells(base, pr, hi, lo)))
    for hi, lo in EVAL[2:]:
        res.append(stats(lab, paste_spells(base, pr, hi, lo, over_d=True)))
    y, cls = targets(lab)
    g_ = cls >= 0
    pos = np.concatenate([pr[g_ & (y[:, 0] > 0), 0], pr[g_ & (y[:, 1] > 0), 1]])
    neg = np.concatenate([pr[g_ & (y[:, 0] == 0), 0][::10], pr[g_ & (y[:, 1] == 0), 1][::10]])
    return name, np.array(res, dtype=float), pos[::5], neg[::5]


def eval_cmd(args):
    import turnover_cls as T
    jobs = [(n, g, args.tag) for n, g, _ in T.ext_list()]
    with Pool(6) as pool:
        res = [r for r in pool.map(eval_match, jobs, chunksize=1) if r is not None]
    names = np.array([r[0] for r in res])
    arr = np.stack([r[1] for r in res])
    pos = np.concatenate([r[2] for r in res])
    neg = np.concatenate([r[3] for r in res])
    print("{} matches; frame AUC of P(inside a short possession): {:.3f}".format(
        len(res), T.roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])))
    labels = ["base v31_d1"] + ["paste hi {:g}".format(h) for h, _ in EVAL] + \
        ["paste hi {:g}, also over D".format(h) for h, _ in EVAL[2:]]
    print("{:<28} {:>8} {:>8} {:>9} | {:>11} {:>11} | {:>9} {:>9}".format(
        "setting", "SkillC", "PFF", "up/down", "caught <2s", "caught 2-5s", "pred/true", "precision"))
    for j, lb in enumerate(labels):
        sc = [arr[names == nm, j, 0].mean() for nm in ("skillcorner", "pff")]
        d = arr[:, j, 0] - arr[:, 0, 0]
        t = arr[:, j, 1:].sum(axis=0)
        print("{:<28} {:>8.4f} {:>8.4f} {:>4}/{:<4} | {:>10.1%} {:>10.1%}  | {:>9.2f} {:>9.1%}".format(
            lb, sc[0], sc[1], int(np.sum(d > 1e-9)), int(np.sum(d < -1e-9)), t[1] / t[0],
            t[3] / t[2], t[5] / t[4], t[6] / t[5]))


def ours_cmd(args):
    """Our match (MTAH_VARIANT=_kt): v31_d1 + spells of the all-84 model pasted (hi, over D)."""
    import torch
    import dfl_data as dd
    import ours_chain as oc
    import turnover_cls as T
    from atd_cache import load_arrays
    from pipeline.features import LENGTH, WIDTH
    codes, ex = oc.ours_v31(dead_prefix=os.path.join("models", "dead_ext_d1"))
    f = {k[2:]: v for k, v in ex.items() if k.startswith("f_")}
    blk = [tuple(b) for b in ex["blk"]]
    n = len(codes)
    period = np.zeros(n, dtype=np.int8)
    for i, (a, b) in enumerate(blk, start=1):
        period[a:b] = i
    c = dict(ex, period=period)
    base = T.base_chain(c, f, blk, ex["pdead"])
    print("base_chain equals ours_v31: {}".format(np.array_equal(base, codes)))
    atd = load_arrays()
    d = {"players": atd["players"], "team": atd["team"], "period": period}
    sign = T.attack_sign(d, period)
    xraw = frame_inputs(c, f, d, dd.blur_ball(atd["ball"]), (LENGTH, WIDTH), ex["pdead"], sign)
    members = args.ens.split(",") if args.ens else [args.tag + (":" + ",".join(DROP) if DROP else "")]
    prs = []
    for mem in members:
        tag, _, drop = mem.partition(":")
        x = np.array(xraw, copy=True)
        if drop:
            x[:, [channel_names().index(ch) for ch in drop.split("+")]] = 0
        ck = torch.load(os.path.join(MODELS, "{}_all.pt".format(tag)), weights_only=False)
        model = make_model(ck["n_in"], ck.get("hid", 128)).cuda()
        model.load_state_dict(ck["state"])
        prs.append(predict(model, ck["mu"], ck["sd"], x))
    pr = np.mean(prs, axis=0)
    print("models: {}".format(members))
    print("frames with P(inside short) >= 0.5: {:.1%} (external mean of true short frames 4.5%)".format(
        float(np.mean(pr[:, :2].max(axis=1) >= 0.5))))
    from report import PUBLIC_GT
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    shorts = [(i + 1, a, b, k) for i, (a, b, k) in enumerate(pm.possession_spells(gt, pm.valid_runs(gt)))
              if (b - a) / FPS < SHORT_S]
    print("public short possessions, peak P(inside a short spell of their team) inside / +-1 s: " +
          ", ".join("#{} ({:.1f} s) {:.2f} / {:.2f}".format(
              i, (b - a) / FPS, float(pr[a:b, 0 if k == H else 1].max()),
              float(pr[max(a - FPS, 0):b + FPS, 0 if k == H else 1].max())) for i, a, b, k in shorts))
    if args.check:
        import xgboost as xgb
        import in_domain_team as idt
        import short_check as K
        T.V2 = True
        ktag, gen, thr = args.check.split(":")
        gen, thr = float(gen), float(thr)
        cols = idt.player_columns(atd["players"], atd["team"], period, (LENGTH, WIDTH))
        ctx = T.match_ctx(c, f, base, ex["pdead"], d, dd.blur_ball(atd["ball"]), (LENGTH, WIDTH), cols)
        spells = K.gen_spells(pr, gen)
        rows, names = [], None
        for a, b, s_ in spells:
            xx = K.spell_inputs(ctx, pr, a, b, s_)
            names = names or list(xx)
            rows.append([xx[k] for k in names])
        bst = xgb.Booster()
        bst.load_model(os.path.join(K.OUT, "models", "{}_{}_g{:g}_all.json".format(
            ktag, args.ens_tag, gen)))
        pk = bst.predict(xgb.DMatrix(np.array(rows, np.float32))) if rows else np.zeros(0)
        new = base.copy()
        ok = (H, A, D) if args.over_d else (H, A)
        for i in np.argsort(pk):
            if pk[i] < thr:
                continue
            a, b, s_ = spells[i]
            a, b = K.refine_edges(pr, a, b, s_, n)
            seg = new[a:b]
            new[a:b] = np.where(np.isin(seg, ok), H if s_ == 0 else A, seg)
        sp = pm.possession_spells(new, pm.valid_runs(new))
        print("check {} (gen {:g}, keep >= {:g}): {} spells, {} kept; {} possessions (base {}), "
              "frames changed {:.2f}%".format(ktag, gen, thr, len(spells), int(np.sum(pk >= thr)),
                                             len(sp), len(pm.possession_spells(base, pm.valid_runs(base))),
                                             100 * np.mean(new != base)))
        if args.name:
            print(oc.write_and_score(new, args.name))
        return
    for hi in args.hi:
        new = paste_spells(base, pr, hi, hi - 0.15, over_d=args.over_d)
        sp = pm.possession_spells(new, pm.valid_runs(new))
        print("hi {:g}{}: {} possessions (base {}), frames changed {:.2f}%".format(
            hi, ", over D" if args.over_d else "", len(sp),
            len(pm.possession_spells(base, pm.valid_runs(base))), 100 * np.mean(new != base)))
        if args.name:
            print(oc.write_and_score(new, "{}_h{:g}".format(args.name, hi).replace(".", "")))


def avg_cmd(args):
    """pred_<tag>.npz = mean of pred_<t>.npz for t in --from (an ensemble of held-out predictions)."""
    zs = [np.load(os.path.join(OUT, "pred_{}.npz".format(t))) for t in args.src]
    keys = set(zs[0].files).intersection(*[z.files for z in zs[1:]])
    np.savez(os.path.join(OUT, "pred_{}.npz".format(args.tag)),
             **{k: np.mean([z[k].astype(np.float32) for z in zs], axis=0).astype(np.float16)
                for k in keys})
    print("pred_{}: mean of {} over {} matches".format(args.tag, args.src, len(keys)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("prep", "train", "eval", "ours", "avg"))
    ap.add_argument("--src", nargs="+", default=[])
    ap.add_argument("--name")
    ap.add_argument("--hi", type=float, nargs="+", default=[0.6])
    ap.add_argument("--over-d", action="store_true")
    ap.add_argument("--drop", default="", help="channels to zero, e.g. bz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ens", default="", help="ensemble for ours, e.g. s1,s1b,s1nz:bz")
    ap.add_argument("--ens-tag", default="e3", help="the held-out ensemble tag the check was trained on")
    ap.add_argument("--check", default="", help="ours: keep/drop check TAG:GEN:THR, e.g. k1:0.6:0.3")
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--tag", default="s1")
    ap.add_argument("--folds", nargs="+", default=["0", "1", "2", "3"])
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()
    global DROP, HID
    HID = args.hid
    if args.drop:
        DROP = args.drop.split(",")
        os.environ["SN_DROP"] = args.drop
    {"prep": prep_cmd, "train": train_cmd, "eval": eval_cmd, "ours": ours_cmd,
     "avg": avg_cmd}[args.cmd](args)


if __name__ == "__main__":
    main()
