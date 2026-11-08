"""Change net that sees the players themselves, not only team summaries. Aggregate only.

    python tools/player_net.py prep [--src xb]                         # tokens, data/cache/player_net/
    CN_DFL=5 python tools/player_net.py train --tag p1 [--folds 0 1 2 3] [--epochs 12]

Per frame, the 12 players nearest the ball (ball = the source inputs' ball, xb: ball net in the gaps)
become tokens: offset to the ball and velocity (x toward the home team's attack, so an H/A swap is a
sign flip), velocity relative to the ball, distance, speed toward the ball, team (+1 home / -1 away),
valid. A shared token MLP + attention pooling (once over all tokens, once signed by team) gives 96
numbers per frame, joined to the change net's per-frame inputs and fed to the same temporal model
(dilated convolutions + bidirectional GRU) with the same targets, folds and output file
(data/cache/change_net/pred_<tag>.npz), so cn_dfl_miss.py / cn_tbias.py read it as any change net.
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
import short_net as S  # noqa: E402
import change_net as C  # noqa: E402

FPS = 25
K_TOK = 12
F_TOK = 10
NEG_TOK = [0, 2, 4, 7]           # dx, vx, rvx, team flip on an H/A swap
OUT = os.path.join(ROOT, "data", "cache", "player_net")
SRC = os.environ.get("PN_SRC", "xb")


def cdiff(v):
    out = np.full_like(v, np.nan)
    out[2:-2] = (v[4:] - v[:-4]) * FPS / 4.0
    out[:2], out[-2:] = out[2], out[-3]
    return out


def tok_path(name, g):
    return os.path.join(OUT, "tok_{}_{}.npy".format(name, g))


def prep_one(job):
    name, g = job
    if os.path.isfile(tok_path(name, g)):
        return None
    import dfl_data as dd
    import turnover_cls as T
    from pipeline import features
    x = np.load(os.path.join(C.OUT, SRC, "{}_{}.npy".format(name, g))).astype(np.float32)
    names = S.SHARED + ["h_" + k for k in S.BASE_TEAM] + ["a_" + k for k in S.BASE_TEAM]
    if name == "dfl":
        import cn_dfl
        d = dd.load(g)
        with np.load(cn_dfl.path(g)) as z:
            blk = [tuple(b) for b in z["blk"]]
        period = np.zeros(len(x), np.int8)
        for i, (a, b) in enumerate(blk, start=1):
            period[a:b] = i
        dd_ = {"players": d["players"], "team": d["team"], "period": period}
        pitch = dd.PITCH
    else:
        import run_v31_external as rx
        c, _ = T.load_cache(name, g)
        dd_, pitch, _ = rx.load_set(name, g)
        period = c["period"]
    sign = T.attack_sign(dd_, period)[:, None]
    L, W = pitch
    px = (dd_["players"][:, :, 0] - 0.5) * L * sign            # toward the home attack
    py = (dd_["players"][:, :, 1] - 0.5) * W
    pvx, pvy = cdiff(px), cdiff(py)
    bx, by = x[:, names.index("bx")][:, None], x[:, names.index("by")][:, None]
    bvx, bvy = x[:, names.index("bvx")][:, None], x[:, names.index("bvy")][:, None]
    dx, dy = px - bx, py - by
    dist = np.hypot(dx, dy)
    dsort = np.where(np.isnan(dist), np.inf, dist)
    order = np.argsort(dsort, axis=1)[:, :K_TOK]
    r = np.arange(len(x))[:, None]
    team = np.where(dd_["team"] == features.HOME, 1.0, -1.0)[None].repeat(len(x), 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        tow = -(pvx * dx + pvy * dy) / np.maximum(dist, 0.3)
    feats = [np.clip(dx, -40, 40) / 10, np.clip(dy, -40, 40) / 10, np.clip(pvx, -12, 12) / 8,
             np.clip(pvy, -12, 12) / 8, np.clip(pvx - bvx, -30, 30) / 8, np.clip(pvy - bvy, -30, 30) / 8,
             np.clip(dist, 0, 60) / 10, team, np.clip(tow, -12, 12) / 8, np.isfinite(dist).astype(np.float64)]
    tok = np.stack([f[r, order] for f in feats], axis=2)
    tok = np.nan_to_num(tok)
    tok[~np.isfinite(dsort[r, order])] = 0.0
    os.makedirs(OUT, exist_ok=True)
    np.save(tok_path(name, g), tok.astype(np.float16))
    return name


def prep_cmd(args):
    import turnover_cls as T
    import dfl_data as dd
    from multiprocessing import Pool
    jobs = [(n, g) for n, g, _ in T.ext_list()] + [("dfl", g) for g in dd.GAMES]
    with Pool(args.jobs) as p:
        done = [r for r in p.map(prep_one, jobs) if r is not None]
    print("tokens written for {} matches ({} per frame x {} features)".format(len(done), K_TOK, F_TOK))


def make_model(n_in, hid=128, d=48):
    import torch
    import torch.nn as nn
    S.N_TGT = C.N_TGT
    temporal = S.make_model(n_in + 2 * d, hid)

    class PNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Sequential(nn.Linear(F_TOK, d), nn.GELU(), nn.Linear(d, d), nn.GELU())
            self.att = nn.Linear(d, 1)
            self.temporal = temporal

        def forward(self, x, tok):
            h = self.tok(tok)                                     # (b, T, k, d)
            valid = tok[..., 9:10] > 0.5
            a = self.att(h).masked_fill(~valid, -1e4)
            w = torch.softmax(a, dim=2)
            team = tok[..., 7:8]
            pooled = torch.cat([(w * h).sum(2), (w * h * team).sum(2)], dim=-1)
            return self.temporal(torch.cat([x, pooled], dim=-1))
    return PNet()


def train_one(keys, epochs, seed=0, log=""):
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    xs, ys, cs = C.load_data(keys)
    toks = [np.load(tok_path(n, g), mmap_mode="r") for n, g in keys]
    mu, sd = S.norm_stats(xs)
    perm, sign = S.swap_index()
    tsign = np.ones(F_TOK, np.float32)
    tsign[NEG_TOK] = -1
    model = make_model(xs[0].shape[1], C.HID).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = sum(len(x) for x in xs) // (C.CROP * 32)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * steps)
    pos_w = torch.tensor([20.0] * C.N_TGT, device="cuda")
    lens = np.array([len(x) for x in xs], dtype=np.float64)
    lens *= np.array([C.DFL_W if n == "dfl" else 1.0 for n, _ in keys])
    pm_ = lens / lens.sum()
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for _ in range(steps):
            bx, bt, by, bc = [], [], [], []
            for _ in range(32):
                m = rng.choice(len(xs), p=pm_)
                a = int(rng.integers(0, len(xs[m]) - C.CROP))
                xb = (xs[m][a:a + C.CROP].astype(np.float32) - mu) / sd
                tb = np.asarray(toks[m][a:a + C.CROP], dtype=np.float32)
                yb = ys[m][a:a + C.CROP]
                cb = cs[m][a:a + C.CROP].astype(np.int64)
                if rng.random() < 0.5:
                    xb = ((xb * sd + mu)[:, perm] * sign - mu) / sd
                    tb = tb * tsign
                    yb = yb[:, C.SWAP_T]
                    cb = np.where(cb == 0, 1, np.where(cb == 1, 0, cb))
                bx.append(xb)
                bt.append(tb)
                by.append(yb)
                bc.append(cb)
            xb = torch.from_numpy(np.stack(bx)).cuda()
            tb = torch.from_numpy(np.stack(bt)).cuda()
            yb = torch.from_numpy(np.stack(by)).cuda()
            cb = torch.from_numpy(np.stack(bc)).cuda()
            out = model(xb, tb)
            mask = (cb >= 0).float()
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                out[..., :C.N_TGT], yb, pos_weight=pos_w, reduction="none").mean(-1)
            ce = torch.nn.functional.cross_entropy(out[..., C.N_TGT:].reshape(-1, 3),
                                                   cb.clamp(min=0).reshape(-1),
                                                   reduction="none").reshape(cb.shape)
            loss = ((bce + 0.3 * ce) * mask).sum() / mask.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
        print("  {} epoch {} loss {:.4f} ({:.0f} s)".format(log, ep + 1, tot / steps, time.time() - t0), flush=True)
    return model, mu, sd


def predict(model, mu, sd, x, tok):
    import torch
    model.eval()
    n = len(x)
    out = np.zeros((n, C.N_TGT + 3), dtype=np.float32)
    chunk, pad = 3000, 250
    with torch.no_grad():
        for a in range(0, n, chunk):
            lo, hi = max(a - pad, 0), min(a + chunk + pad, n)
            xb = torch.from_numpy(((x[lo:hi].astype(np.float32) - mu) / sd)[None]).cuda()
            tb = torch.from_numpy(np.asarray(tok[lo:hi], dtype=np.float32)[None]).cuda()
            o = model(xb, tb)[0].float().cpu().numpy()
            o = o[a - lo:a - lo + min(chunk, n - a)]
            out[a:a + len(o), :C.N_TGT] = 1 / (1 + np.exp(-o[:, :C.N_TGT]))
            e = np.exp(o[:, C.N_TGT:] - o[:, C.N_TGT:].max(axis=1, keepdims=True))
            out[a:a + len(o), C.N_TGT:] = e / e.sum(axis=1, keepdims=True)
    return out


def train_cmd(args):
    import torch
    ext = C.train_list()
    pred_path = os.path.join(C.OUT, "pred_{}.npz".format(args.tag))
    preds = dict(np.load(pred_path)) if os.path.isfile(pred_path) else {}
    for fold in args.folds:
        if fold == "all":
            model, mu, sd = train_one([(n, g) for n, g, _ in ext], args.epochs, 100 * args.seed, "all")
            torch.save({"state": model.state_dict(), "mu": mu, "sd": sd, "n_in": len(mu), "hid": C.HID},
                       os.path.join(C.MODELS, "{}_all.pt".format(args.tag)))
            continue
        k = int(fold)
        model, mu, sd = train_one([(n, g) for n, g, f in ext if f != k], args.epochs, k + 100 * args.seed,
                                  "fold %d" % k)
        for n, g, f in ext:
            if f == k:
                tok = np.load(tok_path(n, g), mmap_mode="r")
                preds["{}_{}".format(n, g)] = predict(model, mu, sd, C.load_x(n, g), tok).astype(np.float16)
        np.savez(pred_path, **preds)
        print("fold {} done, {} matches predicted".format(k, len(preds)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("prep", "train"))
    ap.add_argument("--tag", default="p1")
    ap.add_argument("--folds", nargs="+", default=["0", "1", "2", "3"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    {"prep": prep_cmd, "train": train_cmd}[args.cmd](args)


if __name__ == "__main__":
    main()
