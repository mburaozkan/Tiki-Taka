"""Ball-from-players network: where the unseen ball is, learned on DFL. Aggregate only.

    python tools/ball_net.py train [--hold all|<game>] [--epochs 8]   # models/ball_net_<hold>.pt
    python tools/ball_net.py lomo                                      # 7 leave-one-match-out models + eval

Self-supervised: DFL's ball is known in every frame, so gaps are cut in with our gap pattern
(gap_model.degrade, training seeds 1..N_SEEDS, never the test seed) and the true ball inside them is
the target. For every gap frame the network sees 22 player tokens + one free-ball token:
  player  position, velocity, displacement over the last / next 1 s, team (+1 home / -1 away),
          distance to the last seen ball and to the next seen ball, to the straight-line ball, and
          whether it was the nearest player (<= 3 m) when the ball disappeared / reappeared;
  common  time since the ball was last seen / until it is seen again (log), on every token;
  free    sits on the straight line between the gap's ends.
Output: a weight per token (softmax) and a small offset; ball = sum of weights x (token position +
offset). The weights summed per team give P(the ball is at a home / away player).
Loss: smooth L1 on the ball position (m) + 0.5 x cross-entropy on the token nearest the true ball
(the free token when no player is within 2 m).
eval (lomo): each DFL match by the model that never saw it, on its test-seed gaps (the gaps the
change net sees): ball error, nearest-team right, live frames, vs the straight line and the carrier
rule (tools/ball_fill.py).
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
import ball_fill as BF  # noqa: E402

FPS = 25
N_SEEDS = 3
F = 18
CACHE = os.path.join(ROOT, "data", "cache", "ball_net")
MODELS = os.path.join(ROOT, "models")


def features(ball, players, team, blk, pitch, frames=None):
    """tokens (m, 23, F) float32, mask (m, 23) bool (True = absent), positions (m, 23, 2) metres,
    frame index (m,), for the gap frames (or the given subset of them)."""
    from pipeline import features as pf
    n = len(ball)
    P = BF.to_m(players[:, :, :2], pitch).astype(np.float32)         # (n, p, 2)
    p = P.shape[1]
    V = np.full_like(P, np.nan)
    V[5:] = (P[5:] - P[:-5]) * FPS / 5.0
    V[:5] = V[5]
    Fw = np.full_like(P, np.nan)
    Bw = np.full_like(P, np.nan)
    Fw[:-FPS] = P[FPS:] - P[:-FPS]
    Bw[FPS:] = P[FPS:] - P[:-FPS]
    B = BF.to_m(ball[:, :2], pitch)
    lin = BF.to_m(BF.fill_linear(ball, blk), pitch)
    idx, b0s, b1s, dt0, dt1, c0, c1 = [], [], [], [], [], [], []
    for a, b, s, e in BF.gap_runs(ball, blk):
        b0 = B[a - 1] if a > s else np.array([np.nan, np.nan])
        b1 = B[b] if b < e else np.array([np.nan, np.nan])

        def holder(t, bp):
            if t < 0 or t >= n or np.isnan(bp).any():
                return -1
            d = np.hypot(*(P[t] - bp).T)
            d = np.where(np.isnan(d), np.inf, d)
            j = int(np.argmin(d))
            return j if d[j] <= BF.CARRY_M else -1
        h0, h1 = holder(a - 1, b0), holder(b, b1)
        T = np.arange(a, b)
        idx.append(T)
        b0s.append(np.repeat(b0[None], len(T), 0))
        b1s.append(np.repeat(b1[None], len(T), 0))
        dt0.append((T - a + 1) / FPS if a > s else np.full(len(T), 60.0))
        dt1.append((b - T) / FPS if b < e else np.full(len(T), 60.0))
        c0.append(np.full(len(T), h0))
        c1.append(np.full(len(T), h1))
    if not idx:
        return (np.zeros((0, p + 1, F), np.float32), np.zeros((0, p + 1), bool),
                np.zeros((0, p + 1, 2), np.float32), np.zeros(0, int))
    idx = np.concatenate(idx)
    b0s, b1s = np.concatenate(b0s), np.concatenate(b1s)
    dt0, dt1 = np.concatenate(dt0), np.concatenate(dt1)
    c0, c1 = np.concatenate(c0), np.concatenate(c1)
    if frames is not None:
        keep = np.isin(idx, frames)
        idx, b0s, b1s, dt0, dt1, c0, c1 = (v[keep] for v in (idx, b0s, b1s, dt0, dt1, c0, c1))
    m = len(idx)
    pp, vv, fw, bw = P[idx], V[idx], Fw[idx], Bw[idx]
    L = lin[idx]
    sc = np.array([52.5, 34.0], np.float32)
    X = np.zeros((m, p + 1, F), np.float32)
    X[:, :p, 0:2] = pp / sc
    X[:, :p, 2:4] = vv / 8.0
    X[:, :p, 4:6] = fw / 10.0
    X[:, :p, 6:8] = bw / 10.0
    X[:, :p, 8] = np.where(team == pf.HOME, 1.0, -1.0)[None]

    def dist(q):
        return np.hypot(*(pp - q[:, None]).transpose(2, 0, 1)) / 20.0
    X[:, :p, 9] = dist(b0s)
    X[:, :p, 10] = dist(b1s)
    X[:, :p, 11] = dist(L)
    ar = np.arange(p)[None]
    X[:, :p, 12] = (ar == c0[:, None])
    X[:, :p, 13] = (ar == c1[:, None])
    X[:, :, 14] = np.log1p(dt0)[:, None]
    X[:, :, 15] = np.log1p(dt1)[:, None]
    X[:, p, 0:2] = L / sc
    X[:, p, 16] = 1.0                                            # free token
    X[:, :p, 17] = np.isnan(b0s[:, 0])[:, None] | np.isnan(b1s[:, 0])[:, None]
    mask = np.zeros((m, p + 1), bool)
    mask[:, :p] = np.isnan(pp[:, :, 0])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X[mask] = 0.0
    pos = np.concatenate([np.nan_to_num(pp), L[:, None]], axis=1).astype(np.float32)
    return X, mask, pos, idx


def make_model(f=F, d=64, layers=3):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(f, d), nn.GELU(), nn.Linear(d, d))
            enc = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.0, batch_first=True)
            self.enc = nn.TransformerEncoder(enc, layers)
            self.out = nn.Linear(d, 3)

        def forward(self, x, mask):
            h = self.enc(self.inp(x), src_key_padding_mask=mask)
            o = self.out(h)
            logit = o[..., 0].masked_fill(mask, -1e4)
            off = 3.0 * torch.tanh(o[..., 1:])
            return logit, off
    return Net()


def ball_of(logit, off, pos):
    import torch
    w = torch.softmax(logit, dim=-1)
    return (w[..., None] * (pos + off)).sum(1), w


def build_train(g):
    """training rows for one DFL match: gaps from seeds 1..N_SEEDS, every 2nd frame."""
    path = os.path.join(CACHE, "train_{}.npz".format(g))
    if os.path.isfile(path):
        return path
    import dfl_data as dd
    d, _, truth, blk = BF.dfl_match(g)
    out = {k: [] for k in ("X", "M", "P", "Y")}
    for s in range(1, N_SEEDS + 1):
        ball = dd.prepare_ball(d, s * 7919 + 13)
        X, M, P, idx = features(ball, d["players"], d["team"], blk, dd.PITCH)
        y = BF.to_m(truth[idx, :2], dd.PITCH)
        ok = np.isfinite(y[:, 0])
        sel = np.flatnonzero(ok)[::2]
        out["X"].append(X[sel].astype(np.float16))
        out["M"].append(M[sel])
        out["P"].append(P[sel].astype(np.float16))
        out["Y"].append(y[sel].astype(np.float32))
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, **{k: np.concatenate(v) for k, v in out.items()})
    return path


TOK = 48


def pad(X, M, P):
    """pad the player tokens to TOK - 1 (masked), free token kept last."""
    k = TOK - X.shape[1]
    if k <= 0:
        return X, M, P
    zx = np.zeros((len(X), k, X.shape[2]), X.dtype)
    zp = np.zeros((len(P), k, 2), P.dtype)
    zm = np.ones((len(M), k), bool)
    return (np.concatenate([X[:, :-1], zx, X[:, -1:]], 1), np.concatenate([M[:, :-1], zm, M[:, -1:]], 1),
            np.concatenate([P[:, :-1], zp, P[:, -1:]], 1))


def train(games, epochs, tag, seed=0):
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = [dict(np.load(build_train(g))) for g in games]
    for z in data:
        z["X"], z["M"], z["P"] = pad(z["X"], z["M"], z["P"])
    X = np.concatenate([z["X"] for z in data])
    M = np.concatenate([z["M"] for z in data])
    P = np.concatenate([z["P"] for z in data])
    Y = np.concatenate([z["Y"] for z in data])
    dev = "cuda"
    model = make_model().to(dev)
    bs = 1024
    steps = len(X) // bs
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * steps)
    t0 = time.time()
    for ep in range(epochs):
        perm = rng.permutation(len(X))
        tot = 0.0
        model.train()
        for k in range(steps):
            j = perm[k * bs:(k + 1) * bs]
            x = torch.from_numpy(X[j].astype(np.float32)).to(dev)
            m = torch.from_numpy(M[j]).to(dev)
            p = torch.from_numpy(P[j].astype(np.float32)).to(dev)
            y = torch.from_numpy(Y[j]).to(dev)
            logit, off = model(x, m)
            b, _ = ball_of(logit, off, p)
            dd_ = torch.linalg.norm(p - y[:, None], dim=-1).masked_fill(m, 1e4)
            dd_[:, -1] = 1e4
            near = dd_.argmin(1)
            near = torch.where(dd_.min(1).values <= 2.0, near, torch.full_like(near, p.shape[1] - 1))
            loss = torch.nn.functional.smooth_l1_loss(b, y) + 0.5 * torch.nn.functional.cross_entropy(logit, near)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            tot += float(loss.detach())
        print("  {} epoch {} loss {:.3f} ({:.0f} s)".format(tag, ep + 1, tot / max(steps, 1), time.time() - t0),
              flush=True)
    os.makedirs(MODELS, exist_ok=True)
    torch.save({"state": model.state_dict()}, os.path.join(MODELS, "ball_net_{}.pt".format(tag)))
    return model


def load(tag):
    import torch
    m = make_model().cuda()
    m.load_state_dict(torch.load(os.path.join(MODELS, "ball_net_{}.pt".format(tag)), weights_only=False)["state"])
    m.eval()
    return m


def fill(model, ball, players, team, blk, pitch):
    """(ball (n, 2) normalised with the gaps filled, p_home (n,) P(ball at a home player), NaN when seen)."""
    import torch
    from pipeline import features as pf
    X, M, P, idx = features(ball, players, team, blk, pitch)
    out = ball[:, :2].copy()
    ph = np.full(len(ball), np.nan, np.float32)
    home = np.concatenate([team == pf.HOME, [False]])
    away = np.concatenate([team == pf.AWAY, [False]])
    with torch.no_grad():
        for k in range(0, len(X), 4096):
            x = torch.from_numpy(X[k:k + 4096]).cuda()
            m = torch.from_numpy(M[k:k + 4096]).cuda()
            p = torch.from_numpy(P[k:k + 4096]).cuda()
            logit, off = model(x, m)
            b, w = ball_of(logit, off, p)
            w = w.cpu().numpy()
            out[idx[k:k + 4096]] = b.cpu().numpy() / np.array(pitch) + 0.5
            wh, wa = w[:, home].sum(1), w[:, away].sum(1)
            ph[idx[k:k + 4096]] = wh / np.maximum(wh + wa, 1e-6)
    return out, ph


def eval_match(g, model):
    import dfl_data as dd
    import cn_dfl
    d, ball, truth, blk = BF.dfl_match(g)
    with np.load(cn_dfl.path(g)) as z:
        live = z["lab"] <= 1
    gap = np.isnan(ball[:, 0]) & np.isfinite(truth[:, 0])
    inb = np.zeros(len(ball), bool)
    for s, e in blk:
        inb[s:e] = True
    gap &= inb
    P = BF.to_m(d["players"][:, :, :2], dd.PITCH)

    def near_team(xy):
        q = np.hypot(*(P - BF.to_m(xy, dd.PITCH)[:, None]).transpose(2, 0, 1))
        return d["team"][np.argmin(np.where(np.isnan(q), np.inf, q), axis=1)]
    tt = near_team(truth[:, :2])
    fills = {"linear": BF.fill_linear(ball, blk),
             "carrier": BF.fill_carrier(ball, d["players"], d["team"], blk, dd.PITCH),
             "net": fill(model, ball, d["players"], d["team"], blk, dd.PITCH)[0]}
    res = {}
    for nm, f in fills.items():
        err = np.hypot(*((BF.to_m(f, dd.PITCH) - BF.to_m(truth[:, :2], dd.PITCH))[gap]).T)
        res[nm] = (err, near_team(f)[gap] == tt[gap], live[gap])
    return res


def train_cmd(args):
    import dfl_data as dd
    games = [g for g in dd.GAMES if g != args.hold]
    train(games, args.epochs, args.hold)


def lomo_cmd(args):
    import dfl_data as dd
    from multiprocessing import Pool
    with Pool(4) as pool:
        pool.map(build_train, dd.GAMES)
    res = []
    for g in dd.GAMES:
        p = os.path.join(MODELS, "ball_net_{}.pt".format(g))
        model = load(g) if os.path.isfile(p) and not args.retrain else \
            train([h for h in dd.GAMES if h != g], args.epochs, g).eval()
        res.append(eval_match(g, model))
    print("DFL, ball in the test-seed gaps, each match by the model that never saw it")
    print("{:<8} {:>7} {:>6} {:>6} {:>10} | live: {:>6} {:>10}".format("filler", "median", "<2 m", "<5 m",
                                                                      "team right", "median", "team right"))
    for nm in res[0]:
        err = np.concatenate([r[nm][0] for r in res])
        same = np.concatenate([r[nm][1] for r in res])
        lv = np.concatenate([r[nm][2] for r in res])
        print("{:<8} {:>6.1f}m {:>5.0%} {:>5.0%} {:>9.1%} | {:>12.1f}m {:>9.1%}".format(
            nm, np.median(err), np.mean(err < 2), np.mean(err < 5), same.mean(), np.median(err[lv]), same[lv].mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("train", "lomo"))
    ap.add_argument("--hold", default="all")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()
    {"train": train_cmd, "lomo": lomo_cmd}[args.cmd](args)


if __name__ == "__main__":
    main()
