"""Second-stage team model: a bidirectional GRU that re-decides the team from the first stage's
team log-odds plus the team-reaction features, over long context (experiment 6, step 3).

    from pipeline.refine import inputs, train, predict

We label offline, so the network may read the future: teams reorganise 1-8 s after a turnover,
and the second stage can move a change of possession to where that reaction says it happened.
Per frame inputs (inputs()), averaged over BIN frames (BIN 5: 5 fps; BIN 1: the full 25 fps, with
CONV convolution layers in front of the GRU for the fine timing), home/away symmetric (flip swaps the
team columns and negates the signed ones):
  z            first-stage log-odds of home vs away (clipped), and whether the hand path is live
  ball         seen, gap length, nearest home / away distance (capped)
  touches      touch network P(home touch), P(away touch), picked touch side, last touch side
  reaction     per team: spread, centroid x, push, back line, compactness (pipeline/reaction.py)
               and their 2 / 4 / 6 s future-minus-past changes; pressure and mean speed
Target: the true team on live bins. Output: P(home) per frame (bins repeated), to be decoded by
learned.decode_hybrid like any team probabilities.

Dead-ball head (dead_inputs / dead_targets / decode_dead): the same network on the same inputs
(optionally + the first stage's P(dead)) trained to say whether the ball is dead; predict(...,
symmetric=True). decode_dead re-decides D vs live with a two-state Viterbi over beta x its log-odds
+ gamma for agreeing with the hand path, and `cost` per change.
"""

import numpy as np

# Params
BIN = 5
HIDDEN = 64
LAYERS = 2
CONV = 0                # 5-tap conv layers (dilation 1, 2, 4, ...) before the GRU
CROP_S = 120.0          # seconds per training stretch
PLAYERS = 0             # per-player inputs: the K players per team nearest the ball (0 = off)
PLAYER_FEATS = 9        # see player_inputs()
PLAYER_EMBED = 32
PREDICT_S = 240.0       # prediction windows (s); each keeps its centre half, so >= 60 s context
BATCH = 16
STEPS = 1500
LR = 2e-3
SEED = 0
Z_CLIP = 8.0
REACTION = ("spread", "cx", "push", "back", "compact")
DIFFS = ("", "_d2", "_d4", "_d6")
USE_REACTION = True     # False: the ablation without the team-reaction inputs


def _team_cols(cols, side):
    out = [cols["r_{}_{}{}".format(n, side, d)] for n in REACTION for d in DIFFS]
    out += [cols["press_" + side], cols["speed_" + side]]
    return [np.asarray(x, dtype=np.float64) for x in out]


def inputs(prob, hand, f, cols, flip=False, floor=1e-4):
    """(n_bins, F) float32 inputs; see the module docstring."""
    ph = np.clip(prob[:, 0], floor, 1.0)
    pa = np.clip(prob[:, 1], floor, 1.0)
    z = np.clip(np.log(ph / pa), -Z_CLIP, Z_CLIP)
    live = np.isin(hand, (0, 1)).astype(np.float64)
    seen = np.asarray(f["ball_seen"], dtype=np.float64)
    gap = np.minimum(np.asarray(f["gap_s"], dtype=np.float64), 30.0) / 30.0
    dh = np.minimum(np.nan_to_num(np.asarray(f["d_home"], np.float64), nan=30.0), 30.0) / 10.0
    da = np.minimum(np.nan_to_num(np.asarray(f["d_away"], np.float64), nan=30.0), 30.0) / 10.0
    th = np.asarray(f["touch_p_home"], np.float64)
    ta = np.asarray(f["touch_p_away"], np.float64)
    touch = np.asarray(f["touch"], np.float64)
    last = np.asarray(f["last_touch"], np.float64)
    home, away = ((_team_cols(cols, "home"), _team_cols(cols, "away")) if USE_REACTION
                  else ([], []))
    sign = -1.0 if flip else 1.0
    if flip:
        dh, da, th, ta, home, away = da, dh, ta, th, away, home
    feats = [sign * z, live, seen, gap, dh, da, th, ta, sign * touch, sign * last] + home + away
    x = np.stack([np.nan_to_num(v) for v in feats], axis=1)
    n = len(x) // BIN * BIN
    return x[:n].reshape(-1, BIN, x.shape[1]).mean(axis=1).astype(np.float32)


def targets(lab, flip=False):
    """(y, valid) per bin: 1 = home has it (away when flipped), on bins whose frames are all live."""
    n = len(lab) // BIN * BIN
    b = lab[:n].reshape(-1, BIN)
    home = np.all(b == 0, axis=1)
    away = np.all(b == 1, axis=1)
    y = (away if flip else home).astype(np.float32)
    return y, home | away


def dead_inputs(prob, hand, f, cols, flip=False, use_pd=True, floor=1e-4):
    """inputs() + (use_pd) the first stage's log-odds of dead vs live."""
    x = inputs(prob, hand, f, cols, flip, floor)
    if not use_pd:
        return x
    pd = np.clip(prob[:, 2], floor, 1.0)
    z = np.clip(np.log(pd / np.clip(prob[:, 0] + prob[:, 1], floor, 1.0)), -Z_CLIP, Z_CLIP)
    n = len(z) // BIN * BIN
    zb = z[:n].reshape(-1, BIN).mean(axis=1).astype(np.float32)
    return np.concatenate([x, zb[:, None]], axis=1)


def dead_targets(lab, flip=False):
    """(y, valid) per bin: 1 = dead, on bins whose frames are all dead or all live."""
    n = len(lab) // BIN * BIN
    b = lab[:n].reshape(-1, BIN)
    dead = np.all(b == 2, axis=1)
    live = np.all((b == 0) | (b == 1), axis=1)
    return dead.astype(np.float32), dead | live


def decode_dead(hand, p_dead, blocks, beta, gamma, cost, step=5, floor=1e-4):
    """Hand-like codes (H for live, D, X outside blocks) from a two-state live / dead Viterbi per
    block on `step`-frame bins: dead scores beta * logit(p_dead) + gamma (hand D) or - gamma (hand
    live), live 0, `cost` per change (in frames' units: scores are summed over each bin)."""
    out = np.full(len(hand), 3, dtype=np.int8)
    q = np.clip(p_dead, floor, 1 - floor)
    sc = beta * np.log(q / (1 - q)) + gamma * np.where(hand == 2, 1.0, -1.0)
    for a, b in blocks:
        n = (b - a + step - 1) // step
        seg = np.zeros(n * step)
        seg[:b - a] = sc[a:b]
        e = seg.reshape(n, step).sum(axis=1).tolist()
        # two-state Viterbi, state 1 = dead
        t0, t1 = 0.0, e[0]
        back = bytearray(n)
        for i in range(1, n):
            prev0 = 0 if t0 >= t1 - cost else 1          # best predecessor of live
            prev1 = 1 if t1 >= t0 - cost else 0          # best predecessor of dead
            back[i] = prev0 | (prev1 << 1)
            t0, t1 = (t0 if prev0 == 0 else t1 - cost), (t1 if prev1 == 1 else t0 - cost) + e[i]
        path = np.empty(n, dtype=np.int8)
        st = 1 if t1 > t0 else 0
        for i in range(n - 1, -1, -1):
            path[i] = st
            st = (back[i] >> st) & 1
        out[a:b] = np.where(np.repeat(path, step)[:b - a] == 1, 2, 0)
    return out


def _hold(v):
    """NaNs filled with the last seen value (then the first seen one)."""
    ok = ~np.isnan(v)
    if not ok.any():
        return np.zeros_like(v)
    idx = np.where(ok, np.arange(len(v)), -1)
    np.maximum.accumulate(idx, out=idx)
    first = int(np.flatnonzero(ok)[0])
    return v[np.where(idx >= 0, idx, first)]


def player_inputs(players, team, ball, groups, pitch, k=None, flip=False):
    """(n_bins, 2, k, PLAYER_FEATS) float16: per team (home first, away first when flipped) the k
    players nearest the ball at the first frame of every bin. Per player, in the team's own
    attacking direction (from its mean position per group, as pipeline/reaction.py):
      x up the pitch / 50 m, y / 34 m, x and y relative to the ball / 10 m, velocity toward goal
      and sideways / 5 m/s, distance to the ball / 10 m (cap 5), to the nearest opponent / 5 m
      (cap 4), and 1 for a real player (0 = padding).
    The ball is held at its last seen position through gaps (inputs() has the gap flags)."""
    import warnings
    k = k or PLAYERS
    L, W = pitch
    n = len(players)
    frames = np.arange(0, n // BIN * BIN, BIN)
    px = (players[:, :, 0].astype(np.float64) - 0.5) * L
    py = (players[:, :, 1].astype(np.float64) - 0.5) * W
    vx, vy = np.full_like(px, np.nan), np.full_like(py, np.nan)
    vx[2:-2] = (px[4:] - px[:-4]) * 25 / 4.0
    vy[2:-2] = (py[4:] - py[:-4]) * 25 / 4.0
    bx = _hold((ball[:, 0].astype(np.float64) - 0.5) * L)
    by = _hold((ball[:, 1].astype(np.float64) - 0.5) * W)
    px, py, vx, vy, bx, by = px[frames], py[frames], vx[frames], vy[frames], bx[frames], by[frames]
    g = np.asarray(groups)[frames]
    out = np.zeros((len(frames), 2, k, PLAYER_FEATS), dtype=np.float16)
    sides = (1, 0) if flip else (0, 1)
    for slot, s_ in enumerate(sides):
        mine, theirs = team == s_, team != s_
        x, y = px[:, mine], py[:, mine]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cx = np.nanmean(x, axis=1)
            sign = np.zeros(len(frames))
            for gg in np.unique(g):
                sel = g == gg
                m = np.nanmean(cx[sel]) if np.any(~np.isnan(cx[sel])) else 0.0
                sign[sel] = 1.0 if m < 0 else -1.0
            ox, oy = px[:, theirs], py[:, theirs]
            dopp = np.full(x.shape, np.nan)
            for a in range(0, len(frames), 5000):
                d = np.hypot(x[a:a + 5000, :, None] - ox[a:a + 5000, None, :],
                             y[a:a + 5000, :, None] - oy[a:a + 5000, None, :])
                dopp[a:a + 5000] = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=2)
        dball = np.hypot(x - bx[:, None], y - by[:, None])
        order = np.argsort(np.where(np.isnan(dball), np.inf, dball), axis=1)[:, :k]

        def take(a):
            return np.take_along_axis(a, order, axis=1)
        sg = sign[:, None]
        feats = [sg * take(x) / 50.0, take(y) / 34.0, sg * (take(x) - bx[:, None]) / 10.0,
                 (take(y) - by[:, None]) / 10.0, sg * take(vx[:, mine]) / 5.0,
                 take(vy[:, mine]) / 5.0, np.minimum(take(dball) / 10.0, 5.0),
                 np.minimum(take(dopp) / 5.0, 4.0)]
        valid = ~np.isnan(take(x))
        f = np.stack(feats + [valid.astype(np.float64)], axis=-1)
        f = np.where(np.isfinite(f), f, 0.0)
        f[~valid] = 0.0
        width = min(k, f.shape[1])
        out[:, slot, :width] = f[:, :width].astype(np.float16)
    return out


def config():
    return {"BIN": BIN, "HIDDEN": HIDDEN, "LAYERS": LAYERS, "CONV": CONV, "PLAYERS": PLAYERS}


def _net(n_in, cfg=None):
    import torch
    import torch.nn as nn
    cfg = cfg or config()
    hidden, layers, conv = cfg["HIDDEN"], cfg["LAYERS"], cfg["CONV"]
    k = cfg.get("PLAYERS", 0)

    class Refiner(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(n_in, hidden)
            if k:
                self.player = nn.Sequential(nn.Linear(PLAYER_FEATS, PLAYER_EMBED), nn.ReLU(),
                                            nn.Linear(PLAYER_EMBED, PLAYER_EMBED), nn.ReLU())
                self.mix = nn.Linear(hidden + 4 * PLAYER_EMBED, hidden)
            self.convs = nn.ModuleList([nn.Conv1d(hidden, hidden, 5, padding=2 * 2 ** i,
                                                  dilation=2 ** i) for i in range(conv)])
            self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                              bidirectional=True, dropout=0.1)
            self.out = nn.Linear(2 * hidden, 1)

        def forward(self, x, pl=None):
            h = self.inp(x).relu()
            if k:
                # set encoder: the same small network for every player, masked mean and max
                e = self.player(pl)                                # (B, T, 2, k, E)
                m = pl[..., -1:]
                mean = (e * m).sum(3) / m.sum(3).clamp(min=1.0)
                mx = (e + (m - 1.0) * 1e4).max(3).values * (m.sum(3) > 0)
                team = torch.cat([mean, mx], dim=-1).flatten(2)    # (B, T, 4E)
                h = self.mix(torch.cat([h, team], dim=-1)).relu()
            if len(self.convs):
                c = h.transpose(1, 2)
                for layer in self.convs:
                    c = c + layer(c).relu()
                h = c.transpose(1, 2)
            h, _ = self.gru(h)
            return self.out(h).squeeze(-1)
    return Refiner()


def train(examples, steps=STEPS, seed=SEED, log=print):
    """examples: [(x (bins, F), y, valid)]; returns (model, mean, std)."""
    import torch
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    stacked = np.concatenate([e[0] for e in examples])
    mean, std = stacked.mean(axis=0), stacked.std(axis=0) + 1e-6
    mean[0], std[0] = 0.0, 1.0           # keep the sign of z (symmetric under flip)
    data = [((e[0] - mean) / std, e[1], e[2].astype(np.float32), e[3] if len(e) > 3 else None)
            for e in examples]
    lengths = np.array([len(d[1]) for d in data], dtype=np.float64)
    model = _net(len(mean)).to(dev)
    model.cfg = config()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps)
    total = 0.0
    for step in range(steps):
        model.train()
        xb, yb, mb, pb = [], [], [], []
        for _ in range(BATCH):
            i = int(rng.choice(len(data), p=lengths / lengths.sum()))
            x, y, v, pl = data[i]
            crop = int(CROP_S * 25 / BIN)
            a = int(rng.integers(0, max(1, len(y) - crop)))
            xb.append(x[a:a + crop])
            yb.append(y[a:a + crop])
            mb.append(v[a:a + crop])
            if pl is not None:
                pb.append(pl[a:a + crop].astype(np.float32))
        xb = torch.tensor(np.stack(xb), device=dev)
        yb = torch.tensor(np.stack(yb), device=dev)
        mb = torch.tensor(np.stack(mb), device=dev)
        pt = torch.tensor(np.stack(pb), device=dev) if pb else None
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(xb, pt), yb,
                                                                    reduction="none")
        loss = (loss * mb).sum() / mb.sum().clamp(min=1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        total += loss.item()
        if (step + 1) % 500 == 0:
            log("  step {}  loss {:.4f}".format(step + 1, total / 500))
            total = 0.0
    return model, mean, std


def save(bundle, path):
    import torch
    model, mean, std = bundle
    torch.save({"state": model.state_dict(), "mean": mean, "std": std,
                "cfg": getattr(model, "cfg", config())}, path)


def load(path):
    import torch
    saved = torch.load(path, weights_only=False,
                       map_location=None if torch.cuda.is_available() else "cpu")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = saved.get("cfg", {"BIN": 5, "HIDDEN": 64, "LAYERS": 2, "CONV": 0, "PLAYERS": 0})
    model = _net(len(saved["mean"]), cfg).to(dev)
    model.load_state_dict(saved["state"])
    model.cfg = cfg
    return model, saved["mean"], saved["std"]


def predict(bundle, x_plain, x_flip, n_frames, p_plain=None, p_flip=None, symmetric=False):
    """(n_frames,) P(home), averaged with the flipped prediction, bins repeated to frames.
    p_plain / p_flip: player_inputs() for a model trained with them. symmetric: the target does
    not swap with the teams (the dead-ball head)."""
    import torch
    model, mean, std = bundle
    dev = next(model.parameters()).device
    model.eval()
    step = int(PREDICT_S * 25 / getattr(model, "cfg", config())["BIN"])
    pad = step // 4

    def run(x, pl):
        # overlapping windows (cuDNN rejects very long sequences); keep each window's centre
        x = (x - mean) / std
        out = np.empty(len(x), dtype=np.float32)
        with torch.no_grad():
            for a in range(0, len(x), step - 2 * pad):
                lo, hi = max(0, a - pad), min(len(x), a + step - pad)
                seg = torch.tensor(np.ascontiguousarray(x[lo:hi]), device=dev)[None]
                ps = (torch.tensor(pl[lo:hi].astype(np.float32), device=dev)[None]
                      if pl is not None else None)
                p = torch.sigmoid(model(seg, ps))[0].cpu().numpy()
                keep_hi = min(len(x), a + step - 2 * pad)
                out[a:keep_hi] = p[a - lo:keep_hi - lo]
        return out

    other = run(x_flip, p_flip)
    pb = (run(x_plain, p_plain) + (other if symmetric else 1.0 - other)) / 2.0
    out = np.repeat(pb, getattr(model, "cfg", config())["BIN"])
    if len(out) < n_frames:
        out = np.concatenate([out, np.full(n_frames - len(out), out[-1] if len(out) else 0.5)])
    return out[:n_frames]
