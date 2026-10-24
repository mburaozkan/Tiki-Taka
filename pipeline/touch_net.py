"""Learned touch detector: who touches the ball, frame by frame, from the blurred and gapped ball.

    from pipeline.touch_net import inputs, train, predict

Trained on DFL (tools/train_touch.py) with the ball blurred and gapped like ours, against touch
labels taken from DFL's sharp raw ball and the provider's flags (tools/touch_labels.py). The same
inputs() runs on our match, so the network transfers by construction of its input, not by hope.

Inputs per frame (inputs()), all from positions, so they exist for any tracking provider:
  ball      seen, x, y, vx, vy, speed, velocity change over +/- 3 and +/- 6 frames, time since
            the ball was last seen and until it is seen again, length of the current gap
  players   per team, the N_NEAR players nearest the ball: distance, dx, dy, velocity relative to
            the ball, speed; and how many are within 2 m and 5 m
  pace      mean player speed
flip=True swaps the two team blocks and mirrors the pitch (x and every x-velocity negated): the
same play seen with the teams' roles exchanged, used as augmentation and for the labels' H <-> A.

Model: a residual temporal convolution network (kernel 5, dilations 1-16, about +/- 2.5 s of
context), output logits for none / home touch / away touch per frame.
"""

import numpy as np

from pipeline.features import window_mean

# Params
FPS = 25
N_NEAR = 3
SMOOTH = 2              # +/- frames of position smoothing before velocities
CHANNELS = 64
DILATIONS = (1, 2, 4, 8, 16)
KERNEL = 5
DROPOUT = 0.1
CROP = 1000             # frames per training stretch (40 s)
BATCH = 32
STEPS = 250             # per epoch
EPOCHS = 12
LR = 2e-3
TOL_FRAMES = 2          # frames on either side of a touch that are trained as that touch
TOUCH_WEIGHT = 3.0      # class weight of the two touch classes
SEED = 0
# P(touch) peak needed to call a touch. Touch F1 is flat from 0.5 to 0.7 (0.74), but the decoder
# wants sure touches: DFL spellF1 0.553 at 0.5, 0.567 at 0.7, 0.577 at 0.85. Our match is less
# confident than DFL, and from 0.9 up too few touches survive there (possessions 237, 208), so 0.7.
PICK_THRESHOLD = 0.7
MODEL_PATH = "models/touch_net_all.pt"   # trained on all 7 DFL matches; committed for exact rebuilds


def _velocity(x, y):
    """Centred velocity (m/s) of metre tracks (n,), smoothed over +/- SMOOTH frames."""
    xs, ys = window_mean(x, SMOOTH), window_mean(y, SMOOTH)
    xs[np.isnan(x)], ys[np.isnan(y)] = np.nan, np.nan
    vx, vy = np.full(len(x), np.nan), np.full(len(x), np.nan)
    vx[1:-1], vy[1:-1] = (xs[2:] - xs[:-2]) * FPS / 2.0, (ys[2:] - ys[:-2]) * FPS / 2.0
    return vx, vy


def _gap_clocks(seen):
    """(seconds since the ball was last seen, seconds until it is seen again, gap length)."""
    n = len(seen)
    idx = np.where(seen, np.arange(n), -1)
    np.maximum.accumulate(idx, out=idx)
    since = np.where(idx >= 0, np.arange(n) - idx, n) / float(FPS)
    nxt = np.where(seen, np.arange(n), n)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    until = (nxt - np.arange(n)) / float(FPS)
    return since, until, since + until


def inputs(players, ball, team, pitch, flip=False):
    """(n, n_features) float32 per-frame inputs; NaN replaced by 0 (the seen flag says why)."""
    L, W = pitch
    n = len(ball)
    sign = -1.0 if flip else 1.0
    bx = (ball[:, 0].astype(np.float64) - 0.5) * L
    by = (ball[:, 1].astype(np.float64) - 0.5) * W
    seen = ~np.isnan(bx)
    bvx, bvy = _velocity(bx, by)
    dv = []
    for k in (3, 6):
        d = np.full(n, np.nan)
        d[k:-k] = np.hypot(bvx[2 * k:] - bvx[:-2 * k], bvy[2 * k:] - bvy[:-2 * k])
        dv.append(d)
    since, until, gap = _gap_clocks(seen)
    cols = [seen.astype(np.float64), sign * bx / 50, by / 34, sign * bvx / 10, bvy / 10,
            np.hypot(bvx, bvy) / 10, dv[0] / 10, dv[1] / 10,
            np.minimum(since, 5) / 5, np.minimum(until, 5) / 5, np.minimum(gap, 30) / 30]

    px = (players[:, :, 0].astype(np.float64) - 0.5) * L
    py = (players[:, :, 1].astype(np.float64) - 0.5) * W
    pvx, pvy = np.empty_like(px), np.empty_like(py)
    for s in range(px.shape[1]):
        pvx[:, s], pvy[:, s] = _velocity(px[:, s], py[:, s])
    speed = players[:, :, 2].astype(np.float64)
    dist = np.hypot(px - bx[:, None], py - by[:, None])
    order = (1, 0) if flip else (0, 1)
    for side in order:
        cols_side = np.flatnonzero(team == side)
        d = dist[:, cols_side]
        d_sorted = np.where(np.isnan(d), np.inf, d)
        near = np.argsort(d_sorted, axis=1)[:, :N_NEAR]
        rows = np.arange(n)[:, None]
        sel = cols_side[near]
        dd = d_sorted[rows, near]
        ok = np.isfinite(dd)
        for j in range(N_NEAR):
            s = sel[:, j]
            r = np.arange(n)
            okj = ok[:, j]
            cols += [np.where(okj, np.minimum(dd[:, j], 30) / 5, 0.0),
                     np.where(okj, sign * (px[r, s] - bx) / 5, 0.0),
                     np.where(okj, (py[r, s] - by) / 5, 0.0),
                     np.where(okj, sign * (pvx[r, s] - bvx) / 10, 0.0),
                     np.where(okj, (pvy[r, s] - bvy) / 10, 0.0),
                     np.where(okj, speed[r, s] / 10, 0.0)]
        cols += [np.sum(d_sorted <= 2.0, axis=1) / 3.0, np.sum(d_sorted <= 5.0, axis=1) / 5.0]
    ok_speed = ~np.isnan(speed)
    pace = np.where(ok_speed.any(axis=1),
                    np.where(ok_speed, speed, 0).sum(axis=1) / np.maximum(ok_speed.sum(axis=1), 1),
                    np.nan)
    cols.append(window_mean(pace, FPS) / 5)
    out = np.stack(cols, axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def targets(n, frames, teams, flip=False):
    """(n,) int64: 0 none, 1 home touch, 2 away touch, spread over +/- TOL_FRAMES (nearest wins)."""
    y = np.zeros(n, dtype=np.int64)
    best = np.full(n, TOL_FRAMES + 1)
    for f, t in zip(frames, teams):
        cls = (2 - t) if flip else (1 + t)
        for off in range(-TOL_FRAMES, TOL_FRAMES + 1):
            i = f + off
            if 0 <= i < n and abs(off) < best[i]:
                best[i], y[i] = abs(off), cls
    return y


def _net(n_in):
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.f = nn.Sequential(
                nn.Conv1d(CHANNELS, CHANNELS, KERNEL, padding=d * (KERNEL - 1) // 2, dilation=d),
                nn.GroupNorm(8, CHANNELS), nn.GELU(), nn.Dropout(DROPOUT),
                nn.Conv1d(CHANNELS, CHANNELS, 1))

        def forward(self, h):
            return h + self.f(h)

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Conv1d(n_in, CHANNELS, 1)
            self.blocks = nn.Sequential(*[Block(d) for d in DILATIONS])
            self.out = nn.Conv1d(CHANNELS, 3, 1)

        def forward(self, x):                      # x: (B, T, F)
            return self.out(self.blocks(self.inp(x.transpose(1, 2)))).transpose(1, 2)
    return TCN()


def train(examples, epochs=EPOCHS, seed=SEED, log=print):
    """Train on [(X (n, F), y (n,), valid (n,) bool)]; returns (model, mean, std)."""
    import torch
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    stacked = np.concatenate([x[v] for x, _, v in examples])
    mean, std = stacked.mean(axis=0), stacked.std(axis=0) + 1e-6
    data = [((x - mean) / std, y, v) for x, y, v in examples]
    lengths = np.array([len(y) for _, y, _ in data], dtype=np.float64)
    model = _net(len(mean)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=epochs * STEPS)
    weight = torch.tensor([1.0, TOUCH_WEIGHT, TOUCH_WEIGHT], device=dev)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for _ in range(STEPS):
            xb, yb, mb = [], [], []
            for _ in range(BATCH):
                i = int(rng.choice(len(data), p=lengths / lengths.sum()))
                x, y, v = data[i]
                a = int(rng.integers(0, len(y) - CROP))
                xb.append(x[a:a + CROP])
                yb.append(y[a:a + CROP])
                mb.append(v[a:a + CROP])
            xb = torch.tensor(np.stack(xb), device=dev)
            yb = torch.tensor(np.stack(yb), device=dev)
            mb = torch.tensor(np.stack(mb), device=dev, dtype=torch.float32)
            logits = model(xb)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 3), yb.reshape(-1),
                                                     weight=weight, reduction="none")
            loss = (loss * mb.reshape(-1)).sum() / mb.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        log("  epoch {:>2}  loss {:.4f}".format(epoch + 1, total / STEPS))
    return model, mean, std


def save(bundle, path):
    import torch
    model, mean, std = bundle
    torch.save({"state": model.state_dict(), "mean": mean, "std": std}, path)


def load(path):
    """(model, mean, std) from save(); on the GPU when there is one."""
    import torch
    saved = torch.load(path, weights_only=False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = _net(len(saved["mean"])).to(dev)
    model.load_state_dict(saved["state"])
    return model, saved["mean"], saved["std"]


def predict(model_bundle, x):
    """(n, 3) probabilities none / home / away for one sequence of inputs."""
    import torch
    model, mean, std = model_bundle
    dev = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(((x - mean) / std)[None], device=dev, dtype=torch.float32)
        return torch.softmax(model(xt)[0], dim=-1).cpu().numpy()


def pick(prob, threshold=0.5, half=3):
    """(frames, teams) of touch peaks: local maxima of P(touch) above threshold."""
    score = prob[:, 1] + prob[:, 2]
    local = score.copy()
    for s in range(1, half + 1):
        np.maximum(local[s:], score[:-s], out=local[s:])
        np.maximum(local[:-s], score[s:], out=local[:-s])
    frames = np.flatnonzero((score >= threshold) & (score == local))
    teams = (prob[frames, 2] > prob[frames, 1]).astype(np.int8)
    return frames, teams
