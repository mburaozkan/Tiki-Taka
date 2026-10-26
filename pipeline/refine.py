"""Second-stage team model: a bidirectional GRU that re-decides the team from the first stage's
team log-odds plus the team-reaction features, over long context (experiment 6, step 3).

    from pipeline.refine import inputs, train, predict

We label offline, so the network may read the future: teams reorganise 1-8 s after a turnover,
and the second stage can move a change of possession to where that reaction says it happened.
Per frame inputs (inputs()), averaged over BIN frames (5 fps), home/away symmetric (flip swaps the
team columns and negates the signed ones):
  z            first-stage log-odds of home vs away (clipped), and whether the hand path is live
  ball         seen, gap length, nearest home / away distance (capped)
  touches      touch network P(home touch), P(away touch), picked touch side, last touch side
  reaction     per team: spread, centroid x, push, back line, compactness (pipeline/reaction.py)
               and their 2 / 4 / 6 s future-minus-past changes; pressure and mean speed
Target: the true team on live bins. Output: P(home) per frame (bins repeated), to be decoded by
learned.decode_hybrid like any team probabilities.
"""

import numpy as np

# Params
BIN = 5
HIDDEN = 64
LAYERS = 2
CROP = 600              # bins per training stretch (2 min)
BATCH = 16
STEPS = 1500
LR = 2e-3
SEED = 0
Z_CLIP = 8.0
REACTION = ("spread", "cx", "push", "back", "compact")
DIFFS = ("", "_d2", "_d4", "_d6")


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
    home, away = _team_cols(cols, "home"), _team_cols(cols, "away")
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


def _net(n_in):
    import torch.nn as nn

    class Refiner(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(n_in, HIDDEN)
            self.gru = nn.GRU(HIDDEN, HIDDEN, num_layers=LAYERS, batch_first=True,
                              bidirectional=True, dropout=0.1)
            self.out = nn.Linear(2 * HIDDEN, 1)

        def forward(self, x):
            h, _ = self.gru(self.inp(x).relu())
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
    data = [((x - mean) / std, y, v.astype(np.float32)) for x, y, v in examples]
    lengths = np.array([len(y) for _, y, _ in data], dtype=np.float64)
    model = _net(len(mean)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps)
    total = 0.0
    for step in range(steps):
        model.train()
        xb, yb, mb = [], [], []
        for _ in range(BATCH):
            i = int(rng.choice(len(data), p=lengths / lengths.sum()))
            x, y, v = data[i]
            a = int(rng.integers(0, max(1, len(y) - CROP)))
            xb.append(x[a:a + CROP])
            yb.append(y[a:a + CROP])
            mb.append(v[a:a + CROP])
        xb = torch.tensor(np.stack(xb), device=dev)
        yb = torch.tensor(np.stack(yb), device=dev)
        mb = torch.tensor(np.stack(mb), device=dev)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(xb), yb, reduction="none")
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


def predict(bundle, x_plain, x_flip, n_frames):
    """(n_frames,) P(home), averaged with the flipped prediction, bins repeated to frames."""
    import torch
    model, mean, std = bundle
    dev = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(torch.tensor((x_plain - mean) / std, device=dev)[None]))[0]
        q = torch.sigmoid(model(torch.tensor((x_flip - mean) / std, device=dev)[None]))[0]
    pb = (p.cpu().numpy() + 1.0 - q.cpu().numpy()) / 2.0
    out = np.repeat(pb, BIN)
    if len(out) < n_frames:
        out = np.concatenate([out, np.full(n_frames - len(out), out[-1] if len(out) else 0.5)])
    return out[:n_frames]
