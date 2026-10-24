"""Temporal network for the learned evidence: a bidirectional GRU over the per-frame inputs.

    from pipeline.sequence import train_gru, predict_gru

Same inputs as pipeline/learned.py (matrix(f)), standardised with the training statistics. Trained
on random CROP-frame stretches of the training matches (each also home/away flipped), predicting
H/A/D at every graded frame. Output P(H, A, D) per frame, decoded by the same Viterbi.
Runs on the GPU when there is one.
"""

import numpy as np

from pipeline.decode import H, A, D, X
from pipeline.learned import matrix, flip

# Params
CROP = 750          # frames per training stretch (30 s)
BATCH = 32
HIDDEN = 64
LAYERS = 2
DROPOUT = 0.3
NOISE = 0.1         # Gaussian input noise (in standard deviations) while training
VAL_SHARE = 0.15    # the last part of each training match is held out to pick the best epoch
WINDOW = 1000       # inference window (subsampled frames), overlapping by WINDOW // 4 each side
EPOCH_STEPS = 300
EPOCHS = 10
LR = 2e-3
STEP = 2            # frame subsampling: the network sees 12.5 fps
SEED = 0


def _net(n_in):
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(n_in, HIDDEN), nn.GELU())
            self.gru = nn.GRU(HIDDEN, HIDDEN, num_layers=LAYERS, batch_first=True,
                              bidirectional=True, dropout=DROPOUT)
            self.out = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(2 * HIDDEN, 3))

        def forward(self, x):
            h, _ = self.gru(self.inp(x))
            return self.out(h)
    return Net()


def _prepare(f, labels=None):
    """[(X, y)] for a match and its flipped copy (y None when no labels), subsampled by STEP."""
    out = []
    pairs = [(f, labels), (flip(f), None if labels is None else
                           np.where(labels == H, A, np.where(labels == A, H, labels)))]
    for g, lab in pairs:
        mat = matrix(g)[0][::STEP]
        out.append((mat, None if lab is None else lab[::STEP]))
    return out


def train_gru(matches, epochs=EPOCHS, seed=SEED, log=print):
    """matches: [(f, labels)]. Returns (model, mean, std)."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seqs = [s for f, lab in matches for s in _prepare(f, lab)]
    allx = np.concatenate([x for x, _ in seqs])
    mean = np.nanmean(allx, axis=0)
    std = np.nanstd(allx, axis=0) + 1e-6
    norm = [(np.nan_to_num((x - mean) / std), y) for x, y in seqs]
    cut = [int(len(x) * (1 - VAL_SHARE)) for x, _ in norm]
    val = [(x[c:], y[c:]) for (x, y), c in zip(norm, cut)]
    model = _net(allx.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=epochs * EPOCH_STEPS)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=X)
    crop = CROP // STEP
    best = None
    for ep in range(epochs):
        model.train()
        total = 0.0
        for _ in range(EPOCH_STEPS):
            xb, yb = [], []
            for _ in range(BATCH):
                k = rng.integers(len(norm))
                x, y = norm[k]
                s = rng.integers(0, cut[k] - crop)
                xb.append(x[s:s + crop])
                yb.append(y[s:s + crop])
            xt = torch.tensor(np.stack(xb), dtype=torch.float32, device=dev)
            xt = xt + NOISE * torch.randn_like(xt)
            yt = torch.tensor(np.stack(yb), dtype=torch.long, device=dev)
            logits = model(xt)
            loss = loss_fn(logits.reshape(-1, 3), yt.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += float(loss.detach())
        vloss = _val_loss(model, val, loss_fn, dev)
        log("  epoch {:2d}  train loss {:.3f}  held-out loss {:.3f}".format(
            ep + 1, total / EPOCH_STEPS, vloss))
        if best is None or vloss < best[0]:
            best = (vloss, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep + 1)
    model.load_state_dict(best[1])
    log("  kept epoch {}".format(best[2]))
    return model, mean, std


def _val_loss(model, val, loss_fn, dev):
    import torch
    model.eval()
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for x, y in val:
            logits = _windowed(model, x, dev)
            yt = torch.tensor(y, dtype=torch.long, device=dev)
            n = int((yt != X).sum())
            if n:
                tot += float(loss_fn(logits, yt)) * n
                cnt += n
    model.train()
    return tot / max(cnt, 1)


def _windowed(model, x, dev):
    """Logits (len(x), 3) from overlapping windows; each frame taken from a window's middle."""
    import torch
    n = len(x)
    pad = WINDOW // 4
    out = torch.zeros((n, 3), device=dev)
    start = 0
    while start < n:
        a = max(0, start - pad)
        b = min(n, start + WINDOW - 2 * pad + pad)
        xt = torch.tensor(np.ascontiguousarray(x[a:b])[None], dtype=torch.float32, device=dev)
        logits = model(xt)[0]
        end = min(n, start + WINDOW - 2 * pad)
        out[start:end] = logits[start - a:end - a]
        start = end
    return out


def predict_gru(trained, f):
    """(n, 3) P(H, A, D) per video frame, averaged with the flipped prediction."""
    import torch
    model, mean, std = trained
    dev = next(model.parameters()).device
    model.eval()
    n = len(f["ball_seen"])
    outs = []
    with torch.no_grad():
        for k, (x, _) in enumerate(_prepare(f)):
            p = torch.softmax(_windowed(model, np.nan_to_num((x - mean) / std), dev),
                              dim=-1).cpu().numpy()
            if k == 1:
                p = p[:, [A, H, D]]
            outs.append(p)
    p = (outs[0] + outs[1]) / 2.0
    full = np.repeat(p, STEP, axis=0)[:n]
    if len(full) < n:
        full = np.concatenate([full, np.repeat(full[-1:], n - len(full), axis=0)])
    return full
