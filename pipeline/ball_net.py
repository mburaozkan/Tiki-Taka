"""Ball detector for the tactical video: a small U-Net over three consecutive frames.

    from pipeline.ball_net import train, load, detect

Input: frames t-1, t, t+1 as 9 channels (RGB each), any size divisible by 8. Output: a heatmap
logit per pixel at full resolution; its peak is the ball at frame t. Fully convolutional, so it
is trained on crops (tools/vision_dataset.py) and run on whole frames.

Loss: CenterNet focal loss on a Gaussian target (SIGMA_PX) around the labelled pixel, the usual
choice for one tiny object per image.
"""

import numpy as np

# Params
SIGMA_PX = 2.0
WIDTHS = (32, 64, 128, 128)
EPOCHS = 25
BATCH = 32
LR = 1e-3
SEED = 0
LOG_EVERY = 50


def _net():
    import torch
    import torch.nn as nn

    def block(a, b, dil=1):
        return nn.Sequential(nn.Conv2d(a, b, 3, padding=dil, dilation=dil), nn.BatchNorm2d(b),
                             nn.ReLU(inplace=True), nn.Conv2d(b, b, 3, padding=dil, dilation=dil),
                             nn.BatchNorm2d(b), nn.ReLU(inplace=True))

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            w1, w2, w3, w4 = WIDTHS
            self.e1, self.e2, self.e3 = block(9, w1), block(w1, w2), block(w2, w3)
            self.mid = nn.Sequential(block(w3, w4, 2), block(w4, w4, 4))
            self.d3, self.d2, self.d1 = block(w4 + w3, w3), block(w3 + w2, w2), block(w2 + w1, w1)
            self.pool = nn.MaxPool2d(2)
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.head = nn.Conv2d(w1, 1, 1)
            self.head.bias.data.fill_(-4.0)

        def forward(self, x):
            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2))
            m = self.mid(self.pool(e3))
            d3 = self.d3(torch.cat([self.up(m), e3], 1))
            d2 = self.d2(torch.cat([self.up(d3), e2], 1))
            d1 = self.d1(torch.cat([self.up(d2), e1], 1))
            return self.head(d1)[:, 0]
    return UNet()


def heatmaps(points, size):
    """(N, size, size) Gaussian targets; a point (-1, -1) is an empty map. Peak pixel exactly 1."""
    yy, xx = np.mgrid[0:size, 0:size]
    out = np.zeros((len(points), size, size), dtype=np.float32)
    for i, (x, y) in enumerate(points):
        if x < 0:
            continue
        g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * SIGMA_PX ** 2))
        g[int(round(min(max(y, 0), size - 1))), int(round(min(max(x, 0), size - 1)))] = 1.0
        out[i] = g
    return out


def focal_loss(logits, target):
    import torch
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    neg = 1.0 - pos
    pos_loss = -torch.log(p) * (1 - p) ** 2 * pos
    neg_loss = -torch.log(1 - p) * p ** 2 * (1 - target) ** 4 * neg
    n_pos = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


def train(crops, points, train_idx, epochs=EPOCHS, log=print, seed=SEED):
    """crops: (N, 9, S, S) uint8; points: (N, 2) ball pixel in crop or -1. Mixed precision on the
    GPU (fp16 autocast, loss in fp32); logs every LOG_EVERY steps with timing."""
    import time
    import torch
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    size = crops.shape[-1]
    model = _net().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    steps = max(1, len(train_idx) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=epochs * steps)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        order = rng.permutation(train_idx)
        total = 0.0
        for s in range(steps):
            idx = np.sort(order[s * BATCH:(s + 1) * BATCH])
            x = np.asarray(crops[idx], dtype=np.float32) / 255.0
            pts = points[idx].copy()
            # augment: horizontal flip, time reversal, brightness
            flip = rng.random(len(idx)) < 0.5
            x[flip] = x[flip][:, :, :, ::-1]
            pts[flip, 0] = np.where(pts[flip, 0] >= 0, size - 1 - pts[flip, 0], -1)
            rev = rng.random(len(idx)) < 0.5
            x[rev] = x[rev][:, [6, 7, 8, 3, 4, 5, 0, 1, 2]]
            x *= rng.uniform(0.8, 1.2, size=(len(idx), 1, 1, 1)).astype(np.float32)
            y = heatmaps(pts, size)
            xt = torch.tensor(np.ascontiguousarray(x), device=dev)
            yt = torch.tensor(y, device=dev)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
                logits = model(xt)
            loss = focal_loss(logits.float(), yt)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
            if (s + 1) % LOG_EVERY == 0:
                log("    epoch {:>2} step {:>4}/{}  loss {:.3f}  {:.0f}s".format(
                    ep + 1, s + 1, steps, total / (s + 1), time.time() - t0))
        log("  epoch {:>2}/{}  loss {:.3f}  {:.0f}s".format(ep + 1, epochs, total / steps,
                                                         time.time() - t0))
    return model


def predict_crops(model, crops, idx, batch=128):
    """(peak x, peak y, peak probability) per crop."""
    import torch
    dev = next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16,
                                         enabled=dev.type == "cuda"):
        for s in range(0, len(idx), batch):
            j = np.sort(idx[s:s + batch])
            x = torch.tensor(np.asarray(crops[j], dtype=np.float32) / 255.0, device=dev)
            p = torch.sigmoid(model(x).float())
            flat = p.flatten(1)
            v, k = flat.max(1)
            size = p.shape[-1]
            out.append(np.stack([(k % size).cpu().numpy(), (k // size).cpu().numpy(),
                                 v.cpu().numpy()], 1))
    return np.concatenate(out)


def save(model, path):
    import torch
    torch.save({"state": model.state_dict(), "widths": WIDTHS}, path)


def load(path):
    import torch
    saved = torch.load(path, weights_only=False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = _net().to(dev)
    model.load_state_dict(saved["state"])
    model.eval()
    return model
