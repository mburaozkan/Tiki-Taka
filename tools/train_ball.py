"""Train the ball detector on the video crops and test it on the held-out minutes. Counts only.

    python tools/train_ball.py [--epochs 25]

Crops and labels from tools/vision_dataset.py. Held out: the last 20% of each half. Reports, on
held-out crops:
  positives  median pixel error of the heatmap peak, share within 3 / 6 px, and the share
             detected (peak probability >= threshold) at each THRESHOLDS
  negatives  share with a false detection (peak >= threshold) at each THRESHOLDS
The labels are the ATD ball (blurred, so not exact): errors of 1-3 px are partly the labels'.
Saves data/cache/vision/ball_net.pt; progress in data/cache/vision/train.log (live).
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import DATA_DIR  # noqa: E402
from pipeline import ball_net  # noqa: E402

# Params
VISION = os.path.join(DATA_DIR, "cache", "vision")
THRESHOLDS = (0.1, 0.3, 0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=ball_net.EPOCHS)
    args = parser.parse_args()
    log_path = os.path.join(VISION, "train.log")
    handle = open(log_path, "w", buffering=1)

    def log(msg):
        print(msg, flush=True)
        handle.write(msg + "\n")
    crops = np.load(os.path.join(VISION, "crops.npy"))          # ~3 GB into RAM, fast batches
    with np.load(os.path.join(VISION, "meta.npz")) as m:
        rows = m["rows"]
    points = rows[:, 3:5].astype(np.float32)
    val = rows[:, 5] > 0
    tr, te = np.flatnonzero(~val), np.flatnonzero(val)
    log("train {} crops, held out {}".format(len(tr), len(te)))
    model = ball_net.train(crops, points, tr, epochs=args.epochs, log=log)
    ball_net.save(model, os.path.join(VISION, "ball_net.pt"))

    pred = ball_net.predict_crops(model, crops, te)
    order = np.argsort(te)
    te_sorted = te[order]
    pos = points[te_sorted, 0] >= 0
    err = np.hypot(pred[:, 0] - points[te_sorted, 0], pred[:, 1] - points[te_sorted, 1])
    e = err[pos]
    log("\nheld-out positives ({}): median error {:.1f} px | within 3 px {:.0f}% | within 6 px "
        "{:.0f}%".format(pos.sum(), np.median(e), 100 * np.mean(e <= 3), 100 * np.mean(e <= 6)))
    for th in THRESHOLDS:
        hit = (pred[pos, 2] >= th) & (e <= 6)
        fa = pred[~pos, 2] >= th
        log("  threshold {:.1f}: ball found (within 6 px) {:.0f}% | false detection on crops "
              "without a ball {:.1f}%".format(th, 100 * hit.mean(), 100 * fa.mean()))


if __name__ == '__main__':
    main()
