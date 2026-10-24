"""Review material for the ball detector, written locally for a person to look at. Never shared.

    python tools/review_ball.py [--clip-start S] [--clip-seconds 60]

Writes to data/cache/vision/review/ (gitignored with the rest of data/):
  crops_contact_sheet.png   48 held-out crops (frame t of each), 3x upscaled: blue circle = the
                            ATD ball (the label), red cross = the detector's peak, with its
                            probability; rows: random positives, the worst misses, crops without
                            a ball (any red cross there is a false alarm)
  review_clip.mp4           CLIP_SECONDS of a held-out stretch at full resolution: ATD ball in blue,
                            the detector in red (when its peak passes THRESHOLD), and "ATD: no ball"
                            on frames where the ATD has none
These files contain match video: they stay on this machine (competition terms). The script prints
only file names and counts.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from games import GAME, DATA_DIR, asset_paths, half_video_blocks  # noqa: E402
from pipeline import ball_net  # noqa: E402

# Params
VISION = os.path.join(DATA_DIR, "cache", "vision")
OUT = os.path.join(VISION, "review")
THRESHOLD = 0.3
SCALE = 3
PAD = 8
BLUE, RED, WHITE, YELLOW = (255, 120, 0), (0, 0, 255), (255, 255, 255), (0, 220, 255)


def contact_sheet(model, crops, rows):
    import cv2
    val = np.flatnonzero(rows[:, 5] > 0)
    pos = val[rows[val, 3] >= 0]
    neg = val[rows[val, 3] < 0]
    pred_pos = ball_net.predict_crops(model, crops, pos)
    err = np.hypot(pred_pos[:, 0] - rows[pos, 3], pred_pos[:, 1] - rows[pos, 4])
    rng = np.random.default_rng(0)
    pick_random = rng.choice(len(pos), size=16, replace=False)
    pick_worst = np.argsort(-err)[:16]
    pick_neg = rng.choice(len(neg), size=16, replace=False)
    pred_neg = ball_net.predict_crops(model, crops, neg[np.sort(pick_neg)])
    tiles = []
    for label, idx_list, preds in (("random", pos[np.sort(pick_random)], pred_pos[np.sort(pick_random)]),
                                   ("worst", pos[pick_worst], pred_pos[pick_worst]),
                                   ("no ball", neg[np.sort(pick_neg)], pred_neg)):
        for i, p in zip(idx_list, preds):
            img = np.ascontiguousarray(crops[i, 3:6].transpose(1, 2, 0)[:, :, ::-1])
            img = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
            if rows[i, 3] >= 0:
                cv2.circle(img, (int(rows[i, 3] * SCALE), int(rows[i, 4] * SCALE)), 12, BLUE, 2)
            x, y = int(p[0] * SCALE), int(p[1] * SCALE)
            color = RED if p[2] >= THRESHOLD else YELLOW
            cv2.drawMarker(img, (x, y), color, cv2.MARKER_CROSS, 18, 2)
            cv2.putText(img, "{} p={:.2f}".format(label, p[2]), (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
            tiles.append(cv2.copyMakeBorder(img, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT))
    rows_img = [np.hstack(tiles[k:k + 8]) for k in range(0, len(tiles), 8)]
    sheet = np.vstack(rows_img)
    path = os.path.join(OUT, "crops_contact_sheet.png")
    cv2.imwrite(path, sheet)
    return path


def clip(model, start_frame, seconds):
    import cv2
    import torch
    sys.path.insert(0, os.path.join(ROOT, "starter"))
    from vision_dataset import ball_pixels
    px, py, ok, _ = ball_pixels()
    cap = cv2.VideoCapture(asset_paths()["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame - 1))
    n = int(seconds * 25)
    path = os.path.join(OUT, "review_clip.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (1920, 1080))
    dev = next(model.parameters()).device
    frames = []
    ok_read, img = cap.read()
    frames.append(img)
    found = gaps = found_in_gap = 0
    for k in range(n + 1):
        ok_read, img = cap.read()
        if not ok_read:
            break
        frames.append(img)
        if len(frames) < 3:
            continue
        f = start_frame + k
        stack = np.concatenate([fr[:, :, ::-1].transpose(2, 0, 1) for fr in frames[-3:]], 0)
        x = np.zeros((9, 1088, 1920), dtype=np.float32)
        x[:, :1080] = stack / 255.0
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16,
                                             enabled=dev.type == "cuda"):
            heat = torch.sigmoid(model(torch.tensor(x[None], device=dev)).float())[0, :1080]
        v, idx = heat.flatten().max(0)
        p, j = float(v), int(idx)
        out = frames[-2].copy()
        if ok[f]:
            cv2.circle(out, (int(px[f]), int(py[f])), 14, BLUE, 2)
        else:
            gaps += 1
            cv2.putText(out, "ATD: no ball", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, WHITE, 3)
        if p >= THRESHOLD:
            found += 1
            found_in_gap += not ok[f]
            cv2.drawMarker(out, (j % 1920, j // 1920), RED, cv2.MARKER_CROSS, 26, 3)
        cv2.putText(out, "detector p={:.2f}".format(p), (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    RED if p >= THRESHOLD else YELLOW, 2)
        writer.write(out)
        frames = frames[-2:]
    writer.release()
    return path, found, gaps, found_in_gap, n


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clip-start", type=float, default=None,
                        help="seconds into the held-out part of the first half")
    parser.add_argument("--clip-seconds", type=float, default=60.0)
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    model = ball_net.load(os.path.join(VISION, "ball_net.pt"))
    crops = np.load(os.path.join(VISION, "crops.npy"), mmap_mode="r")
    with np.load(os.path.join(VISION, "meta.npz")) as m:
        rows = m["rows"]
    print("wrote", contact_sheet(model, crops, rows))
    _, first, last = half_video_blocks(GAME)[0]
    held_from = int(last - 0.2 * (last - first))
    start = held_from + int((args.clip_start or 60.0) * 25)
    path, found, gaps, found_in_gap, n = clip(model, start, args.clip_seconds)
    print("wrote {} | frames {} | detector fired on {} | ATD without ball on {} | detector fired "
          "on {} of those".format(path, n, found, gaps, found_in_gap))


if __name__ == '__main__':
    main()
