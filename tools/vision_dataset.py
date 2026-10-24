"""Training crops for a ball detector on the tactical video. Local only; prints counts only.

    python tools/vision_dataset.py                 # one pass over the video, ~15-20 min

Labels come for free from the ATD ball projected through each frame's homography (pitch 0-1 ->
pixels, starter/load_homography.py). The ATD ball is blurred over ~0.24 s, which shifts it only
when its velocity changes, so positives are frames where it is seen on t-1, t, t+1 and moving
steadily (velocity change over +/- STEADY_K frames under STEADY_DV_MS).

Each sample is CROP x CROP pixels at full resolution from frames t-1, t, t+1 (9 channels):
  positive  the ball somewhere in the crop (random offset), heatmap target at its pixel
  negative  a crop of the same frames at least NEG_MIN_PX from the ball (one ball per frame)
Split by time: the last VAL_SHARE of each half is held out.

Writes data/cache/vision/{crops.npy (uint8, N x 9 x CROP x CROP), meta.npz}. The video, the
crops and the positions never leave this machine; the script prints only counts.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, DATA_DIR, asset_paths, half_video_blocks  # noqa: E402

# Params
OUT = os.path.join(DATA_DIR, "cache", "vision")
CROP = 160
N_POS = 9000
NEG_PER_POS = 0.5
NEG_MIN_PX = 120
EDGE_PX = 8                 # the ball at least this far inside the crop
STEADY_K = 6
STEADY_DV_MS = 1.5
VAL_SHARE = 0.2
SEED = 0
W_PX, H_PX = 1920, 1080


def ball_pixels():
    """(px, py, usable) for every frame: the ATD ball projected through the homography."""
    from atd_cache import load_arrays
    from load_homography import load
    ball = load_arrays("tactical")["ball"]
    H, err, valid = load()
    n = len(ball)
    u, v = ball[:, 0].astype(np.float64), ball[:, 1].astype(np.float64)
    ok = ~np.isnan(u) & (valid[:n] > 0)
    px, py = np.full(n, np.nan), np.full(n, np.nan)
    Hn = H[:n]
    den = Hn[:, 2, 0] * u + Hn[:, 2, 1] * v + Hn[:, 2, 2]
    px[ok] = ((Hn[:, 0, 0] * u + Hn[:, 0, 1] * v + Hn[:, 0, 2]) / den)[ok]
    py[ok] = ((Hn[:, 1, 0] * u + Hn[:, 1, 1] * v + Hn[:, 1, 2]) / den)[ok]
    return px, py, ok, ball


def steady(ball):
    """Frames where the ball's velocity barely changes over +/- STEADY_K frames."""
    L, W = GAMES[GAME]["pitch"]
    x, y = ball[:, 0].astype(np.float64) * L, ball[:, 1].astype(np.float64) * W
    vx, vy = np.gradient(x) * 25, np.gradient(y) * 25
    k = STEADY_K
    dv = np.full(len(x), np.inf)
    dv[k:-k] = np.hypot(vx[2 * k:] - vx[:-2 * k], vy[2 * k:] - vy[:-2 * k])
    return np.nan_to_num(dv, nan=np.inf) < STEADY_DV_MS


def plan():
    """Sample list: (frame, crop_x0, crop_y0, ball_x_in_crop or -1, ball_y_in_crop, is_val)."""
    rng = np.random.default_rng(SEED)
    px, py, ok, ball = ball_pixels()
    n = len(px)
    seen3 = ok.copy()
    seen3[1:-1] = ok[:-2] & ok[1:-1] & ok[2:]
    inside = (px > EDGE_PX) & (px < W_PX - EDGE_PX) & (py > EDGE_PX) & (py < H_PX - EDGE_PX)
    val = np.zeros(n, bool)
    in_half = np.zeros(n, bool)
    for _, first, last in half_video_blocks(GAME):
        in_half[first:last + 1] = True
        val[int(last - VAL_SHARE * (last - first)):last + 1] = True
    cand = np.flatnonzero(seen3 & inside & steady(ball) & in_half)
    pick = np.sort(rng.choice(cand, size=min(N_POS, len(cand)), replace=False))
    rows = []
    for f in pick:
        bx, by = px[f], py[f]
        ox = int(np.clip(bx - rng.uniform(EDGE_PX, CROP - EDGE_PX), 0, W_PX - CROP))
        oy = int(np.clip(by - rng.uniform(EDGE_PX, CROP - EDGE_PX), 0, H_PX - CROP))
        rows.append((f, ox, oy, bx - ox, by - oy, val[f]))
        if rng.random() < NEG_PER_POS:
            for _ in range(20):
                nx, ny = int(rng.integers(0, W_PX - CROP)), int(rng.integers(0, H_PX - CROP))
                cx, cy = nx + CROP / 2, ny + CROP / 2
                if np.hypot(cx - bx, cy - by) > NEG_MIN_PX + CROP / 2:
                    rows.append((f, nx, ny, -1.0, -1.0, val[f]))
                    break
    return np.array(rows, dtype=np.float64), len(cand)


def main():
    import cv2
    os.makedirs(OUT, exist_ok=True)
    rows, n_cand = plan()
    frames = rows[:, 0].astype(np.int64)
    need = np.unique(np.concatenate([frames - 1, frames, frames + 1]))
    crops = np.lib.format.open_memmap(os.path.join(OUT, "crops.npy"), mode="w+", dtype=np.uint8,
                                      shape=(len(rows), 9, CROP, CROP))
    by_frame = {}
    for i, f in enumerate(frames):
        for k, g in enumerate((f - 1, f, f + 1)):
            by_frame.setdefault(int(g), []).append((i, k))
    cap = cv2.VideoCapture(asset_paths()["video"])
    last = int(need.max())
    t, j = 0, 0
    while t <= last:
        ok, img = cap.read()
        if not ok:
            break
        if t in by_frame:
            rgb = img[:, :, ::-1]
            for i, k in by_frame[t]:
                ox, oy = int(rows[i, 1]), int(rows[i, 2])
                crops[i, 3 * k:3 * k + 3] = rgb[oy:oy + CROP, ox:ox + CROP].transpose(2, 0, 1)
            j += 1
        t += 1
        if t % 20000 == 0:
            print("  {:>6} / {} frames read".format(t, last + 1), flush=True)
    crops.flush()
    np.savez(os.path.join(OUT, "meta.npz"), rows=rows, crop=CROP)
    pos = rows[:, 3] >= 0
    print("candidates (ball seen, steady, inside the frame): {}".format(n_cand))
    print("samples: {} positive, {} negative | held out: {} positive, {} negative".format(
        int(np.sum(pos & (rows[:, 5] == 0))), int(np.sum(~pos & (rows[:, 5] == 0))),
        int(np.sum(pos & (rows[:, 5] == 1))), int(np.sum(~pos & (rows[:, 5] == 1)))))


if __name__ == '__main__':
    main()
