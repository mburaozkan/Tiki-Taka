"""RF-DETR ball detector (SoccerNet fine-tune, Hugging Face julianzu9612/RFDETR-Soccernet) on the
tactical video's public window. Local only; prints aggregates only.

    python tools/detr_ball.py run          # detections -> data/cache/yolo/window_rfdetr.npz
    python tools/detr_ball.py compare      # vs the YOLO ball (bambezius) on the same frames
    python tools/detr_ball.py touch [model.pt ...]   # touch nets on the window, tracking vs cleaned RF-DETR ball

Model: RF-DETR Large (DINOv2), trained at 560 px; checkpoint in external/rfdetr_sn/weights (pickle
scanned: argparse.Namespace, OrderedDict and tensors only; loaded with weights_only=True). The full
1920-wide frame shrunk to 560 would leave the ball 1-2 px, so each frame is cropped to 560 x 560 at
full resolution around the reference point: the tracking ball projected into the image (homography),
else the last centre. The ball box nearest the reference (within RADIUS px) else the most confident.

compare (kill rule agreed with the user, public window +-5 s):
  1. jitter: per method, ball on the pitch (m) minus its 2-frame Gaussian smooth, median size, on
     frames both methods detect (and their neighbours); go on only if RF-DETR's is about half
     YOLO's or less;
  2. #7 / #10 / #12 / #15: largest velocity change within +-0.5 s of the true start and its rank
     among velocity-change peaks within +-5 s, raw and after the physics filter
     (tools/ball_denoise.py, lam 4); go on only if #12's survives the filter.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "starter"), os.path.join(ROOT, "evaluation")]

OUT = os.path.join(ROOT, "data", "cache", "yolo")
CKPT = os.path.join(ROOT, "external", "rfdetr_sn", "weights", "checkpoint_best_regular.pth")
FPS = 25
CROP = 560
RADIUS = 80.0
CONF = 0.05
STARTS = (7, 10, 12, 15)


OUTLIER_M = 0.5
CLEAN_LAM = 0.5


def clean_ball(xy_m, lam=CLEAN_LAM):
    """(n, 2) metres, NaN gaps -> single-frame outliers (> OUTLIER_M from the 5-frame median) replaced
    by the median, then the physics filter (ball_denoise, lam). Used on our RF-DETR ball and, in
    train_touch --rfdetr, on DFL's simulated one."""
    from scipy.ndimage import median_filter
    from ball_denoise import filter_ball
    xy = xy_m.copy()
    seen = np.isfinite(xy[:, 0])
    edges = np.flatnonzero(np.diff(np.concatenate([[0], seen.astype(np.int8), [0]])))
    for a0, b0 in zip(edges[::2], edges[1::2]):
        seg = xy[a0:b0]
        med = np.column_stack([median_filter(seg[:, k], size=5, mode="nearest") for k in (0, 1)])
        out = np.hypot(*(seg - med).T) > OUTLIER_M
        seg[out] = med[out]
    return filter_ball(xy, lam)


def load_model():
    import argparse as ap_
    import torch
    from rfdetr import RFDETRLarge
    torch.serialization.add_safe_globals([ap_.Namespace])
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    m = RFDETRLarge(pretrain_weights=None, num_classes=3, resolution=CROP)
    m.model.model.load_state_dict(ck["model"], strict=True)
    m.model.model.cuda().eval()
    m.model.device = torch.device("cuda")
    return m


def run_cmd(args):
    import cv2
    from PIL import Image
    from games import asset_paths
    import vision_dataset as V
    from yolo_ball import public_frames
    a, b = public_frames()
    px, py, ok, _ = V.ball_pixels()
    m = load_model()
    cap = cv2.VideoCapture(asset_paths()["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    det = np.full((b - a, 3), np.nan)
    centre = None
    for i in range(b - a):
        good, img = cap.read()
        if not good:
            break
        h, w = img.shape[:2]
        f = a + i
        ref = (px[f], py[f]) if (ok[f] and np.isfinite(px[f])) else None
        if ref is not None:
            centre = ref
        c = centre if centre is not None else (w / 2, h / 2)
        x0 = int(np.clip(c[0] - CROP / 2, 0, w - CROP))
        y0 = int(np.clip(c[1] - CROP / 2, 0, h - CROP))
        crop = Image.fromarray(cv2.cvtColor(img[y0:y0 + CROP, x0:x0 + CROP], cv2.COLOR_BGR2RGB))
        d = m.predict(crop, threshold=CONF)
        k = np.flatnonzero(d.class_id == 0)
        if len(k):
            cx = (d.xyxy[k, 0] + d.xyxy[k, 2]) / 2 + x0
            cy = (d.xyxy[k, 1] + d.xyxy[k, 3]) / 2 + y0
            j = k[np.argmax(d.confidence[k])]
            jj = int(np.argmax(d.confidence[k]))
            r = ref if ref is not None else c
            dd = np.hypot(cx - r[0], cy - r[1])
            if dd.min() <= RADIUS:
                jj = int(np.argmin(dd))
            det[i] = (cx[jj], cy[jj], d.confidence[k][jj])
        if i % 1000 == 0:
            print("  {} / {}".format(i, b - a), flush=True)
    np.savez(os.path.join(OUT, "window_rfdetr.npz"), a=a, b=b, det=det)
    print("frames {}; ball detected {:.0%}".format(b - a, np.mean(np.isfinite(det[:, 0]))))


def to_pitch(det, a, b):
    from load_homography import load as load_h
    from yolo_ball import pitch_of
    from games import GAMES, GAME
    Hm, _, _ = load_h()
    L, W = GAMES[GAME]["pitch"]
    return pitch_of(det[:, 0], det[:, 1], Hm[a:b].astype(np.float64)) * np.array([L, W])


def jitter(xy, use):
    """median |xy - 2-frame Gaussian smooth| over frames in use whose +-2 neighbours are all seen."""
    from scipy.ndimage import gaussian_filter1d
    seen = np.isfinite(xy[:, 0])
    full = seen.copy()
    for s in (1, 2):
        full[s:] &= seen[:-s]
        full[:-s] &= seen[s:]
    sm = gaussian_filter1d(np.nan_to_num(xy), 2.0, axis=0)
    r = np.hypot(*(xy - sm).T)
    k = full & use
    return np.median(r[k]), int(k.sum())


def kinks(xy, a, sp):
    from ball_denoise import dv_of
    from scipy.ndimage import maximum_filter1d
    d = np.nan_to_num(dv_of(xy))
    pk = np.flatnonzero((d >= 1) & (d == maximum_filter1d(d, 7)))
    out = []
    for i in STARTS:
        s = sp[i - 1][0] - a
        w = pk[np.abs(pk - s) <= 12]
        if not len(w):
            out.append("none")
            continue
        p = w[np.argmax(d[w])]
        around = pk[np.abs(pk - p) <= 125]
        out.append("{:.1f} r{}/{}".format(d[p], 1 + int(np.sum(d[around] > d[p])), len(around)))
    return " | ".join(out)


def compare_cmd(args):
    from ball_actions import public_spells
    from ball_denoise import filter_ball
    z = np.load(os.path.join(OUT, "window_rfdetr.npz"))
    a, b = int(z["a"]), int(z["b"])
    dr = z["det"]
    y = np.load(os.path.join(OUT, "public_bambezius.npz"))
    dy = y["det"][a - int(y["a"]):b - int(y["a"])].copy()
    dy[~(dy[:, 2] >= 0.1)] = np.nan                    # as yolo_ball.py refine
    xr, xyo = to_pitch(dr, a, b), to_pitch(dy, a, b)
    gt, sp = public_spells()
    both = np.isfinite(xr[:, 0]) & np.isfinite(xyo[:, 0])
    print("public window +-5 s, {} frames: ball detected RF-DETR {:.0%}, YOLO {:.0%}, both {:.0%}; "
          "median distance between them {:.2f} m".format(b - a, np.mean(np.isfinite(xr[:, 0])),
                                                         np.mean(np.isfinite(xyo[:, 0])), both.mean(),
                                                         np.median(np.hypot(*(xr - xyo)[both].T))))
    jr, nr = jitter(xr, both)
    jy, ny = jitter(xyo, both)
    print("1. jitter (|ball - 2-frame smooth|, median, frames both see): RF-DETR {:.3f} m ({} fr), "
          "YOLO {:.3f} m ({} fr); ratio {:.2f}".format(jr, nr, jy, ny, jr / jy))
    print("2. largest velocity change within +-0.5 s of #" + " / #".join(map(str, STARTS))
          + " (m/s), rank among peaks within +-5 s")
    for nm, xy in (("YOLO", xyo), ("RF-DETR", xr)):
        print("   {:<8} raw      {}".format(nm, kinks(xy, a, sp)))
        print("   {:<8} lam 4    {}".format(nm, kinks(filter_ball(xy, 4.0), a, sp)))


def touch_cmd(args):
    """touch networks on the public window with the cleaned RF-DETR ball (only where the tracking ball
    is seen) vs the tracking ball: new team's touch P at STARTS, touch peaks (P >= 0.5) per minute of
    seen tracking ball inside the graded window."""
    from ball_actions import public_spells
    from atd_cache import load_arrays
    from games import GAMES, GAME
    from pipeline import touch_net as tn
    import pipeline.features as features
    L, W = GAMES[GAME]["pitch"]
    z = np.load(os.path.join(OUT, "window_rfdetr.npz"))
    a, b = int(z["a"]), int(z["b"])
    rf = clean_ball(to_pitch(z["det"], a, b))
    atd = load_arrays("tactical")
    P, B0, T = atd["players"], atd["ball"], atd["team"]
    seen = np.isfinite(B0[a:b, 0])
    print("cleaned RF-DETR ball: jitter {:.3f} m".format(jitter(rf, seen)[0]))
    bb = B0.copy()
    seg = bb[a:b].copy()
    w = seen & np.isfinite(rf[:, 0])
    seg[w, :2] = rf[w] / np.array([L, W])
    bb[a:b] = seg
    gt, sp = public_spells()
    s0, s1 = sp[0][0], sp[-1][1]
    mins = np.isfinite(B0[s0:s1, 0]).sum() / 1500
    lo, hi = a - 200, b + 200
    print("{:<26} {:<9} {:>22} {:>7}".format("model", "ball", "P at #" + "/#".join(map(str, STARTS)), "pk/min"))
    for mp in args.models or [tn.MODEL_PATH]:
        m = tn.load(mp if os.path.isabs(mp) else os.path.join(ROOT, mp))
        path = len(m[1]) > tn.n_base_inputs()
        for nm, ball in (("tracking", B0), ("rfdetr", bb)):
            prob = tn.predict(m, tn.inputs(P[lo:hi], ball[lo:hi], T, (features.LENGTH, features.WIDTH), path=path))
            ps = ["{:.2f}".format(prob[sp[i - 1][0] - lo - 12:sp[i - 1][0] - lo + 13, 1 + sp[i - 1][2]].max())
                  for i in STARTS]
            fr = tn.pick(prob, 0.5)[0] + lo
            print("{:<26} {:<9} {:>22} {:>7.1f}".format(os.path.basename(mp), nm, "/".join(ps),
                                                        np.sum((fr >= s0) & (fr < s1)) / mins))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("run", "compare", "touch"))
    ap.add_argument("models", nargs="*")
    args = ap.parse_args()
    {"run": run_cmd, "compare": compare_cmd, "touch": touch_cmd}[args.cmd](args)


if __name__ == "__main__":
    main()
