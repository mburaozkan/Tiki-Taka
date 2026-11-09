"""SoccerTrack v2 probe: can RF-DETR find the ball in its panoramic video? Aggregates only.

    python tools/soccertrack_probe.py [--match 117092] [--minutes 10] [--conf 0.3]

Data: data/external/soccertrack (Hugging Face atomscott/soccertrack-v2, v1.2, CC BY 4.0). Video
frame k = annotation frame k, 25 fps, 3840 x 1906 panorama.

At every ball-action event in the first --minutes of the 1st half (all classes; the ball track is
exact at events, interpolated between them, so only event frames are used as truth): the whole
panorama is scanned in TILE x TILE crops at full resolution (no prior from the labels), the most
confident ball box is mapped to the pitch through a piecewise-linear map fitted on the 65
hand-annotated pitch keypoints (raw/<match>_keypoints.json), and compared with the labelled ball.
Prints: found (a ball box >= --conf), distance to the label (median, within 1 / 2 / 5 m), and
whether any ball box at all lies within 2 m (the best a smarter picker could do).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
ST = os.path.join(ROOT, "data", "external", "soccertrack")
TILE = 560
Y_RANGE = (400, 1620)
X_RANGE = (150, 3750)
STEP = 500
CHUNK = 16


def keypoint_maps(match):
    """(pixel -> pitch, pitch -> pixel) piecewise-linear maps; pitch in the ball file's centred
    metres (x right, y towards the near touchline)."""
    from scipy.interpolate import LinearNDInterpolator
    kp = json.load(open(os.path.join(ST, "raw", match, "{}_keypoints.json".format(match))))
    P = np.array([[float(t) for t in k.strip("()").split(",")] for k in kp])
    V = np.array([[float(v[0]), float(v[1])] for v in kp.values()])
    near_is_68 = V[P[:, 1] == 68, 1].mean() > V[P[:, 1] == 0, 1].mean()   # larger pixel y = nearer the camera
    C = np.column_stack([P[:, 0] - 52.5, (P[:, 1] - 34.0) * (1 if near_is_68 else -1)])
    return LinearNDInterpolator(V, C), LinearNDInterpolator(C, V)


def scan_area(match, w, h, margin=120):
    """pixel box around the pitch keypoints (+ margin), clipped to the frame."""
    kp = json.load(open(os.path.join(ST, "raw", match, "{}_keypoints.json".format(match))))
    V = np.array([[float(v[0]), float(v[1])] for v in kp.values()])
    global X_RANGE, Y_RANGE
    X_RANGE = (max(int(V[:, 0].min()) - margin, 0), min(int(V[:, 0].max()) + margin, w))
    Y_RANGE = (max(int(V[:, 1].min()) - margin, 0), min(int(V[:, 1].max()) + margin, h))


def tiles(crop=TILE):
    step = int(crop * STEP / TILE)
    xs = list(range(X_RANGE[0], X_RANGE[1] - crop + 1, step)) + [X_RANGE[1] - crop]
    ys = list(range(Y_RANGE[0], Y_RANGE[1] - crop + 1, step)) + [Y_RANGE[1] - crop]
    return [(x, y) for y in sorted(set(ys)) for x in sorted(set(xs))]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--match", default="117092")
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--crop", type=int, default=TILE, help="source crop size, resized to TILE (280 = 2x zoom)")
    ap.add_argument("--max-frames", type=int, default=0, help="use only this many event frames (evenly spaced)")
    ap.add_argument("--half", default="1st", choices=("1st", "2nd"))
    args = ap.parse_args()
    import cv2
    from PIL import Image
    from detr_ball import load_model
    to_pitch, _ = keypoint_maps(args.match)
    ev = json.load(open(os.path.join(ST, "bas", args.match, "{}_12_class_events.json".format(args.match))))["actions"]
    z = np.load(os.path.join(ST, "ball", "{}_{}_ball.npz".format(args.match, args.half)))
    frame_of = {int(f): i for i, f in enumerate(z["frame"])}
    half1 = os.path.join(ST, "videos", args.match, "{}_panorama_{}_half.mp4".format(args.match, args.half))
    nmax = int(args.minutes * 60 * 25)
    cap = cv2.VideoCapture(half1)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scan_area(args.match, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frames = sorted({int(e["frame"]) for e in ev if int(e["frame"]) < min(nmax, total)
                     and int(e["frame"]) in frame_of})
    if args.max_frames and len(frames) > args.max_frames:
        frames = [frames[i] for i in np.linspace(0, len(frames) - 1, args.max_frames).astype(int)]
    T = tiles(args.crop)
    print("{} event frames ({} half, first {:.0f} min); {} tiles of {} px per frame, zoom x{:.1f}".format(
        len(frames), args.half, args.minutes, len(T), args.crop, TILE / args.crop), flush=True)
    m = load_model()
    err, found, near_any = [], 0, 0
    _, to_px = keypoint_maps(args.match)
    near = {"3 m": 0, "5 m": 0, "40 px": 0, "80 px": 0}
    saved = []
    for k, f in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        crops = [Image.fromarray(cv2.cvtColor(cv2.resize(img[y:y + args.crop, x:x + args.crop], (TILE, TILE)),
                                              cv2.COLOR_BGR2RGB)) for x, y in T]
        zf = args.crop / TILE
        dets = []
        for c0 in range(0, len(crops), CHUNK):           # chunks: a whole frame's tiles at once ran out of RAM
            r = m.predict(crops[c0:c0 + CHUNK], threshold=0.05)
            dets.extend(r if isinstance(r, list) else [r])
        cand = []
        for (x0, y0), d in zip(T, dets):
            for j in np.flatnonzero(d.class_id == 0):
                cx = (d.xyxy[j, 0] + d.xyxy[j, 2]) / 2 * zf + x0
                cy = (d.xyxy[j, 1] + d.xyxy[j, 3]) / 2 * zf + y0
                cand.append((float(d.confidence[j]), cx, cy))
        i = frame_of[f]
        truth = np.array([z["x"][i], z["y"][i]], float)
        if cand:
            pts = np.array([c[1:] for c in cand])
            pitch = to_pitch(pts)
            dist = np.hypot(*(pitch - truth).T)
            dmin = np.nanmin(np.where(np.isfinite(dist), dist, np.inf))
            near_any += bool(dmin <= 2.0)
            near["3 m"] += bool(dmin <= 3.0)
            near["5 m"] += bool(dmin <= 5.0)
            tpx = to_px(truth[None])[0]
            dpx = np.hypot(*(pts - tpx).T).min() if np.all(np.isfinite(tpx)) else np.inf
            near["40 px"] += bool(dpx <= 40)
            near["80 px"] += bool(dpx <= 80)
            saved.append(np.column_stack([np.full(len(cand), f), np.array(cand)]))
            conf_on = np.where(np.isfinite(dist), [c[0] for c in cand], -1.0)   # only boxes on the pitch map
            b = int(np.argmax(conf_on))
            if conf_on[b] >= args.conf:
                found += 1
                err.append(dist[b])
        if k % 50 == 0:
            print("  {} / {}".format(k, len(frames)), flush=True)
    err = np.array(err, float)
    n = len(frames)
    print("found (best ball box >= {}): {:.0%} of {} event frames".format(args.conf, found / n, n))
    if len(err):
        ok = np.isfinite(err)
        print("best box vs labelled ball: median {:.2f} m; within 1 m {:.0%}, 2 m {:.0%}, 5 m {:.0%} of found "
              "({} outside the keypoint map)".format(np.nanmedian(err), np.mean(err[ok] <= 1), np.mean(err[ok] <= 2),
                                                   np.mean(err[ok] <= 5), int((~ok).sum())))
    print("any ball box within 2 m of the label: {:.0%} of event frames".format(near_any / n))
    print("any ball box near the label: " + ", ".join("{} {:.0%}".format(k, v / n) for k, v in near.items()))
    print("(best box = most confident ball box that maps onto the pitch)")
    if saved:
        np.save(os.path.join(ROOT, "data", "cache", "soccertrack_probe_{}_c{}.npy".format(args.match, args.crop)),
                np.concatenate(saved))


if __name__ == "__main__":
    main()
