"""Pretrained YOLO ball detectors (Hugging Face) on the tactical video. Local only; prints aggregates.

    python tools/yolo_ball.py probe [--n 300]                 # score the models vs the tracking ball
    python tools/yolo_ball.py run --model bambezius --imgsz 1920   # public window: detections, compare
    python tools/yolo_ball.py render --model bambezius        # private/yolo_public_<model>.mp4 (user)

Models (models/yolo/<name>/, pickles scanned before use): uisikdag (YOLOv8, football players +
ball), bambezius (YOLO11n, SoccerNet ball), acatorcini (YOLOv9, soccer ball). Reference: the ATD
(tracking) ball projected into the frame through the homography (tools/vision_dataset.ball_pixels).
probe: frames across the match with the tracking ball seen and inside the frame; per model and image
size: found (any ball detection), within 10 / 25 px of the tracking ball, median distance, far
detections (> 50 px). run: every frame of the public window (+-5 s); the best ball detection per
frame -> pitch (inverse homography) -> speed and velocity change; vs the tracking ball: coverage,
median distance (m), velocity-change peaks (> 4 m/s within +-3 frames) found by YOLO with none in
the tracking within 0.5 s, and the reverse. Cached in data/cache/yolo/. The video, frames and
positions never leave this machine.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

MODELS = {"uisikdag": "models/yolo/uisikdag/best.pt", "bambezius": "models/yolo/bambezius/best.pt",
          "acatorcini": "models/yolo/acatorcini/v9b_best.pt"}
OUT = os.path.join(ROOT, "data", "cache", "yolo")
FPS = 25


def load_model(name):
    from ultralytics import YOLO
    m = YOLO(os.path.join(ROOT, MODELS[name]))
    ball = [k for k, v in m.names.items() if "ball" in str(v).lower()]
    return m, ball


def best_ball(res, ball_ids, ref=None, radius=80.0):
    """(x, y, conf) of the most confident ball box (with ref = (x, y): the box nearest ref within
    radius px if any, else the most confident), or NaNs."""
    b = res.boxes
    if b is None or len(b) == 0:
        return np.nan, np.nan, 0.0
    cls = b.cls.cpu().numpy().astype(int)
    conf = b.conf.cpu().numpy()
    xy = b.xywh.cpu().numpy()[:, :2]
    sel = np.isin(cls, ball_ids)
    if not sel.any():
        return np.nan, np.nan, 0.0
    i = np.flatnonzero(sel)[np.argmax(conf[sel])]
    if ref is not None and np.isfinite(ref[0]):
        js = np.flatnonzero(sel)
        dd = np.hypot(xy[js, 0] - ref[0], xy[js, 1] - ref[1])
        if dd.min() <= radius:
            i = js[np.argmin(dd)]
    return float(xy[i, 0]), float(xy[i, 1]), float(conf[i])


def read_frames(frames):
    import cv2
    from games import asset_paths
    cap = cv2.VideoCapture(asset_paths()["video"])
    out = {}
    frames = sorted(set(int(f) for f in frames))
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if ok:
            out[f] = img
    return out


def probe_cmd(args):
    import vision_dataset as V
    rng = np.random.default_rng(0)
    px, py, ok, _ = V.ball_pixels()
    inside = ok & (px > 10) & (px < V.W_PX - 10) & (py > 10) & (py < V.H_PX - 10)
    cand = np.flatnonzero(inside)
    frames = np.sort(rng.choice(cand, args.n, replace=False))
    imgs = read_frames(frames)
    print("{} frames with the tracking ball in view".format(len(imgs)))
    print("{:<11} {:>6} | {:>6} {:>7} {:>7} {:>8} {:>6} | conf>=0.25: found / within 10 px".format(
        "model", "imgsz", "found", "<10px", "<25px", "median", ">50px"))
    for name in MODELS:
        m, ball_ids = load_model(name)
        for imgsz in (1280, 1920):
            d, cf = [], []
            for f in frames:
                if f not in imgs:
                    continue
                r = m.predict(imgs[f], imgsz=imgsz, conf=0.05, verbose=False, device=0)[0]
                x, y, c = best_ball(r, ball_ids)
                d.append(np.hypot(x - px[f], y - py[f]) if np.isfinite(x) else np.inf)
                cf.append(c)
            d, cf = np.array(d), np.array(cf)
            fin = np.isfinite(d)
            hi = cf >= 0.25
            print("{:<11} {:>6} | {:>5.0%} {:>6.0%} {:>6.0%} {:>7.1f}px {:>5.0%} | {:.0%} / {:.0%}".format(
                name, imgsz, fin.mean(), np.mean(d < 10), np.mean(d < 25),
                np.median(d[fin]) if fin.any() else np.nan, np.mean(fin & (d > 50)),
                hi.mean(), np.mean(hi & (d < 10))))


def public_frames(pad_s=5):
    import possession_metrics as pm
    from report import PUBLIC_GT
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    idx = np.flatnonzero((gt >= 0) & (gt <= 2))        # graded frames only (X outside the window)
    return max(idx[0] - pad_s * FPS, 0), idx[-1] + pad_s * FPS


def pitch_of(px, py, Hm):
    """pixel -> pitch (0-1) through the inverse of each frame's homography."""
    out = np.full((len(px), 2), np.nan)
    for i in range(len(px)):
        if not np.isfinite(px[i]):
            continue
        try:
            Hi = np.linalg.inv(Hm[i])
        except np.linalg.LinAlgError:
            continue
        v = Hi @ np.array([px[i], py[i], 1.0])
        out[i] = v[:2] / v[2]
    return out


def vel_change(xy_m, k=3):
    v = np.gradient(xy_m, axis=0) * FPS
    dv = np.full(len(xy_m), np.nan)
    dv[k:-k] = np.hypot(*(v[2 * k:] - v[:-2 * k]).T)
    return np.hypot(*v.T), dv


def run_cmd(args):
    import cv2
    from games import asset_paths, GAMES, GAME
    from load_homography import load as load_h
    import vision_dataset as V
    a, b = public_frames()
    px, py, ok, ball = V.ball_pixels()
    cache = os.path.join(OUT, "public_{}.npz".format(args.model))
    if os.path.isfile(cache):                     # detections already made (a full-match pass)
        z = np.load(cache)
        a0 = int(z["a"])
        det = z["det"][a - a0:b - a0]
    else:
        m, ball_ids = load_model(args.model)
        cap = cv2.VideoCapture(asset_paths()["video"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, a)
        det = np.full((b - a, 3), np.nan)
        for i in range(b - a):
            ok_, img = cap.read()
            if not ok_:
                break
            r = m.predict(img, imgsz=args.imgsz, conf=args.conf, verbose=False, device=0)[0]
            det[i] = best_ball(r, ball_ids, (px[a + i], py[a + i]) if args.near else None)
    Hm, _, valid = load_h()
    xy = pitch_of(det[:, 0], det[:, 1], Hm[a:b])
    L, W = GAMES[GAME]["pitch"]
    trk = ball[a:b, :2].astype(np.float64)
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, "window_{}.npz".format(args.model)), a=a, b=b, det=det, xy=xy)
    ym = xy * np.array([L, W])
    tm = trk * np.array([L, W])
    sy, dvy = vel_change(ym)
    st, dvt = vel_change(tm)
    both = np.isfinite(ym[:, 0]) & np.isfinite(tm[:, 0])
    dist = np.hypot(*(ym - tm)[both].T)
    pix = np.hypot(det[:, 0] - px[a:b], det[:, 1] - py[a:b])
    print("public window +-5 s: {} frames; YOLO ball found {:.0%} (tracking ball seen {:.0%})".format(
        b - a, np.mean(np.isfinite(det[:, 0])), np.mean(np.isfinite(tm[:, 0]))))
    print("where both: distance median {:.1f} m, within 1 / 2 m {:.0%} / {:.0%}; pixels median {:.1f}".format(
        np.median(dist), np.mean(dist < 1), np.mean(dist < 2), np.nanmedian(pix)))
    from scipy.ndimage import maximum_filter1d

    def peaks(dv):
        d = np.nan_to_num(dv)
        return np.flatnonzero((d > 4) & (d == maximum_filter1d(d, 7)))
    py_, pt_ = peaks(dvy), peaks(dvt)
    only_y = [p for p in py_ if not np.any(np.abs(pt_ - p) <= 12)]
    only_t = [p for p in pt_ if not np.any(np.abs(py_ - p) <= 12)]
    print("velocity-change peaks > 4 m/s: YOLO {}, tracking {}; YOLO-only {} (tracking-only {})".format(
        len(py_), len(pt_), len(only_y), len(only_t)))
    import possession_metrics as pm
    from report import PUBLIC_GT
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    sp = pm.possession_spells(gt, pm.valid_runs(gt))
    for k in (7, 10, 12):
        s = sp[k - 1][0] - a
        w = slice(max(s - 25, 0), s + 26)
        print("  #{} start +-1 s: YOLO found {:.0%}, YOLO peak {}, tracking peak {}".format(
            k, np.mean(np.isfinite(det[w, 0])), "yes" if any(abs(p - s) <= 25 for p in py_) else "no",
            "yes" if any(abs(p - s) <= 25 for p in pt_) else "no"))


def render_cmd(args):
    import cv2
    from games import asset_paths, GAMES, GAME
    import vision_dataset as V
    z = np.load(os.path.join(OUT, "window_{}.npz".format(args.model)))
    a, b, det, xy = int(z["a"]), int(z["b"]), z["det"], z["xy"]
    L, W = GAMES[GAME]["pitch"]
    px, py, ok, ball = V.ball_pixels()
    sy, dvy = vel_change(xy * np.array([L, W]))
    st, dvt = vel_change(ball[a:b, :2].astype(np.float64) * np.array([L, W]))
    cap = cv2.VideoCapture(asset_paths()["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    path = os.path.join(ROOT, "private", "yolo_public_{}.mp4".format(args.model))
    wr = None
    for i in range(b - a):
        okr, img = cap.read()
        if not okr:
            break
        if wr is None:
            h, w = img.shape[:2]
            wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
        f = a + i
        if np.isfinite(px[f]):
            cv2.circle(img, (int(px[f]), int(py[f])), 12, (255, 128, 0), 2)          # tracking: blue
        if np.isfinite(det[i, 0]):
            cv2.rectangle(img, (int(det[i, 0]) - 9, int(det[i, 1]) - 9),
                          (int(det[i, 0]) + 9, int(det[i, 1]) + 9), (0, 0, 255), 2)  # YOLO: red
        txt = "t {:6.1f}s | tracking spd {:5.1f} dv {:5.1f} | YOLO spd {:5.1f} dv {:5.1f} conf {:.2f}".format(
            i / FPS, np.nan_to_num(st[i]), np.nan_to_num(dvt[i]), np.nan_to_num(sy[i]), np.nan_to_num(dvy[i]),
            det[i, 2] if np.isfinite(det[i, 2]) else 0)
        cv2.rectangle(img, (0, 0), (980, 34), (0, 0, 0), -1)
        cv2.putText(img, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if np.nan_to_num(dvy[i]) > 4:
            cv2.putText(img, "YOLO PEAK", (1000, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if np.nan_to_num(dvt[i]) > 4:
            cv2.putText(img, "TRACK PEAK", (1180, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 128, 0), 2)
        wr.write(img)
    wr.release()
    print("wrote", os.path.relpath(path, ROOT))


def clean_cmd(args):
    """full-match YOLO ball, cleaned and merged with the tracking ball -> ball_clean_<model>.npy."""
    from load_homography import load as load_h
    from atd_cache import load_arrays
    from games import GAMES, GAME
    from scipy.ndimage import median_filter, gaussian_filter1d
    z = np.load(os.path.join(OUT, "public_{}.npz".format(args.model)))      # the full-match pass
    a0 = int(z["a"])
    trk = load_arrays("tactical", raw=True)["ball"][:, :2].astype(np.float64)
    n = len(trk)
    det = np.full((n, 3), np.nan)
    m = min(n - a0, len(z["det"]))
    det[a0:a0 + m] = z["det"][:m]
    Hm, _, valid = load_h()
    Hm = Hm[:n].astype(np.float64)
    okh = (valid[:n] > 0) & np.isfinite(det[:, 0])
    Hi = np.full_like(Hm, np.nan)
    Hi[okh] = np.linalg.inv(Hm[okh])
    v = np.einsum("nij,nj->ni", Hi, np.column_stack([det[:, 0], det[:, 1], np.ones(n)]))
    yx = v[:, :2] / v[:, 2:3]
    yx[~okh] = np.nan
    L, W = GAMES[GAME]["pitch"]
    ym = yx * np.array([L, W])
    tm = trk * np.array([L, W])
    inside = (yx[:, 0] > -0.05) & (yx[:, 0] < 1.05) & (yx[:, 1] > -0.05) & (yx[:, 1] < 1.05)
    med = np.column_stack([median_filter(np.nan_to_num(ym[:, k], nan=1e4), size=7, mode="nearest") for k in (0, 1)])
    consistent = inside & (np.hypot(*(ym - med).T) <= 1.5) & (det[:, 2] >= 0.1)
    near_trk = np.isfinite(tm[:, 0]) & (np.hypot(*(ym - tm).T) <= 3.0)
    run = np.zeros(n, int)
    c = 0
    for i in range(n):
        c = c + 1 if consistent[i] else 0
        run[i] = c
    longrun = np.zeros(n, bool)                       # frames inside a run of >= 5 consistent frames
    i = n - 1
    while i >= 0:
        if run[i] >= 5:
            longrun[i - run[i] + 1:i + 1] = True
            i -= run[i]
        else:
            i -= 1
    mode = os.environ.get("YB_MODE", "both")       # both / fill (gaps only) / replace (no gap filling)
    use_rep = consistent & near_trk
    use_fill = consistent & longrun & ~np.isfinite(tm[:, 0])
    use = {"both": use_rep | use_fill, "fill": use_fill, "replace": use_rep}[mode]
    out = tm.copy()
    out[use] = ym[use]
    sm = out.copy()                                    # light smoothing on finite stretches
    fin = np.isfinite(out[:, 0])
    edges = np.flatnonzero(np.diff(np.concatenate([[0], fin.astype(np.int8), [0]])))
    for s0, s1 in zip(edges[::2], edges[1::2]):
        if s1 - s0 >= 5:
            sm[s0:s1] = gaussian_filter1d(out[s0:s1], float(os.environ.get("YB_SIGMA", "1.5")), axis=0, mode="nearest")
    tag = os.environ.get("YB_TAG", "")
    np.save(os.path.join(OUT, "ball_clean_{}{}.npy".format(args.model, tag)), (sm / np.array([L, W])).astype(np.float32))
    spd = np.hypot(*np.gradient(sm, axis=0).T) * FPS
    spd_t = np.hypot(*np.gradient(tm, axis=0).T) * FPS
    print("frames {}: YOLO detection {:.0%}, consistent {:.0%}; ball from YOLO {:.0%} (replacing tracking {:.0%}, "
          "filling tracking gaps {:.1%}); ball seen before {:.0%} -> after {:.0%}".format(
              n, np.mean(np.isfinite(det[:, 0])), consistent.mean(), use.mean(),
              np.mean(use & np.isfinite(tm[:, 0])), np.mean(use & ~np.isfinite(tm[:, 0])),
              np.mean(np.isfinite(tm[:, 0])), np.mean(np.isfinite(sm[:, 0]))))
    print("ball speed > 30 m/s (implausible): tracking {:.2%}, cleaned {:.2%}".format(
        np.nanmean(spd_t > 30), np.nanmean(spd > 30)))


def refine_cmd(args):
    """tracking ball + a smoothed YOLO correction, only where both exist and agree; gaps untouched.
    -> ball_refined_<model>.npy (normalised x, y; NaN exactly where the tracking ball is NaN)."""
    from load_homography import load as load_h
    from atd_cache import load_arrays
    from games import GAMES, GAME
    from scipy.ndimage import median_filter, gaussian_filter1d
    z = np.load(os.path.join(OUT, "public_{}.npz".format(args.model)))
    a0 = int(z["a"])
    trk = load_arrays("tactical", raw=True)["ball"][:, :2].astype(np.float64)
    n = len(trk)
    det = np.full((n, 3), np.nan)
    m = min(n - a0, len(z["det"]))
    det[a0:a0 + m] = z["det"][:m]
    Hm, _, valid = load_h()
    Hm = Hm[:n].astype(np.float64)
    okh = (valid[:n] > 0) & np.isfinite(det[:, 0])
    Hi = np.full_like(Hm, np.nan)
    Hi[okh] = np.linalg.inv(Hm[okh])
    v = np.einsum("nij,nj->ni", Hi, np.column_stack([det[:, 0], det[:, 1], np.ones(n)]))
    L, W = GAMES[GAME]["pitch"]
    ym = v[:, :2] / v[:, 2:3] * np.array([L, W])
    ym[~okh] = np.nan
    tm = trk * np.array([L, W])
    seen = np.isfinite(tm[:, 0])
    med = np.column_stack([median_filter(np.nan_to_num(ym[:, k], nan=1e4), size=7, mode="nearest") for k in (0, 1)])
    r = ym - tm                                         # YOLO's correction to the tracking ball
    step = np.full(n, np.inf)
    step[1:] = np.hypot(*(ym[1:] - ym[:-1]).T) * FPS    # implied speed between consecutive frames
    use = (seen & np.isfinite(ym[:, 0]) & (np.hypot(*r.T) <= args.radius)
           & (np.hypot(*(ym - med).T) <= 1.5) & (det[:, 2] >= 0.1) & (np.minimum(step, np.roll(step, -1)) <= 35))
    rr = np.where(use[:, None], r, np.nan)
    out = tm.copy()
    edges = np.flatnonzero(np.diff(np.concatenate([[0], seen.astype(np.int8), [0]])))
    for s0, s1 in zip(edges[::2], edges[1::2]):          # per seen stretch: fill short holes, smooth
        seg = rr[s0:s1]
        ok = np.isfinite(seg[:, 0])
        if ok.sum() < 3:
            continue
        idx = np.arange(s1 - s0)
        filled = np.column_stack([np.interp(idx, idx[ok], seg[ok, k]) for k in (0, 1)])
        far = np.ones(len(idx), bool)                   # frames more than 5 from a used one: no correction
        for k in np.flatnonzero(ok):
            far[max(k - 5, 0):k + 6] = False
        filled[far] = 0.0
        out[s0:s1] = tm[s0:s1] + gaussian_filter1d(filled, args.sigma, axis=0, mode="nearest")
    np.save(os.path.join(OUT, "ball_refined_{}.npy".format(args.model)), (out / np.array([L, W])).astype(np.float32))
    sp_t = np.hypot(*np.gradient(tm, axis=0).T) * FPS
    sp_o = np.hypot(*np.gradient(out, axis=0).T) * FPS
    print("tracking ball seen {:.0%}; YOLO correction used on {:.0%} of seen frames (median size {:.2f} m); "
          "unseen frames unchanged: {}".format(seen.mean(), use[seen].mean(), np.nanmedian(np.hypot(*r[use].T)),
                                               bool(np.all(np.isnan(out[~seen, 0])))))
    print("speed > 30 m/s: tracking {:.2%}, refined {:.2%}".format(np.nanmean(sp_t > 30), np.nanmean(sp_o > 30)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("probe", "run", "render", "clean", "refine"))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--model", default="bambezius")
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--radius", type=float, default=3.0, help="refine: max YOLO-tracking distance (m)")
    ap.add_argument("--sigma", type=float, default=1.0, help="refine: smoothing of the correction (frames)")
    ap.add_argument("--near", type=int, default=1, help="1: prefer the ball box nearest the tracking ball")
    args = ap.parse_args()
    {"probe": probe_cmd, "run": run_cmd, "render": render_cmd, "clean": clean_cmd, "refine": refine_cmd}[args.cmd](args)


if __name__ == "__main__":
    main()
