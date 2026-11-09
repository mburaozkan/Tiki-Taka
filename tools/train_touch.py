"""Train and test the touch network on DFL, one match held out at a time. Aggregate only.

    python tools/train_touch.py                 # 7 folds
    python tools/train_touch.py --folds 2       # the first 2 held-out matches only

Saves each fold's model as data/cache/models/touch_net_<held-out match>.pt (for testing the
pipeline on DFL with a network that never saw that match), and with --final also a model
trained on all 7 as touch_net_all.pt (the one used on our match).

Every match's ball is made to look like ours (dfl_data.prepare_ball: dead ball animated,
blurred, our gap pattern); training uses VERSIONS random gap patterns per match, each also with the
teams swapped. The held-out match gets one fixed gap pattern, and both the network and our current
rule (pipeline.features.touches on the same ball) are scored against the teacher touches
(tools/touch_labels.py):
  P / R / F1   touches matched one to one, same team, within TOL_S
  turnover R   recall of the true touches whose team differs from the previous true touch's:
               the touches that change possession, which is what the decoder needs
  offset       median (predicted - true) of matched touches, s
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import dfl_data as dd  # noqa: E402
import touch_labels as tl  # noqa: E402
from gap_model import degrade  # noqa: E402
import pipeline.features as features  # noqa: E402
from pipeline import touch_net as tn  # noqa: E402

# Params
MODEL_DIR = os.path.join(dd.DATA_DIR, "cache", "models")
VERSIONS = 2            # gap patterns per training match
TEST_SEED = 100
TOLS_S = (0.2, 0.5)
THRESHOLDS = (0.3, 0.5, 0.7)
SHORT_S = 3.0           # --target short: possession starts of spells shorter than this


def changes(lab):
    """(frames, teams) where controlled possession starts: H or A after anything else graded."""
    f = np.flatnonzero(np.isin(lab[1:], (dd.H, dd.A)) & (lab[1:] != lab[:-1])
                       & (lab[:-1] != dd.X)) + 1
    return f, (lab[f] == dd.A).astype(np.int8)


def short_changes(lab, max_s=None):
    """changes() limited to possessions shorter than max_s (SHORT_S): where they start."""
    max_s = SHORT_S if max_s is None else max_s
    f, t = changes(lab)
    keep = []
    for i in f:
        j = i
        while j < len(lab) and lab[j] == lab[i]:
            j += 1
        keep.append(j - i < max_s * dd.FPS)
    keep = np.array(keep, dtype=bool)
    return f[keep], t[keep]


YOLO = False           # --yolo: the ball made to look like our YOLO-refined ball (tools/yolo_ball.py refine)
YOLO_SLOW = (0.64, 0.96)      # m, sd of YOLO's slow correction on our match (x, y)
YOLO_FAST = (0.14, 0.21)      # m, sd of its fast jitter (lag-1 autocorrelation 0.84 ~ smoothed sigma 1)
YOLO_SHARE = 0.46             # share of seen frames YOLO corrects on our match


def yolo_like(d, seed):
    """our tracking-like ball, and in random stretches (~YOLO_SHARE of frames) the sharp ball +
    YOLO-like slow offset and fast jitter; same gaps."""
    from scipy.ndimage import gaussian_filter1d
    rng = np.random.default_rng(1000 + seed)
    blur = dd.prepare_ball(d, seed)
    sharp = dd.prepare_ball(d, seed, blur=False)
    n = len(blur)
    L, W = dd.PITCH
    noise = np.zeros((n, 2))
    for k, (slow, fast) in enumerate(zip(YOLO_SLOW, YOLO_FAST)):
        s_ = gaussian_filter1d(rng.standard_normal(n), 5.0)
        f_ = gaussian_filter1d(rng.standard_normal(n), 1.0)
        noise[:, k] = s_ / s_.std() * slow + f_ / f_.std() * fast
    noise /= np.array([L, W])
    sm = sharp.copy()
    ok = np.isfinite(sharp[:, 0])
    sm[ok, :2] = gaussian_filter1d(np.nan_to_num(sharp[:, :2]), 1.0, axis=0)[ok]
    mask = np.zeros(n, bool)                 # stretches of 1-6 s, alternating
    t = 0
    while t < n:
        ln = int(rng.integers(25, 150))
        if rng.random() < YOLO_SHARE:
            mask[t:t + ln] = True
        t += ln
    out = blur.copy()
    use = mask & ok & np.isfinite(blur[:, 0])
    out[use, :2] = sm[use, :2] + noise[use]
    return out


REAL = None            # --yolo-real blur|sharp: real YOLO corrections from our match (tools/yolo_noise.py)
REAL_CUT = 0.95


def yolo_real(d, seed):
    """blur: our tracking-like ball + a pasted real YOLO correction sequence (literal transplant).
    sharp: where the pasted correction is non-zero, the sharp ball (smoothed 1 frame) + only its fast
    part (correction - its 5-frame smooth); elsewhere the tracking-like ball. Same gaps."""
    import yolo_noise
    from scipy.ndimage import gaussian_filter1d
    rng = np.random.default_rng(2000 + seed)
    blur = dd.prepare_ball(d, seed)
    n = len(blur)
    seen = np.isfinite(blur[:, 0])
    c = yolo_noise.paste(n, seen, yolo_noise.load(cut=REAL_CUT), rng) / np.array(dd.PITCH)
    out = blur.copy()
    if REAL == "blur":
        out[seen, :2] = blur[seen, :2] + c[seen]
        return out
    sharp = dd.prepare_ball(d, seed, blur=False)
    ok = np.isfinite(sharp[:, 0])
    sm = gaussian_filter1d(np.nan_to_num(sharp[:, :2]), 1.0, axis=0)
    fast = c - gaussian_filter1d(c, 5.0, axis=0)
    use = seen & ok & (np.hypot(*c.T) > 1e-9)
    out[use, :2] = sm[use] + fast[use]
    return out


RFDETR = False         # --rfdetr: the ball made to look like our cleaned RF-DETR ball (tools/detr_ball.py)
RF_SD = 0.043          # m, white jitter: gives RF-DETR's measured 0.054 m (|ball - 2-frame smooth|, median)
RF_OUT = 0.07          # share of frames with a single-frame wrong detection
RF_OUT_M = (1.0, 8.0)  # its size, m


def rfdetr_like(d, seed):
    """DFL's sharp ball (dead ball animated, our gaps) + RF-DETR-like jitter and single-frame
    outliers, then the same cleaning as our RF-DETR ball (detr_ball.clean_ball)."""
    import detr_ball
    rng = np.random.default_rng(3000 + seed)
    sharp = dd.prepare_ball(d, seed, blur=False)
    xy = sharp[:, :2] * np.array(dd.PITCH)
    seen = np.isfinite(xy[:, 0])
    xy = xy + rng.normal(0, RF_SD, xy.shape)
    out = seen & (rng.random(len(xy)) < RF_OUT)
    ang = rng.uniform(0, 2 * np.pi, out.sum())
    r = rng.uniform(*RF_OUT_M, out.sum())
    xy[out] += np.column_stack([np.cos(ang), np.sin(ang)]) * r[:, None]
    ball = sharp.copy()
    ball[:, :2] = detr_ball.clean_ball(xy) / np.array(dd.PITCH)
    return ball


def prepared(game, seed, target="touch"):
    """Ball made to look like ours for one match, plus the target events (touches, or possession
    changes for the turnover network)."""
    d = dd.load(game)
    ball = (rfdetr_like(d, seed) if RFDETR else yolo_real(d, seed) if REAL else yolo_like(d, seed) if YOLO
            else dd.prepare_ball(d, seed))   # dead ball animated, blurred, our gaps
    frames, teams = (tl.load(game)[:2] if target == "touch" else
                     short_changes(d["controlled"]) if target == "short" else
                     changes(d["controlled"]))
    valid = d["period"] > 0
    return d, ball, frames, teams, valid


def match(pf, pt, tf, tt, tol):
    """(matched pred, matched true, offsets) greedy one-to-one, same team, nearest first."""
    pairs = []
    for i, (f, t) in enumerate(zip(tf, tt)):
        lo, hi = np.searchsorted(pf, [f - tol, f + tol + 1])
        for j in range(lo, hi):
            if pt[j] == t:
                pairs.append((abs(pf[j] - f), i, j, pf[j] - f))
    pairs.sort()
    used_t, used_p, offs = set(), set(), []
    for _, i, j, off in pairs:
        if i in used_t or j in used_p:
            continue
        used_t.add(i)
        used_p.add(j)
        offs.append(off)
    return used_p, used_t, offs


def scores(pf, pt, tf, tt):
    """{tol: (P, R, F1, turnover R, median offset s)}."""
    turnover = np.concatenate(([False], tt[1:] != tt[:-1]))
    out = {}
    for tol_s in TOLS_S:
        mp, mt, offs = match(pf, pt, tf, tt, int(round(tol_s * dd.FPS)))
        p = len(mp) / max(len(pf), 1)
        r = len(mt) / max(len(tf), 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        tr = np.mean([i in mt for i in np.flatnonzero(turnover)]) if turnover.any() else np.nan
        out[tol_s] = (p, r, f1, tr, np.median(offs) / dd.FPS if offs else np.nan)
    return out


def rule_touches(d, ball):
    length, width = features.LENGTH, features.WIDTH
    features.LENGTH, features.WIDTH = dd.PITCH
    f = features.build_features({"players": d["players"], "ball": ball, "team": d["team"]},
                                events=[])
    features.LENGTH, features.WIDTH = length, width
    frames = np.flatnonzero(f["touch"] != 0)
    return frames, (f["touch"][frames] < 0).astype(np.int8)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folds", type=int, default=len(dd.GAMES))
    parser.add_argument("--epochs", type=int, default=tn.EPOCHS)
    parser.add_argument("--final", action="store_true", help="also train on all 7 and save")
    parser.add_argument("--target", choices=("touch", "turnover", "short"), default="touch",
                        help="touches (touch_net_*) or controlled possession changes "
                             "(turnover_net_*)")
    parser.add_argument("--warp", type=float, default=1.0,
                        help="time-warp DFL by this factor (dfl_data.set_warp); models go to "
                             "data/cache/models/warp<k>/")
    parser.add_argument("--yolo", action="store_true",
                        help="train / test on a YOLO-like ball (models touch_net_yolo_*)")
    parser.add_argument("--yolo-real", choices=("blur", "sharp"),
                        help="train / test on DFL + real YOLO corrections from our match "
                             "(models touch_net_yr<b|s><cut>_*)")
    parser.add_argument("--rfdetr", action="store_true",
                        help="train / test on an RF-DETR-like sharp ball (models touch_net_rf_*)")
    parser.add_argument("--cut", type=float, default=0.95, help="--yolo-real: tools/yolo_noise.py cut")
    args = parser.parse_args()
    global MODEL_DIR, YOLO, REAL, REAL_CUT, RFDETR
    YOLO = args.yolo
    REAL, REAL_CUT = args.yolo_real, args.cut
    RFDETR = args.rfdetr
    if args.warp != 1.0:
        dd.set_warp(args.warp)
        MODEL_DIR = os.path.join(MODEL_DIR, "warp{:g}".format(args.warp))
    stem = {"touch": "touch_net", "turnover": "turnover_net", "short": "short_net"}[args.target]
    if dd.TOUCH_TAG:                 # a rebuilt chain's touch networks: plain names in their own folder
        MODEL_DIR = os.path.join(MODEL_DIR, dd.TOUCH_TAG)
    if YOLO:
        stem += "_yolo"
    if RFDETR and not dd.TOUCH_TAG:
        stem += "_rf"
    if REAL:
        stem += "_yr{}{:g}".format(REAL[0], REAL_CUT * 100)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("building inputs ...", flush=True)
    train_sets = {}
    for g in dd.GAMES:
        sets = []
        for v in range(VERSIONS):
            d, ball, frames, teams, valid = prepared(g, seed=v, target=args.target)
            for flip in (False, True):
                x = tn.inputs(d["players"], ball, d["team"], dd.PITCH, flip=flip)
                sets.append((x, tn.targets(len(x), frames, teams, flip=flip), valid))
        train_sets[g] = sets

    rows = {"rule": [], **{"net@{}".format(t): [] for t in THRESHOLDS}}
    chg = {"rule": [], **{"net@{}".format(t): [] for t in THRESHOLDS}}
    for held in dd.GAMES[:args.folds]:
        print("held out {} ...".format(held), flush=True)
        bundle = tn.train([s for g in dd.GAMES if g != held for s in train_sets[g]],
                          epochs=args.epochs, log=lambda *_: None)
        tn.save(bundle, os.path.join(MODEL_DIR, "{}_{}.pt".format(stem, held)))
        d, ball, tf, tt, valid = prepared(held, seed=TEST_SEED, target=args.target)
        cf, ct = changes(d["controlled"])
        prob = tn.predict(bundle, tn.inputs(d["players"], ball, d["team"], dd.PITCH))
        rf, rt = rule_touches(d, ball)
        rows["rule"].append(scores(rf, rt, tf, tt))
        chg["rule"].append(scores(rf, rt, cf, ct))
        for th in THRESHOLDS:
            pf, pt = tn.pick(prob, th)
            rows["net@{}".format(th)].append(scores(pf, pt, tf, tt))
            chg["net@{}".format(th)].append(scores(pf, pt, cf, ct))
        r, n = rows["rule"][-1][0.5], rows["net@0.5"][-1][0.5]
        print("  +/-0.5 s  rule F1 {:.2f} turnover R {:.2f} | net@0.5 F1 {:.2f} turnover R {:.2f}"
              .format(r[2], r[3], n[2], n[3]), flush=True)

    if args.final:
        print("training on all 7 ...", flush=True)
        bundle = tn.train([s for g in dd.GAMES for s in train_sets[g]], epochs=args.epochs,
                          log=lambda *_: None)
        tn.save(bundle, os.path.join(MODEL_DIR, "{}_all.pt".format(stem)))

    print("\nmean over {} held-out matches".format(args.folds))
    print("{:<10} {:>6} {:>32} {:>32}".format("", "", "within 0.2 s: P  R  F1  turnR  off",
                                              "within 0.5 s: P  R  F1  turnR  off"))
    for title, table in (("against its own target ({})".format(args.target), rows),
                         ("against the true possession changes (controlled labels)", chg)):
        print(title)
        for name, rs in table.items():
            line = "{:<10}".format(name)
            for tol in TOLS_S:
                m = np.nanmean([r[tol] for r in rs], axis=0)
                line += "   {:.2f} {:.2f} {:.2f} {:.2f} {:+.2f}s".format(*m)
            print(line)


if __name__ == '__main__':
    main()
