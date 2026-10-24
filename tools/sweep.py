"""Try several decoder settings on our match and on external matches in one go. Aggregate only.

    python tools/sweep.py "base:" "snap1:snap_s=1" "snap2:snap_s=2 touch_switch=10"
    python tools/sweep.py --provider dfl "base:" "sw20:switch=20"
    python tools/sweep.py --provider dfl --labels labels "base:"

Each argument is NAME:KEY=VALUE ... (PARAMS overrides; nothing after the colon = defaults).
Features are built once per match, so every setting sees the same evidence. One line per
setting: our full match (possessions, median length, direct-change share, D share), the public
window (spellF1 / macroF1, a correctness check only, never for choosing), then the external
spellF1 per match and the mean, which is what settings are chosen on, with recall by true
possession length and the median start offset of matched possessions.

Providers: metrica (3 matches, event-derived labels) or dfl (7 matches; --labels controlled, the
default, or labels, the provider's raw possession flags; see tools/dfl_data.py). --degrade cuts
ball gaps like ours into the external ball track (tools/gap_model.py; DFL's ball is never lost).
--ball-fill guesses the ball inside gaps on every match (pipeline/ball_fill.py).
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
import pipeline.features as features  # noqa: E402
from pipeline.decode import PARAMS, emissions, decode_block, X  # noqa: E402
from report import PUBLIC_GT, half_runs, brief_line  # noqa: E402

# Params
LENGTH_EDGES_S = (0, 4, 10, 1e9)   # true possession length buckets for the recall breakdown
RECALL_IOU = 0.5


def recall_by_length(gt, pred, runs, fps):
    """[(matched, total)] per LENGTH_EDGES_S bucket at RECALL_IOU, plus the median |start offset|
    (s) of matched possessions."""
    filled = pm.fill_abstentions(pred, runs)
    gs, ps = pm.possession_spells(gt, runs), pm.possession_spells(filled, runs)
    flags = pm.matched_gt_flags(pm.spell_overlaps(gs, ps), len(gs), RECALL_IOU)
    lengths = np.array([(b - a) / float(fps) for a, b, _ in gs])
    out = []
    for lo, hi in zip(LENGTH_EDGES_S[:-1], LENGTH_EDGES_S[1:]):
        m = (lengths >= lo) & (lengths < hi)
        out.append((int(np.sum(flags & m)), int(m.sum())))
    offsets = []
    for (a, b, k), ok in zip(gs, flags):
        if ok:
            starts = [c for c, d, k2 in ps if k2 == k and d > a and b > c]
            offsets.append(min(abs(c - a) for c in starts) / float(fps))
    return out, float(np.median(offsets)) if offsets else float("nan")


def parse_setting(text):
    name, _, rest = text.partition(":")
    params = dict(PARAMS)
    for pair in rest.split():
        key, _, value = pair.partition("=")
        if key not in params:
            sys.exit("unknown parameter {!r}".format(key))
        params[key] = float(value)
    return name, params


def decode_runs(f, p, runs, n):
    scores = emissions(f, p)
    codes = np.full(n, X, dtype=np.int8)
    for a, b in runs:
        codes[a:b] = decode_block(scores, f, p, a, b)
    return codes


def external_matches(provider, label_key, degrade=False, ball_fill=None):
    """[(features, runs, labels)] and the provider's fps."""
    if provider == "metrica":
        import metrica_data as mod
        key = "labels"
    else:
        import dfl_data as mod
        key = label_key
    length, width = features.LENGTH, features.WIDTH
    features.LENGTH, features.WIDTH = mod.PITCH
    out = []
    for g in mod.GAMES:
        d = mod.load(g)
        ball = d["ball"]
        if degrade:
            from gap_model import degrade as cut
            ball = cut(ball, d["players"], seed=sum(map(ord, str(g))))
        f = features.build_features({"players": d["players"], "ball": ball,
                                     "team": d["team"]}, events=[], ball_fill=ball_fill)
        runs = []
        for per in (1, 2):
            idx = np.flatnonzero(d["period"] == per)
            if len(idx):
                runs.append((idx[0], idx[-1] + 1))
        out.append((f, runs, d[key]))
    features.LENGTH, features.WIDTH = length, width
    return out, mod.FPS


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("settings", nargs="+", help="NAME:KEY=VALUE ...")
    parser.add_argument("--provider", choices=("metrica", "dfl", "none"), default="metrica")
    parser.add_argument("--labels", choices=("controlled", "labels"), default="controlled",
                        help="dfl only: controlled possession (default) or the raw flags")
    parser.add_argument("--degrade", action="store_true", help="cut our gap pattern into the ball")
    parser.add_argument("--ball-fill", choices=("hold", "extrapolate", "interp"))
    args = parser.parse_args()
    settings = [parse_setting(s) for s in args.settings]

    ours = features.build_features(ball_fill=args.ball_fill)
    ours_runs = [(a, b) for _, a, b in half_runs()]
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))

    external, fps = ([], 25) if args.provider == "none" else external_matches(
        args.provider, args.labels, args.degrade, args.ball_fill)
    print("{:<16} {:>5} {:>7} {:>6} {:>6}   {:>15}   {}".format(
        "setting", "poss", "median", "direct", "D", "public sp / mF1",
        "{} spellF1 per match | mean, macroF1 mean   recall at IoU {} by true length {}   median "
        "|start offset| of matched".format(args.provider, RECALL_IOU, " / ".join(
            "{:g}-{}s".format(a, "" if b > 1e8 else "{:g}".format(b))
            for a, b in zip(LENGTH_EDGES_S[:-1], LENGTH_EDGES_S[1:])))))
    for name, p in settings:
        codes = decode_runs(ours, p, ours_runs, len(ours["touch"]))
        line = brief_line(name, codes, gt).split()
        # brief_line: name poss median direct D home spF1 mF1 count
        poss, median, direct, dead, sp, mf = line[1], line[2], line[3], line[4], line[6], line[7]
        ext, mf1, buckets, offs = [], [], np.zeros((len(LENGTH_EDGES_S) - 1, 2)), []
        for f, runs, labels in external:
            pred = decode_runs(f, p, runs, len(labels))
            r = pm.evaluate(labels, pred, bootstrap_n=0)
            ext.append(r["spell_f1_mean"])
            mf1.append(r["macro_f1"])
            rec, off = recall_by_length(labels, pred, runs, fps)
            buckets += np.array(rec)
            offs.append(off)
        tail = (" ".join("{:.3f}".format(s) for s in ext) + " | {:.3f} mF1 {:.3f}".format(np.mean(ext), np.mean(mf1))
                + "   recall " + " ".join("{:.0f}%".format(100 * m / max(t, 1))
                                          for m, t in buckets)
                + "   start off {:.2f}s".format(np.mean(offs)) if ext else "")
        print("{:<16} {:>5} {:>7} {:>6} {:>6}   {:>6} / {:<6}   {}".format(
            name, poss, median, direct, dead, sp[:6], mf[:6], tail), flush=True)


if __name__ == '__main__':
    main()
