"""Try several decoder settings on our match and on the Metrica games in one go. Aggregate only.

    python tools/sweep.py "base:" "snap1:snap_s=1" "snap2:snap_s=2 touch_switch=10"

Each argument is NAME:KEY=VALUE ... (PARAMS overrides; nothing after the colon = defaults).
Features are built once per match, so every setting sees the same evidence. One line per
setting: our full match (possessions, median length, direct-change share, D share), the public
window (spellF1 / macroF1, a correctness check only, never for choosing), then the Metrica
spellF1 per game and the Metrica mean, which is what settings are chosen on.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("settings", nargs="+", help="NAME:KEY=VALUE ...")
    parser.add_argument("--no-metrica", action="store_true")
    args = parser.parse_args()
    settings = [parse_setting(s) for s in args.settings]

    ours = features.build_features()
    ours_runs = [(a, b) for _, a, b in half_runs()]
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))

    external = []
    if not args.no_metrica:
        import metrica_data as md
        length, width = features.LENGTH, features.WIDTH
        features.LENGTH, features.WIDTH = md.PITCH
        for g in md.GAMES:
            d = md.load(g)
            f = features.build_features({"players": d["players"], "ball": d["ball"],
                                         "team": d["team"]}, events=[])
            runs = []
            for per in (1, 2):
                idx = np.flatnonzero(d["period"] == per)
                if len(idx):
                    runs.append((idx[0], idx[-1] + 1))
            external.append((f, runs, d["labels"]))
        features.LENGTH, features.WIDTH = length, width

    print("{:<16} {:>5} {:>7} {:>6} {:>6}   {:>15}   {}".format(
        "setting", "poss", "median", "direct", "D", "public sp / mF1",
        "Metrica spellF1 per game | mean   recall at IoU {} by true length {}   median "
        "|start offset| of matched".format(RECALL_IOU, " / ".join(
            "{:g}-{}s".format(a, "" if b > 1e8 else "{:g}".format(b))
            for a, b in zip(LENGTH_EDGES_S[:-1], LENGTH_EDGES_S[1:])))))
    for name, p in settings:
        codes = decode_runs(ours, p, ours_runs, len(ours["touch"]))
        line = brief_line(name, codes, gt).split()
        # brief_line: name poss median direct D home spF1 mF1 count
        poss, median, direct, dead, sp, mf = line[1], line[2], line[3], line[4], line[6], line[7]
        ext, buckets, offs = [], np.zeros((len(LENGTH_EDGES_S) - 1, 2)), []
        for f, runs, labels in external:
            pred = decode_runs(f, p, runs, len(labels))
            ext.append(pm.evaluate(labels, pred, bootstrap_n=0)["spell_f1_mean"])
            rec, off = recall_by_length(labels, pred, runs, md.FPS)
            buckets += np.array(rec)
            offs.append(off)
        tail = (" ".join("{:.3f}".format(s) for s in ext) + " | {:.3f}".format(np.mean(ext))
                + "   recall " + " ".join("{:.0f}%".format(100 * m / max(t, 1))
                                          for m, t in buckets)
                + "   start off {:.2f}s".format(np.mean(offs)) if ext else "")
        print("{:<16} {:>5} {:>7} {:>6} {:>6}   {:>6} / {:<6}   {}".format(
            name, poss, median, direct, dead, sp[:6], mf[:6], tail))


if __name__ == '__main__':
    main()
