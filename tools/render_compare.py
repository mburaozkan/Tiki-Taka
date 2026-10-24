"""Blind side-by-side check of two submissions where they disagree around set pieces.

    python tools/render_compare.py evaluation/results/v6.csv evaluation/results/v7_rd1.0_rt1.0.csv
    python tools/render_compare.py A.csv B.csv --tally      # after filling in the answer sheet

Finds every stretch where A and B give different labels, near a Smart Tagging restart window
(SET PIECES start until the next phase starts, plus a margin after), and cuts it into numbered
clips. Each clip shows two banners, LEFT and RIGHT. Which submission is on which side is random
per clip, so you judge what you see, not the name. Also drawn: the ball circle, the restart window
("RESTART (tag)"), and the truth where the public window has one.

Writes, all under data/cache/compare/ (gitignored: match video):
  <A>_vs_<B>.mp4   the clips, each after a one-second "clip k / N" card
  key.csv          clip -> which side is A (never shown; read by --tally)
  answers.csv      clip, answer, notes: fill answer with L, R, both (both fine), none (both wrong)
                   or ? (can't tell)
Prints only the clip count, total length and paths. --tally prints only counts per submission.
"""

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from games import GAME, video_path, half_video_blocks  # noqa: E402
from atd_cache import CACHE_DIR, load_arrays  # noqa: E402
from report import PUBLIC_GT  # noqa: E402
from render_labels import FPS, COLOURS, NAMES, banner  # noqa: E402

# Params
OUT_DIR = os.path.join(CACHE_DIR, "compare")
AFTER_WINDOW_S = 10.0   # disagreements this long after a restart window still count
MERGE_S = 4.0           # disagreements closer than this share a clip
PAD_BEFORE_S = 4.0
PAD_AFTER_S = 3.0
MIN_DIFF_S = 0.4        # ignore disagreements shorter than this in total per clip
CARD_S = 1.0
SEED = 7


def clips(a, b, near, valid):
    """[(first, last)] video-frame ranges around the disagreements of a and b inside `near`."""
    diff = np.flatnonzero((a != b) & near & valid)
    if not len(diff):
        return []
    groups, start, prev = [], diff[0], diff[0]
    for i in diff[1:]:
        if i - prev > MERGE_S * FPS:
            groups.append((start, prev))
            start = i
        prev = i
    groups.append((start, prev))
    out = []
    for s, e in groups:
        if np.sum((a[s:e + 1] != b[s:e + 1])) < MIN_DIFF_S * FPS:
            continue
        out.append((max(0, s - int(PAD_BEFORE_S * FPS)), min(len(a) - 1, e + int(PAD_AFTER_S * FPS))))
    return out


def tally(key_path, answers_path, names):
    side = {}
    with open(key_path, newline="") as handle:
        for row in csv.DictReader(handle):
            side[row["clip"]] = row["left_is"]
    counts = {names[0]: 0, names[1]: 0, "both fine": 0, "both wrong": 0, "can't tell": 0,
              "blank": 0}
    with open(answers_path, newline="") as handle:
        for row in csv.DictReader(handle):
            ans = (row.get("answer") or "").strip().upper()
            left = names[0] if side.get(row["clip"]) == "A" else names[1]
            right = names[1] if left == names[0] else names[0]
            key = {"L": left, "R": right, "BOTH": "both fine", "NONE": "both wrong",
                   "?": "can't tell"}.get(ans, "blank")
            counts[key] += 1
    print("judged better:  " + "   ".join("{} {}".format(k, v) for k, v in counts.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--tally", action="store_true")
    parser.add_argument("--everywhere", action="store_true",
                        help="every disagreement, not only those near a restart window")
    args = parser.parse_args()

    names = [os.path.splitext(os.path.basename(p))[0] for p in (args.a, args.b)]
    stem = "{}_vs_{}".format(*names)
    out_dir = os.path.join(OUT_DIR, stem)
    key_path = os.path.join(out_dir, "key.csv")
    answers_path = os.path.join(out_dir, "answers.csv")
    if args.tally:
        tally(key_path, answers_path, names)
        return

    import cv2
    from load_homography import load as load_homography, project
    from pipeline.features import build_features

    a = pm.label_codes(pm.read_label_csv(args.a))
    b = pm.label_codes(pm.read_label_csv(args.b))
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    f = build_features()
    restart = f["tag_restart"] > 0
    near = np.ones(len(a), dtype=bool) if args.everywhere else restart.copy()
    after = int(AFTER_WINDOW_S * FPS)
    for i in ([] if args.everywhere else np.flatnonzero(np.diff(restart.astype(np.int8)) == -1)):
        near[i + 1:i + 1 + after] = True
    valid = np.zeros(len(a), dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        valid[first:last + 1] = True
    ranges = clips(a, b, near, valid)

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(ranges))    # shuffled, so clip number says nothing about time
    left_is_a = rng.random(len(ranges)) < 0.5

    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(answers_path):
        sys.exit("{} exists; move it away before re-rendering".format(
            os.path.relpath(answers_path, ROOT)))
    with open(key_path, "w", newline="") as handle:
        handle.write("clip,left_is\n")
        handle.writelines("{},{}\n".format(k + 1, "A" if left_is_a[k] else "B")
                          for k in range(len(ranges)))
    with open(answers_path, "w", newline="") as handle:
        handle.write("clip,answer,notes\n")
        handle.writelines("{},,\n".format(k + 1) for k in range(len(ranges)))

    ball = load_arrays("tactical")["ball"]
    matrices, _, hvalid = load_homography()
    capture = cv2.VideoCapture(video_path())
    video = os.path.join(out_dir, stem + ".mp4")
    writer, size, total = None, None, 0
    for k, idx in enumerate(order):
        first, last = ranges[idx]
        left, right = (a, b) if left_is_a[k] else (b, a)
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        for frame in range(first, last + 1):
            ok, image = capture.read()
            if not ok:
                break
            if writer is None:
                h, w = image.shape[:2]
                scale = min(1.0, args.width / float(w))
                size = (int(w * scale), int(h * scale))
                writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), FPS, size)
            if frame == first:
                card = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                cv2.putText(card, "clip {} / {}".format(k + 1, len(ranges)),
                            (size[0] // 2 - 160, size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                            (255, 255, 255), 3)
                for _ in range(int(CARD_S * FPS)):
                    writer.write(card)
            image = cv2.resize(image, size)
            for x, side, codes in ((0, "LEFT ", left), (340, "RIGHT ", right)):
                label = int(codes[frame])
                banner(image, x, side + NAMES[label][:1], COLOURS[label], cv2, 330)
            truth = int(gt[frame])
            if truth != pm.X:
                banner(image, 680, "truth " + NAMES[truth][:1], COLOURS[truth], cv2, 200)
            cv2.putText(image, "clip {}".format(k + 1), (size[0] - 150, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            if restart[frame]:
                cv2.putText(image, "RESTART (tag)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 220, 255), 2)
            if not np.isnan(ball[frame, 0]) and hvalid[frame]:
                x, y = project(matrices[frame], ball[frame, 0], ball[frame, 1])
                cv2.circle(image, (int(x * scale), int(y * scale)), 14, (0, 230, 0), 2)
            else:
                cv2.putText(image, "NO BALL", (10, size[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 200, 255), 2)
            writer.write(image)
            total += 1
    capture.release()
    if writer is not None:
        writer.release()
    print("{} clips, {:.1f} min of match video".format(len(ranges), total / float(FPS) / 60.0))
    print("video:   {}".format(os.path.relpath(video, ROOT)))
    print("answers: {}  (answer = L / R / both / none / ?, notes free text)".format(
        os.path.relpath(answers_path, ROOT)))


if __name__ == '__main__':
    main()
