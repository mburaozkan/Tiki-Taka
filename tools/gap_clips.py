"""Is the ball visible in the video while the ATD has lost it? Cut clips to check by eye.

    pip install opencv-python
    python tools/gap_clips.py            # writes clips + an answer sheet to data/cache/gap_clips/
    python tools/gap_clips.py --tally    # after filling in the sheet: aggregate counts only

Picks N random ball gaps (at least MIN_GAP_S long, inside the halves, fixed seed) and writes one
clip per gap: LEAD_S before the gap, then the gap itself (capped at MAX_CLIP_S). The overlay shows
whether the ATD sees the ball on each frame, and a circle where it is when it does.

Everything is written under data/cache/, which is gitignored: clips and the sheet are match data.
Look at them locally only, and never paste the sheet or a frame anywhere. --tally prints counts.

Answer codes for the sheet (one per clip, in the `answer` column):
    V  ball clearly visible for most of the gap
    P  partly: visible for some of it, or hard to make out
    H  hidden: occluded, in a crowd, or in the air out of view
    O  off screen: the camera does not show where the ball is
    U  unclear: could not tell
"""

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, half_video_blocks, video_path  # noqa: E402
from atd_cache import load_arrays, CACHE_DIR  # noqa: E402

# Params
FPS = GAMES[GAME]["fps"]
N_CLIPS = 20
SEED = 0
MIN_GAP_S = 1.0
LEAD_S = 1.0
MAX_CLIP_S = 10.0
MAX_WIDTH = 1920
OUT_DIR = os.path.join(CACHE_DIR, "gap_clips")
SHEET = os.path.join(OUT_DIR, "answers.csv")
CODES = ("V", "P", "H", "O", "U")


def pick_gaps(ball):
    """N random (start, stop) ball gaps inside the halves, sorted by clip number."""
    seen = ~np.isnan(ball[:, 0])
    gaps = []
    for _, first, last in half_video_blocks(GAME):
        window = seen[first:last + 1]
        edges = np.diff(np.concatenate(([1], window.astype(np.int8), [1])))
        for a, b in zip(np.flatnonzero(edges == -1), np.flatnonzero(edges == 1)):
            if a > 0 and b < len(window) and (b - a) >= MIN_GAP_S * FPS:
                gaps.append((first + int(a), first + int(b)))
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(len(gaps), size=min(N_CLIPS, len(gaps)), replace=False)
    return [gaps[i] for i in chosen], len(gaps)


def write_clip(capture, path, start, stop, ball, matrices, valid):
    """Frames [start - lead, min(stop, start + max)) with the ATD overlay."""
    import cv2
    from load_homography import project

    first = max(0, start - int(LEAD_S * FPS))
    last = min(stop, start + int(MAX_CLIP_S * FPS))
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    writer, scale = None, 1.0
    for frame in range(first, last):
        ok, image = capture.read()
        if not ok:
            break
        if writer is None:
            h, w = image.shape[:2]
            scale = min(1.0, MAX_WIDTH / float(w))
            size = (int(w * scale), int(h * scale))
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, size)
        if scale != 1.0:
            image = cv2.resize(image, size)
        has_ball = not np.isnan(ball[frame, 0])
        if has_ball and valid[frame]:
            x, y = project(matrices[frame], ball[frame, 0], ball[frame, 1])
            cv2.circle(image, (int(x * scale), int(y * scale)), 18, (0, 255, 0), 2)
        colour = (0, 160, 0) if has_ball else (0, 0, 200)
        text = "ATD ball SEEN" if has_ball else "ATD ball MISSING - is it visible?"
        cv2.rectangle(image, (0, 0), (620, 44), colour, -1)
        cv2.putText(image, text, (10, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        writer.write(image)
    if writer is not None:
        writer.release()


def make_clips():
    import cv2
    from load_homography import load as load_homography

    if os.path.isfile(SHEET):
        sys.exit("{} exists; delete it to start a fresh sheet".format(SHEET))
    ball = load_arrays("tactical")["ball"]
    gaps, n_total = pick_gaps(ball)
    matrices, _, valid = load_homography()
    os.makedirs(OUT_DIR, exist_ok=True)

    capture = cv2.VideoCapture(video_path())
    if not capture.isOpened():
        sys.exit("cannot open the tactical video")
    rows = []
    for number, (start, stop) in enumerate(gaps, 1):
        name = "clip_{:02d}.mp4".format(number)
        write_clip(capture, os.path.join(OUT_DIR, name), start, stop, ball, matrices, valid)
        rows.append({"clip": name, "gap_s": "{:.1f}".format((stop - start) / float(FPS)),
                     "answer": ""})
        print("  clip {:2d}/{} written".format(number, len(gaps)))
    capture.release()

    with open(SHEET, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["clip", "gap_s", "answer"])
        writer.writeheader()
        writer.writerows(rows)
    print("\n{} clips from {} gaps of {:g}s+ -> {}".format(len(gaps), n_total, MIN_GAP_S, OUT_DIR))
    print("Watch each clip, fill the `answer` column with one of {} (see the docstring),".format(
        "/".join(CODES)))
    print("then run:  python tools/gap_clips.py --tally")


def tally():
    """Counts of each answer, overall and split by gap length. Safe to share."""
    with open(SHEET) as handle:
        rows = [r for r in csv.DictReader(handle)]
    answers = [(r["answer"].strip().upper(), float(r["gap_s"])) for r in rows]
    missing = sum(1 for a, _ in answers if a not in CODES)
    print("GAP CLIP CHECK  {} clips, {} unanswered".format(len(answers), missing))
    for label, keep in (("all", lambda s: True), ("gaps < 3s", lambda s: s < 3),
                        ("gaps 3-10s", lambda s: 3 <= s < 10), ("gaps >= 10s", lambda s: s >= 10)):
        group = [a for a, s in answers if keep(s) and a in CODES]
        print("  {:<12} n={:<3} ".format(label, len(group)) + "  ".join(
            "{} {}".format(code, group.count(code)) for code in CODES))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tally", action="store_true", help="count the filled-in answers")
    args = parser.parse_args()
    tally() if args.tally else make_clips()
