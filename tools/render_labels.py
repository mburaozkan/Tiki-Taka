"""Render the tactical video with a submission's labels on it, to watch what the pipeline thinks.

    python tools/render_labels.py evaluation/results/v5.csv                  # public 5 minutes
    python tools/render_labels.py evaluation/results/v5.csv --start 40:00 --minutes 5
    python tools/render_labels.py evaluation/results/v5.csv --start 2nd --minutes 3
    python tools/render_labels.py evaluation/results/v30.csv --possessions 10 12   # #10..#12

Overlay, per frame:
  - top-left banner: the submission's label (H blue, A red, D grey);
  - next to it, inside the public window only: the ground truth label, and WRONG when they differ;
  - the ATD ball circled in green (via the homography), "NO BALL" in a gap;
  - a touch makes the circle flash in the toucher's colour; "last touch" shows the carried side.

--start takes mm:ss of video time (the burned-in clock), a video frame number, or "2nd" for the
second-half kickoff. --possessions FIRST LAST cuts from PAD_BEFORE_S before true possession
#FIRST to PAD_AFTER_S after #LAST (numbered as in tools/spell_table.py). Writes data/cache/renders/<submission>_<start>.mp4 (gitignored: it is match
video). Watch it locally; never share frames from it. Prints only the output path.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from games import GAME, GAMES, video_path, half_video_blocks  # noqa: E402
from atd_cache import CACHE_DIR, load_arrays  # noqa: E402
from report import PUBLIC_GT  # noqa: E402

# Params
FPS = GAMES[GAME]["fps"]
OUT_DIR = os.path.join(CACHE_DIR, "renders")
SECOND_HALF_KICKOFF = 113742  # video frame, published in DATA.md / FAQ.md
TOUCH_FLASH = 8               # frames the circle stays in the toucher's colour
PAD_BEFORE_S = 8.0            # --possessions: lead-in before the first possession
PAD_AFTER_S = 5.0             # --possessions: run-out after the last one
# BGR colours
COLOURS = {pm.H: (200, 90, 20), pm.A: (40, 40, 210), pm.D: (110, 110, 110), pm.X: (60, 60, 60)}
NAMES = {pm.H: "H  Barcelona", pm.A: "A  Athletic", pm.D: "D  dead ball", pm.X: "X"}


def parse_start(text):
    """mm:ss, a frame number, or '2nd' -> video frame."""
    if text == "2nd":
        return SECOND_HALF_KICKOFF
    if ":" in text:
        minutes, seconds = text.split(":")
        return (int(minutes) * 60 + int(seconds)) * FPS
    return int(text)


def banner(image, x, text, colour, cv2, width=330):
    cv2.rectangle(image, (x, 0), (x + width, 52), colour, -1)
    cv2.putText(image, text, (x + 10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)


def main():
    import cv2
    from load_homography import load as load_homography, project
    from pipeline.features import build_features

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submission")
    parser.add_argument("--start", help="mm:ss, frame, or 2nd (default: public window start)")
    parser.add_argument("--minutes", type=float, default=5.0)
    parser.add_argument("--possessions", type=int, nargs=2, metavar=("FIRST", "LAST"),
                        help="cut around true public possessions #FIRST..#LAST instead")
    parser.add_argument("--width", type=int, default=1280, help="output width in pixels")
    args = parser.parse_args()

    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    pred = pm.label_codes(pm.read_label_csv(args.submission))
    graded = np.flatnonzero(gt != pm.X)
    first = parse_start(args.start) if args.start else int(graded[0])
    last = min(len(pred), first + int(args.minutes * 60 * FPS))

    # True possession number per frame (#1, #2, ...), matching tools/spell_table.py.
    spells = pm.possession_spells(gt, pm.valid_runs(gt))
    spell_no = np.zeros(len(gt), dtype=np.int32)
    for i, (a, b, _) in enumerate(spells):
        spell_no[a:b] = i + 1
    tag = None
    if args.possessions:
        lo, hi = args.possessions
        first = max(0, spells[lo - 1][0] - int(PAD_BEFORE_S * FPS))
        last = min(len(pred), spells[hi - 1][1] + int(PAD_AFTER_S * FPS))
        tag = "p{}-{}".format(lo, hi)

    f = build_features()
    ball = load_arrays("tactical")["ball"]
    matrices, _, valid = load_homography()

    os.makedirs(OUT_DIR, exist_ok=True)
    name = "{}_{}.mp4".format(os.path.splitext(os.path.basename(args.submission))[0],
                              tag or first)
    path = os.path.join(OUT_DIR, name)

    capture = cv2.VideoCapture(video_path())
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    writer, scale, flash, flash_colour = None, 1.0, 0, None
    halves = [(a, b) for _, a, b in half_video_blocks(GAME)]
    for frame in range(first, last):
        ok, image = capture.read()
        if not ok:
            break
        if writer is None:
            h, w = image.shape[:2]
            scale = min(1.0, args.width / float(w))
            size = (int(w * scale), int(h * scale))
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, size)
        if scale != 1.0:
            image = cv2.resize(image, size)

        label = int(pred[frame])
        banner(image, 0, "pred " + NAMES[label], COLOURS[label], cv2)
        truth = int(gt[frame])
        if truth != pm.X:
            number = " #{}".format(spell_no[frame]) if spell_no[frame] else ""
            banner(image, 340, "truth" + number + " " + NAMES[truth][:1], COLOURS[truth], cv2, 350)
            if truth != label:
                cv2.putText(image, "WRONG", (700, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                            (0, 0, 255), 3)
        if not any(a <= frame <= b for a, b in halves):
            cv2.putText(image, "outside the halves (not graded)", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        touch = f["touch"][frame]
        if touch != 0:
            flash, flash_colour = TOUCH_FLASH, COLOURS[pm.H if touch > 0 else pm.A]
        lt = f["last_touch"][frame]
        lt_text = "last touch: " + ("H" if lt > 0 else "A" if lt < 0 else "-")
        cv2.putText(image, lt_text, (10, size[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)

        if not np.isnan(ball[frame, 0]) and valid[frame]:
            x, y = project(matrices[frame], ball[frame, 0], ball[frame, 1])
            colour = flash_colour if flash > 0 else (0, 230, 0)
            cv2.circle(image, (int(x * scale), int(y * scale)), 14, colour, 3 if flash else 2)
        else:
            cv2.putText(image, "NO BALL (ATD gap)", (10, size[1] - 55), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 200, 255), 2)
        flash = max(0, flash - 1)
        writer.write(image)
    capture.release()
    if writer is not None:
        writer.release()
    print("wrote {:.1f} min -> {}".format((last - first) / float(FPS) / 60.0,
                                          os.path.relpath(path, ROOT)))


if __name__ == '__main__':
    main()
