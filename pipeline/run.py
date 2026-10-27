"""Build features, decode, write a submission. Prints label shares only.

    python pipeline/run.py                                  # -> evaluation/results/v1.csv
    python pipeline/run.py --name v1_sw40 --set switch=40   # override any PARAMS entry
    python pipeline/run.py --name v20 --touch-net      # the current submission
    python tools/report.py evaluation/results/v1.csv        # then look at it

The same command with the same parameters always writes the same file.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from pipeline.features import build_features  # noqa: E402
from pipeline.decode import PARAMS, LETTERS, decode  # noqa: E402
import possession_metrics as pm  # noqa: E402
from games import GAME, GAMES  # noqa: E402

# Params
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")


def parse_overrides(pairs):
    """['switch=40', ...] -> PARAMS copy with those values."""
    params = dict(PARAMS)
    for pair in pairs or []:
        key, _, value = pair.partition("=")
        if key not in params:
            sys.exit("unknown parameter {!r}; have {}".format(key, sorted(params)))
        params[key] = float(value)
    return params


def write_submission(path, codes):
    """frame,label CSV; MTAH_VARIANT (tools/atd_cache.variant) is added to the file stem."""
    stem, ext = os.path.splitext(path)
    path = stem + os.environ.get("MTAH_VARIANT", "") + ext
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, LETTERS[c]) for i, c in enumerate(codes))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="v1", help="output file stem")
    parser.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="override PARAMS")
    parser.add_argument("--touch", nargs="*", metavar="KEY=VALUE",
                        help="override TOUCH_* / KEEPER_* constants in pipeline.features, "
                             "e.g. TOUCH_DV_MS=2 TOUCH_R_M=3")
    parser.add_argument("--tag-sign", nargs="*", metavar="CODE=SIGN",
                        help="extra Smart Tagging team readings, e.g. 'players in the box=1'")
    parser.add_argument("--ball-fill", choices=("hold", "extrapolate", "interp"),
                        help="guess the ball inside gaps (pipeline/ball_fill.py)")
    parser.add_argument("--touch-net", action="store_true",
                        help="touches from the touch network trained on DFL "
                             "(models/touch_net_all.pt, from tools/train_touch.py --final)")
    args = parser.parse_args()

    import pipeline.features as features
    for pair in args.tag_sign or []:
        code, _, sign = pair.rpartition("=")
        features.EXTRA_TAG_SIGN[code.strip().lower()] = int(sign)
    for pair in args.touch or []:
        key, _, value = pair.partition("=")
        if not key.startswith(("TOUCH_", "KEEPER_")) or not hasattr(features, key):
            sys.exit("unknown TOUCH_/KEEPER_ constant {!r}".format(key))
        setattr(features, key, float(value) if "." in value or key != "TOUCH_K" else int(value))
    params = parse_overrides(args.set)
    model = None
    if args.touch_net:
        from pipeline import touch_net
        model = touch_net.load(os.path.join(ROOT, touch_net.MODEL_PATH))
    codes = decode(build_features(ball_fill=args.ball_fill, touch_model=model), params)
    path = write_submission(os.path.join(RESULTS_DIR, args.name + ".csv"), codes)

    reason = pm.validate_submission(pm.read_label_csv(path), GAMES[GAME]["n_video_frames"])
    print("{} -> {}  ({})".format(args.name, os.path.relpath(path, ROOT), reason or "valid"))
    print("  " + "  ".join("{} {:5.1f}%".format(LETTERS[k], 100.0 * np.mean(codes == k))
                           for k in range(4)))
    changed = {k: v for k, v in params.items() if PARAMS[k] != v}
    if changed:
        print("  overrides: {}".format(changed))


if __name__ == '__main__':
    main()
