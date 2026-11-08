"""v31_kt rebuilt with the YOLO ball (tools/yolo_ball.py clean). Aggregate only.

    $env:MTAH_VARIANT="_kt"; python tools/ours_yolo.py

Runs ours_chain.ours_v31() with MTAH_BALL=yolo (atd_cache swaps the ball's x / y), writes
evaluation/results/v31_yolo_kt.csv and prints its public line next to v31_kt's; caches the touch
network's P(home / away touch) and the ball for the viewer (data/cache/yolo/v31_yolo_extras.npz).
"""
import os
import sys

MODE = os.environ.get("YOLO_MODE", "touch")      # touch: only the touch inputs see the YOLO-refined ball
if MODE == "touch":                               # ball: the whole chain on the cleaned YOLO ball
    os.environ["MTAH_TOUCH_BALL"] = "yolo"
else:
    os.environ["MTAH_BALL"] = "yolo"
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import ours_chain as oc  # noqa: E402
import possession_metrics as pm  # noqa: E402


def main():
    tp = os.environ.get("YOLO_TOUCH_MODEL")       # e.g. data/cache/models/touch_net_yolo_all.pt
    codes, ex = oc.ours_v31(touch_path=os.path.join(ROOT, tp) if tp else None)
    tag = (os.environ.get("MTAH_BALL_TAG", "") + ("_touch" if MODE == "touch" else "")
           + ("net" if os.environ.get("YOLO_TOUCH_MODEL") else ""))
    print("  " + str(oc.write_and_score(codes, "v31_yolo" + tag)))
    base = pm.label_codes(pm.read_label_csv(os.path.join(ROOT, "evaluation", "results",
                                                         "v31{}.csv".format(os.environ.get("MTAH_VARIANT", "")))))
    print("frames different from v31{}: {:.1%}; possessions {} vs {}".format(
        os.environ.get("MTAH_VARIANT", ""), np.mean(codes != base),
        len(pm.possession_spells(codes, pm.valid_runs(codes))), len(pm.possession_spells(base, pm.valid_runs(base)))))
    np.savez(os.path.join(ROOT, "data", "cache", "yolo", "v31_yolo{}_extras.npz".format(tag)),
             touch_p_home=ex["f_touch_p_home"], touch_p_away=ex["f_touch_p_away"], touch=ex["f_touch"])


if __name__ == "__main__":
    main()
