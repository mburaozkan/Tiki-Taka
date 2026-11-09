"""Touch network on our match's tracking ball or YOLO-refined ball: aggregates only.

    $env:MTAH_VARIANT="_kt"; $env:YOLO_TAGS="-,_l8"; python tools/yolo_touch_check.py <model.pt> [...]

Per model and ball: the new team's max touch P within +-0.5 s of public starts #7 / #10 / #12 / #15
(spell_table numbering), and touch peaks (touch_net.pick at 0.5 and at the deployed 0.7) per minute
of seen tracking ball over the full match.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "starter")]
import possession_metrics as pm  # noqa: E402
from report import PUBLIC_GT  # noqa: E402
from atd_cache import load_arrays  # noqa: E402
from pipeline import touch_net as tn  # noqa: E402
import pipeline.features as features  # noqa: E402

STARTS = (7, 10, 12, 15)


def main():
    atd = load_arrays("tactical")
    balls = [("tracking", atd["ball"])]
    for tag in os.environ.get("YOLO_TAGS", "-").split(","):   # e.g. -,_l8,_l16 (- = plain refined)
        ref = np.load(os.path.join(ROOT, "data", "cache", "yolo", "ball_refined_bambezius{}.npy".format(
            "" if tag == "-" else tag)))
        yb = atd["ball"].copy()
        yb[:len(ref), :2] = ref
        balls.append(("yolo" + ("" if tag == "-" else tag), yb))
    seen_min = np.isfinite(atd["ball"][:, 0]).sum() / 1500
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    sp = pm.possession_spells(gt, pm.valid_runs(gt))
    print("{:<34} {:<9} {:>22} {:>8} {:>8}".format("model", "ball", "P at #" + "/#".join(map(str, STARTS)),
                                                  "pk@0.5", "pk@0.7"))
    for mp in sys.argv[1:] or [tn.MODEL_PATH]:
        m = tn.load(mp if os.path.isabs(mp) else os.path.join(ROOT, mp))
        path = len(m[1]) > tn.n_base_inputs()
        for nm, ball in balls:
            prob = tn.predict(m, tn.inputs(atd["players"], ball, atd["team"], (features.LENGTH, features.WIDTH),
                                           path=path))
            ps = []
            for i in STARTS:
                a, b, k = sp[i - 1]
                ps.append("{:.2f}".format(prob[max(a - 12, 0):a + 13, 1 + k].max()))
            pk = [len(tn.pick(prob, th)[0]) / seen_min for th in (0.5, 0.7)]
            print("{:<34} {:<9} {:>22} {:>8.1f} {:>8.1f}".format(os.path.basename(mp), nm, "/".join(ps), *pk))


if __name__ == "__main__":
    main()
