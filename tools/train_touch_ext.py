"""Touch network retrained with SkillCorner + PFF added to DFL (candidate (a) for short possessions).

    python tools/train_touch_ext.py            # 4 fold models + one on everything

Training data per model:
  DFL, all 7 matches, as tools/train_touch.py (ball made to look like ours, VERSIONS gap patterns,
  each also with the teams swapped; teacher touches from tools/touch_labels.py);
  SkillCorner + PFF matches outside the fold (run_v31_external.fold_of), each with its own
  broadcast ball blurred like DFL (no extra gaps, no swap); touches from the events: every player
  possession's (SkillCorner) / OTB event's (PFF) first touch, plus its release when it lasts
  >= RELEASE_S; only graded frames count.
Saves data/cache/models/touch_ext/touch_f<k>.pt (fold k held out) and touch_all.pt. Prints, per
fold, touch F1 and turnover recall within 0.5 s on the held-out external matches for the deployed
network and the new one (thresholds 0.5 / 0.7).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import dfl_data as dd  # noqa: E402
import train_touch as tt  # noqa: E402
import run_v31_external as rx  # noqa: E402
from pipeline import touch_net as tn  # noqa: E402

# Params
OUT = os.path.join(dd.DATA_DIR, "cache", "models", "touch_ext")
RELEASE_S = 0.3
VERSIONS = 2
SETS = ("skillcorner", "pff")


def event_touches(d):
    """(frames, teams) from the set's events (start, end, team in columns 0-2)."""
    fr, te = [], []
    for a, b, t in d["events"][:, :3]:
        fr.append(int(a))
        te.append(int(t))
        if (b - a) / dd.FPS >= RELEASE_S:
            fr.append(int(b))
            te.append(int(t))
    o = np.argsort(fr, kind="stable")
    return np.array(fr)[o], np.array(te, dtype=np.int8)[o]


def external(name, game):
    d, pitch, ball = rx.load_set(name, game)
    ball = dd.blur_ball(ball)
    frames, teams = event_touches(d)
    x = tn.inputs(d["players"], ball, d["team"], pitch)
    return dict(x=x, y=tn.targets(len(x), frames, teams), valid=d["labels"] != dd.X,
                frames=frames, teams=teams, labels=d["labels"])


def main():
    os.makedirs(OUT, exist_ok=True)
    print("DFL inputs ...", flush=True)
    dfl = []
    for g in dd.GAMES:
        for v in range(VERSIONS):
            d, ball, frames, teams, valid = tt.prepared(g, seed=v)
            for flip in (False, True):
                x = tn.inputs(d["players"], ball, d["team"], dd.PITCH, flip=flip)
                dfl.append((x, tn.targets(len(x), frames, teams, flip=flip), valid))
    print("external inputs ...", flush=True)
    ext = {}
    for name in SETS:
        for g in rx.games_of(name):
            ext[(name, g)] = external(name, g)
    deployed = tn.load(os.path.join(ROOT, tn.MODEL_PATH))
    for k in list(range(rx.FOLDS)) + ["all"]:
        train = list(dfl) + [(e["x"], e["y"], e["valid"]) for (name, g), e in ext.items()
                             if k == "all" or rx.fold_of(name, g) != k]
        print("fold {}: {} training sequences ...".format(k, len(train)), flush=True)
        bundle = tn.train(train, log=lambda *_: None)
        tn.save(bundle, os.path.join(OUT, "touch_{}.pt".format("all" if k == "all" else
                                                               "f{}".format(k))))
        if k == "all":
            continue
        res = {}
        for (name, g), e in ext.items():
            if rx.fold_of(name, g) != k:
                continue
            cf, ct = tt.changes(e["labels"])
            for tag, model in (("deployed", deployed), ("new", bundle)):
                prob = tn.predict(model, e["x"])
                for th in (0.5, 0.7):
                    pf, pt = tn.pick(prob, th)
                    keep = e["valid"][pf]
                    pf, pt = pf[keep], pt[keep]
                    s_t = tt.scores(pf, pt, e["frames"], e["teams"])[0.5]
                    s_c = tt.scores(pf, pt, cf, ct)[0.5]
                    res.setdefault((name, tag, th), []).append((s_t[2], s_c[1], s_c[3]))
        for key in sorted(res):
            m = np.nanmean(res[key], axis=0)
            print("  {:<11} {:<8} @{:g}: touch F1 {:.3f} | possession starts R {:.3f}, "
                  "turnover R {:.3f}".format(*key, *m), flush=True)


if __name__ == "__main__":
    main()
