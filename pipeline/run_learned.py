"""Label BAR-ATH with the learned evidence model trained on external matches.

    python pipeline/run_learned.py --name v21 --source dfl --model gbm --costs 15 10 4
    python pipeline/run_learned.py --name v16 --source metrica          # the 2026-10-24 model
    python pipeline/run_learned.py ... --retrain

Models are trained on all external matches (each also home/away flipped) and saved, so later runs
reuse the same weights and write the same file: models/<model>_<source>.* in the repository when
present (committed, for exact rebuilds), otherwise data/cache/models/.

  metrica  the 3 Metrica games (event labels, rebuilt ball). Failed to transfer (v16).
  dfl      the 7 DFL matches prepared like ours (blurred, gapped ball; touch network trained
           without each match; controlled-possession labels), every gap pattern of
           tools/train_learned_dfl.py. Our match then uses the touch network trained on all 7.

Smart Tagging is not used (the training matches have none). Decoded per half by the same Viterbi,
costs in log-prob units. Prints label shares only.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from games import GAME, GAMES, DATA_DIR, half_video_blocks  # noqa: E402
from pipeline import learned  # noqa: E402
from pipeline.decode import LETTERS  # noqa: E402
from pipeline.run import write_submission, RESULTS_DIR  # noqa: E402

# Params
CACHE_DIR = os.path.join(DATA_DIR, "cache", "models")
REPO_DIR = os.path.join(ROOT, "models")
GBM_WEIGHT = 0.5      # blend: P = w * P_gbm + (1 - w) * P_gru
COSTS = {"switch": 10.0, "to_dead": 25.0, "from_dead": 4.0}


def training(source):
    """[(features, labels)] for every external training match (and gap pattern)."""
    if source == "metrica":
        import metrica_data as md
        from train_external import game_features
        return [(game_features(g), md.load(g)["labels"]) for g in md.GAMES]
    import dfl_data as dd
    from train_learned_dfl import match_features, test_seed, TRAIN_VERSIONS
    out = []
    for g in dd.GAMES:
        lab = dd.load(g)["controlled"]
        for seed in [test_seed(g)] + [test_seed(g) + 1 + v for v in range(TRAIN_VERSIONS)]:
            out.append((match_features(g, seed), lab))
    return out


def model_path(kind, source):
    name = "{}_{}.{}".format(kind, source, "txt" if kind == "gbm" else "pt")
    repo = os.path.join(REPO_DIR, name)
    return repo if os.path.isfile(repo) else os.path.join(CACHE_DIR, name)


def models(source, kinds, retrain=False):
    """{kind: model} for kind in kinds ("gbm", "gru"), trained on `source`; cached on disk."""
    import torch
    from pipeline.sequence import train_gru, _net
    os.makedirs(CACHE_DIR, exist_ok=True)
    paths = {k: model_path(k, source) for k in kinds}
    if retrain or not all(os.path.isfile(p) for p in paths.values()):
        train = training(source)
        paths = {k: os.path.join(CACHE_DIR, os.path.basename(p)) for k, p in paths.items()}
        if "gbm" in kinds:
            step = 3 if source == "dfl" else 2
            gbm = learned.train_gbm([learned.training_rows(f, lab, step=step) for f, lab in train])
            gbm.save_model(paths["gbm"])
        if "gru" in kinds:
            model, mean, std = train_gru(train, log=lambda *_: None)
            torch.save({"state": model.state_dict(), "mean": mean, "std": std,
                        "n_in": len(mean)}, paths["gru"])
    out = {}
    if "gbm" in kinds:
        import lightgbm as lgb
        out["gbm"] = lgb.Booster(model_file=paths["gbm"])
    if "gru" in kinds:
        saved = torch.load(paths["gru"], weights_only=False)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        net = _net(saved["n_in"]).to(dev)
        net.load_state_dict(saved["state"])
        out["gru"] = (net, saved["mean"], saved["std"])
    return out


def main():
    from pipeline.features import build_features
    from pipeline.sequence import predict_gru
    from pipeline import touch_net
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="v16")
    parser.add_argument("--source", choices=("metrica", "dfl"), default="metrica")
    parser.add_argument("--model", choices=("gbm", "gru", "blend"), default="blend")
    parser.add_argument("--costs", nargs=3, type=float, metavar=("SWITCH", "TO_DEAD", "FROM_DEAD"))
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--touch-model", help="touch network weights (default: the committed "
                                              "models/touch_net_all.pt)")
    parser.add_argument("--model-dir", help="read / write the learned models here instead")
    parser.add_argument("--gbm-weight", type=float, default=GBM_WEIGHT)
    parser.add_argument("--hybrid", type=float, metavar="SWITCH",
                        help="hand v20 decides D vs live, the learned model the team, with this "
                             "switch cost (learned.decode_hybrid)")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="input-column prefixes to leave out (learned.EXCLUDE)")
    args = parser.parse_args()
    learned.EXCLUDE.update(args.exclude)

    kinds = ("gbm", "gru") if args.model == "blend" else (args.model,)
    if args.model_dir:
        global REPO_DIR, CACHE_DIR
        REPO_DIR = CACHE_DIR = args.model_dir
    trained = models(args.source, kinds, args.retrain)
    tm_path = args.touch_model or os.path.join(ROOT, touch_net.MODEL_PATH)
    tm = touch_net.load(tm_path) if args.source == "dfl" else None
    f = build_features(events=[], touch_model=tm)
    probs = {}
    if "gbm" in trained:
        probs["gbm"] = learned.predict_proba(trained["gbm"], f)
    if "gru" in trained:
        probs["gru"] = predict_gru(trained["gru"], f)
    prob = (args.gbm_weight * probs["gbm"] + (1 - args.gbm_weight) * probs["gru"]
            if args.model == "blend" else probs[args.model])
    costs = dict(zip(("switch", "to_dead", "from_dead"), args.costs)) if args.costs else COSTS
    blocks = [(first, last + 1) for _, first, last in half_video_blocks(GAME)]
    if args.hybrid is not None:
        from pipeline.decode import decode
        hand = decode(build_features(touch_model=tm))          # v20, Smart Tagging included
        codes = learned.decode_hybrid(hand, prob, blocks, args.hybrid)
    else:
        codes = learned.decode_probs(prob, blocks, costs)
    path = write_submission(os.path.join(RESULTS_DIR, args.name + ".csv"), codes)
    reason = pm.validate_submission(pm.read_label_csv(path), GAMES[GAME]["n_video_frames"])
    print("{} -> {}  ({})".format(args.name, os.path.relpath(path, ROOT), reason or "valid"))
    print("  " + "  ".join("{} {:5.1f}%".format(LETTERS[k], 100.0 * np.mean(codes == k))
                           for k in range(4)))


if __name__ == '__main__':
    main()
