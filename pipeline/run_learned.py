"""Label BAR-ATH with the learned evidence model trained on the Metrica matches.

    python pipeline/run_learned.py --name v16          # GBM + GRU blend, the 2026-10-24 choice
    python pipeline/run_learned.py --name v16 --retrain

Both models are trained on all three Metrica games (each also home/away flipped) and saved to
data/cache/models/, so later runs reuse the same weights and write the same file. Our match's
features come from pipeline.features as for the hand model; Smart Tagging is not used (the
training matches have none). Decoded per half by the same Viterbi, costs in log-prob units.
Prints label shares only.
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
MODEL_DIR = os.path.join(DATA_DIR, "cache", "models")
GBM_WEIGHT = 0.5      # blend: P = w * P_gbm + (1 - w) * P_gru (Metrica leave-one-out: 0.822)
COSTS = {"switch": 10.0, "to_dead": 25.0, "from_dead": 4.0}


def metrica_training():
    """[(features, labels)] for the three Metrica games (cached features)."""
    import metrica_data as md
    from train_external import game_features
    return [(game_features(g), md.load(g)["labels"]) for g in md.GAMES]


def models(retrain=False):
    """(gbm, gru) trained on all Metrica games; cached on disk."""
    import torch
    from pipeline.sequence import train_gru, _net
    os.makedirs(MODEL_DIR, exist_ok=True)
    gbm_path = os.path.join(MODEL_DIR, "gbm_metrica.txt")
    gru_path = os.path.join(MODEL_DIR, "gru_metrica.pt")
    if retrain or not (os.path.isfile(gbm_path) and os.path.isfile(gru_path)):
        train = metrica_training()
        gbm = learned.train_gbm([learned.training_rows(f, lab) for f, lab in train])
        gbm.save_model(gbm_path)
        model, mean, std = train_gru(train, log=lambda *_: None)
        torch.save({"state": model.state_dict(), "mean": mean, "std": std,
                    "n_in": len(mean)}, gru_path)
    import lightgbm as lgb
    gbm = lgb.Booster(model_file=gbm_path)
    saved = torch.load(gru_path, weights_only=False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = _net(saved["n_in"]).to(dev)
    net.load_state_dict(saved["state"])
    return gbm, (net, saved["mean"], saved["std"])


def main():
    from pipeline.features import build_features
    from pipeline.sequence import predict_gru
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="v16")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--gbm-weight", type=float, default=GBM_WEIGHT)
    args = parser.parse_args()

    gbm, gru = models(args.retrain)
    f = build_features(events=[])
    prob = (args.gbm_weight * learned.predict_proba(gbm, f)
            + (1 - args.gbm_weight) * predict_gru(gru, f))
    blocks = [(first, last + 1) for _, first, last in half_video_blocks(GAME)]
    codes = learned.decode_probs(prob, blocks, COSTS)
    path = write_submission(os.path.join(RESULTS_DIR, args.name + ".csv"), codes)
    reason = pm.validate_submission(pm.read_label_csv(path), GAMES[GAME]["n_video_frames"])
    print("{} -> {}  ({})".format(args.name, os.path.relpath(path, ROOT), reason or "valid"))
    print("  " + "  ".join("{} {:5.1f}%".format(LETTERS[k], 100.0 * np.mean(codes == k))
                           for k in range(4)))


if __name__ == '__main__':
    main()
