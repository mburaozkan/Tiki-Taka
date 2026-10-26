"""In-domain team model: pseudo-label a match's own confident frames, train a players-only team
model on them, and use it as an extra team vote where the ball evidence is weak. Aggregate only.

    python tools/in_domain_team.py               # DFL validation (each match treated as unlabelled)
    python tools/in_domain_team.py --ours        # also: our match, on top of v21 -> v24 csvs

Why: the learned team models are trained on DFL and transfer with a shift (JOURNAL.md); our own
match has ~100k frames whose team is clear from the ball alone. A model trained on those frames
sees our teams, our pace and our tracking.

Pseudo-labels (pseudo_labels): ball seen, the nearest player within REACH_M, its margin over the
nearest opponent >= MARGIN_M, co-movement not disagreeing (m_home - m_away has the same sign, or
reads nothing), and the hand path live. Label = the nearest player's team.

Players-only model (no ball input at all, so it also reads in gaps and contested frames): per team
pipeline.reaction features (spread, centroid, push toward goal, back line, compactness, their
future-minus-past differences), pressure (mean distance from each player to the nearest opponent),
mean speed, each with past / future means over CONTEXT_S (up to +/- 10 s). LightGBM binary H vs A,
trained on the match and its home/away-swapped copy.

Combination: in the hybrid decoder's team stage, logit P(H) += w * logit P_po(H) on weak frames
(ball not seen, or nearest player farther than REACH_M, or margin < MARGIN_M). The base team
probabilities on DFL are the held-out learned probabilities (learned_probs_anim.npz).

DFL validation: per match, pseudo-labels from its own realistic features (true labels unused),
model trained on that match only, scored with the true labels. Reports pseudo-label coverage and
precision, the players-only model's accuracy on weak live frames against the base model's, and
hybrid spellF1 per match for each w (folds won vs w = 0).
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
import dfl_data as dd  # noqa: E402
from train_learned_dfl import match_features, test_seed, blocks  # noqa: E402
from pipeline import learned, reaction  # noqa: E402
from pipeline.decode import PARAMS, emissions, decode_block, X, H, A  # noqa: E402

# Params
REACH_M = 2.0
MARGIN_M = 1.5
CONTEXT_S = (2, 5, 10)
WEIGHTS = (0.0, 0.25, 0.5, 1.0)
SWITCHES = (10.0, 20.0)
STEP = 2
PROBS = os.path.join(dd.CACHE, "learned_probs_anim.npz")


def pseudo_labels(f, hand_codes):
    """(team H/A per frame, confident mask, weak mask)."""
    dh = np.asarray(f["d_home"], dtype=np.float64)
    da = np.asarray(f["d_away"], dtype=np.float64)
    seen = f["ball_seen"] > 0
    with np.errstate(invalid="ignore"):
        near = np.fmin(dh, da) <= REACH_M
        margin = np.abs(dh - da) >= MARGIN_M
        team = np.where(dh < da, H, A)
        md = np.asarray(f["m_home"], dtype=np.float64) - np.asarray(f["m_away"], dtype=np.float64)
        agree = np.isnan(md) | (md == 0) | ((md > 0) == (team == H))
    live = np.isin(hand_codes, (H, A))
    strong = seen & near & margin
    return team, strong & agree & live, ~strong


def _pairwise_min(a, b, chunk=20000):
    """Per frame, mean over a's players of the distance to the nearest of b's players."""
    out = np.full(len(a), np.nan)
    for s in range(0, len(a), chunk):
        d = np.sqrt(((a[s:s + chunk, :, None, :] - b[s:s + chunk, None, :, :]) ** 2).sum(-1))
        with np.errstate(invalid="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out[s:s + chunk] = np.nanmean(np.nanmin(d, axis=2), axis=1)
    return out


def player_columns(players, team, groups, pitch):
    """{name: (n,)} players-only inputs, home/away pairs named *_home / *_away."""
    L, W = pitch
    cols = reaction.reaction_features(players, team, groups, pitch)
    pos = np.stack([(players[:, :, 0] - 0.5) * L, (players[:, :, 1] - 0.5) * W], axis=-1)
    pos = pos.astype(np.float32)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for s, side in enumerate(reaction.SIDES):
            other = 1 - s
            cols["press_" + side] = _pairwise_min(pos[:, team == s], pos[:, team == other])
            cols["speed_" + side] = np.nanmean(players[:, team == s, 2].astype(np.float64), axis=1)
    levels = [k for k in cols if "_d" not in k.split("_", 2)[-1]]
    for k in levels:
        x = cols[k]
        for c in CONTEXT_S:
            w = int(c * dd.FPS)
            cols[k + "_p{}".format(c)] = learned.window(x, -w, 0)
            cols[k + "_f{}".format(c)] = learned.window(x, 1, w + 1)
    return cols


def po_matrix(cols, flip=False):
    names = sorted(k for k in cols if "_home" in k)
    out = []
    for k in names:
        a, b = cols[k], cols[k.replace("_home", "_away")]
        out += [b, a] if flip else [a, b]
    return np.stack(out, axis=1).astype(np.float32)


def train_po(cols, team, conf, seed=0):
    import lightgbm as lgb
    idx = np.flatnonzero(conf)[::STEP]
    xs = np.concatenate([po_matrix(cols)[idx], po_matrix(cols, True)[idx]])
    ys = np.concatenate([(team[idx] == H), (team[idx] == A)]).astype(np.float64)
    params = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31,
              "min_data_in_leaf": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8,
              "bagging_freq": 1, "lambda_l2": 1.0, "seed": seed, "verbose": -1}
    return lgb.train(params, lgb.Dataset(xs, ys), num_boost_round=300)


def predict_po(model, cols):
    """P(H) from the players alone, averaged with the swapped prediction."""
    return (model.predict(po_matrix(cols)) + 1.0 - model.predict(po_matrix(cols, True))) / 2.0


def combine(prob, p_po, weak, w, floor=1e-4):
    """(n, 3) probabilities with the team stage's log-odds moved by w * logit(p_po) on weak frames."""
    ph = np.clip(prob[:, H], floor, 1.0)
    pa = np.clip(prob[:, A], floor, 1.0)
    q = np.clip(p_po, floor, 1 - floor)
    z = np.log(ph / pa) + w * weak * np.log(q / (1 - q))
    out = prob.copy()
    live = ph + pa
    out[:, H] = live / (1.0 + np.exp(-z))
    out[:, A] = live - out[:, H]
    return out


def dfl_validation():
    base = {}
    with np.load(PROBS) as c:
        for k in c.files:
            name, g = k.split("__")
            base.setdefault(name, {})[g] = c[k]
    base["blend"] = {g: (base["gbm"][g] + base["gru"][g]) / 2 for g in dd.GAMES}
    res = {}
    print("{:<8} {:>9} {:>9} {:>11} {:>11} {:>11}".format(
        "match", "pseudo %", "pseudo ok", "weak live", "po on weak", "base on weak"))
    for g in dd.GAMES:
        d = dd.load(g)
        f = match_features(g, test_seed(g))
        lab, blk = d["controlled"], blocks(d["period"])
        sc = emissions(f, PARAMS)
        hand = np.full(len(lab), X, dtype=np.int8)
        for a, b in blk:
            hand[a:b] = decode_block(sc, f, PARAMS, a, b)
        team, conf, weak = pseudo_labels(f, hand)
        cols = player_columns(d["players"], d["team"], d["period"], dd.PITCH)
        p_po = predict_po(train_po(cols, team, conf), cols)
        live = np.isin(lab, (H, A))
        wl = weak & live
        po_ok = np.mean((p_po[wl] >= 0.5) == (lab[wl] == H))
        b = base["gbm"][g]
        base_ok = np.mean((b[wl, H] >= b[wl, A]) == (lab[wl] == H))
        print("{:<8} {:>8.1f}% {:>8.1f}% {:>10.1f}% {:>10.1f}% {:>10.1f}%".format(
            g, 100 * conf[lab != X].mean(), 100 * np.mean(team[conf & live] == lab[conf & live]),
            100 * wl.sum() / max(live.sum(), 1), 100 * po_ok, 100 * base_ok), flush=True)
        for name in ("gbm", "blend"):
            for sw in SWITCHES:
                for w in WEIGHTS:
                    pred = learned.decode_hybrid(hand, combine(base[name][g], p_po, weak, w), blk, sw)
                    r = pm.evaluate(lab, pred, bootstrap_n=0)
                    res.setdefault((name, sw, w), []).append(r["spell_f1_mean"])
    print("\nhybrid spellF1 per match | mean | folds won vs w=0")
    for (name, sw, w), s in res.items():
        ref = res[(name, sw, 0.0)]
        print("{:<6} switch {:>4g} w {:<5g} {} | {:.3f} | {}/7".format(
            name, sw, w, " ".join("{:.3f}".format(x) for x in s), np.mean(s),
            sum(x > y + 1e-9 for x, y in zip(s, ref))), flush=True)


def ours(weights):
    """v21 plus the in-domain vote on our match: evaluation/results/v24_w<w>.csv."""
    from games import GAME, GAMES, half_video_blocks
    from atd_cache import load_arrays
    from pipeline.features import build_features
    from pipeline.sequence import predict_gru
    from pipeline import touch_net
    from pipeline.decode import decode
    from pipeline.run import write_submission, RESULTS_DIR
    import run_learned as rl
    tm = touch_net.load(os.path.join(ROOT, touch_net.MODEL_PATH))
    f = build_features(events=[], touch_model=tm)
    trained = rl.models("dfl", ("gbm", "gru"))
    prob = 0.5 * learned.predict_proba(trained["gbm"], f) + 0.5 * predict_gru(trained["gru"], f)
    hand = decode(build_features(touch_model=tm))
    blks = [(a, b + 1) for _, a, b in half_video_blocks(GAME)]
    groups = np.zeros(len(hand), dtype=np.int8)
    for i, (a, b) in enumerate(blks, start=1):
        groups[a:b] = i
    team, conf, weak = pseudo_labels(f, hand)
    atd = load_arrays()
    from pipeline.features import LENGTH, WIDTH
    cols = player_columns(atd["players"], atd["team"], groups, (LENGTH, WIDTH))
    p_po = predict_po(train_po(cols, team, conf & (groups > 0)), cols)
    graded = groups > 0
    print("ours: pseudo-labelled {:.1f}% of graded frames, weak {:.1f}%; players-only agrees "
          "with v21's team on {:.1f}% of v21's live weak frames".format(
              100 * conf[graded].mean(), 100 * weak[graded].mean(),
              100 * np.mean((p_po >= 0.5)[graded & weak & np.isin(hand, (H, A))] ==
                            (learned.decode_hybrid(hand, prob, blks, 10.0) == H)[
                                graded & weak & np.isin(hand, (H, A))])))
    for w in weights:
        codes = learned.decode_hybrid(hand, combine(prob, p_po, weak, w), blks, 10.0)
        path = write_submission(os.path.join(RESULTS_DIR, "v24_w{:g}.csv".format(w)), codes)
        reason = pm.validate_submission(pm.read_label_csv(path), GAMES[GAME]["n_video_frames"])
        print("  v24_w{:g} -> {} ({})".format(w, os.path.relpath(path, ROOT), reason or "valid"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ours", action="store_true")
    parser.add_argument("--skip-dfl", action="store_true")
    args = parser.parse_args()
    if not args.skip_dfl:
        dfl_validation()
    if args.ours:
        sys.path.insert(0, os.path.join(ROOT, "pipeline"))
        ours(WEIGHTS)


if __name__ == '__main__':
    main()
