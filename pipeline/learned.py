"""Learned evidence: per-frame P(H), P(A), P(D) from the features, decoded by the same Viterbi.

    from pipeline.learned import matrix, flip, train_gbm, decode_probs

Inputs are the pipeline features (no Smart Tagging: the external training matches have none), each
with context from before and after the frame: means over past and future windows, time since the
last touch and until the next, the side of the next touch. Everything is home/away symmetric:
flip(f) swaps the two teams, so training on f and flip(f) (labels H <-> A) removes any home bias.

The model replaces the hand-set emission weights in pipeline/decode.py. The decoder (switch and
stoppage costs) stays, now in units of log-probability.
"""

import numpy as np

from pipeline.decode import H, A, D, X, viterbi, emissions, transition_cost, PARAMS

# Params
WINDOWS = (12, 25, 50)           # frames: ~0.5, 1, 2 s, each as a past and a future mean
SWAP = (("d_home", "d_away"), ("c_home", "c_away"), ("m_home", "m_away"), ("g_home", "g_away"),
        ("touch_p_home", "touch_p_away"))
NEGATE = ("touch", "last_touch", "shape", "keeper_hold")
EXCLUDE = set()                  # column-name prefixes left out (domain-shift experiments)
FAR = 30.0                       # a missing distance reads as this far (m)
DECODER = {"switch": 6.0, "to_dead": 6.0, "from_dead": 4.0}   # log-prob units


QNORM_KEYS = ("player_speed", "ball_speed")       # ranked alone
QNORM_POOLED = (("touch_p_home", "touch_p_away"),)  # ranked on their joint distribution


def _ranks(x, groups):
    """x mapped to its empirical CDF (0-1) within each group; NaN stays NaN."""
    out = np.full(len(x), np.nan)
    for g in np.unique(groups):
        idx = np.flatnonzero((groups == g) & ~np.isnan(x))
        if len(idx) > 1:
            order = np.argsort(x[idx], kind="stable")
            r = np.empty(len(idx))
            r[order] = np.arange(len(idx)) / (len(idx) - 1.0)
            out[idx] = r
    return out


def qnormalise(f, groups):
    """Copy of f with pace, ball speed and the touch probabilities replaced by their within-group
    quantiles (groups: one id per frame, e.g. the half), and the touch timers marked for ranking in
    matrix(). Removes level shifts between matches (our live play is slower than DFL's)."""
    g = dict(f)
    groups = np.asarray(groups)
    for k in QNORM_KEYS:
        g[k] = _ranks(np.asarray(f[k], dtype=np.float64), groups)
    for a, b in QNORM_POOLED:
        if a in f and b in f:
            both = _ranks(np.concatenate([f[a], f[b]]).astype(np.float64),
                          np.concatenate([groups, groups]))
            g[a], g[b] = both[:len(groups)], both[len(groups):]
    g["_qgroup"] = groups
    return g


def flip(f):
    """The same match with home and away swapped."""
    g = dict(f)
    for a, b in SWAP:
        if a in f and b in f:
            g[a], g[b] = f[b], f[a]
    for k in NEGATE:
        g[k] = -f[k]
    return g


def window(x, lo, hi):
    """Mean of x over frames t+lo .. t+hi-1, ignoring NaN (NaN if none)."""
    n = len(x)
    ok = ~np.isnan(x)
    cs = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    cn = np.concatenate(([0], np.cumsum(ok)))
    t = np.arange(n)
    a, b = np.clip(t + lo, 0, n), np.clip(t + hi, 0, n)
    cnt = cn[b] - cn[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (cs[b] - cs[a]) / np.maximum(cnt, 1), np.nan)


def touch_timing(touch):
    """Seconds since the last touch, until the next, and the next touch's side (0 if none)."""
    n = len(touch)
    idx = np.where(touch != 0, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    nxt = np.where(touch != 0, np.arange(n), n)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    since = np.where(last >= 0, (np.arange(n) - last) / 25.0, 60.0)
    until = np.where(nxt < n, (nxt - np.arange(n)) / 25.0, 60.0)
    side = np.where(nxt < n, touch[np.minimum(nxt, n - 1)], 0.0)
    return np.minimum(since, 60.0), np.minimum(until, 60.0), side


def hand_columns(f, blocks):
    """The hand model's per-frame scores (H, A, D) and its decoded path (one-hot), for stacking."""
    sc = emissions(f, PARAMS)
    cost = transition_cost(PARAMS)
    path = np.full(len(sc), D, dtype=np.int8)
    for a, b in blocks:
        path[a:b] = viterbi(sc[a:b], cost)
    onehot = np.eye(3)[path]
    return {"hand_sH": sc[:, H], "hand_sA": sc[:, A], "hand_sD": sc[:, D],
            "hand_H": onehot[:, H], "hand_A": onehot[:, A], "hand_D": onehot[:, D]}


def matrix(f, blocks=None):
    """(n, k) float32 design matrix and the column names. With blocks, adds hand_columns."""
    dh = np.nan_to_num(f["d_home"], nan=FAR)
    da = np.nan_to_num(f["d_away"], nan=FAR)
    base = {
        "seen": f["ball_seen"], "gap": f["gap_s"], "out": f["ball_out"],
        "bspeed": np.nan_to_num(f["ball_speed"], nan=-1.0),
        "pspeed": np.nan_to_num(f["player_speed"], nan=-1.0),
        "dh": dh, "da": da, "ddiff": da - dh, "dmin": np.minimum(dh, da),
        "ch": np.nan_to_num(f["c_home"], nan=FAR), "ca": np.nan_to_num(f["c_away"], nan=FAR),
        "touch": f["touch"], "last_touch": f["last_touch"],
        "mh": np.nan_to_num(f["m_home"], nan=-1.0), "ma": np.nan_to_num(f["m_away"], nan=-1.0),
        "shape": np.nan_to_num(f["shape"]), "keeper": f["keeper_hold"],
        "gh": np.nan_to_num(f["g_home"], nan=-1.0), "ga": np.nan_to_num(f["g_away"], nan=-1.0),
    }
    # Touch network probabilities (pipeline/touch_net.py); zeros when it was not run.
    tph = f.get("touch_p_home", np.zeros(len(dh), np.float32)).astype(np.float64)
    tpa = f.get("touch_p_away", np.zeros(len(dh), np.float32)).astype(np.float64)
    base.update({"tph": tph, "tpa": tpa, "tpdiff": tph - tpa})
    since, until, nside = touch_timing(f["touch"])
    if "_qgroup" in f:
        since, until = _ranks(since, f["_qgroup"]), _ranks(until, f["_qgroup"])
    base.update({"t_since": since, "t_until": until, "next_side": nside})
    # Context: past and future means of the signals that move at a change of possession.
    ctx = {"ddiff": np.where(np.isnan(f["d_home"]) | np.isnan(f["d_away"]), np.nan, da - dh),
           "mdiff": f["m_home"] - f["m_away"], "seen": f["ball_seen"],
           "bspeed": f["ball_speed"], "pspeed": f["player_speed"], "touch": f["touch"],
           "out": f["ball_out"], "shape": f["shape"], "tpdiff": tph - tpa, "tpsum": tph + tpa}
    cols = dict(base)
    if blocks is not None:
        cols.update(hand_columns(f, blocks))
    for name, x in ctx.items():
        x = x.astype(np.float64)
        for w in WINDOWS:
            cols["{}_p{}".format(name, w)] = window(x, -w, 0)
            cols["{}_f{}".format(name, w)] = window(x, 1, w + 1)
    names = [k for k in cols if not any(k.startswith(e) for e in EXCLUDE)]
    mat = np.stack([np.asarray(cols[k], dtype=np.float32) for k in names], axis=1)
    return mat, names


def training_rows(f, labels, step=2, blocks=None):
    """(X, y) from a match and its flipped copy, graded frames only, every `step`-th frame."""
    xs, ys = [], []
    for g, lab in ((f, labels), (flip(f), np.where(labels == H, A, np.where(labels == A, H, labels)))):
        mat, _ = matrix(g, blocks)
        keep = np.flatnonzero(lab != X)[::step]
        xs.append(mat[keep])
        ys.append(lab[keep])
    return np.concatenate(xs), np.concatenate(ys)


def domain_classifier(source_mats, target_mat, rounds=200, seed=0):
    """LightGBM telling target rows (our match) from source rows (DFL), classes balanced."""
    import lightgbm as lgb
    xs = np.concatenate(list(source_mats) + [target_mat])
    n_src = len(xs) - len(target_mat)
    ys = np.concatenate([np.zeros(n_src), np.ones(len(target_mat))])
    w = np.where(ys == 1, n_src / float(len(target_mat)), 1.0)
    params = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31,
              "min_data_in_leaf": 500, "feature_fraction": 0.8, "bagging_fraction": 0.5,
              "bagging_freq": 1, "seed": seed, "verbose": -1}
    return lgb.train(params, lgb.Dataset(xs, ys, weight=w), num_boost_round=rounds)


def domain_weights(clf, xs, clip=(0.1, 10.0)):
    """Importance weights p(ours) / p(DFL) per row (classes balanced in training), clipped and
    scaled to mean 1."""
    p = np.clip(clf.predict(xs), 1e-4, 1 - 1e-4)
    w = np.clip(p / (1 - p), *clip)
    return w / w.mean()


def train_gbm(rows, rounds=400, seed=0, weight_fn=None):
    """LightGBM multiclass on [(X, y), ...]; weight_fn(X) gives per-row weights (e.g.
    domain_weights)."""
    import lightgbm as lgb
    xs = np.concatenate([r[0] for r in rows])
    ys = np.concatenate([r[1] for r in rows])
    weight = weight_fn(xs) if weight_fn is not None else None
    params = {"objective": "multiclass", "num_class": 3, "learning_rate": 0.05,
              "num_leaves": 63, "min_data_in_leaf": 200, "feature_fraction": 0.8,
              "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "seed": seed,
              "verbose": -1}
    return lgb.train(params, lgb.Dataset(xs, ys, weight=weight), num_boost_round=rounds)


def predict_proba(model, f, blocks=None):
    """(n, 3) P(H, A, D), averaged with the flipped prediction (H and A swapped back)."""
    p = model.predict(matrix(f, blocks)[0])
    q = model.predict(matrix(flip(f), blocks)[0])[:, [A, H, D]]
    return (p + q) / 2.0


def decode_probs(prob, blocks, costs=DECODER, floor=1e-4):
    """Viterbi over log P within each (first, last+1) block; X elsewhere."""
    scores = np.log(np.clip(prob, floor, 1.0))
    c = np.zeros((3, 3))
    c[H, A] = c[A, H] = costs["switch"]
    c[H, D] = c[A, D] = costs["to_dead"]
    c[D, H] = c[D, A] = costs["from_dead"]
    codes = np.full(len(prob), X, dtype=np.int8)
    for a, b in blocks:
        codes[a:b] = viterbi(scores[a:b], c)
    return codes


def decode_hybrid(hand_codes, prob, blocks, switch, floor=1e-4, discount=None):
    """Hand model decides D vs live, the learned model decides the team.

    Keeps the hand path's D frames (and X outside the blocks). Inside every maximal live stretch
    of the hand path, a two-state Viterbi over log P(H) and log P(A), renormalised to H + A = 1,
    with `switch` (log-prob units) per change of team. The learned model's own D call is not
    used: DFL's ball is frozen while dead, so it learned that a still ball means a stoppage, which
    does not hold on our match (JOURNAL.md, 2026-10-24). discount: optional (n, 2) per-frame
    reduction of the switch cost into H / into A (decode.viterbi).
    """
    out = hand_codes.copy()
    live = (hand_codes == H) | (hand_codes == A)
    pa = np.clip(prob[:, H], floor, 1.0)
    pb = np.clip(prob[:, A], floor, 1.0)
    total = pa + pb
    scores = np.stack([np.log(pa / total), np.log(pb / total), np.full(len(prob), -1e9)], axis=1)
    c = np.zeros((3, 3))
    c[H, A] = c[A, H] = switch
    c[H, D] = c[A, D] = c[D, H] = c[D, A] = 1e9
    for a, b in blocks:
        edges = np.diff(np.concatenate(([0], live[a:b].astype(np.int8), [0])))
        for s0, e0 in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            out[a + s0:a + e0] = viterbi(scores[a + s0:a + e0], c,
                                         None if discount is None else discount[a + s0:a + e0])
    return out

