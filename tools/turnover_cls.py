"""Turnover classifier for short possessions, trained on SkillCorner + PFF. Aggregate only.

    python tools/turnover_cls.py build [--jobs 4]          # candidates + features per match
    python tools/turnover_cls.py train [--tag t1]           # XGBoost GPU, 4 fixed folds + all
    python tools/turnover_cls.py eval [--tag t1] [--jobs 6] # discount grid on the v31_d1 chain

Candidates: every peak (local max over +/- PEAK_HALF_S) of the current touch network's P(touch by
T) >= CAND_P, where the working base (v31_d1, cached chain + held-out dead head d1) has the other
team in possession just before (BEFORE_S). Target: a true possession of T starts within
+/- LABEL_S of the peak (external truth, which counts short possessions).
Inputs (all oriented to T = candidate team, O = the other): touch P of both teams around the peak,
the previous / next picked touches (team relative to T, time), time to the next T / O pick,
first-stage / refiner / run-vote evidence before and after, every paired feature (d_, c_, m_, g_,
touch_p_) at offsets, ball speed / height / seen, ball direction change and travel, ball position
and travel in T's attacking direction, players of T / O near the ball now and where it goes,
press / speed / shape reaction columns, and how long the base's current O spell has lasted.
Model: XGBoost (cuda), each external match scored by the model of the fold that never saw it.

eval: where P(start) = p, the switch into T costs SWITCH - min(w * clip((p - p0) / (1 - p0)),
SWITCH - 1) for +/- half_s around the peak (learned.decode_hybrid(discount=...)); rest = v31_d1.
Prints spellF1 vs base (SkillCorner, PFF, up / down), nested 4-fold, and possessions caught
(IoU >= 0.5) < 2 s and 2-5 s.
"""

import argparse
import glob
import itertools
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from pipeline.decode import H, A, D, X  # noqa: E402

# Params
FPS = 25
CAND_P = 0.3
PEAK_HALF_S = 0.2
BEFORE_S = 0.2
LABEL_S = 0.5
OFFS_S = (-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0)      # offsets for the per-frame features
CTX_N = 3                                           # previous / next picked touches
NEAR_M = 5.0
FOLDS = 4
DEAD_TAG = "d1"
INTER = os.path.join(ROOT, "data", "cache", "ext_v31")
OUT = os.path.join(ROOT, "data", "cache", "turnover_cls")
ROWS = os.environ.get("TC_ROWS", "")   # --rows v2: rows with the interception inputs (rows_v2_*);
V2 = ROWS == "v2"                      # via the environment so the worker processes see it
GRID = {"w": (0.0,), "p0": (0.2, 0.3, 0.4), "half_s": (0.2,), "b": (8.0,), "max_s": (5.0,),
        "tot": (0.0, 45.0), "end_p": (0.0, 0.3, 0.5)}
CHOSEN = {"w": 0.0, "p0": 0.2, "half_s": 0.2, "b": 8.0, "max_s": 5.0}   # 2026-11-07, nested +0.009
PAIRED = ("d", "c", "m", "g", "touch_p")
SINGLE = ("ball_speed", "ball_seen", "gap_s", "player_speed", "ball_out", "keeper_hold")
REACT = ("press", "speed")


def ext_list():
    import run_v31_external as rx
    out = []
    for name in ("skillcorner", "pff"):
        for p in sorted(glob.glob(os.path.join(INTER, "v31_{}_*.npz".format(name)))):
            g = os.path.basename(p)[:-4].split("_", 2)[2]
            out.append((name, g, rx.fold_of(name, g)))
    return out


def rows_path(name, g):
    return os.path.join(OUT, "rows_{}{}_{}.npz".format(ROWS + "_" if ROWS else "", name, g))


def load_cache(name, g):
    with np.load(os.path.join(INTER, "v31_{}_{}.npz".format(name, g))) as z:
        c = {k: z[k] for k in z.files}
    f = {k[2:]: v for k, v in c.items() if k.startswith("f_")}
    return c, f


def base_chain(c, f, blk, pdead, disc=None, boost=None):
    """v31_d1: hand2 rebuilt with the dead head d1, then v31's decoder (+ discount, + boost: (n,)
    extra home log-odds) and backdating."""
    import eval_dead_v30 as ev
    import eval_touch_run as tr
    import train_ext_heads as th
    from pipeline import fixes, learned
    h2 = th.rebuild_hand2(c["hand"], pdead, f, blk)
    q = np.clip(c["p_ref"], 1e-4, 1 - 1e-4)
    z = tr.ALPHA * np.log(q / (1 - q)) + c["vote"]
    p = fixes.add_vote(c["p1"], z if boost is None else z + boost, 1.0)
    codes = learned.decode_hybrid(h2, p, blk, tr.SWITCH, discount=disc)
    codes = fixes.restart_handover(codes, f, max_s=3.0, slow_ms=2.0)
    return fixes.backdate(codes, f, **ev.BACKDATE)


def peaks(p, thr, half):
    """Frames where p >= thr and p is the max over +/- half frames (first of ties)."""
    n = len(p)
    mx = p.copy()
    for s in range(1, half + 1):
        np.maximum(mx[s:], p[:-s], out=mx[s:])
        np.maximum(mx[:-s], p[s:], out=mx[:-s])
    cand = np.flatnonzero((p >= thr) & (p >= mx))
    keep, last = [], -10 ** 9
    for i in cand:
        if i - last > half:
            keep.append(i)
            last = i
    return np.array(keep, dtype=np.int64)


def true_starts(lab):
    """[(frame, team, length_s, after_change)] of every true possession start."""
    runs = pm.valid_runs(lab)
    sp = pm.possession_spells(lab, runs)
    out = []
    for i, (a, b, k) in enumerate(sp):
        prev = sp[i - 1] if i else None
        change = prev is not None and prev[1] == a and prev[2] != k
        out.append((a, k, (b - a) / FPS, change))
    return out


def attack_sign(d, period):
    """(n,) +1 if home attacks toward x = 1 in that frame's period, else -1 (from keeper sides)."""
    import run_v31_external as rx
    from pipeline import features
    out = np.zeros(len(period))
    for per in np.unique(period):
        s = rx.keeper_side(d, features.HOME, per)       # +1: home goal at x = 1
        out[period == per] = -s if s else 1
    return out


def build_match(job):
    name, g, fold = job
    path = rows_path(name, g)
    if os.path.isfile(path):
        return path
    import dfl_data as dd
    import run_v31_external as rx
    import train_ext_heads as th
    from ext_stage1 import cols_for
    c, f = load_cache(name, g)
    with np.load(os.path.join(ROOT, "data", "cache", "ext_heads_{}.npz".format(DEAD_TAG))) as z:
        pdead = th.seeds_mean({k: z[k] for k in z.files if k.startswith(name + "_" + g + "__")},
                              name + "_" + g)
    blk = rx.blocks(c["period"])
    base = base_chain(c, f, blk, pdead)
    d, pitch, ball = rx.load_set(name, g)
    x, meta, names = match_rows(c, f, base, pdead, d, dd.blur_ball(ball), pitch, cols_for(name, g),
                                c["labels"])
    os.makedirs(OUT, exist_ok=True)
    np.savez(path, x=x, meta=meta, names=np.array(names), fold=fold)
    return path


def _speed(ball, a, b, L, W):
    """Mean ball speed (m/s) over frames [a, b)."""
    n = len(ball)
    a, b = max(a, 1), min(b, n)
    if b - a < 1:
        return np.nan
    d = np.hypot(np.diff(ball[a - 1:b, 0]) * L, np.diff(ball[a - 1:b, 1]) * W)
    return float(np.nanmean(d) * FPS) if np.any(~np.isnan(d)) else np.nan


def intercept_inputs(i, side, ball, players, team, L, W, picks, pick_side, ball_z, features):
    """Inputs for the interception / block pattern of short possessions (v2 rows)."""
    n = len(ball)
    x = {}

    def cl(t):
        return min(max(t, 0), n - 1)

    def xy(t):
        return np.array([ball[cl(t), 0] * L, ball[cl(t), 1] * W])

    pre, post = _speed(ball, i - 10, i - 2, L, W), _speed(ball, i + 2, i + 10, L, W)
    x["spd_pre"], x["spd_post"] = pre, post
    x["spd_ratio"] = post / pre if pre and pre > 0.5 else np.nan
    x["spd_post_long"] = _speed(ball, i + 10, i + 25, L, W)
    for k in (3, 5, 12):
        vi, vo = xy(i) - xy(i - k), xy(i + k) - xy(i)
        nv = np.linalg.norm(vi) * np.linalg.norm(vo)
        x["turn_cos_%d" % k] = float(np.dot(vi, vo) / nv) if nv > 1e-6 else np.nan
    z0 = ball_z[cl(i)]
    x["dz_pre"] = z0 - ball_z[cl(i - 8)]
    x["dz_post"] = ball_z[cl(i + 8)] - z0
    zz = ball_z[cl(i - 25):cl(i) + 1]
    x["z_max_pre"] = float(np.nanmax(zz)) if np.any(~np.isnan(zz)) else np.nan
    # the pass being cut: last picked touch of the other team before the peak
    po = picks[(picks < i - 2) & (pick_side != side)]
    pt = picks[(picks < i - 2) & (pick_side == side)]
    if len(po):
        x["o_rel_dt"] = (i - po[-1]) / FPS
        x["o_rel_dist"] = float(np.hypot(*(xy(i) - xy(po[-1]))))
        x["o_rel_spd"] = _speed(ball, po[-1] + 1, po[-1] + 8, L, W)
    else:
        x["o_rel_dt"] = x["o_rel_dist"] = x["o_rel_spd"] = np.nan
    x["t_last_dt"] = (i - pt[-1]) / FPS if len(pt) else 30.0
    # toucher and duel
    tid_t = features.HOME if side == 0 else features.AWAY
    tid_o = features.AWAY if side == 0 else features.HOME
    b0 = xy(i)
    pos0 = players[cl(i), :, :2] * [L, W]
    posm = players[cl(i - 5), :, :2] * [L, W]
    dist = np.hypot(*(pos0 - b0).T)
    for tag, tid in (("t", tid_t), ("o", tid_o)):
        sel = np.flatnonzero((team == tid) & ~np.isnan(dist))
        if not len(sel):
            for k in ("_tou_spd", "_tou_toward", "_n2", "_n10"):
                x[tag + k] = np.nan
            continue
        x[tag + "_n2"] = float(np.sum(dist[sel] < 2.0))
        x[tag + "_n10"] = float(np.sum(dist[sel] < 10.0))
        j = sel[np.argmin(dist[sel])]
        v = (pos0[j] - posm[j]) * (FPS / 5.0)
        ok = not np.any(np.isnan(v))
        x[tag + "_tou_spd"] = float(np.hypot(*v)) if ok else np.nan
        to_ball = b0 - posm[j]
        nb = np.linalg.norm(to_ball)
        x[tag + "_tou_toward"] = float(np.dot(v, to_ball) / nb) if ok and nb > 1e-6 else np.nan
    # where the ball next stops (slowest point in the next 2 s) and who is near it
    seg = [(_speed(ball, t - 2, t + 3, L, W), t) for t in range(i + 5, min(i + 50, n - 3), 3)]
    seg = [s_ for s_ in seg if not np.isnan(s_[0])]
    if seg:
        sp_, ts = min(seg)
        bs = xy(ts)
        x["stop_dt"], x["stop_spd"] = (ts - i) / FPS, sp_
        ds = np.hypot(*(players[ts, :, :2] * [L, W] - bs).T)
        for tag, tid in (("t", tid_t), ("o", tid_o)):
            dd_ = ds[(team == tid) & ~np.isnan(ds)]
            x["stop_min_" + tag] = float(dd_.min()) if len(dd_) else np.nan
    else:
        x["stop_dt"] = x["stop_spd"] = x["stop_min_t"] = x["stop_min_o"] = np.nan
    return x


def match_ctx(c, f, base, pdead, d, ball, pitch, cols):
    """Per-match arrays the per-touch inputs read."""
    from pipeline import features
    n = len(base)
    chg = np.concatenate(([0], np.flatnonzero(np.diff(base) != 0) + 1))
    spell_start = np.zeros(n, dtype=np.int64)
    spell_start[chg] = chg
    picks = np.flatnonzero(f["touch"] != 0)
    return {"c": c, "f": f, "base": base, "pdead": pdead, "ball": ball, "L": pitch[0],
            "W": pitch[1], "cols": cols, "players": d["players"], "team": d["team"], "n": n,
            "sign_home": attack_sign(d, d["period"]), "features": features,
            "q": np.clip(c["p_ref"], 1e-4, 1 - 1e-4),
            "ph1": c["p1"][:, H] / np.maximum(c["p1"][:, H] + c["p1"][:, A], 1e-6),
            "tp": np.stack([f["touch_p_home"], f["touch_p_away"]], axis=1).astype(np.float64),
            "picks": picks, "pick_side": np.where(f["touch"][picks] > 0, 0, 1),
            "half": int(round(PEAK_HALF_S * FPS)), "offs": [int(round(o * FPS)) for o in OFFS_S],
            "spell_start": np.maximum.accumulate(spell_start)}


def touch_feats(ctx, i, side, j):
    """Inputs of a touch of side (0 home, 1 away) at frame i, oriented to that team; j: the frame
    'just before' (for how long the base's current spell has lasted)."""
    c, f, ball, cols, pdead = ctx["c"], ctx["f"], ctx["ball"], ctx["cols"], ctx["pdead"]
    L, W, n, players, team = ctx["L"], ctx["W"], ctx["n"], ctx["players"], ctx["team"]
    sign_home, features, q, ph1, tp = (ctx["sign_home"], ctx["features"], ctx["q"], ctx["ph1"],
                                       ctx["tp"])
    picks, pick_side, half, offs = ctx["picks"], ctx["pick_side"], ctx["half"], ctx["offs"]
    spell_start = ctx["spell_start"]
    x = {}
    sg = sign_home[i] if side == 0 else -sign_home[i]    # T attacks toward +sg in x
    x["peak_p"] = tp[i, side]
    x["o_p_at"] = tp[i, 1 - side]
    for w_ in (0.5, 1.0):
        k = int(w_ * FPS)
        x["o_pmax_prev%g" % w_] = tp[max(i - k, 0):i, 1 - side].max(initial=0)
        x["o_pmax_next%g" % w_] = tp[i + 1:i + 1 + k, 1 - side].max(initial=0)
        x["t_pmax_next%g" % w_] = tp[i + half + 1:i + half + 1 + k, side].max(initial=0)
    # picked touches around
    prv = np.flatnonzero(picks < i - 2)[-CTX_N:][::-1]
    nxt = np.flatnonzero(picks > i + 2)[:CTX_N]
    for r in range(CTX_N):
        for tag, idx in (("prev", prv), ("next", nxt)):
            if r < len(idx):
                x["%s%d_t" % (tag, r)] = float(pick_side[idx[r]] == side)
                x["%s%d_dt" % (tag, r)] = abs(picks[idx[r]] - i) / FPS
            else:
                x["%s%d_t" % (tag, r)] = np.nan
                x["%s%d_dt" % (tag, r)] = np.nan
    for tag, s_ in (("t", side), ("o", 1 - side)):
        after = picks[(picks > i + 2) & (pick_side == s_)]
        x["next_%s_dt" % tag] = (after[0] - i) / FPS if len(after) else 30.0
    # evidence around (oriented)
    for o, k in zip(OFFS_S, offs):
        t = min(max(i + k, 0), n - 1)
        pt = ph1[t] if side == 0 else 1 - ph1[t]
        x["p1_%g" % o] = pt
        lq = np.log(q[t] / (1 - q[t]))
        x["ref_%g" % o] = lq if side == 0 else -lq
        x["vote_%g" % o] = c["vote"][t] if side == 0 else -c["vote"][t]
        x["hand2D_%g" % o] = float(c["hand"][t] == D)
        for p_ in PAIRED:
            a_, b_ = f[p_ + "_home"][t], f[p_ + "_away"][t]
            x["%s_t_%g" % (p_, o)] = a_ if side == 0 else b_
            x["%s_o_%g" % (p_, o)] = b_ if side == 0 else a_
        for s_ in SINGLE:
            x["%s_%g" % (s_, o)] = f[s_][t]
        x["ball_z_%g" % o] = c["ball_z"][t]
        for r_ in REACT:
            a_, b_ = cols[r_ + "_home"][t], cols[r_ + "_away"][t]
            x["%s_t_%g" % (r_, o)] = a_ if side == 0 else b_
            x["%s_o_%g" % (r_, o)] = b_ if side == 0 else a_
    for k_ in ("r_push", "r_cx", "r_back", "r_compact", "r_spread"):
        for dd_ in ("d2", "d4", "d6"):
            a_ = cols["%s_home_%s" % (k_, dd_)][i]
            b_ = cols["%s_away_%s" % (k_, dd_)][i]
            x["%s_t_%s" % (k_, dd_)] = a_ if side == 0 else b_
            x["%s_o_%s" % (k_, dd_)] = b_ if side == 0 else a_
    # ball geometry
    def bxy(t):
        t = min(max(t, 0), n - 1)
        return ball[t, 0] * L, ball[t, 1] * W
    b0 = np.array(bxy(i))
    x["ball_x_att"] = (b0[0] - L / 2) * sg
    x["ball_y_abs"] = abs(b0[1] - W / 2)
    v_in = b0 - np.array(bxy(i - 8))
    for w_ in (0.3, 0.6, 1.0, 2.0):
        bk = np.array(bxy(i + int(w_ * FPS)))
        x["travel_%g" % w_] = float(np.hypot(*(bk - b0)))
        x["adv_%g" % w_] = float((bk[0] - b0[0]) * sg)
    v_out = np.array(bxy(i + 8)) - b0
    nv = np.linalg.norm(v_in) * np.linalg.norm(v_out)
    x["turn_cos"] = float(np.dot(v_in, v_out) / nv) if nv > 1e-6 else np.nan
    # players near the ball now / where it goes
    for o, k in (("0", 0), ("0.5", int(0.5 * FPS)), ("1", FPS), ("2", 2 * FPS)):
        t = min(i + k, n - 1)
        bt = np.array(bxy(t))
        dist = np.hypot(players[t, :, 0] * L - bt[0], players[t, :, 1] * W - bt[1])
        for tag, tid in (("t", features.HOME if side == 0 else features.AWAY),
                         ("o", features.AWAY if side == 0 else features.HOME)):
            dt_ = dist[team == tid]
            dt_ = dt_[~np.isnan(dt_)]
            x["near_%s_%s" % (tag, o)] = float(np.sum(dt_ < NEAR_M))
            x["min_%s_%s" % (tag, o)] = float(dt_.min()) if len(dt_) else np.nan
    x["o_spell_s"] = (i - spell_start[j]) / FPS
    if V2:
        x.update(intercept_inputs(i, side, ball, players, team, L, W, picks, pick_side,
                                  c["ball_z"], features))
    x["dead_p"] = pdead[i]
    return x


def match_rows(c, f, base, pdead, d, ball, pitch, cols, lab=None):
    """(x, meta, names) of the candidates of one match. d: players, team, period (keeper sides);
    ball: blurred, normalised. lab None: no truth (meta label 0)."""
    n = len(base)
    if lab is None:
        lab = np.full(n, X, dtype=np.int8)
    ctx = match_ctx(c, f, base, pdead, d, ball, pitch, cols)
    tp = ctx["tp"]
    starts = true_starts(lab)
    st_f = np.array([s[0] for s in starts])
    st_k = np.array([s[1] for s in starts])
    bef = int(round(BEFORE_S * FPS))
    lab_k = int(round(LABEL_S * FPS))
    rows, meta = [], []
    names = None
    for side in (0, 1):
        T, O = (H, A) if side == 0 else (A, H)
        for i in peaks(tp[:, side], CAND_P, ctx["half"]):
            j = max(i - bef, 0)
            if base[j] != O:
                continue
            x = touch_feats(ctx, i, side, j)
            if names is None:
                names = list(x)
            rows.append([x[k] for k in names])
            near = np.abs(st_f - i) <= lab_k
            hit = np.flatnonzero(near & (st_k == T))
            if len(hit):
                s = starts[hit[np.argmin(np.abs(st_f[hit] - i))]]
                meta.append((i, T, 1, s[2], float(s[3])))
            else:
                meta.append((i, T, 0, np.nan, np.nan))
    return (np.array(rows, dtype=np.float32).reshape(len(rows), -1),
            np.array(meta, dtype=np.float64).reshape(len(meta), 5), names or [])


def build_cmd(args):
    jobs = ext_list()
    with Pool(args.jobs) as pool:
        paths = pool.map(build_match, jobs, chunksize=1)
    tot, pos = 0, 0
    starts = {"short": [0, 0], "mid": [0, 0]}
    for (name, g, _), p in zip(jobs, paths):
        with np.load(p) as z:
            m = z["meta"]
        tot += len(m)
        pos += int(m[:, 2].sum()) if len(m) else 0
    print("candidates {} , positive {} ({:.1%})".format(tot, pos, pos / max(tot, 1)))
    # coverage of true starts after a team change
    for name in ("skillcorner", "pff"):
        cov = {"<2": [0, 0], "2-5": [0, 0], ">=5": [0, 0]}
        for (nm, g, _), p in zip(jobs, paths):
            if nm != name:
                continue
            c, _ = load_cache(nm, g)
            with np.load(p) as z:
                m = z["meta"]
            hit = set()
            for row in m:
                if row[2] == 1:
                    hit.add((int(row[1]), round(row[3], 3)))
            for a, k, ln, chg in true_starts(c["labels"]):
                if not chg:
                    continue
                key = "<2" if ln < 2 else "2-5" if ln < 5 else ">=5"
                cov[key][0] += 1
            cov_hits = {"<2": 0, "2-5": 0, ">=5": 0}
            for row in m:
                if row[2] == 1 and row[4] == 1:
                    ln = row[3]
                    cov_hits["<2" if ln < 2 else "2-5" if ln < 5 else ">=5"] += 1
            for k in cov:
                cov[k][1] += cov_hits[k]
        print("{:<12} candidates at true starts after a team change (base had the other team): ".format(
            name) + "  ".join("{} {}/{} ({:.0%})".format(k, v[1], v[0], v[1] / max(v[0], 1))
                              for k, v in cov.items()))


def load_rows():
    data = []
    for name, g, fold in ext_list():
        with np.load(rows_path(name, g)) as z:
            if len(z["meta"]):
                data.append((name, g, fold, z["x"], z["meta"], list(z["names"])))
    return data


def roc_auc_score(y, p):
    r = np.argsort(np.argsort(p)) + 1.0
    npos = y.sum()
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos))


def average_precision_score(y, p):
    o = np.argsort(-p)
    ys = y[o]
    prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    return float(np.sum(prec * ys) / max(ys.sum(), 1))


def train_xgb(x, y, seed=0, rounds=600, w=None):
    import xgboost as xgb
    dm = xgb.DMatrix(x, label=y, weight=w)
    params = {"objective": "binary:logistic", "eval_metric": "logloss", "device": "cuda",
              "tree_method": "hist", "max_depth": 6, "eta": 0.05, "subsample": 0.8,
              "colsample_bytree": 0.7, "min_child_weight": 5, "seed": seed}
    return xgb.train(params, dm, rounds)


def weights(meta, sw):
    """Positives < 2 s weight sw, 2-5 s (1 + sw) / 2, the rest 1."""
    if sw == 1:
        return None
    ln = meta[:, 3]
    pos = meta[:, 2] == 1
    return np.where(pos & (ln < 2), sw, np.where(pos & (ln < 5), (1 + sw) / 2.0, 1.0))


def auc_vs_neg(yy, pp, sel):
    neg = yy == 0
    return roc_auc_score(np.r_[np.ones(sel.sum()), np.zeros(neg.sum())], np.r_[pp[sel], pp[neg]])


def train_cmd(args):
    import xgboost as xgb
    data = load_rows()
    out = {}
    for k in range(FOLDS):
        tr_ = [d for d in data if d[2] != k]
        x = np.concatenate([d[3] for d in tr_])
        y = np.concatenate([d[4][:, 2] for d in tr_])
        bst = train_xgb(x, y, w=weights(np.concatenate([d[4] for d in tr_]), args.short_weight))
        for d in data:
            if d[2] == k:
                out["{}_{}".format(d[0], d[1])] = bst.predict(xgb.DMatrix(d[3]))
        print("fold {} trained on {} rows ({:.1%} positive)".format(k, len(y), y.mean()), flush=True)
    x = np.concatenate([d[3] for d in data])
    y = np.concatenate([d[4][:, 2] for d in data])
    os.makedirs(os.path.join(OUT, "models"), exist_ok=True)
    bst = train_xgb(x, y, w=weights(np.concatenate([d[4] for d in data]), args.short_weight))
    bst.save_model(os.path.join(OUT, "models", "{}_all.json".format(args.tag)))
    np.savez(os.path.join(OUT, "pred_{}.npz".format(args.tag)), **out)
    # held-out quality
    yy = np.concatenate([d[4][:, 2] for d in data])
    pp = np.concatenate([out["{}_{}".format(d[0], d[1])] for d in data])
    ln = np.concatenate([d[4][:, 3] for d in data])
    print("held-out AUC {:.3f}, AP {:.3f} (base rate {:.3f}); peak P alone AUC {:.3f}".format(
        roc_auc_score(yy, pp), average_precision_score(yy, pp), yy.mean(),
        roc_auc_score(yy, np.concatenate([d[3][:, 0] for d in data]))))
    for thr in (0.3, 0.5, 0.7):
        sel = pp >= thr
        print("  p >= {:.1f}: precision {:.2f}, recall {:.2f} (short < 2 s starts {:.2f}, 2-5 s {:.2f})"
              .format(thr, yy[sel].mean() if sel.any() else np.nan, sel[yy == 1].mean(),
                      sel[(yy == 1) & (ln < 2)].mean(), sel[(yy == 1) & (ln >= 2) & (ln < 5)].mean()))
    pos = yy == 1
    print("  AUC vs negatives: short < 2 s {:.3f}, 2-5 s {:.3f}, >= 5 s {:.3f}".format(
        auc_vs_neg(yy, pp, pos & (ln < 2)), auc_vs_neg(yy, pp, pos & (ln >= 2) & (ln < 5)),
        auc_vs_neg(yy, pp, pos & (ln >= 5))))
    imp = bst.get_score(importance_type="gain")
    names = data[0][5]
    top = sorted(imp.items(), key=lambda kv: -kv[1])[:15]
    print("top inputs (gain): " + ", ".join(names[int(k[1:])] for k, _ in top))


def discount(n, meta, p, w, p0, half_s, switch):
    out = np.zeros((n, 2))
    k = int(round(half_s * FPS))
    v = np.minimum(w * np.clip((p - p0) / (1.0 - p0), 0, 1), switch - 1.0)
    for (i, T), vi in zip(meta[:, :2].astype(int), v):
        if vi <= 0:
            continue
        col = 0 if T == H else 1
        a, b = max(i - k, 0), min(i + k + 1, n)
        out[a:b, col] = np.maximum(out[a:b, col], vi)
    return out


def span_boost(n, meta, p, f, b, p0, max_s, tot=0.0, end_p=0.0):
    """(n,) home log-odds: +/- b x clip((p - p0) / (1 - p0)) toward T from the peak to the next
    picked touch of the other team (at most max_s). tot > 0: every proposal (p >= p0) gets at
    least tot in total over its span (enough to pay both switches). end_p > 0: the span also ends
    at the other team's first touch-P peak >= end_p."""
    out = np.zeros(n)
    tch = np.asarray(f["touch"])
    picks = np.flatnonzero(tch != 0)
    side = np.where(tch[picks] > 0, H, A)
    tp = (np.asarray(f["touch_p_home"]), np.asarray(f["touch_p_away"]))
    v = b * np.clip((p - p0) / (1.0 - p0), 0, 1)
    k = int(round(max_s * FPS))
    for (i, T), vi, pi in zip(meta[:, :2].astype(int), v, p):
        if pi < p0 or (vi <= 0 and tot <= 0):
            continue
        nxt = picks[(picks > i + 2) & (side != T)]
        e = min(int(nxt[0]) if len(nxt) else n, i + k, n)
        if end_p > 0:
            po = tp[1 if T == H else 0]
            hit = np.flatnonzero(po[i + 3:e] >= end_p)
            if len(hit):
                j = i + 3 + int(hit[0])
                while j + 1 < e and po[j + 1] >= po[j]:
                    j += 1
                e = j
        if e <= i:
            continue
        if tot > 0:
            vi = max(vi, tot / float(e - i))
        sgn = 1.0 if T == H else -1.0
        seg = out[i:e]
        out[i:e] = np.where(np.abs(seg) < vi, sgn * vi, seg)
    return out


def eval_match(job):
    name, g, tag, settings = job
    import eval_touch_run as tr
    import run_v31_external as rx
    import train_ext_heads as th
    from ext_stage1 import length_caught
    c, f = load_cache(name, g)
    with np.load(os.path.join(ROOT, "data", "cache", "ext_heads_{}.npz".format(DEAD_TAG))) as z:
        pdead = th.seeds_mean({k: z[k] for k in z.files if k.startswith(name + "_" + g + "__")},
                              name + "_" + g)
    with np.load(rows_path(name, g)) as z:
        meta = z["meta"]
    with np.load(os.path.join(OUT, "pred_{}.npz".format(tag))) as z:
        key = "{}_{}".format(name, g)
        p = z[key] if key in z.files else np.zeros(0)
    blk = rx.blocks(c["period"])
    lab = c["labels"]
    n = len(lab)
    base = base_chain(c, f, blk, pdead)
    res = [(tr.score(lab, base), length_caught(lab, base))]
    for s in settings:
        disc = discount(n, meta, p, s["w"], s["p0"], s["half_s"], tr.SWITCH)             if len(p) and s["w"] > 0 else None
        bst = span_boost(n, meta, p, f, s["b"], s["p0"], s["max_s"], s.get("tot", 0.0),
                         s.get("end_p", 0.0)) if len(p) else None
        codes = base_chain(c, f, blk, pdead, disc, bst)
        res.append((tr.score(lab, codes), length_caught(lab, codes)))
    return name, g, res


def eval_cmd(args):
    import run_v31_external as rx
    settings = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    jobs = [(n, g, args.tag, settings) for n, g, _ in ext_list()]
    with Pool(args.jobs) as pool:
        res = pool.map(eval_match, jobs, chunksize=1)
    sc = np.array([[r[0] for r in x[2]] for x in res])           # (matches, 1 + settings)
    lc = np.array([[np.array(r[1]) for r in x[2]] for x in res])  # (m, 1+s, 2 classes, 2)
    names = np.array([x[0] for x in res])
    folds = np.array([rx.fold_of(x[0], x[1]) for x in res])
    np.savez(os.path.join(OUT, "eval_{}.npz".format(args.tag)), sc=sc, lc=lc, names=names,
             folds=folds)
    gain = sc[:, 1:] - sc[:, :1]
    order = np.argsort(-gain.mean(axis=0))

    def caught_line(col):
        t = lc[:, col].sum(axis=0)
        return "< 2 s {:.1%}, 2-5 s {:.1%}".format(t[0, 1] / max(t[0, 0], 1), t[1, 1] / max(t[1, 0], 1))

    print("base v31_d1: SkillCorner {:.4f}, PFF {:.4f} | caught {}".format(
        sc[names == "skillcorner", 0].mean(), sc[names == "pff", 0].mean(), caught_line(0)))
    for j in order[:8]:
        s = settings[j]
        parts = []
        for nm in ("skillcorner", "pff"):
            gg = gain[names == nm, j]
            parts.append("{} {:+.4f} ({}/{})".format(nm[:5], gg.mean(), int(np.sum(gg > 1e-9)),
                                                     int(np.sum(gg < -1e-9))))
        print("{}: {} | all {:+.4f} | caught {}".format(
            " ".join("{} {:g}".format(k, v) for k, v in s.items()), "  ".join(parts), gain[:, j].mean(),
            caught_line(j + 1)))
    nested = np.zeros(len(sc))
    for k in range(FOLDS):
        j = int(np.argmax(gain[folds != k].mean(axis=0)))
        nested[folds == k] = gain[folds == k, j]
    print("nested 4-fold: {:+.4f} ({}/{})".format(nested.mean(), int(np.sum(nested > 1e-9)),
                                                 int(np.sum(nested < -1e-9))))


def ours_cmd(args):
    """Our match: v31_d1 (MTAH_VARIANT=_kt for the keeper-corrected tracking) + the proposals."""
    import xgboost as xgb
    import dfl_data as dd
    import ours_chain as oc
    import in_domain_team as idt
    from atd_cache import load_arrays
    from pipeline.features import LENGTH, WIDTH
    from pipeline.run import RESULTS_DIR
    import eval_touch_run as tr
    codes, ex = oc.ours_v31(dead_prefix=os.path.join("models", "dead_ext_d1"))
    variant = os.environ.get("MTAH_VARIANT", "")
    ref_path = os.path.join(RESULTS_DIR, "v31_d1{}.csv".format(variant))
    if os.path.isfile(ref_path):
        ref = pm.label_codes(pm.read_label_csv(ref_path))
        print("rebuilt base vs {}: {:.3f}% frames differ".format(os.path.basename(ref_path),
                                                                 100 * np.mean(codes != ref)))
    f = {k[2:]: v for k, v in ex.items() if k.startswith("f_")}
    blk = [tuple(b) for b in ex["blk"]]
    n = len(codes)
    period = np.zeros(n, dtype=np.int8)
    for i, (a, b) in enumerate(blk, start=1):
        period[a:b] = i
    c = dict(ex, period=period)
    base = base_chain(c, f, blk, ex["pdead"])
    print("base_chain equals ours_v31: {}".format(np.array_equal(base, codes)))
    atd = load_arrays()
    cols = idt.player_columns(atd["players"], atd["team"], period, (LENGTH, WIDTH))
    d = {"players": atd["players"], "team": atd["team"], "period": period}
    x, meta, names = match_rows(c, f, base, ex["pdead"], d, dd.blur_ball(atd["ball"]),
                                (LENGTH, WIDTH), cols)
    bst = xgb.Booster()
    bst.load_model(os.path.join(OUT, "models", "{}_all.json".format(args.tag)))
    p = bst.predict(xgb.DMatrix(x))
    s = CHOSEN
    disc = discount(n, meta, p, s["w"], s["p0"], s["half_s"], tr.SWITCH) if s["w"] > 0 else None
    new = base_chain(c, f, blk, ex["pdead"], disc, span_boost(n, meta, p, f, s["b"], s["p0"],
                                                              s["max_s"]))
    print("candidates {}, p >= p0 {}; frames changed {:.2f}%".format(
        len(p), int(np.sum(p >= s["p0"])), 100 * np.mean(new != base)))
    for nm, cd in (("base", base), ("new", new)):
        sp = pm.possession_spells(cd, pm.valid_runs(cd))
        print("{}: {} possessions, D {:.1%}".format(nm, len(sp), np.mean(cd[cd != X] == D)))
    if args.name:
        print(oc.write_and_score(new, args.name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "train", "eval", "ours"))
    ap.add_argument("--name")
    ap.add_argument("--rows", default="", help="v2: rows with the interception inputs")
    ap.add_argument("--short-weight", type=float, default=1.0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--tag", default="t1")
    args = ap.parse_args()
    global ROWS, V2
    ROWS, V2 = args.rows, args.rows == "v2"
    os.environ["TC_ROWS"] = ROWS
    {"build": build_cmd, "train": train_cmd, "eval": eval_cmd, "ours": ours_cmd}[args.cmd](args)


if __name__ == "__main__":
    main()
