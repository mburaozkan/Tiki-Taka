"""Turnover evidence signals, measured before they go into the change net. Aggregate only.

    python tools/turnover_signals.py prep [--jobs 6]   # per-match signals, data/cache/turnover_sig/
    python tools/turnover_signals.py auc [--tag c1dl]

Signals per frame t, signed toward "turnover to H" (the turnover-to-A version is the negative), each
as the change from the past window [t - P - 0.4 s, t - 0.4 s] to the future window
[t + 0.4 s, t + P + 0.4 s], P = 1 / 2 / 4 s; all from the players only unless named "ball":
  flow    team mean running speed toward the opponent goal: H up and A down after a turnover to H;
  press   players within 20 m of the ball running at it (> 2 m/s): the new defenders (A) press;
  spread  team spread (std x + std y): new attackers spread, new defenders compact;
  depth   team centroid x toward the opponent goal (H pushes up, A drops);
  ball    ball displacement toward A's goal (seen frames only; 0 where unknown).
auc: live frames (every 5th), positives = within 2 frames of a true turnover to H (or to A, with the
signal negated). Per provider (SkillCorner / PFF / DFL): AUC of each signal alone; the change net's
own score (max of P(turnover to that team) within +-0.5 s, held-out c1dl); and a logistic model of
net + signals, fitted on the other two providers, so the gain carries over between providers.
Also the AUC among the hard cases (net score < 0.3).
"""

import argparse
import os
import sys
from multiprocessing import Pool

os.environ.setdefault("TC_ROWS", "v2")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from pipeline.decode import H, A, D  # noqa: E402

FPS = 25
OUT = os.path.join(ROOT, "data", "cache", "turnover_sig")
PASTS = (1, 2, 4)
KINDS = ("flow", "press", "spread", "depth", "ball")
GAP = int(0.4 * FPS)


def sig_names():
    return ["{}{}".format(k, p) for k in KINDS for p in PASTS]


def team_series(players, team, ball, pitch, sign_home):
    """per-team (n,) series: flow, press, spread, depth (home, away), ball x toward A's goal."""
    from pipeline import features
    L, W = pitch
    px = (players[:, :, 0] - 0.5) * L
    py = (players[:, :, 1] - 0.5) * W
    pvx = np.full_like(px, np.nan)
    pvy = np.full_like(py, np.nan)
    pvx[5:], pvy[5:] = (px[5:] - px[:-5]) * FPS / 5.0, (py[5:] - py[:-5]) * FPS / 5.0
    bx = (ball[:, 0] - 0.5) * L
    by = (ball[:, 1] - 0.5) * W
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for tag, tid, sgn in (("h", features.HOME, 1.0), ("a", features.AWAY, -1.0)):
            sel = team == tid
            s = (sign_home * sgn)[:, None]
            vx, x, y = pvx[:, sel] * s, px[:, sel] * s, py[:, sel]
            out["flow_" + tag] = np.nanmean(np.clip(vx, -12, 12), axis=1)
            dx, dy = bx[:, None] - px[:, sel], by[:, None] - py[:, sel]
            dist = np.hypot(dx, dy)
            tow = (pvx[:, sel] * dx + pvy[:, sel] * dy) / np.maximum(dist, 0.5)
            out["press_" + tag] = np.nansum((dist < 20) & (tow > 2), axis=1).astype(np.float32)
            out["spread_" + tag] = np.nanstd(x, axis=1) + np.nanstd(y, axis=1)
            out["depth_" + tag] = np.nanmean(x, axis=1)
    out["ball"] = -bx * sign_home          # + toward A's goal (home attacks +x when sign_home = 1)
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def window_mean(v, lo, hi):
    """mean of v over [t + lo, t + hi) for every t, NaN-aware (NaN where the window is empty)."""
    n = len(v)
    ok = np.isfinite(v)
    cs = np.concatenate([[0.0], np.cumsum(np.where(ok, v, 0.0))])
    cn = np.concatenate([[0], np.cumsum(ok)])
    t = np.arange(n)
    a, b = np.clip(t + lo, 0, n), np.clip(t + hi, 0, n)
    cnt = cn[b] - cn[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (cs[b] - cs[a]) / np.maximum(cnt, 1), np.nan)


def signals(players, team, ball, pitch, sign_home):
    s = team_series(players, team, ball, pitch, sign_home)
    out = []
    for kind in KINDS:
        for P in PASTS:
            w = P * FPS

            def delta(v):
                return window_mean(v, GAP, GAP + w) - window_mean(v, -GAP - w, -GAP)
            if kind == "ball":
                x = delta(s["ball"])
            elif kind == "press":
                x = delta(s["press_a"]) - delta(s["press_h"])
            else:
                x = delta(s[kind + "_h"]) - delta(s[kind + "_a"])
            out.append(np.nan_to_num(x).astype(np.float32))
    return np.stack(out, axis=1)


def prep_one(job):
    name, g = job
    path = os.path.join(OUT, "{}_{}.npy".format(name, g))
    if os.path.isfile(path):
        return path
    import dfl_data as dd
    import turnover_cls as T
    if name == "dfl":
        import cn_dfl
        from train_learned_dfl import test_seed
        d = dd.load(g)
        with np.load(cn_dfl.path(g)) as z:
            blk = [tuple(b) for b in z["blk"]]
            n = len(z["lab"])
        period = np.zeros(n, np.int8)
        for i, (a, b) in enumerate(blk, start=1):
            period[a:b] = i
        ball = dd.prepare_ball(d, test_seed(g))
        dd_ = {"players": d["players"], "team": d["team"], "period": period}
        pitch = dd.PITCH
    else:
        import run_v31_external as rx
        c, _ = T.load_cache(name, g)
        d, pitch, ball = rx.load_set(name, g)
        ball = dd.blur_ball(ball)
        dd_ = d
        period = c["period"]
    sign = T.attack_sign(dd_, period)
    x = signals(dd_["players"], dd_["team"], ball, pitch, sign)
    os.makedirs(OUT, exist_ok=True)
    np.save(path, x.astype(np.float16))
    return path


def match_list():
    import turnover_cls as T
    import dfl_data as dd
    return [(n, g) for n, g, _ in T.ext_list()] + [("dfl", g) for g in dd.GAMES]


def prep_cmd(args):
    jobs = match_list()
    with Pool(args.jobs) as pool:
        for i, p in enumerate(pool.imap_unordered(prep_one, jobs), 1):
            if i % 10 == 0 or i == len(jobs):
                print("  {} / {}".format(i, len(jobs)), flush=True)


def load_cls(name, g):
    import change_net as C
    return C.load_cls(name, g)


def auc(y, s):
    """rank AUC (ties averaged)."""
    from scipy.stats import rankdata
    y = np.asarray(y, bool)
    npos, nneg = y.sum(), (~y).sum()
    if npos == 0 or nneg == 0:
        return np.nan
    r = rankdata(s)
    return (r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def samples(name, g, pred):
    """rows for both directions: (y, net score, signals signed toward that team)."""
    cls = load_cls(name, g)
    x = np.load(os.path.join(OUT, "{}_{}.npy".format(name, g))).astype(np.float32)
    n = len(cls)
    assert len(x) == n == len(pred), (name, g)
    pr = pred.astype(np.float32)
    rows = []
    for col, (src, dst), sg in ((0, (A, H), 1.0), (1, (H, A), -1.0)):
        t = np.flatnonzero((cls[:-1] == src) & (cls[1:] == dst)) + 1
        pos = np.zeros(n, bool)
        for k in range(-2, 3):
            pos[np.clip(t + k, 0, n - 1)] = True
        from scipy.ndimage import maximum_filter1d
        net = maximum_filter1d(pr[:, col], size=25)
        live = (cls == H) | (cls == A)
        live[t] = True
        idx = np.flatnonzero(live)
        idx = np.union1d(idx[::5], np.flatnonzero(pos))
        rows.append(np.column_stack([pos[idx], net[idx], sg * x[idx]]))
    return np.concatenate(rows)


def logit_fit(X, y, l2=1.0, iters=25):
    """L2 logistic regression by Newton steps; returns [bias, w...]."""
    Xb = np.column_stack([np.ones(len(X)), X])
    y = y.astype(np.float64)
    w = np.zeros(Xb.shape[1])
    reg = np.eye(len(w)) * l2
    reg[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))
        g = Xb.T @ (p - y) + reg @ w
        Hs = (Xb * (p * (1 - p))[:, None]).T @ Xb + reg
        w -= np.linalg.solve(Hs, g)
    return w


def auc_cmd(args):
    with np.load(os.path.join(ROOT, "data", "cache", "change_net", "pred_{}.npz".format(args.tag))) as z:
        preds = {k: z[k] for k in z.files}
    prov = {"skillcorner": "SkillCorner", "pff": "PFF", "dfl": "DFL"}
    data = {}
    for name, g in match_list():
        key = "{}_{}".format(name, g)
        if key not in preds:
            continue
        data.setdefault(name, []).append(samples(name, g, preds[key]))
    data = {k: np.concatenate(v) for k, v in data.items()}
    names = sig_names()
    print("held-out net {}; rows (live frames / 5 + positives): ".format(args.tag) + ", ".join(
        "{} {} ({:.2%} pos)".format(prov[k], len(v), v[:, 0].mean()) for k, v in data.items()))
    print("\nAUC alone (turnover to that team within 2 frames vs other live frames)")
    print("{:<10} ".format("signal") + " ".join("{:>11}".format(prov[k]) for k in data))
    print("{:<10} ".format("net") + " ".join("{:>11.3f}".format(auc(v[:, 0], v[:, 1])) for v in data.values()))
    for j, nm in enumerate(names):
        print("{:<10} ".format(nm) + " ".join("{:>11.3f}".format(auc(v[:, 0], v[:, 2 + j]))
                                              for v in data.values()))
    print("\nhard cases (net score < 0.3): share of positives, AUC of each signal alone")
    print("{:<10} ".format("") + " ".join("{:>11}".format(prov[k]) for k in data))
    print("{:<10} ".format("pos share") + " ".join(
        "{:>10.0%} ".format(np.mean(v[v[:, 0] > 0, 1] < 0.3)) for v in data.values()))
    for j, nm in enumerate(names):
        print("{:<10} ".format(nm) + " ".join(
            "{:>11.3f}".format(auc(v[v[:, 1] < 0.3, 0], v[v[:, 1] < 0.3, 2 + j])) for v in data.values()))

    def feats(v, cols):
        lg = np.log(np.clip(v[:, 1:2], 1e-4, 1 - 1e-4) / (1 - np.clip(v[:, 1:2], 1e-4, 1 - 1e-4)))
        return np.column_stack([lg] + [v[:, 2 + c] for c in cols])
    groups = [("net + flow", [names.index(n) for n in names if n.startswith("flow")]),
              ("net + press", [names.index(n) for n in names if n.startswith("press")]),
              ("net + spread", [names.index(n) for n in names if n.startswith("spread")]),
              ("net + depth", [names.index(n) for n in names if n.startswith("depth")]),
              ("net + ball", [names.index(n) for n in names if n.startswith("ball")]),
              ("net + players only", [j for j, n in enumerate(names) if not n.startswith("ball")]),
              ("net + all", list(range(len(names))))]
    print("\nlogistic, fitted on the other two providers: AUC all / hard cases (net < 0.3)")
    print("{:<20} ".format("model") + " ".join("{:>15}".format(prov[k]) for k in data))
    line = "{:<20} ".format("net alone")
    for k, v in data.items():
        h = v[:, 1] < 0.3
        line += "{:>7.3f} / {:<5.3f} ".format(auc(v[:, 0], v[:, 1]), auc(v[h, 0], v[h, 1]))
    print(line)
    for nm, cols in groups:
        line = "{:<20} ".format(nm)
        for k, v in data.items():
            tr = np.concatenate([w for k2, w in data.items() if k2 != k])
            Xt = feats(tr, cols)
            mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
            wts = logit_fit((Xt - mu) / sd, tr[:, 0] > 0)
            s = ((feats(v, cols) - mu) / sd) @ wts[1:] + wts[0]
            h = v[:, 1] < 0.3
            line += "{:>7.3f} / {:<5.3f} ".format(auc(v[:, 0], s), auc(v[h, 0], s[h]))
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("prep", "auc"))
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--tag", default="c1dl")
    args = ap.parse_args()
    {"prep": prep_cmd, "auc": auc_cmd}[args.cmd](args)


if __name__ == "__main__":
    main()
