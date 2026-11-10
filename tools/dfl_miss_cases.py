"""Case studies of true possessions the submission misses, on DFL open data only (CC BY 4.0).
Never run on our match: it loads only dfl_data.GAMES.

    python tools/dfl_miss_cases.py [--tag c1dl --alpha 0.52] [--kind D|other|merged|shifted] [--n 8] [--seed 0]

Outcome per true H / A possession as tools/dfl_loss.py (caught / called D / other team / merged /
shifted). For a random sample of one outcome: a timeline from 1 s before the true start to 1 s after its
end (at most 7 s), every 0.2 s: truth, ours, ball (x toward the side the possession team attacks, y,
height, speed, seen by the net), the 3 nearest players of the possession team (N) and the other (O):
id / distance / speed / approach speed; change net P(class = N), P(turnover or restart to N), short
net P(N spell start); and the DFL events nearby.
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
import dfl_data as dd  # noqa: E402
from pipeline.decode import H, A, D  # noqa: E402

FPS = 25
CODE = {H: "H", A: "A", D: "D"}
KINDS = {"caught": 0, "D": 1, "other": 2, "merged": 3, "shifted": 4}


def iou(a, b, c, d):
    return max(0, min(b, d) - max(a, c)) / (max(b, d) - min(a, c))


def outcome(filled, a, b, k, pred):
    ov = [(iou(a, b, c, d), c, d) for c, d, k2 in pred if k2 == k and d > a and c < b]
    best = max(ov)[0] if ov else 0.0
    seg = filled[a:b]
    if best >= 0.5:
        return 0
    if np.mean(seg == D) >= 0.5:
        return 1
    if np.mean(seg == (A if k == H else H)) >= 0.5:
        return 2
    _, c, d = max(ov) if ov else (0, a, b)
    return 3 if (ov and c <= a + 3 and d >= b - 3 and (d - c) > 2 * (b - a)) else 4


def collect(tag, alpha):
    import cn_sweep
    out = []
    for fp in sorted(glob.glob(os.path.join(ROOT, "data", "cache", "cn_e3k", "dfl_*.npz"))):
        g = os.path.basename(fp)[4:-4]
        z, cpr, fin = cn_sweep.dfl_final(g, tag, alpha)
        lab = z["lab"]
        runs = pm.valid_runs(lab)
        filled = pm.fill_abstentions(fin, runs)
        pred = pm.possession_spells(filled, runs)
        for a, b, k in pm.possession_spells(lab, runs):
            out.append(dict(g=g, a=a, b=b, k=k, kind=outcome(filled, a, b, k, pred)))
    return out


def show(c, ctx, tag, alpha):
    import cn_sweep
    from train_learned_dfl import test_seed
    import change_net as C
    g, a, b, k = c["g"], c["a"], c["b"], c["k"]
    if g not in ctx:
        z, cpr, fin = cn_sweep.dfl_final(g, tag, alpha)
        d = dd.load(g)
        ball = dd.prepare_ball(d, test_seed(g))
        lab = z["lab"]
        ctx.clear()
        ctx[g] = dict(lab=lab, ours=pm.fill_abstentions(fin, pm.valid_runs(lab)), cpr=cpr,
                      spr=z["spr"].astype(np.float32), d=d, seen=np.isfinite(ball[:, 0]))
    x = ctx[g]
    lab, ours, cpr, spr, d = x["lab"], x["ours"], x["cpr"], x["spr"], x["d"]
    nt = 0 if k == H else 1
    P = (d["players"][:, :, :2] - 0.5) * np.array(dd.PITCH)
    B = (d["ball"][:, :2] - 0.5) * np.array(dd.PITCH)
    bz = d["ball"][:, 2]
    team = d["team"]
    lo, hi = max(a - FPS, 0), min(b + FPS, a + 7 * FPS, len(lab))
    sgn = 1.0 if np.nanmean(P[lo:hi, team == nt, 0]) < np.nanmean(P[lo:hi, team != nt, 0]) else -1.0
    prev = CODE.get(lab[a - 1], "X") if a > 0 else "-"
    ev, kinds = d["events"], d["event_kind"]
    sel = np.flatnonzero((ev[:, 0] >= lo) & (ev[:, 0] < hi))
    print("\n=== {} {} {:.1f} s (before: {}), ours: {} ===".format(
        g, CODE[k], (b - a) / FPS, prev, [n for n, v in KINDS.items() if v == c["kind"]][0]))
    print("  events: " + "; ".join("{:+.1f}s {} {}{}".format((ev[i, 0] - a) / FPS, "N" if ev[i, 1] == nt else "O",
                                                            kinds[i], "" if ev[i, 2] else " (unctl)") for i in sel))
    print("  {:>5} {:>2} {:>2} | {:>6} {:>5} {:>4} {:>4} {:>2} | {:<34} | {:<34} | {:>4} {:>4} {:>4}".format(
        "t", "tr", "us", "ball x", "y", "z", "spd", "sn", "N team: id dist spd tow", "O team: id dist spd tow",
        "Pcls", "Pchg", "Pst"))
    ids = {}
    for tt in range(lo, hi, 5):
        vb = (B[min(tt + 2, len(B) - 1)] - B[max(tt - 2, 0)]) * FPS / 4.0
        tr = CODE.get(lab[tt], "X")
        line = "  {:>+5.1f} {:>2} {:>2} | {:>6.1f} {:>5.1f} {:>4.1f} {:>4.1f} {:>2} |".format(
            (tt - a) / FPS, tr if tr == "D" or tr == "X" else ("N" if lab[tt] == k else "O"),
            {H: "N" if k == H else "O", A: "N" if k == A else "O", D: "D"}.get(ours[tt], "X"),
            B[tt, 0] * sgn, B[tt, 1], bz[tt] if np.isfinite(bz[tt]) else -1, np.hypot(*vb), "y" if x["seen"][tt] else "-")
        for side in (nt, 1 - nt):
            s_ = np.flatnonzero(team == side)
            dist = np.hypot(*(P[tt, s_] - B[tt]).T)
            dist = np.where(np.isnan(dist), np.inf, dist)
            parts = []
            for j in s_[np.argsort(dist)[:3]]:
                dj = np.hypot(*(P[tt, j] - B[tt]))
                v = (P[min(tt + 2, len(P) - 1), j] - P[max(tt - 2, 0), j]) * FPS / 4.0
                tow = -np.dot(v, P[tt, j] - B[tt]) / max(dj, 0.3)
                key = ids.setdefault(j, ("n" if side == nt else "o") + str(sum(1 for q in ids if team[q] == side) + 1))
                parts.append("{}:{:.1f}/{:.1f}/{:+.1f}".format(key, dj, np.hypot(*v), tow))
            line += " {:<34} |".format(" ".join(parts))
        prev_live = tt > 0 and lab[max(a - 1, 0)] in (H, A)
        pchg = cpr[tt, nt] if prev_live else cpr[tt, 3 + nt]
        line += " {:>4.2f} {:>4.2f} {:>4.2f}".format(cpr[tt, C.N_TGT + nt], pchg, spr[tt, 2 + nt])
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="c1dl")
    ap.add_argument("--alpha", type=float, default=0.52)
    ap.add_argument("--kind", default="D", choices=list(KINDS))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--min-s", type=float, default=0.0)
    ap.add_argument("--max-s", type=float, default=1e9)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cs = collect(args.tag, args.alpha)
    pool = [c for c in cs if c["kind"] == KINDS[args.kind] and args.min_s <= (c["b"] - c["a"]) / FPS < args.max_s]
    print("{} of {} true possessions are '{}'".format(len(pool), len(cs), args.kind))
    rng = np.random.default_rng(args.seed)
    pick = sorted((pool[i] for i in rng.choice(len(pool), min(args.n, len(pool)), replace=False)),
                  key=lambda c: (c["g"], c["a"]))
    ctx = {}
    for c in pick:
        show(c, ctx, args.tag, args.alpha)


if __name__ == "__main__":
    main()
