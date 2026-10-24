"""Why is a submission dead (D) too long? Aggregate answer only.

    python tools/dead_diag.py evaluation/results/v6.csv [--set key=value ...]

Public window (diagnosis only):
  - true vs predicted D share, and frames of each confusion (alive called D, D called alive);
  - each true dead stretch: predicted D start / end offset (s, + = late), labels only;
  - on true-alive frames called D vs true-alive frames called alive: the mean of every term of the
    D score and the alive score, plus ball speed, nearest-player distance and ball-seen share.
Full match: predicted D-run lengths, and the mean of each D term on predicted D frames, split by
whether the ball is seen and moving (> still_ms). Means and shares only.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from report import PUBLIC_GT, FPS  # noqa: E402
from pipeline.features import build_features, runs_of  # noqa: E402
from pipeline.decode import PARAMS, falling, side_evidence  # noqa: E402
from pipeline.run import parse_overrides  # noqa: E402


def terms(f, p):
    """Every additive term of the D score, and the alive score, per frame."""
    near, close, _ = side_evidence(f["d_home"].astype(np.float64), f["d_away"].astype(np.float64),
                                   p["team_scale_m"], p["control_m"])
    gap = np.minimum(f["gap_s"] / p["gap_full_s"], 1.0)
    hold = (p["w_keeper"] > 0) * np.abs(f["keeper_hold"])
    still = (f["ball_speed"] < p["still_ms"]) & (f["ball_seen"] > 0)
    ps = np.where(np.isnan(f["player_speed"]), p["slow_ms"], f["player_speed"])
    slow = falling(ps, p["slow_ms"], p["slow_width_ms"])
    bs = np.where(np.isnan(f["ball_speed"]), 0.0, f["ball_speed"])
    moving = (1.0 - falling(bs, p["moving_ms"], p["moving_width_ms"])) * (f["ball_seen"] > 0)
    moving = moving * (1.0 - f["ball_out"])
    # Gated by the players: during a stoppage the ball also moves (passed back to the taker,
    # kicked away) while everyone walks, so a moving ball only counts when the players move too.
    gap = gap * slow ** p["gap_gate"] * (1.0 - hold)
    moving = moving * (1.0 - slow) ** p["moving_gate"]
    slow = slow * (1.0 - (p["w_moving"] > 0) * moving)
    alone = np.where(f["ball_seen"] > 0, 1.0 - falling(close, p["alone_m"], p["alone_width_m"]),
                     0.0)
    damp = 1.0 - p["still_damp"] * still * slow
    tags = (p["w_tag"] * f["tag_team"] + p["w_restart_team"] * f["tag_restart_team"]
            + p["w_keeper"] * f["keeper_hold"])
    out = {
        "gap": p["w_gap"] * gap, "out": p["w_out"] * f["ball_out"], "still": p["w_still"] * still,
        "slow": p["w_slow"] * slow, "setpiece": p["w_setpiece"] * f["tag_setpiece"],
        "restart": p["w_restart"] * f["tag_restart"], "alone": p["w_alone"] * alone,
    }
    out["D score"] = sum(out.values()) - p["dead_bias"]
    out["alive score"] = np.abs(p["w_team"] * near * damp + tags) + p["w_moving"] * moving
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submission")
    parser.add_argument("--set", nargs="*", metavar="KEY=VALUE")
    args = parser.parse_args()
    p = parse_overrides(args.set)

    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    pred = pm.label_codes(pm.read_label_csv(args.submission))
    graded = np.flatnonzero(gt != pm.X)
    sl = slice(graded[0], graded[-1] + 1)
    f = build_features()
    t = terms(f, p)

    g, q = gt[sl], pred[sl]
    alive_t, dead_t = (g == pm.H) | (g == pm.A), g == pm.D
    print("PUBLIC WINDOW  true D {:.1f}%  predicted D {:.1f}%".format(100 * np.mean(dead_t),
                                                                   100 * np.mean(q == pm.D)))
    print("  alive called D: {:.1f}s   D called alive: {:.1f}s".format(
        np.sum(alive_t & (q == pm.D)) / FPS, np.sum(dead_t & (q != pm.D)) / FPS))
    print("  true dead stretches: predicted D start / end offset (s, + = late)")
    for a, b in runs_of(dead_t):
        pd = q == pm.D
        # Predicted D run overlapping this true stretch the most.
        best = max(((s, e) for s, e in runs_of(pd) if s < b and a < e),
                   key=lambda r: min(r[1], b) - max(r[0], a), default=None)
        if best is None:
            print("    {:5.1f}s  no predicted D".format((b - a) / FPS))
        else:
            print("    {:5.1f}s  start {:+5.1f}  end {:+5.1f}".format(
                (b - a) / FPS, (best[0] - a) / FPS, (best[1] - b) / FPS))

    groups = {"alive called D": alive_t & (q == pm.D), "alive called alive": alive_t & (q != pm.D),
              "D called D": dead_t & (q == pm.D)}
    seen = f["ball_seen"][sl] > 0
    extra = {"ball seen %": 100.0 * seen,
             "ball speed": f["ball_speed"][sl], "nearest m": np.fmin(f["d_home"], f["d_away"])[sl],
             "player speed": f["player_speed"][sl]}
    print("\n  {:<14}".format("term (mean)") + "".join("{:>20}".format(k) for k in groups))
    for name, values in list(t.items()) + list(extra.items()):
        v = values[sl] if name not in extra else values
        print("  {:<14}".format(name) + "".join(
            "{:>20}".format("{:.2f}".format(np.nanmean(v[m])) if m.any() else "-")
            for m in groups.values()))
    print("  {:<14}".format("seconds") + "".join("{:>20.1f}".format(m.sum() / FPS)
                                                 for m in groups.values()))

    print("\nFULL MATCH ({})".format(os.path.basename(args.submission)))
    valid = pred != pm.X
    dead = (pred == pm.D) & valid
    lengths = np.array([(b - a) / FPS for a, b in runs_of(dead)])
    print("  D runs: {}  length p25/p50/p75/p90: {}  share of graded: {:.1f}%".format(
        len(lengths), " / ".join("{:.1f}".format(x) for x in np.percentile(lengths, (25, 50, 75, 90))),
        100 * dead.sum() / valid.sum()))
    moving = (f["ball_seen"] > 0) & (f["ball_speed"] >= p["still_ms"])
    split = {"D, ball moving": dead & moving, "D, ball still": dead & (f["ball_seen"] > 0) & ~moving,
             "D, no ball": dead & (f["ball_seen"] == 0), "alive": valid & (pred != pm.D)}
    print("  {:<14}".format("term (mean)") + "".join("{:>18}".format(k) for k in split))
    for name, values in list(t.items()) + [("player speed", f["player_speed"]),
                                           ("nearest m", np.fmin(f["d_home"], f["d_away"]))]:
        print("  {:<14}".format(name) + "".join(
            "{:>18}".format("{:.2f}".format(np.nanmean(values[m])) if m.any() else "-")
            for m in split.values()))
    print("  {:<14}".format("seconds") + "".join("{:>18.0f}".format(m.sum() / FPS)
                                                 for m in split.values()))

    # Restart lag: inside each predicted D run of 3 s or more, the first kick (ball seen, in the
    # pitch, faster than KICK_MS for KICK_S) is the restart; how long does D carry on after it?
    kick_ms, kick_n = 5.0, int(0.5 * FPS)
    fast = (f["ball_seen"] > 0) & (f["ball_out"] == 0) & (np.nan_to_num(f["ball_speed"]) >= kick_ms)
    lags, none = [], 0
    for a, b in runs_of(dead):
        if b - a < 3 * FPS:
            continue
        runs = [(s, e) for s, e in runs_of(fast[a:b]) if e - s >= kick_n]
        if not runs:
            none += 1
            continue
        lags.append((b - a - runs[0][0]) / FPS)
    lags = np.array(lags)
    print("  restart lag (D runs >= 3 s with a kick inside): {} runs, {} without a kick".format(
        len(lags), none))
    print("    D carries on after the first kick p25/p50/p75/p90: {} s;  over 3 s: {:.0f}%".format(
        " / ".join("{:.1f}".format(x) for x in np.percentile(lags, (25, 50, 75, 90))),
        100 * np.mean(lags > 3)))


if __name__ == '__main__':
    main()
