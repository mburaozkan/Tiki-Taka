"""Does DFL's possession flag follow the nearest team? DFL open data only. Aggregate only.

    python tools/dfl_flag_rule.py [--tag c1dlbd]

At every true turnover (true ball, before our gaps): offset from the true change to the moment the
new team's nearest player becomes nearer the ball than the old team's (the last such crossing within
-3..+3 s); ball in flight at the change (speed > 8 m/s), ball height; split found / missed by the net.
Also: over all live frames, how often the true team = the team nearest the ball (raw and with a
hysteresis of 1 m).
"""
import argparse, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa
import change_net as C  # noqa
import dfl_data as dd  # noqa
import miss_profile as M  # noqa
from pipeline.decode import H, A, D  # noqa
FPS = 25


def main():
    import cn_dfl
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="c1dlbd"); args = ap.parse_args()
    R, agree, agree_h, nlive = [], 0, 0, 0
    for g in dd.GAMES:
        with np.load(cn_dfl.path(g)) as z:
            lab, blk = z["lab"], [tuple(b) for b in z["blk"]]
        dec = C.viterbi(cn_dfl.heldout(args.tag, g), blk, 0.5, -3.0)
        runs = pm.valid_runs(lab)
        filled = pm.fill_abstentions(dec, runs)
        pc = np.array([f for f, k in M.changes(filled, runs) if k == 0])
        d = dd.load(g)
        P = (d["players"][:, :, :2] - 0.5) * np.array(dd.PITCH)
        B = (d["ball"][:, :2] - 0.5) * np.array(dd.PITCH)
        dist = np.hypot(*(P - B[:, None]).transpose(2, 0, 1))
        dist = np.where(np.isnan(dist), np.inf, dist)
        dh = dist[:, d["team"] == 0].min(1)
        da = dist[:, d["team"] == 1].min(1)
        spd = np.hypot(*np.gradient(B, axis=0).T) * FPS
        live = np.isin(lab, (H, A)) & np.isfinite(dh) & np.isfinite(da) & np.isfinite(B[:, 0])
        near = np.where(dh < da, H, A)
        agree += np.sum(near[live] == lab[live]); nlive += live.sum()
        hy = np.full(len(lab), H)
        cur = H
        for t in range(len(lab)):
            if np.isfinite(dh[t]) and np.isfinite(da[t]):
                if cur == H and da[t] < dh[t] - 1.0:
                    cur = A
                elif cur == A and dh[t] < da[t] - 1.0:
                    cur = H
            hy[t] = cur
        agree_h += np.sum(hy[live] == lab[live])
        for t in [f for f, k in M.changes(lab, runs) if k == 0]:
            new = lab[t]
            dn, do = (dh, da) if new == H else (da, dh)
            w = np.arange(max(t - 75, 1), min(t + 76, len(lab)))
            cross = w[(dn[w] < do[w]) & (dn[w - 1] >= do[w - 1])]
            off = (cross[np.argmin(np.abs(cross - t))] - t) / FPS if len(cross) else np.nan
            missed = (np.min(np.abs(pc - t)) if len(pc) else 9999) > 2 * FPS
            R.append([off, spd[t] if np.isfinite(spd[t]) else np.nan, d["ball"][t, 2], dn[t], do[t], missed,
                      dn[t] < do[t]])
    R = np.array(R, float)
    print("live frames: true team = nearest team {:.1%}; with 1 m hysteresis {:.1%}".format(agree / nlive, agree_h / nlive))
    for nm, s in (("all", np.ones(len(R), bool)), ("found", R[:, 5] == 0), ("missed", R[:, 5] == 1)):
        o = R[s, 0]
        print("{:<7} n {:>4} | new team nearer at the change {:>5.1%} | nearest-team crossing within 0.2 / 0.5 / 1 s: "
              "{:>4.0%} / {:>4.0%} / {:>4.0%}, none within 3 s {:>4.0%} | ball > 8 m/s {:>4.0%}, height > 1 m {:>4.0%} | "
              "median dist new / old {:.1f} / {:.1f} m".format(
                  nm, s.sum(), np.mean(R[s, 6]), np.mean(np.abs(o) <= 0.2), np.mean(np.abs(o) <= 0.5),
                  np.mean(np.abs(o) <= 1), np.mean(np.isnan(o)), np.nanmean(R[s, 1] > 8), np.nanmean(R[s, 2] > 1),
                  np.nanmedian(R[s, 3]), np.nanmedian(R[s, 4])))


if __name__ == "__main__":
    main()
