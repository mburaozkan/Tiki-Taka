"""A change net with the ball net in the gaps (+ duel inputs) on our match. Aggregate only.

    $env:MTAH_VARIANT="_kt"; python tools/cn_ours_bd.py [--tag c1dlbd] [--duel]
    $env:MTAH_VARIANT="_kt"; python tools/cn_ours_bd.py --tag c1dld --duel --nofill   # original ball
    $env:MTAH_VARIANT="_kt"; python tools/cn_ours_bd.py --tag c1dldl --duel --loose --nofill

Builds our match's change-net inputs exactly as change_net.ours_setup, then (as for the external
sets) rewrites the gap frames of the ball-derived inputs from the ball net trained on all 7 DFL
matches (models/ball_net_all.pt, tools/cn_ballfill.py), and with --duel appends the duel inputs
(tools/cn_duel.py). Decodes alone (w 0.5, bias -3), alone with v31_d1's D fixed, and mixed with the
base (alpha 0.3 / 0.5); writes evaluation/results/<name>.csv for each and prints the public line.
Nothing but scores and counts is printed.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="c1dlbd")
    ap.add_argument("--duel", action="store_true")
    ap.add_argument("--loose", action="store_true", help="also the loose-ball memory inputs (cn_loose.py; needs --duel)")
    ap.add_argument("--nofill", action="store_true", help="keep the original ball (no ball net in the gaps)")
    args = ap.parse_args()
    if args.duel:
        os.environ["SN_EXTRA"] = "ctl,keep,since,last" if args.loose else "ctl,keep"
    import torch
    import short_net as S
    import change_net as C
    import possession_metrics as pm
    import dfl_data as dd
    import ours_chain as oc
    import turnover_cls as T
    import ball_net as N
    import cn_ballfill as CB
    from atd_cache import load_arrays
    from pipeline.features import LENGTH, WIDTH
    from pipeline.decode import D
    S.N_TGT = C.N_TGT
    codes, ex = oc.ours_v31(dead_prefix=os.path.join("models", "dead_ext_d1"))
    f = {k[2:]: v for k, v in ex.items() if k.startswith("f_")}
    blk = [tuple(b) for b in ex["blk"]]
    n = len(codes)
    period = np.zeros(n, dtype=np.int8)
    for i, (a, b) in enumerate(blk, start=1):
        period[a:b] = i
    c = dict(ex, period=period)
    base = T.base_chain(c, f, blk, ex["pdead"])
    atd = load_arrays()
    d = {"players": atd["players"], "team": atd["team"], "period": period}
    sign = T.attack_sign(d, period)
    pitch = (LENGTH, WIDTH)
    ball = dd.blur_ball(atd["ball"])
    full_team = S.TEAM
    S.TEAM = S.BASE_TEAM                                               # base channels first
    x = S.frame_inputs(c, f, d, ball, pitch, ex["pdead"], sign)
    S.TEAM = full_team
    old = S.SHARED + ["h_" + k for k in S.BASE_TEAM] + ["a_" + k for k in S.BASE_TEAM]
    xo = x.copy()
    gap = np.zeros(n, bool)
    if not args.nofill:
        filled, _ = N.fill(N.load("all"), ball, d["players"], d["team"], blk, pitch)
        gap = ~np.isfinite(ball[:, 0]) & np.isfinite(filled[:, 0])
        gm = CB.geom(d["players"], d["team"], filled, pitch, sign)
        for k, v in gm.items():
            xo[gap, old.index(k)] = v[gap]
    print("our match: {} frames, ball unseen {:.1%}, filled {:.1%}".format(n, np.mean(~np.isfinite(ball[:, 0])), gap.mean()))
    if args.duel:
        import cn_duel as CD
        bx = xo[:, old.index("bx")] * sign
        by = xo[:, old.index("by")]
        un = (xo[:, old.index("seen")] < 0.5) & (bx == 0) & (by == 0)
        ex2 = CD.duel_channels(d["players"], d["team"], np.where(un, np.nan, bx), np.where(un, np.nan, by), pitch)
        if args.loose:
            import cn_loose as CL
            starts = [a for a, _ in blk]
            ex2.update(CL.loose_channels(ex2["h_keep"], ex2["a_keep"], starts))
        xo = np.stack([ex2[nm] if nm[:2] in ("h_", "a_") and nm in ex2 else xo[:, old.index(nm)]
                       for nm in S.channel_names()], axis=1)
    mp = os.path.join(ROOT, "models", "change_net", "{}_all.pt".format(args.tag))   # committed (c1dld)
    if not os.path.isfile(mp):
        mp = os.path.join(C.MODELS, "{}_all.pt".format(args.tag))
    ck = torch.load(mp, weights_only=False)
    model = S.make_model(ck["n_in"], ck.get("hid", 128)).cuda()
    model.load_state_dict(ck["state"])
    pr = S.predict(model, ck["mu"], ck["sd"], S.drop_channels(xo.astype(np.float32)))
    np.savez(os.path.join(C.OUT, "ours_{}.npz".format(args.tag)), pr=pr, base=base, blk=np.array(blk))
    print("  " + str(oc.write_and_score(base, "v31_d1_basecheck")))
    outs = [("alone", C.viterbi(pr, blk, 0.5, -3.0)),
            ("alone_dfix", C.viterbi(pr, blk, 0.5, -3.0, base == D))]
    for al in (0.3, 0.5):
        q, _ = C.mix_pr(pr, base, blk, al, 0.0)
        outs.append(("mix{:g}".format(al).replace(".", ""), C.viterbi(q, blk, 1.0, -3.0)))
    for nm, out in outs:
        name = "v31_d1_{}_{}_kt".format(args.tag, nm)
        sp = pm.possession_spells(out, pm.valid_runs(out))
        print("{:<11} {} possessions, D {:.1%}, frames changed vs base {:.1%}".format(
            nm, len(sp), np.mean(out == D), np.mean(out != base)))
        print("  " + str(oc.write_and_score(out, name)))


if __name__ == "__main__":
    main()
