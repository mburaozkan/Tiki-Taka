"""e3k (short net + keep check) on the 7 DFL matches, leak-free. Aggregate only.

    python tools/dfl_short.py [--ens s1,s1b,s1nz:bz] [--ens-tag e3] [--check k1:0.6:0.3]

Base v31_d1 on DFL from held-out pieces only: stage 1 from v25_inputs (leave one match out), the
team refiner refine_b2 (held out), the run vote, and the d1 dead head's DFL predictions trained
without that match (data/cache/ext_heads_d1.npz, keys dfl_<g>, 'dfllomo'); turnover_cls.base_chain
on top. The short-net ensemble and the keep check are the all-84 models (SkillCorner + PFF only),
so DFL is a provider they never saw. Spells pasted as short_net.py ours (refined edges, over D).
Truth: DFL's controlled labels (note: they drop most landing-touch short possessions, so DFL is
a strict judge for inserts). Prints per match and mean: spellF1 base / e3k, up / down, short caught.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
os.environ.setdefault("TC_ROWS", "v2")
import possession_metrics as pm  # noqa: E402
from pipeline.decode import H, A, D  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ens", default="s1,s1b,s1nz:bz")
    ap.add_argument("--ens-tag", default="e3")
    ap.add_argument("--check", default="k1:0.6:0.3")
    ap.add_argument("--snap", default="", help="W:P change-point snap (tools/short_edges.py)")
    args = ap.parse_args()
    import torch
    import xgboost as xgb
    import dfl_data as dd
    import eval_touch_run as tr
    import eval_dead_v30 as ev
    import train_ext_heads as th
    import turnover_cls as T
    import short_net as S
    import short_check as K
    import in_domain_team as idt
    from eval_mbr import v25_inputs
    from eval_refine import stage1
    from train_learned_dfl import match_features, test_seed, blocks
    from decoder_sweep import stats
    T.V2 = True
    data = v25_inputs()
    with np.load(os.path.join(dd.CACHE, "refine_{}.npz".format(tr.TEAM_TAG))) as z:
        team = {k: z[k] for k in z.files}
    with np.load(os.path.join(ROOT, "data", "cache", "ext_heads_{}.npz".format(T.DEAD_TAG))) as z:
        heads = {k: z[k] for k in z.files if k.startswith("dfl_")}
    members = args.ens.split(",")
    nets = []
    for mem in members:
        tag, _, drop = mem.partition(":")
        ck = torch.load(os.path.join(S.MODELS, "{}_all.pt".format(tag)), weights_only=False)
        model = S.make_model(ck["n_in"], ck.get("hid", 128)).cuda()
        model.load_state_dict(ck["state"])
        nets.append((model, ck, drop))
    ktag, gen, thr = args.check.split(":")
    gen, thr = float(gen), float(thr)
    bst = xgb.Booster()
    bst.load_model(os.path.join(K.OUT, "models", "{}_{}_g{:g}_all.json".format(ktag, args.ens_tag, gen)))
    print("{:<8} {:>8} {:>8} {:>8} | caught <2s base/e3k  2-5s base/e3k | poss true/base/e3k".format(
        "match", "v31_d1", "e3k", "delta"))
    rows = []
    for g in dd.GAMES:
        d = dd.load(g)
        lab, hand, prob = data[g]
        f = match_features(g, test_seed(g))
        blk = blocks(d["period"])
        p1 = stage1(prob, f)
        p_ref = np.mean([team["{}__{}".format(g, s)] for s in range(tr.TEAM_SEEDS)], axis=0)
        vote = tr.run_vote(f, ev.VOTE)
        pdead = th.seeds_mean({k: v for k, v in heads.items() if k.startswith("dfl_" + g + "__")}, "dfl_" + g)
        c = dict(hand=hand, p1=p1, p_ref=p_ref, vote=np.asarray(vote, np.float32), period=d["period"], labels=lab)
        base = T.base_chain(c, f, blk, pdead)
        ball = dd.prepare_ball(d, test_seed(g))           # the ball match_features used (blurred, our gaps)
        c["ball_z"] = (ball[:, 2] if ball.shape[1] > 2 else np.full(len(ball), np.nan)).astype(np.float32)
        dd_ = {"players": d["players"], "team": d["team"], "period": d["period"]}
        sign = T.attack_sign(dd_, d["period"])
        xraw = S.frame_inputs(c, f, dd_, ball, dd.PITCH, pdead, sign)
        prs = []
        for model, ck, drop in nets:
            x = np.array(xraw, copy=True)
            if drop:
                x[:, [S.channel_names().index(ch) for ch in drop.split("+")]] = 0
            prs.append(S.predict(model, ck["mu"], ck["sd"], x))
        pr = np.mean(prs, axis=0)
        cols = idt.player_columns(d["players"], d["team"], d["period"], dd.PITCH)
        ctx = T.match_ctx(c, f, base, pdead, dd_, ball, dd.PITCH, cols)
        spells = K.gen_spells(pr, gen)
        xr, names = [], None
        for a, b, s_ in spells:
            xx = K.spell_inputs(ctx, pr, a, b, s_)
            names = names or list(xx)
            xr.append([xx[k] for k in names])
        pk = bst.predict(xgb.DMatrix(np.array(xr, np.float32))) if xr else np.zeros(0)
        new = base.copy()
        for i in np.argsort(pk):
            if pk[i] < thr:
                continue
            a, b, s_ = spells[i]
            a, b = K.refine_edges(pr, a, b, s_, len(new))
            seg = new[a:b]
            new[a:b] = np.where(np.isin(seg, (H, A, D)), H if s_ == 0 else A, seg)
        if args.snap:
            import short_edges as SE
            w_s, p_min = (float(v) for v in args.snap.split(":"))
            e3k_only = new.copy()
            new = SE.snap(new, pr, w_s, p_min)
            print("  snap vs e3k alone: {:+.4f}".format(stats(lab, new)[0] - stats(lab, e3k_only)[0]))
        sb, sn = stats(lab, base), stats(lab, new)
        rows.append((sb, sn))
        print("{:<8} {:>8.4f} {:>8.4f} {:>+8.4f} | {:>4}/{:<4} of {:<4} {:>4}/{:<4} of {:<4} | {}/{}/{}".format(
            g, sb[0], sn[0], sn[0] - sb[0], sb[2], sn[2], sb[1], sb[4], sn[4], sb[3], sb[5], sb[6], sn[6]), flush=True)
    b = np.array([r[0][0] for r in rows])
    e = np.array([r[1][0] for r in rows])
    print("mean: v31_d1 {:.4f}, e3k {:.4f}, delta {:+.4f} ({}/{} matches up/down)".format(
        b.mean(), e.mean(), (e - b).mean(), int(np.sum(e - b > 1e-9)), int(np.sum(e - b < -1e-9))))


if __name__ == "__main__":
    main()
