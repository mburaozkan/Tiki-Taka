"""Change net mix + e3k short spells: do the two gains add up? Aggregate only.

    python tools/cn_e3k.py ext   [--tag c1chdl]    # 84 external matches (held-out predictions)
    python tools/cn_e3k.py dfl   [--tag c1chdl]    # 7 DFL matches (held out for every model)
    python tools/cn_e3k.py ours  [--tag c1chdl]    # our match: public window, writes the files

Four outputs per match, same settings as already chosen (no new tuning):
  base     v31_d1
  e3k      base + e3 short-net spells kept by the check k1 (gen 0.6, keep >= 0.3, edges refined, over D)
  mix      the change net's class head blended 50 / 50 with the base's codes, decoded with its change
           heads (change_net.mix_pr alpha 0.5, beta 0; viterbi w 1, bias -3), as 2026-11-08
  mix+e3k  mix, then e3k's spells pasted on top (the same spells, the same order)
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
os.environ.setdefault("TC_ROWS", "v2")
import possession_metrics as pm  # noqa: E402
from pipeline.decode import H, A, D  # noqa: E402

ALPHA, W, BIAS = 0.5, 1.0, -3.0
NAMES = ("base v31_d1", "e3k", "mix", "mix+e3k")


def paste(codes, spells):
    out = codes.copy()
    for a, b, team in spells:
        seg = out[a:b]
        out[a:b] = np.where(np.isin(seg, (H, A, D)), team, seg)
    return out


def mixed(pr, base, blk):
    import change_net as C
    q, _ = C.mix_pr(pr, base, blk, ALPHA, 0.0)
    return C.viterbi(q, blk, W, BIAS)


def ext_spells(name, g, n):
    """e3k's kept spells on an external match, in paste order (as short_edges.e3k)."""
    import short_net as S
    import short_check as K
    key = "{}_{}".format(name, g)
    with np.load(os.path.join(S.OUT, "pred_e3.npz")) as z:
        pr = z[key].astype(np.float32)
    with np.load(os.path.join(K.OUT, "e3_g0.6_{}_{}.npz".format(name, g))) as z:
        meta = z["meta"]
    with np.load(os.path.join(K.OUT, "pred_k1_e3_g0.6.npz")) as z:
        pk = z[key] if key in z.files else np.zeros(0)
    out = []
    for i in np.argsort(pk):
        if pk[i] < 0.3:
            continue
        a, b, team = int(meta[i, 0]), int(meta[i, 1]), int(meta[i, 2])
        a, b = K.refine_edges(pr, a, b, 0 if team == H else 1, n)
        out.append((a, b, team))
    return out


def ext_match(job):
    name, g, tag = job
    import turnover_cls as T
    import run_v31_external as rx
    import pair_cls as P
    import change_net as C
    from decoder_sweep import stats
    c, f = T.load_cache(name, g)
    pdead = P.pdead_of(name, g)
    blk = rx.blocks(c["period"])
    base = T.base_chain(c, f, blk, pdead)
    lab = c["labels"]
    with np.load(os.path.join(C.OUT, "pred_{}.npz".format(tag))) as z:
        pr = z["{}_{}".format(name, g)].astype(np.float32)
    sp = ext_spells(name, g, len(base))
    mx = mixed(pr, base, blk)
    return name, g, np.array([stats(lab, x) for x in (base, paste(base, sp), mx, paste(mx, sp))], float)


def report(rows, sets):
    names = np.array([r[0] for r in rows])
    arr = np.stack([r[2] for r in rows])
    print("{:<12} ".format("") + " ".join("{:>22}".format(s) for s in sets)
          + " | {:>10} {:>9} | {:>9}".format("caught <2s", "2-5s", "pred/true"))
    for j, nm in enumerate(NAMES):
        parts = []
        for s in sets:
            v = arr[names == s, j, 0]
            d = v - arr[names == s, 0, 0]
            parts.append("{:.4f} ({:>2}/{:<2} vs base)".format(v.mean(), int(np.sum(d > 1e-9)), int(np.sum(d < -1e-9))))
        t = arr[:, j, 1:].sum(axis=0)
        print("{:<12} ".format(nm) + " ".join("{:>22}".format(p) for p in parts)
              + " | {:>9.1%} {:>9.1%} | {:>9.2f}".format(t[1] / t[0], t[3] / t[2], t[5] / t[4]))
    for s in sets:
        d = arr[names == s, 3, 0] - arr[names == s, 1, 0]
        print("{}: mix+e3k vs e3k {:+.4f} ({}/{})".format(s, d.mean(), int(np.sum(d > 1e-9)), int(np.sum(d < -1e-9))))


def ext_cmd(args):
    import turnover_cls as T
    with Pool(args.jobs) as pool:
        rows = pool.map(ext_match, [(n, g, args.tag) for n, g, _ in T.ext_list()], chunksize=1)
    report(rows, ("skillcorner", "pff"))


def dfl_cmd(args):
    """DFL: base and change-net predictions from cn_dfl's cache (held out); e3k spells as dfl_short."""
    import torch
    import xgboost as xgb
    import dfl_data as dd
    import cn_dfl
    import turnover_cls as T
    import short_net as S
    import short_check as K
    import in_domain_team as idt
    import eval_touch_run as tr
    import eval_dead_v30 as ev
    import train_ext_heads as th
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
    nets = []
    for mem in ("s1", "s1b", "s1nz:bz"):
        tag, _, drop = mem.partition(":")
        ck = torch.load(os.path.join(S.MODELS, "{}_all.pt".format(tag)), weights_only=False)
        model = S.make_model(ck["n_in"], ck.get("hid", 128)).cuda()
        model.load_state_dict(ck["state"])
        nets.append((model, ck, drop))
    bst = xgb.Booster()
    bst.load_model(os.path.join(K.OUT, "models", "k1_e3_g0.6_all.json"))
    rows = []
    for g in dd.GAMES:
        z = dict(np.load(cn_dfl.path(g)))
        cpr = cn_dfl.heldout(args.tag, g)
        if cpr is None:
            cpr = cn_dfl.predict_tag(args.tag, z["x"])
        lab0, base0, blk = z["lab"], z["base"], [tuple(b) for b in z["blk"]]
        d = dd.load(g)
        lab, hand, prob = data[g]
        f = match_features(g, test_seed(g))
        p1 = stage1(prob, f)
        p_ref = np.mean([team["{}__{}".format(g, s)] for s in range(tr.TEAM_SEEDS)], axis=0)
        vote = tr.run_vote(f, ev.VOTE)
        pdead = th.seeds_mean({k: v for k, v in heads.items() if k.startswith("dfl_" + g + "__")}, "dfl_" + g)
        c = dict(hand=hand, p1=p1, p_ref=p_ref, vote=np.asarray(vote, np.float32), period=d["period"], labels=lab)
        base = T.base_chain(c, f, blocks(d["period"]), pdead)
        if not np.array_equal(base, base0):
            print("  note: {} base differs from cn_dfl's cache on {:.2%} of frames".format(g, np.mean(base != base0)))
        ball = dd.prepare_ball(d, test_seed(g))
        c["ball_z"] = (ball[:, 2] if ball.shape[1] > 2 else np.full(len(ball), np.nan)).astype(np.float32)
        dd_ = {"players": d["players"], "team": d["team"], "period": d["period"]}
        x0 = S.frame_inputs(c, f, dd_, ball, dd.PITCH, pdead, T.attack_sign(dd_, d["period"]))
        prs = []
        for model, ck, drop in nets:
            x = np.array(x0, copy=True)
            if drop:
                x[:, [S.channel_names().index(ch) for ch in drop.split("+")]] = 0
            prs.append(S.predict(model, ck["mu"], ck["sd"], x))
        pr = np.mean(prs, axis=0)
        ctx = T.match_ctx(c, f, base, pdead, dd_, ball, dd.PITCH,
                          idt.player_columns(d["players"], d["team"], d["period"], dd.PITCH))
        cand = K.gen_spells(pr, 0.6)
        xr = [list(K.spell_inputs(ctx, pr, a, b, s_).values()) for a, b, s_ in cand]
        pk = bst.predict(xgb.DMatrix(np.array(xr, np.float32))) if xr else np.zeros(0)
        sp = []
        for i in np.argsort(pk):
            if pk[i] < 0.3:
                continue
            a, b, s_ = cand[i]
            a, b = K.refine_edges(pr, a, b, s_, len(base))
            sp.append((a, b, H if s_ == 0 else A))
        mx = mixed(cpr, base, blk)
        r = [stats(lab, x) for x in (base, paste(base, sp), mx, paste(mx, sp))]
        os.makedirs(os.path.join(ROOT, "data", "cache", "cn_e3k"), exist_ok=True)
        np.savez_compressed(os.path.join(ROOT, "data", "cache", "cn_e3k", "dfl_{}.npz".format(g)), lab=lab, base=base,
                            final=paste(mx, sp), cpr=cpr.astype(np.float16), spr=pr.astype(np.float16),
                            tph=f["touch_p_home"].astype(np.float16), tpa=f["touch_p_away"].astype(np.float16),
                            period=d["period"])
        rows.append(("dfl", g, np.array(r, float)))
        print("  {}: base {:.4f}  e3k {:.4f}  mix {:.4f}  mix+e3k {:.4f}".format(g, *[x[0] for x in r]), flush=True)
    report(rows, ("dfl",))


def ours_cmd(args):
    import change_net as C
    import ours_chain as oc
    z = np.load(os.path.join(C.OUT, "ours_{}.npz".format(args.tag)))
    pr, base, blk = z["pr"].astype(np.float32), z["base"], [tuple(b) for b in z["blk"]]
    import short_net as S
    sp = [tuple(int(v) for v in r) for r in np.load(os.path.join(S.OUT, "ours_spells_{}.npy".format(args.spells)))]
    mx = mixed(pr, base, blk)
    for nm, x in (("v31_d1_chk", base), ("v31_d1_e3k_chk", paste(base, sp)),
                  ("v31_d1_{}m".format(args.tag), mx), ("v31_d1_{}m_e3k".format(args.tag), paste(mx, sp))):
        print("  " + str(oc.write_and_score(x, nm)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("ext", "dfl", "ours"))
    ap.add_argument("--tag", default="c1chdl")
    ap.add_argument("--spells", default="v31_d1_e3k_rb")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    {"ext": ext_cmd, "dfl": dfl_cmd, "ours": ours_cmd}[args.cmd](args)


if __name__ == "__main__":
    main()
