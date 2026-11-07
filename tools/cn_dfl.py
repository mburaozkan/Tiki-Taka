"""The change network on DFL (7 full matches it never saw). Aggregate only.

    python tools/cn_dfl.py prep                 # per-match inputs, data/cache/change_net/dfl_*.npz
    python tools/cn_dfl.py eval --tags c1 c1ch  # base v31_d1 vs the net (all-84 model) vs mixes

DFL as the realistic benchmark: dead ball animated, blurred, our gap pattern (dfl_data.prepare_ball,
the features of train_learned_dfl.match_features). The base v31_d1 is held out per match: first stage
(stage1 of the leave-one-match-out v25 inputs), refiner refine_b2 (held out), dead head d1 (held-out
DFL predictions in ext_heads_d1.npz). The change net was trained on SkillCorner + PFF only, so every
DFL match is new to it. Prints spellF1 per setting (mean, matches up / down vs the base), short catches,
and change recall within 0.5 / 1 s.
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
import change_net as C  # noqa: E402
import short_net as S  # noqa: E402
import dfl_data as dd  # noqa: E402

DROPS = {"c1": "", "c1nz": "bz,hold", "c1ch": "drop_ch.txt", "c1raw": "drop_raw.txt", "c1dl": ""}


def path(g):
    return os.path.join(C.OUT, "dfl_{}.npz".format(g))


def prep_cmd(args):
    import eval_dead_v30 as ev
    import eval_touch_run as tr
    import train_ext_heads as th
    import turnover_cls as T
    from train_learned_dfl import test_seed
    with np.load(os.path.join(dd.CACHE, "refine_{}.npz".format(tr.TEAM_TAG))) as cc:
        b2 = {k: cc[k] for k in cc.files}
    with np.load(os.path.join(ROOT, "data", "cache", "ext_heads_d1.npz")) as z:
        heads = {k: z[k] for k in z.files if k.startswith("dfl_")}
    for g in dd.GAMES:
        if os.path.isfile(path(g)):
            continue
        lab, hand, p1, f, blk, _, _ = th.dfl_data()[g]
        d = dd.load(g)
        ball = dd.prepare_ball(d, test_seed(g))
        p_ref = np.mean([b2["{}__{}".format(g, s)] for s in range(tr.TEAM_SEEDS)], axis=0)
        pdead = th.seeds_mean(heads, "dfl_" + g)
        vote = tr.run_vote(f, ev.VOTE)
        period = np.zeros(len(lab), np.int8)
        for i, (a, b) in enumerate(blk, start=1):
            period[a:b] = i
        c = {"hand": hand, "p1": p1, "p_ref": p_ref, "vote": vote, "period": period}
        base = T.base_chain(c, f, blk, pdead)
        dd_ = {"players": d["players"], "team": d["team"], "period": period}
        sign = T.attack_sign(dd_, period)
        x = S.frame_inputs(c, f, dd_, ball, dd.PITCH, pdead, sign)
        np.savez(path(g), x=x.astype(np.float16), lab=lab, base=base, blk=np.array(blk))
        print("  DFL {}: base spellF1 {:.3f}".format(g, C_score(lab, base)), flush=True)


def C_score(lab, codes):
    import eval_touch_run as tr
    return tr.score(lab, codes)


def drop_list(tag):
    v = DROPS.get(tag, "")
    if v.endswith(".txt"):
        v = open(os.path.join(C.OUT, v)).read().strip()
    return [c for c in v.split(",") if c]


def predict_tag(tag, x):
    import torch
    S.N_TGT = C.N_TGT
    x = np.array(x, dtype=np.float32, copy=True)
    dr = drop_list(tag)
    if dr:
        x[:, [S.channel_names().index(c) for c in dr]] = 0
    ck = torch.load(os.path.join(C.MODELS, "{}_all.pt".format(tag)), weights_only=False)
    model = S.make_model(ck["n_in"], ck.get("hid", 128)).cuda()
    model.load_state_dict(ck["state"])
    return S.predict(model, ck["mu"], ck["sd"], x)


def heldout(tag, g):
    """the held-out DFL prediction (folds / hold:dfl) if the tag has one, else None."""
    p = os.path.join(C.OUT, "pred_{}.npz".format(tag))
    if not os.path.isfile(p):
        return None
    with np.load(p) as z:
        return z["dfl_" + g].astype(np.float32) if "dfl_" + g in z.files else None


def eval_cmd(args):
    import miss_profile as M
    from decoder_sweep import stats
    data = {g: dict(np.load(path(g))) for g in dd.GAMES}
    rows = {}
    for tag in args.tags:
        for g in dd.GAMES:
            z = data[g]
            pr = heldout(tag, g)
            if pr is None:
                pr = predict_tag(tag, z["x"])
            lab, base, blk = z["lab"], z["base"], [tuple(b) for b in z["blk"]]
            runs = pm.valid_runs(lab)
            outs = [("base v31_d1", base)] if tag == args.tags[0] else []
            for w, bias in ((0.5, -3.0), (1.0, -3.0), (1.0, -1.5)):
                outs.append(("{} alone w {:g} bias {:g}".format(tag, w, bias), C.viterbi(pr, blk, w, bias)))
            outs.append(("{} alone, base D fixed".format(tag), C.viterbi(pr, blk, 1.0, -3.0, base == D_)))
            for al in (0.3, 0.5):
                q, dm = C.mix_pr(pr, base, blk, al, 0.0)
                outs.append(("{} mix alpha {:g}".format(tag, al), C.viterbi(q, blk, 1.0, -3.0)))
            for nm, cd in outs:
                st = stats(lab, cd)
                bd = M.boundaries(lab, pm.fill_abstentions(cd, runs), runs)
                rows.setdefault(nm, []).append((st, bd))
    names = list(rows)
    ref = np.array([r[0][0] for r in rows["base v31_d1"]])
    print("DFL, 7 matches (held out for every model)")
    print("{:<30} {:>8} {:>6} | {:>10} {:>8} | {:>9} {:>9} | turnover / stop / restart recall 1 s".format(
        "setting", "spellF1", "up/dn", "caught<2s", "2-5s", "pred/true", "precision"))
    for nm in names:
        st = np.array([r[0] for r in rows[nm]])
        v = st[:, 0]
        t = st[:, 1:].sum(axis=0)
        rec = []
        for ty in range(3):
            nt = sum(r[1][ty][0] for r in rows[nm])
            ht = sum(r[1][ty][1][2] for r in rows[nm])
            rec.append(ht / max(nt, 1))
        print("{:<30} {:>8.4f} {:>2}/{:<2}  | {:>9.1%} {:>8.1%} | {:>9.2f} {:>9.1%} | {:.0%} / {:.0%} / {:.0%}".format(
            nm, v.mean(), int(np.sum(v > ref + 1e-9)), int(np.sum(v < ref - 1e-9)),
            t[1] / max(t[0], 1), t[3] / max(t[2], 1), t[5] / t[4], t[6] / t[5], *rec))


D_ = None


def main():
    global D_
    from pipeline.decode import D
    D_ = D
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("prep", "eval"))
    ap.add_argument("--tags", nargs="+", default=["c1"])
    args = ap.parse_args()
    {"prep": prep_cmd, "eval": eval_cmd}[args.cmd](args)


if __name__ == "__main__":
    main()
