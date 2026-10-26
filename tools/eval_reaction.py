"""Team reaction as a vote in v27's team stage (experiment 6, step 1). DFL, aggregate only.

    python tools/eval_reaction.py

Teams reorganise 1-8 s after a turnover, and we label offline, so frame t may read the future.
Votes (home +), from pipeline/reaction.py columns (cached per DFL match by in_domain_team):
  <feat> shift s   tanh(level(t + s) / scale), level = home - away of the feature
  <feat> diff k    tanh(change_k(t) / scale), change_k = future-minus-past difference over k s
feat: spread (the team in possession spreads out) or cx (its centroid sits higher up the pitch);
scale = the match's standard deviation of the signal. Each vote is added to the v27 team stage
(H/A split only; D still comes from the hand path) with weight w.

Per setting: spellF1 per match, mean, delta vs v27, folds won / lost; turnover timing (direct
true H<->A changes: found within 5 s by a predicted change to the same team, median signed offset,
share within 1 s); false switches per hour inside true possessions >= 10 s.
"""

import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
import dfl_data as dd  # noqa: E402
import in_domain_team as idt  # noqa: E402
from eval_mbr import v25_inputs  # noqa: E402
from train_learned_dfl import match_features, test_seed, blocks  # noqa: E402
from pipeline import learned, fixes  # noqa: E402
from pipeline.decode import H, A, D  # noqa: E402

# Params
FEATS = ("spread", "cx")
SHIFTS = (2, 4, 6)
DIFFS = (2, 4, 6)
WEIGHTS = (0.25, 0.5, 1.0)
SWITCH = 10.0
FOUND_S = 5.0
LONG_S = 10.0


def v27_codes(hand, prob, f, blk, extra=None):
    """v27's decoder (touch-margin vote w 0.5 / 2 s, restart handover 3 s / 2 m/s), plus an extra
    home-positive vote."""
    vote = 0.5 * fixes.margin_vote(f, vote_s=2.0)
    if extra is not None:
        vote = vote + extra
    codes = learned.decode_hybrid(hand, fixes.add_vote(prob, vote, 1.0), blk, SWITCH)
    return fixes.restart_handover(codes, f, max_s=3.0, slow_ms=2.0)


def timing(lab, pred, fps=25):
    """(true direct changes, found, signed offsets s, false switches, long-possession seconds)."""
    t_idx = np.flatnonzero(np.isin(lab[1:], (H, A)) & np.isin(lab[:-1], (H, A))
                           & (lab[1:] != lab[:-1])) + 1
    p_idx = np.flatnonzero(np.isin(pred[1:], (H, A)) & np.isin(pred[:-1], (H, A))
                           & (pred[1:] != pred[:-1])) + 1
    offs = []
    for t in t_idx:
        near = p_idx[(np.abs(p_idx - t) <= FOUND_S * fps) & (pred[np.minimum(p_idx, len(pred) - 1)] == lab[t])]
        if len(near):
            offs.append((near[np.argmin(np.abs(near - t))] - t) / float(fps))
    false, long_frames = 0, 0
    edges = np.flatnonzero(np.diff(lab)) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [len(lab)]))
    for a, b in zip(starts, stops):
        if lab[a] in (H, A) and (b - a) >= LONG_S * fps:
            long_frames += b - a
            inner = p_idx[(p_idx > a + fps) & (p_idx < b - fps)]
            false += len(inner)
    return len(t_idx), offs, false, long_frames


def signal(cols, feat, mode, k):
    home = np.asarray(cols["r_{}_home{}".format(feat, "" if mode == "shift" else "_d{}".format(k))],
                      dtype=np.float64)
    away = np.asarray(cols["r_{}_away{}".format(feat, "" if mode == "shift" else "_d{}".format(k))],
                      dtype=np.float64)
    x = home - away
    if mode == "shift":
        s = int(k * dd.FPS)
        x = np.concatenate((x[s:], np.full(s, np.nan)))
    sd = np.nanstd(x)
    return np.tanh(np.nan_to_num(x / (sd if sd > 0 else 1.0)))


def main():
    data = v25_inputs()
    ctx = {}
    for g in dd.GAMES:
        d = dd.load(g)
        ctx[g] = (match_features(g, test_seed(g)), blocks(d["period"]),
                  idt.player_cols_cached(g, d))

    def summary(name, preds):
        s, n_true, offs, false, longf = [], 0, [], 0, 0
        for g in dd.GAMES:
            lab = data[g][0]
            s.append(pm.evaluate(lab, preds[g], bootstrap_n=0)["spell_f1_mean"])
            nt, o, fa, lf = timing(lab, preds[g])
            n_true += nt
            offs += o
            false += fa
            longf += lf
        offs = np.array(offs)
        return s, "found {:.0f}%  median {:+.2f}s  within 1 s {:.0f}%  false/h in long {:.1f}".format(
            100.0 * len(offs) / max(n_true, 1), np.median(offs) if len(offs) else np.nan,
            100.0 * np.mean(np.abs(offs) <= 1.0) if len(offs) else 0.0,
            false / (longf / dd.FPS / 3600.0))

    ref, ref_t = summary("v27", {g: v27_codes(data[g][1], data[g][2], ctx[g][0], ctx[g][1])
                                 for g in dd.GAMES})
    print("v27 reference: {} | {:.3f} | {}".format(" ".join("{:.3f}".format(x) for x in ref),
                                                  np.mean(ref), ref_t), flush=True)
    for feat, (mode, ks) in itertools.product(FEATS, (("shift", SHIFTS), ("diff", DIFFS))):
        for k, w in itertools.product(ks, WEIGHTS):
            preds = {g: v27_codes(data[g][1], data[g][2], ctx[g][0], ctx[g][1],
                                  w * signal(ctx[g][2], feat, mode, k)) for g in dd.GAMES}
            s, t = summary("", preds)
            won = sum(x > y + 1e-9 for x, y in zip(s, ref))
            lost = sum(x < y - 1e-9 for x, y in zip(s, ref))
            print("{:<6} {:<5} {} s  w {:<4g} {} | {:.3f} | {:+.4f} | {}/{} | {}".format(
                feat, mode, k, w, " ".join("{:.3f}".format(x) for x in s), np.mean(s),
                np.mean(s) - np.mean(ref), won, lost, t), flush=True)


if __name__ == '__main__':
    main()
