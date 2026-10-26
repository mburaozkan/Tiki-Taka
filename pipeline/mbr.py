"""Minimum-Bayes-risk decoding of the team stage: pick the labelling with the best EXPECTED spellF1.

    from pipeline.mbr import mbr_hybrid
    codes = mbr_hybrid(hand_codes, prob, blocks, tau=0.1, sample_switch=10.0)

Viterbi returns the single most probable path, but spellF1 rewards whole possessions matched at
IoU 0.3-0.7, so the most probable path is not the one with the best expected score. Here, as in
learned.decode_hybrid, the hand path fixes D vs live; inside each live stretch the team stage is a
two-state HMM over log P(H), log P(A) (renormalised, scaled by tau: frames are strongly correlated,
so the raw product of per-frame probabilities is far too sure) with a per-change cost
sample_switch. Per block (half):
  1. N_SAMPLES team paths are drawn from that HMM's posterior (forward filtering, backward
     sampling), fixed seed;
  2. candidates: decode_hybrid at each of CAND_SWITCHES (the unscaled emissions, as deployed) plus
     the first N_CAND_SAMPLES samples;
  3. the candidate with the highest mean spellF1 against all the samples is returned.
spell_f1() is an exact, faster re-implementation of the scorer's spell matching
(evaluation/possession_metrics.spell_scores) for one block.
"""

import numpy as np

from pipeline.decode import H, A, D
from pipeline.learned import decode_hybrid

# Params
IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)
N_SAMPLES = 200
N_CAND_SAMPLES = 20
CAND_SWITCHES = (5.0, 10.0, 15.0, 20.0, 30.0)
SEED = 0


def spells(codes):
    """(starts, stops, teams) of the maximal H / A runs of one block (no X inside)."""
    edges = np.flatnonzero(np.diff(codes)) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [len(codes)]))
    k = codes[starts]
    keep = (k == H) | (k == A)
    return starts[keep], stops[keep], k[keep]


def spell_f1(gt, pred):
    """Mean F1 over IOUS of the greedy one-to-one same-team matching, as the official scorer."""
    gs, ge, gk = gt
    ps, pe, pk = pred
    if not len(gs) or not len(ps):
        return 0.0
    pairs = []
    j0 = 0
    for i in range(len(gs)):
        while j0 < len(ps) and pe[j0] <= gs[i]:
            j0 += 1
        j = j0
        while j < len(ps) and ps[j] < ge[i]:
            if pk[j] == gk[i]:
                inter = min(ge[i], pe[j]) - max(gs[i], ps[j])
                union = max(ge[i], pe[j]) - min(gs[i], ps[j])
                pairs.append((-inter / float(union), i, j))
            j += 1
    pairs.sort()
    f1s = []
    for th in IOUS:
        used_g, used_p, tp = set(), set(), 0
        for neg, i, j in pairs:
            if -neg < th:
                break
            if i in used_g or j in used_p:
                continue
            used_g.add(i)
            used_p.add(j)
            tp += 1
        p, r = tp / float(len(ps)), tp / float(len(gs))
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))


def _ffbs(lp, switch, n, rng):
    """n samples (n, T) of a two-state chain with log emissions lp (T, 2) and a change cost."""
    T = len(lp)
    stay, move = np.log1p(-np.exp(-switch) / (1 + np.exp(-switch))), -switch - np.log1p(np.exp(-switch))
    trans = np.array([[stay, move], [move, stay]])
    alpha = np.empty((T, 2))
    alpha[0] = lp[0] - np.logaddexp(lp[0, 0], lp[0, 1])
    for t in range(1, T):
        a = np.logaddexp(alpha[t - 1, 0] + trans[0], alpha[t - 1, 1] + trans[1]) + lp[t]
        alpha[t] = a - np.logaddexp(a[0], a[1])
    out = np.empty((n, T), dtype=np.int8)
    p1 = np.exp(alpha[-1, 1])
    out[:, -1] = rng.random(n) < p1
    u = rng.random((n, T))
    for t in range(T - 2, -1, -1):
        nxt = out[:, t + 1]
        # P(s_t = 1 | s_{t+1}) ∝ alpha_t(1) * trans[1, nxt]
        l1 = alpha[t, 1] + trans[1, nxt]
        l0 = alpha[t, 0] + trans[0, nxt]
        out[:, t] = u[:, t] < 1.0 / (1.0 + np.exp(l0 - l1))
    return out


def sample_paths(hand_codes, prob, a, b, tau, switch, n, rng, floor=1e-4):
    """(n, b - a) int8 codes: the hand path's D, sampled teams inside its live stretches."""
    base = hand_codes[a:b]
    out = np.repeat(base[None, :], n, axis=0)
    ph = np.clip(prob[a:b, H], floor, 1.0)
    pa = np.clip(prob[a:b, A], floor, 1.0)
    tot = ph + pa
    lp = tau * np.log(np.stack([ph / tot, pa / tot], axis=1))
    live = (base == H) | (base == A)
    edges = np.diff(np.concatenate(([0], live.astype(np.int8), [0])))
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        draw = _ffbs(lp[s:e], switch, n, rng)
        out[:, s:e] = np.where(draw == 1, A, H)
    return out


def mbr_hybrid(hand_codes, prob, blocks, tau, sample_switch, n_samples=N_SAMPLES,
               n_cand_samples=N_CAND_SAMPLES, cand_switches=CAND_SWITCHES, seed=SEED,
               return_info=False):
    """Codes like learned.decode_hybrid, the team stage chosen by expected spellF1."""
    rng = np.random.default_rng(seed)
    out = hand_codes.copy()
    cand_full = {sw: decode_hybrid(hand_codes, prob, blocks, sw) for sw in cand_switches}
    picks = []
    for a, b in blocks:
        samples = sample_paths(hand_codes, prob, a, b, tau, sample_switch, n_samples, rng)
        s_spells = [spells(s) for s in samples]
        cands = [("viterbi{:g}".format(sw), c[a:b]) for sw, c in cand_full.items()]
        cands += [("sample", samples[i]) for i in range(min(n_cand_samples, n_samples))]
        best = None
        for name, c in cands:
            cs = spells(c)
            u = np.mean([spell_f1(s, cs) for s in s_spells])
            if best is None or u > best[0]:
                best = (u, name, c)
        out[a:b] = best[2]
        picks.append(best[1])
    return (out, picks) if return_info else out
