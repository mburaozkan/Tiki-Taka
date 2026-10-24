"""Semi-Markov (explicit-duration) decoding over H / A / D.

    from pipeline.semimarkov import duration_priors, decode_hsmm

The Viterbi in pipeline/decode.py charges a fixed penalty per change of state, whatever the
length of the segment it opens: a 0.3 s possession costs as much as a 3 s one. Here every segment
is scored as a whole:

    score(segment of state j over frames [u, t)) = sum of the frame scores of j over [u, t)
                                                    + weight * log p_j(t - u)
                                                    - cost(previous state -> j)

with p_j the length distribution of state j's segments (possessions for H and A, stoppages for
D), estimated from labelled matches (duration_priors). Segments run up to MAX_S; a longer stretch
is a chain of same-state segments, joined at no cost, with a flat prior beyond MAX_S. Decoded per
block by dynamic programming over segment end points, O(frames x MAX_S x 3).
"""

import numpy as np

from pipeline.decode import H, A, D, X

# Params
FPS = 25
MAX_S = 30.0            # longest single segment; longer ones are chains (flat prior beyond)
SMOOTH_S = 0.2          # Gaussian smoothing of the empirical length histogram (s)
FLOOR = 1e-6            # probability floor for lengths never seen


def duration_priors(label_arrays, max_s=MAX_S, smooth_s=SMOOTH_S):
    """{"poss": logp, "dead": logp}: log length distributions over 1..max_s*FPS frames, from
    label arrays (H/A/D/X codes). Lengths above max_s count in the last bin."""
    dmax = int(max_s * FPS)
    hist = {"poss": np.zeros(dmax + 1), "dead": np.zeros(dmax + 1)}
    for lab in label_arrays:
        edges = np.flatnonzero(np.diff(lab) != 0) + 1
        starts = np.concatenate(([0], edges))
        stops = np.concatenate((edges, [len(lab)]))
        for a, b in zip(starts, stops):
            k = lab[a]
            if k == X:
                continue
            key = "dead" if k == D else "poss"
            hist[key][min(b - a, dmax)] += 1
    sigma = max(1, int(smooth_s * FPS))
    r = 3 * sigma
    w = np.exp(-0.5 * (np.arange(-r, r + 1) / float(sigma)) ** 2)
    out = {}
    for key, h in hist.items():
        tail = h[dmax]
        body = np.convolve(h[:dmax], w / w.sum(), mode="same")
        p = np.concatenate((body, [tail]))
        p[0] = 0.0
        p = p / p.sum()
        out[key] = np.log(np.maximum(p, FLOOR))
    return out


def decode_hsmm(scores, priors, costs, weight=1.0, max_s=MAX_S):
    """Best H/A/D path for one block. scores: (n, 3) per-frame log-evidence; costs: (3, 3)
    penalty matrix (from, to), diagonal ignored (same-state joins are free)."""
    n = len(scores)
    dmax = min(int(max_s * FPS), n)
    cum = np.vstack([np.zeros((1, 3)), np.cumsum(scores, axis=0)])        # (n+1, 3)
    logdur = np.full((3, dmax + 1), -np.inf)
    logdur[H, 1:] = weight * priors["poss"][1:dmax + 1]
    logdur[A, 1:] = logdur[H, 1:]
    logdur[D, 1:] = weight * priors["dead"][1:dmax + 1]
    c = costs.astype(np.float64).copy()
    np.fill_diagonal(c, 0.0)

    best = np.full((n + 1, 3), -np.inf)
    best[0] = 0.0
    entry = np.full((n + 1, 3), -np.inf)     # entry[u, j] = max_i best[u, i] - c[i, j]
    entry_from = np.zeros((n + 1, 3), dtype=np.int8)
    back_len = np.zeros((n + 1, 3), dtype=np.int32)
    back_prev = np.zeros((n + 1, 3), dtype=np.int8)
    d = np.arange(1, dmax + 1)

    def set_entry(u):
        for j in range(3):
            v = best[u] - c[:, j]
            i = int(np.argmax(v))
            entry[u, j], entry_from[u, j] = v[i], i
    set_entry(0)
    entry[0] = 0.0                            # any state may open the block for free
    for t in range(1, n + 1):
        k = min(dmax, t)
        u = t - d[:k]                         # segment starts
        for j in range(3):
            cand = entry[u, j] + (cum[t, j] - cum[u, j]) + logdur[j, 1:k + 1]
            m = int(np.argmax(cand))
            best[t, j] = cand[m]
            back_len[t, j] = m + 1
            back_prev[t, j] = entry_from[u[m], j]
        set_entry(t)

    path = np.empty(n, dtype=np.int8)
    t, j = n, int(np.argmax(best[n]))
    while t > 0:
        length = back_len[t, j]
        path[t - length:t] = j
        prev = back_prev[t, j]
        t -= length
        j = prev
    return path


def decode_blocks(scores, blocks, priors, costs, weight=1.0, n=None):
    """decode_hsmm per (first, stop) block; X elsewhere."""
    codes = np.full(n if n is not None else len(scores), X, dtype=np.int8)
    for a, b in blocks:
        codes[a:b] = decode_hsmm(scores[a:b], priors, costs, weight)
    return codes
