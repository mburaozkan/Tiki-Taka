"""Hand fixes on top of the hybrid decoder (JOURNAL.md, next steps C; experiment 5).

The hybrid (v25) takes D vs live from the hand path and the team from the learned probabilities,
so a team fix has to enter the learned team stage. Each team fix here is an extra log-odds vote for
home (+) / away (-) per frame, added like the players-only vote (in_domain_team.combine); the
restart handover changes the decoded path itself.

  restart_handover  a team spell right after a stoppage, shorter than max_s, with the players
                    slow, followed by the other team, is the ball handed to the taker: D
  out_vote          a stoppage whose first `look_s` seconds (or the second before it) has the
                    ball out: the restart goes to the team that did NOT touch it last; +/-1 over
                    the first `vote_s` of the next live stretch
  landing_vote      an aerial ball coming down: speed over the previous 0.5 s above fast_ms,
                    dropping below half of it within 0.4 s, a player within reach_m: +/-1 for
                    his team over the next `vote_s`
  margin_vote       every touch (features "touch") votes for its team over the next `vote_s`,
                    weighted by min(margin / 1.5 m, 1), margin = toucher's lead over the nearest
                    opponent
  run_vote          `min_n` picked touches in a row by one team, each within `max_gap_s` of the
                    previous: +/-1 from the first of them (so the change stays at the first touch)
                    until the other team's next touch, at most `hold_s` after the last one
                    (public #15, user 2026-10-28)
"""

import numpy as np

from pipeline.decode import H, A, D

FPS = 25


def add_vote(prob, vote, w, floor=1e-4):
    """(n, 3) probabilities with the team log-odds moved by w * vote (vote > 0: home)."""
    ph = np.clip(prob[:, H], floor, 1.0)
    pa = np.clip(prob[:, A], floor, 1.0)
    z = np.log(ph / pa) + w * vote
    out = prob.copy()
    live = ph + pa
    out[:, H] = live / (1.0 + np.exp(-z))
    out[:, A] = live - out[:, H]
    return out


def _runs(codes):
    edges = np.flatnonzero(np.diff(codes)) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [len(codes)]))
    return list(zip(starts, stops, codes[starts]))


def restart_handover(codes, f, max_s=4.0, slow_ms=1.5):
    """Codes with handover spells after stoppages turned into D."""
    out = codes.copy()
    runs = _runs(codes)
    pace = np.nan_to_num(np.asarray(f["player_speed"], dtype=np.float64), nan=slow_ms)
    for i in range(1, len(runs) - 1):
        a, b, k = runs[i]
        before, after = runs[i - 1][2], runs[i + 1][2]
        if k not in (H, A) or before != D or after not in (H, A) or after == k:
            continue
        if (b - a) <= max_s * FPS and np.mean(pace[a:b]) < slow_ms:
            out[a:b] = D
    return out


def out_vote(hand_codes, f, look_s=3.0, vote_s=5.0):
    """+1 / -1 votes for the team restarting after the ball went out."""
    n = len(hand_codes)
    vote = np.zeros(n)
    out = np.asarray(f["ball_out"]) > 0
    last = np.asarray(f["last_touch"], dtype=np.float64)
    for a, b, k in _runs(hand_codes):
        if k != D or b >= n or hand_codes[b] not in (H, A) or a == 0:
            continue
        lo = max(0, a - FPS)
        hi = min(b, a + int(look_s * FPS))
        idx = np.flatnonzero(out[lo:hi])
        if not len(idx):
            continue
        first_out = lo + idx[0]
        toucher = last[max(0, first_out - 1)]
        if toucher == 0:
            continue
        e = min(n, b + int(vote_s * FPS))
        vote[b:e] = -np.sign(toucher)
    return vote


def landing_vote(ball, d_home, d_away, pitch, fast_ms=12.0, reach_m=3.0, vote_s=2.0):
    """+1 / -1 votes after an aerial ball comes down near a player."""
    L, W = pitch
    x = ball[:, 0].astype(np.float64) * L
    y = ball[:, 1].astype(np.float64) * W
    n = len(x)
    v = np.full(n, np.nan)
    v[1:] = np.hypot(np.diff(x), np.diff(y)) * FPS
    v[v > 40] = np.nan
    k1, k2 = int(0.5 * FPS), int(0.4 * FPS)
    before = np.full(n, np.nan)
    after = np.full(n, np.nan)
    cs = np.concatenate(([0.0], np.cumsum(np.nan_to_num(v))))
    cn = np.concatenate(([0], np.cumsum(~np.isnan(v))))
    t = np.arange(n)
    a, b = np.clip(t - k1, 0, n), t
    with np.errstate(invalid="ignore", divide="ignore"):
        before = np.where(cn[b] - cn[a] > k1 // 2, (cs[b] - cs[a]) / np.maximum(cn[b] - cn[a], 1),
                          np.nan)
        a2, b2 = np.clip(t + 1, 0, n), np.clip(t + 1 + k2, 0, n)
        after = np.where(cn[b2] - cn[a2] > k2 // 2,
                         (cs[b2] - cs[a2]) / np.maximum(cn[b2] - cn[a2], 1), np.nan)
    dh = np.where(np.isnan(d_home), np.inf, d_home)
    da = np.where(np.isnan(d_away), np.inf, d_away)
    with np.errstate(invalid="ignore"):
        land = (before > fast_ms) & (after < before / 2) & (np.minimum(dh, da) <= reach_m)
    vote = np.zeros(n)
    last_end = -1
    for i in np.flatnonzero(land):
        if i <= last_end:
            continue
        e = min(n, i + int(vote_s * FPS))
        vote[i:e] = 1.0 if dh[i] < da[i] else -1.0
        last_end = e
    return vote


def margin_vote(f, vote_s=1.0, full_m=1.5):
    """Touch votes weighted by the toucher's margin over the nearest opponent."""
    touch = np.asarray(f["touch"], dtype=np.float64)
    dh = np.asarray(f["d_home"], dtype=np.float64)
    da = np.asarray(f["d_away"], dtype=np.float64)
    n = len(touch)
    vote = np.zeros(n)
    for i in np.flatnonzero(touch != 0):
        m = abs(dh[i] - da[i]) if not (np.isnan(dh[i]) or np.isnan(da[i])) else 0.0
        e = min(n, i + int(vote_s * FPS))
        vote[i:e] = np.sign(touch[i]) * min(m / full_m, 1.0)
    return vote


def run_vote(f, min_n=2, max_gap_s=3.0, hold_s=2.0, turnover=True):
    """+1 / -1 votes over runs of same-team touches. turnover: only runs that follow a touch by
    the other team (a change of possession, not more touches inside one)."""
    touch = np.asarray(f["touch"], dtype=np.float64)
    n = len(touch)
    idx = np.flatnonzero(touch != 0)
    side = np.sign(touch[idx])
    vote = np.zeros(n)
    i = 0
    while i < len(idx):
        j = i
        while (j + 1 < len(idx) and side[j + 1] == side[i]
               and idx[j + 1] - idx[j] <= max_gap_s * FPS):
            j += 1
        if j - i + 1 >= min_n and (not turnover or (i > 0 and side[i - 1] != side[i])):
            stop = min(n, idx[j] + int(hold_s * FPS))
            if j + 1 < len(idx):
                stop = min(stop, idx[j + 1])
            vote[idx[i]:stop] = side[i]
        i = j + 1
    return vote
