"""Team reaction features: how each team's shape moves, now and in the seconds after.

    from pipeline.reaction import reaction_features
    r = reaction_features(players, team, groups, pitch)     # {name: (n,) float64}

Teams reorganise 1-8 s after a turnover (JOURNAL.md, v13: shape is 88% right on live frames but
votes for the old team around a change). We label offline, so frame t may look ahead: the
future-minus-past differences below show the reaction as opposite-sign changes for the two teams.

Per team (suffix _home / _away), smoothed over 1 s, with x signed toward the goal the team attacks
in that group (half), so the columns mean the same thing in both halves and for both teams:
  spread    log(std x * std y) of the team's players (m^2)
  cx        centroid x (m from halfway, + = toward the goal it attacks)
  push      centroid velocity toward that goal (m/s)
  back      second-deepest player (deepest outfield) x (m, same sign)
  compact   mean distance to the team centroid (m)
and for each, _d2 / _d4 / _d6: mean over [t, t+k] minus mean over [t-k, t], k in seconds.
NaN where a team has fewer than MIN_PLAYERS tracked. Attack direction per group: a team attacks
toward x = 1 if its mean centroid over the group sits in the left half.
"""

import numpy as np

# Params
FPS = 25
MIN_PLAYERS = 6
SMOOTH_S = 1.0
DIFF_S = (2, 4, 6)
NAMES = ("spread", "cx", "push", "back", "compact")
SIDES = ("home", "away")


def _mean_window(x, lo, hi):
    """Mean of x over t+lo .. t+hi-1, ignoring NaN (NaN if none), clipped to the array."""
    n = len(x)
    ok = ~np.isnan(x)
    cs = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    cn = np.concatenate(([0], np.cumsum(ok)))
    t = np.arange(n)
    a, b = np.clip(t + lo, 0, n), np.clip(t + hi, 0, n)
    cnt = cn[b] - cn[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (cs[b] - cs[a]) / np.maximum(cnt, 1), np.nan)


def _levels(px, py):
    """Unsigned per-frame levels for one team: (centroid x, spread, sorted x, compactness)."""
    count = np.sum(~np.isnan(px), axis=1)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cx, cy = np.nanmean(px, axis=1), np.nanmean(py, axis=1)
        area = np.nanstd(px, axis=1) * np.nanstd(py, axis=1)
        compact = np.nanmean(np.hypot(px - cx[:, None], py - cy[:, None]), axis=1)
    bad = count < MIN_PLAYERS
    with np.errstate(divide="ignore", invalid="ignore"):
        spread = np.where(bad | ~(area > 0), np.nan, np.log(area))
    cx = np.where(bad, np.nan, cx)
    compact = np.where(bad, np.nan, compact)
    return cx, spread, compact, bad


def reaction_features(players, team, groups, pitch):
    """{r_<name>_<side>[_d<k>]: (n,) float64}; see the module docstring."""
    L, W = pitch
    n = len(players)
    groups = np.asarray(groups)
    smooth = int(SMOOTH_S * FPS)
    out = {}
    for s, side in enumerate(SIDES):
        px = (players[:, team == s, 0].astype(np.float64) - 0.5) * L
        py = (players[:, team == s, 1].astype(np.float64) - 0.5) * W
        cx, spread, compact, bad = _levels(px, py)
        sign = np.zeros(n)
        for g in np.unique(groups):
            sel = groups == g
            m = np.nanmean(cx[sel]) if np.any(~np.isnan(cx[sel])) else 0.0
            sign[sel] = 1.0 if m < 0 else -1.0
        xs = px * sign[:, None]
        srt = np.sort(np.where(np.isnan(xs), np.inf, xs), axis=1)
        back = np.where(bad | ~np.isfinite(srt[:, 1]), np.nan, srt[:, 1])
        cxs = cx * sign
        levels = {"spread": spread, "cx": cxs, "back": back, "compact": compact}
        cols = {}
        for g in np.unique(groups):
            idx = np.flatnonzero(groups == g)
            a, b = idx[0], idx[-1] + 1
            for name, x in levels.items():
                cols.setdefault(name, np.full(n, np.nan))[a:b] = _mean_window(
                    x[a:b], -smooth // 2, smooth // 2 + 1)
            c = cols["cx"][a:b]
            cols.setdefault("push", np.full(n, np.nan))[a:b] = (
                _mean_window(c, 1, smooth + 1) - _mean_window(c, -smooth, 0)) / SMOOTH_S
            for name in NAMES:
                x = cols[name][a:b]
                for k in DIFF_S:
                    w = int(k * FPS)
                    cols.setdefault("{}_d{}".format(name, k), np.full(n, np.nan))[a:b] = (
                        _mean_window(x, 0, w + 1) - _mean_window(x, -w, 1))
        for key, x in cols.items():
            parts = key.split("_")
            out["r_{}_{}{}".format(parts[0], side, "_" + parts[1] if len(parts) > 1 else "")] = x
    return out


def swap_pairs():
    """(home column, away column) pairs, for learned.flip."""
    pairs = []
    for name in NAMES:
        for suf in [""] + ["_d{}".format(k) for k in DIFF_S]:
            pairs.append(("r_{}_home{}".format(name, suf), "r_{}_away{}".format(name, suf)))
    return pairs
