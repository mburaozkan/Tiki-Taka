"""Turn per-frame features into H/A/D with a three-state HMM decoded by Viterbi.

Per frame, every state gets a score (log-scale evidence, hand-set weights in PARAMS). Every change
of state pays a fixed penalty. Viterbi then finds the labelling with the best total score, which
is what makes possessions long and clean: a switch only happens when the opposing evidence over the
following stretch outweighs the switch penalty.

Two rules fall out of this rather than being coded:
  - a ball gap gives the teams no evidence, so the current possession holds through it;
  - a single flicker of the nearest player never pays for two switches.

Decoded inside each half separately. Frames outside both halves are written as X (never graded).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks  # noqa: E402

# Params
FPS = GAMES[GAME]["fps"]
H, A, D, X = 0, 1, 2, 3
LETTERS = np.array(["H", "A", "D", "X"])

PARAMS = {
    # Team evidence from the nearest player of each side, when the ball is seen.
    "w_team": 2.0,          # max per-frame score for "this side is closest"
    "team_scale_m": 2.0,    # distance difference (m) at which the evidence is ~76% of max
    # Control, not proximity: team evidence decays as exp(-nearest / control_m), so a player 6 m
    # away says almost nothing (v2 used 1 / (1 + d / 4), which still trusted 6 m at 40%).
    "control_m": 1.5,
    # (A ball-in-flight gate was tried in v3 and removed: no effect, Viterbi already ignores the
    # brief nearest-player flicker during a pass.)
    # Carrier evidence: the same rule on c_home/c_away, the distance a player STAYED within of the
    # ball over ~0.5 s. Off: at equal total weight (w_team 1 + w_carrier 1) it changed wrong-team
    # frames by under 0.5 points, so it carries no information the instant term lacks (v4).
    "w_carrier": 0.0,
    # Last touch: the side whose player last changed the ball's velocity (pass, shot, block,
    # interception), carried forward until the next touch. Proximity was a coin flip (57%) inside
    # true possessions under 5 s; this is evidence of a different kind (v5).
    "w_touch": 1.0,
    # Pressing is not winning: where the nearest player is on the side that did NOT touch last,
    # scale its team vote down by this much (an opponent closing in without touching it flipped v5).
    # Only the H/A split uses it; whether the ball is alive still comes from the undamped vote.
    "against_touch": 0.0,
    "carrier_m": 2.0,
    # Co-movement (m_home - m_away): the carrier moves with the ball. 89-96% right where it reads,
    # on 18-43% of live frames (public window). Adds to the team vote (v12).
    "w_comove": 1.0,        # 4 in v12 (tuned on the public window); 1 since v15 (Metrica)
    # Team shape (log spread ratio): the team in possession spreads out. 88% right on public live
    # frames, 95% with the ball missing. H/A split only, tanh-squashed at shape_scale (v13).
    "w_shape": 0.0,
    "shape_scale": 0.5,
    # Keeper hold (keeper_hold): the keeper caught it and the tracker lost the ball in his hands.
    # Team vote for his side, and the gap evidence is switched off while it lasts.
    "w_keeper": 3.0,        # firm: outweighs any dead evidence (1 in v9, 0 before)
    "w_tag": 0.5,           # Smart Tagging phase team vote
    # Guessed ball inside gaps (run.py --ball-fill): how much its nearest-player vote counts in
    # gap frames, for which team (side) and for alive vs dead (alive). No effect without a fill.
    "w_fill_side": 1.0,
    "w_fill_alive": 0.0,
    # Restart windows (tag_restart): SET PIECES start until the next attack/defence phase starts.
    "w_restart": 0.0,       # dead evidence inside the window
    "w_restart_team": 0.0,  # team vote for the set piece's tagged team, just after the window
    # A still ball among slow players is a restart being set up: the taker stands next to it, so
    # the nearest-player vote would call it possession. Scale the team vote down by this much.
    "still_damp": 0.7,
    # Dead-ball evidence. D score = sum of these - dead_bias.
    "dead_bias": 0.45,
    "w_gap": 0.8,           # ball missing, scaled by gap length up to gap_full_s
    "gap_full_s": 8.0,
    # A gap is dead evidence only as far as the players are slow: with everyone running, the
    # tracker has lost a ball in play. 0 in v6 (every gap counted fully; D 52.6%), 1 since v8.
    "gap_gate": 0.0,        # 1 in v8-v12; off since v15: it cost 0.016 spellF1 on Metrica
    "w_out": 1.5,           # ball outside the pitch
    "w_still": 0.4,         # ball (seen) slower than still_ms
    "still_ms": 2.5,
    "w_slow": 0.8,          # players slow, logistic in mean player speed
    "slow_ms": 1.5,
    "slow_width_ms": 0.3,
    "w_setpiece": 0.3,      # inside a SET PIECES tag
    # Ball in play: a seen ball moving faster than moving_ms is live play (a pass, a run), whatever
    # the players' pace. Adds to the alive score of both teams, and switches off the slow term.
    "w_moving": 0.0,
    "moving_ms": 5.0,
    "moving_width_ms": 1.0,
    "moving_gate": 0.0,     # power on (1 - slow): 0 = no gate, 1 = gated by player pace
    "w_alone": 1.0,         # ball seen with no player within alone_m: out, or waiting for a restart
    "alone_m": 8.0,
    "alone_width_m": 1.5,
    # Penalties per change of state, in the same units as the per-frame scores.
    "switch": 30.0,         # H <-> A (12 before touches; 20 in v5; 30 since v6 detects more touches)
    # Possession changes at a touch: within touch_switch_s of a touch by the team taking over, an
    # H <-> A change costs `switch - touch_switch` instead of `switch`. Short possessions that start
    # with a clean touch become affordable, and pressing without a touch stays expensive (v11).
    "touch_switch": 0.0,
    "touch_switch_s": 0.4,
    # Only clean touches earn it: the nearest opponent at least this much (m) farther than the
    # toucher. In the public window touches were right 93% with a 1.5 m margin, 43% under 0.5 m,
    # while the toucher's own distance made no difference (85-92%).
    "touch_margin_m": 1.5,
    # Boundary snap: after Viterbi, move each start of a possession (H or A after anything else)
    # to the nearest touch by that team within snap_s, never past the neighbouring boundaries.
    # The decoder finds the possession; the touch says when it began. 0 = off.
    "snap_s": 0.0,
    "to_dead": 25.0,        # H/A -> D
    "from_dead": 15.0,      # D -> H/A
}


def falling(x, centre, width):
    """Logistic step from 1 (x well below centre) to 0 (well above), overflow-safe."""
    return 1.0 / (1.0 + np.exp(np.clip((x - centre) / width, -50.0, 50.0)))


def side_evidence(home_m, away_m, scale_m, decay_m):
    """+ for home, - for away: which side is closer, faded by how close the closer one is.

    Also returns the closer distance (0 where neither side has a value) and the no-value mask.
    """
    # One side untracked near the ball: treat its distance as far, so the other side wins.
    far = 50.0
    both_nan = np.isnan(home_m) & np.isnan(away_m)
    dh = np.where(np.isnan(home_m), far, home_m)
    da = np.where(np.isnan(away_m), far, away_m)
    close = np.minimum(dh, da)
    score = np.tanh((da - dh) / scale_m) * np.exp(-close / decay_m)
    return np.where(both_nan, 0.0, score), np.where(both_nan, 0.0, close), both_nan


def emissions(f, p=PARAMS):
    """(n_frames, 3) scores for H, A, D."""
    n = len(f["ball_seen"])
    near, close, _ = side_evidence(f["d_home"].astype(np.float64),
                                          f["d_away"].astype(np.float64),
                                          p["team_scale_m"], p["control_m"])
    carrier, _, _ = side_evidence(f["c_home"].astype(np.float64), f["c_away"].astype(np.float64),
                                  p["team_scale_m"], p["carrier_m"])
    in_gap = f["ball_seen"] == 0
    near_alive = np.where(in_gap, p["w_fill_alive"] * near, near)
    near = np.where(in_gap, p["w_fill_side"] * near, near)
    gap = np.minimum(f["gap_s"] / p["gap_full_s"], 1.0)
    still = (f["ball_speed"] < p["still_ms"]) & (f["ball_seen"] > 0)  # NaN compares False
    ps = np.where(np.isnan(f["player_speed"]), p["slow_ms"], f["player_speed"])
    slow = falling(ps, p["slow_ms"], p["slow_width_ms"])
    bs = np.where(np.isnan(f["ball_speed"]), 0.0, f["ball_speed"])
    moving = (1.0 - falling(bs, p["moving_ms"], p["moving_width_ms"])) * (f["ball_seen"] > 0)
    moving = moving * (1.0 - f["ball_out"])
    # Gated by the players: during a stoppage the ball also moves (passed back to the taker,
    # kicked away) while everyone walks, so a moving ball only counts when the players move too.
    gap = gap * slow ** p["gap_gate"] * (1.0 - (p["w_keeper"] > 0) * np.abs(f["keeper_hold"]))
    moving = moving * (1.0 - slow) ** p["moving_gate"]
    slow = slow * (1.0 - (p["w_moving"] > 0) * moving)

    damp = 1.0 - p["still_damp"] * still * slow
    tags = (p["w_tag"] * f["tag_team"] + p["w_restart_team"] * f["tag_restart_team"]
            + p["w_keeper"] * f["keeper_hold"])
    team = (p["w_team"] * near_alive + p["w_carrier"] * carrier) * damp + tags
    oppose = (near * f["last_touch"]) < 0
    team_side = (p["w_team"] * near * (1.0 - p["against_touch"] * oppose)
                 + p["w_carrier"] * carrier) * damp + tags
    # Co-movement decides which team, never whether the ball is alive: a player walking the ball
    # to a restart spot moves with it too (adding it to `team` dropped macroF1 0.88 -> 0.85).
    team_side = team_side + p["w_comove"] * np.nan_to_num(f["m_home"] - f["m_away"])
    # Only with the ball missing: shape lags turnovers by seconds (as a general vote it merged
    # possessions: public 0.46-0.58), but it is 95% right in gaps, where nothing else reads.
    team_side = team_side + p["w_shape"] * (f["ball_seen"] == 0) * np.tanh(
        np.nan_to_num(f["shape"]) / p["shape_scale"])

    # No ball (gap) is handled by w_gap, not here: close is 0 there, so alone is 0.
    alone = 1.0 - falling(close, p["alone_m"], p["alone_width_m"])
    alone = np.where(f["ball_seen"] > 0, alone, 0.0)

    dead = (p["w_gap"] * gap + p["w_out"] * f["ball_out"] + p["w_still"] * still
            + p["w_slow"] * slow + p["w_setpiece"] * f["tag_setpiece"]
            + p["w_restart"] * f["tag_restart"]
            + p["w_alone"] * alone - p["dead_bias"])

    out = np.empty((n, 3))
    # Last touch decides WHICH team, not WHETHER the ball is alive. The alive score max(H, A) stays
    # |team| whatever the touch says; only the split between H and A uses it: H - A = 2 * side.
    # With w_touch 0 this is exactly H = team, A = -team.
    #   Adding it to both (v5_t1) carried it through stoppages and outvoted D: macroF1 0.86 -> 0.73.
    #   Lowering only the non-toucher (v5n) turned team disagreements into D: D share up to 59%.
    side = team_side + p["w_touch"] * f["last_touch"]
    alive = np.abs(team) + p["w_moving"] * moving
    out[:, H] = alive - 2.0 * np.maximum(0.0, -side)
    out[:, A] = alive - 2.0 * np.maximum(0.0, side)
    out[:, D] = dead
    return out


def transition_cost(p=PARAMS):
    """(3, 3) penalty matrix, from-state rows to to-state columns."""
    c = np.zeros((3, 3))
    c[H, A] = c[A, H] = p["switch"]
    c[H, D] = c[A, D] = p["to_dead"]
    c[D, H] = c[D, A] = p["from_dead"]
    return c


def touch_discount(f, p):
    """(n, 2) H <-> A switch discount per frame: [:, 0] for a change into H, [:, 1] into A."""
    n = len(f["touch"])
    out = np.zeros((n, 2))
    if p["touch_switch"] <= 0:
        return out
    k = int(round(p["touch_switch_s"] * FPS))
    dh = np.nan_to_num(f["d_home"], nan=np.inf)
    da = np.nan_to_num(f["d_away"], nan=np.inf)
    for col, side, margin in ((0, 1.0, da - dh), (1, -1.0, dh - da)):
        clean = margin >= p["touch_margin_m"]
        hit = ((f["touch"] == side) & clean).astype(np.float64)
        near = np.convolve(hit, np.ones(2 * k + 1), mode="same") > 0
        out[near, col] = p["touch_switch"]
    return out


def viterbi(scores, cost, discount=None):
    """Best state path maximising sum(scores) - sum(change penalties).

    discount: optional (n, 2) per-frame reduction of the A->H and H->A costs.
    """
    n = len(scores)
    back = np.empty((n, 3), dtype=np.int8)
    total = scores[0].copy()
    back[0] = np.arange(3)
    for t in range(1, n):
        c = cost
        if discount is not None and (discount[t, 0] or discount[t, 1]):
            c = cost.copy()
            c[A, H] -= discount[t, 0]
            c[H, A] -= discount[t, 1]
        cand = total[:, None] - c             # cand[from, to]
        best = np.argmax(cand, axis=0)
        back[t] = best
        total = cand[best, np.arange(3)] + scores[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(total))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def snap_to_touches(path, touch, p=PARAMS):
    """Copy of path with each possession start moved to the nearest touch by its team."""
    k = int(round(p["snap_s"] * FPS))
    if k <= 0:
        return path
    path = path.copy()
    changes = np.flatnonzero(path[1:] != path[:-1]) + 1
    prev = 0
    for n, i in enumerate(changes):
        nxt = changes[n + 1] if n + 1 < len(changes) else len(path)
        new = path[i]
        if new in (H, A):
            lo, hi = max(prev + 1, i - k), min(nxt - 1, i + k)
            hits = np.flatnonzero(touch[lo:hi + 1] == (1.0 if new == H else -1.0)) + lo
            if len(hits):
                j = int(hits[np.argmin(np.abs(hits - i))])
                if j < i:
                    path[j:i] = new
                elif j > i:
                    path[i:j] = path[i - 1]
                i = j
        prev = i
    return path


def decode_block(scores, f, p, first, stop):
    """Viterbi plus the boundary snap over frames first..stop-1 (one half or period)."""
    discount = touch_discount({k: v[first:stop] for k, v in f.items()}, p)
    path = viterbi(scores[first:stop], transition_cost(p), discount)
    return snap_to_touches(path, f["touch"][first:stop], p)


def decode(f, p=PARAMS, game=GAME):
    """Label codes (H/A/D inside the halves, X outside) for every video frame."""
    scores = emissions(f, p)
    codes = np.full(len(scores), X, dtype=np.int8)
    for _, first, last in half_video_blocks(game):
        codes[first:last + 1] = decode_block(scores, f, p, first, last + 1)
    return codes
