"""Every feature we have, on one time axis: the second window of possession_view.py --features.

For the user's eyes only (draws, prints nothing about the data). Sources on our match:
  data/cache/change_net/ours_c1dl.npz   the change net's 47 inputs (x) and c1dl's output (pr)
  data/cache/change_net/ours_c1dlbd.npz c1dlbd's output (ball net in the gaps + duel inputs)
  models/ball_net_all.pt                the ball net: filled ball and P(ball at a home player) in gaps
  tools/cn_duel.py, tools/turnover_signals.py   duel inputs and team-reaction signals, rebuilt here
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

HC, AC = "#2b6cb0", "#c53030"


def load(s):
    """per-frame arrays for the feature window (whole match)."""
    import short_net as S
    import change_net as C
    import cn_duel as CD
    import turnover_signals as TS
    import turnover_cls as T
    out = {}
    z = np.load(os.path.join(C.OUT, "ours_c1dl.npz"))
    names = S.SHARED + ["h_" + k for k in S.BASE_TEAM] + ["a_" + k for k in S.BASE_TEAM]
    x = z["x"].astype(np.float32)
    out["x"] = {nm: x[:, i] for i, nm in enumerate(names)}
    out["pr_c1dl"] = z["pr"].astype(np.float32)
    blk = [tuple(b) for b in z["blk"]]
    p = os.path.join(C.OUT, "ours_c1dlbd.npz")
    out["pr_c1dlbd"] = np.load(p)["pr"].astype(np.float32) if os.path.isfile(p) else None
    period = np.zeros(len(x), np.int8)
    for i, (a, b) in enumerate(blk, start=1):
        period[a:b] = i
    d = {"players": s["players"], "team": np.asarray(s["team"]), "period": period}
    sign = T.attack_sign(d, period)
    bx = out["x"]["bx"] * sign
    by = out["x"]["by"]
    unseen = out["x"]["seen"] < 0.5
    out["duel"] = CD.duel_channels(d["players"], d["team"], np.where(unseen, np.nan, bx),
                                   np.where(unseen, np.nan, by), s["pitch"])
    try:
        import ball_net as N
        import dfl_data as dd
        ball = dd.blur_ball(s["ball"])
        filled, ph = N.fill(N.load("all"), ball, d["players"], d["team"], blk, s["pitch"])
        out["fill"] = filled
        out["p_home_ball"] = ph
    except Exception as e:                       # no GPU / model: skip that panel
        out["fill"], out["p_home_ball"] = None, None
        out["fill_err"] = str(e)[:80]
    out["react"] = TS.signals(d["players"], d["team"], s["ball"], s["pitch"], sign)
    out["react_names"] = TS.sig_names()
    return out


def build(plt, F, a, b, t, strips):
    """the feature figure for frames [a, b); strips(ax) draws the label strips. Returns (fig, axes)."""
    x = F["x"]
    rows = 12
    fig, axes = plt.subplots(rows, 1, sharex=True, figsize=(15, 13),
                             gridspec_kw={"hspace": 0.08, "left": 0.08, "right": 0.88, "top": 0.97,
                                          "bottom": 0.04})
    fig.suptitle("every feature (cursor follows the main window; click to move it)", fontsize=10)
    sl = slice(a, b)

    def team_lines(ax, key, label, lim=None, ls="-"):
        ax.plot(t, x["h_" + key][sl], color=HC, lw=1, ls=ls, label="H " + label)
        ax.plot(t, x["a_" + key][sl], color=AC, lw=1, ls=ls, label="A " + label)
        if lim:
            ax.set_ylim(*lim)

    def legend(ax, ylabel):
        ax.set_ylabel(ylabel, fontsize=8, rotation=0, ha="right", va="center")
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=6, frameon=False)

    ax = axes[0]
    strips(ax)
    ax.set_yticks([])
    ax.set_ylabel("labels", fontsize=8, rotation=0, ha="right", va="center")

    ax = axes[1]
    for tag, ls in (("c1dl", "-"), ("c1dlbd", "--")):
        pr = F["pr_" + tag]
        if pr is None:
            continue
        ax.plot(t, pr[sl, 0], color=HC, lw=1, ls=ls, label="{} turn->H".format(tag))
        ax.plot(t, pr[sl, 1], color=AC, lw=1, ls=ls, label="{} turn->A".format(tag))
        ax.plot(t, pr[sl, 2], color="#718096", lw=0.8, ls=ls, label="{} stop".format(tag))
    ax.set_ylim(0, 1)
    legend(ax, "change\nheads")

    ax = axes[2]
    for tag, ls in (("c1dl", "-"), ("c1dlbd", "--")):
        pr = F["pr_" + tag]
        if pr is None:
            continue
        ax.plot(t, pr[sl, 5], color=HC, lw=1, ls=ls, label="{} P(H)".format(tag))
        ax.plot(t, pr[sl, 6], color=AC, lw=1, ls=ls, label="{} P(A)".format(tag))
        ax.plot(t, pr[sl, 7], color="#a0aec0", lw=0.8, ls=ls, label="{} P(D)".format(tag))
    ax.plot(t, x["h_p1"][sl], color="#2c5282", lw=0.6, ls=":", label="first stage P(H)")
    ax.set_ylim(0, 1)
    legend(ax, "class /\nstage 1")

    ax = axes[3]
    team_lines(ax, "d1", "nearest", (0, 15))
    team_lines(ax, "d2", "2nd", (0, 15), ls="--")
    team_lines(ax, "d3", "3rd", (0, 15), ls=":")
    unseen = x["seen"][sl] < 0.5
    edges = np.flatnonzero(np.diff(np.concatenate(([0], unseen.astype(np.int8), [0]))))
    for ga, gb in zip(edges[::2], edges[1::2]):
        ax.axvspan(t[ga], t[gb - 1], color="#edf2f7", lw=0, zorder=0)
    legend(ax, "to ball (m)\ngrey: unseen")

    ax = axes[4]
    team_lines(ax, "n2", "within 2 m")
    team_lines(ax, "n5", "within 5 m", ls="--")
    team_lines(ax, "n10", "within 10 m", ls=":")
    legend(ax, "players\nnear ball")

    ax = axes[5]
    team_lines(ax, "nspd", "nearest speed")
    team_lines(ax, "ntow", "approach", ls="--")
    ax.axhline(0, color="grey", lw=0.4)
    legend(ax, "nearest\nplayer m/s")

    ax = axes[6]
    du = F["duel"]
    ax.plot(t, np.minimum(du["h_ctl"][sl], 10), color=HC, lw=1, label="H control (|v-vball|)")
    ax.plot(t, np.minimum(du["a_ctl"][sl], 10), color=AC, lw=1, label="A control")
    ax2 = ax.twinx()
    ax2.plot(t, du["h_keep"][sl], color=HC, lw=1, ls="--", label="H keep (s)")
    ax2.plot(t, du["a_keep"][sl], color=AC, lw=1, ls="--", label="A keep (s)")
    ax2.set_ylim(0, 3.2)
    ax2.tick_params(labelsize=6)
    ax.set_ylim(0, 10)
    legend(ax, "duel")

    ax = axes[7]
    ax.plot(t, np.minimum(x["bspd"][sl], 30), color="#2d3748", lw=1, label="ball speed")
    ax.plot(t, np.minimum(x["bacc"][sl] / 10, 30), color="#dd6b20", lw=0.8, label="accel / 10")
    ax.plot(t, x["bz"][sl] * 3, color="#38a169", lw=0.8, label="height x 3")
    ax.plot(t, np.minimum(x["gap"][sl], 10) * 3, color="#a0aec0", lw=0.8, label="gap length x 3")
    ax.set_ylim(0, 30)
    legend(ax, "ball")

    ax = axes[8]
    if F.get("p_home_ball") is not None:
        ph = F["p_home_ball"][sl]
        ax.plot(t, ph, color="#6b46c1", lw=1.2, label="ball net P(ball at H) in gaps")
        ax.axhline(0.5, color="grey", lw=0.4)
        ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, "ball net not available: " + F.get("fill_err", ""), transform=ax.transAxes,
                ha="center", fontsize=7)
    ax.plot(t, x["dead"][sl], color="#a0a0a0", lw=0.8, label="dead head P")
    ax.plot(t, x["handD"][sl] * 0.95, color="#4a5568", lw=0.6, ls=":", label="hand D")
    ax.plot(t, x["out"][sl] * 0.9, color="#d69e2e", lw=0.6, label="ball out")
    ax.plot(t, x["hold"][sl] * 0.85, color="#319795", lw=0.6, label="keeper hold")
    legend(ax, "ball net /\ndead")

    ax = axes[9]
    team_lines(ax, "touch", "touch net P")
    team_lines(ax, "fc", "touch c", ls="--")
    team_lines(ax, "fm", "touch m", ls=":")
    legend(ax, "touch\nfeatures")

    ax = axes[10]
    team_lines(ax, "cx", "centroid x (to goal)")
    ax2 = ax.twinx()
    ax2.plot(t, x["h_spread"][sl], color=HC, lw=0.8, ls="--", label="H spread")
    ax2.plot(t, x["a_spread"][sl], color=AC, lw=0.8, ls="--", label="A spread")
    ax2.tick_params(labelsize=6)
    ax.plot(t, x["h_ref"][sl], color="#2c5282", lw=0.6, ls=":", label="refiner log-odds H")
    ax.plot(t, x["h_vote"][sl], color="#805ad5", lw=0.6, ls="-.", label="run vote H")
    legend(ax, "shape /\nrefiner")

    ax = axes[11]
    nm = F["react_names"]
    for key, col in (("flow2", "#2b6cb0"), ("press2", "#c05621"), ("depth2", "#2f855a"), ("ball2", "#4a5568")):
        ax.plot(t, F["react"][sl, nm.index(key)], color=col, lw=0.9, label=key + " (+ = to H)")
    ax.axhline(0, color="grey", lw=0.4)
    legend(ax, "team\nreaction")
    ax.set_xlabel("seconds from window start")
    for ax in axes:
        ax.tick_params(labelsize=6)
    return fig, list(axes)
