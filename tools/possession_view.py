"""Look at public-window possessions on screen: labels, what each stage believed, and the pitch.

    python tools/possession_view.py 9 12                  # true possessions #9 to #12 (+/- 5 s)
    python tools/possession_view.py 1 2 --pad 8
    python tools/possession_view.py 14 15 --sub evaluation/results/v30_d1n_add.csv   # dh1
    set MTAH_VARIANT=_kf  (PowerShell: $env:MTAH_VARIANT="_kf")  then as above: the keeper-fixed
                          tracking (pipeline/track_fix.py) and v30_kf.csv (build that first)

Always the submission's chain and params (v30 = v29 + touch-run vote + backdating, Params below);
with the default --sub it says so if the rebuild does not reproduce v30.csv.

For the user's eyes only: the window opens locally and nothing about the data is printed (the
match data must not reach a hosted LLM, JOURNAL.md 2026-10-23). Possessions are numbered as in
tools/spell_table.py, and the time axis is in seconds from the window start.

Left, sharing the time axis:
  labels     truth (with the possession numbers), v30 (or --sub), and the hand path (D vs live)
  team       P(home | live): first stage (v25 + touch-margin votes), second stage (v29's
             refiner, 5 seeds), their v29 blend (first + 0.5 x second, in log-odds), and v30's
             team input (that blend + the touch-run vote); purple squares = frames backdating moved
  touches    touch network P(home touch), P(away touch); the picked touches as ticks; the
             touch-run vote as a band at the bottom
  ball track the ball itself, no network: speed and velocity change over +/-3 frames (m/s; dashed:
             the 2 m/s a touch needs), and the old velocity-peak rule's touches as ticks (nearest
             player within 3 m at a peak; orange dots: every peak, with or without a player
             near). Our match has no true touches (Smart Tagging is phases)
  ball       nearest home / away player to the ball (m, up to 15; dashed: the rule's 3 m reach);
             grey bands = ball not seen
  reaction   per team: push toward goal and compactness change over the next 4 s (the second
             stage's strongest new evidence)
  (keepers)  with --keepers (default; --no-keepers hides it): whether each team's keeper is
             visible (pipeline/track_fix.keeper_status), as two strips under the labels
Right: the pitch at the cursor (home blue, away red, ball black; with --keepers the raw tracking,
and players the keeper rule ignores as grey crosses, so the other team's doubled keeper is not
shown as theirs). Click on any plot, drag the
slider, or use left / right (0.2 s; shift: 1 s) and space to play / pause.
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
import dfl_data as dd  # noqa: E402
import in_domain_team as idt  # noqa: E402
from eval_refine import stage1  # noqa: E402
from report import PUBLIC_GT, FPS  # noqa: E402
from pipeline import refine  # noqa: E402
import pipeline.features as features  # noqa: E402

# Params (the submission, v30 = v29 + touch-run vote + backdating; JOURNAL.md 2026-10-29)
SUB = os.path.join(ROOT, "evaluation", "results", "v30{}.csv".format(os.environ.get("MTAH_VARIANT", "")))
REFINE_TAG = "v29"
SEEDS = 5
ALPHA = 0.5             # v29's blend weight
VOTE = {"w": 2.0, "min_n": 2, "max_gap_s": 2.0, "hold_s": 4.0, "turnover": 1}   # v29_tr_a
BACKDATE = {"p_max": 0.7, "max_s": 9.0, "loose_m": 1.5}                          # v29_bd_b
COLORS = {pm.H: "#2b6cb0", pm.A: "#c53030", pm.D: "#a0a0a0", pm.X: "#ffffff"}


def signals():
    """Per-frame arrays over the whole match, rebuilt as tools/tune_refine.ours does for v29."""
    from games import GAME, half_video_blocks
    from atd_cache import load_arrays
    from pipeline.features import build_features, LENGTH, WIDTH
    from pipeline import touch_net
    with np.load(os.path.join(dd.DATA_DIR, "cache", "v25_ours_prob{}.npz".format(os.environ.get("MTAH_VARIANT", "")))) as c:
        hand, prob = c["hand"], c["prob"]
    f = build_features(events=[], touch_model=touch_net.load(os.path.join(ROOT, touch_net.MODEL_PATH)))
    blk = [(a, b + 1) for _, a, b in half_video_blocks(GAME)]
    groups = np.zeros(len(hand), dtype=np.int8)
    for i, (a, b) in enumerate(blk, start=1):
        groups[a:b] = i
    atd = load_arrays()
    cols = idt.player_columns(atd["players"], atd["team"], groups, (LENGTH, WIDTH))
    p1 = stage1(prob, f)
    bundles = [refine.load(os.path.join(ROOT, "models", "refine_{}_s{}.pt".format(REFINE_TAG, s)))
               for s in range(SEEDS)]
    refine.BIN = bundles[0][0].cfg["BIN"]      # v29 runs at bin 2; inputs() reads the module's
    xs = (refine.inputs(p1, hand, f, cols), refine.inputs(p1, hand, f, cols, True))
    p_ref = np.mean([refine.predict(b, xs[0], xs[1], len(hand)) for b in bundles], axis=0)

    def logit(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))

    import eval_touch_run as tr
    from pipeline import fixes
    ph1 = p1[:, 0] / np.maximum(p1[:, 0] + p1[:, 1], 1e-9)
    blend = 1.0 / (1.0 + np.exp(-(logit(ph1) + ALPHA * logit(p_ref))))
    vote = tr.run_vote(f, VOTE)
    p_sub = 1.0 / (1.0 + np.exp(-(logit(ph1) + ALPHA * logit(p_ref) + vote)))
    base = tr.decode(hand, p1, p_ref, f, blk, vote)
    codes = fixes.backdate(base, f, **BACKDATE)
    return {"hand": hand, "p1": ph1, "p_ref": p_ref, "blend": blend, "vote": vote, "p_sub": p_sub,
            "base": base, "codes": codes, "f": f, "cols": cols,
            "players": atd["players"], "team": atd["team"], "ball": atd["ball"],
            "pitch": (LENGTH, WIDTH)}


def ball_track(ball):
    """Ball speed and the velocity change over +/-TOUCH_K frames (m/s), as pipeline.features.touches
    sees them: the raw evidence of a touch, without the network."""
    n = len(ball)
    x = features.window_mean(ball[:, 0].astype(np.float64) * features.LENGTH, features.TOUCH_SMOOTH)
    y = features.window_mean(ball[:, 1].astype(np.float64) * features.WIDTH, features.TOUCH_SMOOTH)
    seen = ~np.isnan(ball[:, 0])
    x[~seen], y[~seen] = np.nan, np.nan
    vx, vy = np.full(n, np.nan), np.full(n, np.nan)
    vx[1:-1], vy[1:-1] = (x[2:] - x[:-2]) * FPS / 2.0, (y[2:] - y[:-2]) * FPS / 2.0
    fast = np.hypot(vx, vy) > features.TOUCH_MAX_MS
    vx[fast], vy[fast] = np.nan, np.nan
    k = features.TOUCH_K
    dv = np.full(n, np.nan)
    dv[k:-k] = np.hypot(vx[2 * k:] - vx[:-2 * k], vy[2 * k:] - vy[:-2 * k])
    return np.hypot(vx, vy), dv


def touch_peaks(dv):
    """Frames where the velocity change is a local maximum over +/-TOUCH_K and >= TOUCH_DV_MS (the
    rule's peak test in pipeline.features.touches, before it asks for a player within TOUCH_R_M)."""
    k = features.TOUCH_K
    ok = np.where(np.isnan(dv), -np.inf, dv)
    local = ok.copy()
    for shift in range(1, k + 1):
        np.maximum(local[shift:], ok[:-shift], out=local[shift:])
        np.maximum(local[:-shift], ok[shift:], out=local[:-shift])
    return (ok >= features.TOUCH_DV_MS) & (ok == local)


def strip(ax, codes, t, y, label, h=0.17):
    """One coloured label bar at height y."""
    edges = np.flatnonzero(np.diff(codes)) + 1
    for a, b in zip(np.concatenate(([0], edges)), np.concatenate((edges, [len(codes)]))):
        ax.axvspan(t[a], t[b - 1] + 1.0 / FPS, ymin=y, ymax=y + h, color=COLORS[int(codes[a])],
                   lw=0)
    ax.text(t[0], y + h / 2.0, label + " ", ha="right", va="center", transform=ax.get_xaxis_transform(),
            fontsize=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("first", type=int, help="first true possession (#, as in spell_table)")
    parser.add_argument("last", type=int, nargs="?", help="last one (default: first)")
    parser.add_argument("--pad", type=float, default=5.0, help="seconds shown either side")
    parser.add_argument("--sub", default=SUB, help="submission to compare with the truth")
    parser.add_argument("--keepers", action=argparse.BooleanOptionalAction, default=True,
                        help="keeper visible per team (pipeline/track_fix.keeper_status) as two "
                             "strips; on the pitch the raw tracking, the players that rule ignores "
                             "(the other team's doubled or swapped keeper) drawn as grey crosses")
    args = parser.parse_args()
    last = args.last or args.first

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    runs = pm.valid_runs(gt)
    spells = pm.possession_spells(gt, runs)
    if not 1 <= args.first <= last <= len(spells):
        sys.exit("possessions are numbered 1..{}".format(len(spells)))
    sub = pm.label_codes(pm.read_label_csv(args.sub))
    s = signals()
    if args.sub == SUB:
        live = np.isin(s["hand"], (pm.H, pm.A))
        if not np.array_equal(s["codes"][live], sub[live]):
            print("note: rebuilding v30 from the params here does not reproduce", args.sub)

    a = max(spells[args.first - 1][0] - int(args.pad * FPS), 0)
    b = min(spells[last - 1][1] + int(args.pad * FPS), len(gt))
    run = next(r for r in runs if r[0] <= spells[args.first - 1][0] < r[1])
    a, b = max(a, run[0]), min(b, run[1])
    t = (np.arange(a, b) - a) / float(FPS)
    f = s["f"]

    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(6, 2, width_ratios=(2.2, 1), height_ratios=(1.1, 1.3, 1, 1, 1, 1),
                            left=0.07, right=0.98, top=0.95, bottom=0.12, hspace=0.25, wspace=0.12)
    axes = [fig.add_subplot(grid[0, 0])]
    axes += [fig.add_subplot(grid[i, 0], sharex=axes[0]) for i in range(1, 6)]
    pitch_ax = fig.add_subplot(grid[:, 1])
    fig.suptitle("public possessions #{}-#{}  ({} vs truth)".format(
        args.first, last, os.path.splitext(os.path.basename(args.sub))[0]))

    ax = axes[0]
    strip(ax, gt[a:b], t, 0.80, "truth")
    strip(ax, sub[a:b], t, 0.61, os.path.splitext(os.path.basename(args.sub))[0])
    hand = np.where(np.isin(s["hand"][a:b], (pm.H, pm.A)), pm.X, pm.D)
    strip(ax, hand, t, 0.42, "hand D")
    if args.keepers:
        from atd_cache import load_arrays
        from games import GAME, half_video_blocks
        from pipeline.track_fix import keeper_status
        raw = load_arrays(raw=True)
        blk = [(x, y + 1) for _, x, y in half_video_blocks(GAME)]
        mode = os.environ.get("MTAH_VARIANT", "")
        cases = None
        if mode in ("_ko", "_kx", "_kd"):   # the team's own "keeper" is the one ignored
            visible, ignored, cases = keeper_status(raw["players"], raw["team"], blk, s["pitch"],
                                                    own_keeper=True, deepest_only=mode == "_kd")
            if mode == "_ko":
                cases = None            # the real keeper keeps the other team's colour
        else:
            visible, ignored = keeper_status(raw["players"], raw["team"], blk, s["pitch"])
        strip(ax, np.where(visible[a:b, 0], pm.H, pm.X), t, 0.23, "H keeper")
        strip(ax, np.where(visible[a:b, 1], pm.A, pm.X), t, 0.04, "A keeper")
    for i, (sa, sb, k) in enumerate(spells):
        if sa < b and sb > a:
            ax.text(((max(sa, a) + min(sb, b)) / 2.0 - a) / FPS, 0.885, "#{}".format(i + 1),
                    ha="center", va="center", transform=ax.get_xaxis_transform(), fontsize=9,
                    color="white", weight="bold")
    ax.set_yticks([])

    ax = axes[1]
    ax.plot(t, s["p1"][a:b], color="#718096", lw=1, label="first stage")
    ax.plot(t, s["p_ref"][a:b], color="#d69e2e", lw=1, label="second stage")
    ax.plot(t, s["blend"][a:b], color="#a0aec0", lw=1, ls="--", label="v29 blend")
    ax.plot(t, s["p_sub"][a:b], color="black", lw=1.6, label="+ run vote (v30)")
    moved = np.flatnonzero(s["codes"][a:b] != s["base"][a:b])
    if len(moved):
        ax.scatter(t[moved], np.full(len(moved), 0.03), s=6, color="#805ad5", marker="s", lw=0,
                   label="backdated")
    ax.axhline(0.5, color="grey", lw=0.5, ls="--")
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(home | live)")
    ax.legend(loc="upper right", fontsize=7, ncol=5)

    ax = axes[2]
    ax.plot(t, f["touch_p_home"][a:b], color=COLORS[pm.H], lw=1, label="P(home touch)")
    ax.plot(t, f["touch_p_away"][a:b], color=COLORS[pm.A], lw=1, label="P(away touch)")
    touch = f["touch"][a:b]
    vote = s["vote"][a:b]
    for side, color in ((1.0, COLORS[pm.H]), (-1.0, COLORS[pm.A])):
        ax.vlines(t[touch == side], 0.92, 1.0, color=color, lw=2)
        ax.fill_between(t, 0, 0.06, where=np.sign(vote) == side, color=color, alpha=0.6, lw=0,
                        step="post")
    ax.text(0, 0.03, "run vote ", ha="right", va="center", fontsize=7,
            transform=ax.get_yaxis_transform())
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("touches")
    ax.legend(loc="upper left", fontsize=7, ncol=2)

    ax = axes[3]
    speed, dv = ball_track(s["ball"])
    rule = features.touches(s["ball"], f["d_home"], f["d_away"], f["gap_s"])[0][a:b]
    ax.plot(t, np.minimum(speed[a:b], 30), color="#4a5568", lw=1, label="ball speed (m/s)")
    ax.plot(t, np.minimum(dv[a:b], 30), color="#dd6b20", lw=1, label="velocity change (m/s)")
    ax.axhline(features.TOUCH_DV_MS, color="#dd6b20", lw=0.5, ls="--")
    for side, color in ((1.0, COLORS[pm.H]), (-1.0, COLORS[pm.A])):
        ax.vlines(t[rule == side], 26, 30, color=color, lw=2)
    peaks = touch_peaks(dv)[a:b]          # what the rule would call a touch if a player were near
    ax.scatter(t[peaks], np.minimum(dv[a:b][peaks], 30), s=12, color="#dd6b20", zorder=3,
               label="velocity peak")
    ax.set_ylim(0, 30)
    ax.set_ylabel("ball track")
    ax.legend(loc="upper left", fontsize=7, ncol=2)

    ax = axes[4]
    unseen = ~np.asarray(f["ball_seen"][a:b], dtype=bool)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], unseen.astype(np.int8), [0]))))
    for ga, gb in zip(edges[::2], edges[1::2]):
        ax.axvspan(t[ga], t[gb - 1], color="#e2e8f0", lw=0)
    ax.plot(t, np.minimum(f["d_home"][a:b], 15), color=COLORS[pm.H], lw=1, label="nearest home")
    ax.plot(t, np.minimum(f["d_away"][a:b], 15), color=COLORS[pm.A], lw=1, label="nearest away")
    ax.axhline(features.TOUCH_R_M, color="grey", lw=0.5, ls="--")
    ax.set_ylim(0, 15)
    ax.set_ylabel("to ball (m)")
    ax.legend(loc="upper right", fontsize=7, ncol=2)

    ax = axes[5]
    cols = s["cols"]
    for side, color in (("home", COLORS[pm.H]), ("away", COLORS[pm.A])):
        for name, ls in (("push", "-"), ("compact", ":")):
            ax.plot(t, cols["r_{}_{}_d4".format(name, side)][a:b], color=color, lw=1, ls=ls,
                    label="{} {} (+4 s)".format(side, name))
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_ylabel("reaction")
    ax.set_xlabel("seconds from window start")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    for ax in axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    cursors = [ax.axvline(0, color="black", lw=0.8) for ax in axes]

    length, width = s["pitch"]
    pitch_ax.set_xlim(0, length)
    pitch_ax.set_ylim(0, width)
    pitch_ax.set_aspect("equal")
    pitch_ax.add_patch(plt.Rectangle((0, 0), length, width, fill=False, color="#2f855a"))
    pitch_ax.axvline(length / 2.0, color="#2f855a", lw=0.8)
    pitch_ax.set_xticks([])
    pitch_ax.set_yticks([])
    home_dots = pitch_ax.scatter([], [], s=40, color=COLORS[pm.H])
    away_dots = pitch_ax.scatter([], [], s=40, color=COLORS[pm.A])
    ball_dot = pitch_ax.scatter([], [], s=25, color="black", zorder=3)
    ignored_dots = pitch_ax.scatter([], [], s=60, color="#718096", marker="x", zorder=4)
    info = pitch_ax.set_title("", fontsize=9)
    letters = {pm.H: "H", pm.A: "A", pm.D: "D", pm.X: "X"}
    team = np.asarray(s["team"])
    if args.keepers:                # draw the raw tracking; ignored players as grey crosses
        draw_players, team = raw["players"], np.asarray(raw["team"])
    else:
        draw_players, ignored, visible, cases = s["players"], None, None, None

    slider = Slider(fig.add_axes([0.07, 0.04, 0.6, 0.025]), "t (s)", 0, t[-1], valinit=0)

    def show(sec):
        i = a + int(round(sec * FPS))
        i = min(max(i, a), b - 1)
        xy = draw_players[i, :, :2] * np.array([length, width])
        skip = ignored[i] if ignored is not None else np.zeros(len(team), dtype=bool)
        side_now = np.where(cases[i], 1 - team, team) if cases is not None else team
        for dots, side in ((home_dots, 0), (away_dots, 1)):
            keep = (side_now == side) & ~skip & ~np.isnan(xy[:, 0])
            dots.set_offsets(xy[keep].reshape(-1, 2))
        ignored_dots.set_offsets(xy[skip & ~np.isnan(xy[:, 0])].reshape(-1, 2))
        ball = s["ball"][i, :2] * np.array([length, width])
        ball_dot.set_offsets(ball.reshape(1, 2) if not np.isnan(ball[0]) else np.empty((0, 2)))
        keepers = ("" if visible is None else "\nkeeper visible: H {}  A {}   (x = ignored)".format(
            "yes" if visible[i, 0] else "NO", "yes" if visible[i, 1] else "NO"))
        info.set_text("{:.1f} s   truth {}   ours {}   P(home) {:.2f}{}{}".format(
            sec, letters[int(gt[i])], letters[int(sub[i])], s["p_sub"][i],
            "" if not np.isnan(ball[0]) else "   NO BALL", keepers))
        for c in cursors:
            c.set_xdata([sec, sec])
        fig.canvas.draw_idle()

    slider.on_changed(show)
    timer = fig.canvas.new_timer(interval=80)
    timer.add_callback(lambda: slider.set_val(min(slider.val + 0.08, t[-1])))
    playing = [False]

    def on_key(event):
        step = 1.0 if event.key and "shift" in event.key else 0.2
        if event.key in ("right", "shift+right"):
            slider.set_val(min(slider.val + step, t[-1]))
        elif event.key in ("left", "shift+left"):
            slider.set_val(max(slider.val - step, 0))
        elif event.key == " ":
            playing[0] = not playing[0]
            (timer.start if playing[0] else timer.stop)()

    def on_click(event):
        if event.inaxes in axes and event.xdata is not None:
            slider.set_val(min(max(event.xdata, 0), t[-1]))

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_click)
    show(0.0)
    if os.environ.get("POSSESSION_VIEW_NO_SHOW"):
        fig.canvas.draw()
        print("ok")
        return
    plt.show()


if __name__ == '__main__':
    main()
