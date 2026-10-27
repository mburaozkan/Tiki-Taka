"""Look at public-window possessions on screen: labels, what each stage believed, and the pitch.

    python tools/possession_view.py 9 12                  # true possessions #9 to #12 (+/- 5 s)
    python tools/possession_view.py 1 2 --pad 8
    python tools/possession_view.py 14 15 --sub evaluation/results/v30_d1n_add.csv
    set MTAH_VARIANT=_kf  (PowerShell: $env:MTAH_VARIANT="_kf")  then as above: the keeper-fixed
                          tracking (pipeline/track_fix.py) and v29_kf.csv

For the user's eyes only: the window opens locally and nothing about the data is printed (the
match data must not reach a hosted LLM, JOURNAL.md 2026-10-23). Possessions are numbered as in
tools/spell_table.py, and the time axis is in seconds from the window start.

Left, sharing the time axis:
  labels     truth (with the possession numbers), v29 (or --sub), and the hand path (D vs live)
  team       P(home | live): first stage (v25 + touch-margin votes), second stage (v29's
             refiner, 5 seeds), and their v29 blend (first + 0.5 x second, in log-odds)
  touches    touch network P(home touch), P(away touch); the picked touches as ticks
  ball       nearest home / away player to the ball (m); grey bands = ball not seen
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

# Params
SUB = os.path.join(ROOT, "evaluation", "results", "v29{}.csv".format(os.environ.get("MTAH_VARIANT", "")))
REFINE_TAG = "v29"
SEEDS = 5
ALPHA = 0.5             # v29's blend weight
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
    xs = (refine.inputs(p1, hand, f, cols), refine.inputs(p1, hand, f, cols, True))
    bundles = [refine.load(os.path.join(ROOT, "models", "refine_{}_s{}.pt".format(REFINE_TAG, s)))
               for s in range(SEEDS)]
    p_ref = np.mean([refine.predict(b, xs[0], xs[1], len(hand)) for b in bundles], axis=0)

    def logit(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))

    ph1 = p1[:, 0] / np.maximum(p1[:, 0] + p1[:, 1], 1e-9)
    blend = 1.0 / (1.0 + np.exp(-(logit(ph1) + ALPHA * logit(p_ref))))
    return {"hand": hand, "p1": ph1, "p_ref": p_ref, "blend": blend, "f": f, "cols": cols,
            "players": atd["players"], "team": atd["team"], "ball": atd["ball"],
            "pitch": (LENGTH, WIDTH)}


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

    a = max(spells[args.first - 1][0] - int(args.pad * FPS), 0)
    b = min(spells[last - 1][1] + int(args.pad * FPS), len(gt))
    run = next(r for r in runs if r[0] <= spells[args.first - 1][0] < r[1])
    a, b = max(a, run[0]), min(b, run[1])
    t = (np.arange(a, b) - a) / float(FPS)
    f = s["f"]

    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(5, 2, width_ratios=(2.2, 1), height_ratios=(1.1, 1.3, 1, 1, 1),
                            left=0.07, right=0.98, top=0.95, bottom=0.12, hspace=0.25, wspace=0.12)
    axes = [fig.add_subplot(grid[0, 0])]
    axes += [fig.add_subplot(grid[i, 0], sharex=axes[0]) for i in range(1, 5)]
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
    ax.plot(t, s["blend"][a:b], color="black", lw=1.6, label="v29 blend")
    ax.axhline(0.5, color="grey", lw=0.5, ls="--")
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(home | live)")
    ax.legend(loc="upper right", fontsize=7, ncol=3)

    ax = axes[2]
    ax.plot(t, f["touch_p_home"][a:b], color=COLORS[pm.H], lw=1, label="P(home touch)")
    ax.plot(t, f["touch_p_away"][a:b], color=COLORS[pm.A], lw=1, label="P(away touch)")
    touch = f["touch"][a:b]
    for side, color in ((1.0, COLORS[pm.H]), (-1.0, COLORS[pm.A])):
        ax.vlines(t[touch == side], 0.92, 1.0, color=color, lw=2)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("touches")
    ax.legend(loc="upper left", fontsize=7, ncol=2)

    ax = axes[3]
    unseen = ~np.asarray(f["ball_seen"][a:b], dtype=bool)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], unseen.astype(np.int8), [0]))))
    for ga, gb in zip(edges[::2], edges[1::2]):
        ax.axvspan(t[ga], t[gb - 1], color="#e2e8f0", lw=0)
    ax.plot(t, np.minimum(f["d_home"][a:b], 30), color=COLORS[pm.H], lw=1, label="nearest home")
    ax.plot(t, np.minimum(f["d_away"][a:b], 30), color=COLORS[pm.A], lw=1, label="nearest away")
    ax.set_ylim(0, 30)
    ax.set_ylabel("to ball (m)")
    ax.legend(loc="upper right", fontsize=7, ncol=2)

    ax = axes[4]
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
        draw_players, ignored, visible = s["players"], None, None

    slider = Slider(fig.add_axes([0.07, 0.04, 0.6, 0.025]), "t (s)", 0, t[-1], valinit=0)

    def show(sec):
        i = a + int(round(sec * FPS))
        i = min(max(i, a), b - 1)
        xy = draw_players[i, :, :2] * np.array([length, width])
        skip = ignored[i] if ignored is not None else np.zeros(len(team), dtype=bool)
        for dots, side in ((home_dots, 0), (away_dots, 1)):
            keep = (team == side) & ~skip & ~np.isnan(xy[:, 0])
            dots.set_offsets(xy[keep].reshape(-1, 2))
        ignored_dots.set_offsets(xy[skip & ~np.isnan(xy[:, 0])].reshape(-1, 2))
        ball = s["ball"][i, :2] * np.array([length, width])
        ball_dot.set_offsets(ball.reshape(1, 2) if not np.isnan(ball[0]) else np.empty((0, 2)))
        keepers = ("" if visible is None else "\nkeeper visible: H {}  A {}   (x = ignored)".format(
            "yes" if visible[i, 0] else "NO", "yes" if visible[i, 1] else "NO"))
        info.set_text("{:.1f} s   truth {}   ours {}   P(home) {:.2f}{}{}".format(
            sec, letters[int(gt[i])], letters[int(sub[i])], s["blend"][i],
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
