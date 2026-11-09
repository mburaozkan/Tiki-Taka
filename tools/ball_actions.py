"""SoccerNet ball-action spotting (T-DEED, public checkpoints) on the broadcast video. Local only;
prints aggregates only.

    python tools/ball_actions.py clip                      # public window +-1 min -> data/cache/ball_actions/public_bc.mp4
    python tools/ball_actions.py run --model SoccerNetBall_challenge1
    $env:MTAH_VARIANT="_kt"; python tools/ball_actions.py check --model SoccerNetBall_challenge1

Mapping broadcast -> tactical: data/cache/broadcast_align.npz gives, per tactical frame, the matched
broadcast frame (formation alignment, about 40% of frames). Live broadcast runs 1:1 with the
tactical clock between cuts to replays, so an unaligned tactical frame takes the offset of the
nearest aligned frames before and after when those two agree within BRIDGE_TOL frames (no replay
in between); otherwise it stays unmapped. Broadcast frames with no tactical frame (replays) are
dropped.

T-DEED (github.com/arturxe2/T-DEED, cloned to external/T-DEED, checkpoints from its README's Google
Drive) runs on the clip at 796x448, stride 2, its own soft-NMS; events -> tactical frames ->
data/cache/ball_actions/<model>_public.npz.

check: events per class per live minute (public window), and per true public start (+-1 s, as in
spell_table's numbering): the strongest event's class and confidence, and whether the player nearest
the ball at that event belongs to the new team. Label-derived numbering and model outputs only.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "starter")]

OUT = os.path.join(ROOT, "data", "cache", "ball_actions")
TDEED = os.path.join(ROOT, "external", "T-DEED")
VIDEO = os.path.join(ROOT, "data", "broadcast", "BAR-ATH_broadcast-001.mp4")
FPS = 25
PAD_S = 60
BRIDGE_TOL = 5
SIZE = (796, 448)
CLASSES = ("PASS", "DRIVE", "HEADER", "HIGH PASS", "OUT", "CROSS", "THROW IN", "SHOT",
           "BALL PLAYER BLOCK", "PLAYER SUCCESSFUL TACKLE", "FREE KICK", "GOAL")


def public_spells():
    import possession_metrics as pm
    from report import PUBLIC_GT
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    return gt, pm.possession_spells(gt, pm.valid_runs(gt))


def bridged():
    """tactical frame -> broadcast frame (-1 unmapped), aligned frames plus bridged gaps."""
    m = np.load(os.path.join(ROOT, "data", "cache", "broadcast_align.npz"))["match"].astype(np.int64)
    n = len(m)
    t = np.arange(n)
    ok = m >= 0
    off = np.where(ok, m - t, 0)
    idx = np.flatnonzero(ok)
    out = m.copy()
    j = np.searchsorted(idx, t)                       # next aligned index
    has_prev, has_next = j > 0, j < len(idx)
    p = idx[np.clip(j - 1, 0, len(idx) - 1)]
    q = idx[np.clip(j, 0, len(idx) - 1)]
    agree = has_prev & has_next & (np.abs(off[p] - off[q]) <= BRIDGE_TOL)
    fill = ~ok & agree
    out[fill] = t[fill] + off[p[fill]]
    return out


def clip_cmd(args):
    import cv2
    os.makedirs(OUT, exist_ok=True)
    gt, sp = public_spells()
    t0, t1 = sp[0][0] - PAD_S * FPS, sp[-1][1] + PAD_S * FPS
    mp = bridged()
    b = mp[t0:t1]
    b0, b1 = int(b[b >= 0].min()), int(b[b >= 0].max()) + 1
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, b0)
    wr = cv2.VideoWriter(os.path.join(OUT, "public_bc.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    k = 0
    for _ in range(b1 - b0):
        ok, fr = cap.read()
        if not ok:
            break
        wr.write(cv2.resize(fr, SIZE))
        k += 1
    wr.release()
    np.savez(os.path.join(OUT, "public_bc_map.npz"), b0=b0, t0=t0, t1=t1)
    win = mp[sp[0][0]:sp[-1][1]]
    print("tactical window +-{} s: {} frames; mapped {:.0%} (aligned {:.0%}); clip {} broadcast frames "
          "({:.1f} min, {:.0%} of them have a tactical frame)".format(
              PAD_S, t1 - t0, np.mean(b >= 0), np.mean(np.load(os.path.join(ROOT, "data", "cache", "broadcast_align.npz"))["match"][t0:t1] >= 0),
              k, k / FPS / 60, len(np.unique(b[b >= 0])) / max(k, 1)))
    print("public window itself mapped {:.0%}".format(np.mean(win >= 0)))


def run_cmd(args):
    clip = os.path.join(OUT, "public_bc.mp4")
    cmd = [sys.executable, "inference.py", "--model", args.model, "--video_path", clip,
           "--frame_width", str(SIZE[0]), "--frame_height", str(SIZE[1]),
           "--inference_threshold", str(args.threshold)]
    subprocess.run(cmd, cwd=TDEED, check=True)
    with open(os.path.join(TDEED, "inference_output", "results_inference.json")) as fp:
        preds = json.load(fp)["predictions"]
    z = np.load(os.path.join(OUT, "public_bc_map.npz"))
    b0, t0, t1 = int(z["b0"]), int(z["t0"]), int(z["t1"])
    mp = bridged()
    inv = {}
    for t in range(t0, t1):
        if mp[t] >= 0:
            inv.setdefault(int(mp[t]), t)
    fr, lab, conf, drop = [], [], [], 0
    for e in preds:
        t = inv.get(b0 + int(e["frame"]))
        if t is None:
            drop += 1
            continue
        fr.append(t)
        lab.append(CLASSES.index(e["label"]))
        conf.append(e["confidence"])
    np.savez(os.path.join(OUT, "{}_public.npz".format(args.model)), frame=np.array(fr), label=np.array(lab),
             conf=np.array(conf, dtype=np.float32))
    print("events {} (threshold {}), mapped to tactical {}, dropped (replay / unmapped) {}".format(
        len(preds), args.threshold, len(fr), drop))


def check_cmd(args):
    from atd_cache import load_arrays
    import pipeline.features as features
    z = np.load(os.path.join(OUT, "{}_public.npz".format(args.model)))
    fr, lab, conf = z["frame"], z["label"], z["conf"]
    gt, sp = public_spells()
    a0, a1 = sp[0][0], sp[-1][1]
    live = np.isin(gt, (0, 1))
    mins = live[a0:a1].sum() / FPS / 60
    inwin = (fr >= a0) & (fr < a1)
    print("{}: public window, events with confidence >= {:.2f}, per live minute ({:.1f} live min)".format(
        args.model, args.min_conf, mins))
    keep = inwin & (conf >= args.min_conf)
    print("  " + ", ".join("{} {:.1f}".format(c, np.sum(keep & (lab == i)) / mins)
                           for i, c in enumerate(CLASSES) if np.any(keep & (lab == i))))
    atd = load_arrays("tactical")
    players, ball, team = atd["players"], atd["ball"], atd["team"]
    L, W = features.LENGTH, features.WIDTH

    def nearest_team(f):
        b = ball[f, :2]
        if not np.all(np.isfinite(b)):
            return None
        d = np.hypot((players[f, :, 0] - b[0]) * L, (players[f, :, 1] - b[1]) * W)
        if not np.any(np.isfinite(d)):
            return None
        return int(team[np.nanargmin(d)])

    print("per true start (+-1 s): strongest event class conf | nearest player's team at it (new / old / ?)")
    caught = 0
    for i, (a, b, k) in enumerate(sp):
        m = (np.abs(fr - a) <= FPS) & (conf >= args.min_conf)
        if not m.any():
            print("  #{:<3} none".format(i + 1))
            continue
        caught += 1
        j = np.flatnonzero(m)[np.argmax(conf[m])]
        nt = nearest_team(fr[j])
        side = "?" if nt is None else ("new" if nt == k else "old")
        print("  #{:<3} {:<24} {:.2f} | {}".format(i + 1, CLASSES[lab[j]], conf[j], side))
    print("starts with an event within +-1 s: {} of {}".format(caught, len(sp)))
    rng = np.random.default_rng(0)
    cand = np.flatnonzero(live[a0:a1]) + a0
    hit, mx = [], []
    for f in rng.choice(cand, 2000):
        m = (np.abs(fr - f) <= FPS) & (conf >= args.min_conf)
        hit.append(m.any())
        mx.append(conf[m].max() if m.any() else 0.0)
    st = [conf[(np.abs(fr - a) <= FPS) & (conf >= args.min_conf)].max(initial=0.0) for a, b, k in sp]
    print("chance (random live frames in the window): event within +-1 s {:.0%}; mean strongest conf "
          "{:.2f} (true starts {:.2f})".format(np.mean(hit), np.mean(mx), np.mean(st)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("clip", "run", "check"))
    ap.add_argument("--model", default="SoccerNetBall_challenge1")
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--min_conf", type=float, default=0.2)
    args = ap.parse_args()
    {"clip": clip_cmd, "run": run_cmd, "check": check_cmd}[args.cmd](args)


if __name__ == "__main__":
    main()
