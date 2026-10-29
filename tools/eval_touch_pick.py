"""Touch picking for quick exchanges (a touch with an other-team touch within 0.5 s, e.g. a shot
blocked straight away), DFL held-out touch_net folds, teacher touches within 0.2 s. Aggregate only.

    python tools/eval_touch_pick.py

tn.pick takes one peak of P(home)+P(away) per +/-3 frames and gives it one team, so an H->A
exchange becomes one touch. Compared: per-team peaks, and ours plus the rival team's peak.
"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools")); sys.path.insert(0, ROOT)
import dfl_data as dd
import train_touch as tt_
from pipeline import touch_net as tn
TOL, K = 5, int(0.5 * dd.FPS)

def team_pick(prob, th, half=3, other=0.0):
    """peaks of each team's P separately; other: also need that team's P > other x the rival's."""
    fs, ts = [], []
    for c in (1, 2):
        s = prob[:, c]
        loc = s.copy()
        for k in range(1, half + 1):
            np.maximum(loc[k:], s[:-k], out=loc[k:]); np.maximum(loc[:-k], s[k:], out=loc[:-k])
        f = np.flatnonzero((s >= th) & (s == loc))
        fs.append(f); ts.append(np.full(len(f), c - 1, np.int8))
    f = np.concatenate(fs); t = np.concatenate(ts); o = np.argsort(f, kind="stable")
    return f[o], t[o]

def combo(prob, th, th2):
    """ours at th, plus per-team peaks >= th2 of the team ours did not pick within 0.5 s."""
    pf, pt = tn.pick(prob, th)
    qf, qt = team_pick(prob, th2)
    add = [(f, t) for f, t in zip(qf, qt)
           if not np.any((np.abs(pf - f) <= K) & (pt == t)) and np.any(np.abs(pf - f) <= K)]
    if not add:
        return pf, pt
    f = np.concatenate([pf, [a for a, _ in add]]); t = np.concatenate([pt, [b for _, b in add]])
    o = np.argsort(f, kind="stable")
    return f[o], t[o].astype(np.int8)

methods = {"ours@0.7": lambda p: tn.pick(p, 0.7),
           "team@0.7": lambda p: team_pick(p, 0.7), "team@0.5": lambda p: team_pick(p, 0.5),
           "team@0.35": lambda p: team_pick(p, 0.35),
           "ours0.7+rival0.3": lambda p: combo(p, 0.7, 0.3),
           "ours0.7+rival0.2": lambda p: combo(p, 0.7, 0.2)}
acc = {m: [0, 0, 0, 0, 0] for m in methods}   # matched, n_pred, n_true, ex matched, ex n
for g in dd.GAMES:
    d, ball, tf, tt, valid = tt_.prepared(g, seed=tt_.TEST_SEED)
    b = tn.load(os.path.join(tt_.MODEL_DIR, "touch_net_{}.pt".format(g)))
    prob = tn.predict(b, tn.inputs(d["players"], ball, d["team"], dd.PITCH, path=len(b[1]) > tn.n_base_inputs()))
    keep = valid[tf]; tf, tt = tf[keep], tt[keep]
    ex = np.zeros(len(tf), bool)
    ex[1:] |= (tt[1:] != tt[:-1]) & (np.diff(tf) <= K); ex[:-1] |= (tt[1:] != tt[:-1]) & (np.diff(tf) <= K)
    for m, fn in methods.items():
        pf, pt = fn(prob); k = valid[pf]; pf, pt = pf[k], pt[k]
        _, mt, _ = tt_.match(pf, pt, tf, tt, TOL)
        a = acc[m]; a[0] += len(mt); a[1] += len(pf); a[2] += len(tf)
        a[3] += sum(i in mt for i in np.flatnonzero(ex)); a[4] += ex.sum()
    print("  done", g, flush=True)
print("\n{:<18} {:>5} {:>5} {:>5} {:>11}".format("pick", "P", "R", "F1", "exchange R"))
for m, a in acc.items():
    p, r = a[0] / a[1], a[0] / a[2]
    print("{:<18} {:>5.2f} {:>5.2f} {:>5.2f} {:>11.2f}".format(m, p, r, 2 * p * r / (p + r), a[3] / a[4]))
