# Team journal: More than a Hack 2027, Phase 1

Our working log. Update it at the end of each working day: what we did, what we learned, and what
comes next. Every number here is an aggregate. Never paste data rows, coordinates or per-frame
labels into this file, because this repository will become public.

---

## The project in one paragraph

Label every frame of the tactical video of FC Barcelona vs Athletic Club with `H` (home has the
ball), `A` (away has the ball), `D` (dead ball) or `X` (abstain). The file has 199,319 rows. It is
ranked on **spellF1**, which matches possessions as whole intervals, with **macroF1** (per frame)
as the tie-break. The baseline to beat is `smooth50` at **spellF1 0.2433** on the full match. We
only have ground truth for the first 5 minutes (17 possessions). The full-match ground truth is
private. **The deadline is November 17, 2026.**

## Ground rules we agreed

- **No match data goes to a cloud LLM, Claude included.** That means no video, tracking,
  tagging or homography, and no extracts of them. Claude writes the code, we run it locally, and
  we share only aggregate output (scores, counts, rates, distributions). Our `tools/` scripts are
  built to print only that kind of output.
- **Don't paste** notebook `.head()` tables, frame plots, or anything from
  `evaluation/reports/`, which lists transitions frame by frame.
- **The public 5 minutes are for checking correctness, never for choosing between models.** On
  those 5 minutes `smooth125` beats `smooth50`, but on the full match the order reverses. Use
  `--bootstrap 1000` before believing any gap.
- Labels must come from an algorithm, the repository must regenerate the exact submitted file, and
  `data/` must never be committed.

## How spellF1 behaves (key points)

- A possession is a maximal run of `H` or of `A`. Each flicker splits a possession and costs
  heavily.
- `X` is forward-filled with the previous label before spellF1 is computed. Abstaining inside a
  possession is free. Abstaining across a change of possession is expensive.
- Being 1 frame late is free, 8 frames late costs 0.89, and 1 second late costs 0.65.

---

## Our tooling

Every script prints aggregates only. Video renders go to `data/cache/` (gitignored) and are
watched locally, never shared.

| file | purpose |
|---|---|
| `tools/atd_cache.py` | parses the ATD once into `data/cache/*.npz`, indexed by video frame |
| `tools/data_report.py` | data quality: ball visibility and gaps, tracking counts, homography, tagging |
| `pipeline/run.py --name N [--set k=v] [--touch K=v] [--tag-sign code=±1]` | features → HMM/Viterbi → `evaluation/results/N.csv` (prints label shares) |
| `tools/report.py [--brief] <csvs>` | full-match plausibility (possessions, median, direct %, D %) plus public spellF1/macroF1 |
| `tools/paired_bootstrap.py A B` | paired bootstrap of the spellF1 difference on the public window |
| `tools/spell_table.py <csvs>` | one row per true public possession (#1–#17): IoU, start/end offsets, misses, spurious |
| `tools/boundary_diag.py <csv>` | each true public change: predicted vs evidence vs touch timing, and the cause |
| `tools/spell_diag.py [--profile]` | why possessions are missed (split, merged, shifted, too short) |
| `tools/dead_diag.py <csv>` | why D is too long: D-score terms per confusion, true stoppage offsets, restart lag |
| `tools/zone_diag.py <csv>` | D by pitch zone, keeper coverage, ball at a keeper, gaps by how the ball was last seen |
| `tools/tag_alignment.py [csv]` | how the Smart Tagging codes line up with the labels; set-piece profile |
| `tools/feature_table.py` | per-feature stats on true H/A/D frames (public window) |
| `tools/error_by_ball.py <csv>` | errors by ball seen / gap length / ball speed / nearest distance |
| `tools/gap_check.py`, `tools/gap_clips.py` | ball gaps vs dead stretches; random gap clips to judge by eye |
| `tools/render_labels.py <csv>` | video with predicted and true labels, possession number, ball circle, touches |
| `tools/render_compare.py A B [--everywhere] [--tally]` | blind LEFT/RIGHT clips where two versions disagree, plus an answer sheet |

Workflow: `python pipeline/run.py --name <name>`, then
`python tools/report.py --brief "evaluation/results/v*.csv"`.

---

## Model direction

1. **Per-frame evidence (features):** ball state and position, nearest player of each team,
   touches, player speed, ball out or still, gaps, keeper hold, Smart Tagging.
2. **Evidence → scores.** These are hand-set weights today. The plan is to learn them (see "Next
   steps": network models).
3. **HMM/Viterbi decoding over H/A/D**, which keeps possessions long and clean (spellF1 punishes
   flicker). This stays whatever produces the scores. A possible upgrade is a semi-Markov model
   with explicit possession lengths.

Networks feed the decoder; they don't replace it. With 17 labelled possessions, an end-to-end
network can't be trained or checked on this match alone, so any learned part needs labels from
outside: external open datasets, or self-supervision.

---

## Version history

Full match: possessions, direct H↔A changes, D share (target ranges ~230–300 possessions,
35–50% direct, 40–50% D). Public window: spellF1 / macroF1, used for diagnosis only. The rebuild
column gives the command for the current code.

| version | what changed | poss | direct | D | public spF1 / mF1 | rebuild |
|---|---|---|---|---|---|---|
| smooth50 | organisers' baseline: nearest player, missing ball = D, 50-frame smoothing | 594 | 64% | 32% | 0.33 / 0.62 | `starter/baseline.py` |
| v1 | HMM/Viterbi over nearest player, ball out/still, player speed, gaps, tags; switch 30 | 188 | 24% | 44% | 0.43 / – | older code |
| v2 | switch 18, `still_damp` for restarts | 175 | 23% | 50% | 0.44 / – | older code |
| v3 | control not proximity: team evidence fades with distance (`control_m` 1.5) | 245 | 36% | 50% | 0.42 / 0.86 | older code |
| v4 | carrier persistence (who stays with the ball) | – | – | – | no gain | switched off |
| v5 | **touch detection**: last touch decides which team; switch 20 | 278 | 42% | 51% | 0.50 / 0.82 | `--set switch=20 w_keeper=0 gap_gate=0 --touch TOUCH_DV_MS=3.0 TOUCH_R_M=2.0` |
| v6 | more sensitive touches (2 m/s, 3 m), switch 30 | 278 | 36% | 53% | 0.58 / 0.85 | `--set w_keeper=0 gap_gate=0` |
| v7 | full Smart Tagging: restart window, restart team, extra codes | 272 | 34% | 54% | 0.55 / 0.84 | trial, off (`w_restart`, `w_restart_team`) |
| v8 | gap evidence only while players are slow (`gap_gate` 1) | 289 | 39% | 47% | 0.60 / 0.88 | `--set w_keeper=0` |
| v9 | keeper hold, 8 s | 295 | 38% | 46% | 0.60 / 0.88 | `--set w_keeper=1 --touch KEEPER_HOLD_S=8.0` |
| v10 | firm keeper hold over the whole gap: a keeper holding the ball is never D | 294 | 41% | 40% | 0.60 / 0.88 | `--set w_comove=0` |
| v11 | cheap switching at clean touches | 235–443 | – | – | 0.51–0.61 | trial, off (`touch_switch`) |
| v12 | co-movement: the carrier moves with the ball (H/A split only), weight 4 | 303 | 41% | 40% | 0.62 / 0.89 | `--set gap_gate=1 w_comove=4` |
| v15 | chosen on Metrica: v6 + keeper hold + co-movement 1, no gap gate. Metrica 0.713 | 288 | 39% | 45% | 0.59 / 0.85 | `--set gap_gate=0 w_gap=0.8` |
| v19 | chosen on DFL with our gaps: v15 + gap gate + w_gap 0.4. DFL 0.567 (v15 0.514) | 294 | 42% | 37% | 0.61 / 0.84 | `python pipeline/run.py --name v19` |
| v20 | v19 + touch network (trained on DFL, threshold 0.7). DFL blur+gaps 0.567 (v19 0.511) | 277 | 42% | 37% | 0.64 / 0.87 | `python pipeline/run.py --name v20 --touch-net` |
| **v21** | **v20 decides D; learned GBM+GRU (trained on DFL) decides the team. DFL 0.579 (5/7)** | **261** | **40%** | **37%** | **0.66 / 0.88** | `python pipeline/run_learned.py --name v21 --source dfl --model blend --hybrid 10` |
| v29_tr_a | submission 2026-10-28 (morning): v29 + touch-run vote (two same-team touches after a turnover back that team, w 2, gap 2 s, hold 4 s). DFL 0.573 (+0.0024, 6/1) | 263 | 40% | 37.4% | 0.727 / 0.886 | v29's chain, then `python tools/eval_touch_run.py --ours "w=2 min_n=2 max_gap_s=2 hold_s=4 turnover=1" --name v29_tr_a` |
| v29_bd_b | submission 2026-10-28, renamed v30 on 2026-10-29: v29_tr_a + backdating (a change of team moves back to the other team's previous touch when that was weak (P < 0.7), within 9 s, nothing between, ball loose >= 1.5 m). DFL 0.5755 (+0.0024 vs v29_tr_a, 5/1; nested +0.0018) | 263 | 40% | 37.4% | 0.812 / 0.901 | v29's chain, then `python tools/eval_backdate.py --ours "p_max=0.7 max_s=9 loose_m=1.5" --name v29_bd_b` |
| v30 | submission 2026-10-29 to 2026-10-30: v29_bd_b under its own version number (byte-identical). The dead-ball head candidate that used to be called v30 is now **dh1** (`v30_d1n_add.csv`) | 263 | 40% | 37.4% | 0.812 / 0.901 | v29's chain, then `python tools/eval_backdate.py --ours "p_max=0.7 max_s=9 loose_m=1.5" --name v30` |
| **v31** | **submission (since 2026-10-30): v30 + dead-ball head (d1n, add-only) + remove gate (a D stretch with mean head P(dead) < 0.1 is made live) protected by the ball-out rule (never a stretch with the ball seen out of play). DFL 0.6295 (+0.054 vs v30, 7/0)**; checked on the full match by video | 238 | 40% | 48.0% | **0.812 / 0.917** | v29's chain (v25, v29 builds), then `python tools/eval_dead_v30.py --ours 0.5 0.5 25 --gate 0.1 0 --protect --name v31` (models/dead_d1n_s0..2.pt; rebuild byte-identical, ef2ea808…) |

Tried and rejected along the way:
- symmetric touch encodings (v5);
- `against_touch`, damping an opponent who is close but didn't touch (v6);
- moving-ball alive evidence, with and without the player gate (v8);
- the Smart Tagging hypothesis for late turnovers (`v5_notag`);
- v11: cheap switching at clean touches (`touch_switch`), and contested touches not updating the
  last touch (`TOUCH_MIN_MARGIN_M`).

What each version taught us:
- **v1–v3:** the decoder removes flicker. Team evidence must mean *control* (very close), not
  proximity.
- **v5–v6:** who touched it last beats who is nearest (81% vs 57% right in short possessions).
  The user's video review found the missed touches: dribbles and deflections.
- **v7:** the tags behave as designed across the match (live in restart windows 28% → 4%), but
  nothing can check them. Set-piece intervals start in the stoppage, and the next phase starts
  inside them.
- **v8:** 74% of our D was ball-tracking gaps. A gap with everyone running is play, not a
  stoppage.
- **v9–v10:** the keeper is tracked fine. The error was the ball lost in his hands after a catch.
  The tags give the keeper's team less reliably than positions (10/15 and 7/15).

---

## Log

### 2026-10-22: kickoff

- Read all the documentation. Understood the task, the metrics and the rules (above).
- Discussed the HMM/Viterbi approach vs a sequence model, and chose a hybrid (see Model direction).
- Toured `starter/`. Found that the baseline's big flaw is that a missing ball means `D`,
  so it splits possessions every time the ball is occluded.

### 2026-10-23: data arrived, baselines reproduced, first findings

- Unpacked the data. The alignment checks for the tracking and the tagging both passed.
- Rebuilt the baselines. **The scoreboard matches the README exactly** (`smooth50` public spellF1
  0.3256), so our setup is correct.
- Wrote `tools/` (report scripts and the ATD cache).

**Data findings (`data_report.py`):**
- Ball visible on 59.3% of graded frames. Gaps are long rather than flickery: half 1 has 69 gaps,
  median 11.6 s; half 2 has 81 gaps, median 9.8 s.
- About 10 players tracked per team per frame; tracks last a median of 30–41 s. Team-level
  features only.
- When visible, the ball track is clean: median speed about 5 m/s, jumps over 40 m/s on 0.1–0.4%
  of steps. The ball is outside the pitch on 2.7% / 5.7% of visible frames (half 1 / half 2).
- Broadcast: about 5.7 players per team, tracks last ~9 s, no sync to tactical. **Dropped for
  now.**
- Smart Tagging covers 81–84% of graded frames; 32 of 59 set pieces have team `N/A`. Durations
  look like fixed-length windows.

**Baseline findings (`report.py`, `smooth50`):**
- It claims about **594 possessions** over the full match (median under 5 s). Scaling the public
  window (17 possessions in 5 minutes) suggests the truth is roughly 350, so it probably
  over-counts about 1.5×. That estimate is rough and comes from a small sample.
- On the public window the biggest error is **dead ball called live (15.4% of frames)**, then wrong
  team (11.1%). 3 of the 17 true possessions get no matching prediction at all.
- 12 of the 20 true changes are direct H↔A, so most turnovers happen in open play.

**Hypothesis tested, and mostly rejected (`gap_check.py`):** "ball missing = dead ball".
- Ball missing on 33.6% of dead frames vs 6.5% of live frames. When the ball is missing, the frame
  is dead 56% of the time, against a base rate of about 20%.
- 2 of the 4 dead stretches have the ball visible throughout. 4 gaps fall inside live play
  (median 2.7 s, up to 9.3 s).
- Conclusion: a missing ball is **weak evidence** for D, not a detector. Dead-ball detection needs
  combined signals. Inside live play, a gap should mean **the current team keeps the ball**.
  (Only 4 dead stretches in the sample, so the offsets are not reliable.)

**Pipeline v1 written** (`pipeline/`), tested on synthetic data only so far:
- `features.py`: ball seen, nearest home/away distance to the ball, ball out of bounds, ball speed
  (1 s median), length of the gap a frame sits in, mean player speed (2 s mean), Smart Tagging
  set-piece flag and a team vote from the phase codes (attacking codes: +team, defending codes:
  the other team; the signs come from the code names and still need checking).
- `decode.py`: 3-state HMM/Viterbi per half. Hand-set per-frame scores, plus a penalty for every
  change (switch 30, to dead 25, from dead 15). A gap gives no team evidence, so possession holds
  through it.
- `run.py --name v1 [--set key=value ...]` writes `evaluation/results/<name>.csv`.
- `tools/feature_table.py`: per-feature stats on true H/A/D frames in the public window.

**Feature table (public window, direction checks only):**
- Mean player speed is the best dead-ball signal: median 1.20 m/s on D frames against
  2.04 (H) and 2.44 (A). The quartiles barely overlap.
- The nearest player gives a vote on 93.5% of live frames and is right 77.8% of the time.
  The Smart Tagging team vote gives one on 70.9% and is right 76.8%, so **the code signs are
  correct**.
- The set-piece tag is on for 45% of D frames vs 9% of live ones. Ball out is on for only 7% of D.
  The ball's median speed is 2.5 m/s on D frames vs 5.5–6.7 m/s in play.
- At restarts the ball sits still next to the taker, so the nearest-player vote outvotes D.
  Added `still_damp`, which weakens the team vote when the ball is still and the players are slow.

**v1 → v2** (`tools/report.py --brief` gives one line per file):

| run | possessions | median | direct changes | D | public spellF1 |
|---|---|---|---|---|---|
| smooth50 | 594 | 4.6 s | 64% | 32% | 0.33 |
| v1 (switch 30) | 188 | 13.0 s | 24% | 44% | 0.43 |
| v2 (switch 18) | 175 | 12.2 s | 23% | 50% | 0.44 |
| v2_sw12 | 213 | 9.9 s | 32% | 50% | 0.42 |

- The flicker is gone: no possessions under 1 s (the baseline had 55–69 per half).
- **I corrected the targets.** The ball is in play about 55 of the ~106 graded minutes, and the
  public window has about 4.2 possessions per live minute. That gives **~230–300 possessions**
  (not 350) and **D around 40–50%**. About 100–130 stoppages add two changes each, so the
  realistic direct-change share is **~35–50%**. The public window's 60% comes from an unusually
  live stretch.
- The home share is 65% against the baseline's 57%. That may be real (Barcelona usually has
  60–70% of the ball), so it is **not used as a check**.
- The switch penalty is the only setting that moves the possession count. `dead_bias` and
  `still_damp` move D by a few points and do little else. **v2 defaults: switch 12,
  dead_bias 0.45.** We are still short on possessions, and more tuning won't fix that: the team
  evidence (78% right per frame) can't separate a real turnover from noise.

**NN / vision discussion.** We decided against an end-to-end RNN: there are only 17 labelled
possessions, training on them would cost us our only check, and the output would still need HMM
decoding. Neural nets will feed evidence into the HMM instead:
(a) a self-supervised model that predicts the ball from the players (free labels on the ~94k
frames where the ball is seen);
(b) a YOLO ball detector trained on automatic labels (ATD ball projected through the homography),
run on the gap frames. GPU: RTX 5070 8 GB (PyTorch needs CUDA 12.8 or newer).

**Gap clip check** (`tools/gap_clips.py`: 20 random gaps of 1 s or more, judged by eye locally):
- Of 19 answered: ball visible 9, partly visible 5, hidden 2, off screen 3.
- **About 3/4 of gaps show the ball in the video**, so the ATD gap is mostly a tracker failure.
  **A vision ball detector is worth building.**
- The sample is small and long-gap-heavy (14 of the 19 gaps were 10 s or longer, none under 3 s).
- **The notes changed the conclusion.** The ball is visible, but most gaps are **stoppages**. The
  tracker drops the ball when it is held (keeper, throw-in, referee), standing still for a restart
  (free kick, corner, after a foul), out of play or swapped (ball change, goal celebration), and
  sometimes in the air on a pass. Live-play gaps with genuinely uncertain possession are the
  minority (about 3 of 16 notes).
- So a vision detector would mostly find dead or held balls, where the label is already clear.
  **Vision drops to "maybe later".** A long gap is a strong D signal, and where the ball reappears
  (touchline, corner, goal area) plus the nearest player there gives the restart type and team.
  That is tracking geometry, with no GPU needed.

**Where the errors are** (`tools/error_by_ball.py`, public window, v2):
- 99.7% of wrong frames have the ATD ball seen. Gaps of 3 s or more have 0% error. 19 of the 20
  true changes happen with the ball seen. **Metrica's ball tracking is not our bottleneck; reading
  the ball is.** (Caveat: the ball is seen on 88% of public-window frames vs 59% match-wide.)
- Wrong team: 35% when the ball is above 15 m/s (passes in flight), and 16–21% when the nearest
  player is 1.5–8 m away (about 2/3 of all wrong-team frames).
- Dead called alive: 45% when nobody is within 8 m (52% of those frames are truly dead).

**v3** (control, not proximity):

| run | possessions | median | direct changes | D | public spellF1 / macroF1 |
|---|---|---|---|---|---|
| v2 | 221 | 10.1 s | 35% | 48% | 0.42 / 0.78 |
| v3 (w_team 1) | 156 | 14.6 s | 19% | 52% | 0.46 / 0.85 |
| **v3_wt2** | **245** | 8.2 s | **36%** | 50% | 0.42 / 0.86 |
| v3_nocontrol | 274 | 7.2 s | 39% | 49% | 0.43 / 0.78 |

- **Control decay** `exp(-d/1.5 m)`: dead called alive on ball-seen frames dropped from 9.7% to
  1.6%. It also weakened the team evidence, so `w_team` went from 1 to 2 to get the turnovers back.
- **"Ball alone"** (nobody within 8 m → D): small gain, kept.
- **Flight gate** (no team evidence above 12 m/s): no effect, **removed**.
- v3 defaults: w_team 2, control_m 1.5, w_alone 1, switch 12. All full-match counts are inside the
  target ranges.
- Remaining: wrong team on ~16% of ball-seen frames, even with a player within 1.5 m (15%).
  Nearest-player evidence has hit its limit.

**v4: carrier persistence (tried, switched off).** `c_home`/`c_away` measure how close each team's
best player stayed to the ball over ±6 frames.
- Added on top of v3 (total team weight 4), the model over-segmented: 372 possessions, 26 vs 17 in
  the public window. **Lesson: changing evidence also changes its overall strength.** Compare at
  equal total weight, or the decoder's balance moves.
- At equal weight (w_team 1 + w_carrier 1): 227 possessions; wrong team 13.1% → 12.7% of
  ball-seen frames; total wrong frames 1106 → 1075. **No real gain, so `w_carrier` = 0.** The code
  stays.
- v4_mix had fewer wrong frames but a lower public spellF1 (0.42 → 0.37). spellF1 is decided by a
  few possessions crossing IoU thresholds (one possession ≈ 0.05 here), not by frame accuracy.
- `tools/spell_diag.py` was written to diagnose this. On the baselines: **6 of the 17 true
  possessions are under 5 s and both baselines match only 1 of them.** The most common miss is a
  "merged" prediction that runs too long.

**spellF1 diagnosis (v2/v3).** We matched **0 of the 6 short true possessions** (under 5 s, median
3.0 s). That is 35% of what's scored. A missed short turnover costs about 3 matches: the short
possession itself ("no same-team pred") plus both neighbours ("merged").
`spell_diag.py --profile` showed the evidence was missing, not the decoder: inside short
possessions the ball is seen 100% of the time, but the nearest player is right only 57% (carrier
55%) and Smart Tagging 30% (its phases lag behind turnovers).

**v5: touch detection.** A touch is a peak in the ball's velocity change (smoothed ±2 frames,
|v(t+3) − v(t−3)| ≥ 3 m/s) with a player within 2 m; the nearest player at that moment gets it.
`last_touch` carries the side forward and resets after a 3 s gap.
- Quality (public window): **15.7 touches per live minute**, touching side = true owner **89%**,
  `last_touch` right **81% inside short possessions** (vs 57% for the nearest player).
- Encoding mattered. Last touch must decide *which team*, not *whether alive*:
  - added to both sides, it carried through stoppages and outvoted D (macroF1 0.86 → 0.73);
  - lowering only the non-toucher turned team disagreements into D (D 59%);
  - final: alive score = |team evidence|, and the split between the teams uses team + touch.
    D stays at 50.5% at any touch weight.
- A higher switch penalty (12 → 20) cut the full-match count from 332 to 278 **with no change
  on the public window**, so the trimmed possessions were extras.

| run | possessions | direct changes | D | public spellF1 / macroF1 | matched at 0.5 | short | long |
|---|---|---|---|---|---|---|---|
| v3 | 245 | 36% | 50% | 0.42 / 0.86 | 7/17 | 0/6 | 4/6 |
| **v5** (w_touch 1, switch 20) | **278** | **42%** | **51%** | **0.50 / 0.82** | **12/17** | 2/6 | 6/6 |

- `tools/paired_bootstrap.py` (same resample for both models): v5 − smooth50 = +0.18
  [+0.07, +0.29]; v5 − v3 = +0.08 [−0.01, +0.18]. **Warning:** smooth125 − smooth50 also comes out
  "significant" (+0.10, 96%) on the public window, but it is worse on the full match. The public
  bootstrap can't see the other 85 minutes. **We trust v5 because of how it works**
  (81% vs 57% evidence; whole-possession wrong-team misses down from 5 to 1), and because its
  full-match counts are plausible.

---

**Per-possession view** (`tools/spell_table.py`: labels only, possessions numbered, relative
offsets). `tools/render_labels.py` draws the labels, ball and touches on the video, with the
possession number (`truth #k`).
- **v5's main failure: turnovers detected 6–7 s late.** The starts of #2, #4, #9 and #14 are
  +6.6 to +7.2 s late, and each late change costs both neighbours.
- A true dead stretch between #13 and #14 is called H for 13.9 s. #15 (3.5 s H before a stoppage)
  is called A.
- Smart Tagging is **not** the cause: without it the late starts are identical and everything
  else gets worse (public spellF1 0.44; D 56%).

---

**What the video showed (our own viewing of the v5 render, numbered possessions):**
- #2: the ball changes hands with a **header**. #4: the **keeper** takes it.
- #9 and #11: an opponent **runs close without touching** the ball, and we flip anyway (#9 flips
  2–3 times).
- Between #13 and #14: after a **foul**, a Barcelona player passes the ball to Athletic for the free
  kick, and we call H from that pass.
- Start of #14: Athletic **touches it several times**, but we only switch after a long pass.
- #15: Athletic plays it, it **deflects off a Barcelona player** out for a corner. The truth gives
  #15 to H, so the last deflection counts.
- In short: we miss real touches (dribbles, deflections) and react to pressing without contact.

**v6: two fixes tried (aggregates; each touch setting is a `--touch` override in `pipeline/run.py`).**
- (a) `against_touch`: damp the nearest-player vote when it disagrees with the last touch.
  0.5 / 0.8 / 1.0 → public 0.53 / 0.52 / 0.52 spellF1, but macroF1 drops to 0.79–0.81, and it does
  not fix #9 or #14. **Off (0).**
- (b) **More sensitive touches**: velocity change 3 → 2 m/s, radius 2 → 3 m. This fixes what the
  video showed: #9 start +7.2 → +1.2 s, #14 +6.6 → +1.2 s, #15 caught, and the H during the
  #13–14 stoppage is gone. But the full match jumps to 329 possessions (too many).
- (b) with the switch penalty raised 20 → 30 brings the count back to **278** (the same as v5):

| run | possessions | median | direct | D | public spellF1 / macroF1 | public poss |
|---|---|---|---|---|---|---|
| v5 | 278 | 7.5 s | 42% | 50.8% | 0.50 / 0.82 | 22/17 |
| v6 sensitive touches, switch 20 | 329 | 5.6 s | 48% | 51.3% | 0.54 / 0.88 | 23/17 |
| **v6** (switch 30) | **278** | 6.9 s | 36% | 52.6% | **0.58 / 0.85** | 17/17 |

- Paired bootstrap: v6 − v5 = +0.07 [+0.02, +0.13], better in 99% of draws.
- **Caveat:** we picked the switch penalty with the public window in view. The full-match count
  matching v5 is reassuring, but direct changes at 36% sit at the low edge, and D at 52.6% is just
  above our range. An honest out-of-match check (external data) is the real test.
- **Still wrong in v6:** #2 (header) and #4 (keeper) are as late as before, because touch logic
  can't see a ball in the air or one held in the hands. #10, #12 and #15 (under 4 s) are now missed
  entirely, since switch 30 makes short spells expensive. #6 now starts 9 s late (a false touch).

---

**Smart Tagging, used fully (v7 trial, `tools/tag_alignment.py`, aggregates only):**
- The phase codes come in mirrored pairs (BUILD UP = DEFENDING IN ATTACKING THIRD, and so on).
  Two of the attacking codes we already use agree with the true team only 54% (BALL IN FINAL
  THIRD) and 44% (BALL IN THE BOX) of the time in the public window. That's a small sample.
- SET PIECES intervals (59; median 35 s) start in the stoppage and run on into play. The next
  attack/defence phase starts inside them, a median of 26 s before the set piece ends. In 21 of 27
  cases, that phase gives the ball to the team tagged on the set piece.
- New features (weights 0 by default):
  - `tag_restart`: dead evidence from the set-piece start until the next phase starts. 57 windows,
    median 6.1 s, capped at 20 s.
  - `tag_restart_team`: a 5 s vote for the tagged team afterwards. Throw-ins and kick-offs are
    tagged N/A, so they get no team.
  - `--tag-sign` lets PLAYERS IN THE BOX vote too.
- **Full match (prediction-only checks):** inside the restart windows, v6 calls the ball alive
  28% of the time; v7 (w_restart 1, w_restart_team 1) calls it alive 4%. After the restart, v6
  agrees with the tagged team 78% of the time; v7 agrees 96%. So the features do what they were
  designed to do.
- **Score:** the public window holds only 1 set piece. There, v7 is equal or worse: 0.55 vs 0.58,
  because the window ran 2.3 s past the true restart at #1. Full-match D rises 52.6% → 53.8–55%.
  PLAYERS IN THE BOX makes no difference at either sign.
- **Verdict: unproven, kept off.** The public window can't judge this change, and Metrica has no
  Smart Tagging. To decide, watch v6 vs v7 at full-match set pieces.
- **Blind check built** (`tools/render_compare.py A B`, then `--tally`): 40 clips (8 min) where v6
  and v7 disagree near a restart window. Clips are shuffled, and LEFT/RIGHT is random per clip.
  Waiting for the user's verdicts.

---

**Too much D (v8; `tools/dead_diag.py`, aggregates only).**
- The user watched the v6/v7 clips: "they both stay D for so long". No per-clip answers were filled
  in. LEFT/RIGHT is random per clip, so "right feels better" says nothing about which version.
- **Diagnosis:** 74% of v6's full-match D is ball-tracking gaps. A gap adds up to +0.8 D evidence
  even while the players run. Restart lag: after the first kick inside a D stretch, v6 stays D for
  another 12 s (median); 81% of stretches run on for more than 3 s. In the public window, the live
  play called D has the ball moving at 7.5 m/s with slowish players (1.9 m/s).
- **Tried: "ball moving fast = in play" (`w_moving`).** It made things worse (public 0.49; 354
  possessions): the ball moves during stoppages too (passed back, kicked away), and two true
  stoppages ended 10–14 s early. Gated by player pace (`moving_gate`) it was public 0.55 and 312
  possessions. **Off.**
- **Adopted: gap evidence gated by player pace (`gap_gate` 1).** A gap counts as dead only as far as
  the players are slow.

| run | possessions | median | direct | D | public spellF1 / macroF1 | public D (true 19.6%) |
|---|---|---|---|---|---|---|
| v6 | 278 | 6.9 s | 36% | 52.6% | 0.58 / 0.85 | 24.6% |
| **v8** (gap_gate 1) | **289** | 7.4 s | **39%** | **47.2%** | **0.60 / 0.88** | 20.5% |

- All four true stoppages in the public window start and end within 1.3 s. Paired bootstrap v8 − v6:
  +0.02 [−0.04, +0.14] (64%), so the public window can't separate them. We adopt v8 because full-
  match D and direct changes are now inside our ranges, and because it is the fix for what the user
  saw.
- Still open: restart lag is 9.6 s median, and there are 2,161 s of D in gaps with slow players
  (keeper holding the ball? a slow build-up?). Blind clips are ready: `data/cache/compare/v6_vs_v8/`
  (60 clips, 13 min, `render_compare.py --everywhere`).

---

**Goal area and keeper (v9; `tools/zone_diag.py`, aggregates only).**
- The user saw "the keeper kicking, or even the ball near the six-yard box, makes it D".
- The keeper is not missing from the tracking: each team has a player within 16.5 m of a goal
  87–89% of the time, and with the ball in a six-yard box someone is within 3 m 70% of the time.
- The six-yard box is 71% D in the public truth (goal kicks are set up there; we predict 63%).
  Penalty area: true 28%, predicted 28%. So D near the goal is often right.
- **Gaps that start with the ball at a keeper in his box (a catch):** 16 gaps, 580 s, median 18.5 s,
  86% predicted D. The truth gives a keeper's catch to his team (#4), so this is the error.
- New feature `keeper_hold` (+1/−1): a gap starting with the ball within 3 m of a keeper in his box,
  with no ball out in the 5 s before (that would be a goal kick). For KEEPER_HOLD_S, it gives a team
  vote and turns off the gap evidence.
- 15 holds. 11 of 15 balls reappear within 17 m of that goal, and players walk (1.2 m/s after 8 s).
  So many of these turn into a short restart or a slow build-up, not a long punt.

| run | possessions | direct | D | D at keeper gaps | public |
|---|---|---|---|---|---|
| v8 | 289 | 39% | 47.2% | 86% | 0.60 / 0.88 |
| **v9** (w_keeper 1, hold 8 s) | 295 | 38% | 46.2% | 76% | 0.60 / 0.88 |
| v9, hold 15 s | 294 | 39% | 45.5% | 68% | same |
| v9, hold the whole gap | 293 | 40% | 42.7% | 38% | same |

- **Adopted: hold 8 s** (the laws allow 6 s in the hands). The public window has no keeper catches,
  so it can't judge this. How long to hold is open; the clips are in
  `data/cache/compare/v9_vs_v9_k1.0_hold60/` (10 clips, 5 min).

- **v10, at the user's call:** a keeper holding the ball is never D. The user suggested taking the
  team from the tags (the last defending team); we checked that first:
  - the last DEFENDING IN DEFENSIVE THIRD team matches the keeper's team (from positions) in 10/15;
  - the tag team vote during the hold matches in 7/15 (the phases switch late).
  - So the team comes from positions, and the hold is firm (w_keeper 3) over the whole gap.

| run | possessions | direct | D | during keeper holds (518 s) | public |
|---|---|---|---|---|---|
| v9 | 295 | 38% | 46.2% | D 76%, keeper's team 24% | 0.60 / 0.88 |
| **v10** | **294** | **41%** | **40.0%** | **D 0%, keeper's team 100%** | 0.60 / 0.88 |

- Risk: the hold runs until the ball reappears, so a stoppage after the keeper releases the ball,
  inside the same gap, is now called possession. Full-match D (40.0%) is at the bottom of our
  range. v9 is `--set w_keeper=1 --touch KEEPER_HOLD_S=8.0`.

### 2026-10-23 (later): every public change point, diagnosed (`tools/boundary_diag.py`)

For each of the 20 true changes in the public window, the tool lines up three timings relative to
the true change: our predicted change, when the evidence first favours the new label (held 1 s),
and the new team's first detected touch. Labels and relative seconds only.

- **v10: 9 of 20 on time (within 1 s).** By kind: turnovers 4/12, stoppages 3/4, restarts 2/4.
  **Turnovers are the problem.**
- The causes:
  - **Evidence late (a touch is missed):** #2 start +6.6 s (the header), #8 +2.1, #9 +1.2, and
    the end of #5, a stoppage +1.3 s inside a ball gap.
  - **Evidence early (pressing):** #12 −2.3 s, #13 −2.0, #4 −1.1. An opponent gets close before
    winning the ball. For #12, no touch by the new team is detected at all.
  - **Decoder holds on:** #3 (2.8 s) and #10 (0.8 s). The new team's touch is detected on time
    (−0.1 s, +0.5 s), but the possessions are too short to pay the switch cost twice.
- **Touch accuracy (all 116 public touches, about 29 per live minute):**
  - the toucher's own distance doesn't matter (85–92% right at every distance);
  - the margin over the nearest opponent decides it: right 43% under 0.5 m, 79% at 0.5–1.5 m,
    93% over 1.5 m.
- **Tried (v11): possession changes cheaply at a clean touch** (`touch_switch`, with a clean margin
  of 1.5 m), with the stoppage costs raised along with the switch. Every variant was the same or
  worse: public 0.51–0.61, 235–443 possessions. #3 and #10 didn't come back. The cheap switch gives
  false turnovers at the ~7% of wrong clean touches. Also tried: contested touches don't update
  the last touch (`TOUCH_MIN_MARGIN_M` 0.5) → 0.55. **Both off; v10 unchanged.**
  - Side finding: a switch cost above to_dead + from_dead (40) makes the decoder route turnovers
    through a fake stoppage (direct changes dropped to 0% at switch 60).
- **Conclusion:** the decoder isn't the bottleneck. Touch quality is. Detection is about 88% right
  and misses headers and weak touches, which is too unreliable to switch on and too sparse at
  real changes. This is direct evidence for next step B1, a learned touch/event detector trained
  on external event data.

**v12: co-movement (the user's idea).** The player who has the ball moves with it: same direction
and speed. A defender closing in is near the ball but moves differently.
- Feature `m_home`/`m_away`: for each team, the best player within 2.5 m of the ball, scored by
  exp(−|v_player − v_ball| / 2 m/s), averaged over ±0.5 s. It only reads with the ball moving
  ≥ 1.5 m/s, because a still ball matches every standing player.
- **Accuracy on public live frames:**

| team signal | coverage of live frames | right team | within 3 s of a turnover |
|---|---|---|---|
| nearest player | 94% | 78% | 95% coverage, 67% right |
| last touch | 89% | 86% | 90% coverage, 68% right |
| **co-movement** (margin ≥ 0.05 / 0.2 / 0.4) | 43% / 28% / 18% | **89% / 91% / 96%** | 17% coverage, 74% right |

- **As team evidence it hurt** (public 0.57–0.58, macroF1 0.85): a player walking the ball to a
  restart spot co-moves too, so it kept stoppages alive. **As H/A split only** (like the touch,
  never alive evidence):

| run | possessions | direct | D | public spF1 / mF1 | on-time changes |
|---|---|---|---|---|---|
| v10 | 294 | 41% | 40.0% | 0.60 / 0.88 | 9/20 |
| w_comove 1 / 2 | 296 / 297 | 41% | 40% | 0.61 / 0.60 | – |
| **v12** (w_comove 4) | **303** | 41% | 40.5% | **0.62 / 0.89** | **11/20** |

- What changed (`boundary_diag.py`): #3 start +6.2 → −0.1 s (the short possession is caught); #2
  +9.8 → +5.0; #14 start +1.2 → −0.6. The pressing cases #12 and #13 didn't change; the ball is
  probably too slow there for co-movement to read.
- Bootstrap v12 − v10: +0.02 [−0.03, +0.09] (68%). The weight was picked on the public window and
  303 possessions is just over our range, so check this on the external set. v10 is
  `--set w_comove=0`.

**v13 trial: team shape (`shape` = log of home spread / away spread; spread = std x · std y).**
- The team in possession spreads out. It's right on 88% of public live frames, and **95% with the
  ball missing**, without any ball data.
- As an H/A vote it **hurt** (w 0.5–2: public 0.46–0.58, merged possessions down to 222). Shape
  reorganises seconds after a turnover, so it votes for the old team exactly at the changes.
- Gated to ball gaps only: the public window is unchanged (0.62; it has few live gaps), and the
  full match goes to 309–313 possessions. Nothing can check that yet. **Off (`w_shape` 0);** keep it
  as an input for the learned model.
- Lesson: per-frame accuracy isn't the goal. What matters is evidence that is right **at the
  changes**, and every hand-made signal so far is weakest there.

### 2026-10-24: Metrica open data, the first out-of-match test

**Setup** (`tools/metrica_data.py`, `tools/run_external.py`):
- The 3 Metrica sample matches are in `data/external/metrica/` (public open data, uncommitted).
- They are loaded into the same arrays as our match: positions, speeds computed from positions,
  team per slot, period. Our features and decoder then run on them unchanged; Metrica has no Smart
  Tagging.
- Each match is scored with the competition's own `evaluate()` over the full match. Predictions
  and labels go to `data/external/results/`.
- **The ball is tracked 59–68% of the time in Metrica too** (ours: 59%), with 22 players per
  frame. So it looks like our data without any degrading.

**Labels from the events, calibrated to the competition's definition.** With every
recovery/ball-lost event starting a possession, Metrica had 5.8 possessions per live minute
(median 4.9 s) against the public window's 4.2 (7.6 s).
- Rule now: a possession is started by a controlled action (pass, carry, shot, set piece). A
  recovery counts from its own frame when the same team's next controlled action confirms it.
  D runs from ball out / foul / shot out / goal until the next controlled action.
- Result: 4.0 per live minute; under 2 s 14% (public 12%); under 5 s 34% (public 35%).
- Games 1–2 have no carry events, so they get fewer short possessions (about 200 per match;
  game 3 has 374).

**Results (spellF1 / macroF1, mean of the 3 full matches, about 770 possessions):**

| version | Metrica | our public window |
|---|---|---|
| v5 | 0.704 / 0.878 | 0.50 |
| v6 | 0.711 / 0.877 | 0.58 |
| v10 | 0.696 / 0.863 | 0.60 |
| v12 | **0.682** / 0.862 | **0.62** |
| **v15** = v6 + keeper hold + co-movement 1, no gap gate | **0.713 / 0.880** | 0.59 |

- **The public window ranked the versions wrongly.** v12 was the best there and is the worst on
  Metrica. The two changes picked with the public window in view don't generalise:
  - the gap gate (v8) costs 0.016;
  - co-movement at weight 4 (v12) costs about 0.015; weight 1 is neutral to slightly positive.
- v15 beats v12 in every match: +0.039, +0.037, +0.019.
- Switch cost on Metrica (with v15's features): 10 → 0.693, 15 → 0.706, 20 → 0.709, 30 → 0.713.
  Switch 30 stays.
- Before calibration (looser labels), the best switch was 10 and the order was the same: v5 > v6
  > v10 > v12.
- v15 still over-predicts D on Metrica: 45.7 / 43.4 / 33.5% vs 37.4 / 34.4 / 26.8%.
- **v15 is the new default.** On our match: 288 possessions, 39% direct, D 44.5%, public
  0.59 / 0.85. v12 is `--set gap_gate=1 w_comove=4`.
- **Honest estimate:** the hand-built model scores about 0.71 on full matches of similar data, not
  0.62 and not 0.75. Hand tuning has levelled off: v5 to v15 all land between 0.68 and 0.71.

### 2026-10-24 (later): learned models, trained on Metrica

**Setup:**
- LightGBM 4.7 and PyTorch 2.11 (CUDA 12.8, RTX 5070) installed into the `barcelona` venv
  with `uv pip`.
- `pipeline/learned.py`: 70 inputs per frame (our features, plus means over ±0.5/1/2 s before and
  after, and touch timing), home/away symmetric (`flip`). Gradient boosting gives P(H/A/D), and
  the same Viterbi decodes it.
- `pipeline/sequence.py`: a bidirectional GRU over 30 s stretches (12.5 fps). Early stopping on the
  last 15% of each training match.
- `tools/train_external.py [--model gbm|gru] [--stack]`: leave one Metrica match out.
- `pipeline/run_learned.py`: trains on all of Metrica and labels our match. The weights are saved
  in `data/cache/models/`, so reruns give the identical file.

**Labels made consistent across the 3 matches** (`tools/metrica_data.py`):
- Game 3 alone records carries, and that gave it twice the possessions (374 vs about 200).
- Now only event types all three record count (pass, shot, set piece). A recovery is confirmed by
  the same team's next pass/shot/set piece **or its next "ball lost"**.
- Possessions per live minute 5.1 / 4.5 / 4.9 (public 4.2), median 6.8 / 8.6 / 7.0 s (public 7.6).
- All Metrica scores below use these labels. The 0.713 from the morning was on the old labels and
  isn't comparable.

**Metrica leave-one-out (spellF1 / macroF1):**

| model | game 1 | game 2 | game 3 | mean |
|---|---|---|---|---|
| hand v15 (switch 20, the best hand setting) | 0.782 | 0.806 | 0.789 | 0.793 / 0.887 |
| GBM | 0.806 | 0.822 | 0.799 | 0.809 / 0.878 |
| GBM stacked on the hand model | | | | 0.806 / 0.879 |
| GRU | | | | 0.802 / **0.909** |
| **GBM + GRU 50/50** | **0.817** | **0.834** | **0.814** | **0.822** / 0.896 |

**But it doesn't transfer to our match** (v16 = the blend trained on all of Metrica):
- 414 possessions, D 28.8%, public 0.49 / 0.67 (hand v15: 0.59 / 0.85).
- The team decision transfers fine (true H → H 86%, true A → A 81%).
- **Dead balls don't:** 66% of the true stoppages in the public window are called live.
- **Why: Metrica's ball is not real tracking.**
  - In Metrica's stoppages the ball is present only 0–10% of the time; ours 66%. So "ball missing
    = dead" is almost a perfect rule there.
  - In live play its ball sits on the carrier (median nearest player 0.2 m; ours 1.7 m) and is very
    smooth (frame-to-frame acceleration median 0.8 m/s²; ours 5.7).
  - Its "59–68% ball seen" is mostly the dead time, not gaps in play.
  - The ball looks reconstructed from the events.
- Also checked: touches are rarer in our match (23 vs 33–38 per ball-seen minute), but removing
  the touch inputs doesn't fix the transfer (public macroF1 0.66 → 0.67, Metrica 0.81 → 0.70).
- **Consequence:** Metrica is a poor proxy for anything that involves the ball (dead balls, gaps,
  co-movement, touches). Today's v15 choices made on Metrica (gap gate off, co-movement 1) are
  doubtful too: in Metrica every gap really is dead. v15 stays the default for now; decide again
  on real tracking.

**Next: DFL/Sportec open data** (`kloppy.sportec.load_open_tracking_data`): 7 Bundesliga matches,
real optical tracking with the ball during stoppages, CC BY 4.0, frame-level ball status and
possession. Rerun the hand model and the learned models there, then check the transfer to our
match again.

### 2026-10-24 (evening): guessing the ball inside gaps (`pipeline/ball_fill.py`)

**What the gaps look like (full match, in the halves):** ball seen 59%. Apart from 17 tiny ones,
there are no gaps under 3 s (the provider probably fills short gaps itself). There are 133 gaps
of 3 s or more, and 44% of gap time is in gaps of 30 s or more. The public window has the ball
seen 88% of the time, so it says little about gaps.

**Simple guesses, no learning:** `hold` (last position), `extrapolate` (last velocity over 0.4 s,
slowing with a 2 s time constant, kept on the pitch), `interp` (straight line to where the ball
reappears). `run.py --ball-fill <method>`. The guess feeds only the nearest-player distances.
`w_fill_side` (1) and `w_fill_alive` (0) set how much it counts in gap frames.

**Fake gaps cut from the seen ball** (`tools/ball_fill_bench.py`, 1500 gaps, real lengths):

| gap | method | median error | team right | team right, first 2 s |
|---|---|---|---|---|
| 3–10 s | hold / extrapolate / interp | 14 / 12 / 7 m | 68 / **73** / 60% | 74 / **85** / 65% |
| 10–30 s | | 24 / 24 / 15 m | 64 / **68** / 57% | 76 / **85** / 69% |
| ≥ 30 s | | 33 / 34 / 23 m | 63 / **65** / 55% | 75 / **87** / 74% |

**On our match it changes almost nothing:**
- `v17_ext` changes 474 frames (0.24%), with 53 team flips. The public window is identical
  (0.588 / 0.854); full match 289 possessions, D 44.7%.
- With `w_fill_alive=1`: `v17_ext_alive` 296 possessions, D 43.2%; `v17_int_alive` D 41.3%.
  The public window is unchanged or slightly worse.
- **Why:** v15 calls 57.5% of gap frames D. In the rest, the HMM already carries the team from
  before the gap ("hold"), and the guess mostly agrees. A few seconds into a gap the guessed ball
  is 12–34 m off, so it can't say much more.
- **Lesson:** the open question in gaps is live vs dead, not which team. The public gap errors
  were all live play called D. The ball's position doesn't answer that; player motion (the gap
  gate) or real tracking from outside (DFL) might. Kept as an option; v15 stays the default.

**Retry without gap → D (`w_gap=0`), the guessed ball voting alive and team (v18):**
- `v18_nogap` (no fill), `v18_ext`, `v18_int`, `v18_hold`: full match 296–321 possessions,
  D 25–33%. The public window is the same for all of them: 0.571 / 0.778. Inside public gaps, 58%
  of frames are now wrong, nearly all true dead balls called live.
- With the gap off, the fill method makes no difference in the public window. What matters is
  whether a gap counts as dead, not where the ball is guessed.
- Middle way, `v18_ext_gate` (gap evidence only while players are slow, plus the fill): 312
  possessions, D 38.2%, public 0.611 / 0.884, gap error 5.9% (v15 15.9%). That is v12's gap gate
  again. Paired bootstrap vs v15: +0.023, 5–95% [−0.041, +0.135], so no clear difference.
- So a missing ball must still count as dead evidence. Whether gating it by player pace is right
  is the open question, and only real tracking (DFL) can settle it.

### 2026-10-24 (night): why spellF1 is low, and possession boundaries

**Why spellF1 (0.59) is low next to 83% right-team frames (v15, public):** spellF1 counts
possessions, not seconds. The 6 true possessions under 4 s are 6% of live time but 35% of the
score. 5 of 17 are missed completely (4 under 4 s, one of 8 s), and 6 sit at IoU 0.59–0.69,
just under the 0.7 threshold (F1 at IoU 0.7: 0.29). Each possession is worth about 0.06.

**Boundary diagnosis (v15):** in 7 of 20 true changes the evidence is on time but the decoder
switches late or early. The new team's first touch is within ~1 s of the true change in 13 of 16
cases.

**New tools:** `decode.snap_to_touches` (`snap_s`), shared by `decode()` and `run_external.py`
via `decode_block`; `tools/sweep.py` (settings on our match and Metrica in one run, with Metrica
recall by true possession length).

**Results (Metrica spellF1 mean; Metrica recall at IoU 0.5 for <4 s / 4–10 s / >10 s):**

| setting | ours: poss, direct | public | Metrica | recall |
|---|---|---|---|---|
| v15 (switch 30) | 288, 39% | 0.588 | 0.768 | 37 / 81 / 88% |
| switch 20 | 343, 51% | 0.550 | **0.793** | 53 / 83 / 89% |
| switch 15 | 382, 56% | 0.562 | 0.790 | 56 / 83 / 89% |
| touch_switch 20 | 356, 51% | 0.576 | 0.791 | 55 / 83 / 88% |
| snap 1–3 s | 288, 39% | 0.565–0.588 | 0.762–0.766 | |

- Snapping doesn't help: the bad boundaries are 6–12 s off, and the near ones are already close.
- A lower switch cost finds short possessions on Metrica (37% → 53–59%), but on our match it adds
  55–150 possessions.
- **Metrica can't judge boundary timing:** matched starts are a median 0.07 s from the truth,
  because its ball is rebuilt from the same events the labels come from. Touches there are exact
  by construction; ours come from real, noisier tracking. That is likely why a lower switch helps
  there and adds possessions here.
- **So boundary work needs DFL** (real tracking plus event timestamps) to tune the switch cost and
  touch thresholds, and to train a touch detector. v15 stays the default.

### 2026-10-24 (late): DFL open data, the realistic test set

**Repository:** work moved to our own repository (Tiki-Taka, branch `mirac`). It holds the
competition starter code (MIT) plus ours; data stays out (`MTAH_DATA_DIR` points at it).

**DFL / Sportec open data** (`tools/dfl_data.py`): 7 German league matches (IDSSE, Bassek et
al., Scientific Data 2025, CC BY 4.0).
- Real optical tracking at 25 fps, 22 players. The ball is tracked 100% of the time, dead frames
  included.
- Download from `ndownloader.figshare.com/files/<id>`; the springernature.figshare.com links
  kloppy uses return empty files to scripts. Parsed with our own streaming reader (about 12 s per
  match). The rules only forbid other possession data for *our* fixture.
- Every ball frame carries the provider's `BallStatus` (alive/dead) and `BallPossession`
  (1 home / 2 guest). That is the same kind of per-frame label the competition's ground truth
  comes from (CHALLENGE.md). Checked: possession 1 = home (nearest player agrees 86–94%), speed is
  in km/h.
- **The raw flags switch on any touch:** 6.5–11 possessions per live minute, median 3.0–4.7 s,
  25–37% under 2 s. Public window: 4.2, 7.6 s, 12%.
- **Controlled-possession filter:** a flagged possession counts only if its team makes a
  controlled action inside it, ±1 s. Controlled actions: a pass or cross, a shot, a ball action
  that isn't a clearance, a won tackle with `PossessionChange`, a keeper claim, or a restart.
  Otherwise it goes to the previous team. Result: 4.3–7.3 per live minute, median 5.3–9.4 s,
  8–22% under 2 s. Five matches look like the public window; the two Bundesliga 1 matches stay
  choppier.
- Event clocks are offset from the tracking by −0.1 to −1.6 s per match, with about ±0.6 s
  scatter per event. The offset is estimated per match from passes, using the ball speeding up.
  Exact touch times must come from the tracking, not the event clock.

**Gap model** (`tools/gap_model.py`): cuts gaps like ours into DFL's ball.
- A two-state chain over frames, with start and end chances per band of player pace and gaps of
  at least 3 s.
- Calibrated on DFL so that the missing-ball share per pace band matches ours: ours 81/42/11/10%,
  DFL 82/41/12/9%, 40% overall (ours 41%).
- In DFL, the simulated gaps cover 62% of stoppage time and 21% of live play.

**Results** (`tools/sweep.py --provider dfl [--degrade] [--labels labels]`, mean of 7 matches):

| setting | DFL full ball | DFL + our gaps | same, raw flags | Metrica | public |
|---|---|---|---|---|---|
| v15 | 0.623 / 0.882 | 0.514 / 0.794 | 0.504 | 0.768 | 0.588 / 0.854 |
| switch 20 | 0.599 | 0.502 | | 0.793 | 0.550 |
| gap gate on | 0.623 | 0.556 / 0.825 | 0.529 | | 0.611 / 0.883 |
| **gate, w_gap 0.4 (v19)** | 0.623 | **0.567 / 0.833** | **0.537** | 0.741 | 0.611 / 0.840 |
| gate, w_gap 0.2 | | 0.572 / 0.837 | | | 0.571 / 0.778 |
| gate + extrapolated ball | | 0.551 | | | 0.629 |

- **DFL behaves like our match** (v15 0.62 with the full ball against public 0.59); Metrica
  (0.77) was inflated by its rebuilt ball.
- **Ball gaps cost about 0.11 spellF1** (0.623 → 0.514).
- **The gap gate wins in 7/7 matches under both label definitions**, +0.042. It was turned off
  in v15 on Metrica's advice; Metrica's ball is missing only in stoppages. w_gap 0.4 adds +0.011
  (7/7). 0.2 loses our public stoppages (macroF1 0.78), so we stop at 0.4.
- Switch 20: better on raw flags, worse on controlled labels. It depends on the label definition,
  so switch stays 30.
- Guessing the ball inside gaps doesn't help on realistic data either. Closed.
- **Short possessions are still the weak spot:** recall at IoU 0.5 is 20% under 4 s (60% for
  4–10 s, 81% over 10 s). Matched starts are a median 0.2–0.5 s from the truth.
- **v19 = v15 + gap gate + w_gap 0.4 is the new default.** Our match: 294 possessions, 42%
  direct, D 37.1%, public 0.611 / 0.840 (the public macroF1 drop is about one stoppage; watch it).

**Next:** the touch detector trained on DFL (events give the touches, tracking gives the exact
moment), aimed at the short possessions. Then the learned evidence, retrained on DFL.

### 2026-10-24 (night, later): our ball is blurred; touch labels on DFL

**Ball realism** (`tools/ball_realism.py`, live play with the ball seen):
- DFL matches ours on speed (6.0–6.6 vs 6.4 m/s) and distance to the nearest player (1.4–1.7 vs
  1.9 m). Metrica's ball sat on the carrier (0.3 m) and was far too smooth.
- **Our ball track is time-blurred.** Sharp velocity changes are almost absent (over ±1 frame,
  ≥4 m/s: 1.5 per live minute; DFL 17–40). At the 0.5 s scale the counts are close.
  **DFL's ball smoothed with a Gaussian of σ = 6 frames (0.24 s) matches our profile at every
  scale** (`dfl_data.blur_ball`, `sweep.py --blur`). This is why touches on our match look soft and
  late.
- DFL with our blur and our gaps: v15 0.458 / 0.772, **v19 0.511 / 0.812 (7/7)**, v19 + switch 20
  0.490. The blur alone costs 0.056 and moves matched starts from 0.45 s to 0.9–1.0 s late.
  Possessions under 4 s: 12% found.

**Touch labels** (`tools/touch_labels.py`): a teacher on DFL's sharp raw ball. A student (next)
will see only the blurred, gapped ball.
- Source 0, the sharp ball: a velocity-change peak ≥ 2 m/s with a player within 2 m, while
  alive. It finds 83% of tagged passes within 1 s, 76% of shots, but only 44–57% of dribbles,
  tackles and claims, and 0–15% of restarts (those frames are flagged dead).
- Source 1, restarts: BallStatus dead → alive, by the team that takes over.
- Source 2, possession switches: BallPossession changing team while alive.
- Merged (same team within 0.2 s = one touch): 46–49 touches per live minute in the five
  2. Bundesliga matches, 71–78 in the two Bundesliga 1 matches (whose ball is also the noisiest).
  91% of touches are by the team the flags give the ball to.
- Event matching rises with the window (29% at ±0.3 s, 66% at ±1 s): the hand-tagged event clock
  scatters, so exact touch times come from the tracking.

**Next:** the student network: blurred and gapped DFL ball plus players in, P(touch by home / by
away) per frame out. Train on 6 matches, test on the 7th, and compare with our current touch rule.

### 2026-10-24 (night, last): the touch network, v20

**Touch network** (`pipeline/touch_net.py`, `tools/train_touch.py`):
- A residual temporal conv net, about ±2.5 s of context, output none / home / away touch per
  frame.
- Inputs: the ball (seen, position, velocity, velocity change over 0.12 and 0.24 s, gap clocks);
  the 3 nearest players per team (distance, relative position and velocity, speed); counts
  within 2 m and 5 m; mean pace.
- Trained on DFL with the ball blurred and gapped like ours (2 gap patterns per match, teams
  swapped and pitch mirrored as augmentation), against the teacher touches.
- **7-fold held-out, ±0.5 s:** our rule P 0.90, R 0.38, F1 0.53, turnover touches 35%; **network
  at 0.5: P 0.85, R 0.67, F1 0.74, turnover touches 55%**, on time (0.00 s). Better in 7/7.
- Transfer check: with the ball seen in live play it fires 40.4 per minute on our match against
  40.6–45.0 on DFL (rule: 25 against 25–28). But its mean P(touch) is higher on ours (0.31 vs
  0.18–0.20): less sure.

**In the decoder** (`--touch-net`; network touches replace the rule's in touch / last_touch;
DFL with blur + gaps, each match decoded by the network trained without it):

| touch threshold | DFL sp / mF1 | raw flags | public sp / mF1 | our possessions |
|---|---|---|---|---|
| rule (v19) | 0.511 / 0.812 | 0.451 | 0.611 / 0.840 | 294 |
| 0.5 | 0.553 / 0.829 | | 0.513 / 0.815 | 294 |
| **0.7 (v20)** | **0.567 / 0.835** | | **0.640 / 0.866** | 277 |
| 0.8 | 0.574 / 0.836 | 0.512 (7/7) | 0.582 / 0.858 | 270 |
| 0.85 | 0.577 / 0.837 | | 0.593 / 0.848 | 261 |
| 0.9 / 0.95 | 0.575 / 0.575 | | 0.428 / 0.252 | 237 / 208 |

- The decoder wants sure touches: DFL rises with the threshold up to about 0.85.
- At 0.5 the public window showed the failure mode: spurious turnovers inside long possessions
  (#1, #4, #6, #11), a missed end of #9. #3 (2.8 s) was fixed, from a start 14.7 s early to
  0.1 s early.
- Our match is less confident than DFL: from 0.9 up too few touches survive and possessions
  collapse. **0.7 chosen:** within 0.01 of the DFL best, better than v19 in 7/7, away from that
  cliff.
- Extra knobs on top of the network (touch weight 2–3, cheaper switch at touches, switch 20)
  don't help on DFL.
- **v20 = v19 + touch network at 0.7.** Our match: 277 possessions, 42% direct, D 37.3%, public
  0.640 / 0.866. The weights (trained on DFL only, 0.5 MB) are committed in `models/` so the
  submission rebuilds exactly; a rebuild gave an identical file.
- Still open: the shortest possessions (16% of those under 4 s on DFL; public #10 and #12 still
  missed).

### 2026-10-24 (late night): learned evidence on DFL, and DFL's frozen dead ball

**Learned evidence trained on DFL** (`tools/train_learned_dfl.py`): GBM, GRU and a 50/50 blend on
the existing inputs plus the touch network's probabilities (each match's from the network trained
without it). DFL with blur + gaps, controlled labels, 7 rounds holding one match out, decoder costs
picked on the same rounds:

| model | DFL spellF1 / macroF1 | vs hand v20 |
|---|---|---|
| hand v20 | 0.567 / 0.835 | |
| GBM | 0.639 / 0.897 | 7/7 |
| GRU | 0.637 / 0.903 | 6/7 |
| blend | 0.651 / 0.905 | 7/7 |

**But it doesn't transfer whole:** trained on all 7 and run on our match (v21_blend), 299
possessions and D 39% (plausible, unlike v16), but public 0.575 / 0.710. Team errors fall (wrong
team 7.2% vs 9.0%), yet 13.6% of ball-seen public frames are stoppages called live (v20: 2.4%).

**Cause: DFL's ball is frozen while dead.** With the ball seen in true stoppages: DFL ball speed
0.0, still 98%, touch P > 0.5 on 0.5% of frames. Ours: 2.5 m/s, still 48%, touch P 18% (players
carry and pass the ball to the restart). The learned model took a perfectly still ball to mean
"dead". This also flatters every model's dead-ball detection on DFL, the hand model's included.

**Hybrid** (`learned.decode_hybrid`): the hand path decides D vs live; inside each live stretch a
two-state Viterbi over the learned P(H) vs P(A) decides the team.
- DFL: blend hybrid 0.579 / 0.843 at switch 10 (GBM 0.574, GRU 0.568). That is **+0.012 over v20,
  5/7 matches** (worse in the two Bundesliga 1 matches). Most of the learned 0.651 came from the
  frozen-ball stoppages.
- Our match: **v21 = v20 + blend hybrid, switch 10.** 261 possessions, D 37.3% (unchanged), home
  share 57.9% (v20 61.8%), public 0.658 / 0.877, wrong team 7.5%.
- Weights in `models/` (`gbm_dfl.txt` 8.5 MB, `gru_dfl.pt`); a rebuild gave an identical file.

**Next:** give DFL's dead ball realistic motion (carried and passed to the restart spot), so the
learned model can also learn stoppages that look like ours. That is the way to the rest of the
0.65.

### 2026-10-25: a moving dead ball for DFL, and the semi-Markov decoder

**Dead-ball animation** (`dfl_data.animate_dead_ball`, `prepare_ball`, `sweep.py --animate`).
Each DFL dead stretch: the ball lies where play stopped (10–50% of the stretch), is passed between
the 4 players nearest the restart spot (6–15 m/s, rests of 0.3–1.5 s, 0.5–3 m off), and the
taker brings it to the restart spot.

| ball in stoppages, seen | out | speed | still | >8 m/s | nearest | <1.5 m | >8 m |
|---|---|---|---|---|---|---|---|
| DFL as provided | 29% | 0.0 | 98% | 1% | 4.4 m | 13% | 23% |
| DFL animated | 11% | 3.3 | 44% | 20% | 2.4 m | 28% | 9% |
| ours (public true D / full-match v21 D) | 10/16% | 2.5/2.4 | 48/51% | 13/19% | 2.9/3.3 m | 35/28% | 17/22% |

Remaining difference: our ball lies far from everyone more often (about 20% vs 9%).

**Retrained on animated DFL** (touch network, then GBM / GRU; 7 rounds holding one match out):
- Touch network unchanged (F1 0.74). Our old rule's precision drops 0.90 → 0.69: it fires on the
  moving dead ball.
- **Hand v20 0.567 → 0.535**: the frozen ball had been flattering its dead-ball rules. Learned
  blend 0.651 → 0.639. Hybrid 0.542.
- **Transfer still fails, the other way round:** trained on animated DFL, the full learned model
  calls 21.7% of our ball-seen live frames dead (D 57.9% over the match; public 0.275 / 0.725). Its
  team decision keeps improving (wrong team 5.9%). Learned stoppage detection is not trusted on
  our match. v22 hybrid (new touch network): 231 possessions, public 0.524 / 0.853, worse than
  v21. **v21 stays.**

**Semi-Markov decoder** (`pipeline/semimarkov.py`, `tools/eval_hsmm.py`): each segment is scored
as a whole, plus the log-probability of its length (priors from the other six DFL matches).
- Synthetic test: beats the tuned Viterbi (accuracy 0.977 vs 0.973, 94% of short segments vs
  77%).
- On DFL (animated + blur + gaps), no gain. Hand scores: 0.539 vs 0.535 Viterbi. Blend: 0.639 vs
  0.637. Settings that find twice the short possessions (23–28% vs 14%) score 0.47–0.51: each
  real short possession comes with a false one.
- **Short possessions are limited by the evidence, not the decoder.** Kept, off.

**Realistic benchmark now:** DFL with an animated dead ball, blur and our gaps. Hand v20 0.535,
v21-style hybrid 0.542, learned blend 0.639 (doesn't transfer).

### 2026-10-25 (later): why learned stoppage detection fails on our match (time-boxed probe)

Question: the learned blend scores 0.639 on DFL (animated dead ball) against 0.535 for the hand
model, but calls our live play dead (D 58%). Which inputs differ? Our live frames are v21's live
calls (its D comes from the hand model), with the ball seen, compared with DFL's true live frames.

- **Shift ranking:** the largest gaps are time since / until the last thresholded touch (+0.65 /
  +0.51 SD; on the frames called dead +1.40). On our match the touch network is less sure, so
  fewer touches pass 0.7 and live play looks touchless.
- **Without the touch timers** (`--exclude t_since t_until next_side`): D still 53%. With switch
  30 > 15 + 4, every turnover went through a fake stoppage (0% direct); with sane costs (10/15/4,
  6/15/4, 10/25/6) D is still 53%.
- **LightGBM contributions** (pred_contrib) on our wrongly-dead live frames vs DFL live: the
  biggest push to dead is the touch-probability sum over ±2 s (ours 0.01–0.03, DFL 0.19), then
  player pace (1.5 vs 2.5 m/s) and ball speed (4.4 vs 6.7 m/s). 15% of our live frames get
  P(D) > 0.5; DFL's: 0%.
- **Without touch amounts too** (`--exclude ... tpsum`): D 52%, 16% of public ball-seen live
  frames still called dead. The model moves on to pace and ball speed.
- **Conclusion:** our live play is slower than DFL's (Barcelona's calm build-up runs at a pace DFL
  mostly shows in stoppages). Dropping inputs can't remove that without removing pace, which the
  hand model's stoppage rules need too. Learned stoppage detection is dropped. The hybrid
  (hand D, learned team) stays: **v21**.
- `--exclude` added to `train_learned_dfl.py` and `run_learned.py`.

### 2026-10-25 (night): touches, turnovers, team, oracles, the video, and other sources

**Public possessions under v21** (`tools/spell_table.py`): #2 starts 4.9 s late; #6 starts 7.6 s
early; #9 runs 12.3 s long, so #10 (0.8 s) is missed and #11 starts 20.7 s early; #12 (1.3 s)
IoU 0.03; #14 runs 9.6 s long and #15 (3.5 s) is missed (v20 had it at 0.81).

**Touch network at every true public change** (new team's best P within ±3 s; first touch ≥ 0.7):
- #2: P 0.10. The impact is in the data (13 → 4.7 → 1.7 m/s, turning 48° then 108°), but the
  nearest player is the other team's (away 2.2 m, home 3.4 m).
- #10, #12: P 0.53, below the 0.7 cutoff.
- #8, #9: touch found 1.6–2.1 s late (the second touch).
- #15: found (0.74, −0.5 s), but v21's learned team decision overruled it.

**Camera** (`tools/camera.py`): planar calibration from the homographies puts the tactical camera
at x 53 m (halfway line), y 112 m (44 m behind the far touchline), 29 m up, focal ~2300 px,
consistent over the match. Airborne balls would appear displaced if the ATD mapped them through
the ground plane, but **they are not displaced**:
- where the ATD gives a height (8.7% of seen frames), high balls sit close to players (>3 m: 1.6 m),
  and un-projecting makes it worse (5.3 m);
- where the height switches on or off, the ball moves a normal 0.22–0.33 m (uncorrected
  positions would jump 1–24 m).

**Ball-path inputs for the touch network** (`touch_net.path_columns`: per team, lateral distance
and position along the ball's path ahead, players near the path, first interceptor): touch F1
0.74, turnover touches 0.54, the same as without them.

**Turnover network** (`train_touch.py --target turnover`, trained on controlled possession
changes). Finding true changes within ±0.5 s on held-out DFL:

| model | precision | recall | F1 |
|---|---|---|---|
| touch rule | 0.07 | 0.36 | 0.11 |
| touch network (0.5) | 0.07 | 0.51 | 0.12 |
| turnover network (0.3) | 0.31 | 0.35 | 0.33 |

As soft cheaper switches in the decoder (`w_turnover`): DFL 0.534 → 0.538 at best.

**Oracle tests** (DFL realistic, touch-network pipeline 0.534 / 0.811):

| perfect … | spellF1 | macroF1 | short (<4 s) found |
|---|---|---|---|
| change points (as cheaper switches) | 0.574–0.592 | 0.820 | 26–27% |
| dead-ball evidence | 0.577 | 0.848 | 16% |
| team evidence | **0.732–0.763** | 0.873–0.884 | 28–33% |

**The bottleneck is knowing which team has the ball.**

**Team network** (`tools/train_team.py`: inputs plus touch probabilities, ±5 s context, live
frames). Per-frame team accuracy on held-out DFL: hand 82.1%, learned blend 88.6%, **team network
87.9%**. Hybrid decoder 0.538. Its errors: 57% within 2 s of a change (error rate 24–35% there),
36% inside ball gaps, 5.5% error in settled play with the ball seen.

**Ball detector on the tactical video** (`tools/vision_dataset.py`, `pipeline/ball_net.py`,
`tools/train_ball.py`, `tools/review_ball.py`; local only):
- Ball diameter 4.6–8.1 px (median 5.9). The camera pans; 26% of the pitch corners are in view
  on average. OpenCV decodes 244 frames/s.
- Labels: the ATD ball projected through the homography, on frames where it is steady. 7,315
  ball crops and 3,619 without, 160 px, 3 frames; the last 20% of each half held out.
- U-Net with focal loss. The first run spilled GPU memory (7.8 / 8 GB); rerun with fp16, batch 32,
  crops in RAM: 53 s per epoch.
- **Held out: median error 13.2 px, 33% within 6 px; at P ≥ 0.3, found 26% with 0.9% false
  alarms.** Training loss fell steeply late, so likely memorising. Labels possibly off the ball.
- **Dropped at the user's call:** we need gap filling and possession logic, not a new ball track.

**Signed boundary offsets** (DFL realistic): turnovers −0.04 s median (no systematic delay; the
0.5–0.65 s "start offset" is spread, not lag); restarts +0.80 s (probably the dead-ball
animation; our public restarts are early in 3 of 4); stoppage starts −0.16 s. No shift.

**Carrier-following in gaps** (`features.gap_carrier`, `w_gapcarry`): follow the last carrier; if
the ball comes back near the other team, change at the moment an opponent was closest. DFL
0.532–0.534 (no gain). It answers on 47% of live gap frames and is right 77% of the time, while
the decoder alone is right 87%; on the decoder's wrong gap frames it is right only 21%. 19% of
turnovers happen inside a gap. Off.

**Smart Tagging vs possession:**
- Public window: the tag names a team on 71% of live frames and agrees 77% of the time.
- At the 12 true turnovers it switches to the new team within 10 s once (3.6 s early); 9 never
  switch.
- Full match vs v21's turnovers (proxy): 72 of 148 never switch, timing spread ±4 s.
- **Tags follow phases of play, not possession.** Kept at the hand model's 0.5 weight only.

**Broadcast ATD, a first look:** 164,488 frames. 61% of frames have ≥ 10 players tracked
(tactical 90%). On those, the ball is seen 79% of the time (tactical 60%). Not sharper (peaks/min
3.8 / 3.7 / 14.0 vs tactical 6.8 / 7.4 / 16.8). Not time-aligned with tactical; would need
alignment by player formations to fill tactical gaps.

**Ideas not yet tried:**
- Broadcast ATD as a second ball source.
- More training data for the team / touch networks: PFF FC World Cup 2022 open data (64 matches,
  tracking + events); check licence and format.
- A systematic search of the hand model's ~30 parameters on realistic DFL.
- Training the networks on segments / boundaries instead of frames.

### 2026-10-25: broadcast data check; possession-weighted training started

**Broadcast recording, basic check** (ATD + homography only; the broadcast video file is not in
data/, and isn't needed):

| | tactical | broadcast |
|---|---|---|
| frames | 199,319 (133 min) | 164,488 (110 min, includes replays and cut-aways) |
| track slots / per team | 46 / 24 + 22 | 51 / 26 + 25 |
| homography valid | 100% | 66% |
| ≥ 10 players tracked | 90% | 61% |
| usable (≥ 10 players + homography) | 120 min | 67 min, 193 stretches, median 12.8 s, 90% of the time in stretches ≥ 10 s |
| ball seen where usable | 60% | **79%** |

The broadcast covers about two-thirds of the match in alignable stretches and has the ball more
often there. Its ball was not sharper in the earlier peak check (possibly provider-filled).
Alignment to tactical (by player formations) and a ball-to-ball comparison where both exist come
next.

**Possession-weighted team network** (`tools/train_team.py --weighting possession`): every true
possession gets the same total loss weight (mean length / own length, capped at 20), and frames
within ±1 s of a change get 4× that. Training-side weights: median 1, p99 18.8, max 80. Running;
it will be compared with the unweighted team network (hybrid 0.538) on DFL spellF1 and
short-possession recall.

### 2026-10-25: possession-weighted training, result

Team network with per-possession loss weights (`--weighting possession`), 7 rounds holding one
DFL match out (realistic prep), then the hybrid decoder (hand D vs live, network for the team):
- Per-frame team accuracy 85.6% (unweighted 87.9%).
- Over the switch cost, from saved held-out probabilities:

| team network | switch 20 | 30 | 45 | 60 | 90 |
|---|---|---|---|---|---|
| unweighted: spellF1 / short found | 0.538 / 14% | 0.541 / 13% | **0.543** / 11% | 0.540 / 9% | 0.538 / 8% |
| possession-weighted | 0.502 / 17% | 0.517 / 16% | 0.530 / 14% | 0.530 / 12% | 0.538 / 10% |

- **No gain.** At equal short-possession recall the weighted network is behind (14%: 0.530 vs
  0.538). It finds more short possessions only with as many false switches, the same pattern as
  the semi-Markov decoder: the evidence for a 1–2 s possession isn't separable from noise with
  this ball. Unweighted, the team-network hybrid peaks at 0.543 (switch 45), against 0.534 for
  the touch-network pipeline.

### 2026-10-25: broadcast ATD aligned to tactical; as a ball source, no gain

**Alignment** (`tools/broadcast_align.py`): a broadcast frame and a tactical frame are scored by the
median distance from each broadcast player to the nearest tactical player of the same team. Per
usable broadcast stretch, a whole-match search for the middle frame gives the offset; every frame
is then checked at that offset ±2 and accepted under 1 m (slow-motion replays drop out).
- Orientation: same as tactical. 0.26 m median, against 2.95–3.21 m for every flipped / swapped
  version.
- 92% of 193 stretches aligned; 52% of tactical in-half frames get a broadcast frame (median
  score 0.26 m).
- Where both have the ball (74% of aligned frames): distance median 0.81 m, p75 2.0 m, p90 5.3 m,
  57% within 1 m.

**Coverage of our ball gaps** (2,586 s inside the halves): aligned 27%, **broadcast ball on 8%**.
Live-play gaps (v21's live calls, 1,022 s): 13% filled. Stoppage gaps (1,564 s): 5%. In the
public window, gap frames within 2 s of a true change: 189, filled: 0. During our gaps the TV is
mostly on replays and close-ups.

**Pipeline with the gaps filled by the broadcast ball** (v20 settings): 295 possessions (277), D
38.9% (37.3%), public 0.560 / 0.838 (0.640 / 0.866). Worse: the fills chop long gaps into short
pieces (changing the gap / stoppage evidence), and the hand-over between two balls that disagree
by 1–5 m creates jumps and fake touches.

**Conclusion:** the broadcast is not a useful ball source for us. Low coverage exactly where we
need it, so even a smoothed fill has little to win. The alignment is kept
(`data/cache/broadcast_align.npz`).

### 2026-10-26: ATD channel check (does our ATD carry ball-dead or owning-team?)

`tools/atd_channels.py` prints the export's structure only (channel ids, field counts).
- **XML DataFormatSpecification:** per track slot `x`, `y`, `speed` (46 slots tactical, 51
  broadcast), and the ball `x`, `y`, `s`. Nothing else: no ball-status, owning-team or any other
  channel.
- **.txt:** every one of the 199,319 / 164,488 lines has 3 `:` groups, 46 / 51 player items and a
  ball item, each with exactly 3 fields. `atd_cache.py` already reads every field.
- **kloppy:** `ball_state` and `ball_owning_team` are `None` on every frame.
- Provider parameters: half start/end, attack direction, and per slot `position_type`. Two slots
  are tagged `Goalkeeper` (one per team), tracked on 90% of frames, but their median x is 0.24 /
  0.84 of the pitch, so the slot does not stay on the keeper (identities swap). Not usable as a
  keeper id.
- **Conclusion: negative.** The uncorrected versions of the GT's ball-dead / owning-team fields
  are not in what we were given; there is nothing to add as a feature or decoder prior.

### 2026-10-26: pace shift, (a) per-half quantile normalisation

Acceptance rule from here on: a change must raise mean spellF1 on realistic DFL (animated dead
ball, blur, gaps; leave one match out) **and** win in at least 5 of 7 folds.

`learned.qnormalise` (`--qnorm` in `train_learned_dfl.py` and `run_learned.py`): player pace,
ball speed and the touch probabilities (home and away ranked together) replaced by their
quantile within the half; the touch timers (time since / until the last touch) ranked in
`matrix()`. Everything derived from them (context windows) follows.

New tool `tools/transfer_check.py`: per submission, full-match D, possessions, and on the public
window true D called live, true live called D (all / ball seen), spellF1 / macroF1.

| GBM (animated DFL) | DFL full model | DFL hybrid | ours: full D | public live → D | public D → live | public spF1 |
|---|---|---|---|---|---|---|
| plain | 0.633 | 0.547 | 56.6% | 24.6% | 11.2% | 0.263 |
| quantile-normalised | 0.618 (1/7) | 0.549 (3/7) | 53.2% | 15.9% | 16.2% | 0.389 |
| v21 (submission) | | 0.579 | 37.3% | 0.0% | 24.6% | 0.658 |

- **Rejected.** It moves the transfer in the right direction (live play called D 24.6% → 15.9%)
  but nowhere near enough (target: D 37–50%, live called D close to 0), and it costs 0.015 on
  DFL. The hybrid gain (+0.002) is noise (3/7).
- Likely reason: a quantile within the half also depends on how much of the half is dead, and our
  match has more dead time than DFL, so a live frame's pace rank still differs.

### 2026-10-26: pace shift, (b) time-warped DFL and (c) domain weighting

**(b) Time warp** (`dfl_data.set_warp`, `--warp` in `train_touch.py`, `train_learned_dfl.py`,
`run_learned.py`): every DFL half resampled onto a clock stretched 1.25× (positions interpolated,
speeds / 1.25, labels, events and teacher touches moved to the new clock), then animated, blurred
and gapped as before. Touch network, features and GBM all retrained on it.
- One match, as a check: possessions per live minute 6.3 → 5.0 (public window 4.2), median
  5.3 → 6.6 s, live pace 2.26 → 1.81 m/s.
- Touch network (7 folds, ±0.5 s, at 0.5): F1 0.67 (unwarped 0.74); slower touches are harder to
  place.
- DFL scores on the warped matches (not comparable with the unwarped benchmark; every model
  drops): hand v20 0.473, full GBM 0.573, hybrid 0.474.

**(c) Domain weighting** (`learned.domain_classifier` / `domain_weights`, `--domain-weight`): a
GBM tells our frames from DFL's on the model inputs; DFL training rows get p(ours)/p(DFL),
clipped to 0.1–10, mean 1. The weights are very uneven (median 0.31, p90 2.0, effective sample
15%): the two domains separate easily.
- DFL: full GBM 0.627 (plain 0.633, 2/7), hybrid 0.545 (0.547, 2/7).

**Transfer to our match (full learned GBM, `tools/transfer_check.py`):**

| GBM trained on | full D | poss | public live → D | public D → live | public spF1 / mF1 |
|---|---|---|---|---|---|
| animated DFL (plain) | 56.6% | 323 | 24.6% | 11.2% | 0.263 / 0.724 |
| (a) quantile-normalised | 53.2% | 315 | 15.9% | 16.2% | 0.389 / 0.766 |
| (b) warped 1.25× | 51.9% | 317 | 17.4% | 17.3% | 0.339 / 0.751 |
| (a) + (b) | 53.2% | 287 | 17.4% | 28.7% | 0.425 / 0.736 |
| (c) domain-weighted | 58.0% | 352 | 26.0% | 8.8% | 0.270 / 0.706 |
| v21 (hand D, learned team) | 37.3% | 261 | 0.0% | 24.6% | 0.658 / 0.877 |

**Conclusion: the pace shift is not the whole story; experiment 2 is rejected.** Normalising
or warping pace each removes about a third of the live play called dead (25% → 16–17%), and
together they do no better. So the rest of the difference is not a level shift in pace. Domain
weighting fails because the domains are too easy to tell apart (effective sample 15%). Learned
stoppage detection still does not transfer, so it cannot replace the hand D path. **v21 stays.**
(1.15 and 1.35 were not run: 1.25 and the combination already show the ceiling of this route.)

### 2026-10-26: in-domain team model (pseudo-labels from the match itself)

`tools/in_domain_team.py`. Pseudo-labels: ball seen, nearest player within 2 m and ≥ 1.5 m closer
than the nearest opponent, co-movement not disagreeing, hand path live; label = that player's
team. A **players-only** LightGBM (no ball input: per-team spread, centroid, push toward goal,
back line, compactness and their future-minus-past changes from `pipeline/reaction.py`, plus
pressure and mean speed, each with past / future means up to ±10 s) is trained on one match's own
pseudo-labels, home/away swapped as augmentation. Its log-odds are added (weight w) to the
hybrid's team stage on **weak** frames (ball not seen, or nobody within 2 m, or margin < 1.5 m).

**DFL validation, each match treated as unlabelled** (true labels only for scoring):
- Pseudo-labels cover 15–24% of graded frames and are right 91–95% of the time.
- Weak frames are 63–71% of live play. There the players-only model is right 79–83%; the base
  learned GBM is right 76–89%, so the players-only model is worse in 6 of 7 matches (better only
  in the hardest one, J03WN1).

| base | switch | w 0 | w 0.25 | w 0.5 | w 1 |
|---|---|---|---|---|---|
| GBM | 10 | 0.540 | 0.544 (3/7) | 0.542 (2/7) | 0.522 (1/7) |
| GBM | 20 | 0.547 | **0.552 (4/7)** | 0.546 (3/7) | 0.536 (2/7) |
| blend | 10 | 0.541 | 0.546 (4/7) | 0.537 (2/7) | 0.522 (1/7) |
| blend | 20 | 0.542 | **0.549 (4/7)** | 0.549 (3/7) | 0.540 (3/7) |

- Best +0.005 to +0.007 at w 0.25, **4/7 folds: below the 5/7 rule, so not accepted.** Most of
  the gain comes from J03WN1 (+0.02 to +0.05), the match where the base model is weakest.
- Our match (v21 + vote, `v24_w<w>.csv`): pseudo-labels cover 21% of graded frames, weak frames
  75%; the players-only model agrees with v21's team on 79% of v21's live weak frames. w 0
  rebuilds v21 exactly. w 0.25: 271 possessions (261), 41% direct, D unchanged, public 0.684 /
  0.879 (v21 0.658). The public window rose, but it is a check only.
- Reading: without the ball, team shape and pressure give about 80% on hard frames, the same
  level the ball-based model already reaches there, so the vote mostly adds noise. It helps only
  where the base model is poor.
- Kept as the candidate (`v24_w0.25`); variants to strengthen it are running (pseudo-labels from
  the decoded path, DFL-trained players-only model, a gate by base-model uncertainty).

**Variants** (`in_domain_team.py --variants`; DFL, each match treated as unlabelled):
- Pseudo-label sources for the players-only model: *strong* (as above); *decoded* (strong plus
  frames of the decoded hybrid path ≥ 2 s from any decoded change with the base model ≥ 0.8 sure:
  23–37% of graded frames, right 94–99%); *dfl* (the other 6 DFL matches' true live frames);
  *dfl+decoded* (both).
- Players-only accuracy on weak live frames: strong 81.1%, decoded 83.9%, dfl 83.8%,
  dfl+decoded 85.5%.
- Gates: *weak* (as above) or *unsure* (weight 1 − |2 P(H) − 1| of the base model).

| source, gate weak | GBM sw 20, w 0.25 / 0.5 | blend sw 10, w 0.25 / 0.5 | blend sw 20, w 0.25 / 0.5 |
|---|---|---|---|
| strong | +0.005 (4/3) / −0.001 | +0.004 (4/3) / −0.004 | +0.008 (4/3) / +0.008 (3/4) |
| decoded | +0.005 (4/3) / +0.006 (3/4) | +0.008 (7/0) / +0.005 (3/4) | +0.009 (4/3) / +0.008 (5/2) |
| dfl | +0.009 (5/1) / +0.008 (5/2) | +0.008 (5/2) / +0.008 (5/2) | +0.004 (4/3) / +0.012 (6/1) |
| **dfl+decoded** | **+0.010 (6/1) / +0.011 (5/1)** | **+0.009 (6/1) / +0.010 (5/2)** | +0.007 (4/3) / +0.010 (5/2) |

(wins / losses against no vote; base means 0.547 / 0.541 / 0.542; the unsure gate is equal or
worse throughout.) **dfl+decoded with the weak gate wins under every base at ≥ 5/7**, which
makes it unlikely to be a lucky pick among the 48 variants.

**On our match (v25 = v21 + this vote, blend switch 10 as in v21):** decoded pseudo-labels
cover 34% of graded frames; the players-only model agrees with v21's team on 81.5% of v21's live
weak frames.

| version | poss | direct | D | home | public spF1 / mF1 | DFL (blend sw 10) |
|---|---|---|---|---|---|---|
| v21 | 261 | 40% | 37.3% | 57.9% | 0.658 / 0.877 | 0.541 |
| v25_w0.25 | 259 | 39% | 37.3% | 59.9% | 0.671 / 0.880 | 0.550 (6/7) |
| **v25_w0.5** | **264** | 40% | 37.3% | 60.4% | **0.684 / 0.881** | **0.551 (5/7)** |
| v25_w1 | 268 | 41% | 37.3% | 60.2% | 0.684 / 0.883 | not run |

- **v25_w0.5 passes the rule** (+0.010 on DFL, 5/7) and is better on the public window
  (0.684 vs 0.658), with plausible full-match counts. It is the new candidate on both references.
  Rebuild: `python tools/in_domain_team.py --ours --skip-dfl --source dfl+decoded`.

### 2026-10-26: MBR decoding of the team stage (experiment 4)

`pipeline/mbr.py`, `tools/eval_mbr.py`. The hand path fixes D vs live (as in v25); inside each
live stretch, 200 team paths are sampled from a two-state HMM over the v25 team probabilities
(emissions scaled by a temperature tau, change cost "sample switch", forward filtering / backward
sampling, seed 0). Candidates: the v25 decoder at switch 5 / 10 / 15 / 20 / 30 plus 20 of the
samples; per half, the one with the best mean spellF1 against all samples is kept. A fast
spellF1 (`mbr.spell_f1`) matches the official scorer exactly (200 random cases, max difference 0).

DFL with the v25 pipeline held out (reference: v25 decoder, switch 10: **0.551**):

| tau | sample switch 5 | 10 | 20 |
|---|---|---|---|
| 0.02 | 0.545 (2/5) | 0.464 | 0.378 |
| 0.05 | 0.551 (3/4) | 0.507 | 0.457 |
| 0.1 | 0.552 (4/3) | 0.530 | 0.500 |
| 0.2 | 0.551 (1/0) | 0.545 (2/5) | 0.526 |

- Nested (setting chosen on the other six matches): **0.548, −0.003, 2 won / 4 lost. Rejected.**
- Where MBR picks a Viterbi candidate, it is only choosing the switch cost per half (no gain).
  Where it picks a sample, the score drops by up to 0.17: a single sample carries the posterior's
  noise, and the samples agree with each other more than with the truth. The sampled posterior is
  not calibrated well enough for expected-spellF1 choices; with these probabilities Viterbi at a
  fixed switch is already as good as this gets.
- On our match, the best setting (tau 0.1, switch 5 → picks switch 20 / 15) is `v26.csv`: 250
  possessions (v25 264), public 0.684 / 0.881, the same as v25 (the public window does not change).

### 2026-10-26: hand fixes on top of v25 (experiment 5)

`pipeline/fixes.py`, `tools/eval_fixes.py`. In the hybrid the hand path gives only D vs live, so
team fixes enter the team stage as extra log-odds votes (like the players-only vote); the restart
handover edits the decoded path. DFL, the v25 pipeline held out, reference **0.551**:

| fix | best setting | DFL | folds | note |
|---|---|---|---|---|
| A restart handover: a team spell ≤ max_s right after a stoppage, players slower than slow_ms, followed by the other team → D | max_s 3, slow 2 m/s | 0.553 (+0.002) | 5/2 | small; 1.5 m/s does almost nothing |
| B ball out: vote for the team that did not touch it last, first vote_s of the restart | w 1, 5 s | 0.552 (+0.0005) | 4/2 | premise right only 58.7% of 150 restarts (the last touch before the ball goes out is often missed); w 2 hurts |
| C aerial landing: fast ball (> 12 m/s) halving its speed near a player → vote for his team 2 s | w 0.5 | 0.551 | 2/5 | worse at every weight |
| **D touch margin: every network touch votes for its team over vote_s, × min(margin / 1.5 m, 1)** | **w 0.5, 2 s** | **0.559 (+0.0075)** | **5/2** | w 1 / 1 s +0.006 (5/2); w 2 hurts |
| **A + D** | max_s 3 / slow 2 + w 0.5 / 2 s | **0.560 (+0.009)** | **5/2** | |

On our match (`--ours`, from v25's saved inputs):

| version | poss | direct | D | public spF1 / mF1 | DFL |
|---|---|---|---|---|---|
| v25_w0.5 | 264 | 40% | 37.3% | 0.684 / 0.881 | 0.551 |
| v27_1 = D (w 0.5, 2 s) | 269 | 41% | 37.3% | 0.684 / 0.881 | 0.559 (5/7) |
| v27_2 = D (w 1, 1 s) | 276 | 42% | 37.3% | 0.655 / 0.881 | 0.557 (5/7) |
| v27_3 = A | 262 | 40% | 37.4% | 0.684 / 0.881 | 0.553 (5/7) |
| **v27_4 = A + D** | **266** | 40% | 37.4% | **0.684 / 0.881** | **0.560 (5/7)** |

- **A + D passes the rule** (+0.009, 5/7) and leaves the public window unchanged. The touch-margin
  vote works where the earlier hand-model version did not (v11: touch margin on the hand path),
  because here it only nudges the learned team stage, it does not decide switches.
- Settings were picked from a grid on the same 7 matches (about 20 settings), so the true gain is
  probably a little smaller; the neighbouring D settings all give +0.005 to +0.0075 at 5/7, so it
  is not a single lucky setting.
- B and C rejected. Rebuild of v27_4: `python tools/in_domain_team.py --ours --skip-dfl --source
  dfl+decoded --weights 0.5`, then `python tools/eval_fixes.py --skip-dfl --ours "A max_s=3
  slow_ms=2 + D w=0.5 vote_s=2" --name v27`. **Adopted as the submission (`v27.csv`)**; the
  rebuild is byte-identical to v27_4.

### 2026-10-26: future team reaction, step 1: votes in the team stage (experiment 6)

`tools/eval_reaction.py`. The reaction features (`pipeline/reaction.py`, attack direction per half
from each team's mean position) as an extra vote in v27's team stage (H/A split only), DFL held
out, reference v27 **0.560** (direct turnovers found within 5 s 54%, median offset 0.00 s, 65%
within 1 s; 44.6 false switches per hour inside true possessions ≥ 10 s):
- **spread shifted ahead** (shape at t + 2 / 4 / 6 s): worse everywhere, −0.002 to −0.019, at
  most 2/7. The team's shape still follows the old possession too long, or flips with play
  that is not a turnover.
- **spread change** (future minus past over 2 / 4 / 6 s): −0.000 to −0.005, at most 4/7, and
  more false switches in long possessions (up to 48.9 / h): teams reshape inside possessions too.
- **centroid shifted / change**: best +0.0017 (change over 2 s, w 1), 4/7; everything within
  ±0.002 or worse.
- Turnover timing barely moves in any setting (found 49–54%, 62–66% within 1 s).
- **No setting passes; rejected.** As a separate vote, team reaction adds nothing to a team stage
  that already reads touches, co-movement, the players-only vote and the touch-margin vote.
  Step 3 (the second-stage network, which can combine it with everything else) is running.

### 2026-10-26: step 3, the second-stage bidirectional team model (user's idea)

`pipeline/refine.py`, `tools/eval_refine.py`. A 2-layer bidirectional GRU (hidden 64) at 5 fps over
2-minute stretches. Inputs per step: v27's first-stage team log-odds, the hand path's live flag,
ball seen / gap / nearest distances, touch probabilities and touch sides, and per team every
reaction feature (spread, centroid, push, back line, compactness, each with its 2 / 4 / 6 s
change) plus pressure and pace: 54 inputs. Target: the true team on live bins; home/away flip as
augmentation. Leave one DFL match out. Caveat: the training matches' first-stage probabilities
come from models that saw the held-out match (mild stacking leak).
- Per-frame team accuracy on live frames: first stage 89.5%, refined 89.7%.

| decoder (then v27's restart handover) | DFL | vs v27 | folds | turnovers found / within 1 s | false / h in long |
|---|---|---|---|---|---|
| v27 | 0.560 | | | 54% / 65% | 44.6 |
| refined alone, switch 10 / 20 / 30 | 0.552 / 0.548 / 0.543 | −0.009 to −0.017 | ≤ 2/7 | 45–50% / 66–67% | 41.5–47.2 |
| **v27 log-odds + 0.5 × refined** | **0.567** | **+0.007** | **6/7** | 53% / 66% | 44.3 |
| v27 log-odds + 1 × refined | 0.566 | +0.006 | 5/7 | 54% / 66% | 44.3 |

- **The blend passes the rule (+0.007, 6/7)**; alone, the second stage is worse. Gains in
  J03WMX, J03WN1 and J03WR9 (+0.012 to +0.024), the rest within ±0.009.
- **Ablation without the reaction inputs** (`--no-reaction`): per-frame 89.0%, blend 0.5 → 0.560
  (−0.000), blend 1 → 0.559. **The gain comes from the team-reaction features**, read together
  with everything else by the bidirectional network, not from a second pass over the first stage.
  (As a separate hand vote, step 1, they gave nothing.)
- **v28 on our match** (refiner trained on all 7 DFL matches, `models/refine_v28.pt`, 0.5 MB;
  `python tools/eval_refine.py --ours 0.5 --skip-dfl` after the v25 build; a rebuild from the
  saved model is byte-identical): the team differs from v27 on 5.5% of live frames; 275
  possessions (v27 266), 42% direct, D 37.4%, home share 56.1% (v27 60.1%); public 0.684 / 0.882
  (v27 0.684 / 0.881). It passes the rule on DFL and leaves the public window unchanged.

**Tuning the second stage** (`tools/tune_refine.py`: 3 seeds per fold, held-out P(home)
averaged; v27 log-odds + alpha × refined at alpha 0.25 / 0.5 / 0.75 / 1 and switch 10 / 15 / 20; a
model trains in about 17 s). Prediction now runs in overlapping 4-minute windows keeping each
centre (cuDNN rejects a 200k-step sequence); at 5 fps this gives exactly the same scores.

| configuration | per-frame | mean gain over the 12 settings | settings at 6–7/7 | alpha 0.5, switch 10 (v28's) |
|---|---|---|---|---|
| **5 fps, hidden 64, 2 layers** | 90.1% | **+0.0077** | 9 of 12 | **+0.0066, 6 won / 0 lost** |
| 25 fps, 2 conv layers in front | 89.7% | +0.0053 | 4 of 12 | +0.0061, 5/2 |

- **25 fps does not help.** Turnovers within 1 s rise slightly (68% vs 67%), but the GRU carries
  five times more steps for the same context and the overall gain falls. What the network adds is
  the slow team reaction, not frame-exact timing.
- Nested selection of (alpha, switch) is noisy because the grid is flat (ensemble nested +0.0005 to
  +0.005); every setting of the 5 fps model beats v27, so a fixed setting is the honest choice.
- Around 5 fps (3 seeds each; v27 = 0.560):

| configuration | per-frame | mean gain, 12 settings | settings ≥ 6/7 | alpha 0.5 / sw 10 | ensemble nested |
|---|---|---|---|---|---|
| **12.5 fps** (bin 2) | 90.1% | **+0.0100** | **12 of 12** | +0.0092 (6/1) | **0.570 (+0.010, 5/2)** |
| 5 fps (v28's) | 90.1% | +0.0077 | 9 of 12 | +0.0066 (6/0) | 0.561 |
| 5 fps, 3000 steps | | +0.0076 | 9 of 12 | +0.0066 (5/2) | 0.564 |
| 5 fps, 3 layers | 89.8% | +0.0066 | 8 of 12 | +0.0036 (4/3) | 0.569 |
| 5 fps, hidden 128 | 89.8% | +0.0061 | 6 of 12 | +0.0044 (5/2) | 0.573 |
| 2.5 fps (bin 10) | 89.8% | +0.0058 | 7 of 12 | +0.0052 (6/1) | 0.567 |
| 5 fps, 4-minute stretches | 89.8% | +0.0054 | 9 of 12 | +0.0055 (7/0) | 0.569 |
| 25 fps + 2 conv | 89.7% | +0.0053 | 4 of 12 | +0.0061 (5/2) | 0.566 |

- **12.5 fps is the sweet spot**: finer than 5 fps without making the GRU carry 25 fps sequences;
  every setting beats v27 in at least 6 of 7 matches. Bigger or deeper networks, longer stretches
  and longer training do not help (7 training matches are enough for the small network).
- Last round: 12.5 fps with 5 seeds: mean +0.0085, nested 0.571 (+0.010, 6/1); 8.3 fps (bin 3):
  +0.0082, 12 of 12 at ≥ 6/7, nested 0.569 (7/0); 12.5 fps + 1 conv layer: +0.0070. All within noise
  of each other. In every configuration switch 20 beats switch 10 (alpha 0.5–0.75: +0.010 to
  +0.011). Check: **v27 alone at switch 20 is worse** (0.5575; 15: 0.5587; 30: 0.5565; 10: 0.5601),
  so the gain is the second stage's, which lets the decoder use a stricter switch.
- **Chosen: 12.5 fps, 5-seed ensemble, alpha 0.5, switch 20** (the middle of the best region):
  DFL **0.571 (+0.011, 6/1)** vs v27 0.560.

**v29 on our match** (`python tools/tune_refine.py --bin 2 --seeds 5 --tag v29 --ours 0.5 20` after
the v25 build; 5 models `models/refine_v29_s0..4.pt`, 0.5 MB each, trained on all 7 DFL matches;
a rebuild is byte-identical): team differs from v27 on 4.2% of live frames.

| version | poss | median | direct | D | home | public spF1 / mF1 | DFL |
|---|---|---|---|---|---|---|---|
| v27 | 266 | 9.3 s | 40% | 37.4% | 60.1% | 0.684 / 0.880 | 0.560 |
| v28 (5 fps, 1 model, sw 10) | 275 | 9.0 s | 42% | 37.4% | 56.1% | 0.684 / 0.881 | 0.567 |
| **v29** | **249** | 10.8 s | 38% | 37.4% | 57.5% | **0.697 / 0.882** | **0.571** |

- **Adopted as the submission.** The full rebuild chain (v25 inputs, then v29) gives a
  byte-identical file.
- **v29 is the best version on both references**: public 0.697 (previous best v24_w1 0.690, which
  loses on DFL) and DFL 0.571. Full-match counts stay inside the plausible ranges (249
  possessions, 38% direct, D 37.4%).

**Team network context.** Correction: the existing team network already sees about ±10 s (one
5-tap convolution per block, dilations up to 64: 2 × 127 frames per side; the docstring said
±5 s). `train_team.py --context 20 --path` adds dilation 128 (±20 s): per-frame accuracy 88.1%
(±10 s: 87.9%), hybrid best 0.537 at switch 45 (±10 s: 0.543). **No gain; rejected.**

### 2026-10-27: second stage with per-player inputs (set encoder)

`refine.player_inputs`: every 12.5 fps step, per team, the K players nearest the ball, each with 8
numbers in the team's own attacking direction (position, position relative to the ball, velocity
toward goal and sideways, distance to the ball, distance to the nearest opponent) plus a valid flag.
One shared per-player MLP (9 → 32 → 32), masked mean and max per team, mixed into the GRU's input,
so player order and track swaps do not matter. `tune_refine.py --players K`. Same protocol as v29's
configuration (12.5 fps, 5 seeds, leave one match out, 12 alpha × switch settings; v27 = 0.560):

| configuration | per-frame | mean gain, 12 settings | settings ≥ 6/7 | alpha 0.5 / sw 20 (v29's) | ensemble nested |
|---|---|---|---|---|---|
| **no player inputs (v29)** | 90.1% | **+0.0085** | | **0.571 (+0.011)** | **0.571** |
| 11 players per team | 89.7% | +0.0063 | 8 of 12 | 0.567 (+0.0070, 7/0) | 0.565 (+0.005, 6/1) |
| 6 players per team | 89.5% | +0.0051 | 6 of 12 | 0.565 (+0.0050, 6/1) | 0.565 (+0.005, 5/2) |

- **Rejected.** Raw per-player inputs lower per-frame accuracy and the gain. 11 players are more
  consistent per fold (7/0 at five settings) but smaller; seed 0 alone with 6 players loses (2/5).
  With 7 training matches the network overfits the extra inputs; the hand-made team summaries
  (reaction features) already carry what it can use. Code kept, off by default (`PLAYERS = 0`);
  v29 is unchanged.

### 2026-10-27: the dead-ball side; second-stage dead-ball head

**Where v29 loses on DFL** (`tools/eval_dead.py`, aggregate): true D called live 13.0%, true live
called D 12.7%; 92% of stoppages >= 2 s are found, but the called restart is late (median end
+1.3 s; within 1 s 33%, within 3 s 62%). Bounds: v29 0.571; **D vs live from the labels 0.698**;
team from the labels 0.748. On the public window v29 never calls live frames D but calls 24.6%
of true D live (`tools/transfer_check.py`).

**Dead-ball head** (`refine.dead_inputs / dead_targets / decode_dead`, `tools/eval_dead_refine.py`):
the v29 second-stage network (12.5 fps, 3 seeds) trained to say whether the ball is dead, with
(d1) or without (d1n, `--no-pd`) the first stage's P(dead). `decode_dead`: two-state live/dead
Viterbi over beta x its log-odds + gamma for agreeing with the hand path, cost per change
(verified equal to the 3-state Viterbi); then the team as v29 (alpha 0.5, switch 20, handover).
Grid beta (0.5, 1) x gamma (0, 0.5, 1, 2) x cost (10, 25, 50, 100); v29 = 0.571.

| mode | DFL nested | grid | public spF1 / mF1 | pub D>live | full D | poss |
|---|---|---|---|---|---|---|
| v29 | 0.571 | | 0.697 / 0.882 | 24.6% | 37.4% | 249 |
| free, d1 | 0.657 (+0.086, 7/0) | +0.071 to +0.089, all 7/0 | 0.618–0.630 | 53–56% | 41% | 254–266 |
| free, d1n | 0.650 (+0.079, 7/0) | +0.055 to +0.080, all 7/0 | 0.637–0.658 | 45–62% | 44% | 238–240 |
| near (only within 3 s of a hand D edge), d1 | 0.606 (+0.035) | all 7/0 | 0.675 / 0.851 | 34.9% | 34.9% | 249 |
| near, d1n | 0.597 (+0.027) | all 7/0 | 0.675 / 0.858 | 32.7% | 36.2% | 242 |
| add (hand D kept, head may only add D), d1 | 0.615 (+0.045) | all 7/0 | 0.675 / 0.916 | 12.1% | 48.9% | 239 |
| **add, d1n (0.5, 0.5, 25)** | **0.612 (+0.041, 7/0)**; setting 0.611 | +0.032 to +0.041, all 7/0 | **0.697 / 0.897** | 19.3% | 48.8% | 230 |

- Per-bin dead vs live on DFL: head 97.3% (d1) / 96.4% (d1n), hand 87.3%, first stage 97.1%.
  The free head rewrites 15–20% of our match's D vs live and removes real stoppages on the
  public window: the same domain gap as the learned D call (2026-10-24). Rejected.
- **Add-only without the first stage's P(dead) is the first dead-ball change that does not hurt
  the public window**: spellF1 unchanged (0.697), macroF1 0.882 -> 0.897, public D>live 24.6% ->
  19.3%, live>D still 0%. Milder settings (gamma 1–2) give the same public number with D 47–48%.
  Caveats: full-match D 48.8% and 230 possessions sit at the edge of the plausible ranges (37–50%,
  ~230–300); direct share 33% (v29 38%).
- Candidate **v30 = v30_d1n_add** (`python tools/eval_dead_refine.py --tag d1n --no-pd --mode add
  --ours 0.5 0.5 25` after the v25 and v29 builds; `models/dead_d1n_s0..2.pt`); rebuild
  byte-identical (5e4e5839…). Not yet the submission.

### 2026-10-27: leak-free second stage on 3 gap patterns (in progress, paused)

The second stage (v29) is validated on match g with training inputs from first-stage models that
saw g (stacking leak), and it sees one gap pattern per match. `tools/clean_stage1.py` retrains the
first stage (GBM + GRU + players-only vote, combined as v25) without each PAIR of DFL matches (21)
and without each single match (7, for the deployed model), and predicts every target on 3 gap
patterns (test_seed + 0, 1, 2). Cached per call in `data/cache/dfl/s1_<excluded>_<target>_<v>.npz`
(hand codes in `hand_<g>_<v>.npy`); a rerun only builds what is missing.
`tools/tune_refine.py --stage1 {lomo, clean, clean3}` trains the second stage on those inputs
(the held-out match's own inputs stay the v25 ones, so the numbers compare with v29's).

**Paused 2026-10-27 02:35** at 138 of 147 input files (all 21 pairs done; singles done for some
matches, the remaining 3 singles missing; all cached files verified readable). See "Resume here".

**Resumed 2026-10-27:** the first-stage build is complete (147 of 147 files). Team accuracy of the
first stage on live frames, mean over matches x 3 gap patterns: leak-free pairs 88.9%, single
exclusion 89.6%. Next: the second stage on these inputs (`--stage1 clean`, then `clean3`).

**Leak-free second stage, 1 gap pattern** (`tune_refine.py --bin 2 --seeds 5 --stage1 clean --tag
b2s5c1`; held-out inputs still v25, so comparable with v29 / b2): per-frame team accuracy 90.1%.
Grid mean over alpha x switch **+0.0066** (leaky b2: +0.0085); v29's cell (alpha 0.5, switch 20)
0.567, +0.0065, 4/7 (leaky 0.571, 6/7); best cell alpha 0.5 / switch 15 0.570, +0.0096, 5/7;
nested 0.570 (leaky 0.571); single seed nested 0.562. So the leak inflated v29's gain by about
0.002 on the grid mean and 0.004 on its own cell; the second stage still helps without it (all 12
cells positive, 4–6 of 7 folds). Next: `--stage1 clean3` (3 gap patterns).

**Decision (user, 2026-10-27):** keep v29 as the submission for now; v30 (dead-ball head) stays a
candidate. Continue with clean3.

**Public window, possession by possession, v29 (`tools/spell_table.py`):** 8 of 17 possessions
at IoU >= 0.7, 4 at 0.5–0.7, 1 at 0.3–0.5 (#11), 4 missed: #2 (IoU 0.27: the turnover after #1
found 5.8 s late), #10, #12 and #15 (H spells of 0.8, 1.3 and 3.5 s, read as A throughout).
Biggest timing errors: #5 > D > #6, where the 5.5 s stoppage is missed (A ends 2.0 s early, H
starts 7.5 s early); #9–#12, where A runs on 12.3 s past #9, the two short H spells are missed and
a spurious 3.2 s H sits inside #11 (k 2); #14 runs on 9.6 s over the missed #15 into the stoppage
(v30 trims that to 6.7 s, the only difference between the two). One spurious possession in total.

**Leak-free second stage, 3 gap patterns** (`--stage1 clean3 --tag b2s5c3`): per-frame team
accuracy 90.4% (clean 90.1%). Grid mean **+0.0082** (clean +0.0066, leaky b2 +0.0085), every cell
positive, 5–7 of 7 folds (one 7/0 cell); v29's cell (alpha 0.5, switch 20) 0.570, +0.0097, 6/7
(clean 0.567 4/7, leaky 0.571 6/7); best cell alpha 0.75 / switch 20 0.571, +0.0104, 6/7; nested
0.568 (clean 0.570, leaky 0.571), seed 0 nested 0.566. The 3 gap patterns win back what the leak
had added, and the folds are steadier, but on DFL it ties v29 rather than beating it. A v31 built
this way would be v29 without the leak, at the same DFL score.

**Viewer for the problem possessions** (`tools/possession_view.py 9 12`, local window only,
prints nothing about the data): truth / v29 / hand-D strips, P(home) of the first stage, the
second stage and the v29 blend, touch-network probabilities and picked touches, nearest player
per team with ball gaps shaded, push / compactness change over 4 s, and the pitch at a cursor
(slider, arrows, space to play). For the user to inspect #1–#2, #5–#6, #9–#13 and #14–#15.

**User's observation in the viewer (#9–#12):** Athletic's keeper is carried by a track slot whose
team is Barcelona, which is what makes us switch. DATA.md: slots "appear, vanish and swap identity",
and we give each slot one fixed team from the metadata (`atd_cache.team_of_each_track`), so a
swapped slot keeps the wrong team for its whole life. Every input (nearest player per team,
touches, pseudo-labels, reaction) inherits it, and the DFL training data (clean team ids) never
showed it. The user also suspects errors in the truth. CHALLENGE.md: the truth is derived
mechanically from the *corrected* tracking (D where it marks the ball dead, otherwise the team it
names as owning), nothing smoothed or hand-adjusted, and "if your reading disagrees with it, the
ground truth is right"; the private full match is built the same way. Improving the provided
tracking is explicitly allowed (CHALLENGE.md), so a per-frame team re-assignment of slots is legal.

**User's notes from the viewer (2026-10-27), recorded before any change:**
1. **#9–#12:** two "keepers" are tracked, but wrongly. The slot with Athletic's team stands still in
   Athletic's goal (likely a frozen ghost track). The moving one, carried by a Barcelona slot, is
   really Athletic's keeper. Barcelona's keeper is not detected at all. That impostor is what makes
   us switch to H. The user: we don't need Barcelona's keeper, so the fix is to drop ghosts, cut
   slots into pieces at gaps / impossible jumps, and decide each piece's team by position (keeper
   rule: the deepest player in a goal belongs to the team defending it; at most 11 per team).
2. **#1–#2:** by eye, the truth's switch from #1 (A) to #2 (H) happens too early; v29 switches
   5.8 s later and looks more accurate to the user. Context: the truth comes mechanically from the
   corrected tracking's owner flag (CHALLENGE.md), and both the public window and the private full
   match are built that way. Across the other public possessions our starts are on time or early
   (−2.6 to +0.2 s), so #2 is an outlier, not a sign that the truth runs systematically early.

**User's review of the remaining public stretches (2026-10-27, viewer, in words):**
3. **#5–#6:** not a stoppage on screen; the ball is only missing from the tracking. The truth's
   5.5 s D looks wrong to the user, and v29 (H through it) looks correct. Same class as #1–#2:
   truth vs eye.
4. **#14–#15: a real miss, fixable.** Athletic lose the ball; #15 has two clear Barcelona touches;
   then the ball goes out of play and the players stop, but the keeper kicks the ball right after
   it went out, and we carry A on over all of it. Two faults: the two H touches don't turn the
   possession, and a kick after the ball is out (players stopped) keeps play "live" for us.
5. **#3: keeper tracking again.** When one keeper is not visible, the tracker doubles the other
   one (the visible keeper shows up for both teams). Same family as #9–#12; the keeper fix must
   cover duplicates as well as ghosts and wrong-team slots.
6. **#4–#7:** fine for now.
7. **#17:** fine; there the keeper hands the ball to a defender to take the goal kick (we start H
   1.3 s before the truth, which keeps D until the kick).

Summary: tracking (keepers: ghost, wrong team, duplicate) in #3 and #9–#12; our own misses in
#14–#15 (touches not turning the possession, play kept live after the ball goes out); truth vs eye
in #1–#2 and #5–#6 (not to be tuned toward); #4, #7, #17 fine.

**Decision (user):** do (1) the keeper tracking fix and (2.1) "ball out + players stopped = D, a
kick right after does not restart play". (2.2) "two opposing touches force a switch" is on hold:
the possession must still change at the *first* touch, not the second.

**Keeper tracking, measured** (`tools/keeper_diag.py`, full match, graded frames, aggregate;
checked first on a synthetic match with one of each error planted):

| check | seconds | % graded | ball within 10 m | episodes >= 1 s |
|---|---|---|---|---|
| impostor keeper in Barcelona's goal (an Athletic slot deepest, >= 2 m behind every Barcelona player, <= 6 m from the line) | 262 | 4.1% | 4% | 13 |
| impostor keeper in Athletic's goal (the #9–#12 case) | 30 | 0.5% | 9% | 8 |
| the same with frozen defenders ignored | 262 / 31 | | | |
| no defender in the box (keeper not tracked), Barcelona / Athletic goal | 909 / 1276 | 14.3% / 20.1% | 0% | |
| a slot still (< 0.15 m over 3 s), in a box | 3423 | 53.9% | 0% | 34 |
| two slots <= 0.3 m apart, same / different teams / keeper zone | 80 / 79 / 10 | 1.3% / 1.2% / 0.2% | 5% / 18% / 11% | 11 / 4 / 2 |
| (two slots <= 1 m apart, different teams: mostly real duels) | 1264 | 19.9% | 20% | 436 |
| a slot jumping > 1 m in one frame | 12 | 0.2% | 11% | 5 |

- **Impostor keepers are the real, measurable error:** 13 episodes (262 s) in Barcelona's goal and
  8 (30 s) in Athletic's. Most happen with the ball far away, but where it comes near they decide
  the team (#9–#12).
- **Stillness can't identify ghosts:** real keepers stand nearly still half the match. A still slot
  in a box is almost never near the ball, so ghosts matter only through the impostor check.
- **True duplicates (<= 0.3 m) are rare**, in the keeper zone 10 s in total. So the doubling in #3
  is more likely an impostor than two slots on top of each other.
- **Identity swaps happen across gaps, not as jumps** (12 s of jumps in the whole match). So the
  fix works per piece: a slot's continuous tracked run, which is where a new person can start.

**Keeper fix** (`pipeline/track_fix.fix_keepers`; on with `MTAH_VARIANT=_kf`, which also suffixes
every submission and the v25 inputs cache, so v29's files are untouched). A piece of team T counts
as the other team's keeper when, within a half:
- it spends >= 80% of its frames in the box of the goal the other team defends, and on >= 50% of
  them it is deeper than every *moving* defender there (a still one may be a ghost), for >= 2 s;
- or (a swap without a gap) it is the deepest for >= 1 s while a still defender (the ghost) stands
  in that box. That stretch is flipped, widened to the piece's whole stay in the box.
A flipped stretch moves to a new slot of the other team. Synthetic tests: ghost + impostor across a
gap and without one, both flipped; a striker in the box with the keeper deeper, not flipped.
- Whole pieces only: 8 pieces, 391 s. Impostors in Barcelona's goal 262 -> 24 s; in Athletic's
  goal 30 -> 30 s (those swaps happen without a gap).
- With the second pass ungated: 58 stretches, 1169 s, impostors 0 / 0. Too many: with Athletic's
  keeper untracked (20% of the time), any Barcelona attacker deepest in the box was flipped.
- **Gated on a still defender (kept):** 18 stretches, 585 s (9 from each team); impostors left
  23 s / 22 s, none with the ball within 10 m.

**v29 rebuilt on the keeper-fixed tracking (v29_kf;** same models, `MTAH_VARIANT=_kf`, v25 then
tune_refine `--ours 0.5 20`): full D 37.3% (v29 37.4%), 238 possessions (249), public **0.667 /
0.890** (v29 0.697 / 0.882), 0 spurious possessions (v29 1). Per possession:
- #11: the wrong 3.2 s H is gone (k 2 -> 1, IoU 0.31 -> 0.38), as intended. But that blip had
  been splitting our long A, so now one A runs from #9 to #13: #9 IoU 0.62 -> 0.47 (end +22.2 s),
  #13 0.54 -> 0.09. #10 and #12 are still missed (the short H spells were never ours).
- **#4 breaks:** IoU 0.97 -> 0.76, split in 2 (k 2), end −9.2 s. A new A appears inside Barcelona's
  41.6 s possession, likely a wrong flip there.
- #1–#2 slightly better (late turnover 5.8 -> 5.3 s; #2 IoU 0.27 -> 0.33, now a hit).
Not adopted; waiting for the user's check in the viewer (#4, #9–#13, #3).

**User: in #9–#13 the keeper is still Barcelona's.** `tools/keeper_window.py 9 13` (seconds only):
the rule had flipped Athletic's keeper only while he was inside the box; a Barcelona slot was still
within 25 m of Athletic's goal for 6.0 s. Flipped stretches now extend to the whole stay within
30 m of the goal (`EXTEND_M`): 18 stretches, 623 s; in the window 3.0 s within 25 m remain (maybe
a real Barcelona forward). Rebuilt v29_kf: public **0.653 / 0.889** (first fix 0.667; v29 0.697),
238 possessions, D 37.3%. Per possession the same as the first fix, plus #5 0.72 -> 0.70.
**#4:** without the fix, an Athletic slot is in Barcelona's box for 40.2 of the 41.6 s, deeper
than every Barcelona player on most of its piece. That is Barcelona's keeper under an Athletic slot
(the 262 s of impostors in Barcelona's goal). The fix gives it back to Barcelona (5.5 s left), so
the flip there is probably right, yet #4 breaks (a new A at its end). To check in the viewer.

**User's rule instead (v29 stays the base; nothing flipped):** a team's keeper is *visible* when
its deepest player is inside its own box and >= 2 m behind the rest of the team (the defensive
line). While it is, a player of the other team behind that line and within 30 m of that goal for
>= 2 s is the doubled or swapped keeper from the far side, and is ignored (NaN). With no visible
keeper, nothing changes. `pipeline/track_fix.keeper_status / ignore_impostors`, `MTAH_VARIANT=_ki`.
Synthetic: impostor ignored (16 s); no visible keeper, nothing ignored; a striker staying >= 2 s
behind the line is ignored too (a known side effect of the rule).
- Full match: keeper visible Barcelona 81%, Athletic 74% of graded frames; ignored 1454 s
  (Barcelona slots 599 s, Athletic slots 870 s).
- In #4: Barcelona's keeper visible 90%, Athletic's 0%, an Athletic slot behind Barcelona's line
  ignored for 36.5 s (the doubled keeper, as the user saw in #3). #11: 6.5 s of a Barcelona slot
  ignored.
- **v29_ki: public 0.640 / 0.886**, D 39.2%, 235 possessions. Per possession the same as both
  fixes before: #11's wrong H gone, #2 now a hit, but #9–#13 merge into one A (#13 IoU 0.09) and #4
  breaks (0.97 -> 0.76, k 2, end −9.2 s). #3 0.61 -> 0.55, #16 0.94 -> 0.85.
- So all three ways (flip, flip extended, ignore) give the same #4 / #9–#13 result: the keeper is
  handled as the rule intends, and what changes is how the models react without the wrong player.
- Viewer `--keepers` (default on): strips for "H keeper" / "A keeper" visible, and on the pitch
  the raw tracking with the ignored players as grey crosses.

**User: in these cases the other team's slot is the real keeper, so remove the team's own one.**
`ignore_impostors(mode=...)`: "own" (`_ko`) ignores the defending team's own deepest player in the
same cases; "own_flip" (`_kx`) does that and also gives the other team's slot (the real keeper) to
the defending team. Synthetic: as intended; in "own" a striker >= 2 s behind the line removes the
real keeper for that time (side effect). Viewer follows the mode (grey cross = removed; in `_kx`
the real keeper is drawn in his corrected colour).

| version | public spF1 / mF1 | full D | poss | #2 | #3 | #4 | #9 | #11 | #13 | #16 |
|---|---|---|---|---|---|---|---|---|---|---|
| v29 | **0.697** / 0.882 | 37.4% | 249 | 0.27 | 0.61 | 0.97 | 0.62 | 0.31 (k2) | 0.54 | 0.94 |
| v29_ki (other team's slot removed) | 0.640 / 0.886 | 39.2% | 235 | 0.33 | 0.55 | 0.76 | 0.47 | 0.38 | 0.09 | 0.85 |
| **v29_ko (own removed)** | **0.684** / 0.881 | 38.0% | 259 | 0.34 | 0.55 | 0.96 | 0.62 | 0.31 (k2) | 0.57 | 0.94 |
| v29_kx (own removed + real keeper flipped) | 0.653 / 0.883 | 38.7% | 244 | 0.35 | 0.61 | 0.76 | 0.47 | 0.38 | 0.09 | 0.73 |

- v29_ko keeps #4 (0.96) and turns #2 into a hit, but keeps the wrong H inside #11: there the
  removed "own" keeper is Athletic's ghost, and the real keeper still carries Barcelona's label.
  Loses on #3 and #5 (−0.03 to −0.06). Net −0.013, near the noise of the public window (~0.01).
- v29_kx, the tracking as the user describes it (both stretches right), gives #4's break and the
  #9–#13 merge again. So once the keepers are right in both places, our models get #4's end and
  #9–#13 wrong: in v29 the wrong keepers were hiding real model errors there.

**Decision (user):** v29 stays. **The stoppage after #5 (5.5 s), across versions** (share of it
called D): v5 / v6 / v15 93%; v8–v12 69%; **v19 onwards 0%** (v19, v20, v21, v29, and v30 too). The
ball is missing on 93% of it. The user's guess is right: since v19 a ball gap counts as dead
evidence only as far as the players are slow (`gap_gate` 1), at half the weight (`w_gap` 0.8 ->
0.4), so with the ball lost and players still moving we keep the possession. (v8–v12 had the gate
at w_gap 0.8: 69%.) v19's change was chosen on DFL with our gap pattern (+0.042, 7/7). Note: the
user judged this stretch earlier, by eye, as "not a stoppage, the ball is just missing".
**Correction (user):** it is a real stoppage, a foul. The ball is visible on video, but the
tracking loses it after the whistle (likely held or standing among players). So the truth's D is
right, and #5–#6 is our miss, not a truth-vs-eye case. That leaves only #1–#2 in that class.

**Missing ball as dead-ball evidence, swept on top of v29** (`tools/eval_gap_dead.py`; new hand
params `w_ease` / `ease_gate` / `ease_ms` 1 / `ease_lag_s` 2, off by default, so v29 is unchanged:
the default rebuild matches v29's 0.571). Only the hand path's D vs live changes; team as v29. On
DFL the team probabilities are v29's held-out ones; on our match the second stage is rerun.

| setting | DFL | vs v29 | folds | public spF1 / mF1 | full D | poss | foul after #5 called D |
|---|---|---|---|---|---|---|---|
| v29 (w_gap 0.4) | 0.571 | | | 0.697 / 0.882 | 37.4% | 249 | 0% |
| w_gap 0.4 + ease 0.5 | 0.563 | −0.008 | 0/7 | 0.697 / 0.902 | 39.1% | 244 | 93% |
| w_gap 0.4 + ease 0.5 + gate | 0.563 | −0.008 | 0/7 | 0.697 / 0.902 | 39.6% | 243 | 93% |
| w_gap 0.4 + ease 1 | 0.553 | −0.018 | 0/7 | 0.697 / 0.902 | 40.3% | 243 | 93% |
| w_gap 0.6 | 0.561 | −0.010 | 0/7 | 0.697 / 0.910 | 39.1% | 245 | 0% |
| w_gap 0.6 + ease 0.5 | 0.556 | −0.015 | 0/7 | 0.697 / 0.929 | 40.5% | 241 | 93% |
| w_gap 0.8 (v8-like) | 0.557 | −0.014 | 1/6 | 0.675 / 0.927 | 40.0% | 244 | 69% |
| (the other 8 settings) | 0.539–0.552 | −0.019 to −0.032 | 0/7 | 0.606–0.697 | 40.7–42.7% | | 93% |

- **The easing rule catches the foul** (0% -> 93%) and lifts public macroF1 (0.882 -> 0.902–0.929),
  but **public spellF1 does not move** (0.697 at best). #5 and #6 were already hits, so catching
  the stoppage only improves their boundaries, which the frame-level metric sees and spellF1
  barely does.
- **On DFL every setting loses, in every fold** (best −0.008, 0/7; nested picks v29's). More missing
  ball as dead evidence, gated or not, adds false stoppages elsewhere. That is the same reason
  v19 lowered w_gap. **Rejected; v29 stays.** The params stay in decode.py, off.

**A teammate's per-possession table (public window), against v29.** Their status column marks
IoU >= 0.5 (the headline threshold) as OK; the ranking averages 0.3–0.7. Recall in points (one
per threshold, 85 in total): **teammate 43 / 85 (0.51), v29 55 / 85 (0.65)**; their spurious count is
not in the table, so their spellF1 is unknown.
- **They get #15** (IoU 0.71, Barcelona's 3.5 s spell that v29 misses) because their #14 stops
  about 10 s early; their #14 itself scores only 0.30.
- v29 is clearly better on #3, #5, #7, #11, #13, #14; they are better on #2 (0.36 vs 0.27) and #9
  (0.72 vs 0.62, their #9 ends early). Both miss #10 and #12.

**Viewer bug: the second-stage and blend curves were out of time (2026-10-28).**
`tools/possession_view.py` built the second-stage inputs with `refine.BIN` at its default 5, but
v29's refiner runs at bin 2 (`--bin 2`). The predictions were spread over the wrong frames, so the
"second stage" and "v29 blend" P(home) curves did not belong to the frames under them. The labels,
touches, ball and pitch panels were right. Fixed: the bin now comes from the saved model's config.
Check: re-decoding with bin 2 reproduces v29.csv exactly (0 frames differ); with bin 5 it differs
on 8932 frames. The P(home) 0.82 the user read in #15 came from this bug.

**#2 and #15, with the corrected signals** (`tools/switch_diag.py 2 15`, relative seconds and
aggregates only):
- **#15 (H, 3.5 s):** P(H) over the spell: first stage 0.37, second 0.38, blend 0.32; the blend
  favours H on 9% of it. Summed blend log-odds toward H −69 (a switch costs 20, two needed). Two H
  touches are picked (−0.5 s, P 0.74, margin 1.8 m; +0.7 s, P 0.73, margin 0.3 m), then A touches
  at +3.8 s (margin 10.9 m, during the true D) and +5.7 s. The hand path stays live until +9.6 s
  (truth D from +3.5 s). So the team models do not believe H here, and the touches alone don't
  move them.
- **#2 (H, 8.1 s):** P(H) first 0.44, second 0.21, blend 0.33. The first picked touch is A at
  −0.2 s (P(A) 0.67, nearest player Athletic by 1.5 m): this is the header the user sees as
  Barcelona's. The next H touch, the pick-up, is at +6.0 s (P 0.96), and v29 switches at +5.8 s,
  already on it. So the pick-up is already used; what is missing is the header (or a pick-up
  earlier than the detected one).

**#15: what the dead-ball side sees.** After the true D starts (+3.5 s) the tracking ball is seen
and inside the pitch (out 0% over +3.5..+5 s, 4% over +5..+7 s), and the players are no slower
than in the H spell (1.1–1.3 vs 1.35 m/s). So "ball out + players stopped" has nothing to fire
on here; (b) dropped for #15.

**Runs of same-team touches as a team vote** (`fixes.run_vote`, `tools/eval_touch_run.py`):
`min_n` picked touches in a row by one team, each within `max_gap_s`, after a touch by the other
team (turnover 1): ±w on the blend's log-odds from the first of them (the change stays at the first
touch) until the other team's next touch, at most `hold_s` after the last. Added to v29's blend
before the team Viterbi; D vs live untouched. DFL with v29's held-out second stage (refine_b2):

| setting | DFL | vs v29 | folds | public spF1 / mF1 | poss | #13 | #14 | #15 | #7 |
|---|---|---|---|---|---|---|---|---|---|
| v29 | 0.571 | | | 0.697 / 0.882 | 249 | 0.54 | 0.78 | miss | 0.76 |
| **w 2, n 2, gap 2, hold 4 (v29_tr_a)** | 0.573 | +0.0024 | **6/1** | **0.727** / 0.886 | 263 | 0.78 | 0.97 | **0.83** | 0.63 |
| w 1, n 2, gap 4, hold 1 (v29_tr_b) | 0.574 | +0.0031 | 5/2 | 0.697 / 0.881 | 264 | 0.54 | 0.78 | miss | 0.73 |
| w 1, n 2, gap 2, hold 4 (v29_tr_c) | 0.573 | +0.0024 | 4/3 | 0.703 / 0.885 | 257 | 0.60 | 0.97 | 0.68 | 0.72 |
| best by mean: w 1, n 2, gap 2, hold 4 | 0.574 | +0.0034 | 4/3 | | | | | | |
| nested (setting picked on the other six) | 0.568 | −0.0031 | 4/3 | | | | | | |

- v29_tr_a fixes #15 (two H touches, then held until Athletic's kick), and with it #14's run-on
  (end +9.6 -> −0.5 s) and #13's early start; #12 is touched (0.13). Cost: #7 0.76 -> 0.63 (ends
  0.9 s late), #8 starts 0.9 s late. 16 of 17 found, 1 spurious (an A over the true D after #15,
  the keeper's kick).
- On DFL the gain is small (+0.002–0.003) and the grid is flat; the 6/1 setting passes the >= 5/7
  rule, but nested selection loses (−0.003), so the setting itself is not stable across matches.
  Turnover 1 beats 0; min_n 3 does worse.

**Decision (user, 2026-10-28): v29_tr_a becomes the submission.** `validate_submission.py`: valid;
a rebuild (`tools/eval_touch_run.py --ours "w=2 min_n=2 max_gap_s=2 hold_s=4 turnover=1"`) is
byte-identical. Next: #2.

**#2, by the tracking (aggregates):** the header (−1..0 s) has Athletic nearest (1.5 m vs 3.0 m),
so the touch is credited to Athletic (P 0.67). Then the ball is loose (2.6 m/s) for 4 s, with a
real Athletic player (one continuous track, 1082 s long) running beside it, 3.0 m on average, min
0.7 m, and Barcelona's nearest about 6 m away; Barcelona reaches it at +4..+5 s (0.2 m, P(H touch)
0.42), picked at +6.0 s (0.96). No tracking error here: the truth gives the loose ball to the
header's team, and our only error is the header's team.

**Backdating** (`fixes.backdate`, `tools/eval_backdate.py`, on top of v29_tr_a): a decoded change
to team T at a touch by T moves back to the previous picked touch when that was the other team's
with P(touch) < `p_max`, within `max_s`, no picked touch between, and the ball loose (mean nearest
player >= `loose_m`). DFL (v29's held-out second stage), 36 settings:

| setting | DFL | vs v29_tr_a | folds | public spF1 / mF1 | #1 | #2 | #15 |
|---|---|---|---|---|---|---|---|
| v29_tr_a (submission) | 0.5731 | | | 0.727 / 0.886 | 0.53 | 0.27 | 0.83 |
| p_max 0.7, max_s 3–9, loose 2.5 (v29_bd_a, max_s 9) | 0.5757 | +0.0025 | 5/1 | 0.727 / 0.886 | 0.53 | 0.27 | 0.83 |
| **p_max 0.7, max_s 3–9, loose 1.5 (v29_bd_b, max_s 9)** | 0.5755 | +0.0024 | **5/1** | **0.812 / 0.901** | **0.96** | **0.96** | 0.83 |
| p_max 0.8, max_s 9, loose 1.5 (probe) | 0.5744 | +0.0013 | 4/2 | 0.752 / 0.893 | 0.96 | 0.96 | 0.17 |
| p_max 1.01 (any previous touch) | 0.512–0.552 | −0.02 to −0.06 | 0/7 | | | | |
| nested (picked on the other six) | 0.5750 | **+0.0018** | 5/2 | | | | |

- Only weak earlier touches may be backdated (p_max 0.7): with 0.8, #15's own second touch
  (P 0.73) gets backdated over and #15 breaks; with any touch, DFL loses in every fold.
- max_s does not matter on DFL (3 = 6 = 9); #2 needs >= 6.2 s (header to the picked pick-up).
- v29_bd_b changes nothing else in the public window (#3–#17 as v29_tr_a). #1's end and #2's
  start now sit on the truth (−0.2 s): the earlier note that by eye the truth switched too early
  at #1–#2 was the header.
- Nested is positive (+0.0018, 5/2), unlike the touch-run vote (−0.003).

**Decision (user, 2026-10-28): v29_bd_b becomes the submission.** `validate_submission.py`: valid;
rebuild (`tools/eval_backdate.py --ours "p_max=0.7 max_s=9 loose_m=1.5" --name v29_bd_b`, on v29's
chain) byte-identical. Full match: 263 possessions, median 9.6 s, direct 40%, D 37.4%; public
0.812 / 0.901, 16 of 17 found, 1 spurious.

### 2026-10-29: v29_bd_b becomes v30, the submission; points per possession

**Decision (user, 2026-10-29):** v29_bd_b is renamed **v30** and is the submission. Rebuilt with
`python tools/eval_backdate.py --ours "p_max=0.7 max_s=9 loose_m=1.5" --name v30` on v29's chain:
byte-identical to v29_bd_b.csv, `validate_submission.py` valid, public 0.8121 / 0.9007, 263
possessions. The dead-ball head candidate that was called v30 (v30_d1n_add, public 0.697 / 0.897,
DFL 0.612 on top of v29) is now called **dh1**; its files keep their names.

Points per true possession in the public window (labels only). spellF1 = mean over IoU
thresholds 0.3–0.7 of 2·TP / (n_gt + n_pred); with 17 true and 16 predicted possessions, one
threshold hit is worth 0.0121 and a possession hit at all five 0.0606.

| # | context | len | IoU | thresholds hit | points |
|---|---|---|---|---|---|
| 1 | \|>[A]>H | 6.6 s | 0.96 | 5/5 | 0.0606 |
| 2 | A>[H]>A | 8.1 s | 0.96 | 5/5 | 0.0606 |
| 3 | H>[A]>H | 2.8 s | 0.62 | 4/5 | 0.0485 |
| 4 | A>[H]>A | 41.6 s | 0.97 | 5/5 | 0.0606 |
| 5 | H>[A]>D | 7.4 s | 0.72 | 5/5 | 0.0606 |
| 6 | D>[H]>A | 28.6 s | 0.76 | 5/5 | 0.0606 |
| 7 | H>[A]>H | 3.2 s | 0.63 | 4/5 | 0.0485 |
| 8 | A>[H]>A | 42.4 s | 0.98 | 5/5 | 0.0606 |
| 9 | H>[A]>H | 20.0 s | 0.62 | 4/5 | 0.0485 |
| 10 | A>[H]>A | 0.8 s | miss | 0/5 | 0 |
| 11 | H>[A]>H | 16.0 s | 0.31 | 0/5 | 0 |
| 12 | A>[H]>A | 1.3 s | 0.13 | 0/5 | 0 |
| 13 | H>[A]>D | 3.6 s | 0.78 | 5/5 | 0.0606 |
| 14 | D>[A]>H | 35.7 s | 0.97 | 5/5 | 0.0606 |
| 15 | A>[H]>D | 3.5 s | 0.83 | 5/5 | 0.0606 |
| 16 | D>[A]>D | 7.6 s | 0.94 | 5/5 | 0.0606 |
| 17 | D>[H]>\| | 11.8 s | 0.90 | 5/5 | 0.0606 |
| | | | | total | **0.8121** |

- #11 scores 0 despite IoU 0.31: its best overlap is the same predicted A spell that is matched
  to #9 (A runs on 12.3 s past #9), and matching is one-to-one.
- Lost: #9–#12 0.194 together (#10–#12 0.182, #9 0.012), #3 / #7 / #9 one threshold each (IoU
  just above 0.6). The spurious A (5.9 s, between #15 and #16, true D) costs through n_pred.

Video for the user's check of #10–#12: `python tools/render_labels.py evaluation/results/v30.csv
--possessions 10 12` (new option: cut by true possession number, 8 s before #10 to 5 s after #12)
-> `data/cache/renders/v30_p10-12.mp4` (local only, 0.5 min).

`tools/possession_view.py` now follows the submission: default `--sub` is v30.csv, and it rebuilds
v30's chain with the submission's params (touch-run vote w 2 / min_n 2 / gap 2 s / hold 4 s /
turnover; backdating p_max 0.7 / max_s 9 / loose 1.5). Team panel: v29 blend (dashed) and v30's
team input (blend + run vote); purple squares where backdating moved frames; run vote as a band
under the touches. With the default --sub it warns if the rebuild differs from v30.csv: it
reproduces it on all live frames. For the user's check: `python tools/possession_view.py 10 12`.

**User (viewer, 2026-10-29): the keeper problem persists.** It looks as if Barcelona's keeper plays
while a Barcelona player runs at him; the overlay ignores both, yet the touch still counts as
Barcelona's; it should be Athletic's. Athletic's own "keeper" stays still near the penalty area.
Note: the viewer's `--keepers` overlay only draws the rule; v30 runs on the raw tracking, so
nothing is removed from its inputs. `_kx` (own removed + other team's slot flipped) would flip
every other-team slot behind the line, the pressing attacker too. New mode **`_kd`**
(`ignore_impostors(mode="own_flip_deepest")`, `keeper_status(deepest_only=True)`): as `_kx`, but
per frame only the deepest such slot (the real keeper) goes to the defending team; the attacker
keeps his team. Synthetic (ghost still at 14 m, keeper slot at 2 m, attacker at 6 m): ghost removed,
keeper flipped, attacker kept (`_kx`: attacker flipped too).

v30's chain on it (`MTAH_VARIANT=_kd`: in_domain_team, then eval_backdate `--ours "p_max=0.7
max_s=9 loose_m=1.5" --name v30` -> `v30_kd.csv`, valid): **public 0.6625 / 0.888** (v30 0.812 /
0.901), 257 possessions, D 38.7%.

| version | #1 | #2 | #4 | #9 | #11 | #12 | #13 | #16 |
|---|---|---|---|---|---|---|---|---|
| v30 | 0.96 | 0.96 | 0.97 | 0.62 | 0.31 | 0.13 | 0.78 | 0.94 |
| v30_kd | 0.56 | 0.35 | 0.76 (k2) | 0.47 | 0.38 | MISS | 0.09 | 0.73 |

Same as `_kx` / `_ki` before (#4 breaks, #9–#13 merge into one A, #16 starts late), plus #1–#2
lose backdating's gain. Not adopted; v30 stays. Viewer: `$env:MTAH_VARIANT="_kd"` then
`python tools/possession_view.py 9 13` (v30_kd, removed ghost = grey cross, flipped keeper in
Athletic's colour).

**#10 (0.8 s H, always missed), user from the video:** a Barcelona player hits the ball and it
goes straight off an Athletic player: two touches, H then A. The viewer shows neither tick (only
the correct A touch after). On DFL (7 held-out touch_net folds, teacher touches within 0.2 s):
touches with an other-team touch within 0.5 s ("exchange", 1923) are caught 0.39 vs 0.60 for the
rest (ball seen: 0.50 vs 0.77); only 8% have no signal (peak < 0.3). The pick is the loss:
`touch_net.pick` takes one peak of P(home)+P(away) per +/-3 frames and gives it one team.
`tools/eval_touch_pick.py`:

| pick | P | R | F1 | exchange R |
|---|---|---|---|---|
| ours (0.7) | 0.91 | 0.58 | 0.71 | 0.39 |
| per team 0.5 | 0.82 | 0.63 | 0.71 | 0.48 |
| per team 0.35 | 0.72 | 0.68 | 0.70 | 0.57 |
| **ours + rival's peak >= 0.3 within 0.5 s** | 0.85 | 0.61 | 0.71 | **0.55** |
| ours + rival's peak >= 0.2 | 0.82 | 0.61 | 0.70 | 0.59 |

Touch level only; next: the chain on DFL (spellF1) with the extra rival touches. Keeper network
idea parked (user: keep v30; a corrupted-DFL benchmark first if it comes back).

**User (viewer, #10):** at the true start of #10 the red P(away touch) peaks, the blue does not.
So the network does not see Barcelona's hit as a home touch at all; a rival-peak pick could not
add it there.

**Rival touches through v30's chain on DFL** (`pipeline/fixes.rival_touches`,
`tools/eval_rival_touch.py`; second stage refine_b2 unchanged; the extra touches enter the margin
vote, run vote, backdating and restart handover): every setting loses. Best min_p 0.4 / within
0.3 s: 0.5702 (−0.0053, 1/6); nested the same; min_p 0.2 / 1 s −0.0151 (0/7). Split (min_p 0.4 /
0.3 s): post-steps only −0.0039 (2/5), first stage only −0.0007 (1/4). More touches at the touch
level (exchange recall 0.39 -> 0.55 at equal F1) do not help the spells. **Not adopted; v30 stays.**

**User (viewer, #10):** both teams close to the ball, ball tracked, players right; the ball track
probably never shows the Barcelona hit. Our match has no true touches (Smart Tagging is phases
only), so the viewer got a "ball track" panel instead: ball speed, velocity change over +/-3
frames (the old rule's signal, 2 m/s threshold dashed) and the old velocity-peak rule's touches,
none of it from the network. To check whether the hit is in the ball data at all.

**User (viewer, #10):** velocity change is above 2 m/s the whole time, yet no touch; both P(touch)
drop there, P(home touch) to 0. Viewer: orange dots on every velocity peak the rule considers
(with or without a player near), "to ball" zoomed to 15 m with the rule's 3 m reach dashed.

**Short possessions on DFL** (v30's chain, 7 matches; best IoU with one predicted same-team spell):

| true length | n | share | mean IoU | IoU >= 0.5 | any overlap |
|---|---|---|---|---|---|
| < 1 s | 82 | 4.1% | 0.03 | 0.02 | 0.26 |
| 1–2 s | 203 | 10.2% | 0.07 | 0.04 | 0.22 |
| 2–5 s | 490 | 24.6% | 0.28 | 0.28 | 0.50 |
| > 5 s | 1215 | 61.1% | 0.67 | 0.74 | 0.92 |

14% of possessions are under 2 s and v30 almost never catches them (2–4%); 2–5 s, a quarter of
all, only 28%. #10 (0.8 s) and #12 (1.3 s) are this class, not a one-off.

**Turnover network as cheaper switches under v30** (`tools/eval_turnover_switch.py`; held-out
turnover_net per DFL match; `learned.decode_hybrid(discount=...)`, default None = v30 unchanged).
Switch into T costs 20 − min(w × clip((peak P(change to T) − p0) / (1 − p0)), 19), peak over
±half_s. DFL reference rebuilds v30 exactly (0.5755).

| setting | DFL | vs v30 | folds |
|---|---|---|---|
| **w 19, p0 0, half 0.2 s** | **0.5773** | **+0.0018** | **6/1** |
| w 19, p0 0, half 0.5 s | 0.5766 | +0.0010 | 5/1 |
| w 19, p0 0, half 0.1 s | 0.5770 | +0.0014 | 4/3 |
| w 25 / 40 / 80, half 0.1 s | 0.5768 / 0.5760 / 0.5756 | | |
| w 19, p0 0.3 / 0.5 | 0.5749–0.5750 | −0.0006 | |
| w 5–10 | 0.5751–0.5757 | ~0 | |

Nested +0.0018 (6/1) on the first grid, +0.0014 (5/2) on the extended one; w 19 / half 0.2 is the
peak (beyond the cap, worse). Short possessions barely move: caught 1–2 s 0.04 -> 0.06, < 1 s and
2–5 s unchanged. **v30_to** (`--ours "w=19 p0=0 half_s=0.2"`, valid, 0.16% of live frames differ):
**public 0.800 / 0.900** (v30 0.812 / 0.901), 266 possessions; per possession as v30 except #5
0.72 -> 0.69 (below the 0.7 threshold) and #4 end +0.2 -> +0.3; #10 still missed, same spurious A.
A small DFL gain (about backdating's size), public −0.012 from one threshold crossing. Candidate
only; v30 stays.

**Keeper right without losing points (user: keep v30).** Variant rebuilds recompute every input
(hand path, columns, touches, second stage) from the changed tracking, so first a **local splice**
(`tools/eval_keeper_splice.py`): v30's inputs everywhere, the `_kd` inputs only where the rule
fires (+ margin), then v30's chain once. The rule fires on **1454 s** (Athletic's goal 599 s,
Barcelona's 871 s) against ~290 s of measured impostors.

| splice (_kd) | public spF1 / mF1 | changed possessions |
|---|---|---|
| both goals, margin 0–8 s | 0.642–0.663 / 0.886–0.888 | as v30_kd |
| Athletic's goal only | 0.774 / **0.911** | #9 0.62 -> 0.47, #11 0.31 -> 0.38, #12 0.13 -> MISS, #13 0.78 -> 0.09 |
| Barcelona's goal only | 0.686–0.706 / 0.876–0.878 | #1 0.96 -> 0.56, #2 0.96 -> 0.35, #4 0.97 -> 0.76 (k2), #16 0.94 -> 0.73 |

- Barcelona's goal: the rule is mostly wrong there (871 s), and that is what broke #1, #2, #4, #16.
- Athletic's goal (the user's scene): with the keeper right one A runs #9–#13; v30's wrong keeper
  made an H spell near #12 that split it. The cost is the short-possession problem (#10, #12 unseen).

**Evidence gate** (`keeper_status(evidence=True)`, mode `own_flip_evidence`, variant **`_ke`**): a
case run counts only if on >= 50% of it the slot is within 1 m of the team's deepest player (a
doubled keeper) or that player is still (a ghost). The rule then fires on **63.7 s** (Athletic's
goal 40.1 s, Barcelona's 23.6 s). Splice: Barcelona's goal changes nothing (0.00%, #1/#2/#4/#16
as v30); Athletic's goal 0.28% of live frames, public **0.774 / 0.911** (v30 0.812 / 0.901),
the same #9–#13 change. Plus turnover switches (w 10 / 19): 0.761, no help. **v30_ke.csv** written
(splice, margin 0, both goals) for the user's check. Not adopted: −0.038 spF1 on public for
+0.010 mF1; v30 stays.

**User: #12 is the same story as #10** (Athletic's keeper kicks, a Barcelona player heads it, it
touches an Athletic player, then Barcelona has it): a Barcelona contact straight into an
Athletic one, never seen. Solving it would solve #9–#13.

**Short-possession network** (`train_touch.py --target short`: touch_net architecture and inputs,
target = starts of controlled possessions shorter than 3 s (`SHORT_S`), 7 held-out folds +
`short_net_all.pt`). Held-out, within 0.5 s of a short start: **net@0.3 P 0.07 R 0.04**, @0.5
0.09 / 0.03 (the touch rule: P 0.01, R 0.30). It does not learn them. As cheaper switches in v30's
chain (`eval_turnover_switch.py --net short_net`): w 10 +0.0001 (1/0), w 19 −0.0006, w 40 −0.0013;
nested −0.0013; caught 1–2 s 0.04 -> 0.05. **Negative.** From this tracking (our ball pattern) the
start of a short possession is not learnable with these inputs; the turnover network (+0.0018)
remains the only small gain.

**Broadcast ball at the possession starts** (`tools/broadcast_contacts.py`, public window, flags
only; broadcast mapped by `broadcast_align.npz`). Per true start, ±0.5 s: the broadcast is
aligned at only 2 of 17 starts (#5, #15; 10% of the windows), **0% at #9–#13**, so it cannot help
there (the TV is on something not alignable: replay, close-up). Tactical ball: seen 96%, a touch
rule velocity peak at 71% of starts, one with the new team's player nearest within 3 m at 41%.
#10 and #12: no peak within ±0.5 s; within ±1 s #10 has a peak with a Barcelona player nearest
within 3 m (#12 a peak, Athletic's player nearest). Broadcast direction closed for #10 / #12.

### 2026-10-30: dead-ball head (dh1) on top of v30

`tools/eval_dead_v30.py`: the dead-ball head (d1n, add-only: the hand path's D is kept, the head
may only add D; held-out probabilities `dead_d1n.npz`, deployed `models/dead_d1n_s0..2.pt`, no
retraining) extends the hand path's D, then v30's chain (touch-run vote + backdating) runs as is.

| | DFL (7, leave one out) | public spF1 / mF1 | full-match D | possessions |
|---|---|---|---|---|
| v30 | 0.5755 | 0.812 / 0.901 | 37.4% | 263 |
| **v30_dh** (0.5 / 0.5 / 25, dh1's setting) | **0.6149 (+0.039, 7/0)** | **0.812 / 0.917** | 48.8% | 243 |
| best cell 0.5 / 0 / 25 | 0.6160 (+0.040, 7/0) | | | |
| nested | 0.6144 (+0.039, 7/0) | | | |

- All 32 grid cells win 7/0 on DFL. True D called live 13.0% -> 2.1%; live called D 12.7% -> 13.1%.
- Our match: 11.4% of frames go live -> D, none D -> live, team changes 0.02%. Public spellF1
  unchanged; per possession as v30 (16/17 matched).
- Caveat as with dh1: full-match D 48.8% is at the top of the plausible range (37–50%).
  **v30_dh.csv** written (valid). Candidate; v30 stays until checked on the full match.

**Where live play is called D** (`tools/dead_overcall.py`, DFL): each true-live frame called D
is put by its called D stretch into early (before a true stoppage it overlaps), late (after it:
restart called late) or phantom (the stretch overlaps no true stoppage). Oracle = those frames
take the labelled team (upper bound of a fix).

| | live called D | early | late | phantom | phantom stretches |
|---|---|---|---|---|---|
| v30 | 12.7% | 2.4% (oracle +0.026) | 4.8% (+0.059) | 5.5% (+0.092) | 19.5 / 90 min, median 3.9 s |
| v30_dh | 13.1% | 2.6% (+0.026) | 4.9% (+0.060) | 5.6% (+0.094) | 19.3 / 90 min, median 4.0 s |

- The head doesn't add any of this; it is the hand path's D. Phantom D (D out of nowhere) is
  the largest single loss left on DFL, then late restarts.
- Caveat: on the public window live play is never called D, so part of DFL's phantom D may come
  from our degradation of DFL (gap pattern, animated dead ball).

**What phantom D looks like** (`tools/phantom_diag.py`, DFL, v30_dh). Note: the bidirectional
second stage only decides the team; D vs live is the hand path's (plus the add-only head).

| frames | ball seen | gap (s) | ball speed | player speed | head P(dead) |
|---|---|---|---|---|---|
| phantom | 0.35 | 9.9 | 6.5 | 1.53 | 0.16 |
| true stoppage called D | 0.37 | 22.5 | 3.3 | 1.05 | 0.96 |
| live called live | 0.85 | 2.1 | 7.0 | 2.49 | 0.01 |

- 143 phantom stretches: ball missing throughout 22%, partly 18%, seen throughout 59%. Players
  slower than in normal play and the ball often missing: the hand rule's slow + gap terms.
- **The dead-ball head already separates them:** mean P(dead) per stretch, phantom median 0.00,
  stretches holding a true stoppage 0.91. Below 0.3: 88% of phantom stretches, 3% of true ones
  (below 0.5: 93% / 5%). So a stretch-level removal gate on the head is the next test (the free
  head, frame by frame, removed real stoppages on our public window, 2026-10-27).

**Remove gate** (`eval_dead_v30.py --gate`): after the add-only head, every D stretch whose mean
head P(dead) is below T (optionally only if <= MAX_S s) is made live; the team decoder fills it.

| | DFL | vs v30 | folds | live called D | public spF1 / mF1 | full D |
|---|---|---|---|---|---|---|
| v30_dh (no gate) | 0.6149 | +0.039 | 7/0 | 13.1% | 0.812 / 0.917 | 48.8% |
| gate T 0.05 | 0.6290 | +0.054 | 7/0 | 10.7% | | |
| gate T 0.1 | 0.6284 | +0.053 | 7/0 | 10.5% | 0.776 / 0.850 | 47.6% |
| gate T 0.3 | 0.6274 | +0.052 | 7/0 | 10.0% | 0.776 / 0.850 | 46.7% |
| gate T 0.5 | 0.6254 | +0.050 | 6/1 | 9.4% | 0.776 / 0.850 | 44.4% |
| nested | 0.6259 | +0.050 | 6/1 | | | |

- DFL: +0.014 over v30_dh (all cells 7/0 up to T 0.3). Length caps (10 / 20 s) do worse.
- **Public: −0.036 spF1, −0.067 mF1 at every T.** Per possession (`spell_table.py`) one change
  only: the true stoppage between #16 and #17 (14.5 s) is removed whole, so #16 runs on +1.6 s
  (IoU 0.94 -> 0.78) and #17 starts 12.9 s early (0.90 -> 0.48). The head's P(dead) is below 0.1
  on a real stoppage of ours: 1 of the window's 4 stoppages (DFL: 3% of true-stoppage stretches).
  The domain gap again. **Rejected** as it is; v30_dh stays the candidate.

**Why the head misses that stoppage** (`tools/stoppage_profile.py`): the public window's true
stoppages >= 2 s, numbered S1–S4 (S4 is the one before #17), against DFL's 705 (held-out head):

| | length | ball seen | longest gap | team speed | hand D | head P(dead) |
|---|---|---|---|---|---|---|
| S1 | 5.5 s | 7% | 5.1 s | 1.55 | 0% | 0.31 |
| S2 | 20.4 s | 100% | 0 | 1.12 | 95% | 0.51 |
| S3 | 18.3 s | 20% | 14.6 s | 1.16 | 64% | 0.79 |
| **S4** | 14.5 s | **100%** | 0 | **2.13** | 91% | **0.01** |
| DFL median (p10–p90) | 19.9 s | 42% | 9.5 s | 1.18 (0.82–1.71) | 99% | 0.97 (0.69–1.00) |

- S4 is not a gap problem: the ball is seen throughout. The players move at 2.13 m/s, the 97th
  percentile of DFL's stoppages; the head reads "players running" as live play.
- All four of our stoppages sit in DFL's bottom 12% of head P(dead) (S1–S3 at 5–12th
  percentile). The head is generally less sure of our stoppages, not only S4.
- DFL's head misses (< 0.1) are 2–5% at every ball-seen share, so more gaps in training would
  not reach S4.

**Ball-out rule (user's idea):** S4 is a shot that goes out; the players run back while the
keeper sets up the goal kick, so the ball is seen and out of play. `eval_dead_v30.py`:
`--protect` keeps the remove gate off every D stretch in which the ball is seen out of play
(phantom stretches on DFL have ball_out 0.00); `--hold S` makes D from the ball going out until
the first touch with it back on the pitch, at most S s.

| | DFL | vs v30 | folds | live called D | public spF1 / mF1 | full D | poss |
|---|---|---|---|---|---|---|---|
| v30_dh | 0.6149 | +0.039 | 7/0 | 13.1% | 0.812 / 0.917 | 48.8% | 243 |
| gate 0.1, no protect | 0.6284 | +0.053 | 7/0 | 10.5% | 0.776 / 0.850 | 47.6% | 235 |
| **gate 0.1 + protect (v30_dhgp01)** | **0.6295** | **+0.054** | **7/0** | 10.5% | **0.812 / 0.917** | 48.0% | 238 |
| gate 0.3 + protect | 0.6292 | +0.054 | 7/0 | 10.1% | 0.812 / 0.917 | 47.5% | 233 |
| nested (protect, T grid) | 0.6278 | +0.052 | 7/0 | | | | |
| hold 60 s, no gate | 0.6030 | +0.027 | 6/1 | 15.3% | 0.812 / 0.919 | 50.4% | 244 |
| gate 0.3 + protect + hold 60 | 0.6182 | +0.043 | 6/1 | 12.2% | 0.812 / 0.919 | 49.2% | 235 |

- Protect brings the public window back exactly (per possession as v30_dh; #16 0.94, #17 0.90)
  and is slightly better on DFL too. All T from 0.05 to 0.5 win 7/0 with it.
- Hold costs 0.011–0.012 on DFL (restarts called late: live called D up) for +0.002 public mF1:
  rejected.
- **v30_dhgp01** (`--ours 0.5 0.5 25 --gate 0.1 0 --protect`, valid): DFL +0.054 over v30,
  public unchanged, 0.85% of the match goes D -> live. New candidate; v30 stays the submission.
- **Adopted as v31 (user, 2026-10-30)** after watching the full match: mostly better, a few small
  tweaks left. Rebuild gives the same file byte for byte (ef2ea808…).

**Full-match render for a video check:** `tools/render_labels.py --compare B.csv` shows a second
submission's label next to the first and DIFF where they differ. Both halves rendered as
v30_dhgp01 vs v30 (`data/cache/renders/`, gitignored). Differences of at least 1 s: 55 (42 D
added, 13 D removed).

### 2026-11-02: does DFL have the #10 / #12 pattern? (long kick, then a short possession)

#10 and #12 are both a long kick by Athletic's keeper, a Barcelona header, then straight off an
Athletic player. In v31, missing #10 merges #9 and #11 into one A spell: #9 ends 12.3 s late
and #11 (IoU 0.31) scores 0, so a rough split near #10 would win most of #10-#12's ~0.18.
Before building a "long keeper kick -> contest at the landing" rule, a check of DFL
(`tools/kick_shorts.py`; long kick = Play / GoalKick / FreeKick whose next on-ball event is
>= 35 m away within 5 s, deep = within 25 m of the own goal line; window 6 s; short < 2 s):

| labels | long kicks (deep) | deep: other team short, then kicker | all long: same | short possessions | after a long kick (deep) |
|---|---|---|---|---|---|
| controlled (benchmark truth) | 219 (89) | 4 (4.5%) | 10 (4.6%) | 285 | 11 (3.9%) / 4 (1.4%) |
| raw provider flags | 219 (89) | 12 (13.5%) | 21 (9.6%) | 919 | 25 (2.7%) / 12 (1.3%) |

- DFL has very few of these: 4 cases in 7 matches by our truth, 12 by the raw flags. Too few to
  train on or to tune a rule on.
- Our controlled-label filter removes two thirds of them. It hands possessions without a
  controlled action of their team to a neighbour, and a header that is a defensive clearance is not
  controlled. The public truth keeps #10 / #12 (a header into the opponent), so on this exact
  pattern **DFL's truth uses a different definition from the organisers'**. DFL can't judge
  a fix here; at best it can bound the damage a rule does elsewhere (false splits).
- Base rate: after a deep long kick the other team gets a short spell 4.5-13.5% of the time,
  so a rule needs a strong landing signal to be right more often than wrong.

**Same pattern in Metrica's 3 games** (`tools/kick_shorts_metrica.py`, hand-tagged events,
same long-kick definition; pattern = the other team's first event after the kick, then the
kicker's own event again within 2 s):

| kicks | n | no contact by the other team | contact, kicker back < 2 s | contact, other team keeps |
|---|---|---|---|---|
| deep | 46 | 20 | **16** | 10 |
| not deep | 68 | 59 | 4 | 5 |

- The pattern is common after deep kicks here (16 / 46), but the contacts are mostly duels:
  CHALLENGE AERIAL-LOST 7, GROUND-LOST 3, GROUND-FAULT-LOST 2, RECOVERY 2 (+5 single others).
  Metrica tags both sides of a duel, so some are the loser of a header the kicker's team won.
  Metrica's labels (built from passes / set pieces / confirmed recoveries) don't make these
  possessions either.
- All we have, then: 4 (DFL truth) to 12 (DFL raw flags) plus ~16 duel-like Metrica cases, none
  labelled with the organisers' definition. More data is needed for this pattern.

**More external data for short possessions (checked 2026-11-02):**
- **SkillCorner open data** (GitHub, MIT, credit SkillCorner): 10 A-League 2024/25 matches,
  broadcast tracking at 10 fps (players extrapolated off camera, `is_detected` flag, ball in
  `ball_data`), per-match `dynamic_events.csv` with `player_possession` rows (start / end per
  player possession, so short possessions are labelled directly) and `phases_of_play.csv`.
  At 10 fps a 0.8 s possession is 8 frames.
- **PFF FC World Cup 2022** (form on PFF's blog, instant access): all 64 matches, broadcast
  tracking (`{game}.jsonl.bz2` per game), event data with possession events, metadata and rosters.
  Details (fps, ball height, event types) are in their PDF specs, still to check.
- Plan: PFF if its events mark brief contacts (challenges, touches); SkillCorner as a quick free
  second set. Both go through the DFL treatment (our gap pattern, blur), leave one match out.

### 2026-11-04: SkillCorner open data, the #10 / #12 pattern counted

The user is downloading PFF World Cup 2022 into `data/external/pff/`. Meanwhile SkillCorner:
`data/external/skillcorner/repo` (git clone + `git lfs pull`, gitignored). The repo now has
**20 matches** (A-League 2024/25), not 10; tracking ~90 MB per match, 10 fps.

- Tracking per frame: `ball_data` (x, y, **z**, `is_detected`), 22 `player_data` (x, y,
  `is_detected`), a per-frame `possession` (player, group) and the camera's field-of-view corners.
  In one match: ball position present in 73% of frames, detected (not extrapolated) in 60%;
  players detected 58%. Broadcast, so its gaps look more like ours than DFL's.
- `dynamic_events.csv`: 18,710 `player_possession` rows (median 1.2 s, quarter under 0.5 s) with
  start types (pass_reception, pass_interception, recovery, restarts), end types and `is_header`.

`tools/kick_shorts_skillcorner.py` (same definitions as the DFL / Metrica checks: long kick
>= 35 m within 5 s, deep = within 25 m of own goal line, other team's spell < 2 s then the kicker
back within 2 s):

| kicks | n | kicker keeps | other team short, kicker back | other team keeps |
|---|---|---|---|---|
| deep | 454 | 71 | **46 (10.1%)** | 337 |
| not deep | 1711 | 357 | 183 | 1171 |

- 46 deep cases, against 4 (DFL truth) / 12 (DFL raw) / ~16 duel-like (Metrica). 21 of the 46
  contacts are headers, 30 start as a pass interception, 15 as a recovery; 41 are a single player
  possession.
- Here these contacts **are labelled as possessions** (a player possession of the other team),
  which matches the organisers' treatment of #10 / #12, unlike DFL's controlled filter.
- Base rate: 1,027 of 3,929 team spells are < 2 s and between two spells of the other team (923
  of them a single player possession). Plenty of short possessions to train on, not just after
  long kicks.

**Next:** a loader (`tools/skillcorner_data.py`, like `dfl_data.py`): tracking to our frame
format, labels from the player possessions (team spells, D from the restarts / gaps between
possessions), then check the per-frame `possession` field against them. Then PFF once it is in.

### 2026-11-04 (later): SkillCorner loader; PFF World Cup 2022 checked

**`tools/skillcorner_data.py`** (cache in `data/cache/skillcorner/`). The 10 fps tracking is
resampled to 25 fps (linear between consecutive frames only) into DFL's arrays: players (x, y,
speed from positions), `pdet`, ball (detected only, NaN otherwise), `ball_est` (with SkillCorner's
extrapolation), `tracked`, team, period, labels, and the player possessions as events (start,
end, team, header, start / end type). Labels:
- inside a player possession: its team; between two in live play: the earlier team (the switch
  is at the next team's first touch);
- dead (interruption after the earlier one, restart start type or interruption before the next
  one): the earlier team until the ball is first seen outside the lines, D until the ball was last
  still or slow before the restart, then the restart team;
- before the first / after the last possession of a period, and frames with no players
  (replays, cut-aways): X.

| | poss / live min | median | < 2 s | D | graded frames | ball seen (graded) | players detected |
|---|---|---|---|---|---|---|---|
| 20 matches (range) | 3.9-6.3 | 6.2-9.0 s | 10-19% | 17-35% | 66-81% | 75-85% | 49-66% |
| public window | 4.2 | 7.6 s | 12% | | | | |

- The profile is close to the public window's, much closer than DFL's raw flags (6.5-11 / min).
- SkillCorner's per-frame `possession` field agrees with these labels on 86% of H / A frames
  (89% inside possessions, 83% in flights; no time offset helps, best +0.002 at -0.5 s), in
  disagreement runs of median 1.4 s. It is a looser automatic label; the events stay the truth.

**PFF FC World Cup 2022** (`data/external/pff/FIFA World Cup 2022/`, 64 matches, 5.3 GB):
- Tracking: Sportlogiq broadcast tracking at the video's 29.97 fps, 11 + 11 players (estimated
  off screen: 69% of player positions in one match are ESTIMATED), ball with z in 74% of frames
  (63% of those VISIBLE, the rest estimated). Each frame carries the game / possession event it
  belongs to. A smoothed ball placed on the carrier from the events is also there (not usable as
  input: it comes from the labels).
- Events: OTB game events ("a player has control over the ball", start / end time, team) with
  possession events (pass, cross, clearance, challenge, rebound, shot, carry, touch), body part
  (HE = head), challenge type and outcome, and `sequence` = PFF's own team possession
  ("an uninterrupted possession by one team").

`tools/kick_shorts_pff.py`, same definitions, from the OTB events:

| kicks | n | kicker keeps | other team short, kicker back | of those, own PFF sequence | other team keeps |
|---|---|---|---|---|---|
| deep | 1176 | 474 | 172 | **56** | 530 |
| not deep | 2053 | 1218 | 125 | 51 | 710 |

- OTB events include both sides of a duel: 97 of the 172 deep contacts are challenges, 96 have two
  OTB events, 58 are headers. PFF gives the contact its own sequence (a team possession) in 56 of
  them, about a third; in the others the kicker's team keeps the sequence.
- Base rate: 8,322 of 19,933 OTB team runs are < 2 s between two runs of the other team, but only
  2,549 of them are their own PFF sequence. For labels, PFF's `sequence` is the possession, not
  the OTB team.

Cases of the deep kick -> short contact pattern, labelled as possessions: SkillCorner 46 (20
matches), PFF 56 (64), against 4 on DFL. Enough to train and check a rule on.

**Next:** a PFF loader (`tools/pff_data.py`): tracking at 29.97 fps resampled to 25, labels from
`sequence` (team from the sequence's OTB events, D from OUT game events and restarts), the raw
(not smoothed) ball. Then the short-possession check: v31's chain on SkillCorner and PFF with the
DFL treatment (our gaps, blur), how many of these 46 + 56 it catches, before any retraining.

### 2026-11-04 (evening): PFF loader; two kinds of PFF sequence

**Two `sequence` fields.** The event files' `sequence` column was added in May 2025 (change log)
and is coarser: in match 10502, 295 sequences against 311 in the tracking files' game events,
a third of them holding OTB events of both teams, and a different numbering (39 of 2,145 OTB
events share the number). The tracking files carry the original: `sequence` + `home_ball` per
game event, `home_ball` constant inside a sequence and equal to the toucher's team in 96% of OTB
events. The loader uses the tracking version; the event file only gives set piece type and
possession event kind, matched by start time (2,147 / 2,147).

**`tools/pff_data.py`** (cache `data/cache/pff/`, `--jobs N`, ~10 min for 64 with 4): raw (not
smoothed) players and ball resampled 29.97 -> 25 fps into DFL's arrays, plus `pvis` / `bvis`
(VISIBLE vs ESTIMATED), events (start, end, team, tracking sequence, kind, start time). Labels:
the sequence's team from its first to last OTB event; between sequences the earlier team until an
OUT event (D from it to the restart), D from the earlier end when the next starts with a set
piece, else the earlier team until the next starts; X before kick-offs, after END, in VID.

| 64 matches | poss / live min | median | < 2 s | < 5 s | D | ball present | ball visible | players visible |
|---|---|---|---|---|---|---|---|---|
| min | 3.2 | 5.0 s | 9% | 30% | 28% | 53% | 29% | 25% |
| median | 4.6 | 7.1 s | 14% | 39% | 42% | 68% | 41% | 38% |
| max | 7.5 | 10.3 s | 20% | 50% | 52% | 79% | 54% | 51% |
| public window | 4.2 | 7.6 s | 12% | | | | | |

- Possession profile close to the public window again. D is higher than SkillCorner's (25%):
  PFF's D runs from the OUT tag to the restart touch, and restart touches come late (the taker's
  OTB starts when he plays it).
- The players are mostly estimated (median 38% visible), more than SkillCorner (49-66% detected).

**Pattern count with the tracking sequences** (`tools/kick_shorts_pff.py`; `--event-seq` gives the
earlier numbers): of the 172 deep long kicks followed by a < 2 s contact of the other team and the
kicker back, **152 are their own sequence** (event-file sequences: 56). Not deep: 89 of 125.
Short OTB runs between two of the other team: 4,067 of 8,322 are their own sequence (event file:
2,549). So PFF's original sequences count nearly every such contact as a team possession, as the
organisers do with #10 / #12; the 2025 column merges most of them into the kicker's possession.

Cases of the pattern labelled as possessions: SkillCorner 46 (20 matches) + PFF 152 (64).

**Next:** v31's chain on SkillCorner and PFF with the DFL treatment (our gaps, blur): spellF1
overall and how many of the 46 + 152 short contacts it catches, before any retraining.

### 2026-11-04 (night): v31 unchanged on SkillCorner and PFF

`tools/run_v31_external.py <skillcorner|pff>` (results in `data/external/results/v31_<set>.csv`,
~50-70 s per match). Our match's deployed recipe, nothing trained on the target's truth: deployed
touch network, hand path, gbm_dfl + gru_dfl blend, the players-only team model trained on the 7 DFL
matches plus this match's own pseudo-labels (the unsupervised step our match gets too), stage 1,
refine_v29 x5, touch-run vote, dead head d1n x3 (add-only), ball-out protected gate T 0.1,
backdating. Ball: the set's own broadcast ball (gaps and moving dead ball already there), blurred
like DFL; no animation, no extra gaps. Extra time (3 PFF matches) is decoded as its own blocks.

Short truth spells = H / A spells < 2 s with spells of the other team directly on both sides (no D
in between); caught = a predicted spell of that team with IoU >= 0.5. Deep-kick subset = those
starting at the landing of a deep long kick of the other team (event positions, as before).

| set | matches | v25 | v31 | v31 - v25 | short caught v25 / v31 | after deep kick v25 / v31 |
|---|---|---|---|---|---|---|
| SkillCorner | 20 | 0.5785 | **0.5989** | +0.020 (18 / 2) | 13 / **22** of 213 | 2 / 2 of 13 |
| PFF | 64 | 0.5231 | **0.5366** | +0.014 (47 / 17), 95% CI +0.009 to +0.018 | 56 / **86** of 1,442 | 5 / 8 of 120 |
| DFL (for scale, leave one out) | 7 | | 0.6295 | | | |

- v31's steps over v25 carry to two new providers without retraining (+0.014 to +0.020,
  most matches up). Levels are below DFL: broadcast tracking, players largely estimated (PFF median
  38% visible), PFF's fine-grained sequences.
- **Short possessions are missed almost everywhere:** 6-10% of short truth spells caught, 6-15%
  after a deep long kick. The #10 / #12 weakness is general, not specific to our match, and these
  sets give 213 + 1,442 labelled short spells (13 + 120 after deep kicks) to work on.
- Only 13 of SkillCorner's 46 event-level deep-kick cases are < 2 s spells in the frame labels:
  the possession runs to the next team's touch, so the flight after the contact often takes it
  past 2 s (and a replay cut can end a spell early).

**Next:** train on the short spells. Candidates: (a) the touch / turnover network retrained with
SkillCorner + PFF (broadcast-like ball, contacts labelled), (b) a short-possession head on the
second stage that proposes brief other-team spells at landings, judged leave-one-match-out on
SkillCorner + PFF and then DFL + the public window.

### 2026-11-05: short possessions, (a) touch network with SC + PFF, (b) short head: both dropped

Setup: 4 fixed folds over the 84 external matches (every 4th match of each set,
`run_v31_external.fold_of`); v31's intermediates cached per match (`--save`, `data/cache/ext_v31/`).
`tools/ours_chain.py` rebuilds v31 on our match in one process with any touch network (checked:
0 frames differ from v31.csv). The user decides on the public spellF1.

**(a) Touch network + SkillCorner + PFF** (`tools/train_touch_ext.py`, models in
`data/cache/models/touch_ext/`): DFL as before plus the other folds' external matches, touches
from the events (possession / OTB start, plus the release if >= 0.3 s). On held-out matches touch
F1 rises (SC 0.63 -> 0.83, PFF 0.49 -> 0.56) but recall of possession starts falls (~0.57 -> 0.48
at pick 0.5; ~0.3-0.4 at the deployed 0.7: less confident). v31 chain with the fold networks
(`run_v31_external.py --touch-dir ... --pick 0.5 --tag tx`):

| | v31 | (a) | up / down | short spells caught |
|---|---|---|---|---|
| SkillCorner (20) | 0.5989 | 0.6037 | 10 / 10 | 22 -> 18 of 213 |
| PFF (62 of 64) | 0.5368 | 0.5725 | 54 / 8 | 85 -> 40 of 1403 |
| public window, pick 0.7 / 0.5 / 0.35 | 0.812 | 0.614 / 0.675 / 0.679 (macro 0.933-0.943) | | |

The PFF gain comes with PFF-made labels on both sides; on our match the network merges possessions
(195-223 instead of 238) and loses 0.13-0.20 public spellF1. Dropped.

**(b) Short-possession head** (`tools/short_head.py`): LightGBM per frame on v31's outputs and
evidence seen from v31's team T vs the other O (touch probabilities, distances, co-movement, goal
distances, ball height / speed / gaps, refiner and stage-1 P(T), vote, P(dead), position in
v31's spell, 0.5 / 1 / 2 s context). Target: truth is O there and that spell is < 3 s. Runs of
smoothed P > T (min length 0.2-1 s, max 4 s) inside one v31 spell go to O.
- 4-fold CV: best +0.003 (T 0.5, 1 s; 55 / 27), nested +0.0022 (48 / 35); short spells caught
  108 -> 164 of 1,655, with false insertions eating most of it.
- Public window: unchanged at every setting (0.812 / 0.917); max smoothed P inside the window is
  0.23, so it never proposes #10 / #12. Full match: 0-7 insertions. Dropped.

Neither catches #10 / #12. Next (user's idea): a hand rule at the landing of a long kick, the
team that closes on the landing point gets the ball (distance, closing speed, touch), checked on
SkillCorner + PFF and then the public window.

### 2026-11-05: landing rule for long kicks (user's idea): dropped

`tools/landing_rule.py`. Kick = a v31 touch pick of team T while v31 says T; landing L = the first
touch-network peak after it (P >= 0.3), which must be >= 35 m from the kick point within 5 s
(the first version took the first far peak and so chained short passes: 18,653 "long kicks",
fixed). Who wins at L: `dist` (O's nearest player closer by a margin), `close` (distance minus
0.5 s x closing speed), `touch` (max P(O touch) within +/- 0.4 s beats T's), `both`. If O wins and
v31 says T: O until T's next touch within KEEP s (2 / 3 / 5), or KEEP 0 = turnover (O until v31's
next change, at most 10 s). Deep-only or all kicks; margins 0-4 m.

- SkillCorner + PFF: 1,614 long kicks (19 per match), 540 deep. Best setting +0.0001 (touch,
  deep, keep 2: 15 / 18 matches, fired 42, precision 0.31); turnover variant -0.0003 to -0.0006.
- Why: the closer team does predict who gets the ball (O has it within 1 s after 47-49% of
  landings when O is closer, 17-25% when T is), and so does the touch side (50% vs 15%). But
  "O briefly, then T back" follows only 4-6% of long kicks whoever is closer, and v31 already
  gives O most of the real turnovers (31% of deep landings vs 36% in the truth). Where the rule
  disagrees with v31, v31 is right 70-90% of the time.
- Public window: 26 long kicks in our match (8 deep), 3 in the window (1 deep, though #10 and #12
  both start from keeper kicks). The short variants never fire (0.812 / 0.917 unchanged); the
  turnover variant fires once in the window: 0.800.

Dropped. v31 stays the submission. #10 / #12 remain open: on our tracking even the deep keeper
kicks before them are found only in part.

## Current situation

**Best so far, on both references (2026-10-26).** The public window is the main reference (DFL is
only our match after our own changes to it), but both are tracked and worked from:

| reference | best | score | the other reference |
|---|---|---|---|
| public spellF1 (5 min, 17 possessions) | **v29** (v27 + second-stage network, 12.5 fps, 5 seeds) | 0.697 / 0.882 | DFL 0.571 |
| realistic DFL (7 matches, folds won) | **v29** | 0.571 vs 0.560 (6/7, passes) | public 0.697 / 0.882 |
| **submitted file (since 2026-10-30)** | **v31** (v30 + dead-ball head + remove gate with the ball-out rule) | public 0.812 / 0.917 | DFL 0.6295 (v30 0.5755) |
| previous submission (2026-10-29 to 2026-10-30) | v30 (= v29_bd_b: v29 + touch-run vote + backdating) | public 0.812 / 0.901 | DFL 0.5755 (v29_tr_a 0.5731, v29 0.571) |
| previous submission (2026-10-28, morning) | v29_tr_a (v29 + touch-run vote) | public 0.727 / 0.886 | DFL 0.573 (v29 0.571, 6/1; nested −0.003) |
| previous submission (2026-10-26 to 2026-10-28) | v29 (v27 + second-stage bidirectional team network) | public 0.697 / 0.882 | DFL 0.571 (v27 0.560, v25 0.551, v21 0.541) |

- **DFL open data is the main test set (2026-10-24):** real tracking, provider possession flags
  filtered to controlled possession, our gap pattern cut in. v19 scores 0.567 / 0.833 there.

- **Metrica is useful for the team decision but not for the ball:** its ball is reconstructed
  (absent in stoppages, sitting on the carrier), so models trained on it misread our dead balls.
  Learned models beat the hand model on Metrica (0.822 vs 0.793), but transfer badly to our match
  (v16: public 0.49 / 0.67).
- **The submission is v29** (2026-10-26): v27's pipeline plus the second-stage bidirectional
  team network (12.5 fps, 5 seeds, alpha 0.5, switch 20). Rebuild, from a clean state:
  `python tools/in_domain_team.py --ours --skip-dfl --source dfl+decoded --weights 0.5`, then
  `python tools/tune_refine.py --bin 2 --seeds 5 --tag v29 --ours 0.5 20` (models in `models/`:
  gbm_dfl, gru_dfl, touch_net_all, po_team_v25, refine_v29_s0..4). The full chain gives a
  byte-identical file. Before it, **v27**: v25 plus the touch-margin vote and the restart
  handover. Rebuild: `python tools/in_domain_team.py --ours --skip-dfl --source dfl+decoded
  --weights 0.5`, then `python tools/eval_fixes.py --skip-dfl --ours "A max_s=3 slow_ms=2 + D
  w=0.5 vote_s=2" --name v27` (byte-identical on rebuild). Before it, v25_w0.5:
  v21 plus the in-domain players-only team vote
  (`python tools/in_domain_team.py --ours --skip-dfl --source dfl+decoded --weights 0.5`; model
  `models/po_team_v25.txt`, a rebuild gives a byte-identical file). Before it: v21
  (`python pipeline/run_learned.py --name v21 --source dfl --model blend
  --hybrid 10`): v20 decides D vs live, the learned blend decides the team. On our match: 261
  possessions, D 37.3%, public 0.66 / 0.88. Realistic test (DFL, blur + gaps): 0.579 / 0.843.
- **DFL's ball is frozen while dead**; `dfl_data.prepare_ball` animates it. The realistic benchmark
  is DFL with an animated dead ball, blur and gaps: v20 0.535, hybrid 0.542. Learned stoppage
  detection still doesn't transfer to our match, in either training setup.
- The semi-Markov decoder doesn't help: short possessions are limited by the evidence.
- **Next:** the shortest possessions; learned evidence (GBM / GRU) retrained on DFL with the
  touch network as input.
- The public window misranked our versions. It is now only a correctness check, as the ground
  rules always said.
- Hand-built versions all land between 0.68 and 0.71 on Metrica. The next gain has to come from
  learning.
- `pipeline/`, `tools/` and this journal are **not committed yet**.

## Resume here (paused 2026-10-27 02:35)

**Update 2026-10-30: the submission is v31** (v30 + dead-ball head + remove gate with the ball-out
rule; public 0.812 / 0.917, DFL 0.6295; rebuild byte-identical). Next: the keeper problem.

**Update 2026-10-29: the submission is v30 (= v29_bd_b; the old dead-ball "v30" is now dh1)** (v29 + touch-run vote + backdating, public
0.812 / 0.901, DFL 0.5755; validated, rebuild byte-identical). #2 and #15 are fixed. Still open in
the public window: #9–#12 (keepers, the short H spells #10 / #12), #6's early start, #5's stoppage.
The notes below are from before.

State: **v29 was the submission** (public 0.697 / 0.882, DFL 0.571). **v30 candidate** = v29 +
add-only dead-ball head (public 0.697 / 0.897, DFL 0.612, +0.041 7/7; full D 48.8%, 230 poss at
the edge of plausible): waiting on the user's decision whether to submit it.

1. Finish the leak-free first-stage build (~20 min, only the missing files):
   `python tools/clean_stage1.py`  (ends with an aggregate accuracy line, pair vs single)
2. Second stage, leak-free, compare with v29's configuration (`--tag b2`, 5 seeds: grid mean
   +0.0085, alpha 0.5 / switch 20 = 0.571; nested 0.571):
   `python tools/tune_refine.py --bin 2 --seeds 5 --stage1 clean --tag b2s5c1`   (~40 min: how
   much the leak inflated v29's gain)
   `python tools/tune_refine.py --bin 2 --seeds 5 --stage1 clean3 --tag b2s5c3`  (~70 min: the
   candidate, 3 gap patterns)
3. If clean3 wins: `--ours 0.5 20 --tag v31` (deployed models use the single-exclusion inputs),
   public check with `tools/transfer_check.py`, rebuild identity, then put the dead-ball head
   (`tools/eval_dead_refine.py`, its team probabilities now come from refine_b2 / refine_v29:
   update TEAM_TAG and the model names) on top, and ask the user which becomes the submission.
4. Dead-ball follow-ups: train the dead head on the clean3 inputs too; try add-only with a cap
   on the added D (full D <= ~45%).

## Next steps

**Next session (agreed 2026-11-02): more data for the short possessions (#10 / #12).** Missing
#10 merges #9 and #11 into one A spell (#9 ends 12.3 s late, #11 scores 0); missing #12 hurts
#11 from the other side. Catching #10 and #12 fixes #9-#12 together (~0.18 public spellF1).
DFL (4-12 cases) and Metrica (~16, duel-like) have too few of the long kick -> short contact
pattern, and their truths don't count those contacts the organisers' way.
1. **PFF FC World Cup 2022** (64 matches): the user requests access. Check the spec PDFs:
   tracking fps, ball height, broadcast gaps, and whether the possession events mark brief
   contacts (a header straight into an opponent). Data goes to `data/external/pff/`.
2. **SkillCorner open data** (10 matches, 10 fps, `player_possession` events): download, count
   short possessions and the long kick -> short contact pattern (as `tools/kick_shorts.py`).
3. If either has enough cases: a loader like `tools/dfl_data.py`, the DFL treatment (gap
   pattern, blur), then the touch / short-possession networks retrained with it, and v31's chain
   checked leave one match out plus the public window.

**Queue (2026-10-26), judged on realistic DFL, leave one match out, ≥ 5/7 folds:**
1. ATD channel check: done, negative.
2. Pace shift: done, rejected ((a) quantile normalisation, (b) time warp, (c) domain weighting:
   none makes learned stoppage detection transfer).
3. In-domain team model: done, accepted as v25 (players-only vote, DFL + decoded pseudo-labels).
4. MBR decoding: done, rejected (nested −0.003 on DFL).
5. Hand fixes: done. Touch-margin vote + restart handover pass (v27_4, +0.009, 5/7); ball-out
   restart team and aerial landing rejected.
6. Future team reaction: done. As a hand vote no gain; in the second-stage bidirectional network
   +0.007 on DFL (6/7), v28. Original plan: **future team reaction (user's idea):** teams reorganise 1–8 s after a turnover, so an offline
   labeller can use the future. Per-team spread, centroid, push toward goal, back-line height,
   compactness, and future-minus-past differences (k = 2, 4, 6 s). Step 1: hand `shape_shift`
   (H/A split only). Step 2: inputs to the team network / GBM / GRU, context ±10 s. Also report
   turnover timing and the false-switch rate inside long possessions. Step 3: a second-stage
   bidirectional network (we label offline) that reads the first stage's held-out team
   probabilities plus the reaction features over about ±30 s and re-decides the team.
   Features started in `pipeline/reaction.py` (not wired in yet).
   Per-player inputs to the second stage (set encoder): rejected (+0.0063 vs +0.0085).

**A. Out-of-match validation: done for Metrica (2026-10-24).** Remaining: DFL open data as a
second provider, and the `--gt` option on spell_table / boundary_diag for external matches.
1. Load the Metrica sample data (3 games, tracking + events) with kloppy. Build H/A/D labels from
   the events: possession from passes/receptions, D from out-of-play and restarts until the
   restart event.
2. Degrade the tracking so it looks like ours: drop the ball in gaps with our length distribution
   (more when held or still), add jitter, lose players at the edges.
3. Run v1–v10 on it. Do the version history's choices hold up outside our match? Re-tune switch,
   touch thresholds and the keeper hold there, not on the public window.
4. Then add the Sportec/DFL open data (7 matches): more games, a different provider.

**B. Network models (they need A for labels and checking).** The boundary diagnosis makes B1 the
priority: 11 of 20 public changes are off by more than 1 s, and most of those come down to touch
quality.
1. **Touch/event detector (first network).** A small temporal CNN or transformer over ±1 s of
   tracking (ball path plus nearby players), trained on external games where the events give
   passes, receptions, headers, tackles and keeper catches with timestamps. Output: P(touch by
   home / away) per frame, replacing our hand-made velocity-peak rule. This is aimed at our
   biggest remaining misses (headers #2, deflections, interceptions), which hand rules can't see.
2. **Learned emissions.** Gradient boosting first, then a ±2 s transformer: from our features
   (plus the touch detector) to P(H/A/D) per frame, decoded by the same Viterbi. Trained on external
   games, checked on a held-out external game, and run on ours. The public window stays a sanity
   check only.
3. **Ball in gaps (self-supervised).** Predict the ball's position or owning team from the 22
   players, trained on our own ~94k ball-seen frames (no labels needed) and applied in gaps. Only
   if live-play gaps still matter after v8/v10.
4. **YOLO ball detector:** last. The gap clips showed missing balls are mostly dead or held.
   GPU: RTX 5070 8 GB, PyTorch with CUDA 12.8 or newer.
- Risks:
  - domain shift between providers and leagues (handled by degrading the tracking and by
    validating on held-out games);
  - external labels define possession slightly differently from the organisers'.

**C. Hand-built fixes, meanwhile (cheap; check each on the external set once it exists):**
1. Aerial balls: when the ball lands or reappears after a flight, the team that receives it gets a
   touch (#2).
2. Short possessions: a lower switch cost right after a strong touch (#10, #12, #15).
3. Touch confidence: weight by the size of the velocity change and the toucher's distance (#6).
4. Restart handover: a pass among slow players during a stoppage is not a possession.
5. Restart team: the set piece's tagged team, or for N/A throw-ins, the side the ball went out on.

**D. Housekeeping:**
- Commit `pipeline/`, `tools/` and `JOURNAL.md` on a branch.
- Start the technical document: approach, assets used and dropped, possession rules, version
  history.

## Open questions

- What does the ground truth call a contested or loose ball, or a header duel? #2 suggests
  possession changes at the first controlled touch.
- Is a keeper's slow build-up after a catch possession or D in the truth? The public window has
  no example. v10 assumes possession.
- Team: 3 to 5 members are required. Who is on it?
