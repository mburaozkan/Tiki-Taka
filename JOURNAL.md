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

## Current situation

- **DFL open data is the main test set (2026-10-24):** real tracking, provider possession flags
  filtered to controlled possession, our gap pattern cut in. v19 scores 0.567 / 0.833 there.

- **Metrica is useful for the team decision but not for the ball:** its ball is reconstructed
  (absent in stoppages, sitting on the carrier), so models trained on it misread our dead balls.
  Learned models beat the hand model on Metrica (0.822 vs 0.793), but transfer badly to our match
  (v16: public 0.49 / 0.67).
- **The submission is v21** (`python pipeline/run_learned.py --name v21 --source dfl --model blend
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

## Next steps

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
