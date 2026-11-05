# Gözde's journal: More than a Hack 2027, Phase 1

My working log on branch `gozde` (started from `mirac`). The team log is `JOURNAL.md` (Miraç).
Same rules: aggregates only, no data rows, coordinates, frame numbers or per-frame labels, and no
match data (video, tracking, plots) to a cloud LLM.

---

### 2026-11-04: setup on a MacBook Air (M1, no GPU)

- Data unpacked into `data/tactical/`; `tools/atd_cache.py` reads 199,319 frames, 46 slots.
- **Python must be native arm64.** The Anaconda base env was x86_64 under Rosetta (torch 2.2.2):
  very slow, and torch + LightGBM (two OpenMP runtimes) hung with no output. Fix: a new env
  `CONDA_SUBDIR=osx-arm64 conda create -n tiki -c conda-forge python=3.12`, then
  `conda install -c conda-forge pytorch lightgbm numpy pandas matplotlib opencv`.
- **CPU-only loading:** the models were saved on CUDA; the four `torch.load` calls
  (`touch_net`, `run_learned`, `refine`, `ball_net`) now use `map_location="cpu"` when there is no
  GPU (unchanged on a GPU machine). `tools/in_domain_team.py --ours` loaded the 7 DFL matches even
  when `models/po_team_v25.txt` exists; DFL is now loaded only when retraining.
- Rebuild on CPU matches the journal exactly: v25 (decoded pseudo-labels 34.0%, players-only
  agreement 81.5%), v30 263 possessions, public 0.8121 / 0.9006; v31 238, D 48.0%,
  0.8121 / 0.9166. Times on M1: v25 1 min, v30 2.5 min, v31 4.5 min.

### 2026-11-05: keeper swaps by teleport (_kt)

**Observation (#9–#13, viewer):** Barcelona's keeper vanishes during #11 and returns after #13,
while a Barcelona-coloured "keeper" plays in Athletic's goal next to a frozen Athletic slot.
`tools/keeper_swap.py` (aggregates only): Barcelona's keeper slot stops being tracked, reappears
6.1 s later 83 m away (13.6 m/s) as Athletic's keeper for 13.6 s, then slides back to its own goal
without a gap or a jump (one 2937 s piece). Athletic's own deepest slot is still 74–78% of that
time. One swap explains both symptoms. (JOURNAL.md says the Athletic-goal swaps happen without a
gap: this one has a 6 s gap.)

**Rule** (`pipeline/track_fix.fix_teleports`, `MTAH_VARIANT=_kt`): physics, not position, finds
the swap. A slot reappearing >= 20 m away faster than 12 m/s (after a gap or a >= 5 m one-frame
jump) is a new person. From there, its stay within 30 m of the other team's goal is that goal's
keeper if it lasts >= 2 s and he is deeper than every moving player there on >= 50% of it: moved
to a new slot of the defending team. The defending team's frozen slot is removed during the stay;
during the swap gap (no real keeper tracked) it walks from where it froze to where he reappears.
The swapped slot's own keeper is bridged from where he vanished to where he settles back at his
goal; the slide back is removed. Synthetic tests: swap across a gap, slide back without a jump,
a teleporting attacker (not flipped), a still real keeper before the swap (kept), a stay running
to the end of the half.

Mistakes on the way, caught in the viewer: the first version flipped whole pieces (missed the
2937 s piece); widening the ghost 30 s back removed the real keeper standing still before the
swap; removing the ghost in the gap left no keeper at all. All three fixed as above.

**Full match (`tools/teleport_diag.py`):** 16 teleports, 2 keeper swaps flipped, 244 s
(Barcelona's keeper under an Athletic slot 230.4 s; Athletic's keeper under a Barcelona slot
13.6 s, 11.4 s of it in #9–#13), ball within 10 m of a flipped keeper 15.8 s, ghost removed
19.6 s. The earlier rules fired on 1454 s (`_ki`) and 63.7 s (`_ke`).

**v31_kt** (v31's chain on `_kt`): public **0.7742 / 0.9263** (v31 0.8121 / 0.9166), 233
possessions (238), D 48.0%. Only #9–#13 change: one A from #9 to #13 (#9 0.62 -> 0.47, #11 0.31 ->
0.38, #12 0.13 -> MISS, #13 0.82 -> 0.09), as with every correct keeper fix, because #10 / #12 are
not seen. #1–#8 and #14–#17 unchanged (unlike `_ki` / `_kx` / `_kd`). The ghost removal and the
bridges change no label.

**Full-match differences (`tools/diff_spots.py`):** 27.6 s in 11 stretches, 5 of >= 1 s (1 public,
4 at the end of the second half, all A -> H). Watched on video (`render_labels.py --compare`):
**5/5 v31_kt right** (3 Barcelona's keeper has the ball, 1 Barcelona attack, 1 the public #11
stretch).

**Viewer:** `possession_view.py --frames START SECONDS --compare B.csv` for any stretch outside
the public window; with `_kt` the keeper strips and the pitch show the corrected tracking.

**Not adopted yet:** team decision. Public −0.038 comes from #9–#13 alone; in the private full
match that is roughly 1–2 possessions out of ~240, against the late-match fixes.

**Next:** #10 / #12 (short possessions after a long keeper kick), the only way to win the public
loss back.