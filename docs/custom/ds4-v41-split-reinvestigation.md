# Re-investigation: would splitting DeepSeek V4.1 Flash across the Spark and the Mac speed up token generation?

**Date:** 2026-09-22
**Question:** re-examine the earlier conclusion that splitting `DeepSeek-V4.1-Flash-Q2.gguf`
across the DGX Spark (CUDA) and the Mac Studio M4 Max (Metal) is "not worth it,
no visible improvement", focusing on **token-generation (decode) speed**.
**Method:** no hardware runs (both machines are in use). This is a re-derivation
from the repository's code and its **measured** pipeline data, including the
GLM 5.3 Q4_K split that was built and measured on *this same pair* after the
original V4.1 analysis was written.
**Prior work:** `ds4-technical-analysis.md` §17-§18, `ds4-v41-split-design.md`
(especially §11.15).

---

## 1. Verdict

**For token generation, the earlier conclusion is correct: a layer split does not
speed up decode. It can only recover decode lost to SSD streaming.** For this
model that means *no measureable gain at short/mid context, and a real but modest
~1.4-1.6x gain only at the deepest contexts (250K-500K)*.

Two measured facts, taken on this exact Mac+Spark pair (GLM 5.3, `ds4-glm53-q4-split-design.md` §0),
settle it:

| Topology (same model, same 403K prompt, ctx 524288) | prefill | decode |
| --- | ---: | ---: |
| Mac + Spark pipeline | 346.41 t/s | 9.26 t/s |
| **Mac alone, whole model resident** | 186.89 t/s | **19.71 t/s** |

When the model fits on one machine, the split is **2.1x slower at decode**. Decode
is the *sum* of the two stages; a split cannot beat the faster single machine, it
can only replace streaming with residency.

V4.1 Q2 does not fit resident on one machine (151.8 GiB of main weights vs ~110 GiB
usable), so the honest baseline is single-machine **streaming**, and there the
measured GLM shape applies: split decode is a **wash to 262K** and **~1.5x at
~479K**, because the single machine's decode is expert-cache bound and degrades
with depth while the pair's does not.

**Prefill is the opposite** (it is a `max` of stages, not a `sum`): the same pair
measured **4.0-4.6x** the single machine. But that is ingest speed, not token
generation.

So: the user's recollection is right for the metric they asked about. The earlier
write-up, however, got there with partly wrong reasoning and understated two
things (see §6).

The end goal - a responsive large-context coding agent - is analysed in §9. The
conclusion for response speed is the same, but the priorities are different: most
of the felt improvement is configuration (never re-prefill, fewer reasoning
tokens), not a split.

---

## 2. What the earlier investigation concluded, and on what evidence

From `ds4-v41-split-design.md` §11.15 (the "why this port is not recommended"
section):

1. **Decode is already a wash.** Mac alone streaming decodes 14.0-15.0 t/s at
   8K-32K and 10.5 t/s at 262K; the split was *predicted* at ~13.7 t/s. So no gain.
2. **Prefill is not the split's advantage at realistic scale.** The Mac alone
   reached 317-404 t/s on 131K-token rows, "at or above the ~410 t/s a split was
   predicted to deliver".
3. **1M context fits on the Mac alone** (103.90 GiB planned), so the capacity
   argument disappears.
4. **An M5-class Mac dominates everything** (measured Flash baseline 790/40 vs
   343.76/26.76 t/s).

The mechanism was stated correctly: decode `T = sum(stage)`, prefill `= max(stage)`
(the repository says the same in `docs/DISTRIBUTED.md`: *"Use pipeline mode
primarily for capacity and long-prefill throughput, not as a guaranteed decode
speedup"*). A V4.1 pipeline port was scoped at ~700-1,100 lines.

---

## 3. What is new since then: a measured Mac+Spark pipeline

The GLM 5.3 Q4_K split (`docs/custom/ds4-glm53-q4-implementation-log.md`,
`ds4-glm53-q4-split-design.md`) is a working two-machine pipeline on this exact
pair - Mac coordinator `0:20`, Spark worker `21:output` - and it was measured
cold at the owner's working depth. It is the empirical answer to the V4.1
question's *shape*, because the structural asymmetry is identical.

**Decode is insensitive to the split** (log §3.3, §4.3): moving one layer to the
Mac costs 15.8 t/s of prefill but buys only **1.30 ms (~1.3%)** of decode, because
prefill is `max` and decode is `sum`. At ~121 ms/token neither machine is
GPU-bound (Mac ~71%, Spark ~62%): a batch-1 step is weight-read bound, which is
why adding a second serial stage does not help.

**Decode vs the single machine, by depth** (log §4.4):

| depth | pair | Mac alone (streaming) | ratio |
| ---: | ---: | ---: | ---: |
| ~40K | 10.03 t/s | 8.84 t/s | 1.13x |
| ~287K | 8.30 t/s | 8.00 t/s | 1.04x |
| **~479K** | **7.395 t/s** | **4.63-5.02 t/s** | **~1.5-1.6x** |

The pair's advantage is *zero at 32K and grows with depth*, purely because
single-machine decode is expert-cache bound and falls ~40% from 262K to 512K while
the resident pair falls ~11%.

**Prefill is the split's real win**: 4.0-4.6x the single Mac at every depth
(415.0 vs 82.47 t/s at ~287K), because the pair holds all weights resident while
the single machine streams. That is the `max`-of-stages effect.

---

## 4. Why token generation cannot improve (the physics, with numbers)

1. **A decode step is serial across stages.** One token must traverse every layer
   before the next is sampled, so latency is `Mac stage + Spark stage + wire`. The
   best a split can do is put layers on the faster decoder; it never adds
   throughput. Measured proof: a resident Mac alone is 19.71 t/s vs the pair's
   9.26 t/s on the same model and prompt.
2. **The only decode benefit is residency.** If a single machine must stream
   experts from SSD, its decode is cache-bound and degrades with depth; a resident
   pair does not. That is worth ~1.5x *at 500K*, and ~nothing at 32K.
3. **The Spark stage is the slower decoder**, so leaning the split toward prefill
   (Spark-heavy, as GLM does) costs decode nothing only because decode is already
   `sum`-dominated and context-diluted at depth.
4. Therefore **for "token generation speed" specifically, no**: the metric is
   exactly the one a pipeline cannot move, except by removing streaming, which only
   matters at depth.

This matches the earlier conclusion. What the GLM measurements add is that
"no visible improvement" is *too strong at 500K*: there it is a measurable ~1.5x.

---

## 5. Applying it to V4.1 Q2

**Shape:** 340.60 GiB file; **151.8 GiB main weights** (plus ~188.8 GiB disk-only
Engram tables); 40 layers; `n_embd` 5120, `n_hc` 4 → the wire record is
`4 x 5120 = 20,480` f32 = **80 KiB/token**. Neither machine can hold the main
weights resident, so - like GLM Q4_K - the meaningful baseline is single-machine
**streaming**, and the split is the only residency option.

**Legal cuts are fixed by shared group KV** (`ds41_kv_source` / `owner()`,
`ds4.c:1384`, `:40623`): groups are layers 0-7, 8-13, 14-19, 20-39, so a slice
boundary is legal only after **7, 13 or 19**. Engram layers 1 and 14 must be owned
by whichever rank holds them.

**Predicted decode**, by analogy with the measured GLM shape (V4.1 Q2's single-Mac
decode is 14.2 t/s at 32K and 10.5 t/s at 262K, from §11.15):

| depth | Mac alone (streaming, measured) | split (predicted) | gain |
| ---: | ---: | ---: | ---: |
| 32K | 14.2 t/s | ~14 t/s | none |
| 262K | 10.5 t/s | ~10-11 t/s | none (wash) |
| ~500K | not measured | ~1.4-1.5x the Mac | modest |

**Cut orientation - a correction.** The earlier document fixed the Mac *upstream*
for cut 13 (`Mac 0-13 / Spark 14-39`). Because decode is a sum weighted by each
machine's per-layer time (Mac ~1.375 ms/layer, Spark ~2.283 ms/layer), the
decode-optimal *feasible* orientation is the **opposite**: `Spark 0-13 / Mac
14-39` (26 Mac layers, ~98 GiB - fits), predicted ~14.8 t/s at 64K versus ~13.7
at the balanced 20/20 and ~12.7 for the earlier Mac-upstream cut 13. Even the best
orientation is a wash against the Mac-alone streaming at ≤262K.

**Prefill - the earlier "parity" claim is the weak link.** §11.15 rested on
Mac-alone V4.1 Q2 prefill of **317-404 t/s** at 131K-262K. On the same machine,
GLM 5.3 Q4_K alone measures **82-85 t/s**, a 4.5x gap for a comparably-sized
model; and 374-404 t/s would put an M4 Max *streaming Q2* above the repository's
own *resident Q4* M3 Ultra reference (341.75 t/s). Those V4.1 Mac figures have no
raw artifacts in the tree and are internally suspect. If they are wrong, the V4.1
split's prefill win is large (GLM-like, ~4x), not parity - but that still does not
change the **decode** answer.

---

## 6. Corrections to the earlier write-up

1. **It compared across depths.** The split's 13.7 t/s prediction was calibrated
   at ~64K; the Mac-alone 10.5 t/s is at 262K. At equal depth the split is a wash,
   and *below* the Mac alone at 32K. The conclusion survives; the arithmetic that
   produced it did not.
2. **It missed the decode-optimal cut orientation** (Mac downstream at cut 13;
   §5 above).
3. **Its load-bearing prefill number is unverified** (317-404 t/s), inconsistent
   with both the M3 Ultra resident Q4 figure and the GLM single-Mac figure, and not
   archived. This is the weakest claim in the withdrawal.
4. **"No visible improvement" is right for short/mid context but wrong at 500K**,
   where the measured GLM analogue is ~1.5x. For a 250K-500K coding workload, that
   is the one place the split's decode case is real.

---

## 7. Implementation status and cost (code-verified in the current tree)

Not implemented, and the blockers are unchanged:

- **Engine gate is still closed** (`ds4.c:~70628`): V4.1 requires
  `distributed.role == NONE && !load_slice`.
- **No slice-capable V4.1 graph**: `ds41_graph_alloc` (`ds4.c:40311`) unconditionally
  opens the Engram tables for `blk.1` and `blk.14` (`ds4.c:40349`), so even a
  downstream slice cannot allocate.
- **The V4.1-specific hazards** the earlier design identified remain: shared
  compressed/index caches partitioned by group (ships state, not caches), Engram
  ownership on both open *and* per-token read, and the 4-float `g->pre` mixer
  carry (`ds4.c:~41135`) that must join the wire or the first downstream layer
  silently uses a stale mixer.
- `ds4_engine_hidden_f32_values` (`ds4.c:71820`) still returns
  `N_HC * N_EMBD` for every non-GLM family, so the carry is not yet on the wire.

Scope: **~700-1,100 lines plus a split-vs-whole oracle**, and "as much measurement
as code". This is the main reason it is not worth doing *for decode alone*.

---

## 8. Cheaper options that already exist

1. **Zero-code session handoff** (`ds4-v41-split-design.md` §18.2): prefill on the
   Spark, save the V4.1 KV checkpoint, `scp` it, resume decode on the Mac. Gets the
   Spark's prefill without a port; does not fix the Mac's streaming decode.
2. **A second like machine.** Two Sparks in TP measured **21.9 t/s decode** and
   ~400 t/s prefill (vs one Spark 9.3 t/s). For this model, TP across identical
   backends is the architecture that actually raises decode.
3. **An M5-class Mac**, which dominates every configuration in the earlier report
   (matched Flash baseline 790/40 vs 343.76/26.76 t/s).
4. **Stay single-machine**: raise `--kv-disk-space-mb` so the long ingest is
   amortised, and keep one long session alive.

---

## 9. The real goal: overall code-agent experience (large context + faster response)

The question behind this document is not "decode t/s" in isolation - it is a
large-context coding agent that feels responsive. That reframing changes which
measured numbers are load-bearing, because a turn's latency is not one number.

**A per-turn budget**, assembled from the repo's measurements (V4.1 Q2 single-Mac
figures from `ds4-v41-split-design.md` §11.15; pipeline figures are the GLM 5.3
analogues from `ds4-glm53-q4-implementation-log.md`, the only measured Mac+Spark
pipeline on this pair):

| Step | V4.1 Q2, single M4 Max (streaming) | Mac+Spark pipeline |
| --- | ---: | ---: |
| Cold ingest, 262K | **~11.7 min** | ~3 min (415 t/s) |
| Continued prefill, 2K tool result | ~11 s | ~5 s |
| **Decode of the assistant turn (incl. reasoning)** | **10.5 t/s @262K -> ~95 s / 1000 tk** | ~11 t/s -> ~91 s |
| Decode at 500K | ~7-8 t/s (degrades) | ~11 t/s (**~1.5x**) |
| Re-prefill after compaction / think-level change | another full ingest | same |

Two consequences:

- **"Large context" is a residency and re-prefill problem, not a split problem.**
  1M fits on the Mac alone (103.90 GiB plan), and the split's unique value is
  *speed* at depth and *ingest*, not capacity.
- **"Faster response" is a decode problem.** Per turn, decode dominates the
  continued prefill by roughly an order of magnitude - and decode is exactly the
  quantity a layer split barely moves (§4).

### 9.1 M4 + M5 in this frame

- **TP: not viable.** The two generations run different kernels (`g_metal4_tensor_api_enabled`
  is M5/M6/A19/A20 only; M4 falls back), and TP is a lockstep numerical contract.
  The check is not at startup either: `ds4_tp_hello_fixed` carries no device field
  (`ds4_tp.c:1964`), so the pair would bind and then diverge.
- **Pipeline: viable**, and it helps the agent profile - both stages resident, so
  the M5's TensorOps path only makes its own stage faster; prefill ~4-5x and depth
  decode ~1.5x.
- But decode is `sum(stage)`, so a pipeline does **not** beat an M5 *standalone* at
  short/mid context. If the intent of adding an M5 is response speed, the better
  uses are M5 standalone or two M5s in TP.

| Option | Large context | Response | Cost |
| --- | --- | --- | --- |
| M5 standalone | good | best of the mixed pair | none |
| M4+M5 pipeline | good, resident | prefill great; decode wins only at 500K | the V4.1 port (works today for GLM/Flash) |
| M4+M5 TP | n/a | n/a | not viable |

### 9.2 The largest wins are configuration, not hardware

For a large-context agent at ~10 t/s these dominate any split:

1. **Never re-prefill.** `--kv-disk-space-mb 65536` - the 4096 MiB default holds
   less than one 250K checkpoint (~1.6 GiB on disk) while 64 GB holds ~14, so a
   session survives across days instead of paying the ingest again (§11.15, §17.8).
2. **Keep the thought level stable.** V4.1 puts the effort level right after BOS,
   so any change is a *head* change -> full re-prefill.
3. **Spend fewer reasoning tokens.** At ~10 t/s a 2000-token block is >3 min;
   `/think 25` is the largest felt-latency win and costs nothing.
4. **Right-size context** (`--ctx 524288` day-to-day) so the expert cache stays
   larger than it would at 1M.

### 9.3 A response win that already exists: residency

On this same M4 Max a **resident** model decodes ~2.5-3.5x faster than streaming
V4.1 Q2 (measured 2026-09-19, `ds4-technical-analysis.md` §17.7):

| Model on the M4 Max | Residency | Decode |
| --- | --- | ---: |
| DeepSeek V4.1 Flash Q2 | streams | 10.5 t/s @262K |
| DeepSeek V4 Flash (IQ2/Q4K hybrid) | **resident** (106.53 GiB plan) | 27.9 t/s; **34.8 t/s with DSpark** |

If the task tolerates V4 Flash quality, that is a larger response improvement than
any split, with no port. (V4.1 has no drafter shipped, so there is no speculation
path for it.)

### 9.4 What this changes about the recommendation

- If V4.1 quality is required: run it on the Mac alone with §9.2's settings; treat
  the Spark pipeline as a *prefill/depth* upgrade only, after the port.
- The go/no-go for that port is one unmeasured number - the **V4.1 Q2 single-Mac
  cold 262K ingest** (§5). If it is really ~12 min, the pipeline buys little and is
  not worth 700-1,100 lines; if it is ~50 min, the pipeline buys the same 4-5x as
  GLM.
- If hardware is being added anyway: **two like machines (TP)** beat any M4+M5
  combination for response speed.

---

## 10. What would need the hardware, and the ask

Confirming a *V4.1-specific* number, rather than the GLM analogy, needs:

- The `DeepSeek-V4.1-Flash-Q2.gguf` model on whichever machine runs it (it is not
  on the Mac or the Spark today - only GLM is), and
- For a real split number, the ~700-1,100-line port first (it does not exist).

Without the port, the cheap checks are:

1. **Re-measure single-Mac V4.1 Q2 cold prefill/decode** at 262K and 500K (this
   alone would settle §5's suspect 317-404 t/s), and
2. If wanted, **re-run the GLM topology A/B** (pair vs resident Mac vs streaming
   Mac) at 500K once the pair is free.

Both need a machine and the model. **I am not touching the Spark or the Mac - both
are busy (the Spark is running the GLM worker, the Mac the GLM coordinator). Tell
me when they are free and which of the two checks you want.**

---

## 11. Recommendation

- **Do not port the V4.1 pipeline for token-generation speed.** Decode is the one
  quantity a layer split cannot improve; it is a wash to 262K and only ~1.4-1.5x
  at 500K, at the cost of ~700-1,100 lines and a permanently coupled pair.
- **For the end goal (§9) - a responsive large-context coding agent - the
  priorities are different, and mostly cheaper.** Absorb the re-prefill cost
  (`--kv-disk-space-mb 65536`), keep the thought level stable, spend fewer
  reasoning tokens, and right-size context. If the task tolerates V4 Flash, a
  resident V4 Flash + DSpark on the Mac alone (27.9-34.8 t/s) beats every streaming
  V4.1 configuration.
- **If V4.1 quality is required and the Spark is to be used**, the split is a
  *prefill/depth* upgrade, not a response upgrade: expect ~4-5x ingest and ~1.5x
  decode at 500K, ~nothing below 262K.
- **If hardware is being added anyway**, prefer a like machine: two M5s in TP, or
  two Sparks (21.9 t/s measured). An M4+M5 pair cannot TP and, as a pipeline, only
  matches the Mac+Spark profile.
- **Revisit the port only if** a V4.1 slice port lands for another reason (capacity
  or prefill), or the measured V4.1 Mac-alone cold ingest shows the streaming
  penalty is far larger than the GLM analogy suggests.
