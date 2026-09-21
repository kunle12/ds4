# GLM 5.3 Flash Q4 across a Mac Studio + DGX Spark

**Status:** implemented, tuned and running in daily use. The Q4_K dispatch gap that
this plan was written to close is closed, the pair serves requests end to end, and
the layer split has been swept and chosen. What remains is verification breadth
(§6), not capability.
**Scope:** `GLM-5.3-Flash-Q4_K.gguf` as a two-machine pipeline — Mac coordinator,
DGX Spark worker — for 250K–500K-token coding sessions.
**Companion:** `ds4-glm53-q4-implementation-log.md` is the reference for the current
system: configuration, the measured split frontier, bottleneck analysis, verification
status and operating notes. This document keeps the design rationale, the acceptance
criteria, and the plan's own status.

---

## 0. Outcome

The plan's verdict was that the blocker was **one missing dispatch path — the Q4_K
instantiations of ten GLM MoE kernels** — and that everything else was already in the
tree. That reading was correct: the port landed, and then measurement changed the
answer to *which* dispatch should be the default.

| Requirement | State |
| --- | --- |
| GLM 5.3 layer slicing (pipeline) | works; slice-aware graph alloc, per-slice memory accounting |
| Wire protocol carrying the GLM 5.3 mHC block | **fixed** (3 hunks in `ds4.c`); validated byte-identical against single-host |
| Memory fit for Q4 resident | measured: 82.22 GiB coordinator / 104.73 GiB worker at the recommended split, ctx 524288 |
| Link | 10GbE direct, 0.46 GiB/s single-stream against ~26 MB/s needed; no tunnel required |
| Q4_K *arithmetic* on CUDA | existed twice over; what was missing was the GLM dispatch |
| GLM routed MoE on CUDA **for Q4_K** | **done** — ported, then superseded: a homogeneous Q4_K trio now routes to the pre-existing generic dispatch, which prefills 2.7× faster at identical output (§8) |
| Mac as coordinator (inbound TCP) | **resolved by the OS, not by us**: macOS 26.7 accepts these binaries on non-loopback addresses, so the direct link works and the tunnel is a fallback rather than the transport |
| Spark thermal envelope | protection installed and verified; at ~74 % duty the board peaks at 83.5 °C with no throttling |

**What the split delivers, measured** (cold, greedy, ctx 524288):

| | pair (`0:20`) | pair (`0:23`) | Mac alone (`--ssd-streaming`) |
| --- | ---: | ---: | ---: |
| prefill, 286,646 tk | **415.0 t/s** | 356.3 t/s | 82.47 t/s at 262K |
| ingest | **11.5 min** | 13.4 min | 53 min at 262K |
| decode | **121.1 ms/token** | 120.5 ms | 125 ms at 262K |

So the plan's payoff estimate was directionally right and numerically pessimistic:
it predicted the split would "roughly halve long-ingest wall time" against the
single-Mac streaming path, and it does better than that — **~4.6× prefill**. Its
decode estimate (~10–14 t/s, "near the harmonic mean") did not survive contact with
the serial structure: decode is `sum(stage)`, so it lands near a single machine's
minus the streaming penalty, and the pair's advantage only appears at depth.

**The Q4 quality advantage is still not an argument for the split**, exactly as
stated here: it is available single-machine with `--ssd-streaming` — **−34.4 % NLL
against Q2, better on 98 of 100 cases** — and that route does reach 512K on this Mac
(13.09 GiB decode, 102.00 GiB prefill transient of a 115.19 GiB budget, 5.84 GiB KV,
70.90 GiB expert cache). What it cannot hold is *speed at depth*: decode falls to
**4.63–5.02 t/s** at 512K from 8.00 at 262K, and **MTP is a ~21 % loss** there. The
split is a **speed-at-depth, ingest-time and predictability** decision — not a
quality or capacity one. That conclusion is unchanged.

**The topology effect, measured** (2026-09-19, Q2, the same 403,351-token prompt at
ctx 524288 on both sides, so no quantisation or depth confound):

| Topology | prefill | decode |
| --- | ---: | ---: |
| Mac coordinator + Spark worker | **346.41 t/s** | 9.26 t/s |
| Mac alone (whole model resident) | 186.89 t/s | **19.71 t/s** |
| Spark alone, 512K | *wedges the box* | — |

Two conclusions this forced, both contradicting the earlier `[INFERENCE]`:

* The split **doubles prefill and halves decode** (1.85× / 0.47×) for a
  *whole-model-resident* comparison. It is a throughput-versus-latency trade.
* For Q4 that comparison is not the relevant one, because the single-machine Q4 route
  cannot be resident — it must stream, and streaming is what costs it its decode. The
  pair's Q4 decode is comparable at 262K and ~1.5× better at 479K (§4 of the log).

**Why the split wins at depth, in one line:** single-machine decode is expert-cache
bound and loses ~40 % from 262K to 512K, while the pair loses ~11 % because all
177.77 GiB stay resident.

---

## 1. Goal and acceptance criteria

Primary goal: a 250K–500K-token coding session on the pair, at Q4_K precision, with
the model fully resident and the Spark inside its thermal envelope.

| # | Criterion | Status |
| --- | --- | --- |
| 1 | **Correctness.** Pipeline greedy continuation matches a single-Mac Q4_K run for ≥ 128 tokens; logits within the repo's cross-backend tolerance | **not run** — the one substantive gap |
| 2 | **Boundaries.** Clean greedy output across the pooled-DSA boundary (2 048 → 2 056) and the prefill-work boundary (4 096 → 4 100) | **not run** |
| 3 | **Throughput.** ≥ 150 t/s prefill and ≥ 10 t/s decode at 32K on the pair | **met**, and exceeded at depth — see below |
| 4 | **Capacity.** 262 144-token cold ingest in one session; 524 288 context allocates and runs | **met** — ~287K and ~479K cold ingests at ctx 524288 |
| 5 | **Thermal.** Board ≤ 88 °C for the whole ingest with the guard installed; zero `HW Thermal Slowdown` events; completion without intervention | **met for one session** (~1 h, 61 samples, peak 83.5 °C, no throttling) — not a soak |
| 6 | **State.** Snapshot save/load across the split round-trips | **half met** — save verified, load path exercised on a live pair; fresh-pair and roles-swapped restore remain |

**Criterion 3, as measured.** At ctx 32768 with a 28 657-token prompt the pair
prefills at **389.0 t/s** (2.6× the bar) and decodes at 10.2–10.35 t/s. The split was
then swept, and **Mac `0:20` / Spark `21:output` is the recommended configuration**:
at ctx 524288 on a 286,646-token prompt it measures **415.0 t/s prefill and 121.1 ms
per token**, against 356.3 t/s and 120.5 ms for `0:23` / `24:output` — **+16.5 %
prefill at depth with decode unchanged**. It needs
`DS4_GLM_MEMORY_GUARD_RESERVE_GB=14` on the Spark, because the guard's 18 GiB default
refuses the larger slice.

The rule behind the choice, which is the plan's most useful surviving generalisation:
**moving one layer to the Mac costs 15.8 t/s of prefill and buys only 1.30 ms of
decode**, because prefill is `max(stage)` while decode is `sum(stage)`. Set the split
for prefill. (Log §3.3, §4.3.)

The prefill bar was missed for as long as the GLM-specific Q4_K kernels held the
default — and the fix was routing, not tuning: `--dist-activation-bits 16`,
`--dist-prefill-chunk` and `--dist-prefill-window` change nothing measurable.

**Criterion 6 detail.** Save verified 2026-09-19 for the `0:23` / `24:output` split at
ctx 4096: a 649-token cold prompt wrote `bf072cd0…kv` (165.08 MiB, `save=18.2 ms`)
with the data connections observed on `127.0.0.1:55911`, so the worker's slice really
crossed the link. The load was exercised and reported `cached_tokens: 819`; the
completion text was **not** compared, and equivalence on a fresh pair is not shown.
Two defects were cleared first: the worker's advertised data port was ephemeral until
pinned with `--listen`, and a KDA-layer sizing guard made any slice containing a KDA
layer unsizeable.

Non-goals: tensor parallelism across Metal+CUDA (architecturally excluded), Q4 on a
single Spark, MTP under the split (`ds4_engine_has_mtp` requires
`distributed.role == NONE`).

---

## 2. What was established (evidence)

Measured on the target pair unless noted.

| Item | Result |
| --- | --- |
| `GLM-5.3-Flash-Q4_K.gguf` binding | accepted as `glm5-next`/GLM 5.3 Flash; 177.77 GiB; `q4_k` is 163.27 GiB in 129 tensors = 43 MoE layers × gate/up/down; 46 blocks of which 45 executable |
| Per-layer weights (GGUF tensor table) | embedding and head 1.182 GiB each; blk.0–2 dense at 0.408 GiB each; blk.3–44 MoE at ~4.08 GiB each; blk.45 MTP 4.019 GiB, unmapped |
| **Pair, Q4_K, `0:20`** | **415.0 t/s prefill / 121.1 ms per token** at 286,646 tk, ctx 524288; 455.2 t/s and 103.0 ms at 39,865 tk |
| **Pair, Q4_K, `0:23`** | 356.3 t/s / 120.5 ms at 286,646 tk; 398.8 t/s and 98.5 ms at 39,865 tk |
| Split sensitivity | −15.8 t/s prefill and −1.30 ms decode per layer moved to the Mac as **endpoint averages over `0:20`…`0:27`**; decode is near-linear, prefill is not (18.8 / 10.6 / 22.0 t/s per layer across the three segments) |
| Memory bandwidth | Mac 295.1 GB/s read (369.0 copy), Spark 100.7 (123.2) — the Mac streams memory 2.9× faster |
| Memory admission | `0:20`: Mac 82.22 GiB of 115.19, Spark 104.73 of 107.61 (reserve 14). `0:27`: Mac 111.64 of 115.19 |
| Budget derivation | `min(0.99 x base, base − reserve)`, base = `hw.memsize` (Apple) or the CUDA working set; 18 GiB GLM 5.3 reserve, runtime-tunable. Spark's OS-visible total is 121.61 GiB; the Mac's 115.19 comes from `iogpu.wired_limit_mb=120000` |
| GPU duty | prefill: Mac 99–100 %, Spark ~57 % (`0:23`) rising to ~74 % (`0:20`). Decode: ~71 % Mac, ~62 % Spark, never ≥90 % |
| CUDA GLM routed MoE | was `ds4_cuda.cu:32103` Q2_K-only; ported, then the generic dispatch was promoted as default (§8) |
| CUDA coordinator-side slice prefill | `CUDA tensor read failed: unspecified launch failure` for chunks ≥ 512 rows; **reproduced on pristine `8db1d1d`**; still open (WS 10) |
| GLM 5.3 slice payload | `N_HC × N_EMBD` = 16 384 f32/token = 64 KiB; the wire size function said `N_EMBD` — **fixed** |
| Wire traffic | Mac→Spark 174.85 MB/s mean during prefill (one ~256 MiB burst per 4096-token chunk), Spark→Mac 0.20 MB/s — 871× one-way |
| Q2 pipeline | 32 768 ctx: 380.6 t/s prefill, 12.5 t/s decode with caps on (383.5 / 12.8 uncapped) |
| Q4_K on the Mac alone (SSD streaming) | 32 768: 84.2 / 8.8 · 262 144: 82.5 / 8.0 (53 min ingest), 99.84 GiB plan, 5 435/12 384 experts cached |
| Spark thermals | idle 43–50 °C; ~74 % duty peaks at 83.5 °C board with no throttling; uncapped reaches 90 °C with `HW Thermal Slowdown` and 69 s `SW Power Capping` |
| Distributed snapshot | save verified 2026-09-19 (165.08 MiB, 18.2 ms, data socket on `127.0.0.1:55911`); load path exercised, `cached_tokens: 819`; equivalence on a fresh pair not shown |
| Quality, Q2 vs Q4_K (100-case GLM 5.3 Flash fixture, Metal) | Q4_K `0.300477636 / 90 / 9.480` vs Q2 `0.458177271 / 90 / 7.390`; paired **98/100 cases better**, NLL **−34.4 %**, first-token match equal |
| Q4_K single-machine at 512K | viable (13.09 GiB decode / 102.00 GiB prefill transient of 115.19), but decode falls to **4.63–5.02 t/s** and MTP is a 22 % loss — no lever remains on that route |

---

## 3. The gaps

### 3.1 CUDA GLM routed MoE was Q2_K-only — **closed**

The original gate:

```c
/* ds4_cuda.cu:32103, before */
if (gate_type != 10u || up_type != 10u || down_type != 10u) {
    fprintf(stderr, "ds4: glm routed moe: unsupported types %u/%u/%u\n", …);
    return 0;
}
```

It is gone. The structure around it was type-independent as predicted, so the work
was the weight-side block read and dot for Q4_K (144-byte super-block) in place of
Q2_K (84-byte); activations were already Q8_K on every path.

| Kernel | Role | Was Q2_K-only? |
| --- | --- | --- |
| `glm_moe_expert_map_kernel` | build per-(token,expert) pair lists | no |
| `glm_moe_build_expert_tiles8_kernel` | build expert-tile8 lists | no |
| `glm_routed_moe_gateup_expert_tile8_kernel` | prefill gate/up, tile8 | **yes** |
| `glm_routed_moe_down_expert_tile8_terms_kernel` | prefill down, partial terms | **yes** |
| `glm_routed_moe_down_terms_reduce_kernel` | reduce down terms | no |
| `glm_routed_moe_gateup_expert_kernel` | expert-major gate/up | **yes** |
| `glm_routed_moe_down_expert_kernel` | expert-major down | **yes** |
| `glm_routed_moe_gateup_warp_kernel` | decode gate/up, warp per pair | **yes** |
| `glm_routed_moe_gateup_tok2_reuse_kernel` | decode, 2 tokens/expert reuse | **yes, still refuses** |
| `glm_routed_moe_down_warp_kernel` | decode down | **yes** |
| `glm_routed_moe_batch_q2K_{gateup,down}_kernel` | small-batch fallback | **yes** |

Two kernel-path defects surfaced and were fixed while porting, both worth keeping in
mind as *classes*: a hardcoded `256u` against a 288-expert model (wrong output that
looked plausible, never a crash), and a missing `swiglu_clamp` in the CUDA
definition, masked by AArch64 passing floats in the FP register file. A regression
test (`make test-glm53-moe-q4k`) now fails pre-fix on the first of those.

The tok2 path still refuses Q4_K by name: it is a small instantiation by the same
recipe, but verifying it needs a GLM MTP support model and there is none here.
Speculative decoding must not be silently wrong, so it refuses rather than runs
unverified.

### 3.2 CUDA coordinator-side slice prefill — **still open, still optional**

Only needed if the Spark must lead. Not needed for the goal, since the Mac is the
coordinator, which is also where the user sits. WS 10.

### 3.3 macOS inbound TCP — **superseded**

This was settled on 2026-09-19 as "the firewall route is closed, the tunnel is the
transport". That is no longer true and the plan should not carry it as current: on
macOS 26.7 the same binaries accept non-loopback connections and the pair runs on the
link addresses directly. The tunnel kit is retained as a documented fallback, because
the original block was real when measured — but it is not the operational path, and
the two-forward arrangement below is now only a fallback recipe.

---

## 4. Design

### 4.1 Type-generic GLM routed MoE — as built

Templating, not forking, and it held up: one body per kernel with `template
<typename block_t>` plus `glm_moe_dot` / `glm_moe_dot8` overloads, so the block
stride and dot come from `sizeof(block_t)`. Forking a parallel `*_q4K_*` family would
have doubled a tuned surface, and the case for that is now stronger than it was — the
generic dispatch turned out to be the faster path (§8), so the ported family is a
hatch-covered fallback rather than the default.

Items 1–7 of the original design, as resolved:

1. **Type traits** — reused the vendored `cuda/mmq` vecdots and block definitions.
2. **Dispatch** — the `type != 10` gate became a supported-combination predicate with
   the same shape as Metal's, defaulting to `{Q2_K, Q4_K}` and failing loudly
   otherwise.
3. **Instantiations** — prefill (tile8 + terms/reduce), expert-major, decode warp,
   small-batch fallback: done for Q4_K. tok2 still refuses (§3.1).
4. **Byte accounting** — came from the caller as predicted; the GLM layout validator
   accepts a Q4_K routed set for a slice.
5. **Streaming lookahead** — type-independent, confirmed.
6. **Numerics** — the CUDA path accumulates `mid` in f32 and re-quantizes to Q8_K;
   Metal uses an FP16 intermediate for routed prefill. Cross-backend logits are
   therefore *not* bit-identical by construction, and the acceptance test is a
   tolerance gate rather than equality. Measured agreement after the fixes: mean
   |Δ| 0.0727 against the Metal reference, top-16 16/16.
7. **Escape hatch** — `DS4_CUDA_GLM_MOE_TYPES` narrows which types the GLM-specific
   dispatch accepts (`q2k` restores its historical single-type set; default `q2k,q4k`).
   It is not a route selector: a homogeneous Q4_K trio is claimed by the generic
   dispatch before that entry is consulted.

### 4.2 Topology: Spark as worker, Mac as coordinator

Unchanged, and still the right choice — it keeps the interactive frontend (CLI,
agent, server) on the machine the user sits at, and sidesteps §3.2 entirely. Direction
is performance-neutral at a balanced cut.

**Access path — direct link, no tunnel.** The coordinator binds the link address and
the worker dials it:

| Socket | Binds | Reachable by |
| --- | --- | --- |
| coordinator HTTP (`--host/--port`) | per `~/bin/llm_config.json` | the owner's clients — independent of this link |
| coordinator distributed (`--listen`) | `192.168.2.1:9911` | the worker, over the direct link |
| worker data listener (`--listen`) | `192.168.2.2:55911` | the coordinator, over the direct link |

Use the **literal IPv4** of the direct link on both sides, never a hostname — an mDNS
resolution failure killed an earlier session-scoped run mid-ingest.

*Fallback*, if a future macOS release blocks non-loopback accepts for a locally built
binary again: the loopback tunnel, with `-R 9911:127.0.0.1:9911` for control and
`-L 55911:127.0.0.1:55911` for the data connection, the worker pinned with
`--listen 127.0.0.1 55911`, and the coordinator deriving the worker's address from
the accepted socket. Supervise it with `launchd` `KeepAlive`, since a drop aborts a
run mid-ingest.

One exposure remains, unchanged and independent of the transport: the HTTP API has
**no authentication**, so serving clients beyond a trusted network needs a proxy or
VPN.

### 4.3 Thermal envelope — measured, not assumed

The Spark hard-locked under sustained load once (board 90 °C, `HW Thermal Slowdown`
accumulating, box unreachable). The response was the protection kit — a clock cap at
2100 MHz, a CPU cap at 2.4 GHz, and a board-zone governor — installed as a
prerequisite rather than an option, and it has held:

| condition | board peak | throttling |
| --- | ---: | --- |
| uncapped | 90 °C | `HW Thermal Slowdown` + 69 s `SW Power Capping` |
| capped + guard, 32K→128K sweep | 81 °C | none over 269 samples |
| capped + guard, 1 h at ~57 % duty | 83.9 °C | none over 61 samples |
| capped + guard, ~74 % duty (`0:20`) | **83.5 °C** | none; SM clock 2093 MHz |

**Loading the Spark harder is essentially free on this unit**: duty nearly doubling
moved the peak by 0.4 °C. So the throughput gain at the recommended split is paid in
duty cycle, not thermal headroom — which also means the 83.9 °C figure was not a
duty-limited ceiling. Two practices from this plan stand and should stay: every
throughput number is recorded **with** the peak board temperature, and runs are
bounded with cool-downs.

**The precondition is still mandatory:**

```sh
pgrep -ax ds4 || echo clear      # expect "clear"
free -g | head -2                # expect the worker's slice to be free
sudo pkill -x ds4                # only if the first line was not empty
```

Two `ds4` processes on that box is not a supported configuration, not even
transiently: the guard's `TARGET_KILL=ds4` at `ZONE_ABORT=95 °C` is a backstop for the
hardware, not a licence to over-subscribe memory.

**Treat the admission numbers as necessary, not sufficient.** The guard prints
`required=… GiB budget=… GiB` at startup, and a *whole-model* 512K run plans ~98.88 GiB
against 121 GiB on this box: the Mac admits and completes it, the Spark does **not** —
the GPU driver refuses the allocations outright (`NVRM: … Out of memory
[NV_ERR_NO_MEMORY]`), logging stops inside 7 s, and the box resets ~19 min later with
the watchdog never armed. The guard's budget derives from system RAM and does not
account for NVRM's own reservation, so it cannot authorise a whole-model run there.
**The split slices are the supported shape** — the recommended `0:20` / `21:output`
plans 82.22 GiB on the Mac and 104.73 GiB on the Spark — and the pair ran a 403K
ingest at 67 °C with the machine responsive.

---

## 5. Work breakdown, with status

Estimates were for a developer fluent in this codebase and its QA habits; the work
came in roughly as scoped, and the surprises were in *which* path won rather than in
the effort.

| # | Workstream | Files | Est. | Status |
| --- | --- | --- | ---: | --- |
| 1 | Type-traits + dispatch predicate for GLM MoE (`{Q2_K, Q4_K}`), loud failure otherwise, env escape hatch | `ds4_cuda.cu` | 0.5–1 d | **done** |
| 2 | Q4_K instantiations: prefill tile8 gate/up, down terms + reduce | `ds4_cuda.cu` | 2–3 d | **done** |
| 3 | Q4_K instantiations: expert-major gate/up + down | `ds4_cuda.cu` | 1–2 d | **done** |
| 4 | Q4_K instantiations: decode warp-per-pair, tok2-reuse, down warp, small-batch | `ds4_cuda.cu` | 2–3 d | **partial** — warp, down warp and small-batch land; tok2 and scalar still refuse Q4_K by name |
| 5 | CPU/GPU parity harness for the GLM MoE Q4_K path | `tests/`, `Makefile` | 1–2 d | **partial** — `make test-glm53-moe-q4k` covers warp/small-batch, tile8, expert-major and tile8-off, and is validated to fail pre-fix; tok2, scalar, empty experts, tile tails and scratch reuse uncovered; the clamp is checked on real weights rather than synthetically |
| 6 | Cross-machine oracle: pipeline vs single-host Q4_K, logit tolerance + `--dist-replay-check` | `tests/`, `QA_BEFORE_RELEASES.md` | 1–2 d | **open** — criterion 1, the remaining gate that matters |
| 7 | Boundary gates (2 048→2 056, 4 096→4 100), snapshot round-trip across the split | `QA_BEFORE_RELEASES.md` | 1–2 d | **partial** — live-pair save and load verified; fresh-pair and roles-swapped restores and both sweeps not run |
| 8 | Long-context endurance: 262K cold ingest, 524K alloc, thermal logging per frontier | `ds4_bench.c`, `speed-bench/` | 1–2 d | **done for one session** — ~287K and ~479K ingests at ctx 524288, peak board logged, no throttling |
| 9 | Docs + release gates | docs, `QA_BEFORE_RELEASES.md` | 0.5–1 d | **done** — `DISTRIBUTED.md`, `DGX_SPARK.md`, `MODELS.md`, `SERVER.md` and `QA_BEFORE_RELEASES.md` §10/§16 all carry the current split and numbers |
| 10 | *(optional)* CUDA coordinator-side slice prefill fix | `ds4_cuda.cu` | 1–3 d | **open, optional** — not needed while the Spark is the worker |
| 11 | *(optional)* IQ2_XXS for the same GLM MoE path | `ds4_cuda.cu` | 1–2 d | **deferred** — one instantiation of the same template, once Q4_K is through QA |

The critical path (1 → 2 → 5 → 6) landed through 5; **6 is the remaining gate that
matters**. The order of work held as predicted — prefill first, then decode, then the
QA matrix — which is what let a long ingest be measured behind the guard early.

---

## 6. Validation plan, and what has passed

Layered, cheapest first; each layer must pass before the next is trusted.

1. **Kernel parity (unit).** *Partial.* `tests/test_glm53_moe_q4k.c` builds a
   synthetic 288-expert MoE through the public `ds4_gpu_*` API and compares against a
   host mirror of the q8_K quantizer, the Q4_K block dot and the MoE math —
   bit-exact, and deliberately shaped around the regression: it fills `mid` with NaN
   and asserts no selected pair is left unwritten, using a `selected` list containing
   expert 287, expert 256, expert 255, a negative slot and an unselected expert. It
   **fails** with the old 256-expert grid (`57 selected pairs had no mid row written`)
   and passes with the fix. Uncovered: tok2, scalar, empty experts, tile tails,
   scratch reuse.
2. **Graph parity (integration).** *Passed for the CUDA Q4_K path.* Full-vocabulary
   logits against the Metal reference on the same prompt: mean |Δ| 0.0727, max 0.197,
   top-16 16/16 after the clamp fix, and the corrected tile8 path is byte-identical to
   the warp path. The 100-case GLM 5.3 Q4 suite passes on the single machine.
3. **Boundary gates.** *Not run.* 2 048→2 056 and 4 096→4 100 rendered-token sweeps,
   non-finite-logit checks, a multi-token exact-output task.
4. **Cross-machine correctness.** *Not run.* Pipeline vs single-host greedy agreement
   for ≥ 128 tokens, then logits; `--dist-replay-check` after a forced route drop;
   worker restart mid-ingest and mid-decode. This is criterion 1 and the largest
   remaining gap.
5. **Capacity / long context.** *Capacity passed, state half.* 262 144 cold ingest and
   524 288 allocation are both done — bettered, in fact, with ~287K and ~479K ingests.
   Save and load on a live pair are verified; the **fresh-pair** restore and the
   **roles-swapped** restore remain (criterion 6).
6. **Thermal endurance.** *One session.* A 286K ingest with the guard armed: peak
   board 83.5 °C at ~74 % duty, zero thermal events, completion without intervention.
   The plan asked for the ingest repeated twice to show it is not a one-off; that
   repeat has not been run.
7. **Agent-level.** *Not run.* `tests/test_agent_compaction.py` at ≥ 64K and a real
   read/edit/test task through `ds4-agent` — the actual workload.

---

## 7. Risks, and how each resolved

| Risk | Impact | Outcome |
| --- | --- | --- |
| The Q4_K expert *layout* differs from the Q2_K assumption (byte offsets/rows) | silent wrong output | **materialised in a form the plan did not anticipate** — not the layout but a hardcoded 256-expert bound against a 288-expert model. Caught by comparing logits against a *different implementation*, not by kernel-level checks: the fault was in the routing data feeding the kernels |
| Templating the tuned kernels perturbs Q2_K codegen | Q2_K regression | **did not materialise** — a Q2 standalone regression on the pair produced byte-identical output against a `8655de7` baseline |
| Cross-backend numeric drift larger than tolerance (CUDA f32 mid vs Metal FP16) | fails acceptance | **measured, and smaller than feared** after the fixes: mean |Δ| 0.0727, top-16 16/16. The tolerance framing was right; bit-identity was never available |
| Spark thermal trip during endurance runs | lost work, hardware risk | **measured away for this workload** — ~74 % duty peaks at 83.5 °C with no throttling, and the guard has never intervened |
| Tunnel drops mid-ingest | aborted 250K run | **moot** — the tunnel is no longer the transport (§3.3); the direct link is. The fallback recipe is retained, with its own supervision advice |
| Time: the port expands (decode variants, IQ2_XXS, boundary bugs) | slip | **contained** — prefill-first was the right order, and the ported kernels turned out not to be the default anyway (§8) |

The risk that cost the most was not in this table: **reasoning from shared code to
shared inputs.** "The two paths are templated over the weight type, therefore the fault
must be type-dependent" silently assumed both paths compute the same work. They do not.
That is recorded in the implementation log's Appendix B.

---

## 8. Rollout and repository hygiene

* **No new user-facing flags.** Type support is a capability of the CUDA GLM path;
  `DS4_CUDA_GLM_MOE_TYPES` is diagnostic, documented with the other environment
  variables.
* **`DS4_GLM_GENERIC_MOE_Q4K` — removed 2026-09-20, and promoted rather than
  dropped.** The two dispatch paths were measured against each other on the pair
  before the gate came out, which is what "do not leave two dispatch paths
  unexamined" was asking for, and the answer was not the one the workstreams assumed:
  the generic dispatch prefills at **258.9 t/s** against the ported GLM-specific
  kernels' **95.3 t/s** — 2.7× — with **byte-identical** output over 64 greedy tokens.
  A homogeneous Q4_K trio now routes to the generic dispatch unconditionally, and the
  ported kernels are test-only coverage (`make test-glm53-moe-q4k`) rather than a
  runtime-selectable fallback — the predicate has no switch back. Mechanism: the
  generic path uses tensor-core tile16 Q4_K
  kernels, the ported ones do not. This is the difference between criterion 3 passing
  and failing.
* **Docs updated in the same change set** — and this is the part that had drifted.
  `docs/DISTRIBUTED.md`, `docs/DGX_SPARK.md`, `docs/MODELS.md`, `docs/SERVER.md` and
  `QA_BEFORE_RELEASES.md` §10/§16 all still described the superseded `0:23` split and
  dismissed the `0:20` rebalance as "short context only". All five now carry the
  recommended split, the reserve it needs, and the current numbers.
* The QA habit stands: **a physical two-machine run is the gate**; a parser or unit
  test is not.

---

## 9. Fallbacks (current numbers)

| Need | Use | Numbers |
| --- | --- | --- |
| **Best available on the pair** | **Q4_K, Mac `0:20` / Spark `21:output`, Spark reserve 14** | **415.0 t/s prefill, 121.1 ms/token at 286,646 tk**; 11.5 min ingest; ~74 % Spark duty, peak board 83.5 °C |
| Q4 quality, long context, one machine | Mac alone + `--ssd-streaming` | 82.5 t/s prefill, 8.0 t/s decode at 262K (53 min ingest, 99.84 GiB plan); **4.63–5.02 t/s at 512K** |
| Fastest decode on the pair | `0:27` / `28:output` | 93.9 ms/token at 39,865 tk — ~4.6 ms better than `0:23`, at the cost of 13.5 % prefill and only 3.55 GiB free on the Mac |
| Q2 throughput on the pair | Q2 pipeline | 380.6 t/s prefill, 12.5 t/s decode at 32K, board 66–77 °C |
| Q2, one machine | **Mac resident** — not the Spark at 512K | Mac, whole model, ctx 524288, 403K prompt: **186.89 t/s prefill, 19.71 t/s decode**. The Spark's repo-QA figure (531 / 14.35 t/s) is at a smaller context; a whole-model 512K run **wedges the box** |

---

## 10. Open questions for the owner

1. ~~Is Q4_K the target, or should the same work also cover **IQ2_XXS** in the same
   pass?~~ Answered 2026-09-19: **Q4_K only**, IQ2_XXS left as WS 11 — still deferred,
   now behind Q4_K's QA rather than behind the port.
2. ~~Grant `ds4` Local Network permission on the Mac, or standardise on the
   supervised tunnel?~~ Answered 2026-09-19 as "the tunnel is standard", and
   **superseded 2026-09-20**: macOS 26.7 accepts these binaries on non-loopback
   addresses, so the direct link is the transport and the tunnel is a documented
   fallback.
3. ~~Is the Spark's hard-lock RMA-worthy on this unit?~~ Answered 2026-09-19:
   **accept the cap + guard workaround**, revisit only if a caps-armed run trips. The
   caps-armed runs since include a full hour at ~74 % duty with no trip, so the answer
   now rests on evidence rather than on absence of testing.
4. ~~Does the 500K target need to hold with `--mtp` off?~~ Answered 2026-09-19:
   **yes, and MTP stays a non-goal.** The decision rests on MTP being excluded under a
   layer split, and the single-machine measurement — a **22 % loss at 512K** — removes
   the worry that this gives something up.
5. **New, answered 2026-09-20: which split?** **Mac `0:20` / Spark `21:output`**, with
   `DS4_GLM_MEMORY_GUARD_RESERVE_GB=14` on the Spark. Measured **+16.5 % prefill at
   depth** against `0:23` with decode unchanged, because the split is ~3.1× more
   sensitive for prefill (`max(stage)`) than for decode (`sum(stage)`). This is the
   same answer the owner's instinct pointed at — put the compute-bound layers on the
   Spark — and the measurement now supports it: the Spark is the faster machine per
   layer at prefill, the Mac at decode, and prefill is where the choice has leverage.
