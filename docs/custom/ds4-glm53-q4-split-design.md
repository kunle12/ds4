# GLM 5.3 Flash Q4 across a Mac Studio + DGX Spark — closing the gap

**Status:** design / work plan. The implementation is *not* done.
**Scope:** make `GLM-5.3-Flash-Q4_K.gguf` runnable as a two-machine pipeline
(Mac coordinator + Spark worker) for 250K–500K-token coding sessions.

---

## 0. Verdict

The blocker is **one missing quantized-matmul path in one function**, not the
architecture. Everything else the split needs is already in the tree and has
been exercised end to end:

| Requirement | State |
| --- | --- |
| GLM 5.3 layer slicing (pipeline) | works; slice-aware graph alloc, per-slice memory accounting |
| Wire protocol carrying the GLM 5.3 mHC block | **fixed during this work** (3 hunks in `ds4.c`); validated byte-identical against single-host |
| Memory fit for Q4 resident | measured: 88.60 GiB coordinator / 86.32 GiB worker, caches included, at 512K ctx |
| Link | 10GbE direct, measured 0.46 GiB/s single-stream vs ~26 MB/s needed |
| Q4_K *arithmetic* on CUDA | exists twice over: vendored MMQ (`ds4_mmq_dense_impl<GGML_TYPE_Q4_K>`) and DeepSeek MoE Q4_K kernels |
| GLM routed MoE on CUDA **for Q4_K** | **missing** — `ds4_cuda.cu:32103` accepts `type 10` (Q2_K) only |
| Mac as coordinator (inbound TCP) | blocked by macOS for adhoc-signed binaries; workaround in use (`ssh -R` tunnel) or grant Local Network permission |
| Spark thermal envelope | protection now installed and verified live |

Payoff `[INFERENCE]`: pipeline prefill is `max(stage)` not `sum(stage)`, so the
split should roughly halve long-ingest wall time versus the single-Mac streaming
path (measured today: 82.5 t/s ⇒ 53 min for a cold 250K). Decode is `sum(stage)`
and will land near the harmonic mean of the two machines, i.e. ~10–14 t/s versus
8.0 t/s streaming. The main win is **ingest time and the removal of the
SSD-streaming dependency** (no expert cache to size, no hotlist to tune).

---

## 1. Goal and acceptance criteria

Primary goal: a 250K–500K-token coding session on the pair, at Q4_K precision,
with the model fully resident and the Spark inside its thermal envelope.

Acceptance (all must hold on the target hardware):

1. **Correctness.** Pipeline greedy continuation matches a single-Mac Q4_K run on
   the same prompt for ≥ 128 tokens, and full-vocabulary logits agree within the
   repository's tolerance for cross-backend comparison (`tests/` oracle style;
   the TS/TP precedent is logit tolerance, not bit-identity).
2. **Boundaries.** Clean greedy output across the pooled-DSA boundary (2 048 →
   2 056 rendered tokens) and the prefill-work boundary (4 096 → 4 100), as
   `QA_BEFORE_RELEASES.md` §6 requires.
3. **Throughput.** ≥ 150 t/s prefill and ≥ 10 t/s decode at 32K on the pair
   (today: 380.6 / 12.5 t/s for **Q2** on the pair; 82.5 / 8.0 t/s for **Q4** on
   one Mac).
4. **Capacity.** 262 144-token cold ingest completes in one session; 524 288
   context allocates and runs.
5. **Thermal.** Board (`acpitz`) stays ≤ 88 °C for the whole ingest with the
   guard installed; zero `HW Thermal Slowdown` events attributable to the run;
   no hard-lock, and the run completes without operator intervention.
6. **State.** Snapshot save/load across the split round-trips (DSV4/DSVL path).

Non-goals for this work: tensor parallelism across Metal+CUDA (architecturally
excluded), Q4 on a single Spark, MTP under the split (`ds4_engine_has_mtp`
requires `distributed.role == NONE`).

---

## 2. What was established during this work (evidence)

Measured on the target pair unless noted.

| Item | Result |
| --- | --- |
| `GLM-5.3-Flash-Q4_K.gguf` binding | accepted as `glm5-next`/GLM 5.3 Flash; 177.77 GiB; **`q4_k` is 163.27 GiB in 129 tensors = 43 MoE layers × gate/up/down**; everything else BF16/Q8_0/F32 |
| CUDA GLM routed MoE | `ds4_cuda.cu:32103` → `glm routed moe: unsupported types 12/12/12`; decode delegates to the same batch function |
| CUDA coordinator-side slice prefill | `CUDA tensor read failed: unspecified launch failure` for chunks ≥ 512 rows; **reproduced on pristine `8db1d1d`**; 26-token slices fine |
| macOS inbound | handshake completes, socket unowned in `CLOSE_WAIT`, `accept()` never returns; a 20-line `cc` listener fails identically, Apple `nc` works; loopback exempt |
| GLM 5.3 slice payload | `N_HC × N_EMBD` = 16 384 f32/token; the wire/buffer size function said `N_EMBD` — **fixed** |
| Q2 pipeline (Mac coord + Spark worker) | 32 768 ctx: 383.5 t/s prefill, 12.8 t/s decode (uncapped) |
| Q2 pipeline with thermal caps on | 32 768 ctx: **380.6 t/s prefill, 12.5 t/s decode**; board 66–77 °C, `slowdown=Not Active` (vs 90 °C board and accumulating HW slowdown uncapped) |
| Q4_K on the Mac alone (SSD streaming) | 32 768: 84.2 t/s / 8.8 t/s · 262 144: 82.5 t/s / 8.0 t/s (53 min ingest), 99.84 GiB plan, 5 435/12 384 experts cached, no thermal warning |
| Spark thermals | idle 43–50 °C; under pipeline prefill board 60–90 °C, GPU die ~10 °C cooler; `HW Thermal Slowdown` + 69 s `SW Power Capping` observed uncapped |

---

## 3. The gaps

### 3.1 CUDA GLM routed MoE is Q2_K-only (the critical path)

```c
/* ds4_cuda.cu:32103 */
if (gate_type != 10u || up_type != 10u || down_type != 10u) {
    fprintf(stderr, "ds4: glm routed moe: unsupported types %u/%u/%u\n", …);
    return 0;
}
```

The implementation around it (`ds4_gpu_glm_routed_moe_batch_tensor`,
`ds4_cuda.cu:32068`) is a complete, tuned pipeline whose **structure is
type-independent** — only the inner K-quant dot products are Q2_K-specific:

| Kernel | Role | Q2_K here? |
| --- | --- | --- |
| `glm_moe_expert_map_kernel` | build per-(token,expert) pair lists | no |
| `glm_moe_build_expert_tiles8_kernel` | build expert-tile8 lists | no |
| `glm_routed_moe_gateup_expert_tile8_kernel` | prefill gate/up, tile8 | **yes** |
| `glm_routed_moe_down_expert_tile8_terms_kernel` | prefill down, partial terms | **yes** |
| `glm_routed_moe_down_terms_reduce_kernel` | reduce down terms | no |
| `glm_routed_moe_gateup_expert_kernel` | expert-major gate/up | **yes** |
| `glm_routed_moe_down_expert_kernel` | expert-major down | **yes** |
| `glm_routed_moe_gateup_warp_kernel` | decode gate/up, warp per pair | **yes** |
| `glm_routed_moe_gateup_tok2_reuse_kernel` | decode, 2 tokens/expert reuse | **yes** |
| `glm_routed_moe_down_warp_kernel` | decode down | **yes** |
| `glm_routed_moe_batch_q2K_{gateup,down}_kernel` | small-batch fallback | **yes** |

Activations are already quantized to Q8_K by `q8_K_quantize_kernel` for *all*
paths, so the activation side needs no change: the work is the **weight-side**
block read + dot for Q4_K (144-byte super-block, 4.5 bpw) in place of Q2_K
(84-byte, 2.5625 bpw).

### 3.2 CUDA coordinator-side slice prefill (only if the Spark must lead)

Not needed for the goal as long as the Mac is the coordinator, which is also
where the user sits. Fix it only if the Spark-headed topology is wanted.

### 3.3 macOS inbound TCP (operational, not code)

`ds4` is adhoc/linker-signed, so non-loopback inbound is blackholed. Two
options, both already demonstrated: grant the binary Local Network permission
(preferred, durable) or keep a supervised `ssh -R` tunnel. This must be settled
before any long session, because a tunnel drop aborts a run mid-ingest.

---

## 4. Design

### 4.1 Make the GLM routed MoE type-generic (core change)

Templating, not forking. The ten kernels above share one body and differ only in
the weight block type; a `template <ggml_type TYPE>` (or a small traits struct
supplying block struct, block size and `vec_dot_*_q8_K`) keeps one body per
kernel and adds Q4_K (and later IQ2_XXS) as instantiations. Forking a parallel
`*_q4K_*` kernel family would double a tuned surface and guarantee drift.

Concretely:

1. **Type traits.** Reuse the vendored `cuda/mmq/vecdotq.cuh` (`vec_dot_q4_K_q8_1`
   and friends) and `ggml-common.h` block definitions rather than writing new
   arithmetic; the repo already vendors this tier and links it.
2. **Dispatch.** Replace the `type != 10` gate with a supported-combination
   predicate mirroring Metal (`ds4_gpu_glm_gate_pair_type_supported` /
   `ds4_gpu_glm_down_type_supported`, `ds4_metal.m:38792` area) so the two
   backends advertise the same set: `{Q2_K, Q4_K}` gate/up × down, with the
   existing loud failure for anything else.
3. **Instantiations.** Prefill (tile8, terms+reduce), expert-major, decode
   warp-per-pair, tok2-reuse, small-batch fallback — each for Q2_K and Q4_K.
   Instantiate only what the dispatch can select, to keep compile time and
   register pressure in check.
4. **Byte accounting.** `gate_expert_bytes` / `gate_row_bytes` / mid stride come
   from the caller, already derived from the GGUF tensor type
   (`tensor_expert_bytes`), so Q4_K needs no new plumbing there — but verify the
   GLM layout validator accepts a Q4_K routed set for a *slice* (it must; Metal
   does).
5. **Streaming lookahead.** `cuda_resolve_weight_ptr(map, off, 256 * expert_bytes, …)`
   is a fixed expert *count*, type-independent; confirm the SSD-streaming path
   (unused in the resident split) is not silently assumed Q2_K-sized.
6. **Numerics to match or document.** The CUDA path accumulates `mid` in f32 and
   re-quantizes to Q8_K; Metal uses an FP16 intermediate for routed prefill
   (`QA_BEFORE_RELEASES.md` §6 discusses this). Cross-backend results are *not*
   bit-identical by construction — so the acceptance test is a logit tolerance,
   not equality, and the doc must say which intermediate the CUDA Q4_K path uses
   so the difference is explicit rather than accidental.
7. **Escape hatch.** `DS4_CUDA_GLM_MOE_TYPES=q2k` forces the old behaviour, so an
   existing Q2_K deployment cannot regress silently; the default is
   `q2k,q4k`.

### 4.2 Keep the Spark as worker; Mac as coordinator

Already the working topology. It also keeps the interactive frontend (CLI,
agent, server) on the machine the user sits at, and it sidesteps gap 3.2
entirely. Direction is performance-neutral at a balanced cut (the repo's own
`ds4-v41-split-design.md` §11.10 conclusion), so this costs nothing.

**Access path, and its one constraint.** macOS's Application Firewall blackholes
inbound TCP for the adhoc-signed `ds4`, so the coordinator is reached through a
persistent reverse tunnel (`~/ds4-tunnel/`, boot-persistent launchd daemon) that
binds `9911` on the Spark's loopback and forwards it to the Mac's loopback.
Generation, prefill and decode all work through it (verified end to end).
**Snapshots do not**: the worker's HELLO advertises only a port, and the
coordinator derives the worker's address from the accepted socket — which through
the tunnel is `127.0.0.1`. So acceptance criterion 6 (snapshot round-trip) needs
either a direct connection (ALF allow + explicit binds) or a small protocol
addition carrying the worker's reachable host. Decide this before WS 7.

### 4.3 Thermal envelope is part of the deliverable, not an afterthought

The Spark hard-locks under sustained load; `~90 °C` board is reachable within
minutes of pipeline prefill. Therefore:

* the clock cap (2100 MHz), CPU cap and the board-zone governor ship as
  prerequisites, not options (`~/thermal-protect` on the Spark, installed and
  verified live);
* every throughput number in QA must be recorded **with** the peak board
  temperature for the run; a Spark figure without one is not evidence;
* runs are bounded (frontier sweeps) with cool-downs; 250K/500K ingests are
  treated as endurance tests with the guard armed.

---

## 5. Work breakdown

Estimates assume a developer fluent in this codebase and its QA habits.

| # | Workstream | Files | Effort |
| --- | --- | ---: | --- |
| 1 | Type-traits + dispatch predicate for GLM MoE (`{Q2_K, Q4_K}`), loud failure otherwise, env escape hatch | `ds4_cuda.cu` | 0.5–1 d |
| 2 | Q4_K instantiations: prefill tile8 gate/up, down terms + reduce | `ds4_cuda.cu` | 2–3 d |
| 3 | Q4_K instantiations: expert-major gate/up + down | `ds4_cuda.cu` | 1–2 d |
| 4 | Q4_K instantiations: decode warp-per-pair, tok2-reuse, down warp, small-batch fallback | `ds4_cuda.cu` | 2–3 d |
| 5 | CPU/GPU parity harness for the GLM MoE Q4_K path (new `tests/test_glm53_moe_q4k.c` modelled on `cuda/mmq/test/test_mmq_parity.cu` and `tests/test_glm53_kda.c`) | `tests/`, `Makefile` | 1–2 d |
| 6 | Cross-machine oracle: pipeline vs single-host Q4_K, logit tolerance + `--dist-replay-check` | `tests/`, `QA_BEFORE_RELEASES.md` | 1–2 d |
| 7 | Boundary gates (2 048→2 056 pooled DSA, 4 096→4 100 prefill), snapshot round-trip across the split | `QA_BEFORE_RELEASES.md` | 1–2 d |
| 8 | Long-context endurance: 262K cold ingest, 524K alloc, thermal logging wired into the bench CSV | `ds4_bench.c`, `speed-bench/` | 1–2 d |
| 9 | Docs + release gates: `docs/DGX_SPARK.md` ("Q4 does not fit resident" → the split), `MODELS.md`, QA §6 wording, `docs/DISTRIBUTED.md` | docs, `QA_BEFORE_RELEASES.md` | 0.5–1 d |
| 10 | *(optional)* CUDA coordinator-side slice prefill fix, if the Spark must lead | `ds4_cuda.cu` | 1–3 d |
| 11 | *(optional)* IQ2_XXS for the same GLM MoE path (unlocks the full GLM 5.3 IQ2 artifacts) | `ds4_cuda.cu` | 1–2 d |

Critical path: **1 → 2 → 5 → 6**, about **1–1.5 weeks**; the full set with QA and
docs is **2–3 weeks**. Nothing here requires new architecture, new file formats
or a new transport.

**Order of work matters:** land the *prefill* path first (2) so a long ingest can
be measured behind the guard, then decode (4), then the QA matrix. Each
workstream lands green independently and the whole thing is inert until a Q4_K
file is loaded, so partial progress is not a half-broken state for Q2_K users.

---

## 6. Validation plan

Layered, cheapest first; each layer must pass before the next is trusted.

1. **Kernel parity (unit).** New GLM-MoE Q4_K test against the CPU reference
   kernels already in `ds4.c` (F16/F32/Q8_0/Q2_K/IQ2_XXS oracles exist), for
   prefill tile8, expert-major, decode warp and small-batch shapes, including
   empty experts, tile tails and scratch reuse. Extend
   `cuda/mmq/test/test_mmq_parity.cu` for the dense Q4_K entry if the dispatch
   uses it.
2. **Graph parity (integration).** One prompt, CUDA Q4_K vs CNC/Metal Q4_K, full
   vocabulary logits within tolerance; then the official continuation fixture
   and the 100-case GLM 5.3 Q4 suite from `QA_BEFORE_RELEASES.md` §3/§6.
3. **Boundary gates.** 2 048→2 056 and 4 096→4 100 rendered-token sweeps (`--dump-tokens`
   against the real template), non-finite-logit checks, multi-token
   exact-output task.
4. **Cross-machine correctness.** Pipeline vs single-host: greedy token
   agreement for ≥ 128 tokens, then logits; `--dist-replay-check` after a forced
   route drop; worker restart mid-ingest and mid-decode.
5. **Capacity/long-context.** 262 144 cold ingest, then 524 288 alloc + a short
   generation; snapshot save, restore on a fresh pair, and restore with the
   roles swapped (topology-neutral checkpoint).
6. **Thermal endurance.** Same runs with the guard armed at 88 °C band: record
   peak board per frontier, require zero thermal events and completion without
   intervention; repeat the 262K ingest twice to show it is not a one-off.
7. **Agent-level.** `tests/test_agent_compaction.py` at ≥ 64K and a real
   read/edit/test task through `ds4-agent`, since that is the actual workload.

---

## 7. Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Q4_K dot products are correct but the *expert layout* differs from the Q2_K assumption (byte offsets/rows) | silent wrong output | parity harness first (WS 5); assert per-expert byte counts against `tensor_expert_bytes` |
| Templating the tuned kernels perturbs Q2_K codegen (register pressure, shared-memory layout) | Q2_K regression | instantiate and benchmark both; keep the Q2_K path byte-run-identical where possible and gate with the existing Q2 fixtures |
| Cross-backend numeric drift larger than tolerance (CUDA f32 mid vs Metal FP16) | fails acceptance | choose and document the intermediate; if drift is structural, adopt the repo's precedent of a tolerance + selected-token gate rather than forcing bit-identity |
| Spark thermal trip during endurance runs | lost work, hardware risk | caps installed; guard aborts at 95 °C; bounded runners with cool-downs; peak-board recorded |
| macOS inbound permission never granted and the tunnel drops mid-ingest | aborted 250K run | prefer the permission grant; if tunnelling, supervise with restart-on-failure and snapshot before long steps |
| Time: the port expands (decode variants, IQ2_XXS, boundary bugs) | slip | ship prefill first; the fallbacks in §9 remain valid throughout |

---

## 8. Rollout and repository hygiene

* No new user-facing flags; the type support is a capability of the CUDA GLM
  path. `DS4_CUDA_GLM_MOE_TYPES` is diagnostic, documented with the other
  env vars.
* Update, in the same change set: `docs/DGX_SPARK.md` §GLM 5.3 (state that Q4 is
  the pipeline target and needs both machines), `MODELS.md` (the two-machine Q4
  row), `docs/DISTRIBUTED.md` (a Q4 pipeline example), and
  `QA_BEFORE_RELEASES.md` §6 (replace "Q4 … not supported in this pass" with the
  new gates).
* Keep the QA habit this repo enforces: a physical two-machine run is the gate;
  a parser/unit test is not.

---

## 9. Fallbacks if this slips (all measured today)

| Need | Use | Numbers |
| --- | --- | --- |
| Q4 quality, long context, one machine | Mac alone + `--ssd-streaming` | 82.5 t/s prefill, 8.0 t/s decode at 262K; 53 min cold ingest; 99.84 GiB plan |
| Best throughput available on the pair | Q2 pipeline (Mac coord + Spark worker) | 380.6 t/s prefill, 12.5 t/s decode at 32K, board 66–77 °C |
| Q2, one machine | Spark or Mac resident | Spark GLM 5.3 Q2: 531 t/s prefill, 14.35 t/s decode (repo QA) |

---

## 10. Open questions for the owner

1. Is Q4_K the target, or should the same work also cover **IQ2_XXS** (the
   released GLM 5.3 IQ2 artifacts) in the same pass? It is one more
   instantiation of the same template (WS 11) but widens the QA matrix.
2. Grant `ds4` Local Network permission on the Mac, or standardise on the
   supervised tunnel?
3. Is the Spark's hard-lock RMA-worthy on this unit (field diagnostic
   PowerStress)? The caps are a workaround; a defective unit will still trip.
4. Does the 500K target need to hold with `--mtp` off? MTP is excluded under the
   split, so planning should not assume speculative speedup.
