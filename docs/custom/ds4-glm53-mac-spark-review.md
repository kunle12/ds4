# GLM 5.3 Flash on Mac + DGX Spark — Code and Documentation Review

**Status:** reviewed, then fixed with the owner's consent. §9 records what was applied and
what remains open. The review itself is §1–§8.
**Scope:** the GLM 5.3 Flash Q4_K pipeline split (Mac Studio M4 Max coordinator, DGX Spark
GB10 worker): the docs under `docs/custom/` plus the code they describe.
**Constraint applied:** no structural change to the inference engine is proposed. Every
suggestion below is either a comment/doc edit, an *opt-in* environment switch, or a local
simplification that leaves all existing paths and other models untouched.
**Method:** read the design doc, implementation log and technical analysis; then read the
code the documents cite (`ds4.c`, `ds4_cuda.cu`, `ds4_distributed.c`, `ds4_server.c`) and
checked each factual claim against it. File:line references are to the working tree as of
this review.

---

## 1. Verdict

The system is well engineered and the documentation is unusually honest — it records its
own withdrawn claims, measurement conventions and open gates. The measured conclusions
(the generic MoE dispatch is faster; `0:20` beats `0:23` on prefill; decode is
weight-traffic bound) are supported by the code and the numbers.

What this review found is **not** a broken pipeline. It is:

* **one claim that the code contradicts** — a documented escape hatch that does not exist
  (§3, **F1**), which means a body of ported Q4_K kernels is dead at runtime while the docs
  say it is reachable;
* **one documented "linear" scaling model that the project's own table disproves**, with an
  overstated sensitivity ratio (§3, **F3**);
* **a decode fast-path gate that contradicts its own comment**, with a measurable-looking
  efficiency opportunity behind it (§3 **F4**, §6 **E1**);
* a handful of smaller doc/number inconsistencies (§3 **F2/F5/F6**), two robustness nits
  (§4), and one clear local duplication (§5).

Nothing here invalidates a published performance number. The two items worth real thought
are **F1** (is the ported kernel family dead code, or is the hatch meant to be real?) and
**E1** (should distributed decode use the single-token graph?). Both are owner decisions.

---

## 2. What I verified and found sound

Recorded first, because a review that only lists faults is misleading.

| Claim / area | Evidence | Result |
| --- | --- | --- |
| Wire width `N_HC × N_EMBD` (64 KiB/token) for GLM 5.3 | `ds4.c:71801-71812`; the GLM slice path sizes chunks with the same rule at `ds4.c:74470-74472`, `73532`, and `ds4_distributed.c:2685,3537,7287` | consistent; the KDA/mHC handling is coherent |
| `--dist-activation-bits` width is re-validated on receive | `ds4_distributed.c:7287-7330` recomputes `n_tokens × hidden_f32_values` and rejects a mismatch before touching KV | good; fail-before-work ordering |
| Worker-side token validation | `ds4_distributed.c:7275-7285` checks vocabulary range and the token-hash chain | good |
| Route blob is re-validated by the receiver | contiguity/NUL/port/trailing-bytes checks, `ds4_distributed.c` route validate | good |
| KDA payload sizing bug | `glm_layer_payload_tensor_bytes` (`ds4.c:60986-61037`) now returns conv+recurrent state for KDA layers, matching the sizing loop at `ds4.c:60944-60961` | the fix is internally consistent on both the sizing and writing sides |
| Served model id fix | `ds4_server.c:14951-14965` derives `base`/`-chat`/`-reasoner` from `server_model_id_from_engine` | correct; removes the 5.2 literal |
| MTP exclusion under a layer split | `ds4.c:62201-62205` requires `distributed.role == NONE` | matches the docs |
| Guard "plan admitted or refused as a whole" | budget path `ds4.c:45428-45459`; slice span accounting via `weights_model_map_spans` | matches the operating guidance |
| Numeric spot-checks | 415.0/356.3 = **+16.5 %** ✓; NLL 0.300477636/0.458177271 = **−34.4 %** ✓; 121.1 ms = 8.26 t/s ✓; −15.8 t/s = 3.96 % of 398.8 ✓; −1.30 ms = 1.32 % of 98.5 ✓ | arithmetic in the docs is correct |
| Cleanup paths in pipelined prefill | `ds4_distributed.c:3604-3777`, three exit paths | no double-free/double-destroy found; convoluted but correct |
| `DS4_GLM_MOE_TRACE` / dispatch tracing | `ds4_cuda.cu:32195-32220` and the trace at dispatch | the trace exists as documented |

The generic-dispatch promotion itself (impl-log §8 / change 6) is well supported: the
predicate is explicit, the A/B (258.9 vs 95.3 t/s, byte-identical over 64 tokens) is
recorded, and the removal of `DS4_GLM_GENERIC_MOE_Q4K` is justified.

---

## 3. Confirmed contradictions (doc vs code)

### F1 — *High.* The "reachable through `DS4_CUDA_GLM_MOE_TYPES`" claim is false; the ported Q4_K kernels are unreachable at runtime.

**Claimed in two places:**
* `ds4.c:46090-46093` (comment on `glm_graph_layer_uses_generic_routed_moe`): *"The
  GLM-specific Q4_K kernels remain reachable through `DS4_CUDA_GLM_MOE_TYPES` and are still
  covered by `make test-glm53-moe-q4k`."*
* impl-log §10, change 6: *"the ported kernels remain reachable via
  `DS4_CUDA_GLM_MOE_TYPES`"*; and §3.1, which lists the Q4_K instantiations as a
  "hatch-reachable fallback".

**What the code does:**
* `glm_graph_layer_uses_generic_routed_moe` (`ds4.c:46076-46100`) returns **true
  unconditionally** for a homogeneous Q4_K trio (`ds4.c:46094-46098`).
* Both dispatch sites consult it *first* and return through the generic path:
  * decode/one-tensor: `ds4.c:48464-48504` → `ds4_gpu_routed_moe_one_tensor`;
  * prefill/batch: `ds4.c:48608-48642` → `ds4_gpu_routed_moe_batch_tensor`.
  The GLM-specific entries (`ds4_gpu_glm_routed_moe_one_tensor` at `ds4.c:48506`,
  `ds4_gpu_glm_routed_moe_batch_tensor` at `ds4.c:48676`) are only reached when the
  predicate is **false**.
* `DS4_CUDA_GLM_MOE_TYPES` is read only inside the GLM-specific dispatch, by
  `glm_moe_types_allowed` (`ds4_cuda.cu:32158-32170`), and it can only **narrow** the
  accepted type set (`strstr` membership). It cannot re-route Q4_K back to the GLM-specific
  path.

**Consequence:** with the shipped Q4_K GGUF, setting `DS4_CUDA_GLM_MOE_TYPES=q4k` (or
anything else) changes nothing, because that function is never called for Q4_K. The ported
Q4_K instantiations described in impl-log §3.1/§5 are dead code in the default build; the
regression test `make test-glm53-moe-q4k` still covers the kernels because it drives the
GLM entry points directly, which is why the claim survived review.

There is a real asymmetry worth noting while here: the predicate's IQ2_XXS branch tests
only the **gate** tensor (`ds4.c:46083`) because the shipped recipe is a mixed trio
(IQ2_XXS gate/up, Q2_K down), but the Q4_K branch requires **all three** homogeneous. That
is fine for the current GGUFs; it is the reason the two branches do not look like each
other.

**Options (owner decision, no structural change required):**
1. **Correct the comment and the docs** and accept the kernels as test-only coverage. Cheapest.
2. **Make the hatch real:** add one env check *inside the predicate* (e.g.
   `DS4_GLM_MOE_Q4K_GENERIC=0` forces the GLM-specific dispatch for Q4_K), restoring the
   documented A/B without touching any kernel. One predicate line + comment.
3. **Delete the ported Q4_K instantiations** and the test. Larger call; contradicts the
   "keep the fallback covered" stance and the plan's WS 4/5, so not recommended without a
   deliberate decision.

### F2 — *Low.* The technical-analysis snippet of `ds4_engine_hidden_f32_values` is inverted.

`docs/custom/ds4-technical-analysis.md` §7.1 shows:

```c
if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA) return DS4_N_EMBD;
return (uint64_t)DS4_N_HC * DS4_N_EMBD;
```

The code (`ds4.c:71801-71812`) is:

```c
if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA)
    return ds4_model_is_glm53() ? (uint64_t)DS4_N_HC * DS4_N_EMBD
                                : (uint64_t)DS4_N_EMBD;
return (uint64_t)DS4_N_HC * DS4_N_EMBD;
```

The doc's version omits the 5.3/5.2 distinction and assigns `N_EMBD` to the DSA family
outright — i.e. it describes GLM 5.3 as a single stream, the exact defect the impl-log says
was fixed. The impl-log §10 change 1 describes the code correctly, so this is stale text in
the analysis doc only. Fix the snippet.

### F3 — *Medium.* "Both slopes are linear across `0:20`…`0:27`" is contradicted by the project's own frontier table; the "~4×" ratio is overstated.

impl-log §3.3 and split-design §2 present the sensitivity as `−15.8 t/s` prefill and
`−1.30 ms` decode **per layer moved**, and state both are linear. Compute the per-layer
deltas from the frontier table (impl-log §3.3, `m` = layers on the Mac):

| segment | Δlayers | Δprefill | per-layer | Δdecode | per-layer |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0:20`→`0:23` | 3 | 455.2→398.8 = 56.4 | **18.8 t/s** | 103.0→98.5 = 4.5 | 1.50 ms |
| `0:23`→`0:26` | 3 | 398.8→366.9 = 31.9 | **10.6 t/s** | 98.5→95.2 = 3.3 | 1.10 ms |
| `0:26`→`0:27` | 1 | 366.9→344.9 = 22.0 | **22.0 t/s** | 95.2→93.9 = 1.3 | 1.30 ms |

* Prefill per-layer cost varies by **2.1×** (10.6 … 22.0). `15.8` is the endpoint average
  over 7 layers ((455.2−344.9)/7), not a slope the data supports locally. This is physically
  expected — dense layers 0–2 are ~0.4 GiB, MoE layers ~4 GiB, and KDA vs sparse layers
  differ — so a *linear* model was never going to hold across a range that includes those
  boundaries.
* Decode is close enough to linear (1.10–1.50 ms) that the "−1.30 ms/layer, landed on the
  fit" statement is fair.
* "Differ by ~4×" (impl-log §3.3) / "~4× more sensitive" (split-design): in percent terms
  it is **4.0 % vs 1.3 % ≈ 3.1×**. In range-fraction terms the two are identical (both
  spans are 7 layers), so the only meaningful comparison is the percent one, and it is 3.1×,
  not 4×.

**Why it matters:** the decision it justified (`0:20`) is right — `0:20` is the best
measured point and the Mac is still the binding stage — but a future re-sweep that predicts
`0:21` from "−15.8 t/s/layer" would be off by ~3 t/s, and the "linear" claim would send
someone looking for a bug when the curve bends. Reword to "endpoint-average slope over
`0:20`…`0:27`; not uniform per layer (18.8 / 10.6 / 22.0), steepest where MoE layers move"
and "~3.1× in percent terms".

### F4 — *Medium (correctness of reasoning, efficiency of execution).* The slice-decode gate contradicts its own comment and excludes exactly the distributed-decode shape on Metal/CUDA.

`ds4.c:74489-74496`:

```c
/* The token graph accepts both embeddings and inter-node hidden
 * states. Keep the resident ROCm continuation path opt-in until
 * remote output and timing tests validate it across GLM quants. */
if (remaining == 1 && pos > 0 &&
    ((!input_hc && !output_hc) ||
     rocm_layer_slice_token_decode)) {
    ... glm_graph_forward_token(...);
```

* The comment says the token graph accepts hidden in **and** out — i.e. the distributed
  decode case — but the gate enables it only when there is **neither** input nor output
  (`!input_hc && !output_hc`), or on ROCm with an opt-in env. On the documented pair, a
  distributed decode step on the coordinator passes `input_hc == NULL`,
  `output_hc == hidden` (non-NULL); on the worker `input_hc == hidden`, `output_hc == NULL`.
  Both are excluded, so both stages take the next branch — `glm_graph_forward_tokens` with
  `chunk == 1` — the multi-token graph used for prefill.
* The KV-only single-token case (`!input_hc && !output_hc`, n=1) essentially never happens
  in the documented topology, so in practice the dedicated decode graph is **never** used
  by the pipeline on Metal or CUDA.

**Is this a measured problem?** Not proven. Decode measures 121.1 ms/token on the pair
against 8.00 t/s = 125 ms on the single Mac, both at 262K, so the pipeline is not visibly
paying a large penalty. But the single-machine number is also expert-cache bound, so the
comparison does not isolate graph cost. The honest framing: **the fast path is bypassed by
construction, and the comment says it should not be.** See §6 E1 for the proposed opt-in
A/B (mirroring the existing ROCm flag) and the measurement gate. No structural change.

### F5 — *Low.* Two small numeric inconsistencies between the two docs.

* **MTP loss:** split-design §0 says **22 %**; impl-log §4.5 says **21 %** ("3.88 t/s
  against 4.90/5.02 bracketing it" computes to 20.8–22.7 % depending on the bracket). Pick
  one number and cite the bracket; e.g. "≈21 % against the 4.90–5.02 t/s bracket".
* **Split labelling:** impl-log §4.4's "pair" column at ~287 000 is the `0:23`
  measurement (356.30 t/s / 8.30 t/s = 120.5 ms), while the recommended configuration is
  `0:20` (415.0 t/s / 121.1 ms). The table is internally consistent but does not say which
  split each column is, so a reader may attach the `0:23` figure to the recommended split.
  Label the column (or add a footnote). The ratio conclusions are unaffected — the two
  splits decode within 0.6 ms.

### F6 — *Low.* `METAL.md` omits the machine the GLM pipeline runs on.

`docs/METAL.md:21`: "The same build supports M3 and M5 Macs." The GLM 5.3 split docs
explicitly target a **Mac Studio M4 Max**. Add M4 (or make the sentence generational:
"M-series"). One-word fix.

---

## 4. Robustness findings

### R1 — *Medium.* Any pipelined-prefill failure shuts down the first-hop socket, including failures that had nothing to do with the connection.

`ds4_distributed.c:3704-3714`:

```c
if (rc == 0) dist_prefill_sender_finish(&sender);
else        dist_prefill_sender_cancel(&sender);
pthread_join(sender_tid, NULL);
if (rc == 0 && sender.rc != 0) { ...; rc = 1; }
if (rc != 0) {
    shutdown(plan->entry[0].fd, SHUT_RDWR);
}
```

`rc != 0` includes a **coordinator-local** failure — e.g. `ds4_session_eval_layer_slice`
returning non-zero for the local Mac slice, or `dist_prefill_sender_acquire_slot` returning
NULL. In those cases the transport was healthy, but the code tears down the first-hop fd.
The reviewer thread is woken (that is the point of the `shutdown`), the worker sees EOF,
`dist_worker_clear_sessions` drops **all** per-session KV, the coordinator's monitor bumps
the generation, and the next request replays the entire transcript — a ~11.5-minute ingest
at 287K for a fault that was local and might not recur.

This is *safe* (the recovery path is the documented one and does exactly what the docs
say), but it is coarse. Two non-structural options:

1. Keep the behaviour and **document it** in the operating notes: "any prefill failure
   invalidates the route and forces a transcript replay; there is no partial recovery."
2. Narrow the teardown: signal the sender/reader to stop with their own cancel flags and
   only `shutdown()` the fd when the *reader* failed (the case where a blocked read has to
   be woken). Behaviour-preserving.

Recommend (1) now — it is free — and consider (2) only if a local-error replay is ever
observed in practice. Verify on the live pair before changing anything.

### R2 — *Low.* `DS4_CUDA_GLM_MOE_TYPES` parsing is permissive and per-call.

`glm_moe_types_allowed` (`ds4_cuda.cu:32158-32170`) uses `strstr(set, "q2k")` /
`strstr(set, "q4k")` and is called on every GLM MoE dispatch (42 layers per chunk). Two
nits:

* Substring matching means `DS4_CUDA_GLM_MOE_TYPES=fooq4kbar` enables Q4_K while
  `=q4_k` (underscore, a natural typo given the tensor name) **silently disables it**, and
  the failure surfaces later as a dispatch refusal rather than at parse time.
* The env is re-read per call. The comment at `ds4_cuda.cu:32195-32199` says this is
  deliberate (a test can set it around one dispatch). Fine to keep, but see E1 for the cost.

Suggest: parse once into a small bitmask with an explicit tokenizer, and `fprintf` a
warning for unrecognised tokens. Purely local.

### R3 — *Observation, not a defect.* Unauthenticated HTTP is already documented.

`ds4_server.c` has no auth (split-design §4.2 notes it). The GLM pair exposes
`--host 0.0.0.0 --port 8081`; the docs say "trusted network or proxy". No change proposed,
just confirming the exposure is real and acknowledged.

---

## 5. Simplicity improvement

### S1 — *Low.* `hidden_dim` is recomputed inline instead of using the helper that exists for it.

`ds4.c:74470-74472`:

```c
const uint64_t hidden_dim = ds4_model_is_glm53()
    ? (uint64_t)DS4_N_HC * DS4_N_EMBD
    : (uint64_t)DS4_N_EMBD;
```

This is a hand-copy of `ds4_engine_hidden_f32_values(e)` (`ds4.c:71801`), which the same
file already calls for the output head (`ds4.c:73532`) and which `ds4_distributed.c` uses
for every wire sizing. If a future GLM changes the width rule, the helper is the natural
place to edit and this copy would silently drift — reintroducing exactly the "wire width
4× too small" class of defect the impl-log documents.

Replace the two lines with `const uint64_t hidden_dim = ds4_engine_hidden_f32_values(e);`.
One line, no behaviour change. (The `e` is already in scope.)

---

## 6. Efficiency opportunities (all opt-in, none structural)

### E1 — *Candidate, needs measurement.* Distributed decode runs the multi-token graph per token on Metal/CUDA.

Direct consequence of **F4**. Proposal, mirroring the existing precedent:

1. Extend the ROCm opt-in pattern to Metal/CUDA with a distinct env, e.g.
   `DS4_GLM_LAYER_SLICE_TOKEN_DECODE=1` (the ROCm name is
   `DS4_ROCM_GLM_LAYER_SLICE_TOKEN_DECODE`; a backend-neutral name could cover both, with
   the ROCm env kept as an alias).
2. Gate it exactly as today (default **off**), so no existing run changes behaviour and no
   other model is touched. The engine structure and graph encoders are untouched — only the
   condition that selects which existing forward function is called.
3. Measure the pair at 262K and 479K with `DS4_DIST_DECODE_PROFILE=1`, same prompt/cap,
   `cached_tokens: 0`, and compare against the recorded 121.1 ms / 120.5 ms. Require
   cross-machine oracle parity (criterion 1) before considering it more than experimental.
4. If it wins, consider changing the default for the distributed GLM decode path only after
   the oracle passes; if it does not, **delete the dead ROCm asymmetry or document why it
   stays** (today the code reads as if Metal were validated and ROCm were not, when in
   fact *neither* uses the token graph for distributed decode).

Why it might matter: decode is the metric that actually degrades at depth (the whole reason
the pair exists), and the pair's decode is documented as not GPU-bound. If the prefill graph
adds per-token launch/encode overhead, this is where it shows. If it does not, the finding
is closed with a measurement and the comment gets fixed — either way the reasoning error
goes away.

### E2 — *Micro.* Repeated `getenv` in hot paths.

* `glm_moe_types_allowed` — per MoE dispatch (`ds4_cuda.cu:32163`), ~42 calls per prefill
  chunk.
* `DS4_DIST_DISABLE_PREFILL_ACK_ONLY` — per prefill chunk (`ds4_distributed.c:3692`).
* `DS4_ROCM_GLM_LAYER_SLICE_TOKEN_DECODE` — per slice call (`ds4.c:74473-74479`).

Each is sub-microsecond, so this is not a performance bug today; it is a pattern that will
grow if more env switches are added to these paths. Optional: cache in a `static` with a
test-only reset, or accept and note. Do **not** change without measuring — the project's own
convention (Appendix B) is not to tune what has not been shown to matter.

### E3 — *Note only.* Transient hidden buffers.

Each pipelined-prefill send slot allocates `max_hidden_bytes` = `chunk_cap × 64 KiB`
(≈ 256 MiB at 4096) and the default send depth is 2 (`ds4_distributed.c:468-481`), so the
coordinator holds ≈ 512 MiB of transient slot buffers per stream. Against an 82.22 GiB plan
this is noise; recorded only so the number is on file. No change proposed.

---

## 7. Proposed plan (awaiting consent)

Ordered cheapest/safest first, each item independently landable. **Nothing structural to the
inference engine; no other model's path is touched.**

**Phase 1 — documentation and comment corrections (no runtime effect).**
1. **F1:** replace the `ds4.c:46090-46093` comment and the impl-log §10 / §3.1 wording with
   the truth: the generic dispatch is unconditional for Q4_K, and the GLM-specific Q4_K
   kernels are currently reachable only from `make test-glm53-moe-q4k`. State the intended
   disposition (see item 5).
2. **F2:** fix the `ds4_engine_hidden_f32_values` snippet in `docs/custom/ds4-technical-analysis.md` §7.1.
3. **F3:** reword "linear" to "endpoint-average slope, not uniform per layer (18.8 / 10.6 /
   22.0 t/s)" and "~4×" to "~3.1× in percent terms" in impl-log §3.3 and split-design §0/§2.
4. **F5/F6:** unify the MTP % (21 vs 22), label the impl-log §4.4 split column, add M4 to
   `docs/METAL.md:21`.
5. **Owner decision surfaced by F1:** keep the ported Q4_K kernels as test-only coverage
   (correct the docs), or add a one-line predicate env to make them a genuine A/B hatch, or
   schedule them for deletion. I recommend "keep as documented test coverage" for now, with
   the note that the hatch does not exist.

**Phase 2 — decode fast-path A/B (opt-in env only, requires measurement).**
6. **E1:** add the Metal/CUDA opt-in for the single-token slice-decode path, default off,
   then run the protocol in §6 E1. Land only with numbers and oracle parity; otherwise write
   the negative result down and fix the comment.

**Phase 3 — small robustness/simplicity (no behaviour change).**
7. **S1:** use `ds4_engine_hidden_f32_values(e)` at `ds4.c:74470`.
8. **R2:** strict env tokenizer + warning for `DS4_CUDA_GLM_MOE_TYPES`.
9. **R1:** add the "any prefill failure replays the transcript" sentence to the operating
   notes; only investigate narrowing the `shutdown()` if a local-fault replay is observed.

**Phase 4 — verification debts already tracked by the docs (not new work).**
10. Criterion 1 (cross-machine oracle), criterion 2 (boundary gates), criterion 6
    (fresh-pair and roles-swapped restore). E1's acceptability depends on (10); schedule
    accordingly.

**Explicitly not proposed:** any change to the MoE kernels, the graph encoders, the wire
record layout, the route/handshake protocol, the guard arithmetic, or the split default. The
review's whole point is that the existing design is sound enough to keep.

---

## 8. Out of scope / not reviewed

* `ds4-v41-split-design.md` and the DeepSeek V4.1 pipeline (separate model and plan).
* Tensor parallelism (`ds4_tp.c`), in-box multi-GPU placement, vision, and the CUDA/ROCm
  backends' unrelated paths — consulted only where the GLM slice path shares code.
* The Spark's host-side thermal kit (scripts outside the repository) — the docs' claims were
  taken as recorded.
* A live two-machine run. Every claim above is from source and the project's own recorded
  measurements; F4/E1 in particular **should not be treated as a measured regression** until
  Phase 2 is run on the pair.

---

## 9. Fixes applied (this change set)

Applied after consent. No structural change to the inference engine; every runtime change
either preserves the default or is behind an opt-in switch that is off by default.

**Code**

| Item | File | Change |
| --- | --- | --- |
| **S1** | `ds4.c:74470` | `hidden_dim` now comes from `ds4_engine_hidden_f32_values(e)` instead of a locally re-derived copy of the width rule. Behaviour identical; removes the drift vector that produced the original 4× wire bug. |
| **F4/E1** | `ds4.c:74473-74496` | The GLM slice single-token gate now accepts a backend-neutral opt-in, `DS4_GLM_LAYER_SLICE_TOKEN_DECODE=1`, so the decode graph is reachable for inter-node hidden state on Metal/CUDA as well as ROCm. **Default off** — unset, behaviour is byte-for-byte the previous batch-graph path. ROCm keeps `DS4_ROCM_GLM_LAYER_SLICE_TOKEN_DECODE` as an alias. The comment now describes what the gate actually does. |
| **F1** | `ds4.c:46085-46100` | Predicate comment corrected: the generic dispatch is the only selector for a homogeneous Q4_K trio, `DS4_CUDA_GLM_MOE_TYPES` cannot re-route it, and the ported Q4_K kernels are test-only coverage. No code-path change. |
| **R2** | `ds4_cuda.cu:32158-32200` | `DS4_CUDA_GLM_MOE_TYPES` is parsed as exact comma/space tokens (`glm_moe_types_parse`) with a one-time warning for unrecognised tokens, replacing `strstr` substring matching. Default `q2k,q4k` unchanged. |

**Docs / comments**

| Item | File | Change |
| --- | --- | --- |
| **F2** | `ds4-technical-analysis.md` §7.1 | `ds4_engine_hidden_f32_values` snippet corrected to match the code (GLM 5.3 → `N_HC × N_EMBD`). |
| **F3** | impl-log §3.3; split-design §2, §0/§10 | "Both slopes linear / differ by ~4×" replaced with the endpoint-average framing and the real per-segment figures (18.8 / 10.6 / 22.0 t/s; ~3.1× in percent terms). |
| **F5** | impl-log §4.4, §4.5; split-design §0 | Pair column labelled `0:23` with the `0:20` figure noted as an improvement; MTP loss unified to ~21 %. |
| **F6** | `METAL.md:21` | Build support now reads M3, M4 and M5. |
| **F1** | impl-log §10 change 6, §3.1; split-design §4.1 item 7, §8 | "Hatch-reachable later / reachable via `DS4_CUDA_GLM_MOE_TYPES`" corrected to test-only coverage. |
| **R1** | impl-log §9 | New operating note: any pipelined-prefill failure tears down the first-hop connection and replays the transcript; no partial recovery. |
| **E1** | impl-log §3.1; technical-analysis §7.2.11 | The new `DS4_GLM_LAYER_SLICE_TOKEN_DECODE` switch documented with its default. |

**Verification done here**

* `make ds4.o` and `make -j4` (all five Metal targets) compile and link clean with
  `-Wall -Wextra`.
* `ds4_cuda.cu` **was not compiled** — there is no `nvcc` on the Mac. The R2 change uses
  only `strncmp`/`fprintf`/`getenv` (already included); it must be checked with
  `make cuda-spark` on the Spark before being trusted.
* No runtime run was performed: the E1 switch is off by default, so the pair's behaviour is
  unchanged. Enabling it still needs the A/B in §6 E1 and cross-machine oracle parity.

**Measured after the change set (2026-09-21, on the live pair)**

* **Single-Mac `--ssd-streaming` regression found and fixed.** While building the
  criterion-1 oracle, the single-Mac Q4_K streaming *generation* path was found
  broken by `ce4d214` — the Q4_K generic-dispatch promotion, which was A/B'd only
  on the resident pair. Under streaming the generic dispatch reads expert weight
  ranges the streaming map has not covered
  (`Metal model range … not covered by mapped model views`), so prefill or decode
  fails; `--dump-logits` (sync prefill) still worked, which is why prefill-only
  checks missed it. Fix: keep the generic dispatch for resident graphs and restore
  the GLM-specific one while a graph is streaming. The pair is unchanged
  (382.3 t/s prefill / 92.9 ms decode, i.e. generic still in use) and single-Mac
  generation works again. Full evidence:
  `ds4-glm53-ssd-streaming-regression.md`.
* **E1 A/B: the switch wins, and the gain grows with depth.** With
  `DS4_GLM_LAYER_SLICE_TOKEN_DECODE=1`, distributed decode is **+7.7 %** at
  ~11 K, **+15.4 %** at ~285 K and **+19.6 %** at ~473 K (median inter-chunk
  gap; the identical-output wall time agrees within 0.2 %). Prefill is unchanged
  and the completion text is byte-identical at all three depths. The 285 K
  run reproduced the documented baseline (415 t/s prefill, 121.1 ms decode).
  Full numbers and method: `ds4-glm53-e1-decode-graph-ab.md`. The owner then
  applied it: the switch is set in `~/bin/llm_config.json` **and** the engine
  default for a single-token GLM 5.3 step on Metal/CUDA (explicit falsy value
  opts out; ROCm and non-5.3 GLM stay opt-in).

**Remaining items — disposition**

All tracked items are now resolved or explicitly deferred; the full list is in
`ds4-glm53-open-items.md`. In short:

* Single-Mac `--ssd-streaming` 32K/262K re-measured and matching the documented
  figures; the 512K/MTP numbers are deferred as ~1.8 h per configuration with no
  current decision depending on them.
* Criterion 6 met (fresh-pair and roles-swapped restores); WS10 closed as not
  reproducible; the agent-level real task passes.
* Decisions: keep the ported Q4_K kernels as test-only coverage; do not implement
  a streaming-aware generic dispatch; a QA gate now requires a single-host
  streaming check for any dispatch-selection change.
* Thermal soak repeated (peak 75 °C, no throttling); the EEE ablation is deferred
  as an infrastructure risk.
