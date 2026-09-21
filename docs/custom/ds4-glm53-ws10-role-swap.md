# WS10 — CUDA coordinator-side slice prefill: not reproducible on the current build

**Reported:** *"CUDA coordinator-side slice prefill: `CUDA tensor read failed:
unspecified launch failure` for chunks ≥ 512 rows; reproduced on pristine
`8db1d1d`; still open (WS 10)."*
**Investigated:** 2026-09-21, on the live pair. **Outcome: does not reproduce;
roles-swapped topologies work.**

---

## 1. Method

Roles **swapped** from the operational topology: the DGX Spark GB10 runs the
CUDA **coordinator** (`--layers 0:20`) and the Mac M4 Max runs the Metal
**worker** (`--layers 21:output`), ctx 8192. This forces the coordinator's local
slice prefill through the CUDA backend, which is the reported crash site.
`CUDA_LAUNCH_BLOCKING=1` was set on the coordinator so a failing launch would
surface at its own call rather than as a sticky error at the later D2H copy.

Prompt: 5 018 tokens of `speed-bench/promessi_sposi.txt` (plus an 851-token one);
`--dist-prefill-chunk` varied to hit the reported condition exactly.

## 2. Results

| model | prompt rows | prefill chunk | outcome |
| --- | ---: | ---: | --- |
| Q4_K | 851 | session cap | OK — full logits produced |
| Q4_K | 5 018 | 4 096 | OK — 100 % |
| Q4_K | 5 018 | **512** (10 chunks) | OK — 100 % |
| Q2 | 5 018 | 4 096 | OK — 100 % |

No `unspecified launch failure`, no `tensor read failed`, no non-finite logits, in
any run.

**Cross-coordinator agreement.** The 5 018-token prompt run with the Spark as
coordinator was compared against the same prompt with the Mac as coordinator
(both `0:20`, same `ctx_cap` 4096):

| metric | value |
| --- | ---: |
| mean \|Δ\| | 0.352405 |
| max \|Δ\| | 2.590694 |
| top-1 / top-8 / top-16 / top-64 | 1/1 · 8/8 · 13/16 · 56/64 |
| argmax equal | yes |

So a roles-swapped pair produces the same distribution as the operational one,
to the same order as the cross-backend comparisons elsewhere.

## 3. Why it no longer reproduces

Two independent reasons, either sufficient:

1. **The report predates the GLM work.** It was measured against pristine
   `8db1d1d`, the fork point, before the Q4_K port, the expert-map and KDA
   fixes, the memory guard and the mixed-prefill handling. One of those resolved
   it; a definitive bisect would require rebuilding the fork point, which is not
   worth doing because the current tree is the deployment target.
2. **The GLM-specific CUDA dispatch it was reported against is now unreachable
   for shipped models.** The graph predicate sends a homogeneous Q4_K trio to the
   generic dispatch (resident) and the shipped Q2 recipe is `IQ2_XXS` gate/up
   with `Q2_K` down (`--inspect`: 86 + 43 routed tensors), which the predicate
   also sends to the generic path. The GLM-specific CUDA entry is now reached
   only for Q4_K under `--ssd-streaming` (the regression guard) and for a
   hypothetical homogeneous `Q2_K` trio, which no shipped GLM 5.3 model has.

## 4. Consequence

* **Roles-swapped (Spark-led) works.** The WS10 dependency on criterion 6's
  roles-swapped snapshot restore is removed.
* The operational recommendation is unchanged — the Mac leads because the user
  sits there and it holds the interactive frontend — but that is now a
  preference, not a correctness constraint.
* If pristine `8db1d1d` is ever rebuilt, re-running the 512-row-chunk roles-swap
  is the cheap way to attribute the fix; otherwise WS10 can be closed.
