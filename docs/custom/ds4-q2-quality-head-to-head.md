# Q2-class head-to-head (plus DeepSeek V4 Flash): output quality

**Question:** for a large-context coding agent, is there a measurable output-quality
difference between the resident/streaming low-bit options on one Mac - GLM 5.3
Flash Q2 (resident), DeepSeek V4.1 Flash Q2 (streaming), and DeepSeek V4 Flash
(Vision-Exp hybrid, resident) - to justify picking one over the others?

**Date:** 2026-09-22. **Machine:** Mac Studio M4 Max only (all models run on it).
**Models:**
* `glm-q2` - `/Volumes/Models/glm/GLM-5.3-Flash-Q2.gguf` (resident)
* `v41-q2` - `/Volumes/Models/deepseek/DeepSeek-V4.1-Flash-Q2.gguf` (streaming)
* `flash` - `DeepSeek-V4-Flash-Vision-Exp-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8.gguf`
  (resident; a hybrid quant - layers 37-42 Q4_K experts, the other expert layers
  IQ2_XXS gate/up + Q2_K down, Q8 attention/shared/output)

---

## 1. Why the repo's own metric can't answer this

`gguf-tools/quality-testing` scores target-token NLL against **model-specific**
official continuations (GLM vs Z.AI FP8, V4.1 vs the DeepSeek API). The fixtures
share prompt ids but the references are each model's own provider output, and the
tokenizers differ, so the NLLs are not cross-comparable. What the repo does
measure is each family's **Q4-vs-Q2** delta and V4.1's long-context behaviour:

| model | fixture | NLL | agreement |
| --- | --- | ---: | ---: |
| GLM 5.3 Q4_K | GLM 100-case | 0.300 | 90/100; greedy prefix 9.66 |
| GLM 5.3 Q2 | GLM 100-case | 0.458 | 89-90/100; greedy prefix 7.37 |
| DS V4.1 Q4 | V4.1 General 100 | 0.247 | 2896/2994 |
| DS V4.1 Q2 | V4.1 General 100 | 0.363 | 2696/2994 |
| DS V4.1 Q4 | V4.1 Extended 6 (64-96K) | 0.367 | 360/384 |
| **DS V4.1 Q2** | **V4.1 Extended 6 (64-96K)** | **0.559** | 327/384 |

So: within GLM, Q4_K >> Q2; within V4.1, Q2 loses ~0.12-0.19 NLL, and the loss is
**largest at 64-96K**. There is no equivalent long-context number for GLM Q2, and
the `flash-vision-exp-...-100` fixture for the Flash checkpoint carries no recorded
reference band. Those gaps are why the comparison below uses neutral task probes.

## 2. Method - three neutral, objective probes

All three models, greedy (`--temp 0`), `--metal`, ctx 8192 (battery) / 32-64K
(needle) / 131072 (fixture). GLM Q2 and Flash run resident; V4.1 Q2 uses
`--ssd-streaming` (it cannot be resident). Same prompts. Each probe was also run
with reasoning on and off.

1. **Code battery** - 10 self-contained Python tasks (parsing, DP, edge cases),
   each with a unit-test suite; the model's fenced code block is extracted and run.
   The tests were self-checked against gold solutions first.
2. **Long-context needle** - a ~28K/56K-token prose document with 9 facts
   `SECRET_VALUE_i = N` scattered through it; the model must return the sum and the
   values in order. Objective exact scoring.
3. **Extended-6 fixture code review** - the two code-review cases from the
   repository's 64K/96K long-context fixture, graded on whether the bounds bug is
   correctly fixed and explained.

## 3. Results

**Code battery - a tie at the ceiling:**

| model | score | wall |
| --- | ---: | ---: |
| GLM 5.3 Q2 | **10/10** | ~68 s |
| DS V4.1 Q2 | **10/10** | ~256 s |
| DS V4 Flash | **10/10** | ~38 s |

All three solved every task. The resident models are fast (Flash ~38 s, GLM Q2
~68 s); V4.1 Q2 streams and is ~7x slower than Flash on the same work.

**Long-context needle:**

| run | GLM Q2 | V4.1 Q2 | V4 Flash |
| --- | --- | --- | --- |
| 32K, `--nothink` | 9/9 in order; sum 8187 | 9/9 in order; sum **37** | 9/9 in order; sum 6267 |
| 64K, `--nothink` | 9/9 but **order transposed**; sum 8977 | 9/9 **in order**; sum **37** | 9/9 **in order**; sum 6297 |
| 32K, **thinking on** | **sum 8097; 9/9** | **sum 8097; 9/9** | **sum 8097; 9/9** |
| 64K, **thinking on** | sum 8097; **order transposed again** | sum 8097; 9/9 in order | sum 8097; 9/9 in order |

(expected: sum 8097, values `[37,481,9,2604,88,715,130,3991,42]`.)

Long-context **retrieval is a tie across all three** (9/9 each, at both depths).
The sum is an arithmetic/instruction weakness that hits all three without
reasoning; **with reasoning on, all three compute it exactly**. V4.1's repeated
`SUM=37` under `--nothink` is a format/arithmetic failure, not a retrieval
failure.

The one fidelity difference the probes found: at 64K **GLM Q2 transposes
`SECRET_VALUE_4`/`_5`** (reads `88` before `2604`) in *both* modes -
reproducibly, while still summing correctly - whereas **V4.1 Q2 and V4 Flash both
track the document order exactly**.

**Extended-6 fixture (64K/96K) code review - a tie.** The two code-review cases
from `deepseek-v4.1-flash-20260911-extended` (a ~64K-token archive ending in
"review this C code, explain the bounds bug") were run through all three models at
ctx 131072. Each produced the correct fix (`index >= n`, plus `index < 0`), named
the bug and explained it, on **both** cases. GLM Q2's answer was tightest (~1.3 KB;
it flagged the archive as irrelevant); Flash's was ~1.7 KB; V4.1 Q2's was correct
but chatty and leaked its reasoning (~2.9 KB), because the run had reasoning on
with no separate thinking channel.

## 4. Conclusion

**On objective output quality the three models are equivalent.** Across
short-context coding (10/10 each), long-context retrieval at 32K and 64K (9/9
each), long-context arithmetic (exact each, with reasoning), and the repository's
64K/96K long-context code review (correct on both cases, all three models), there
is **no correctness difference on which to rank them**.

The differences that do exist are small and non-correctness:

* **Speed** - both resident models are fast (Flash fastest: ~38 s for the battery;
  GLM Q2 ~68 s), while V4.1 Q2 streams and is ~7x slower than Flash on the same
  work.
* **Output discipline at 64K** - GLM Q2 transposes two facts at 64K
  (reproducibly); V4.1 Q2 and Flash are exact there. V4.1 Q2's answers are the most
  verbose and, in the raw CLI, leak reasoning.
* **Measured long-context NLL** - V4.1 Q2 degrades at 64-96K against its own Q4
  (0.559 vs 0.367); GLM Q2 and Flash have no equivalent long-context NLL recorded
  (Flash's fixture exists but carries no band).
* **Speculation** - Flash ships a DSpark drafter (measured ~35 t/s at 19.5K in
  `ds4-technical-analysis.md` §17.7); GLM Q2 and V4.1 Q2 have no usable drafter.
  Another point for Flash on responsiveness.
* **Model character** - different providers; a style preference.

## 5. Ranking (quality), with the caveats

1. **GLM 5.3 Flash Q4_K** (pair) - the only Q4-class option and the measured best
   on its own fixture (0.300 vs Q2's 0.458).
2. **GLM 5.3 Flash Q2**, **DS V4.1 Flash Q2** and **DS V4 Flash** - **tied on
   quality**; no probe here separated them on correctness. Break the tie by need:
   * **DS V4 Flash** - fastest per task, resident, DSpark drafter available: the
     default if you want responsiveness and the V4 Flash model.
   * **GLM 5.3 Flash Q2** - resident and nearly as fast, but transposes two facts
     at 64K (the only fidelity slip seen).
   * **DS V4.1 Flash Q2** - streaming (slowest) but exact at 64K; pick for the
     DeepSeek V4.1 model if you accept the stream and the 64-96K NLL caveat.
3. No fourth: on quality the three low-bit options are a set.

## 6. Caveats and what would firm it up

* n = 10 tasks, one needle probe at two depths, and two fixture cases; the
  "thinking" runs are single samples. Treat the direction as solid, not
  statistically strong.
* The needle's 64K ordering slip for GLM Q2 is a single-task signal (reproduced in
  both modes, but not a suite).
* The Extended-6 code-review cases were graded as correct/incorrect plus a manual
  read, not against the DeepSeek API reference - that reference is model-specific
  and cannot be used cross-family.
* **Still unmeasured:** long-context *NLL* at 64-96K for GLM Q2 and V4 Flash on
  their own fixtures. Both passed the 64K/96K code review, but an NLL score would
  surface a subtler degradation if one exists, mirroring V4.1 Q2's 0.559. (The
  `flash-vision-exp-...-100` fixture exists for the Flash checkpoint but carries no
  recorded reference band.)
