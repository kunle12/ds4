# Q2 head-to-head: GLM 5.3 Flash Q2 vs DeepSeek V4.1 Flash Q2 (output quality)

**Question:** for a large-context coding agent, is there a measurable output-quality
difference between the two Q2 models — GLM 5.3 Flash Q2 (resident, Mac) and
DeepSeek V4.1 Flash Q2 (streaming, Mac) — to justify picking one over the other?

**Date:** 2026-09-22. **Machines:** Mac Studio M4 Max only (both models run on it).
**Models:** `/Volumes/Models/glm/GLM-5.3-Flash-Q2.gguf` and
`/Volumes/Models/deepseek/DeepSeek-V4.1-Flash-Q2.gguf`.

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
**largest at 64-96K**. No GLM Q2 long-context number is recorded. That is the only
measured quality asymmetry between #2 and #3 — everything else needed a neutral
task benchmark.

## 2. Method - three neutral, objective probes

Both models, greedy (`--temp 0`), `--metal`, ctx 8192 (battery) / 32-64K (needle)
/ 131072 (fixture). GLM Q2 resident, V4.1 Q2 `--ssd-streaming` (it cannot be
resident). Same prompts. Each probe was also run with reasoning on and off.

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

**Code battery — a tie at the ceiling:**

| model | score | wall |
| --- | ---: | ---: |
| GLM 5.3 Q2 | **10/10** | ~68 s |
| DS V4.1 Q2 | **10/10** | ~256 s |

Both solved every task. GLM Q2 was ~3.8x faster here (resident at ~29 t/s vs
V4.1 streaming).

**Long-context needle:**

| run | GLM Q2 | V4.1 Q2 |
| --- | --- | --- |
| 32K, `--nothink` | values 9/9 in order; sum 8187 (fail) | values 9/9 in order; sum **37** (fail) |
| 64K, `--nothink` | values 9/9 but **order transposed**; sum 8977 (fail) | values 9/9 **in order**; sum **37** (fail) |
| 32K, **thinking on** | **sum 8097 correct; values 9/9** | **sum 8097 correct; values 9/9** |
| 64K, **thinking on** | sum 8097 correct; **order transposed again** | sum 8097 correct; values 9/9 **in order** |

(expected: sum 8097, values `[37,481,9,2604,88,715,130,3991,42]`.)

Long-context **retrieval is essentially a tie at both depths** (each 9/9). The sum
is an arithmetic/instruction weakness that hits both without reasoning; **with
reasoning on, both compute it exactly**. V4.1's repeated `SUM=37` under
`--nothink` is a format/arithmetic failure, not a retrieval failure.

One asymmetry survives: at 64K **GLM Q2 transposes `SECRET_VALUE_4`/`_5`** (reads
`88` before `2604`) in *both* modes - reproducibly, while still summing correctly -
whereas V4.1 Q2 tracks the document order exactly. It is a single-task signal, but
the probes found no other fidelity difference.

**Extended-6 fixture (64K/96K) code review - a tie.** The two code-review cases
from `deepseek-v4.1-flash-20260911-extended` (a ~64K-token archive ending in
"review this C code, explain the bounds bug") were run through both models at
ctx 131072. Both produced the correct fix (`index >= n`, plus `index < 0`), named
the bug and explained it, on **both** cases. GLM Q2's answer was tighter (~1.3 KB;
it flagged the archive as irrelevant); V4.1 Q2's was correct but chatty and leaked
its reasoning (~2.9 KB), because the run had reasoning on with no separate
thinking channel.

## 4. Conclusion

**On objective output quality the two Q2 models are equivalent.** Across
short-context coding (10/10 each), long-context retrieval at 32K and 64K (9/9
each), long-context arithmetic (exact each, with reasoning), and the repository's
64K/96K long-context code review (correct on both cases, both models), there is
**no correctness difference on which to rank #2 over #3**.

The differences that do exist are small and non-correctness:

* **Speed** - GLM Q2 is resident and ~3x faster per token than V4.1 Q2 streaming.
* **Output discipline at 64K** - GLM Q2 transposes two facts at 64K
  (reproducibly); V4.1 Q2 is exact there. V4.1 Q2's answers are more verbose and,
  in the raw CLI, leak reasoning.
* **Measured long-context NLL** - V4.1 Q2 degrades at 64-96K against its own Q4
  (0.559 vs 0.367); GLM Q2's equivalent has not been scored.
* **Model character** - different providers; a style preference.

## 5. Ranking (quality), with the caveats

1. **GLM 5.3 Flash Q4_K** (pair) - the only Q4-class option and the measured best
   on its own fixture (0.300 vs Q2's 0.458).
2. **GLM 5.3 Flash Q2** and **DS V4.1 Flash Q2** - **tied on quality**. No probe
   run here separated them on correctness. The practical tie-break is **speed**
   (GLM Q2 ~3x faster, resident) versus **long-context fidelity** (V4.1 Q2 was
   exact at 64K where GLM Q2 transposed two facts). Pick GLM Q2 for
   responsiveness; V4.1 Q2 if the 64K ordering fidelity or the DeepSeek character
   matters more.
3. No third: on quality the two Q2 options are a pair.

## 6. Caveats and what would firm it up

* n = 10 tasks, one needle probe at two depths, and two fixture cases; the
  "thinking" runs are single samples. Treat the direction as solid, not
  statistically strong.
* The needle's 64K ordering slip for GLM Q2 is a single-task signal (reproduced in
  both modes, but not a suite).
* The Extended-6 code-review cases were graded as correct/incorrect plus a manual
  read, not against the DeepSeek API reference - that reference is model-specific
  and cannot be used cross-family.
* **Still unmeasured:** a GLM Q2 long-context *NLL* number at 64-96K on its own
  fixture. The code review was correct on both models, but an NLL score would
  surface a subtler degradation if one exists, mirroring V4.1 Q2's 0.559.
