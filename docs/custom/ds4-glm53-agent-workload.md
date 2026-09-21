# Agent-level workload on the pair (and a CUDA shared-expert fix it surfaced)

**Status:** passed 2026-09-21. A real read/edit/test coding task completes
end-to-end through `ds4-agent` running as the coordinator across the split.

---

## 1. Setup

* `ds4-agent` as **coordinator** (Mac, Metal, `layers 0:20`) with the Spark as
  worker (CUDA, `21:output`), ctx 8192.
* Headless one-shot: `--non-interactive -p "<task>" --chdir /tmp/agent-task
  --trace /tmp/agent.trace`. No TUI, no `pyte`.
* Task repo: `main.py` with `add(a,b) = a - b` (the bug) and `test_main.py`
  asserting `add(2,3) == 5`, `add(-1,1) == 0`, `mul(4,5) == 20`; the test fails
  before the run.
* Prompt: *"running python3 test_main.py fails. Read the files, fix main.py so
  all assertions pass, then run python3 test_main.py and tell me whether it
  passed."*

## 2. Result

The agent, in one turn:

1. listed the working directory,
2. read `main.py` and `test_main.py`,
3. identified that `add` used subtraction,
4. edited `return a - b` → `return a + b`,
5. ran `python3 test_main.py` → `ALL TESTS PASSED`,
6. reported the fix and the result.

After the run `main.py` is correct and the test passes (`rc=0`). So the actual
workload — an agent doing read/edit/run over a split model — works on the pair.

## 3. Finding: the CUDA worker's fused shared-expert swiglu was bypassed

The worker log carried **24 ×**
`CUDA glm shared swiglu one failed: operation would make the legacy stream
depend on a capturing blocking stream` — exactly one per decode layer (the worker
holds 24). It appeared in **every** run with the E1 token-graph decode on, and
**never** when E1 was off (`e1-worker-*-on` vs `*-off` logs).

**Cause.** `glm_shared_gate_up_swiglu_one_kernel` (and its tok2 sibling) were
launched with no stream argument — the legacy stream — while the E1 token graph
keeps a capture in flight; every other decode-island kernel launches on
`cuda_decode_stream()`. The launch is rejected
(`cudaErrorStreamCaptureImplicit`), `cuda_ok()` returns 0, and the caller
(`DS4_GLM_ENCODE_FFN_BATCH_SHARED`) falls back to the unfused matmul+swiglu path.
That fallback is why output stayed correct — including criterion 1's
byte-identical continuation — but the fused fast path was silently disabled
during decode, with an error printed per layer.

**Fix.** Launch both GLM shared swiglu kernels on `cuda_decode_stream()`
(`ds4_cuda.cu`). In eager mode the helper returns the legacy stream, so behaviour
is unchanged; during capture the launch now rides the capture stream like the
rest of the island.

## 4. Verification of the fix

Rebuilt and installed on the Spark (`ds4_cuda.cu`, 22:31). A fresh pair run at
~11.5 K tokens:

| check | before | after |
| --- | ---: | ---: |
| worker `shared swiglu one failed` lines | 24 | **0** |
| decode median gap | 93.70 ms | 93.78 ms |
| output sha256 | `a3898c57…` | `a3898c57…` |

The fused shared-expert path now runs under capture without error, and the decode
aggregate is unchanged — the fallback's cost was below the run-to-run noise at
this depth, so this is a correctness/robustness fix for the intended fast path
rather than a measured speed-up.

## 5. Notes

* `pyte` is not installed, so `tests/test_agent_compaction.py` — a PTY
  near-full-context regression — was not run. The headless task above is the
  substantive agent check; the compaction test additionally needs the script to
  forward the distributed `--role/--layers/--listen` arguments.
