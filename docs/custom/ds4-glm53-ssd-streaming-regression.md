# GLM 5.3 Q4_K single-Mac `--ssd-streaming` regression (`ce4d214`) and fix

**Found:** 2026-09-21, while building the criterion-1 oracle.
**Status:** fixed in the working tree; verified on the Mac.
**Severity:** high for the single-Mac route, zero for the pipeline pair.

---

## 1. What broke

Commit `ce4d214` *"glm moe: promote the generic Q4_K dispatch, because it is
2.7x faster"* made `glm_graph_layer_uses_generic_routed_moe()` return true for a
homogeneous Q4_K trio **unconditionally**. The A/B behind it was run on the
**resident pair** (Mac coordinator `0:20`, Spark worker `21:output`), where the
generic tensor-core tile16 kernels are indeed ~2.7× faster at prefill
(258.9 vs 95.3 t/s). It was never run against a **single-Mac
`--ssd-streaming`** graph.

Under `--ssd-streaming` the generic routed-MoE dispatch reads expert weight
ranges that the streaming map does not cover, so the run dies:

```
ds4: Metal model range 2.55..3.82 GiB is not covered by mapped model views
ds4: Metal model range 3.82..5.08 GiB is not covered by mapped model views
ds4: Metal model range 5.08..6.35 GiB is not covered by mapped model views
ds4: GLM prefill failed            # or: "GLM decode failed at position ..."
```

The error comes from `ds4_gpu_wrap_model_range()` (`ds4_metal.m:12858`): a
tensor read needs a model range that no registered model view covers. The
ranges are the dense/early-weight region; the generic dispatch's weight access
is not aware of the streaming window that the GLM-specific dispatch uses.

## 2. Evidence

Reproduced through every frontend, at several contexts and prompt sizes:

| frontend | invocation | ctx | prompt | result |
| --- | --- | ---: | ---: | --- |
| CLI `--dump-logits` | sync prefill only | 32768 | 1466 | **OK** |
| CLI generate | `--nothink`, temp 0 | 32768 | 1466 | fail (decode, pos 1460) |
| CLI generate | `--ssd-streaming-cache-experts 16GB` | 32768 | 1466 | fail (decode) |
| CLI generate | auto cache | 32768 | ~200 | fail (prefill) |
| CLI generate | `-p "Reply with exactly: OK"` | 1024 | tiny | fail (prefill) |
| `ds4-server` + HTTP | chat completion | 32768 | 1460 | fail (`metal GLM decode failed`) |
| CLI generate | | 262144 | 1466 | fail (decode) |
| `ds4-bench` | `--gen-tokens 128` | 4096 | 1448 | fail (decode) |

The `--dump-logits` path (which prefills via `ds4_session_sync` and copies
logits) worked throughout, which is why prefill-only checks did not catch this.

**Bisect.** A binary built from `6e1f447` (`ce4d214^`, the immediately preceding
commit) produces `OK` and reports `GLM prefill: 4.97 t/s, generation: 4.95 t/s`
at ctx 1024 / 16 GB cache. HEAD fails the same command. `ce4d214`'s only source
change is the predicate (verified with `git show ce4d214 -- ds4.c`), so the
regression is uniquely attributable to it.

**Consequence.** The documented single-Mac Q4_K `--ssd-streaming` figures —
32 768: 84.2 / 8.8, 262 144: 82.5 / 8.0, 99.84 GiB plan — were measured before
`ce4d214`, and the "Mac alone + `--ssd-streaming`" fallback in
`ds4-glm53-q4-split-design.md` §9 currently cannot generate. (The earlier
suspicion of a 2048-token dense/indexed "boundary" bug was a symptom of this
same regression, not a separate defect: crossing 2048 changes which MoE dispatch
path a chunk takes.)

## 3. Fix

Keep the generic dispatch for resident graphs, and restore the GLM-specific
dispatch while a graph is streaming:

```c
/* ds4.c, above glm_graph_layer_uses_generic_routed_moe */
static bool g_glm_ssd_streaming_active;      /* set in glm_graph_alloc_slice */

if (l->ffn_gate_exps->type == DS4_TENSOR_Q4_K &&
    l->ffn_up_exps->type == DS4_TENSOR_Q4_K &&
    l->ffn_down_exps->type == DS4_TENSOR_Q4_K) {
    return !g_glm_ssd_streaming_active;
}
```

`g_glm_ssd_streaming_active` is assigned next to `g->ssd_streaming = ssd_streaming`
in `glm_graph_alloc_slice`, so it reflects the graph actually in use.

This is the smallest change that restores the documented behaviour:

* **Streaming** → GLM-specific dispatch (pre-`ce4d214` behaviour, known to work).
* **Resident** (the pair, both slices) → generic dispatch, so the measured
  2.7× prefill gain is untouched.
* No other model family is affected: the branch is inside the GLM routed-MoE
  predicate.

## 4. Verification

| check | result |
| --- | --- |
| single-Mac `--ssd-streaming`, ctx 1024, 16 GB cache, 16 tokens | `stdout: OK`, generation 6.89 t/s |
| pair, 8 K, default (E1 on) | 382.3 t/s prefill, 92.9 ms decode — unchanged from before the fix, i.e. generic dispatch still in use |

## 5. Follow-ups

1. **Re-measure the single-Mac Q4_K `--ssd-streaming` numbers** with the fix
   before quoting them again (the old figures predate `ce4d214`).
2. The real fix for the 2.7× could be to make the **generic** dispatch
   streaming-aware, so the single-Mac route also gets the faster kernels. This
   guard is the minimal safe restore, not that optimisation.
3. Any future dispatch promotion needs a single-host `--ssd-streaming` check in
   its A/B — the pair-only A/B is what let this through.
