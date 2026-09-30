# Plan: in-place DeepSeek V4 rewind for the DSpark speculative boundary

Status: IMPLEMENTED + HARDENED (revision 4) — built and installed; awaiting end-to-end test.
Date: 2026-09-30

Implementation: `ds4.c` (session fields + logits buffer,
`ds4_session_dspark_note_rewind_frontier`, recorded in both DSpark verify paths,
`ds4_session_rewind_speculative_boundary`, invalidation/flag clearing),
`ds4_server.c` (`server_generation_rewind` fast path under `inference_mu` + generation
log), `ds4.h` (declaration). Built clean (`-Wall -Wextra`); `ds4-server` installed to
`/Users/xun/bin/ds4-server` (sha `e9e6226f…`). See §8 for the hardening pass.

## 1. Problem

An agent (OpenCode) request intermittently **hangs** when the model enters tool-call
syntax during DeepSeek V4 Flash + DSpark speculative decoding (~109K context):

```
generate_job_inner (ds4_server.c:14269) → server_generation_rewind (12608)
 → ds4_session_sync → metal_graph_prefill_layer_major
  → ds4_gpu_wait_command_buffer → -[MTLCommandBuffer waitUntilCompleted]  (never returns)
```

`sample`: 2893/2919 in `waitUntilCompleted`; CPU 0.8%; log frozen. The client sees a
stalled stream → `Decode error (200 POST …)`.

Trigger: in `ds4_server.c:14265-14276`, a speculative block commits `ntok` tokens but
the DSML tracker stops early (`kept < ntok`), so the server rewinds to
`pos = block_start + kept (- 1 if resample)` and re-evaluates under the new mode.
`ds4_session_rewind` cannot roll DeepSeek's compressor frontier back, sets
`checkpoint_valid=false` (`ds4.c:85306-85308`), and the caller rebuilds the whole
context → the prefill hangs.

## 2. Root cause

Requirements already encoded by `spec_frontier_snapshot`/`spec_frontier_restore`
(`ds4.c:62289-62362`): per-layer compressor frontiers `layer_attn_state_kv/score[]`,
`layer_index_state_kv/score[]`; counters `layer_n_comp[]`/`layer_n_index_comp[]`
(`ds4.c:17200-17201`); `mtp_n_raw`; DSpark window
`dspark_cache_start/token_start/len`. Raw SWA / compressed row bodies are append-only
and invisible beyond the counters (comment `ds4.c:62364-62367`) and need no restore.

The DSpark verifier already snapshots this at block start (`ds4.c:80478` argmax,
`81031` stochastic); it is discarded before returning, so the rewind cannot use it.

## 3. Design (revised)

**Persist the verifier's block-start frontier on the session; expose a dedicated
DeepSeek-only rewind that restores it and replays the retained tokens, including the
`logits` at the restore position. Call it only from `server_generation_rewind`.**

Rationale for the dedicated API (not modifying `ds4_session_rewind`): other callers
(agent, `REUSE_MEMORY_REWIND` at `ds4_server.c:13525`) must keep today's
rebuild/truncate behaviour; a stale persisted snapshot must never be restorable from
those paths. Scoping to the server generation-rewind guarantees the snapshot was taken
by the immediately preceding verify in the same request.

### 3.1 New session state (`#ifndef DS4_NO_GPU`)

```c
ds4_spec_frontier ds_rewind_frontier;  /* block-start frontier (metadata)          */
int               ds_rewind_pos;       /* verifier `start` at snapshot time        */
int               ds_rewind_end;       /* == start + draft_n                       */
bool              ds_rewind_valid;
float            *ds_rewind_logits;    /* s->logits at snapshot (V floats)         */
```

`ds4_spec_frontier` is at `ds4.c:60207-60215` (before the session struct). Metadata
only (~0.7 KiB); add `ds_rewind_logits` (~505 KiB) allocated lazily under the same
condition the graph uses for spec verifiers (`need_spec_verifier`, `ds4.c:73159-73162`),
next to `spec_row_logits` (`ds4.c:73265`); freed with the session.

### 3.2 Record on every DeepSeek DSpark verify

In `ds4_session_eval_dspark_speculative_argmax` (`ds4.c:80313`) after
`spec_frontier_snapshot(&frontier, s)` succeeds (`ds4.c:80478`), and in
`ds4_session_eval_dspark_speculative_stochastic` (`ds4.c:80906`) at its snapshot
(`ds4.c:81031`), with `start = s->checkpoint.len` and `draft_n`:

```c
if (have_frontier) {
    s->ds_rewind_frontier = frontier;
    s->ds_rewind_pos   = start;
    s->ds_rewind_end   = start + draft_n;
    s->ds_rewind_valid = s->ds_rewind_logits != NULL;  /* logits required for kept==1 */
    if (s->ds_rewind_logits)
        memcpy(s->ds_rewind_logits, s->logits, (size_t)DS4_N_VOCAB * sizeof(float));
}
```

`start` is `s->checkpoint.len` at snapshot time: after the seed probe pushed
`first_token` in the normal path (`start = block_start + 1`), or at the block start
under fused seed batching. The replay is driven by the checkpoint token vector, so
both align; `s->logits` at this point is exactly the distribution the server needs
when `pos == base`.

### 3.3 Dedicated rewind API

New in ds4.c (declared in ds4.h):

```c
/* Returns 0 when the live graph was rewound in place to `pos`; nonzero to let the
 * caller fall back to the rebuild path. DeepSeek V4 DSpark only. */
int ds4_session_rewind_speculative_boundary(ds4_session *s, int pos);
```

Body (as built):
- Require: `s->ds_rewind_valid` **and** `s->checkpoint_valid`, family `DEEPSEEK4`,
  not `s->distributed`, `!(s->engine && s->engine->tp.active)`,
  `ds_rewind_pos <= pos <= ds_rewind_end`, `pos < s->checkpoint.len`.
- Save `saved_len = s->checkpoint.len`.
- Copy the retained tokens `s->checkpoint.v[base..pos-1]` (`base = ds_rewind_pos`) into a
  local `DS4_DSPARK_MAX_BLOCK_SIZE` (16) array **before** mutating (`n < 16` is provable).
- `spec_frontier_restore(&s->ds_rewind_frontier, s)` (restores counters, frontiers,
  DSpark window, `mtp_n_raw`); on failure it can have copied only some layers, so set
  `checkpoint_valid=false` and `ds_rewind_valid=false` and return -1.
- `s->checkpoint.len = base;` then:
  - if `pos == base`: `memcpy(s->logits, s->ds_rewind_logits, V*4)` (the `kept==1` case —
    the verifier overwrote `s->logits` with the block-end distribution);
  - else: replay `for i in [base,pos)` with
    `metal_graph_eval_token_raw_swa(&s->graph, &e->model, &e->weights, tok, checkpoint.len, s->logits)`
    then `token_vec_push`, mirroring `ds4_session_sync_internal` (`ds4.c:76363-76384`).
    On any replay failure restore `s->checkpoint.len = saved_len`, clear validity, and
    return -1 so the caller's rebuild re-prefills the intended prompt (ds4_session_rewind
    would otherwise early-return on the truncated vector).
- Success: `checkpoint.len = pos`, `checkpoint_valid = true`, `mtp_draft_valid = false`,
  `ds4_session_dspark_capture_invalidate(s)` (as the normal rewind does),
  `session_greedy_splitkv_reset(s)`, `ds_rewind_valid = false`, return 0.

### 3.4 Server hook

`server_generation_rewind` (`ds4_server.c:12608`): before the existing
`ds4_session_rewind` + `common_prefix` rebuild decision, try the in-place path:

```c
pthread_mutex_lock(&s->inference_mu);
if (ds4_session_rewind_speculative_boundary(slot->session, pos) == 0) {
    server_log(DS4_LOG_GENERATION,
               "ds4-server: in-place DSpark boundary rewind to %d", pos);
    pthread_mutex_unlock(&s->inference_mu);
    return 0;
}
ds4_session_rewind(slot->session, pos);   /* fallback: rebuild */
```

`pos` here is exactly `block_start + kept (- resample?1:0)`; `base = ds_rewind_pos =
block_start+1`. `kept >= 1` is guaranteed (the seed token is proved non-stop before the
speculative eval, `ds4_server.c:13957-13974`), so `pos >= base`.

### 3.5 Lifecycle

- `ds_rewind_valid = false` in `ds4_session_invalidate`, in `ds4_session_rewind` (so a
  fallback never leaves a restorable flag behind), and on every exit of the fast API.
- Because only `server_generation_rewind` consumes it and it runs immediately after the
  verify in the same request, no cross-request staleness is reachable.
- Success is observable in the generation log:
  `ds4-server: in-place DSpark boundary rewind to N`.

### 3.6 Scope / guards

- Family `DEEPSEEK4` only (runtime check; `DS4_MODEL_FAMILY` is `g_ds4_shape.family`,
  `ds4.c:923`).
- TP off (single rank) for v1; TP falls back to the current rebuild.
- `ds4-agent` callers are intentionally left on the rebuild path (follow-up).

## 4. Impact

- Memory: ~505 KiB logits buffer per DSpark session; metadata ~0.7 KiB. Tensor bodies
  reuse existing `spec_attn_state_*` buffers.
- Speed: replaces an O(context) rebuild (minutes + hang) with <=16 single-token evals.
- Correctness: exact pre-verify frontier + replay through the authoritative target
  kernel; logits restored at the boundary.

## 5. Tests

1. Clean build `ds4-server`, `ds4`, `ds4-agent` (`-Wall -Wextra`).
2. Existing model-backed rewind/spec tests unchanged (no `ds4_session_rewind` change).
3. Manual repro: V4 Flash + `--dspark`, force tool syntax mid-speculation; confirm no
   hang, the `in-place DSpark boundary rewind to N` log line, no giant prefill, and sane
   tool output; cross-check against a `--dspark-strict` run of the same prompt.
4. Regression spot-checks: V4.1 and GLM configs unchanged (never set `ds_rewind_valid`).

## 6. Reflection (adversarial review outcome)

An independent review found:
- **D1 (fixed):** `start` is `block_start+1` (seed already pushed), so at `kept==1`,
  `pos==start` and the replay loop is empty; without saving `s->logits` the next sample
  used the verifier's block-end logits → silently wrong tokens. Now `ds_rewind_logits`
  is saved at snapshot and restored when `pos==base`.
- **D2 (fixed):** `ds4_session_tp_active` does not exist; use `!(engine && engine->tp.active)`.
- **D3/D5 (avoided):** scoping to a dedicated API called only from
  `server_generation_rewind` removes stale-snapshot restore from `REUSE_MEMORY_REWIND`
  and other callers; no epoch needed.
- **D4 (scoped out):** `ds4-agent` rewind still uses the rebuild path; documented
  follow-up.
- **D6 (corrected):** the blocked path is the *opportunistic/argmax* DSpark function
  (`ds4.c:85179-85186`), not stochastic; `DS4_DSPARK_MAX_BLOCK_SIZE` is 16 (not 6);
  batched gate is `ds4_server.c:14001`; `DS4_MODEL_FAMILY` is runtime.
- `enable_frontier_snapshot` is confirmed true for DSpark
  (`need_spec_verifier` at `ds4.c:73159-73162` → `enable_mtp`).
- Residual: if the fallback rebuild is ever reached it can still hang; the fix removes
  the common trigger, not every possible rebuild.

## 7. Rollback

Additive, gated, single-file changes to `ds4.c` + a 3-line hook in `ds4_server.c` +
one declaration in `ds4.h`. Revert by removing the hook (server then rebuilds exactly
as today). Last-good binary backed up in `/Users/xun/bin`.

## 8. Implementation review (round 2)

An independent adversarial review of the final diff found **no wrong-output or crash
path** in the core rewind, and verified: exit-state consistency on all paths (including
mid-replay failure), entry conditions (stale flag, seed batching, resample, TP, batched),
frontier validity, logits position, no Qwen4/GLM/DS41 regression, lifecycle/compile.
Corrections applied after that review:

- **restore-failure**: `spec_frontier_restore` can fail partway; now clears
  `checkpoint_valid` and `ds_rewind_valid` before returning so no caller can believe a
  half-mutated graph is valid.
- **flag hygiene**: `ds4_session_rewind` now clears `ds_rewind_valid`, so a fallback
  cannot leave a restorable flag pointing at stale `spec_*` tensor bodies.
- **guards**: added `!checkpoint_valid` and `distributed` entry guards (defense in depth).
- **simplification**: dropped the per-token capture note in the replay (it was undone by
  the capture invalidation at the end), and corrected the helper comment about seed
  batching.
- Verified: session is `xcalloc`'d so new fields are zeroed; `ds_rewind_logits` is
  allocated exactly for DSPARK+dspark and freed on both failure and session free; the
  `n<16` bound cannot overflow `retained[]`.

Known, intentional limitations:
- If the fast path is unavailable (no frontier snapshot) the caller still falls back to
  the full rebuild, which can still hang; the fix removes the common trigger, not every
  possible one.
- `ds4-agent` callers and DS4.1 remain on the rebuild path (follow-up).
- Under `--mtp-exact-sampling`, the `resample` rewind to `base-1` (kept==1) is rejected
  by the `pos >= base` bound and falls back — correct, just not accelerated.
- The unrelated `openai_finish_reason` mapping (fix 1 from earlier) deliberately turns
  internal generation errors into a valid terminal reason; noted as a separate tradeoff.

Final diff: `+161` lines, zero deletions; `make all` clean under `-Wall -Wextra`;
`ds4-server` installed at `/Users/xun/bin/ds4-server` (sha `e9e6226f…`).
