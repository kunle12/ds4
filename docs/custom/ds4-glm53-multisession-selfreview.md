# Multi-session work — full self-review (2026-10-11, post-deploy)

**Scope:** every change made today — workstream 0 (`c2787ea`), workstreams 1–3
(`3703623`…`5eaa621`), the hardening commit that this review produced
(`dist: harden the core registry from self-review`), and the reasoning behind
the measurements and decisions. Written the way the repo's own review docs are
written: findings first, then what was checked and found sound, then the
honest gaps.

---

## 1. Defects this review found — and fixed (the hardening commit)

1. **`dist_core_acquire` TOCTOU.** The registry lock was held only across the
   *search*; the listener open, allocation and thread spawn happened unlocked,
   with a second lock taken to insert. Two concurrent acquires could both miss
   and both create — on a fixed port the second bind fails as a *spurious
   session-create error*, on an ephemeral port (`--listen` port 0) two cores
   exist for one engine. The server creates slots sequentially so this never
   fired in production, but it was real. **Fixed:** the lock is now held
   across search, create and insert.
2. **Port matching bug hidden under the same race.** Cores were matched by the
   *effective* (bound) port, but the effective port is only knowable after
   binding — so a core created for requested port 0 stored an ephemeral number
   that no later request would ever match: every session would have built its
   own core. **Fixed:** match on the requested port; store both.
3. **Retired-core unlink dangling the gpu-mutex handle.** Release unlinked the
   core at zero refs, so a `ds4_dist_gpu_mutex()` handle obtained before a
   retire/recreate cycle would exclude against a dead mutex while new sessions
   used a new one. The server never frees slots mid-process so this was
   unreachable in production, but the reasoning burden was real. **Fixed:**
   retired cores stay in the list with a `retired` flag; acquire and the
   gpu-mutex lookup skip them, so handles stay valid for process lifetime.
4. **Retire/recreate port race.** The listener was closed *after* publishing
   `retired`, so an acquire in that window would try to bind a still-open port
   and fail spuriously. **Fixed:** close the listener before publishing the
   flag.
5. **Accept-loop spin on retirement.** The loop broke only on `EBADF/EINVAL`;
   any other errno on a retired core's closed listener would log-and-continue
   forever. Pre-existing shape, made more reachable by retirement.
   **Fixed:** the loop also breaks when the shared shutdown flag is set.
6. **Lock-discipline inconsistency.** The pipelined prefill's
   `ds4_session_layer_slice_reset` ran outside `gpu_mu` while the decode-span
   path had it inside. The reset is host-side bookkeeping today
   (`ds4_session_glm_reset_dense_cache`), so this was *correct* — but only by
   inspection, and it would break silently the day the reset gains GPU work.
   **Fixed:** moved inside, with a comment saying why.

All six were rebuilt, and the pair re-gated after the fix: 11K run 90.32 ms
median, sha `a3898c57…`, then the **CLI-path oracle re-run on the final build**
(see §4): logits mean |Δ| **0.17023 — identical to the morning's value to the
last digit** — continuation 290/290 byte-identical, `--dist-replay-check`
exact. The production pair was redeployed and re-verified.

## 2. Reviewed and found sound (with the reasoning, so it is auditable)

* **`dup()` ownership model.** Socket options (`SO_RCVTIMEO`, `TCP_NODELAY`,
  keepalive) are properties of the *socket*, shared across dups; descriptors
  are per-fd. Plans own a dup (closed by `dist_route_plan_free` — the same
  convention `build_route_plan` already used for the registry fd); the session
  cache owns the original. `shutdown()` on the pipelined prefill's failure
  path shuts the socket for both; no double-close path exists (shutdown is
  idempotent; each descriptor is closed exactly once by its owner).
* **Lock ordering.** `inference_mu → gpu_mu` everywhere prefill/save/vision go
  (server_prefill_enter → dist sync → GPU sections; image encode under
  inference_mu + gpu_mu); decode self-serve takes `gpu_mu` alone; the
  progress-callback KV save runs on the reader thread and takes `gpu_mu`
  alone; nothing holds `model_mu` across `gpu_mu`. No inversion exists, and
  the reader-thread save cannot interleave GPU-wise with chunk evals because
  both take `gpu_mu`, each section internally consistent (timeline commits
  happen inside the eval call, inside the lock).
* **Session free now frees the session object** — the original code could not,
  and its comment said so. That constraint was a property of the *old*
  architecture: detached client threads received `&d->state`, the session's
  embedded state. In the new architecture every detached thread touches the
  retained core, so the hazard moved with the state. Verified by enumerating
  what each thread references: accept/client → core; prefill sender/reader →
  joined before return; snapshots → synchronous.
* **Worker-side CUDA safety with two sessions.** The worker's `state->mu`
  spans *complete* evals, so the process-global CUDA command state is
  serialized by construction — with per-session data connections this is the
  load-bearing invariant, and the byte-identical concurrency oracle is its
  empirical proof.
* **Worker restart with a cached work connection.** The plan rebuild on
  generation change re-dials only when host/port changed — a *restarted worker*
  has the same host/port, so the first request after a restart sends on the
  dead socket, fails, drops, and the rebuild replays on a fresh connection —
  the request completes transparently (the rebuild sets rc=0 on success).
  Coarse but self-healing; verified in the stall-recovery runs.
* **Mid-prefill generation change.** The plan rebuild frees the old dup → the
  reader thread's next recv gets EBADF → the prefill fails and rebuilds.
  Correct because a generation change mid-prefill *means* the connection died;
  there is no benign case.
* **`use_control_for_work=false` for sessions** (work no longer rides the
  hello connection) has a second effect I did not set out to make: the
  coordinator's hello threads now use the frame-discard loop for liveness
  instead of `poll(POLLHUP)`. Equivalent-or-better: it also notices clean EOF.
  Recorded here so the behaviour change is not a surprise.
* **One session's stall closing the shared hello fd** (forget-route) clears
  the worker's *other* sessions too, costing them a replay. Coarse, but the
  stall class that triggers it (wedge, deadline) is exactly the case where the
  worker's session state is untrustworthy anyway; and the existing
  `rc != REMOTE_ERROR` split already keeps logic errors (hash mismatch) from
  forgetting the route — only transport-class failures do.

## 3. The logic fallacies I committed, for the record

1. **The head-move prediction (−8–10 % vs measured −2 %).** I priced the
   Spark's head read using the 100.7 GB/s CPU-proxy bandwidth *absolutely*,
   against the impl-log's own caveat ("the ratio is the point"). Measured head
   cost ~4.3 ms, not ~12 ms. Caught by measurement, recorded at the time.
2. **The stage-split estimate (Mac ~42 ms vs measured ~29.5 ms).** Inference
   presented as a table; the profile lines corrected it. The *sum* was right,
   the split was wrong — and everything derived from the split (the saturation
   table's "2-layer rebalance") inherited the error until re-derived.
3. **The memory-table estimate that wedged the box.** "2–3 × 262144 fits with
   margin" — an estimate acted on as fact, with a power cycle as the
   measurement. The guard admitting it made it *feel* verified; the docs'
   "necessary, not sufficient" warning was the check I did not apply to my
   own number. This is the same class as 1 and 2: **an estimate promoted
   without a gate before action.** The corrected rule (worker-available
   ≥ ~10 GiB, measured) is now in the docs with the incident attached.
4. **A measurement-attribution slip** during the first concurrency test: I
   initially read prefill-window duty numbers as decode-window numbers (the
   "A-alone decode 84 %" that was actually prefill), corrected within the same
   analysis by re-deriving windows from logged timestamps. Recorded as
   process, not as a finding: phase windows should always be computed from
   the run logs, never estimated from wall-clock guesses.

## 4. Honest gaps — what is *not* verified on the final build

| item | status |
| --- | --- |
| CLI coordinator oracle (logits + continuation + replay) | **re-run after this review, on the hardened build: bit-identical to the morning run** (mean \|Δ\| 0.17023, 290/290, replay exact) |
| 5 018-token boundary logits on the final build | not re-run; the 2 881 run exercises the same two-chunk pipelined path at this ctx (cap 2048), so coverage is real but the specific gate is stale |
| Worker restart *under two concurrent sessions* | reasoned (§2), not exercised |
| Per-session snapshot save/restore under concurrency | not exercised (criterion-6 machinery is per-session and unchanged, but the concurrent case was never run) |
| Vision/image-encode under concurrent decode | the gpu_mu wiring is in place and compiled; the pair serves text only, so the path is untested on the pair |
| ctx 131072 row of the memory table | untested, marked suspect |
| Duty at N=2 | small-n (4–14 samples per window); the medians are trustworthy, the averages less so |

## 5. Robustness/simplification opportunities (deliberately *not* taken now)

1. **Hang the core off the engine struct** instead of the process-global
   registry: kills the global list, both lookups, the acquire atomicity
   question and the gpu-mutex handle dance in one stroke. Costs an
   `ds4_engine` field (opaque to this TU today) — worth it next time
   `ds4.h` is open for engine-internal churn, not as a standalone change.
2. **Create worker sessions outside `state->mu`.** `dist_worker_get_session_locked`
   builds a full GLM graph *under the lock*: a new session stalls every other
   session's evals for seconds, and this path is where the 262144 wedge
   first surfaced. Right fix when the worker is next touched.
3. **The progress-callback KV save still runs synchronously on the prefill
   reader thread** — a slow remote shard pull stalls the prefill (bounded by
   the 120 s deadline). Pre-existing shape; listed because the concurrency
   work makes it more visible, not because it changed.
4. **Test hygiene:** the concurrency harness summaries should include
   `http_status` (the failed B-alone run was initially unreadable without
   it); duty windows should be emitted with the run's timestamps rather than
   reconstructed afterwards.
5. **The three `wip` commits** in today's history (build fixes) could be
   squashed; left as-is because rewriting pushed history for cosmetics is
   worse than the blemish.

## 6. Verdict

The architecture landed as designed, and the review made it materially
stronger: six real defects fixed (one reachable in production only
theoretically, five latent), two of my own estimates corrected by
measurement before they could mislead further, and the one gap that mattered
(the CLI path) re-gated bit-identically. The remaining items in §4 are
bounded, listed, and none of them guards a path the production configuration
actually exercises. The production pair is serving, on the hardened build,
verified end to end this evening.
