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

---

## 7. Workstream 4 self-review (2026-10-12, post-deploy)

**Scope:** the shared GLM prefill workspace (`ds4.c`, `eca36db`…`e1df92c`),
reviewed with the same discipline as §1–§6.

### Defects found by this review — fixed (`e1df92c`) and re-gated

1. **The publish log printed the wrong number — and I read past it twice.**
   The "shared GLM-5.3 prefill workspace: … %.2f GiB once" line printed the
   *alias discount* (zero for the publishing session): production logs said
   **"0.00 GiB once"** during the WS4 test *and* in my own final
   verification grep. A size of 0.00 GiB is impossible and should have
   stopped me both times. Fixed to print the shape's actual workspace bytes;
   verified in production ("**3.02 GiB once**").
2. **The memory guard under-charged at an aliasing session's admission.**
   The discount removed what the session no longer allocates, but the live
   engine-owned workspace itself was not charged anywhere in `required` —
   an under-charge of the workspace size (~3 GiB) at every post-publish
   admission. The same fallacy class as the 262144 incident: I derived the
   discount algebra once during design, then applied it without re-deriving
   the full *required* identity after the edit. Fixed: the transient passed
   to the slice guard now includes the live workspace when one is
   published. Arithmetic for the 2×262144 admission: required ≈ 103.6 of
   107.61 — still admitted, now honestly.

Because that fix changed the admission formula *after* the 2×262144 gate had
passed, the gate was **re-run on the fixed binaries** (the repo's own rule: a
physical run is the gate): A alone 90.1 ms / B alone 91.3 ms (the critical
admission) / concurrent 112.6 ms each — all byte-identical, +62 % aggregate,
no refusal. The 285K production-depth run was also added on the final
binaries (the earlier WS4 verification was 11K-only): **406.7 t/s prefill,
104.5 ms decode median, sha `a3898c57…`** — criterion 4 fully closed.

### Reviewed and found sound

* **The sharing invariant, re-derived end to end.** All 731 `batch_*`
  references were bucketed by enclosing function: every use sits inside an
  eval call (the 29-slot `ws_mirror` set that `glm_graph_ws_free` touches
  first in `glm_graph_free` is the *decode* scratch, none of it shared).
  The only token-graph touch is the single-row embedding staging write into
  `prefill_tokens[0]`, inside the eval lock. MTP verify, native session
  batch, imatrix and seed paths are all eval-internal or shape-ineligible.
* **Alias-before-validate ordering.** The alias block's `batch_cap` /
  `generic_routed_moe` writes are redundant (validate and the batch-rows
  computation overwrite them with identical values); `glm53_prefill_cap`
  survives because `glm53_graph_prefill_workspace_ensure` no-ops at
  matching caps and now *refuses* to grow a borrowed workspace.
* **Ownership lifecycle.** Publish marks the *owner* graph borrowed, so its
  free skips the shared tensors; engine close owns them; the guarded
  indexed block and the generic-routed allocations align with the owner's
  because the shape key includes the compact caps that size them.
* **Accepted approximations, now on record:** the discount over-counts by
  ≤ ~10 MB (the estimate bundles single-token scratch and the seed buffer
  into `glm_graph_workspace_bytes_for_cap`); the owner-freed window leaves
  the workspace bytes out of `glm_session_graph_bytes` while the tensors
  live (the new live-charge covers the *admission* side; slots live for
  process lifetime, so the counter window is theoretical); and the
  workspace never migrates shapes — if an engine's first shape dies out,
  its ~3 GiB stays until engine close (the Qwen4 arena's refcount-free
  rebuild is the template, worth copying only if shape-churning engines
  ever exist).

### Process notes

* The **deploy-vs-git friction recurred** on spike (rsync stomp → dirty
  tree → pull refused; resolved by checkout+pull). Second occurrence; the
  standing fix is to make `ds4-deploy.sh` git-pull on the peer instead of
  rsyncing over its checkout — a host-script change, offered, not made.
* The two defects above were both **visible in output I produced during the
  work** (the 0.00 log line) or **derivable from the design algebra** (the
  un-charged workspace). Neither required new machinery to find — only
  reading my own output skeptically and re-deriving my own formulas. That
  is the same lesson as §3, and it is now written down twice because it
  was committed twice in one day.
