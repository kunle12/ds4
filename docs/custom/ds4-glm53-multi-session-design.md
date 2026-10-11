# GLM 5.3 Flash multi-session pipeline — concurrent clients on Mac + Spark

**Status:** workstreams 0–3 **implemented, gated and deployed 2026-10-11**
(`ds4-glm53-multisession-results.md` has the measurements, including the
corrected memory table and the ctx-262144 incident). The §4.1 head-move
prerequisite is landed (the `21:44` headless worker, in production). What
remains of this design: workstream 4 (shared GLM prefill workspace — now
*necessary* for per-session ctx above 65536, see the incident), workstream 5
(dynamic prefill chunk policy) and workstream 6 (disk-KV eviction).
**Scope:** serving multiple concurrent HTTP clients on the GLM 5.3 Flash Q4_K
Mac+Spark pipeline, with the owner's stated top priority — ~100 % GPU duty on
both machines under load — as the acceptance target (re-derived honestly in
the analysis §7.2 after measurement: the Spark reaches ~100 % at ~14 t/s
aggregate; the Mac's decode ceiling on any prefill-healthy split is ~42 %).
**Companion:** `ds4-glm53-gpu-utilisation-analysis.md` is the analysis this
follows from; its §7 has the saturation arithmetic (re-derived 2026-10-11 from
measured stage times), and its §4.1 (output head on the coordinator) has
**landed**: +1.18 GiB of Spark memory — the enabler for the second session —
plus −2 % decode and −90 % return payload, with the oracle re-passed and
improved. Workstream 0 comes
from `ds4-glm53-response-data-timeout.md`: the code agent's
`Timed out waiting for response data` failures are unbounded result waits on
stalled worker connections, and the fix is standalone-landable before any of
the concurrency work.

---

## 1. Goal and acceptance criteria

| # | Criterion | Measurable gate |
| --- | --- | --- |
| 1 | **Concurrency.** N clients decode simultaneously, each with its own session and KV | **passed 2026-10-11** — 2 interleaved sessions at ctx 65536 produced byte-identical continuations to their alone-runs (`ds4-glm53-multisession-results.md` §2) |
| 2 | **Saturation.** At N ≥ 2 decoding clients: Spark duty rises toward 100 %, Mac duty toward T_mac/T_spark | **measured at N=2** — Spark 64.5 %→76.8 % avg (median sample 96 %), Mac 48.7 %→56.1 % avg; the trend matches the §7.2 arithmetic, and higher N is memory-bound (see the incident, results §3) |
| 3 | **Throughput.** Aggregate decode rises with N | **measured at N=2: 17.8 t/s vs 11.0 single-stream (+62 %)**; per-client gap 91→112 ms |
| 4 | **Single-client regression-free.** One client alone behaves exactly as today | **passed** — decode 91–95 ms and sha `a3898c57…` at every stage; production config re-verified after deploy |
| 5 | **Correctness gates on the live pair** | criterion-1 oracle and boundary gates passed for the headless topology 2026-10-11 (`ds4-glm53-ws0-headmove-results.md` §4); **the CLI-path oracle (logits + continuation + replay) was re-run on the final hardened build and is bit-identical to the morning run** (`ds4-glm53-multisession-selfreview.md` §4); per-session snapshot round-trip under concurrency not yet re-run |
| 6 | **Admission.** Memory over-subscription is refused loudly, not OOM'd | **refuted by measurement**: the guard admitted a second session at ctx 262144 whose creation wedged the GB10 (results §3) — the budget is necessary-not-sufficient; the operating rule is ≥ ~10 GiB worker-available after all sessions, which the guard does not enforce |
| 7 | **Bounded stalls** (workstream 0, independent of 1–6). A worker-side stall that does not close the TCP connection produces a visible failure within the result deadline, not a silent hang | **passed 2026-10-11** — worker `SIGSTOP` mid-decode failed the request at 120.0 s, clean client termination, worker re-dialled after resume, fresh request succeeded; re-verified on the WS1–3 architecture |

Expected per-client latency, stated now to set expectations: at saturation each
client's inter-token gap is ≈ N × T_spark (114 ms at N=2 post-head-move, 171 ms
at N=3) — compare today's multi-client reality, which is N × full round trip
(≈ N × 106 ms) *plus* session re-sync replays between prompts. Concurrency is
strictly better for every client than today's queueing; it is not better than
one client alone on a quiet pair, and cannot be.

---

## 2. What exists today (all verified in source)

The asymmetry this design exploits: **the worker is already concurrency-ready;
the coordinator is single-session by construction.**

* **Worker: ready as-is.** Sessions are created on demand, keyed by session_id
  (`dist_worker_get_session_locked`, `ds4_distributed.c:6798-6821`) — no
  pre-registration handshake. The data listener accepts any number of
  connections, one thread each (`:7892-7932`), and each connection runs a read
  thread plus a prefetch queue (depth 2, `:724-736`) decoupled from an eval
  thread (`:7757-7865`). All evals — decode or prefill, any session — serialise
  behind `state->mu` (`:7455-7520`), which is correct for one GPU. Wire payload
  decode happens *before* the lock (`:7429-7453`), so input conversion of the
  next request overlaps the current eval.
* **Worker scheduling seam.** Because each connection owns its queue and eval
  thread, a decode token arriving on session B's connection waits only for the
  eval currently holding the lock — not behind session A's queued prefill
  chunks. A shared-FIFO design would not have this property.
* **Coordinator: one session, one of everything.** Each `ds4_dist_session`
  opens its own listener on `--listen` and spawns its own accept thread and
  worker registry (`ds4_dist_session_create`, `:5402-5481`). The server creates
  one session per slot (`ds4_server.c:16014`), so `--batched-session N > 1` on a
  coordinator fails at the second bind (`dist_open_listener`, `:1193-1234`,
  `SO_REUSEADDR` without `SO_REUSEPORT`) — **multi-session is not merely
  unbatched today, it cannot start.**
* **Wire: one request in flight per connection.** Result frames are matched
  strictly against the sent request id and a mismatch is fatal
  (`:2390-2394`), which is what forces the blocking send→recv per token and
  forbids two coordinator threads sharing one fd. The frames already carry
  session ids everywhere.
* **Outbound coordinator→worker connections are established practice.** Every
  KV snapshot save/load dials the worker's advertised data port per operation
  (`dist_connect_endpoint` at `:4938`, `:5055`; the hello carries the data
  port, `:1881`). The worker's data read loop handles WORK frames on those
  connections identically to the control connection.
* **Server: batching machinery that excludes this mode.** `decode_worker_main`
  coalesces pending decodes (2 ms window) and calls `ds4_sessions_eval_batch`
  (`ds4_server.c:13307-13409`), but the native batch predicates refuse
  distributed sessions (`s->distributed`, `ds4.c:79151`; the GLM 5.3 native
  session batch also requires the full graph — `has_token_embd &&
  has_output_head`, `ds4.c:77663`), so it falls to the serialised loop
  (`ds4.c:80263-80275`). The global `inference_mu` serialises whole round
  trips (`ds4_server.c:13223`).
* **GPU submission is single-threaded state.** `ds4_gpu_begin_commands` uses a
  process-global current command buffer (`g_batch_cb`, `ds4_metal.m:9512-9528`)
  — concurrent local evals from two threads would race. Any concurrency design
  must keep exactly one thread (or one mutex) owning GPU command submission.

---

## 3. Design

Three changes, ordered so each is independently landable and the pair's
behaviour with one client is byte-identical until the last one.

### 3.1 Shared coordinator core (fixes the port collision)

Hoist the listener, accept thread, worker registry and generation counter from
per-`ds4_dist_session` state to one engine-level coordinator core. Sessions
retain: `session_id`, `request_id` counter, plan cache keyed by generation
(the existing `plan_generation` logic, `:4332-4348`), and their connections.
A worker re-registration bumps the shared generation and invalidates every
session's plan cache at once — which is exactly the semantics the per-session
copies were approximating.

### 3.2 Per-session work connections (fixes request matching)

Each session sends its WORK frames over its own outbound connection to the
worker's data listener, dialled lazily like the snapshot connections and held
open across tokens. The strict per-fd request matching stays exactly as is —
it becomes per-session, which is what makes concurrency possible without a
multiplexer: a session's blocking recv waits only for its own token, and the
worker's per-connection queues provide the decode-over-prefill scheduling seam
of §2. Prefill pipelining (sender + reader threads over the same fd,
`:3513-3777`) is unchanged per session. Snapshot connections are unchanged.
No frame, no version handshake, no worker change.

### 3.3 Shrunken critical section (fixes the serialization)

Today `inference_mu` covers eval + send + wait + recv + sample. The wire wait
(~65 % of the token period on the Mac) holds the lock for nothing. Replace it,
for distributed sessions, with a mutex that covers **every GPU-issuing path**
and nothing else:

* the local slice eval in `dist_coordinator_eval_span` (`:2733`),
* the per-chunk local evals in the pipelined prefill loop (`:3669`),
* the head-from-hc eval (`:2642`),
* KV-shard snapshot reads/writes against the local graph (today these
  serialise against decode via `inference_mu` too — the disk-KV paths at
  `ds4_server.c:11289-11308`, `:11421`, `:11457-11469` — so the narrowed lock
  must keep them excluded from concurrent command submission),
* image-encoder execution (same reasoning, `ds4_server.c:10203-10210`).

It is released during all wire waits and host-side work. Slot threads then run
their own full round trips concurrently; the Mac's GPU time-slices their local
evals (its natural cap, T_mac per token), the Spark's GPU stays fed (its cap,
T_spark per token), and the saturation arithmetic of the analysis §7.1 applies
directly. Think of it as *narrowing* `inference_mu` to the GPU critical
sections, not removing it.

The engine keeps its single-threaded GPU discipline: one mutex around all
GPU-issuing work is the whole change — `g_batch_cb` is only ever touched by the
mutex holder. Non-distributed engines keep today's `inference_mu` behaviour
exactly; the server's batched `decode_worker_main` branch for distributed
sessions (the serialised fallback) is no longer taken.

**Prefill-under-decode policy (required, not optional).** A 4096-token
prefill chunk occupies the Spark's eval lock for ~6–7 s (chunk period ≈ 4096 /
455 t/s × 0.74 duty), so a concurrent client's decode token can spike from
~114 ms to seconds while another client ingests. First version: run
multi-client configs with `--dist-prefill-chunk 512` (or 256), bounding the
spike at ~0.4–0.8 s at a modest prefill cost — the knob already exists and
flows through `dist_coordinator_prefill_chunk_cap` (`:3441`). A dynamic
policy (shrink chunk while any other session has decode pending) is a later
refinement behind the same knob; the 2048 sub-chunk granularity the engine
already uses at long context shows the chunk size is not load-bearing for
correctness.

### 3.4 Memory and admission

The honest constraint. Per-session marginal on each side is KV plus per-session
graph scratch; from the recorded plans, the Mac's `0:20` plan at ctx 524288
(82.22 GiB) decomposes to ≈ 77.0 weights + 2.66 KV + ~2.5 scratch, so a second
session costs ~5 GiB on the Mac (33 GiB slack — fine) and ~6 GiB on the Spark
(**2.88 GiB slack today, ~4.06 after the head move — not enough**). Practical
multi-session configurations are therefore:

| sessions × ctx | Spark marginal (est.) | fits after head move? |
| --- | ---: | --- |
| 2 × 524288 | ~6 GiB | no, unless reserve ≤ 12 — not recommended |
| 2 × 262144 | ~4.2 GiB | **no — measured 2026-10-11: wedges the GB10** (results §3) |
| 2 × 131072 | ~3 GiB | untested; suspect — do not assume |
| **2 × 65536 (+ chunk 2048)** | ~2 GiB | **yes — measured: works, ≥ 10 GiB worker-available throughout** |

> **Correction 2026-10-11.** The original table here estimated "2–3 × 262144
> fits with margin" — that estimate is withdrawn. The first live attempt at
> 2 × 262144 froze the box inside the NVIDIA driver under memory pressure
> (power cycle required) even though the guard admitted it, proving again that
> the guard budget and the idle `free -g` figure are necessary, not
> sufficient. The operative rule is now: keep the worker's *available* memory
> ≥ ~10 GiB after all sessions exist, and size ctx from measurement only.
> `ds4-glm53-multisession-results.md` §3 has the incident and the evidence.

These are estimates; the guard prints actuals (`required=… budget=…` and the
per-session graph accounting) and criterion 6's admission behaviour is "refuse
the whole plan", so the server must treat a refused session-create as "resident
set full" — queue the request (first version) rather than crash. Two memory
levers belong to this workstream because they decide how many clients fit:

* **Shared GLM prefill workspace across sessions** (precedent:
  `share_session_prefill_workspace` for the ds4 graph, `ds4.c:73210-73235`):
  the chunk-cap-sized prefill buffers are the bulk of the scratch and only one
  session prefills at a time under §3.3's mutex; sharing them cuts the
  per-session marginal toward KV-only. **Promoted from optional to necessary**
  by the 262144 incident — it is the only route to per-session ctx above
  65536.
* **Disk-KV eviction for idle sessions** (criterion 6 already proves the
  save/restore across the pair): more logical clients than resident sessions,
  at the cost of a replay on wake — a natural fit for the server's existing
  kv-disk policy machinery.

---

### 3.5 Single-session impact: none, by construction and by gate

The owner's main scenario stays one interactive client, so this is a design
requirement, not a hope. What the three changes do to a single-session run:

| change | single-session delta | expected cost |
| --- | --- | --- |
| shared coordinator core (3.1) | one listener/accept thread/registry instead of one per session — with one session, identical | none |
| per-session work connection (3.2) | WORK frames move to a kept-alive outbound data connection, dialled once lazily; frames, ordering, prefetch depth (2) and eval mechanics are byte-for-byte the same class as today's control-connection path | one ~1 ms dial at session start; one extra idle fd and thread pair on the worker |
| narrowed GPU mutex (3.3) | the lock is uncontended with one session either way; the wire wait is simply never a section anyone contends for | none measurable |

There is no batching tax by design: distributed sessions bypass
`decode_worker_main`'s coalescing entirely (§3.3), so no 2 ms batch-formation
window is inserted into the single-client token loop. Numerics are untouched
(same graphs, same frames, same per-session order), and no env switch or
permanent variant is introduced — the repo's rule against semantic variants
behind flags applies, so the gate is measurement, not configuration:

* **criterion 4 is a hard gate after every phase:** prefill t/s and decode
  median gap within noise of the recorded baseline, continuation sha256
  byte-identical at ~11 K / ~285 K / ~473 K (the E1-harness method).
* the existing release gates keep running alongside it — in particular the
  single-host `--ssd-streaming` generation gate added to
  `QA_BEFORE_RELEASES.md` after the `ce4d214` regression, since the shared
  coordinator core touches `ds4.c` session creation.

If any phase fails its gate, it does not land — the phases are ordered so
1–2 are pure structure with zero behavioural delta, while 0 and 3 are the two
phases that change runtime behaviour, each carrying its own gate (criterion 7
for 0, criterion 4 for 3).

---

## 4. Workstreams

| # | Workstream | Files | Est. | Notes |
| --- | --- | --- | ---: | --- |
| 0 | Bounded result wait + dist socket keepalive tuning | `ds4_distributed.c` | 0.5–1 d | **done** (`c2787ea`), stall-verified twice, including on the WS1–3 architecture |
| 1 | Shared coordinator core | `ds4_distributed.c`, `ds4.h` | 1–2 d | **done 2026-10-11** — registry/listener/accept hoisted to a refcounted per-engine core; single-session byte-identical |
| 2 | Per-session work connections | `ds4_distributed.c` | 1–2 d | **done 2026-10-11** — sessions dial the worker's data listener; plans own dups; failure drops and re-dials; strict matching is per-session |
| 3 | Shrunken local-eval mutex + server concurrency | `ds4_distributed.c`, `ds4_server.c` | 2–3 d | **done 2026-10-11** — shared `gpu_mu` around the GPU sections; dist slots self-serve decode; image encoder joins the exclusion; one-shot sync in batched mode (the 269→430 t/s quantum fix); +62 % aggregate at 2 clients, outputs byte-identical |
| 4 | Shared GLM prefill workspace | `ds4.c` | 1–2 d | **done 2026-10-11** — 50-tensor shared set, engine-owned, offset-table aliasing; a second session adds KV only (~2 GiB at 262144 vs ~4.2 before); 2×262144 gated byte-identical with ≥ 10 GiB available; the memory table above is superseded by the results doc §5 |
| 5 | Dynamic prefill chunk policy | `ds4_distributed.c` | 1 d | later; behind the existing knob |
| 6 | Disk-KV eviction for pipeline sessions | `ds4_server.c` | 2 d | later; reuses criterion-6 transport |

Order 0 is independent of the concurrency work and can land alone; 1 → 2 → 3
is the minimum for criteria 1–3. Each step keeps the previous
behaviour observable behind it, per the repo's rule that the Metal default,
SSD streaming, CUDA and distributed paths must not be affected by fixes to
other parts — verified here by the criterion-4 single-client A/B after each
step.

---

## 5. Risks

| Risk | Mitigation |
| --- | --- |
| GPU command races (`g_batch_cb` is process-global) | the single local-eval mutex is the design, not an optimisation; two-session stress + criterion-1 oracle gate every phase |
| Decode starvation during another client's ingest | §3.3's chunk policy; first version is an operator knob, spike bounded ~0.4–0.8 s |
| Spark memory over-admission (its budget already needed `reserve 14`) | guard admission refuses Nth session; server queues; head move adds 1.18 GiB; real numbers read from the guard lines in §1 gates |
| Duty "regression" optics: one quiet client still reads ~40 %/65 % | documented in the analysis §7.2 — saturation is a load property, not a config property |
| Worker loss mid-flight across N sessions | existing per-session rebuild machinery (`dist_coordinator_rebuild_from_transcript`) replays per session; a prefill failure still costs only that session's transcript (review R1 note stands) |
| Session-cache/prefix semantics across slots | unchanged: each client keeps its own session; no re-sync replays between alternating clients (a second, independent win) |

---

## 6. Non-goals

* No wire protocol change; no version bump; no worker-side behavioural change
  (workstream 0 tunes shared socket options compiled into both binaries — the
  worker's eval and queue logic is untouched).
* No true batched decode (analysis §4.3) — it multiplies per-pass throughput but
  does not by itself raise duties, since the stages still alternate within a
  pass; revisit after saturation is measured.
* No split change (the `0:22` rebalance is evaluated separately per the analysis
  §7.4 step 3, once ≥ 2 clients exist to measure with).
* No speculative decoding under the split (stays excluded, `ds4.c:62256`).

---

## 7. Measurements to run when the machines free up

1. Everything in the analysis doc §5 (stage budget, head-move A/B, duty
   alignment) — items 1–3 are prerequisites for trusting this document's
   projections.
2. Per-session marginal bytes as printed by the guard at ctx 262144 and 131072
   on both hosts — replaces the §3.4 estimates with actuals.
3. `serve_concurrency_bench.py --concurrency 2..4` against the interleaved
   build, reading: aggregate t/s (expect ≈ 1/T_spark), per-client ITL
   (expect ≈ N × T_spark), and both meters during steady state (expect
   Spark ~100 %, Mac ≈ T_mac/T_spark).
4. The concurrency oracle of criterion 1: two interleaved sessions, each
   byte-identical to its alone-run.
