# GLM 5.3 Flash on Mac + Spark — GPU utilisation analysis

**Status:** analysis only. No experiment was run: both machines are busy, so every
code path below was read in source at `24444e5` and every number is either the
project's own recorded measurement or an estimate marked as such. §5 lists the
runs that would confirm the estimates once the pair is free. §6 records the
owner's decisions (concurrent clients wanted; ~100 % utilisation is the top
priority), and §7 derives what that implies: the ceiling for each configuration,
and the recommended sequence. The follow-on design is
`ds4-glm53-multi-session-design.md`.
**Scope:** why the two GPUs of the GLM 5.3 Flash Q4_K pipeline (Mac Studio M4 Max
coordinator `0:20`, DGX Spark GB10 worker `21:output`) read ~40 % (Mac) and ~65 %
(Spark) most of the time, occasionally 100 % on both; and what, at code level,
could raise utilisation and throughput.
**Owner observation addressed:** "most of the time GPUs only go to 40 % (Mac) to
65 % (Spark); sometimes both 100 %."

The one-line answer: **the pair is in one of three regimes at any moment —
pipelined prefill (Mac-bound: ~100 %/74 %), strictly serial decode (alternating:
~40 %/65 %, sum ≈ 100 %), or idle (0 %/0 %) — and a duty-cycle meter blended
across those regimes is what produces the observation.** Neither decode figure is
a defect: decode is a serial pipeline at batch 1, so both GPUs cannot be busy at
once. The real levers are listed in §4; the cheapest measured-bounded one moves
the output head to the coordinator (§4.1, configuration-only candidate), and the
big ones are cross-session interleaving and true batched decode (§4.2/§4.3).

---

## 1. What the pair actually executes, per phase

The engine is a *layer-range pipeline*: the coordinator holds layers 0:20 plus
the token embedding (and, notably, also the output head — see §4.1), the worker
holds 21:44 plus the output head. Per request the server drives three shapes of
work, and they have different parallel structure.

### 1.1 Decode: one token, strictly serial

A generated token walks the whole route before the next one can start. The call
chain, with anchors:

```
server_eval_token (ds4_server.c:13283)
  └─ ds4_session_eval → ds4_session_eval_internal (ds4.c:77396)
       └─ ds4_dist_session_eval (ds4_distributed.c:5650)          [1 token]
            └─ dist_coordinator_eval_span (ds4_distributed.c:2670)
                 ├─ 1. local: ds4_session_eval_layer_slice        (Mac GPU busy)
                 │     ds4_distributed.c:2733, layers 0:20
                 ├─ 2. send 64 KiB hidden to worker               (Mac GPU idle)
                 │     ds4_distributed.c:2575 (blocking)
                 ├─ 3. worker: eval 21:44 + head, return logits   (Spark GPU busy,
                 │     ds4_distributed.c:7501 under state->mu      Mac GPU idle)
                 ├─ 4. recv ~605 KiB logits + memcpy              (both idle)
                 │     ds4_distributed.c:2597, 2628
                 └─ 5. sample + detokenise + HTTP on the Mac CPU  (both idle)
```

Three structural facts fall out of the code:

* **The two stages cannot overlap for a single token.** Step 3 needs step 2's
  payload; step 1 of token *N+1* needs the sample that only exists after step 5
  of token *N*. This is why decode cost is `sum(stage)` (impl-log §2.3) and why
  both GPUs can never be near 100 % in this phase.
* **One request in flight per connection.** `dist_recv_result_alloc` fails on a
  request-id mismatch (`ds4_distributed.c:2390`), so the coordinator blocks
  send→recv per token; there is no pipelining of decode requests, and two
  coordinator threads cannot share the worker fd.
* **Everything engine-side is serialised across sessions anyway.** The server
  holds a global `inference_mu` around every eval (`ds4_server.c:13223`), and the
  batched path's `ds4_sessions_eval_batch` falls back to a serialised loop for
  distributed sessions because the native batch predicates refuse them
  (`s->distributed`, `ds4.c:79151`; the GLM 5.3 native session batch additionally
  requires a full graph — `has_token_embd && has_output_head`, `ds4.c:77663` —
  which no slice session has). The worker itself is ready for interleaving —
  its data listener accepts multiple connections, one thread each
  (`ds4_distributed.c:7892`), with a receive thread + prefetch queue decoupled
  from the eval thread (`:7757`), and evals serialised only by `state->mu`
  (`:7455`) — but nothing on the coordinator ever sends it concurrent work.

Per token at the wire: 64 KiB out (`N_HC×N_EMBD×4`), `154,880×4 ≈ 605 KiB`
logits back (oracle doc: vocab 154,880; logits always 32-bit,
`ds4_distributed.c:7581`). The docs measured this as ~2 ms of a ~121 ms token
(impl-log §5) — the link is not the constraint.

### 1.2 Prefill: chunk-pipelined, coordinator-bound

Prefill is the phase with *overlap by design* (`dist_coordinator_prefill_prompt_pipelined`,
`ds4_distributed.c:3513`): chunks of `--dist-prefill-chunk` (4096 by default at
this ctx) flow through a slot pipeline — a sender thread transmits chunk *k*
while the Mac computes chunk *k+1* (`:3614`), a reader thread consumes results
(`:3343`), and non-final chunks are ack-only so the worker does not send hidden
states back mid-prefill (`:3692`). The flow window defaults to
`remote_stages + 2` = 3 chunks in flight (`:3476-3504`); the worker's receive
queue holds up to 2 further chunks (`dist_worker_prefetch_depth`,
`:724-736`).

Its cost is therefore `max(stage_Mac, stage_Spark)` (impl-log §2.3), and the
measured regime at the recommended `0:20` split is **Mac-bound**: Mac at
99–100 %, Spark at ~74 % duty — the Spark finishes a chunk faster than the Mac
produces one and blocks on its socket read. The project's own words (impl-log
§5): *"the Spark idle time is blocked rather than spare"*. The measured duty
at the older `0:23` split was ~57 % — more Mac layers, longer Spark waits.

### 1.3 Idle and host phases

Between requests, on session-cache hits (a repeated 39 846-token prompt returns
in 0.191 s — no GPU work), during KV snapshot save/load, and during the
coordinator's CPU work (sampling, detokenisation, HTTP streaming), both GPUs
read 0 %. Long sessions spend most of their wall clock here if the client is
not streaming.

---

## 2. Why the meters read 40 % / 65 % / 100 %

### 2.1 What the meters measure

Neither tool reports FLOP efficiency. `nvidia-smi utilization.gpu` (the Spark's
meter here, and the tool behind the project's own duty figures) is *the percent
of the sample period during which at least one kernel was resident*; `macmon`'s
GPU channel on Apple Silicon is the IOReport residency class — the same kind of
duty measure. Both sample on ~1 s intervals, so both report phase blends. A decode kernel that is 100 % resident
but memory-stalled still counts as 100 % while it runs. So each machine's meter
reads: **high while that machine holds the token, zero while it waits** — and a
sampling interval longer than a token period returns the *average residency* of
the phase it happened to cover.

### 2.2 The three regimes, and the expected meter reading

| phase | Mac GPU | Spark GPU | binding constraint | code |
| --- | ---: | ---: | --- | --- |
| prefill (ingest, replay, restore) | 99–100 % | ~74 % duty | Mac stage (`max`) | §1.2 |
| decode, batch 1 | **~40 %** | **~65 %** | neither — serial `sum(stage)`, weight traffic | §1.1 |
| idle / cache-hit / CPU work | 0 % | 0 % | none | §1.3 |

"Sometimes both 100 %" is regime 1: during a long prefill the Mac is pegged and
the Spark bursts at ~96 % for most samples (the project's own sampling saw
"three of four samples at 96 %, one at 7 %", impl-log §3.3), so a coarse
monitor frequently catches both near 100 % — especially during the 11.5-minute
ingests and any forced transcript replays, which are all prefill.

### 2.3 Reconciling 40/65 with the recorded 71/62

The recorded decode duties (~71 % Mac, ~62 % Spark, impl-log §5) predate the E1
decode-graph change (`1a8d45f`, A/B in `ds4-glm53-e1-decode-graph-ab.md`), which
defaulted single-token slice steps to the lean decode graph
(`ds4.c:74556-74595`) and made decode **+15.4 % faster at ~285 K** with
identical output. The time E1 removed was GPU-resident-but-unproductive work:
the indexed batch graph cost ~1.4 ms/layer where the decode graph does the same
math in ~0.79 ms (the comment at `ds4.c:51784-51790` records the measurement).
Removing it lowered *busy time* while raising t/s — so the post-E1 duty-cycle
figures sit **below** the recorded ones, and the owner's 40 % is consistent
with the current build. Two things follow:

* duty and speed moved in **opposite** directions across that change — a
  caution against treating utilisation as the objective (it recurs in §4.1,
  where the recommended change makes the Spark's *duty* drop while decode gets
  faster);
* the recorded 71 %+62 % = 133 % could never have been a same-window pair
  reading for a serial pipeline — at least one of those figures was taken over
  a window that mixed regimes. Same-window decode duties must sum to ≤ 100 %.

### 2.4 The per-token budget the observation implies (estimates)

Taking the post-E1 decode period at ~285 K (~106–107 ms median gap) and the
owner's duties at face value:

| stage | estimate | basis |
| --- | ---: | --- |
| Mac stage (21 layers + embed + readback) | ~42 ms (~40 %) | 40 % × 106 ms |
| Spark stage (24 layers + head) | ~69 ms (~65 %) | 65 % × 106 ms |
| wire (64 KiB out, 605 KiB back) + copy/sample | ~3–5 ms | docs' ~2 ms + logits memcpy |
| **sum** | **~106–112 ms** | measured 106–107 ms |

The sum closing on the measured period is the check: the pipeline has almost no
dead time beyond the structural serialisation — E1 already squeezed the fat
out. The 42/69 split is also consistent with the machines' measured bandwidths
(295.1 vs 100.7 GB/s, a 2.9× ratio) and with the per-layer decode asymmetry the
split sweep measured (−1.30 ms per layer moved Mac-ward, impl-log §4.3). These
stage times are **inferred, not measured**; `DS4_DIST_DECODE_PROFILE=1`
(`ds4_distributed.c:752`) prints exactly this breakdown per token and per span,
on both sides — §5.

---

## 3. Can both GPUs be 100 %?

**At decode, no — not for a single request, ever.** The stages alternate, so
`duty_Mac + duty_Spark ≤ ~100 %` minus wire and host time. Today's ~40 %/65 %
is that bound essentially met (§2.4). Raising *both* numbers requires more
tokens per weight-pass — i.e. batching (§4.2/§4.3) — not better kernels.

**At prefill, only with a stage-balanced split — which memory forbids here.**
Prefill is `max(stage)`; the Mac is the binding stage, so the only way to load
the Spark's remaining 26 % is to move layers Spark-ward, and the Spark is at
104.73 of 107.61 GiB at `0:20` (impl-log §3.2). The guard reserve is already
trimmed to 14 GiB to admit this split. So the Spark's prefill idle is
unharvestable for this model on this hardware; the levers for prefill *speed*
are Mac-side kernel work or a faster coordinator — the chunk/window knobs were
already A/B'd as no-ops (impl-log §3.1).

**What 100 % is worth:** nothing by itself. The metrics that matter are
TTFT/ingest time and inter-token latency, plus aggregate t/s if more than one
client is served. §4 is ordered by gain against those metrics.

---

## 4. Levers, ranked

### 4.1 Move the output head to the coordinator — configuration-only candidate, ≈ −8–10 % decode

The single biggest asymmetry inside the decode `sum(stage)` is not a layer —
it is the **output head**: 1.182 GiB of weights read once per token, on the
*slower-memory* machine. Estimated per token:

| | on Spark (today) | on Mac | delta |
| --- | ---: | ---: | ---: |
| head weight read (1.182 GiB) | ~12.0 ms (100.7 GB/s) | ~4.1 ms (295.1 GB/s) | **−7.9 ms** |
| wire back to Mac | 605 KiB logits | 64 KiB hidden | **−1.2 ms** |

Net ≈ **−9 ms on a ~106 ms token ≈ −8.5 % decode at 285 K** (−10 % at 11 K),
prefill unchanged up to ~0.6 s per long ingest (the final chunk returns its
hidden state instead of logits, ~256 MiB, plus a one-row head matmul on the
Mac — `ds4_session_eval_output_head_from_hc` evaluates only the last row,
`ds4.c:73616`). The Spark's plan drops 1.18 GiB — which is also the difference
between "a second deep session does not fit" and "it might" (§4.2).

The topology is **already implemented and route-validated** — this is the
surprise the code inspection turned up:

* the coordinator binds the head *for exactly this case*: "A coordinator may
  need to apply the output head locally when the last worker advertises N:M
  rather than N:output" (`ds4.c:70648-70652`); the Mac's own map line reads
  `0:20+output` (impl-log §2.1), so the head is resident on the Mac **today**,
  double-mapped;
* route validation, plan completeness and the logits-flag derivation all accept
  a headless final worker with `local_can_output_head`
  (`ds4_distributed.c:1948-1950`, `:2011-2069`, `:2270`);
* both prefill (`allow_hidden`, `:3581`; final head eval `:3762-3773`) and decode
  (`DS4_DIST_RESULT_HIDDEN_STATE` → head-from-hc, `:2640-2658`) paths exist;
* the pipelined-prefill precondition holds for it (`:3433-3437`).

So the experiment is a launch-config change: worker `--layers 21:44` instead of
`21:output`. Costs and caveats:

* **Numerics:** the head matmul moves CUDA→Metal; the cross-backend tolerance
  gates and criterion 1 oracle must be re-run for this topology (the repo's own
  convention: a physical two-machine run is the gate). The head is a plain Q8_0
  matmul, so drift should be small — but that is what the oracle is for.
* **Duty will look *worse* on the Spark** (~58 %) while decode is faster —
  expect that, and do not let the meter re-litigate the decision.
* Prefill at very short prompts crosses one extra 64 KiB payload in the other
  direction — negligible.
* `--dist-activation-bits 16` would then halve *both* wire directions (hidden
  out and back); measured "no gain" on the pair (impl-log §3.1) and it changes
  numerics — leave off unless profiling says otherwise.

### 4.2 Cross-session interleaved decode — the utilisation lever if >1 client is served

Today the pair serves one request at a time, and a second concurrent session is
not merely unbatched — it is not possible to create: each server slot's
`ds4_session_create` would spawn its own coordinator listener on the same
`--listen` port (`ds4_distributed.c:5422`, one `ds4_dist_session` per slot at
`ds4_server.c:16014`), and the second bind fails. The recommended pipeline
server therefore runs exactly one resident session.

If concurrent clients matter, the moderate change is: keep one shared
coordinator state (listener, worker registry, route plan) at engine level and
let N sessions issue their per-token round trips *concurrently*, over per-
session data connections to the worker's existing multi-connection listener
(`:7892`) — or over the shared fd behind a request-id dispatcher (the frames
already carry request ids; `:2390` is the only strict-matching blocker). Relax
the server's `inference_mu` for distributed sessions (`ds4_server.c:13223`).
The worker needs no change: its sessions are keyed by id, its evals serialise
on `state->mu` — which is correct, since its GPU should run one eval at a time.

Expected effect with 2 concurrent decoding sessions (stage times from §2.4,
after §4.1: Mac ~46 ms, Spark ~57 ms): while session A waits on the Spark,
session B's Mac stage runs in that window. Aggregate decode becomes bound by
the longer stage, ≈ 1/57 ms ≈ **17.5 t/s vs 9.4 t/s single-stream** (+85 %
throughput at 2 sessions; the Spark's duty → ~100 %, the Mac's → ~80 %);
per-request latency is roughly unchanged at 2 sessions and grows with queue
depth beyond. **Memory is the honest constraint:** per-session KV at ctx 524288
is ~2.66 GiB on a 21-layer slice (impl-log §3.1), ~3 GiB on the worker slice,
and the Spark's slack is 2.88 GiB today — §4.1's 1.18 GiB plus a reserve trim
to 13 admits a *second* deep session, not more. Many sessions need smaller
per-session ctx (KV scales with it).

### 4.3 True batched decode across sessions — the big win, bigger change

Interleaving still pays every weight read per token. Batching B sessions' tokens
into one graph pass reads each stage's weights once for B tokens — the standard
serving-engine lever, and the reason decode is weight-bound at batch 1
(impl-log §5). Rough shape at B=4 (post-§4.1): Spark pass ≈ its ~4.8 GiB active
weights / 100.7 GB/s ≈ ~50 ms → ~4 tokens/50 ms ≈ **80 t/s aggregate** vs 9.4
today; both GPUs ~100 % duty during passes.

What exists already, as the template: the GLM 5.3 full-graph session-batch
encoders (`glm53_graph_encode_native_session_batch`, `ds4.c:78075`, with
per-row views and a shared prefill workspace) — currently gated to full-graph
sessions (`ds4.c:77653-77685`) — and the two-Spark TP mode's batched decode
(`--batched-session 8`, `decode_worker_main` at `ds4_server.c:13307`; the TP
wire carries batch rows, `ds4_tp.c`). What is missing for the pipeline: a
multi-row work frame (today `ds4_dist_work_fixed` is one session's contiguous
token span, `ds4_distributed.c:2516-2546`) or worker-side coalescing; row-batch
encoders for slice graphs (per-row `input_hc`/`output_hc` already exist in the
single-token slice API, `ds4.c:74580-74581`); and per-session KV memory as in
§4.2. This is a protocol + graph change, squarely against the "no structural
change without owner consent" rule — recorded here with the numbers so the
decision can be made on purpose rather than by default.

### 4.4 Micro-items (≤ 1–2 % each; do not chase these first)

* The 605 KiB logits return per token is 32-bit fixed (`ds4_distributed.c:7581`);
  §4.1 removes most of it. Sampling on the worker would remove the rest but
  moves sampling policy across the wire — not recommended.
* `dist_coordinator_eval_span` mallocs its 64 KiB hidden buffer per token
  (`:2706-2712`) — trivial, but a per-session reusable buffer is one line if
  anyone is in there anyway.
* The two ~1 ms wire crossings per token are structural (hidden out, result
  back); `DS4_DIST_SOCKET_BUFFER_MB` is already 128 MiB.

### 4.5 Non-levers (checked, so nobody reopens them)

* **MTP/speculative decoding** — excluded under a layer split by
  `ds4_engine_has_mtp` (`ds4.c:62256-62260`), for the structural reason that
  the draft head and its routing would be needed on both sides; and *measured*
  a ~21 % decode loss at depth on the single Mac (impl-log §4.5). At 250–500 K
  tokens — this pair's whole purpose — acceptance collapses; it is not a lever
  here even if the exclusion were lifted.
* **Split rebalance for prefill** — memory-refused Spark-ward
  (104.73/107.61 GiB), and Mac-ward *reduces* prefill (the frontier:
  455.2 → 398.8 t/s moving three layers Mac-ward, impl-log §3.3). Decode does
  not respond to the split (sum, ~1.3 ms/layer) — the sweep already proved it.
* **`--dist-prefill-chunk` / `--dist-prefill-window` / activation bits** —
  measured no-ops on this pair (impl-log §3.1); the link is at a few percent of
  capacity (174.85 MB/s mean vs 0.46 GiB/s single-stream).
* **SSD-streaming on either side** — irrelevant to this pair's resident mode.

---

## 5. What to run when the machines free up

Ordered cheapest-first; each item states what it confirms.

1. **Per-token stage budget (confirms §2.4).** With the pair serving a decode
   stream at ~285 K, set `DS4_DIST_DECODE_PROFILE=1` on the coordinator (and
   keep the worker log). Expected: `span … local≈40±5 ms remote≈65±5 ms
   total≈106 ms` (`ds4_distributed.c:2768-2782`), `send≈0.2 ms`,
   `wait_result≈70 ms`; worker side `eval≈65±5 ms`, `lock_wait≈0`
   (`:7588-7602`). If `local` is not ≈40 ms, §2.4's split is wrong and §4.1's
   gain changes accordingly (recompute from the printed `eval` times).
2. **Head-on-Mac A/B (§4.1).** Same method as the E1 A/B
   (`speed-bench/e1-decode-ab/e1_measure.py`, nonce-prefixed cold prefill,
   median inter-chunk gap, identical-output wall time as second estimator):
   worker `--layers 21:output` vs `--layers 21:44`, ~285 K prompt, expect
   ~−8 % decode, prefill within noise. Then the correctness gates for the new
   topology: criterion-1 oracle (byte-identical continuation, logits
   tolerance), boundary gates 2048→2056 / 4096→4100, and `--dist-replay-check`
   (all three exist and passed for the current topology, `ds4-glm53-oracle.md`).
3. **Same-window duty (confirms §2.2).** Sample both meters aligned to phases:
   `nvidia-smi --query-gpu=utilization.gpu -l 1` on the Spark during a decode
   window (expect ~65 % average, samples oscillating ~100/0), during a prefill
   window (~74 %), and the Mac equivalents. This replaces the blended reading
   with regime-attributed readings.
4. **Concurrency characterisation (informs §4.2/§4.3 decisions).** Today a
   second pipeline session cannot exist (§4.2), so nothing to measure yet — but
   `speed-bench/serve_concurrency_bench.py` against the pair documents the
   single-session queueing behaviour (expected: requests serialise; aggregate
   ≈ single-stream t/s) and is the harness to reuse after any interleaving
   change.
5. **Prefill stage ratio (confirms the Spark-idle-is-blocked claim).** The
   coordinator's `--debug` telemetry prints per-hop `eval=` ms per chunk
   (`ds4_distributed.c:2410-2425`); expect Spark/Mac ≈ 0.74 at `0:20`.

---

## 6. Owner decisions (recorded 2026-10-11)

1. **What produced the 40 %/65 % reading?** `macmon` on the Mac,
   `nvidia-smi` on the Spark. Both are residency/duty measures with ~1 s
   sampling (§2.1), so the reading is a phase blend and §2.2 is the whole
   story; macmon's GPU channel is the same IOReport residency class as the
   project's own duty figures. No further reconciliation needed.
2. **Is the workload ever more than one concurrent client?** **Yes — the owner
   wants multiple concurrent clients**, and states **~100 % GPU utilisation on
   both machines as the top priority**. Those two wants are the same project
   (§7): a single request's decode can never occupy both GPUs at once, so
   "100 % on both" is only reachable through concurrency. §4.2 is therefore
   the next workstream, §4.1 is its cheapest enabling step, and §7 sets the
   expectations and the split decision it implies.

---

## 7. The path to ~100 % on both machines

### 7.1 What saturation means for a two-stage pipeline

With N concurrent decoding sessions and a scheduler that always has another
token ready when a stage finishes, a two-stage pipeline saturates the slower
stage and leaves the faster one idle in proportion to the stage-time ratio:

```
duty_Mac   = T_mac  / max(T_mac, T_spark)
duty_Spark = T_spark / max(T_mac, T_spark)
aggregate t/s = 1 / max(T_mac, T_spark)
```

So "100 % on both" requires **(a) concurrency to remove the bubbles** (the
§4.2 work) **and (b) T_mac ≈ T_spark** (a stage-balanced decode split). Neither
alone is sufficient: today's serial single-request decode is all bubbles
(40 %/65 %), and concurrency against an unbalanced split caps the Mac at
T_mac/T_spark.

### 7.2 Decode duty ceilings by configuration (stage times from §2.4, estimates)

| configuration | T_mac | T_spark | Mac duty at saturation | Spark duty | prefill (0:20 basis) |
| --- | ---: | ---: | ---: | ---: | ---: |
| today, head on Spark, `0:20` | ~42 ms | ~69 ms | 61 % | 100 % | 455 t/s |
| §4.1: head on Mac, `0:20` | ~46 ms | ~57 ms | **81 %** | 100 % | ~455 t/s (unchanged) |
| §4.1 + 2 layers Mac-ward (~`0:22`) | ~50 ms | ~52 ms | **~96 %** | 100 % | ~425 t/s (−7 %) |

The remaining few percent are wire and scheduling bubbles (~2–3 ms per
round trip) — **96 %/100 % is the realistic decode ceiling**; a meter showing
both pinned at 100 % will still only appear during prefill or with sampling
coarse enough to blur the bubbles.

Two expectations to set explicitly:

* **Saturation buys aggregate throughput, not per-token latency.** At
  saturation the pair produces ~1 token per 52–57 ms ≈ **18–19 t/s across all
  clients** (vs 9.4 t/s single-stream today); each client's inter-token gap
  becomes ≈ N × T_spark (114 ms at N=2 post-head-move) — still far better than
  today's multi-client reality of N × full round trip plus session re-sync
  replays between prompts. A single client alone will continue to read
  ~40 %/65 % on the meters — that is correct behaviour, not a regression.
* **Duty is not FLOPs.** During decode both GPUs are memory-bound; 100 % duty
  means "a kernel is resident 100 % of the time", which is exactly the state
  in which the hardware cannot do more batch-1 work than it is already doing.

### 7.3 Prefill stays Mac-bound — and that is the one place 100/100 is refused

Prefill is `max(stage)` with the Mac the binding stage (99–100 %) and the Spark
blocked at ~74 %. Balancing prefill would require moving layers Spark-ward,
and the Spark is at 104.73 of 107.61 GiB at `0:20` — memory refuses, which the
split sweep already established (impl-log §3.3). With concurrent clients the
Spark's prefill idle cannot be harvested either: a new client's prefill needs
the Spark's stage too, and if the Spark is saturated by decode the prefill
queues. **Recommended acceptance: prefill remains ~100 %/74 %, decode becomes
~96 %/100 % at load.** If the last 7 % of prefill matters more than the last
15 points of Mac decode duty, keep `0:20` and accept ~81 % — that is the
§7.2 middle row, and it is the recommended default because prefill is the
phase every new client pays.

### 7.4 Recommended sequence

1. **§4.1 head-on-coordinator A/B** (configuration + oracle gate) — ~−8 % decode
   single-stream, stage balance 42/69 → 46/57, +1.18 GiB Spark slack (the
   memory enabler for step 2), prefill unchanged.
2. **§4.2 multi-session interleaving** (the design doc
   `ds4-glm53-multi-session-design.md`) — concurrent clients, both GPUs
   toward their §7.2 ceilings, aggregate ~18–19 t/s at saturation; per-client
   ITL ≈ N × T_spark under load.
3. **Split decision, measured, not assumed:** re-run the split sweep's
   decode points at `0:22`-vs-`0:20` with the head on the Mac *and* ≥2
   concurrent clients, reading duties from both meters; take the 2-layer
   shift only if the measured prefill cost is acceptable (§7.2's third row).
4. **§4.3 true batching** only if per-pass throughput beyond ~19 t/s is ever
   needed — it multiplies throughput but does not by itself raise duties
   (the stages still alternate per pass); it is a later, larger decision.
