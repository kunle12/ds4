# Multi-session pipeline — implementation results (2026-10-11)

**Status:** workstreams **0–3 of the design are implemented, gated and
deployed** to `~/bin` on both hosts; the production pair serves on the new
binaries with the `21:44` headless worker. Concurrent-client serving works and
is measured. One incident occurred and is recorded with its evidence (§3):
the first concurrency attempt at ctx 262144 wedged the Spark hard enough to
need a power cycle — the memory table in the design doc is corrected
accordingly, and the working configuration is bounded by *measured* headroom,
not the guard budget.

---

## 1. What was implemented (`3703623`…, all in `ds4_distributed.c`,
## `ds4_distributed.h`, `ds4_server.c`)

| Workstream | Change |
| --- | --- |
| 1 — shared coordinator core | One listener, accept thread, worker registry and GPU mutex per engine (`ds4_dist_core`), refcounted by session create/free and retained for process lifetime (detached threads touch it). Sessions embed the read-only launch template and share the core's mutable state (`ds4_dist_shared`). The second bind on the coordinator port — the thing that made multi-session serving impossible — is gone. |
| 2 — per-session work connections | Each session dials the worker's data listener privately and sends WORK frames there; the plan owns a `dup()` of the session's fd, so plan lifetime stays self-contained while the connection cache survives rebuilds. Strict per-fd request matching becomes per-session. Transport failure drops the connection and forces a plan rebuild + re-dial. The registry's hello connection is now purely registration/liveness. |
| 3 — narrowed GPU exclusion | A shared `gpu_mu` covers exactly the coordinator's GPU-command sections: the local slice eval (decode span + per-chunk prefill), coordinator-side head application (both call sites), KV shard save/load, and — via the new public `ds4_dist_gpu_mutex/lock/unlock` — the server's image encoder. The remote round trip runs **without** the GPU lock. In the server, distributed slots self-serve decode in their own thread (the batched decode worker is bypassed; it has no native batch for slice sessions anyway), and prefill keeps `inference_mu`. |
| follow-up fix | Batched-mode prefill quanta (2048) fed the dist sync spans no larger than one chunk, silently switching prefill onto the serial span path — measured **269 t/s against ~370–430**. Distributed sessions now sync the whole suffix in one call; the dist prefill chunks internally, reports progress per chunk and yields the GPU lock between chunks, so other slots' decode steps still interleave. |

## 2. Gates and measurements

**Single-session regression (criterion 4)** — production config, ctx 524288,
non-batched: 11K run decode median 92.66 ms pre-fix / **91.0 ms** on the
final deployed binaries, completion sha `a3898c57…` identical in every run,
prefill ~371–380 t/s (unchanged envelope).

**Bounded stalls (criterion 7)** — re-run on the new architecture: a worker
`SIGSTOP` mid-decode fails the request at **exactly 120.0 s** with a clean
client stream end; server-side `finish=error "distributed route incomplete"`;
after `SIGCONT` the worker re-dials within seconds, clears its sessions and a
fresh request succeeds. The semantics survived the per-session-connection
refactor.

**Concurrency oracle (criterion 1) + saturation (criteria 2–3)** — ctx 65536,
`--batched-session 2`, `--dist-prefill-chunk 2048`, worker available ≥ 10 GiB
throughout:

| run | TTFT | decode median gap | sha |
| --- | ---: | ---: | --- |
| A alone | 27.40 s | 91.03 ms | `a3898c57…` |
| B alone | 27.34 s | 90.75 ms | `a3898c57…` |
| A+B concurrent | 52.9 / 53.4 s | **112.3 ms each** | `a3898c57…` both |

* **Output byte-identical under concurrency** — the interleaving does not
  touch numerics, exactly as designed (per-session graphs, serialized GPU
  sections).
* **Aggregate decode ≈ 17.8 t/s vs 11.0 single-stream (+62 %)**; per-client
  latency 91→112 ms — the designed N-sharing trade.
* Both prefills shared the Mac at full rate (2 × 11,478 tokens / 53.2 s ≈
  431 t/s aggregate).
* Duty during the concurrent decode (phase-aligned samplers): **Spark
  64.5 %→76.8 % avg (median sample 96 %), Mac 48.7 %→56.1 % avg (median
  67.3 %)** — the overlap is real and the Spark climbs toward saturation as
  predicted; with only 2 sessions the window is short and edge samples drag
  the averages.

## 3. The ctx-262144 incident — and the corrected memory table

The first concurrency attempt (ctx 262144, batched 2, guard reserve 14) wedged
the Spark hard: the second worker session's creation (~4.2 GiB marginal
against ~5.6 GiB of guard-budget slack) put the box under memory pressure and
the NVIDIA driver never came back — the previous boot's journal records
`task nvidia-smi blocked on an rw-semaphore likely owned by task ds4` and
repeated `Under memory pressure, flushing caches`, and the box needed a power
cycle. The guard had **admitted** the allocation (~106.2 of its 107.61 GiB
budget), which is the design-doc §3.4 estimate ("2–3 × 262144 fits with
margin") being proven wrong by measurement.

Corrected guidance, from measurement only:

| configuration | worker state | verdict |
| --- | --- | --- |
| 2 sessions × ctx 262144 | ~106.2/107.61 budget, ~1.4 GiB nominal slack | **wedges the GB10** — withdrawn |
| 2 sessions × ctx 131072 | ~104.5 est., ~3 GiB nominal slack | untested; suspect — do not assume |
| **2 sessions × ctx 65536** | ~102/107.61 budget, **≥ 10 GiB available measured** | **works** — the gated configuration |
| 1 session × ctx 524288 (production) | 103.55/107.61 | works (unchanged) |

The operative rule: for N-session serving, size ctx so the worker's *available
memory* stays ≥ ~10 GiB after all sessions exist — the guard budget and the
idle `free -g` figure are necessary, not sufficient (NVRM's own reservation
and allocation-time pressure are invisible to both), exactly as the
split-design doc's §4.3 warning says, now with a concrete incident attached.
The design's workstream 4 (shared GLM prefill workspace, which would cut the
per-session scratch toward KV-only) is the lever that could raise the
per-session ctx ceiling; it is now *necessary* rather than optional for
anything above 65536 per session.

## 4. Operating the concurrent configuration

```sh
# Worker: same as production, smaller ctx
DS4_GLM_LAYER_SLICE_TOKEN_DECODE=1 DS4_GLM_MEMORY_GUARD_RESERVE_GB=14 \
  ~/bin/ds4 --cuda -m ~/mlmodels/glm/GLM-5.3-Flash-Q4_K.gguf \
  --role worker --layers 21:44 --coordinator 192.168.2.1 9911 \
  --listen 192.168.2.2 55911 --ctx 65536

# Mac: batched slots enable concurrent clients
DS4_GLM_LAYER_SLICE_TOKEN_DECODE=1 ~/bin/ds4-server ... \
  --role coordinator --layers 0:20 --listen 192.168.2.1 9911 \
  --ctx 65536 --batched-session 2 --dist-prefill-chunk 2048 \
  --host 0.0.0.0 --port 8081
```

Notes: `--dist-prefill-chunk 2048` halves the per-session prefill workspace
(measured no prefill-rate cost on this pair); a long prefill on one slot
monopolises the Mac's prefill for its duration (one-shot syncs serialize
prefills; decode steps of other slots still interleave between chunks);
watch the worker's `free -g` when adding sessions. The production
configuration (single session, ctx 524288) is unchanged and remains the
default in `~/bin/llm_config.json`.
