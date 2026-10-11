# Workstream 0 + the head move — measured results (2026-10-11)

**Status:** both landed. Workstream 0 (bounded result wait) is implemented at
`c2787ea`, deployed to `~/bin` on both hosts and stall-verified on the live
pair. The output-head-on-coordinator topology (`21:44` headless worker) is
adopted in `~/bin/llm_config.json` after passing its oracle gates. The
production pair was restarted on the new topology and serves normally.
**Method:** every number below is from the live pair (titan / spike), the
`e1-decode-ab` harness, `DS4_DIST_DECODE_PROFILE=1` span lines, phase-aligned
`macmon`/`nvidia-smi` sampling, and the criterion-1 oracle method from
`ds4-glm53-oracle.md`.

---

## 1. Baseline (pre-change build `e407bdf`, worker `21:output`)

Cold 285,164-token prefill + greedy decode, the E1 harness:

| metric | value | documented reference |
| --- | ---: | --- |
| prefill | 409.8 t/s (TTFT 695.56 s) | 415.0 t/s ✓ |
| decode median gap | 108.53 ms (104.29–115.46) | 106.45–121 ms ✓ |
| completion sha256 | `a3898c57…` | ✓ identical |

Stage budget at depth (span profile, per token):

| stage | measured |
| --- | ---: |
| Mac local (0:20) | **~29.5 ms** |
| Spark remote (21:44 + head) | **~73.5 ms** |
| send / copy / payload | 0.10 ms / 0.04 ms / 0.59 MiB logits |

Phase-aligned duty (macmon `gpu_active_ratio` / `nvidia-smi utilization.gpu`,
1 s sampling): **prefill Mac 99.1 % avg, Spark 69.2 % avg with 96 % median
samples; decode Mac 32.9 % avg (median 38.6), Spark 65.7 %**; idle 0 %.
This is the owner's "40 % Mac / 65 % Spark, sometimes both 100 %" observation
confirmed and regime-attributed — and it corrects the analysis doc's §2.4
estimate: the Mac stage is ~29 ms, not ~42 ms, so the imbalance is larger.

## 2. Workstream 0 — bounded result wait (`c2787ea`)

`DS4_DIST_RESULT_TIMEOUT_SEC` (default 120 s) on the coordinator's response
waits (decode span, pipelined-prefill reader, KV-snapshot dials) plus
30 s / 10 s / 3 TCP keepalive on all dist sockets.

* **Regression gate** (~11 K, post-change build): prefill 380.5 t/s,
  decode median 94.99 ms (93.08–102.35), sha identical, no error — inside the
  documented envelope (93.70 ms reference, 92.20–98.36 range).
* **Stall test** (worker `SIGSTOP`ed mid-decode — the wedge class: kernel
  alive, answering keepalive probes, no FIN): the request terminated
  **119.9 s after the freeze**, the client saw a clean stream end, the
  coordinator logged `wait_result=120003.7 ms … rc=1` and
  `finish=error "distributed route incomplete: missing layer 21"`, and after
  `SIGCONT` the worker saw EOF, cleared its sessions, **re-dialled within
  seconds**, and a fresh request succeeded. Before this change the same freeze
  hung the request indefinitely — the code agent's 300 s chunk timeout was the
  only thing that fired, and the server stayed blocked.
* Semantics as designed: bounded, visible server-side, self-healing, and the
  worker's legitimately idle read loop is untouched.

## 3. The head move — worker `21:44` (headless), head on the coordinator

Configuration-only (the topology `ds4.c:70648-70652` was built for): the
Mac already maps `0:20+output`; the worker drops the 1.182 GiB head and
returns hidden states, which the Mac folds through its own head
(`output_head=3.22 ms` inside the remote round trip).

| metric | `21:output` baseline | `21:44` headless | delta |
| --- | ---: | ---: | ---: |
| prefill, 285 K | 409.8 t/s | 407.3 t/s | unchanged (noise) |
| decode median gap | 108.53 ms | **106.40 ms** | **−2.0 %** |
| completion text | `a3898c57…` | `a3898c57…` | **byte-identical** |
| completion tokens | 80 | 103 | near-tie tokenisation flip, same text |
| Spark plan | 104.73 GiB | **103.55 GiB** | **+1.18 GiB slack** |
| return payload/token | 0.59 MiB logits | 0.06 MiB hidden | −90 % |
| Mac stage at depth | ~29.5 ms | ~30 ms | unchanged |
| Spark remote at depth | ~73.5 ms | ~71.3 ms | −2.2 ms |

**Why the measured gain is −2 %, not the predicted −8–10 %:** the prediction
priced the Spark's head read at ~12 ms using the 100.7 GB/s CPU-proxy
bandwidth figure. The measured head cost on the Spark is only **~4.3 ms**
(back-solved from the stage deltas: the GB10's GPU-side reads run ~3× the
proxy — the impl-log's own caveat "a CPU-side proxy, so both sit below spec —
the ratio is the point" was the warning ignored). The analysis doc's §4.1
arithmetic is corrected accordingly. The move is still adopted because the
memory (+1.18 GiB, the multi-session enabler), the payload, the small decode
gain, and the *improved* numerics all point the same way, at zero measured
prefill cost.

A second observation the stage data forces: the Spark's context-independent
per-layer floor is ~2.2 ms/layer (~53 ms at pos 17 → ~67 ms at 285 K), so at
depth the remote stage is *not* purely weight-traffic — the context-dependent
term (+14 ms over 285 K) and per-layer overheads dominate the difference. The
"decode is weight-bandwidth-bound" framing holds for the fixed component, not
for the depth term.

## 4. Oracle gates for the headless topology (all pass)

Criterion-1 method (`ds4-glm53-oracle.md`), ctx 32768, both sides freshly
run today; the pair runs the `21:44` worker:

| gate | result | old topology (2026-09-21) |
| --- | --- | --- |
| continuation, 2,881 tk | **290/290 byte-identical** | 290/290 |
| logits, 2,881 tk | mean \|Δ\| **0.170**, max 1.16, argmax equal, top-8 8/8, top-16 14/16, top-64 59/64, all finite | 0.307 / 1.98 / 7/8 / 15/16 / 56/64 |
| boundary, 5,101 tk (crosses 2048 + 4096) | mean \|Δ\| **0.103**, max 0.68, argmax equal, top-8 7/8, top-16 15/16, top-64 63/64 | 0.127 / 0.78 / 8/8 / 15/16 / 62/64 |
| `--dist-replay-check` | **exact across reset/replay** | exact |

The agreement is *better* than the old topology's on the headline metric
(mean |Δ| −45 %): the output head now runs on Metal on both sides of the
comparison, removing the last cross-backend component from the reference path.

## 5. What this changes in the companion documents

* `ds4-glm53-response-data-timeout.md` — workstream 0 implemented and
  stall-verified; the failure the owner reported is bounded and self-healing
  on the deployed binaries.
* `ds4-glm53-gpu-utilisation-analysis.md` — §2.4 stage budget corrected to
  the measured 29.5/73.5 (→ 30/71 headless); §4.1's predicted gain corrected
  to the measured −2 %; §7.2 re-derived (below).
* `ds4-glm53-multi-session-design.md` — criterion 7 (bounded stalls) passed;
  the head move is landed, so the design's memory arithmetic uses the 103.55
  GiB worker plan; workstream 0 is done.
* `DISTRIBUTED.md` / implementation log §1 — the recommended GLM 5.3 Q4_K
  pipeline worker is now `--layers 21:44`.

**§7.2 of the analysis, re-derived from measurement:** at the measured stage
ratio (30/71 at depth), multi-session decode saturation lands at
**Spark ≈ 100 %, Mac ≈ 42 %, aggregate ≈ 1/71 ms ≈ 14 t/s**. Balancing the
decode stages would need ~7 layers moved Mac-ward — ~110 t/s of prefill by the
measured endpoint slope — which is not a trade worth making. The honest
conclusion for the owner's "100 % on both" priority: prefill stays
~100 %/~70 %, decode saturation is ~42 %/100 % on any prefill-healthy split,
and the throughput lever that remains is true batching (design §4.3), whose
per-pass stage floors should be re-derived from these measurements when that
workstream starts.
