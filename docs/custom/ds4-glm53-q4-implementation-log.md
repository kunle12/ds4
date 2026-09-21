# GLM 5.3 Flash Q4_K across a Mac Studio and a DGX Spark

How to run `GLM-5.3-Flash-Q4_K.gguf` as a two-machine pipeline — Mac Studio M4 Max
as coordinator, DGX Spark GB10 as worker — and why it is configured the way it is.

This document states the current position. The history of how each fact was
established, including the mistakes, is in Appendix B; the evidence for individual
numbers is cited inline.

---

## 1. Summary

**The model.** `glm5-next`, 46 blocks of which 45 are executable (block 45 is the
MTP block and is not mapped), 320.76 B parameters, 177.77 GiB on disk. Weights are
Q4_K for the 129 routed-expert tensors and BF16/Q8_0/F32 elsewhere.

**The pairing.** The model does not fit on either machine alone, so it is split by
layer range: the coordinator holds a prefix and the worker holds the rest plus the
output head. Both machines keep their half resident, which is the point — the
single-machine alternative has to stream experts from SSD and is bound by that.

**Recommended configuration** (ctx 524288, all 45 layers resident):

```sh
# Spark — worker, started FIRST (it retries until the coordinator listens)
DS4_GLM_MEMORY_GUARD_RESERVE_GB=14 /home/xun/bin/ds4 --cuda \
  -m /home/xun/mlmodels/glm/GLM-5.3-Flash-Q4_K.gguf --role worker \
  --layers 21:output --coordinator 192.168.2.1 9911 --listen 192.168.2.2 55911 --ctx 524288

# Mac — coordinator
/Users/xun/dev/ds4/ds4-server --chdir /Users/xun/dev/ds4 --metal \
  -m /Users/xun/mlmodels/glm/GLM-5.3-Flash-Q4_K.gguf --role coordinator \
  --layers 0:20 --listen 192.168.2.1 9911 --ctx 524288 --host 0.0.0.0 --port 8081
```

**What it delivers**, measured cold at 286 646 prompt tokens:

| | pair (`0:20`) | pair (`0:23`) | Mac alone (`--ssd-streaming`) |
| --- | ---: | ---: | ---: |
| prefill | **415.0 t/s** | 356.3 t/s | 82.47 t/s at 262K |
| 286K ingest | **11.5 min** | 13.4 min | 53 min at 262K |
| decode | **121.1 ms/token** | 120.5 ms/token | 8.00 t/s = 125 ms at 262K |
| 524K decode | — | — | 4.63–5.02 t/s = 199–216 ms |

So: **~4–5× the single machine's prefill, and decode that is comparable at 262K
but ~1.6× better at 512K**, because the single-machine path degrades with depth
(its expert cache cannot grow) while the pair does not.

**Status.** Working end to end and in daily use, serving
`glm-5.3-flash` / `-chat` / `-reasoner` at context 524288 over HTTP. What remains
is verification breadth, not capability — see §7.

---

## 2. Architecture and data flow

### 2.1 What each machine holds

The split is a contiguous layer range, chosen at launch. The coordinator holds
`0:N`; the worker holds `N+1:output`, which always includes the output head. Both
log the slice they map:

```
Mac   : restricting metal model map to layers 0:20+output (44 spans, 76.47 GiB tensor span)
Spark : restricting cuda model map to layers 21:44+output (50 spans, 98.45 GiB tensor span)
```

Exact per-layer weights, from the GGUF tensor table:

| slice | layers | weights |
| --- | ---: | ---: |
| `token_embd` / output head | — | 1.182 GiB each |
| blk.0–2 — the only dense layers | 3 | 0.408 GiB each |
| blk.3–44 — MoE, 288 experts, 8 active | 42 | ~4.08 GiB each |
| blk.45 — MTP, not mapped without `--mtp` | 1 | 4.019 GiB |

The first three blocks being an order of magnitude smaller matters when reasoning
about a prefix split: a coordinator taking `0:N` gets those three almost free.

### 2.2 Per-token flow

```
HTTP client
    │
    ▼
┌──────────────────────────────────────────────────┐
│ Mac Studio M4 Max — coordinator                  │
│   embed → layers 0:N                             │
│   + HTTP, tokenisation, session/KV, token loop   │
└──────────────────────────────────────────────────┘
    │  N_HC × N_EMBD × 4 B = 64 KiB per token
    │  (measured: 174.85 MB/s mean during prefill,
    │   one ~256 MiB burst per 4096-token chunk)
    ▼
┌──────────────────────────────────────────────────┐
│ DGX Spark GB10 — worker                          │
│   layers N+1:44 + output head                    │
└──────────────────────────────────────────────────┘
    │  ~0.2 MB/s back — essentially just the sampled token
    └──────────────► Mac embeds it and continues
```

Traffic is 871× one-way in favour of Mac→Spark. That is measured, not assumed:
across a cold 8 274-token prefill the Mac sent 174.85 MB/s mean (peaks 260.31,
147.06, 117.20) and received 0.20 MB/s. The three bursts reconcile with the chunk
transfers — 8 274 tokens at ~64 KiB is ~525 MB, and the bursts sum to ~524 MB.

The 64 KiB/token figure is `N_HC × N_EMBD × 4`, where GLM 5.3's mHC block carries
four residual streams (`N_HC = 4`). It is 4× a plain f32 hidden state, which is
what the wire width must be for this model — getting it wrong was the defect that
made pipeline mode impossible at first (§10, change 1).

### 2.3 The structural fact that governs tuning

- **Prefill is chunked and pipelined.** Tokens are processed in 4096-token chunks
  (a 2048 sub-chunk granularity appears at long context), and successive chunks
  overlap across the two stages. Its cost is therefore
  $\max(\text{Mac stage}, \text{Spark stage})$.
- **Decode is strictly serial.** One token walks the Mac's layers, crosses the
  wire, walks the Spark's layers, and comes back. Its cost is
  $\text{Mac stage} + \text{Spark stage} + \text{wire}$.

That asymmetry — `max` versus `sum` — is why the layer split should be chosen for
prefill, and it is quantified in §4.4. It is also why loading the Spark harder
costs nothing at decode: moving a layer between machines only *swaps* which stage
pays for it.

---

## 3. Configuration

### 3.1 The knobs that matter

| Flag / variable | Host | Effect |
| --- | --- | --- |
| `--layers 0:N` / `N+1:output` | both | the split; the only real tuning lever (§3.3) |
| `--ctx` | both | must match; KV is small for this model (2.66 GiB at 524288 on a 21-layer slice) |
| `--role coordinator\|worker` | both | coordinator owns the prefix, HTTP, tokenisation, and the token loop |
| `--listen A P` | both | coordinator: the control port the worker dials. Worker: its data port |
| `DS4_GLM_MEMORY_GUARD_RESERVE_GB` | both | guard reserve; see §3.3 |
| `DS4_GLM_MEMORY_GUARD_FRACTION` | both | guard fraction, default 0.99 |
| `iogpu.wired_limit_mb` | Mac | sysctl; raises the Mac's guard budget (§3.3) |
| `DS4_CUDA_GLM_MOE_TYPES` | Spark | narrows the types the GLM-specific MoE dispatch accepts (`q2k,q4k` default); it is not a route selector — a homogeneous Q4_K trio is served by the generic dispatch before that entry is consulted (§3.1) |
| `DS4_GLM_MOE_TRACE=1` | Spark | one line per MoE dispatch: type, tokens, experts, used, path |
| `DS4_GLM_LAYER_SLICE_TOKEN_DECODE` | both | GLM slice single-token steps use the decode graph even with inter-node hidden state. Default **on** for GLM 5.3 on Metal/CUDA — measured +7.7 % / +15.4 % / +19.6 % decode at ~11K / ~285K / ~473K with identical output; set `0` to opt out. ROCm and non-5.3 GLM stay opt-in |
| `--dist-activation-bits 16` | both | halves the wire payload; no measured gain, changes numerics |
| `--dist-prefill-chunk`, `--dist-prefill-window` | both | no measured effect on this pair |

Start the worker first. It retries every second until the coordinator's control
port answers, so ordering is a convenience rather than a requirement.

### 3.2 Memory budgets, and how they are computed

The GLM guard sizes itself from the host, and the two machines differ for two
separable reasons. From `ds4.c`:

```
base   = hw.memsize                              (Apple)
       | cudaMemGetInfo(total) x devices          (CUDA)
budget = min(0.99 x base,  base - reserve)        reserve = 18 GiB for GLM 5.3
                                                   on a 108-160 GiB host
Apple:   if iogpu.wired_limit_mb is set  ->  budget = wired_limit - 2 GiB
```

| | Mac | Spark |
| --- | ---: | ---: |
| OS-visible total | 128.00 GiB (`hw.memsize`) | 121.61 GiB (`MemTotal`) |
| guard base | 128.00 GiB | 121.61 GiB |
| default budget (base − 18) | 110.00 GiB | 103.61 GiB |
| override in effect | `iogpu.wired_limit_mb=120000` | none |
| **effective budget** | **115.19 GiB** (= 117.19 − 2) | **103.63 GiB** |

The 11.56 GiB gap is **6.37 GiB of hardware** — the GB10 does not expose ~6.4 GiB
of its 128 GB to Linux — plus **5.19 GiB of tuning**, the Mac's raised wired limit.
On identical settings the two machines are ~6.4 GiB apart. Neither number is a
capability difference: both the reserve and the fraction are runtime-settable.

Measured plans (ctx 524288, the numbers the engine prints at startup):

| split | Mac plan | Spark plan |
| --- | ---: | ---: |
| `0:20` / `21:output` | 82.22 GiB of 115.19 | **104.73 GiB of 107.61** |
| `0:23` / `24:output` | 94.88 GiB of 115.19 | 92.07 GiB of 103.63 |
| `0:26` / `27:output` | 107.15 GiB of 115.19 | 79.80 GiB of 103.63 |
| `0:27` / `28:output` | **111.64 GiB of 115.19** | 75.31 GiB of 103.63 |

A plan is admitted or refused as a whole, so the guard is the authority on whether
a split fits — read it in the coordinator's and worker's logs rather than
estimating. The 18 GiB reserve on the Spark is why the recommended configuration
sets `DS4_GLM_MEMORY_GUARD_RESERVE_GB=14`: without it, `0:20` is refused.

**The Spark cannot run the whole model, and the guard cannot tell you that.** Given
the whole 177.77 GiB the engine's guard admits it and the GPU driver then refuses the
allocations outright (`NVRM: … Out of memory [NV_ERR_NO_MEMORY] …
_memdescAllocInternal`), logging stops within 7 s, and the box resets ~19 min later
with the watchdog never armed. The guard's budget derives from system RAM and does
not account for NVRM's own reservation, so it is not authoritative for whole-model
runs on that host. This is why the split exists at all.

### 3.3 Choosing the split

Measured frontier, cold, 39 865-token prompt, ctx 524288 (`m` = layers held by the
Mac):

| split | m | Spark | prefill | engine chunk rate | decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0:20` | 21 | 24 + head | **455.2 t/s** | ~485 t/s | 103.0 ms |
| `0:23` | 24 | 21 + head | 398.8 t/s | — | 98.5 ms |
| `0:26` | 27 | 18 + head | 366.9 t/s | — | 95.2 ms |
| `0:27` | 28 | 17 + head | 344.9 t/s | — | **93.9 ms** |

**The rule: set the split for prefill.** The two sensitivities differ by ~3.1× in
percent terms, but only decode is close to linear per layer:

- moving one layer to the Mac costs **15.8 t/s of prefill** as an
  **endpoint average over `0:20`…`0:27`**. It is not uniform: the three segments
  cost **18.8 / 10.6 / 22.0 t/s per layer** (a 2.1× spread), because the layers
  being moved differ — dense blocks 0–2 are ~0.4 GiB each, MoE blocks ~4 GiB, and
  KDA and sparse layers do different work. Read 15.8 as an average, not a local
  slope.
- moving one layer to the Mac buys **1.30 ms of decode** (~1.3%), which the
  segment figures confirm as near-linear (1.50 / 1.10 / 1.30 ms per layer).

Prefill is `max(stage)`, so rebalancing recovers the *whole* difference between the
stages; decode is `sum(stage)`, so moving a layer only swaps one stage's cost for
the other's and just the per-layer difference shows. Since decode barely responds,
optimise the quantity that does.

**Where it stops, and why it is a memory ceiling rather than a balance point.** The
recommended `0:20` leaves the Spark at 104.73 GiB of a 107.61 GiB budget (reserve
14). At that split the Spark runs ~74% duty — three of four samples at 96%, one at
7% — and the coordinator is still the binding stage, so prefill has not yet
crossed its balance. Going further needs a reserve below 12 GiB, i.e. under 1 GiB
of margin, which is not worth attempting: the Mac remains the constraint, holding
32 GiB it cannot usefully spend, because a layer moved to the Mac *costs* prefill.

---

## 4. Performance

All numbers below are cold (`cached_tokens: 0`) and greedy, measured at the HTTP
boundary with a streaming request: time-to-first-token is the prefill and
inter-token gaps are the decode, from the same request. Prompts are nonce-prefixed
because the coordinator serves a repeated prompt prefix from session cache — a
repeated 39 846-token prefill returned in **0.191 s** with `cached_tokens: 39846`,
which is not a prefill and must never be quoted as one.

### 4.1 Prefill

| depth | pair `0:20` | pair `0:23` | Mac alone (SSD streaming) |
| ---: | ---: | ---: | ---: |
| 39 865 | 455.2 t/s | 398.8 t/s | 84.23 t/s at 32K |
| 286 646 | **415.0 t/s** | 356.3 t/s | 82.47 t/s at 262K |
| 479 000 | — | ~320.9 t/s | ~80 t/s |

Prefill is depth-robust — 415 t/s at 286K against 455 t/s at 40K — because
attention is a small part of a chunk's work while every weight is read for every
token. The engine's own per-chunk telemetry in the coordinator log is the cleaner
figure (it excludes pipeline fill) and reads ~485 t/s at 40K and ~460 t/s at 61K
on the recommended split.

**The pair's prefill advantage over one machine is not parallel compute.** The Mac
alone sets the rate at 99–100% GPU. The advantage comes from the Mac holding 24
*resident* layers instead of 45 *streamed* ones: at 82.47 t/s over 45 layers a
streamed layer costs ~1.10 s against 0.406 s resident, i.e. 2.71×, times 45/24 =
1.875× fewer layers, giving ~5.1× before pipeline overhead against 4.3–5.0×
measured.

### 4.2 Decode

| depth | pair `0:20` | pair `0:23` | Mac alone |
| ---: | ---: | ---: | ---: |
| 39 865 | 103.0 ms | 98.5 ms | 8.84 t/s = 113 ms at 32K |
| 286 646 | 121.1 ms | 120.5 ms | 8.00 t/s = 125 ms at 262K |

Decode degrades slowly with depth (the DSA attention term grows with context) and
is insensitive to the split (§3.3). **It is not GPU-bound on either machine**: at
~121 ms/token the Mac sits near 71% and the Spark near 62%, and neither reaches
90%. A batch-1 step is dominated by reading that layer's weights, so the SMs spend
much of the step waiting on memory — which is also why MTP does not rescue it
(§4.5) and why the pipeline does not help much either.

### 4.3 Split sensitivity, measured

| quantity | slope, per layer moved Mac-ward |
| --- | ---: |
| prefill | **−15.8 t/s** (−4.0% of 398.8) |
| decode | **−1.30 ms** (−1.3% of 98.5) |

The `0:27` decode figure landed on the linear fit to 0.0 ms, so the model is not a
loose approximation — it is the behaviour. The mechanism is in §2.3.

**Validated at the operating depth** — 286 646 tokens, the same prompt and cap as
the §4.1 row, so the comparison is like for like:

| split | prefill | 286K ingest | decode |
| --- | ---: | ---: | ---: |
| `0:20` | **415.0 t/s** | **11.5 min** | **121.1 ms** |
| `0:23` | 356.3 t/s | 13.4 min | 120.5 ms |

**+16.5% prefill for decode unchanged** (0.6 ms, inside the run's own gap spread of
116.1–125.8 ms). The decode penalty visible at 40K (4.6%) *disappears at depth*: the
context-dependent attention term grows and dilutes the per-layer difference between
the machines. The deeper the context, the better the Spark-heavy split looks — and
this workload is deep.

### 4.4 Against the single-machine alternative

Same model, same Q4_K weights, greedy, cold prefill. Standalone figures are the Mac
alone with `--ssd-streaming`.

| depth (prompt tk) | prefill: pair | prefill: standalone | ratio | decode: pair | decode: standalone | ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ~40 000 | 386.19 | 84.23 at 32K | 4.6× | 10.03 t/s | 8.84 t/s at 32K | 1.13× |
| ~287 000 | 356.30 | 82.47 | 4.3× | 8.30 t/s | 8.00 t/s | 1.04× |
| ~479 000 | ~320.9 | ~80 | ~4.0× | 7.395 t/s | 4.63–5.02 t/s | **~1.5–1.6×** |

**Prefill is 4.0–4.6× at every depth**; the difference is the SSD cache — the pair
holds all 177.77 GiB resident, while the single-machine path is bound by whichever
experts happen to be cached. The pair column is the **`0:23` / `24:output`** split,
the series with a matched standalone run at every depth; the recommended
**`0:20` / `21:output`** measures **415.0 t/s** at ~287K against 356.30 here, so
these ratios are conservative.

**Decode's advantage grows with depth**, from nothing at 32K to ~1.5× at 479K,
because single-machine decode is cache-bound and falls ~40% from 262K to 512K while
the pair falls ~11%. The pair's decode case therefore strengthens exactly where the
owner works — 500K sessions — and is nearly absent at 32K. (The pre-§4.4 estimate
of ~6.5 t/s for pair decode at depth was pessimistic: 7.395 was measured at 479K.)

**Why decode does not scale like prefill:** pipeline parallelism overlaps prefill
chunks, so throughput adds a second machine's work. A decode step cannot overlap —
one token traverses both stages in series — so its latency is roughly a single
machine's, minus the streaming penalty the pair avoids.

**Capacity.** The pair is also what makes the model usable at 500K at all: the
standalone needs `--ssd-streaming` and peaks at 102.00 GiB of a 115.19 GiB budget
during prefill, ~13 GiB of slack, with the expert cache stuck at 5 378 of 12 384.

### 4.5 Quality

Q4_K against Q2 on the GLM 5.3 Flash 100-case fixture (`score_official`,
`compare_scores.py`): Q4_K is materially better on loss and agreement, and its
per-case advantage is what justifies the larger footprint. MTP is **not** worth
enabling — at 512K on the single machine it is a 21% decode *loss* (3.88 t/s against
4.90/5.02 bracketing it) because acceptance collapses at depth while the verify
cost remains. It is also excluded in distributed mode by `ds4_engine_has_mtp`,
since speculation would need the head and its routing on both sides of the split.

---

## 5. What limits what

Measured duty and utilisation, so the bottleneck is not inferred from throughput:

| phase | Mac GPU | Spark GPU | binding |
| --- | ---: | ---: | --- |
| prefill, `0:23` | **99–100%** | ~57% duty | coordinator |
| prefill, `0:20` | **99–100%** | ~74% duty | coordinator |
| decode | ~71% | ~62% | neither |

**Prefill is coordinator-GPU-bound.** The Mac is pegged; the Spark alternates 96%
busy with gaps because it finishes its chunk and waits. That the Spark idle time is
*blocked* rather than spare is worth being explicit about, because it changes the
conclusion: there is no headroom to harvest on the worker by making it work harder.
The levers are the split (§3.3) and a faster coordinator.

**Decode is bound by weight traffic, not by compute.** Neither GPU is saturated, and
the reason is that a batch-1 step reads a layer's weights and does very little
arithmetic with them. This is why the split barely moves decode (§3.3), why MTP is
a loss (§4.5), and why the pipeline's decode advantage over one machine is modest.

**Memory bandwidth explains the per-layer asymmetry** between the machines
(streaming read, best of five, 8 GiB buffers; a CPU-side proxy, so both sit below
spec — the ratio is the point):

| host | read-only | copy |
| --- | ---: | ---: |
| Mac M4 Max | **295.1 GB/s** | 369.0 GB/s |
| Spark GB10 | **100.7 GB/s** | 123.2 GB/s |

The Mac streams memory 2.9× faster, which is why it is the faster side per layer at
decode (1.30 ms/layer, §4.3) while the Spark is the faster side at prefill.

**The link is not a constraint.** 10GbE direct, RTT 0.94–1.21 ms, single-stream TCP
0.46 GiB/s. Prefill uses ~27 MB/s average (peaks near 260 MB/s during a chunk
transfer) and decode far less — single-digit percent of the link. The wire is
visible in decode only as one round trip per token, ~2 ms of ~121 ms.

**The Spark's memory budget is what bounds the split** (§3.2, §3.3): it is the
binding constraint, not compute.

---

## 6. Thermal, power and link

### 6.1 The Spark is the thermally constrained machine

This unit hard-locked under sustained load once (board 90 °C, `HW_THERMAL_SLOWDOWN`
accumulating, `SW Power Capping` 69 s, box unreachable, silent hard-lock with no
panic or OOM). An NVIDIA field diagnostic fails such units on PowerStress
(`020000600139`); RMA was declined on 2026-09-19, so the mitigation is the
protection kit rather than replacement.

**The kit** (host-level, outside the repository, backed up on the Mac — §11):
`gb10-thermal-guard.service` sampling the board and clamping the GPU clock,
`gb10-cpu-cap.service` + `.timer` holding the CPU at 2.4 GHz against a 2.81 GHz
hardware maximum, and a GPU clock ceiling of 2100 MHz. Idle settles at 47–50 °C.

**It works, and loading the Spark harder is essentially free:**

| condition | board peak | throttling |
| --- | ---: | --- |
| uncapped | **90 °C** | `HW Thermal Slowdown` active, `SW Power Capping` 69 s |
| capped + guard, 32K→128K sweep | 81 °C | none over 269 samples, 0 interventions |
| capped + guard, 1 h at ~57% duty (`0:23`) | 83.9 °C | none over 61 samples |
| capped + guard, ~74% duty (`0:20`) | **83.5 °C** | `hw_thermal_slowdown` and `sw_power_cap` Not Active; SM clock 2093 MHz |

Duty nearly doubling moved the peak by 0.4 °C. The 83.9 °C figure was therefore not
a duty-limited ceiling, and the Spark-heavy split carries no thermal penalty. GPU
power stays at 41–49 W under load, 9–11 W idle.

### 6.2 The link

10GbE point-to-point: Mac `en0` (10Gbase-T) to Spark `enP7s7` (10000 Mb/s full
duplex), `192.168.2.1` ↔ `192.168.2.2`, RTT 0.94–1.21 ms.

The Spark's `r8127` NIC flaps intermittently — a known DGX Spark issue — and a flap
aborts a run and can leave the host off the network until the interface state is
re-established. EEE is disabled as mitigation and the flap count has been **4 across
every session measured**, including an hour of sustained prefill at ~74% duty and a
32-minute 479K ingest. One NIC transmit error total; zero receive errors. The
remaining untested item is the deliberate ablation — EEE back on under the same load
— so the counterfactual rests on the pre-change baseline rather than a controlled
comparison (Appendix A).

**No tunnel is required.** It was, when measured on 2026-09-19, because macOS
Application Firewall blocked inbound connections for adhoc-signed binaries. On
macOS 26.7 the same binaries accept non-loopback connections and the pair runs on
the link addresses directly. The tunnel kit is retained as a documented fallback
rather than deleted, because the original block was real when measured.

---

## 7. Verification status

Against the plan's acceptance criteria:

| # | Criterion | Status |
| --- | --- | --- |
| 1 | Cross-machine oracle: pipeline vs single-host logits | **met 2026-09-21** — greedy continuation byte-identical over 290 bytes (~200 tokens); argmax equal, top-8 7/8, top-16 15/16, mean \|Δ\| 0.307 (`ds4-glm53-oracle.md`) |
| 2 | Boundary gates (2048→2056, 4096→4100) | **met 2026-09-21** — 2 891- and 5 018-token prompts cross both with clean output and agreeing logits (`ds4-glm53-oracle.md`) |
| 3 | ≥150 t/s prefill and ≥10 t/s decode at 32K | **met**, with margin on prefill (389.0 t/s) and 2–3% on decode (10.2–10.35 t/s) |
| 4 | Long-context capacity: 262K ingest, 524K allocation | **met** — ~287K and ~479K cold ingests at ctx 524288, all weights resident |
| 5 | Endurance with peak board logged per frontier | **met for one session** (~1 h, 61 samples, peak 83.5 °C, no throttling) — not a soak |
| 6 | Distributed snapshot round-trip | **half met** — save verified, load path exercised on a live pair; fresh-pair and roles-swapped restore not run |

**Independently verified:**

- **Slice-payload correctness.** A loopback parity test — two processes on the Mac
  over the split boundary, Q2, 48 greedy tokens against a single-host run — produced
  **byte-identical** output. Any mHC-carry error diverges immediately, so this
  validates the wire width end to end.
- **Q4_K kernels.** Logits compared against the Metal reference and against the
  alternative CUDA paths; the corrected tile8 path is byte-identical to the warp
  path and agrees to mean |Δ| 0.109 over the reference's top-16.
- **The swiglu clamp.** Plumbing it moved logits *closer* to the reference
  (top-16 15/16 → 16/16), i.e. it had been binding and the earlier agreement was
  masking it.
- **A regression test that fails pre-fix.** `make test-glm53-moe-q4k` builds a
  synthetic 288-expert MoE through the public API, fills `mid` with NaN, and asserts
  no selected pair is left unwritten — with a `selected` list containing expert 287,
  expert 256, expert 255, a negative slot and an unselected expert. Restoring the
  old 256-expert grid makes it fail with `57 selected pairs had no mid row written`.
- **Functional end to end**, on the recommended configuration: `/v1/models` serves
  all three ids at context 524288, and a nonce-instructed chat request returned
  exactly `ACK-c55df5`.

**Distributed snapshot (criterion 6, half met).** The save is verified and the load
path was exercised on a live pair: a 649-token cold prompt wrote `bf072cd0…kv`
(165.08 MiB, `save=18.2 ms`) with the data connections observed on
`127.0.0.1:55911`, and re-sending the prompt reported `cached_tokens: 819,
cache_write_tokens: 0` with `cache hit … load=126.8 ms`. What is *not* shown is
equivalence — the same output after a restore on a **fresh** pair — and that is the
remaining half. Getting there required fixing two defects (§8): the worker advertised
an ephemeral data port until pinned with `--listen`, and a KDA-layer sizing guard made
any slice containing a KDA layer unsizeable as soon as a context existed.

**Now verified, 2026-09-21.** The cross-machine oracle (criterion 1) was run: the
pipeline's greedy continuation is byte-identical to a single-Mac Q4_K
`--ssd-streaming` run over 290 bytes (~200 tokens) on a 2,891-token prompt that
crosses the dense/indexed boundary, and the full logits agree on argmax with
top-8 7/8, top-16 15/16 and mean |Δ| 0.307 — cross-backend numeric drift, not a
structural divergence (`ds4-glm53-oracle.md`). Getting there surfaced and fixed a
single-Mac `--ssd-streaming` generation regression from the Q4_K generic-dispatch
promotion (`ds4-glm53-ssd-streaming-regression.md`). `--dist-replay-check` was
also found unwired on the coordinator path and is now implemented in the CLI dump
path; it passes exact on the pair.

---

## 8. Defects found and their disposition

| Defect | Evidence | Disposition |
| --- | --- | --- |
| GLM 5.3 slice wire width: `N_EMBD` where the tape uses `N_HC × N_EMBD` | three tape sites plus git history showing the contract was never updated for 5.3 | **fixed** — bit-exact loopback parity |
| CUDA GLM routed MoE accepted only Q2_K | `ds4_cuda.cu:32103`; `unsupported types 12/12/12` from both roles | **fixed** — Q4_K path ported, and the generic dispatch promoted as default |
| 256-expert bound against a 288-expert model | the literal `256u` in the expert map, grids, sizing and weight lookaheads | **fixed** at all nine sites; regression test fails pre-fix |
| Swiglu clamp missing from the CUDA GLM MoE signature | header and caller both passed it; the CUDA definition did not, working only by AArch64 ABI accident | **fixed** — three signature mismatches found and corrected |
| Metal kernel sources resolved only from the working directory | `metal/…` and `./metal/…` candidates only | **fixed** — executable-relative lookup; installed binaries work from any directory |
| GLM 5.3 KDA layers made any containing slice unsizeable | `distributed KV shard tensor size overflow`; a uniform `compact_live` was rejected for KDA layers, which hold no KV rows | **fixed** — guard removed, span asserted non-zero; snapshot round-trip then worked |
| Spark hard-locks under sustained load | board 90 °C, `HW_THERMAL_SLOWDOWN`, NVIDIA PowerStress failure | **mitigated** — caps and governor; RMA declined. A caps-armed run has not tripped |
| guard `set -u` exit on first `set_cap`; slow-down field off by one; initial `board=0C` | journal `board: parameter not set`, exit 2 | **fixed**, and stub-verified |
| Install hang: `plymouth-quit-wait` stalling the boot for 17 min | `is-system-running = starting` | **fixed** — unit reordered to `After=local-fs.target`; default target `multi-user` |
| macOS ALF blocks inbound for adhoc-signed binaries | a 20-line `cc` listener failed identically while Apple-signed `nc` worked | **worked around**, then made moot — no tunnel needed on macOS 26.7 |
| CUDA coordinator-side slice prefill crashes on chunks ≥512 rows | reported on pristine `8db1d1d` | **not reproducible 2026-09-21** — roles-swapped Q4_K/Q2 prefill works at 512/4 096-row chunks; the GLM-specific CUDA path it targeted is unreachable for shipped models (`ds4-glm53-ws10-role-swap.md`) |
| MTP tok2 and scalar debug paths still refuse Q4_K by name | deliberate | **open** — tok2 needs a GLM MTP support model to verify against, and verification is the point of the port |

---

## 9. Operating notes

**Starting the pair.** Worker first (it retries until the coordinator's control port
answers, so ordering is convenience not requirement), then the coordinator. The
coordinator is ready when `/v1/models` returns the model — not when the port opens,
which happens before the weights are mapped.

**Launching on the Mac needs a supervisor.** A bare `nohup … &` from a
non-interactive or backgrounded shell does not survive; use a terminal, a launchd
job, or `hub start`. Note that **macOS has no `setsid`** — invoking it fails silently
and the process never starts, which is a confusing way to lose twenty minutes.

**`ssh` inside a backgrounded job hangs** unless given `-n`: it inherits the job's
stdin pipe and waits on it. Several harness scripts stalled at their first remote
call because of this, leaving the coordinator unlaunched.

**Is it actually working?** `ps -o pcpu` reads 0.0% while the model is prefilling,
because the work is on the GPU. Check that the log is growing, or that the
coordinator/worker GPU is busy — not the process CPU.

**Measuring.** One request in flight at a time. A killed HTTP client does **not**
cancel the work it started: both stages stay busy draining in-flight chunks for
~40 s, and a second measurement started in that window is contended. The same
applies to the session cache — resubmitting an identical prompt can return in
0.191 s, so any prefill figure needs `cached_tokens` beside it.

**A failed prefill costs the whole transcript.** Any failure in the pipelined
prefill tears down the first-hop connection — including a coordinator-local one —
which drops the workers' per-session KV; the next request then replays the full
token prefix (~11.5 min at 287K). There is no partial recovery, by design.

**Checking whether a split fits.** Read the guard line in each log rather than
estimating; a plan is admitted or refused as a whole.

### Quick acceptance commands

```sh
# link health — the r8127 flaps, and a flap aborts a run
ssh -n 192.168.2.2 'ethtool --show-eee enP7s7 | grep "EEE status"'      # expect: disabled
ssh -n 192.168.2.2 'journalctl -b 0 --no-pager | grep -c "r8127: enP7s7: link down"'

# thermal protection (Spark)
ssh -n 192.168.2.2 'systemctl is-active gb10-thermal-guard.service gb10-cpu-cap.service'
ssh -n 192.168.2.2 '/usr/local/bin/gb10-thermal-status.sh | head -20'
ssh -n 192.168.2.2 'nvidia-smi --query-gpu=clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.sw_power_cap --format=csv,noheader'

# the served surface
curl -s http://192.168.2.1:8081/v1/models | grep -o 'glm-5.3-flash[a-z-]*' | sort -u

# memory admission actually granted (compare against §3.2)
grep -E "memory:|restricting" /path/to/coordinator.log | tail -2
ssh -n 192.168.2.2 'grep -E "memory:|restricting" /tmp/worker.log | tail -2'

# a real completion through the pipeline, before trusting any number
curl -s http://192.168.2.1:8081/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"Reply with exactly ACK"}],"max_tokens":16,"temperature":0}'

# long one-shot runs: detach from the terminal
#   a `nohup … &` started from an agent/SSH PTY can leave ds4 blocked on
#   /dev/ttys000 with the model unloaded and the log frozen. Feeding the prompt on
#   stdin from a file works, as do `ssh -T localhost` and a launchd job.

# distributed snapshot (save + load), see Appendix B phase J–K
~/bin/ds4-server -m ~/mlmodels/glm/GLM-5.3-Flash-Q4_K.gguf --role coordinator --layers 0:20 \
  --listen 192.168.2.1 9911 --ctx 4096 --host 127.0.0.1 --port 18080 \
  --kv-disk-dir /tmp/ds4kv --kv-cache-min-tokens 512
#   a cold prompt above min-tokens forces a save; re-sending it must report
#   cached_tokens == prompt_tokens and no cache_write_tokens
```

---

## 10. Source changes made to run this

All in the working tree of the `customisation` branch; the first three are what make
GLM 5.3 pipeline mode possible at all.

| # | Anchor | Change | Why |
| --- | --- | --- | --- |
| 1 | `ds4.c:71777` `ds4_engine_hidden_f32_values` | return `N_HC * N_EMBD` for `ds4_model_is_glm53()`, `N_EMBD` otherwise | the function described GLM 5.2's single stream while GLM 5.3's mHC block carries four, so every wire and buffer size was 4× too small |
| 2 | `ds4.c:74441`, GLM branch of `ds4_session_eval_layer_slice` | `hidden_dim = glm53 ? N_HC*N_EMBD : N_EMBD` | per-token chunk stride across a slice boundary |
| 3 | `ds4.c:73521` `ds4_session_eval_output_head_from_hc` | write into `gg->hc_cur` for glm53 | the head collapses the HC block; writing `gg->cur` left it stale |
| 4 | `ds4_metal.m:4754`, `:4821` | new `ds4_metal_executable_dir()`; search `<exe_dir>/metal/…` as well as `metal/…` and `./metal/…` | kernel sources were CWD-relative, so an installed binary only worked when started from a directory containing `metal/` |
| 5 | `ds4.c:61000` `glm_layer_payload_tensor_bytes`, KDA branch | stop rejecting the header's uniform `compact_live` counts; return the conv + recurrent state span and assert non-zero | the guard made any slice containing a KDA layer unsizeable as soon as a context existed |
| 6 | `ds4.c:46076` `glm_graph_layer_uses_generic_routed_moe` | the Q4_K branch is unconditional | the Q4_K-specific kernels were the default and measured **2.7× slower** than the pre-existing generic dispatch at identical output, so the generic one was promoted; the ported kernels are now reached only from `make test-glm53-moe-q4k`, because `DS4_CUDA_GLM_MOE_TYPES` is read inside the GLM-specific dispatch and cannot re-route a homogeneous Q4_K trio to it |
| 7 | `ds4_server.c:14951` `send_models`, GLM branch | derive ids from `server_model_id_from_engine()` plus `-chat` / `-reasoner` | the branch keys on the GLM family, which covers 5.2 and 5.3 alike, so a 5.3 model was advertised as `glm-5.2*` |

The CUDA GLM MoE port itself (type predicate, Q4_K prefill/decode/expert-major
kernels, the expert-count and swiglu-clamp fixes) is a larger body of work in
`ds4_cuda.cu`, summarised in §8 and detailed in git history.

**Host-level, outside the repository:** the Spark thermal-protection kit, the launchd
tunnel job (now optional), and `~/ds4-deploy.sh`. All are backed up on the Mac.

---

## 11. Environment and artifacts

### Mac Studio M4 Max

| Path | What |
| --- | --- |
| `~/dev/ds4/` | source tree, branch `customisation`, fork point `8db1d1d`, remote `kunle12/ds4` |
| `~/dev/ds4/docs/custom/ds4-glm53-q4-split-design.md` | the plan and its workstreams |
| `~/dev/ds4/docs/custom/ds4-glm53-q4-implementation-log.md` | this document |
| `~/bin/` | `ds4`, `ds4-server`, `ds4-agent`, `ds4-bench`, `ds4-eval` + `metal/` (26 kernels) |
| `~/bin/llm_config.json` | the launcher's model list, carrying the recommended split |
| `~/ds4-deploy.sh` | rebuild + install to `~/bin` on either host; `check` verifies source parity (md5) and prints both hosts' binaries |
| `~/ds4-tunnel/` | tunnel kit — **no longer required**, retained as a fallback, README says so |
| `~/thermal-protect/` | backup of the Spark protection kit |
| `~/Library/Logs/ds4-tunnel.log` | tunnel log, silent while the peer answers |

### DGX Spark GB10

| Path | What |
| --- | --- |
| `~/dev/ds4/` | a real checkout of upstream `main` at `8db1d1d` (the fork point), with the local work applied as uncommitted changes — 12 modified files plus `docs/custom/`, `docs/NETWORK_SETUP.md` and the WS-5 test. `~/ds4-deploy.sh` keeps its *content* matched to the Mac's tree rather than its git state; `ds4.c` md5 is identical on both hosts (`0ce1a1b7…`) |
| `~/bin/` | the same five binaries, built with `make -j20 cuda-spark` (`sm_121`) |
| `~/mlmodels/glm/` | Q2 + Q4_K GGUFs and the vision encoder |
| `~/thermal-protect/` | protection kit source |
| `/usr/local/bin/gb10-thermal-{guard,cpu-cap,status}.sh` | installed |
| `/etc/systemd/system/gb10-thermal-guard.service`, `gb10-cpu-cap.{service,timer}` | installed and enabled |

Services: `gb10-thermal-guard` (active, `Restart=always`), `gb10-cpu-cap` + timer
(re-applies the CPU cap every 5 min). Default target `multi-user` (headless).

---

## Appendix A. Measurements still open

* **The cross-machine oracle** (criterion 1) — pipeline against single-host logits.
  The one correctness gate not yet run; `--dist-replay-check` already exists in the
  tree.
* **Boundary gates** (criterion 2) — 2 048→2 056 and 4 096→4 100, where the
  dense-to-sparse attention boundary sits.
* **Fresh-pair and roles-swapped snapshot restore** (criterion 6) — the save and the
  load path are verified, equivalence on a fresh pair is not.
* **An EEE-on/off ablation on the link** — the mitigation holds under load (flap
  count 4 across every session), but the counterfactual was never run.
* **The greedy-divergence gap** — the 64-token divergence is asserted to be a
  near-tie and has never been measured.
* **MTP under a layer split** — excluded by design and measured as a loss on one
  machine; a full implementation would need the head and its routing on both sides.

---

## Appendix B. Corrections, and the reasoning errors behind them

Each entry is either a claim that was published and turned out wrong, or a reasoning
error worth not repeating. They are kept because several cost hours.

### Measurement conventions adopted, because these failed more than once

1. **Sample twice before stating a rate.** The SSD-streaming cost was reported as a
   flat ~2.5 min/case from one cold window; across two later windows it is 25–30 s
   once the cache warms.
2. **Report `cached_tokens` beside every prefill figure.** A repeated 39 846-token
   prefill returns in 0.191 s from session cache — an apparent 208 617 t/s that is
   not a prefill.
3. **Do not derive a rate from a near-zero interval.** The engine's own per-chunk
   telemetry printed `chunk=170666712.37 t/s` once when the divisor collapsed.
4. **Compare like with like.** "Mean |Δ| 0.085 → 0.0727" set a vocabulary mean
   against a top-16 mean; like for like it is 0.1085 → 0.0727.
5. **Never quote two numbers whose bases differ.** The MTP decision was briefly
   argued from Q2-pair-at-32K against Q4-single-at-262K: different model, different
   depth, so it supported nothing.

### Reasoning errors, and what would have caught them

6. **A bit-identity proves nothing unless the runs took different paths.** This was
   the pivotal wrong step in the Q4_K work: "Q2_K tile8 ≡ Q2_K warp, byte-identical"
   was read as proof that the tile8 machinery was sound. It showed nothing, and the
   fault lay in *shared* code — the expert map — which can hit one path and not the
   other. Comparing the inputs each path sees, or printing which path each took,
   would have found it immediately; that trace now exists (`DS4_GLM_MOE_TRACE=1`)
   and shows prefill takes tile8 and decode takes warp.
7. **Do not generalise from a single observation.** Twice: the SSD rate above, and
   inferring memory exhaustion from one unreachable host.
8. **A cause asserted but never measured stays asserted.** The 64-token greedy
   divergence is still described as a near-tie and was never measured.

### Claims published and withdrawn

9. The Spark going offline was **thermal** — then a later outage was a **link flap**,
   not memory either, and the memory arithmetic in that incident was wrong.
10. `ds4-server` does **not** lack `--chdir`; it is parsed, just undocumented in
    `--help`.
11. The Spark does **not** need `ufw` rules in the Mac-leads direction.
12. `gb10-thermal-status.sh` has **no** defect; my `sed` range was truncating its
    output.
13. The tunnel was **not** failing from data volume; the Spark's userland was wedged.
14. "The Spark cannot hold more than ~22 layers" — **withdrawn**; the experiment that
    would have tested it was invalidated by a link flap.
15. "The tunnel is required" — **no longer**; macOS 26.7 accepts these binaries on
    non-loopback addresses.
16. "The balance point is unreachable at any context" — **wrong**. It treated the
    Spark's 103.63 GiB guard budget as hardware when the reserve is a runtime
    setting, and the Mac's own 115.19 GiB is itself the product of such an override
    (§3.2). The pushback that caught it was the observation that both machines are
    sold as 128 GB.

### Phase index

Provenance for anything above; the narrative itself is in git history.

| Phase | Subject |
| --- | --- |
| A | Analysis: GGUF, `--inspect`, per-slice admission, the 10GbE link |
| B | The slice wire-width defect: found, traced, fixed, validated bit-exact |
| C | First cross-machine attempt; macOS ALF; the Q4 blocker on CUDA |
| D | Q4 measured where it could run: the Mac alone with `--ssd-streaming` |
| E | Spark thermal protection: the incident, the kit, the capped sweep |
| F | Deployment: binaries, the `--chdir` workaround, the tunnel kit |
| G–I | Plan, repository organisation, fork and remote workflow |
| J–K | Wedge and recovery; distributed snapshot; rebuild and publication |
| L–P | The Q4_K CUDA port: predicate, kernels, the expert-count bug, expert-major, the clamp |
| Q | Self-audit: the fallacies and tooling failures recorded above |
| R–X | Q2 regression, dispatch A/B, criterion 3, load balance, the tunnel leaving the operational path, both server configs, the served model id |
| Y | End-to-end acceptance on the live pair; the depth comparison in §4.4 |
| Z | Which stage is saturated, and the measured traffic direction |
| AA | Correction: the Spark's guard budget is a setting, not a limit |
| AB | Split sweep, the recommended configuration, and the thermal result |
