# E1 decode graph A/B — GLM 5.3 Q4_K on Mac + Spark

**Question.** Does the GLM slice single-token decode fast path
(`DS4_GLM_LAYER_SLICE_TOKEN_DECODE=1`, the switch added in the review change set)
beat the default, which routes a one-token distributed step through the indexed
batch graph?

**Answer.** Yes, at every depth tested, with identical output text:
**+7.7 % decode at ~11 K, +15.4 % at ~285 K, +19.6 % at ~473 K.** Prefill is
unchanged. No throttling; Spark peak 74–76 °C.

**Verdict.** The switch is a real win and the gain grows with depth, which is the
workload that matters. Recommend enabling it for the GLM pair. The full
cross-machine oracle (criterion 1) is still the separate gate before making it a
hard default in the engine; enabling it in the launcher config is reversible and
needs no code change.

---

## 1. Method

Pair: Mac Studio M4 Max coordinator (`--layers 0:20`) + DGX Spark GB10 worker
(`--layers 21:output`), ctx 524288, `DS4_GLM_MEMORY_GUARD_RESERVE_GB=14`.
Both hosts ran the `code review` commit (`1d066f2`); source md5 matched
(`ds4.c 3d00b907…`) and both binaries were rebuilt for their own backend
(installed 2026-09-21 17:49 Mac / 17:52 Spark).

Each configuration is a fresh pair of processes (the env is process-level), so
every number is a **cold** prefill with `cached_tokens: 0`. Prompt is a fixed
nonce plus a slice of `speed-bench/promessi_sposi.txt`. A single streaming chat
request per run; the completion is `temperature 0` and produces the digits
`0…39`, so all runs emit the same 109-character text.

- Prefill = time to first content chunk (TTFT), tokens/s over `prompt_tokens`.
- Decode = median inter-chunk gap in ms. All runs deliver **79 content chunks**
  of the same text, so the gaps are directly comparable; the total wall time over
  the identical output is reported as a second estimator.
- Correctness = SHA-256 of the full completion text, plus the token counts when
  they agree.

Scripts: `speed-bench/e1-decode-ab/e1_measure.py` (HTTP client) and the driver
described in §5. Spark board temperature sampled every 20 s on the worker.

## 2. Results

| run | prompt tk | prefill | decode median gap | decode wall (identical output) | text sha256 |
| --- | ---: | ---: | ---: | ---: | --- |
| 8K off | 11,478 | 380.63 t/s | 100.90 ms | 7.914 s | `a3898c57…` |
| 8K on | 11,478 | 381.80 t/s | **93.70 ms** | **7.349 s** | `a3898c57…` |
| 286K off | 285,161 | 414.50 t/s | 122.80 ms | 9.618 s | `a3898c57…` |
| 286K on | 285,161 | 414.81 t/s | **106.45 ms** | **8.349 s** | `a3898c57…` |
| 479K off | 472,975 | 376.70 t/s | 137.51 ms | 10.786 s | `a3898c57…` |
| 479K on | 472,975 | 376.80 t/s | **114.94 ms** | **9.020 s** | `a3898c57…` |

Effect, two independent estimators:

| depth | median-gap speed-up | identical-output wall speed-up | prefill |
| ---: | ---: | ---: | ---: |
| ~11 K | +7.7 % | +7.7 % | unchanged |
| ~285 K | +15.4 % | +15.2 % | unchanged |
| ~473 K | +19.6 % | +19.6 % | unchanged |

Gap ranges do not overlap in any pair:

| depth | off min–max | on min–max |
| ---: | ---: | ---: |
| 11 K | 98.68–108.11 ms | 92.20–98.36 ms |
| 285 K | 117.66–132.72 ms | 104.22–111.77 ms |
| 473 K | 133.36–147.77 ms | 110.12–123.53 ms |

The 285 K off figure (**122.80 ms** median; 121.75 ms mean) reproduces the
documented `0:20` baseline of **121.1 ms**, and prefill (**414.50 t/s**)
reproduces the documented **415.0 t/s** — the harness is calibrated against the
log.

## 3. Correctness

- **8 K, identical prompt (`nonce corr8k`), separate processes:** identical
  `prompt_tokens` (11,478), identical `completion_tokens` (93), identical text
  SHA-256. This is a clean like-for-like: same input, same token count, same
  output.
- **285 K and 473 K:** identical text SHA-256. At 473 K the token counts also
  agree (103 each); at 285 K the counts differ (80 vs 101) while the text is
  byte-identical — the two graphs picked different tokens that detokenise to the
  same string, i.e. a near-tie flip of the kind the log already documents for the
  alternative CUDA path, not a divergence in output. The identical text is the
  user-visible correctness result.
- Not run: the full cross-machine logit oracle (criterion 1). The above is
  output-level equivalence at three depths, not the logit-tolerance gate.

## 4. Conditions

| run | Spark peak | throttle samples | coordinator plan | worker plan |
| --- | ---: | ---: | ---: | ---: |
| 286K off/on | 75 / 76 °C | none active | 82.22 GiB / 115.19 | 104.73 GiB / 107.61 |
| 8K off/on | 75 / 74 °C | none active | 82.22 GiB | 104.73 GiB |
| 479K off/on | 75 / 75 °C | none active | 82.22 GiB | 104.73 GiB |

`hw_thermal_slowdown` and `sw_power_cap` were `Not Active` in every sample. The
worker logged the documented `104.73 GiB of 107.61` plan and the coordinator
`82.22 GiB of 115.19`.

The 479 K prompt is `promessi_sposi.txt` repeated to ~1.52 MB (the file alone is
~415 K tokens; measured density ~3.17 bytes/token).

## 5. Reproduction

Start, per configuration (env applied to **both** processes):

```sh
# Spark worker, started first
ssh -n xun@192.168.2.2 "DS4_GLM_LAYER_SLICE_TOKEN_DECODE=<0|1> \
  DS4_GLM_MEMORY_GUARD_RESERVE_GB=14 \
  nohup /home/xun/bin/ds4 --cuda -m /home/xun/mlmodels/glm/GLM-5.3-Flash-Q4_K.gguf \
  --role worker --layers 21:output \
  --coordinator 192.168.2.1 9911 --listen 192.168.2.2 55911 --ctx 524288 \
  </dev/null >/tmp/e1-worker.log 2>&1 &"

# Mac coordinator
DS4_GLM_LAYER_SLICE_TOKEN_DECODE=<0|1> \
  /Users/xun/bin/ds4-server --chdir /Users/xun/dev/ds4 --metal \
  -m /Users/xun/mlmodels/glm/GLM-5.3-Flash-Q4_K.gguf \
  --role coordinator --layers 0:20 --listen 192.168.2.1 9911 \
  --ctx 524288 --host 0.0.0.0 --port 8081
```

Then, once `/v1/models` answers:

```sh
python3 e1_measure.py --url http://192.168.2.1:8081 --model glm-5.3-flash \
  --prompt-file /Users/xun/dev/ds4/speed-bench/promessi_sposi.txt \
  --bytes <37000|916700|1520000> --max-tokens 256 --nonce <unique> --repeats 1
```

Measure one configuration per process; the env cannot be changed at runtime.
Repeated identical requests are **not** served from session cache in this setup
(`live kv cache miss … reason=token-mismatch`), so each sample costs a full cold
prefill.

## 6. Decision — applied 2026-09-21

Both were applied at the owner's direction:

1. **Launcher config**: `DS4_GLM_LAYER_SLICE_TOKEN_DECODE=1` added to the GLM
   `worker_cmd` and `launch_cmd` in `~/bin/llm_config.json` (backup:
   `llm_config.json.pre-e1`).
2. **Engine default**: `ds4_session_eval_layer_slice` now enables the decode graph
   for a single-token GLM 5.3 step on Metal/CUDA without any switch. An explicit
   falsy value (`0`/`false`/`off`/`no`) opts out; ROCm and non-5.3 GLM stay
   opt-in, since neither was measured here.

Criterion 1 (the cross-machine oracle) remains open. It is the gate that would
raise confidence from "output-identical at three depths" to "logit-equivalent to
a single host"; running it is still recommended.

## 7. Post-change verification (default flip)

After the default was flipped and both hosts rebuilt (`ds4.c md5 a8416a4d…` on
both), the switch state was re-checked at ~11 K:

| run | median gap | decode wall | completion tk | text sha256 |
| --- | ---: | ---: | ---: | --- |
| no env (new default) | 92.70 ms | 7.276 s | 93 | `a3898c57…` |
| `DS4_GLM_LAYER_SLICE_TOKEN_DECODE=0` | 99.95 ms | 7.835 s | 93 | `a3898c57…` |
| env-ON reference (§2) | 93.70 ms | 7.349 s | 93 | `a3898c57…` |
| env-OFF reference (§2) | 100.90 ms | 7.914 s | 93 | `a3898c57…` |

The default tracks the ON reference and the explicit `0` tracks the OFF reference,
so the flip and its opt-out both behave as intended. Output is identical in all
four.
