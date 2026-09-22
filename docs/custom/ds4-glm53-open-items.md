# Remaining GLM 5.3 items — disposition (2026-09-22)

The items left after criterion 1/2/6, the E1 switch, WS10 and the agent workload.
Each is either done, characterized, decided, or deferred with a reason.

## Done

### Thermal soak repeat
Second cold 286K ingest on the pair (E1 default, same conditions as the first):

| | value |
| --- | ---: |
| prompt tokens | 285,159 |
| prefill | 413.4 t/s |
| decode (median gap) | 107.4 ms |
| output sha256 | `a3898c57…` (identical) |
| Spark peak | **75 °C** |
| throttle samples | 0 (`hw_thermal_slowdown`/`sw_power_cap` Not Active ×36) |

The documented hour-long ~74 % duty soak peaked 83.5 °C; this shorter ingest
repeats cleanly and confirms the earlier no-throttling result was not a one-off
for the ingest workload.

### Single-host streaming QA gate
Added to `QA_BEFORE_RELEASES.md` (GLM 5.3 section): any routed-MoE
**dispatch-selection** change must be measured on a single-host `--ssd-streaming`
**generation** run as well as on the resident pair, with the `ce4d214` regression
as the worked example. A pair-only A/B is what let that through, and
prefill-only checks (`--dump-logits`) would not have caught it either.

## Characterized

### Token-id equivalence across paths
Observed: completion token **counts** differ while the text is byte-identical —
E1 off/on on the pair (80 vs 101 at 285K) and across topologies (136 vs 118 for
the same snapshot prompt). Identical text with differing counts is a near-tie
argmax/tokenisation flip caused by small cross-path float differences, not a
divergence; the consumer parses text (tool syntax, stop strings), so behaviour is
unaffected. A strict token-**id sequence** comparison is not currently possible:
the CLI `--dump-logprobs` is single-position top-k, and the agent `--trace` (which
does log ids) embeds a session timestamp in its system prompt, so two runs are not
prompt-identical. Recorded, no defect.

## Decisions

### Ported Q4_K kernels — keep as test-only coverage
They lose to the generic dispatch on a resident graph (95.3 vs 258.9 t/s) and are
unreachable at runtime for a homogeneous Q4_K trio; they remain exercised by
`make test-glm53-moe-q4k`. Re-exposing them at runtime would restore an
unexamined second dispatch path. Revisit only if a future recipe cannot use the
generic path.

### Streaming-aware generic dispatch — not implemented
Making the generic entry points read experts from the streaming cache would touch
code shared by every model — a design change, not a guard — and the streaming
route is I/O-bound (82–85 t/s against the pair's resident 415 t/s), so the 2.7×
kernel advantage likely does not transfer. The streaming guard is the correct
minimal state. (Design item recorded in
`ds4-glm53-ssd-streaming-regression.md` §6.)

## Deferred (cost or risk, not blockers)

### Single-Mac 512K and MTP-at-512K re-verify
The old numbers (decode 4.63–5.02 t/s; "MTP a ~21 % loss at 512K") predate the
`ce4d214` fix. A `ds4-bench` re-run (frontier 524288, ~1.8 h per configuration)
**did not complete**: the first attempt stopped correctly for a short prompt
(506,160 vs 524,288 tokens), and the re-launch got through the memory plan — which
reproduced the documented **102.00 GiB** exactly — then died silently during the
first 2048-token prefill chunk, with no error line, no jetsam/watchdog entry and
the process gone. So the 512K decode figure stays **unverified, and the single-Mac
Q4_K route at 512K may not run on this build at all**; that needs its own
investigation. MTP-at-512K also stays deferred: `ds4-bench` has no GLM `--mtp`, so
it needs a CLI run, and MTP is a non-goal under the split. The 32K/262K single-Mac
figures have already been re-measured and match
(`ds4-glm53-ssd-streaming-regression.md` §5).

### EEE-on/off link ablation
The mitigation (EEE disabled) holds under load — 4 flaps across every session.
The counterfactual needs EEE re-enabled on the Spark under sustained load, which
risks the r8127 flap that aborts runs and can leave the host off the network.
Deferred as an infrastructure risk for a low-value result.
