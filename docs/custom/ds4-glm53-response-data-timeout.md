# `Timed out waiting for response data` — the GLM 5.3 pipeline under the code agent

**Status:** root cause identified and **fixed**. The bounded result wait and
the keepalive tuning are implemented at `c2787ea`, deployed to `~/bin` on both
hosts, and stall-verified on the live pair (a worker `SIGSTOP` mid-decode now
fails the request at the 120 s deadline, the worker re-dials on resume, and a
fresh request succeeds — previously the same freeze hung indefinitely).
Measured evidence: `ds4-glm53-ws0-headmove-results.md` §2. §5's items 1–2 are
done; the rest of that section stands as written for the future.
**Symptom:** occasionally, the code agent (OpenCode) against the Mac+Spark
GLM 5.3 Flash pair aborts a request with `Timed out waiting for response data`.

---

## 1. What the error actually is

The string is OpenCode's, not ds4's — it is the **per-chunk idle timeout** of the
AI SDK's HTTP transport (`packages/ai/src/route/transport/http.ts` in
`anomalyco/opencode`):

```ts
RequestExecutor.responseStream(response).pipe(
  Stream.timeoutOrElse({
    duration: timeoutDuration(request.http?.chunkTimeout),
    orElse: () => Stream.fail(timeout("read", "Timed out waiting for response data", http)),
  }),
```

* It arms **after the response headers arrive**, and fires when the response
  byte stream delivers **no bytes at all** for the timeout window. SSE comment
  lines (`: prefill`) count as bytes — the timeout wraps the raw stream before
  SSE framing.
* The window defaults to `DEFAULT_HTTP_TIMEOUT_MS = 300_000` — **5 minutes**
  (`packages/ai/src/schema/options.ts`). Per-provider overrides exist
  (`http.chunkTimeout`, `http.headerTimeout`, `http.timeout`); `false` disables.
* Its sibling "Timed out waiting for response **headers**" is a different
  failure (request queued before first output) and is *not* the one seen here.

So the question is precise: **what makes the pair's HTTP response go silent for
more than 5 minutes mid-request?** On a healthy pipeline the answer is
"nothing": prefill emits SSE keepalives (below), decode emits a delta every
~106 ms. The gaps that trip a 5-minute timer require a *stall*, and the
distributed layer has one by construction.

## 2. What the pair promises when healthy

* **Prefill:** `server_progress_cb` (`ds4_server.c:12694`) writes SSE headers on
  the first chunk-completion callback and then a `: prefill` comment per
  callback, spaced ≥ 5 s (`:12717`). Because the emission piggybacks on chunk
  completions, the real cadence is the chunk period — ~10 s at depth
  (4096 tokens at ~415 t/s). Worst healthy gap: ~10 s, 30× under the timeout.
  This is also why the server's own comment says the mechanism exists: "so
  HTTP/TCP idle timeouts on the client side don't close the connection while
  the server is busy doing prefill".
* **Decode:** content / reasoning / tool-argument deltas flow per token
  (`openai_sse_stream_update`, `ds4_server.c:8035`; tool-call arguments are
  streamed live via `sse_chat_tool_call_args_delta_n` — the DSML tracker holds
  only partial split markers, `:7754`). Worst healthy gap: ~0.2 s.
* **KV snapshot save/load** runs on separate data connections and takes tens
  of seconds at depth — never minutes on the response path.

A healthy pair cannot trip this error. The observation "occasionally" is the
signature of an *event*, not a cadence.

## 3. The mechanism: the coordinator's result read has no deadline

Every decode token and every prefill chunk ends with the coordinator blocking
for the worker's result frame: `dist_recv_result_alloc` → `dist_read_full` → a
plain blocking `recv` loop (`ds4_distributed.c:1096-1109`). The socket setup
(`dist_set_socket_low_latency`, `:1027-1070`) installs:

* `TCP_NODELAY`, buffers, `SO_SNDTIMEO` = **60 s** — sends are bounded;
* `SO_KEEPALIVE` — with **OS-default parameters**, i.e. the first probe only
  after ~2 h idle (`net.inet.tcp.keepidle` / `tcp_keepalive_time` = 7200 s);
* **no receive timeout, by default, deliberately** (`:1046-1048`):

> *Do not install a receive timeout by default. Worker control sockets can be
> legitimately idle while a KV snapshot is transferred on a separate data
> connection; timing out that read drops an otherwise healthy route.*

(`DS4_DIST_SOCKET_RECV_TIMEOUT_SEC` can opt in — see §5 for why that env is the
wrong knob for this failure.)

The consequence: when the worker side **stalls without closing the TCP
connection**, the in-flight request freezes *mid-stream* — prefill progress
stops (so no keepalives), decode stops (so no deltas), and the coordinator
blocks in `recv` for anywhere from ~8 minutes (TCP retransmission exhaustion,
if the request bytes were unacked) to hours (socket idle from the kernel's
view, keepalive not due for 2 h). OpenCode's timer fires first, at exactly
300 s: **`Timed out waiting for response data`**.

Note the contrast that confirms the diagnosis: a worker that **dies cleanly**
sends FIN/RST, the read errors immediately, and the request takes the *fast*
failure path — route teardown → `dist_coordinator_rebuild_from_transcript`
(`:2948`) → either a transparent replay (stream stays alive via the same
keepalives) or a prompt SSE error. That path produces visible errors or slow
responses, **never this client-side timeout**. Only the *silent-stall* class
of failure produces 5 minutes of dead air.

## 4. Why the stall happens on this pair (recorded events)

The pair's own logs document exactly this class:

| event | evidence | stall signature |
| --- | --- | --- |
| **r8127 NIC link flap** | impl-log §6.2: "flaps intermittently — a known DGX Spark issue — and a flap aborts the run"; 4 flaps across every measured session | link drops mid-request without FIN/RST reaching the peer; with ~2 h keepalive defaults the socket hangs far past 300 s |
| **Spark wedge / silent hard-lock** | impl-log §8 and Appendix B: "silent hard-lock with no panic or OOM", "the box resets ~19 min later with the watchdog never armed", "the Spark's userland was wedged" | a frozen-but-alive worker answers TCP probes; the socket looks healthy; `recv` blocks until the box resets (~19 min ≫ 300 s) |
| **worker paused under memory pressure / driver stall** | the GB10's shared CPU+GPU memory; the guard's `TARGET_KILL` backstop | same signature as the wedge |

So: occasionally one of the pair's known infra events freezes the worker side
without a clean close; the coordinator has no deadline on the result read; the
agent's chunk timer is the only thing in the system that eventually fires.

**Why failures cluster when they do.** After OpenCode aborts at 300 s, the
server only notices at its next write (keepalive or delta fails, or the client
FIN is seen in the token loop); the work keeps draining "for ~40 s"
(impl-log §9), and — if the worker had actually gone — the retry either queues
behind that drain (header-timeout territory) or meets a dead route and pays a
full transcript replay (~11.5 min at 287K) before first output. One 5-minute
freeze therefore tends to be followed by a slow or failed retry.

## 5. What to do about it

1. **Engine fix (small, recommended): give the result wait a request-scoped
   deadline.** Not a socket timeout — a bounded `poll()` around the
   *result-frame* read in `dist_recv_result_alloc` (and the prefill result
   reader's per-chunk read), armed per request and torn down after. The
   no-recv-timeout comment's concern does not apply here: the coordinator reads
   the work socket only while a result is outstanding, and the longest
   legitimate wait is one deep prefill chunk (~10–20 s) — a default of ~120 s
   is 6× margin. On expiry the span fails, and the existing machinery does the
   right thing: route teardown → rebuild/replay or prompt SSE error — visible,
   retryable, and bounded, instead of a silent 5-minute hang. The worker needs
   nothing: its upstream send is already bounded by the 60 s `SO_SNDTIMEO`,
   and its read loop idles between requests for potentially hours, which is
   exactly why the deadline must be request-scoped rather than on the socket.
2. **Complement: aggressive TCP keepalive on dist sockets** (`TCP_KEEPALIVE` /
   `TCP_KEEPIDLE` ≈ 30 s, interval 10 s, count 3). This detects a *dead peer*
   within ~1 minute **without** disturbing legitimately idle connections (the
   OS answers probes for a healthy socket), closing the wedge/link-flap
   detection gap that the 2 h OS defaults leave open. Cheap, standard, and
   orthogonal to (1).
3. **Do not** set `DS4_DIST_SOCKET_RECV_TIMEOUT_SEC` globally for this: it is
   a blanket `SO_RCVTIMEO` on every dist socket, so it would tear down the
   worker's read loop every N idle minutes between requests — the route churn
   it causes is worse than the hang it cures. The env stays what its comment
   says: a diagnostic for specific cases.
4. **Agent-side palliative (config-only, today):** raise the ds4 provider's
   `http.chunkTimeout` in OpenCode (e.g. 600 000–900 000) so an occasional
   slow recovery (replay after a flap) does not abort an otherwise-alive
   stream. This masks the symptom rather than fixing the stall, but it buys
   tolerance while (1) is implemented. Keep `http.headerTimeout` at its
   default, since queueing/replay should fail fast there.
5. **Diagnosis when it next happens:** the 300 s gap is a fingerprint — check
   the Spark's `journalctl -k | grep "r8127: enP7s7: link down"` and the guard
   status first, as the operating notes already say; then the coordinator log
   for how long after the abort the request actually errored (the delta between
   the client's 300 s and the server's eventual error tells wedge vs link-flap:
   minutes = retransmission exhaustion, hours = idle-socket freeze).

## 6. Ruled out

* **Keepalive cadence** — worst healthy gap ~10 s at depth, 30× under the
  timeout; and the keepalives demonstrably work (routine 11.5-min ingests
  complete against the same 300 s timer).
* **KV snapshot save/load** — tens of seconds at depth, on separate data
  connections, off the response path.
* **Tool-call suppression** — the OpenAI chat path streams tool-call argument
  deltas live; only the Responses/Codex protocol withholds them
  (`ds4_server.c:8751-8754`), and OpenCode does not use that protocol here.
* **Queueing behind another request** — that stalls *before* headers and
  produces the sibling "waiting for response **headers**" timeout, a different
  message.
