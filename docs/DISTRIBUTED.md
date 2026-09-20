# Inference Across Machines

[README](../README.md)

There are two modes:

| Mode | Split | Main use |
| --- | --- | --- |
| Tensor parallelism | Routed experts and per-layer work across two Macs, or two Sparks for V4.1 Q2 | Resident inference with lower per-token work on each GPU |
| Pipeline parallelism | Complete layer ranges across several machines | Fit larger models and overlap long prefills |

These are separate from [tensor parallelism across CUDA cards](CUDA_MULTI_GPU.md).
Network protocols have no authentication or encryption. Use trusted machines
and a trusted network; run the same commit on every peer. Model paths and
artifacts must agree. Update all TP peers together when changing versions.

## Tensor parallelism between two Macs

This is a 50/50 split with exactly one worker. Do not pass `--layers`.
Routed experts are sharded; attention partitioning depends on the model. Both
GPUs work on the same token and exchange partial results. This can reduce generation
latency, but the gain depends on the model, link, and comparison setup.

Two 128 GB Macs are useful for V4 Flash Q4/MXFP4 or GLM 5.3 Flash Q4.
GLM 5.2 IQ2_XXS is another tested capacity setup. V4.1 Flash Q2 also runs
on two 128 GB Macs, with disk-only Engram tables.
A larger quant may need larger machines even though its tensor layout is supported.

### Link setup

Use a Thunderbolt cable. RDMA requires an active verbs device with an
IPv4-mapped GID; a working ping alone does not establish that.

```sh
rdma_ctl status
ibv_devinfo -v
```

Addresses must be on the cabled member interfaces, not only the Thunderbolt
bridge. For example, after checking which interfaces are active:

```sh
# Machine A, example member interface en1.
sudo ifconfig en1 inet 10.99.0.2/30 alias
# Machine B, example member interface en6.
sudo ifconfig en6 inet 10.99.0.1/30 alias
```

For the large tested shards on otherwise idle 128 GB Macs, the setup raised
the per-boot GPU wired-memory limit on both machines:

```sh
sudo sysctl iogpu.wired_limit_mb=120000
```

This is specific to that memory configuration. It grants a larger GPU budget;
it does not create more RAM. Check model, context, and system headroom before
raising a limit on your machine.

### Start the pair

Download the same model on both machines. For GLM 5.3 Flash Q4:

```sh
./download_model.sh glm53-q4
```

Start the worker first; it retries while the coordinator loads:

```sh
# Machine B.
./ds4 --tensor-parallel --role worker \
  --coordinator 10.99.0.2 9911 --transport rdma --ctx 8192

# Machine A.
./ds4 --tensor-parallel --role coordinator \
  --listen 10.99.0.2 9911 --transport rdma --ctx 8192
```

The verbs device and GID are selected automatically. If ambiguous, specify
`--rdma-device` and `--rdma-gid-index` from `ibv_devinfo`. Use `--transport tcp`
on both peers when RDMA is unavailable. Do not probe the waiting coordinator
with `curl` or `nc`: it may treat the connection as a worker handshake.

Keep workers running in a terminal or managed session and retain both logs.
Do not treat repeated handshake or RDMA timeouts as successful QA merely
because a retry works.

The coordinator can be `ds4`, `ds4-agent`, `ds4-server`, or `ds4-bench`;
workers run `ds4`. For models with vision support, pass the same `--vision FILE`
to both for image input.
For GLM MTP, enable `--mtp` on both. For DeepSeek DSpark, both need the
matching support model and DSpark options. V4.1 supports vision but not
speculative decoding.

TP disk-cache restore currently rebuilds the exact saved token prefix on both
ranks rather than restoring the coordinator alone. Expect prefill on restore.
See [speculation](SPECULATIVE_DECODING.md) and [serving](SERVER.md).

## Tensor parallelism between two Sparks

V4.1 Flash Q2 text inference supports one GPU per rank, with a 50/50 expert
split. Each Spark holds about 81 GiB of weights, plus context and runtime
buffers. Engram tables stay on disk. Do not add `--ssd-streaming` or
`--cuda-tensor-parallel`: those select different memory/execution modes.

Build the same commit with `make cuda-spark` on both machines and download
`ds41f-q2` on both. RDMA needs the libibverbs development headers at build time,
its runtime library, and an active RoCEv2 link. Check `ibv_devinfo -v` and use
the addresses of the directly connected ports, not the management network.

For example, with the coordinator at `172.31.250.1` on the direct link:

```sh
# Worker.
./ds4 --cuda -m gguf/DeepSeek-V4.1-Flash-Q2.gguf --ctx 32768 \
  --tensor-parallel --role worker --coordinator 172.31.250.1 9911 \
  --transport rdma

# Coordinator.
./ds4 --cuda -m gguf/DeepSeek-V4.1-Flash-Q2.gguf --ctx 32768 \
  --tensor-parallel --role coordinator --listen 172.31.250.1 9911 \
  --transport rdma
```

The link address selects the matching verbs device and GID. If selection is
ambiguous, use `--rdma-device` and `--rdma-gid-index`. The transport stages
CUDA results through host memory; this is RoCE, not GPUDirect. TCP is also
available with `--transport tcp` on both peers.

The coordinator can be `ds4-agent`, `ds4-server` or `ds4-bench`. Match context
sizes on both sides. With five or more ready sessions, the server batches decode
across both GPUs. Smaller groups run in order, which is faster at those sizes.
Use `--batched-session 8` on the server; allow memory for all eight contexts.
Vision, DSpark and other model/quant layouts are not supported by CUDA network TP.

## Pipeline parallelism

Each process maps only its assigned layers, retaining that slice of the KV
state. Layer ranges are inclusive. `N:output` includes the final layer and
output head. Activations travel from one stage to the next over TCP.

For Flash Q4 on two machines, download `ds4f-q4` on both, then start each side.
Replace the example address with your coordinator's reachable address:

```sh
# Machine A.
./ds4 --role coordinator --layers 0:19 --listen 10.99.0.2 9911

# Machine B.
./ds4 --role worker --layers 20:output --coordinator 10.99.0.2 9911
```

Normally give the output head to the final worker. With several workers,
choose non-overlapping ranges covering the entire model. Workers register
their ranges with the coordinator; intermediate workers forward activations
directly to the next stage.

Long prefill chunks can occupy different stages simultaneously. A single
generation stream cannot use that overlap: each token must finish the route
before the next one is sampled. Use pipeline mode primarily for capacity and
long-prefill throughput, not as a guaranteed decode speedup.

### GLM 5.3 Flash Q4_K across a Mac and a Spark

The same model file on both machines; the Mac coordinates and the Spark works.
Q4_K does not fit on the Spark alone, so this split is what makes it usable there.
Neither machine can hold the whole model, and the split is a contiguous layer range
chosen at launch: the coordinator takes `0:N`, the worker `N+1:output` plus the head.

**Recommended: coordinator `0:20`, worker `21:output`.** Measured at ctx 524288 on a
286,646-token prompt, cold: **415.0 t/s prefill and 121.1 ms per token** — an
11.5-minute ingest. The previously used `0:23` / `24:output` gives 356.3 t/s and
120.5 ms at the same depth, so the Spark-heavy split is **+16.5 % prefill with decode
unchanged**. At 39,865 tokens the same comparison is 455.2 against 398.8 t/s and
103.0 against 98.5 ms.

The two hosts talk **directly** over the point-to-point link — no tunnel. The
coordinator listens on the link address and the worker dials it:

```sh
# Spark — worker. The 14 GiB guard reserve is what lets this larger slice fit.
DS4_GLM_MEMORY_GUARD_RESERVE_GB=14 \
~/bin/ds4 --cuda -m ~/mlmodels/glm/GLM-5.3-Flash-Q4_K.gguf \
  --role worker --layers 21:output --coordinator 192.168.2.1 9911 \
  --listen 192.168.2.2 55911 --ctx 524288

# Mac — coordinator, serving HTTP on 8081.
~/bin/ds4-server -m ~/mlmodels/glm/GLM-5.3-Flash-Q4_K.gguf \
  --role coordinator --layers 0:20 --listen 192.168.2.1 9911 \
  --ctx 524288 --host 0.0.0.0 --port 8081
```

Start the worker first — it retries until the coordinator's control port answers, so
ordering is convenience rather than requirement. The worker's `--listen` is its data
listener, which the coordinator dials directly for snapshots. Use the **literal
IPv4** of the direct link on both sides, never a hostname — an mDNS resolution
failure killed an earlier session-scoped run mid-ingest.

Operational notes for this pair:

* **macOS has no `setsid`.** Wrapping a launch in it fails silently and the process
  never starts; start the worker in a terminal or under `launchd`.
* **Give any `ssh` inside a backgrounded job `-n`,** or it consumes the job's stdin
  and hangs.
* **Judge readiness by the served payload** — a real completion, or `/v1/models` —
  never by the port being open, which happens before the weights are mapped.
* After a failed run, check `journalctl -k | grep "r8127: enP7s7: link down"` on the
  Spark: that NIC flaps intermittently and a flap aborts the run.

If a future macOS release refuses non-loopback accepts for a locally built binary
again — it did on 2026-09-19, which is why `~/ds4-tunnel` exists — fall back to the
loopback form with the two forwards it provides rather than fighting the OS.

**Memory here is a setting, not a hardware limit.** At `0:20` the Spark plans
104.73 GiB of a 107.61 GiB budget and the Mac 82.22 GiB of 115.19 GiB, so this split
is the default at full context rather than a short-context option. The guard sizes
itself as `min(0.99 x base, base - reserve)`, where the base is `hw.memsize` on Apple
and the CUDA device's recommended working set elsewhere, and the GLM 5.3 reserve is
18 GiB by default — 14 GiB here, set at runtime with
`DS4_GLM_MEMORY_GUARD_RESERVE_GB` (and `DS4_GLM_MEMORY_GUARD_FRACTION`). The Spark's
OS-visible total is 121.61 GiB, because the GB10 keeps ~6.4 GiB of its 128 GB from
Linux, and the Mac's 115.19 GiB comes from `iogpu.wired_limit_mb=120000`.

**Choose the split for prefill, not decode.** Moving one layer from the Spark to the
Mac costs **15.8 t/s of prefill** and buys only **1.30 ms of decode**, because
prefill costs `max(stage)` while decode costs `sum(stage)`: rebalancing recovers the
whole stage difference for prefill, but for decode it merely swaps one stage's cost
for the other's. Prefill is coordinator-bound (Mac GPU at 99-100 %, Spark ~74 % duty
at `0:20`); decode is **not** GPU-bound on either machine (~71 % Mac, ~62 % Spark,
never above 90 %) and is weight-bandwidth-bound at batch 1 — which is why the split
barely moves it, and why speculation does not rescue it either.

### Full PRO Q4

For two 512 GB Mac Studios, use the split artifacts:

```sh
# Machine A.
./download_model.sh pro-q4-layers00-30
./ds4 -m gguf/DeepSeek-V4-Pro-Q4K-Layers00-30.gguf \
  --role coordinator --layers 0:30 --listen 10.99.0.2 9911

# Machine B.
./download_model.sh pro-q4-layers31-output
./ds4 -m gguf/DeepSeek-V4-Pro-Q4K-Layers-31-output.gguf \
  --role worker --layers 31:output --coordinator 10.99.0.2 9911
```

These downloads do not change `ds4flash.gguf`. Startup is expensive because
each side must make its model slice resident.

### Tuning and recovery

Keep default chunk sizes first. `--dist-prefill-window N` controls the number
of chunks in flight; `--dist-prefill-chunk N` overrides the session-derived
chunk size. `--debug` shows route and per-hop timings.

Activations use 32-bit transport by default. `--dist-activation-bits 16` halves
the payload; `8` is more aggressive. These change numerical precision on the
wire, not weights or KV storage. Validate output when changing them.

A disconnected worker invalidates the route. In-flight work can fail; later
requests need a complete route before proceeding, and the coordinator can
replay the saved token prefix to rebuild worker state. Pipeline snapshots
serialize all layer slices into one payload and redistribute them when loaded.

For protocol details, see [ds4_distributed.c](../ds4_distributed.c)
and [ds4_tp.c](../ds4_tp.c).
