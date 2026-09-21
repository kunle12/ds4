# Criterion 6 — distributed snapshot restore (fresh pair and roles-swapped)

**Status:** met 2026-09-21. Both remaining halves — a fresh-pair restore and a
roles-swapped restore — now pass with identical output.
**Depends on:** the `ce4d214` streaming fix only indirectly (the snapshot work
predates it); no change was needed for this test.

---

## 1. Setup

Q4_K, ctx 4096, pipeline `0:20` / `21:output`, coordinator started with
`--kv-disk-dir /tmp/ds4kv-crit6 --kv-cache-min-tokens 512`. Prompt: 772 tokens
(a nonce plus a `promessi_sposi.txt` slice plus *"output the numbers 0 through
39"*), deterministic at temperature 0. The first cold request writes a 163.82 MiB
`.kv` snapshot.

## 2. Fresh-pair restore

Run 1 saves; run 2 is a brand-new pair on the same disk directory.

| metric | run 1 (save) | run 2 (fresh pair) |
| --- | ---: | ---: |
| `prompt_tokens` | 772 | 772 |
| `cached_tokens` | 0 | **772** |
| `cache_write_tokens` | 772 | **0** |
| `completion_tokens` | 136 | 136 |
| output sha256 | `a3898c57…` | `a3898c57…` (identical) |

## 3. Roles-swapped restore

The snapshot saved by the **Mac** coordinator was copied to the **Spark**, which
then ran the coordinator (CUDA `0:20`) with the Mac as the worker (Metal
`21:output`) — the roles reversed relative to the save.

```
0921 22:26:21 ds4-server: kv cache hit text tokens=772 text=2490 quant=4 key=token-text load=173.8 ms file=/tmp/ds4kv-crit6/24c1226d….kv
```

| metric | Mac coordinator (save) | Spark coordinator (restore) |
| --- | ---: | ---: |
| `prompt_tokens` | 772 | 772 |
| `cached_tokens` | 0 | **772** |
| `cache_write_tokens` | 772 | **0** |
| output sha256 | `a3898c57…` | `a3898c57…` (identical) |

The disk snapshot is split across whatever route exists at load time, so a
topology change (here a full role swap) is absorbed.

## 4. Notes for the next person

* **A first attempt returned HTTP 500** with
  `kv cache load failed … distributed route incomplete: missing layer 21`. That
  was a test-harness timing error, not a restore defect: the HTTP port (and
  `/v1/models`) is up before the worker has connected, so the route was not yet
  complete. A restore test must wait for the **worker connection**, not just for
  `/v1/models`. The corrected harness does.
* `completion_tokens` differed between the two topologies (136 vs 118) while the
  output text was byte-identical — the same near-tie tokenisation difference seen
  in the E1 A/B (`ds4-glm53-e1-decode-graph-ab.md` §3), not a divergence.

## 5. Interpretation

The checkpoint format is topology-independent as designed: one save loads on a
fresh pair and on the reversed roles, and both reproduce identical output.
Criterion 6 is met.
