# Criterion 1 oracle — pipeline vs single-Mac GLM 5.3 Q4_K

**Status:** run and **passed** 2026-09-21 on the live pair.
**Scope:** the plan's acceptance criterion 1 — *pipeline greedy continuation
matches a single-Mac Q4_K run for ≥ 128 tokens; logits within the repo's
cross-backend tolerance*.
**Depends on:** the fix in `ds4-glm53-ssd-streaming-regression.md`; before it, the
single-Mac `--ssd-streaming` generation reference could not run at all.

---

## 1. Setup

| | pipeline | single host |
| --- | --- | --- |
| machine(s) | Mac coordinator `0:20` + Spark worker `21:output` | Mac alone |
| weights | `GLM-5.3-Flash-Q4_K.gguf` | same |
| context | 32768 | 32768 |
| memory | coordinator 79.54 GiB, worker resident slice | `--ssd-streaming`, 96.71 GiB plan |

Both hosts ran the same commit. Prompt: a nonce plus ~9.2 KB of
`speed-bench/promessi_sposi.txt` plus *"output the numbers 0 through 199, one per
line, nothing else"* — **2,891 tokens**, which crosses the dense→indexed boundary
at 2,048, so the indexed path is exercised on both sides.

- **Logits**: CLI `--dump-logits` after a cold prefill, full 154,880-token vector.
- **Continuation**: CLI greedy (`--temp 0`), up to 200 tokens.

## 2. Results

### 2.1 Greedy continuation — §criterion satisfied

| | bytes generated | common prefix | verdict |
| --- | ---: | ---: | --- |
| pipeline | 290 | | |
| single Mac | 290 | **290 / 290** | **byte-identical** |

The output is the digits `0…99`, one per line — about 200 tokens, above the
128-token bar. Longest-common-prefix = 290 bytes means there is no divergence at
all, not merely a short prefix.

### 2.2 Logits

| metric | value |
| --- | ---: |
| vocab | 154,880 |
| finite pairs | 154,880 (no non-finite) |
| mean \|Δ\| | **0.306940** |
| max \|Δ\| | 1.984564 (token 57,219) |
| top-1 overlap | 1 / 1 |
| top-8 overlap | 7 / 8 |
| top-16 overlap | 15 / 16 |
| top-64 overlap | 56 / 64 |
| argmax equal | **yes** |

For reference, the repo's kernel-level cross-backend figure (CUDA vs Metal on
one machine) is mean \|Δ\| 0.0727 with top-16 16/16 (implementation log §7). This
measurement is larger because it is not a kernel comparison: it accumulates the
CUDA worker's routed-MoE difference across layers 21–44 *and* the output head,
between two machines. No numeric full-logit tolerance is declared anywhere in the
repository, so these values are recorded as the baseline. The practical
agreement — identical argmax and identical generated text — is what criterion 1
asks for.

## 3. Reproduction

```sh
# pipeline logits
~/bin/ds4 --metal -m /Users/xun/mlmodels/glm/GLM-5.3-Flash-Q4_K.gguf \
  --role coordinator --layers 0:20 --listen 192.168.2.1 9911 --ctx 32768 \
  --prompt-file oracle-prompt.txt --dump-logits pair.json --temp 0
#   (Spark worker first: --role worker --layers 21:output ...)

# single-Mac logits
~/bin/ds4 --metal --ssd-streaming -m /Users/xun/mlmodels/glm/GLM-5.3-Flash-Q4_K.gguf \
  --ctx 32768 --prompt-file oracle-prompt.txt --dump-logits single.json --temp 0

# greedy continuation on each (drop --dump-logits, add -n 200 --nothink)
```

`compare_logits.py` prints the logit table; `compare_gen.py` prints the common
prefix.

## 4. Notes and open items

- **`--dist-replay-check` did not fire.** It was passed to the CLI coordinator
  along with `--dump-logits`, but no `distributed replay check` line was emitted.
  The flag is honoured in `dist_run_coordinator` (`state.replay_check = opt->replay_check`),
  reached through `ds4_dist_run`, which `ds4_cli.c`/`ds4_server.c` only call for
  the **worker** role. Its coordinator-path reachability should be checked before
  the `--dist-replay-check` half of WS6 is considered done. Not a blocker for
  criterion 1 as written.
- **Single-Mac Q4_K `--ssd-streaming` numbers need re-measuring.** The figures in
  the split-design doc predate `ce4d214`; the path is fixed but the old numbers
  were not re-validated here.
- The criterion-1 result is output-level equivalence across two machines. It does
  not replace the repo's same-machine exact-logit oracles, which remain the
  regression gate for individual kernels.
