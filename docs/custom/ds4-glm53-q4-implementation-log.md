# GLM 5.3 Flash Q4 across Mac + Spark — implementation log

**Purpose.** Running record of what has been done, with the evidence that backs
each claim, so work can be resumed or audited without re-deriving it.
**How to use.** Append new entries at the end of §3 and update §1/§2 in place.
One entry per change or measurement; never rewrite history — correct it in a new
entry.
**Related documents.** `ds4-glm53-q4-split-design.md` (the plan and its
workstreams), `ds4-technical-analysis.md` (engine), `ds4-v41-split-design.md`
(the analogous V4.1 port study).

Timestamps are local (AEST). Entries marked `≈` are reconstructed from ordering
rather than read from a log; all quoted numbers and command outputs are
verbatim from the session.

---

## 1. Status summary (as of 2026-09-19 18:00)

**Objective.** Run `GLM-5.3-Flash-Q4_K.gguf` (177.77 GiB, `glm5-next`, 45
executable layers) as a two-machine pipeline — Mac Studio coordinator + DGX
Spark worker — for 250K–500K-token coding sessions, with the Spark inside its
thermal envelope.

| Area | State |
| --- | --- |
| GLM 5.3 layer-slice correctness (wire width) | **done, validated bit-exact** |
| Cross-machine Q2 pipeline | **working and measured** |
| Distributed snapshot round-trip across the split | **save verified 2026-09-19** (165 MiB checkpoint, worker's shard fetched over the data forward); load path exercised and reported a hit — equivalence still needs the fresh-pair restore (§6.1 #8) |
| Q4_K on the pair | **working and measured** — WS 2, WS 3 and WS 5's first slice landed (log Phases M–P): correct output (logits byte-identical to the warp path, top-16 16-of-16 against the Metal reference), **100.6 t/s prefill / 11.4 t/s generation** against 40.8 / 5.7 for Mac-only streaming — 2.47× and 1.99×, so the ≥2× gate passes. Phases N and O fixed the pre-existing 256-vs-288 hardcoding in the expert map and the expert-major grid, and tightened a third site (the tile slack) that was sufficient rather than broken; Phase P fixed a swiglu clamp the CUDA MoE ignored, which improved agreement with the Metal reference (mean \|Δ\| 0.085 → 0.073). `make test-glm53-moe-q4k` passes bit-exact. The MTP tok2 and scalar paths still refuse Q4_K by name, unverified |
| Q4_K on the Mac alone | **working and measured** (SSD streaming) |
| Spark thermal protection | **installed, enabled, verified live**; re-armed by itself after the 2026-09-19 power cycle |
| Access path (macOS ALF workaround) | **installed as a boot-persistent launchd daemon, verified**; carries both forwards (`-R` control, `-L` data for snapshots) |
| Binaries deployed to `~/bin` on both hosts | **rebuilt 2026-09-19 from byte-identical sources and reinstalled** (`ds4.c` md5 `98c92891…` on both) |
| Spark wedge, 2026-09-19 | two resident workers left the box with a live kernel and dead userland; **physical power cycle**, then a mandatory clean-slate precondition (plan §4.3) |
| Plan for closing the Q4 gap | **written** (`ds4-glm53-q4-split-design.md`) |
| Fork and branch | **pushed** — `customisation` on `kunle12/ds4`, HEAD `1b82376`, 9 commits ahead of upstream `8db1d1d` |

**Next action:** workstream 1 of the plan — type traits + dispatch predicate for
the GLM routed MoE (`{Q2_K, Q4_K}`), then WS 2 (Q4_K prefill), which is the first
milestone that makes a 262K Q4 ingest measurable behind the guard.

**Nothing is running** except the tunnel daemon (Mac), the thermal guard (Spark),
and — as of this writing — the Q2 pair parked in the working configuration that
§10 restarts, so the rebuilt binaries can be exercised without a reload.

---

## 2. Source changes

| # | File / anchor | Change | Why | Verified |
| --- | --- | --- | --- | --- |
| 1 | `ds4.c:71777` (`ds4_engine_hidden_f32_values`) | return `N_HC * N_EMBD` when `ds4_model_is_glm53()`, `N_EMBD` otherwise | GLM 5.3's mHC block is the slice payload; the function described GLM 5.2's single stream, so every wire/buffer size was 4× too small | built both backends |
| 2 | `ds4.c:74441` (GLM branch, `ds4_session_eval_layer_slice`) | `hidden_dim = glm53 ? N_HC*N_EMBD : N_EMBD` | per-token chunk stride across a slice boundary | built both |
| 3 | `ds4.c:73521` (`ds4_session_eval_output_head_from_hc`) | write into `gg->hc_cur` for glm53 | the head collapses the HC block; writing `gg->cur` left it stale | built both |
| 4 | `ds4_metal.m:4754` + `:4821` | new `ds4_metal_executable_dir()`; search `<exe_dir>/metal/…` in addition to `metal/…` and `./metal/…` | kernel sources were CWD-relative only, so an installed binary (`~/bin/ds4-server` + `~/bin/metal`) only worked if started from a directory containing `metal/` | built, verified from `/tmp` |
| 5 | `ds4.c:61000` (`glm_layer_payload_tensor_bytes`, KDA branch) | the branch no longer rejects the payload header's **uniform** `compact_live`/index counts; it returns the conv + recurrent state span and asserts it is non-zero | the guard made any slice containing a KDA layer unsizeable as soon as a context existed, so the first distributed checkpoint failed with `distributed KV shard tensor size overflow` | snapshot save **and** load verified end to end (§11) |
| 6 | `ds4.c:46076` (`glm_graph_layer_uses_generic_routed_moe`) | the Q4_K branch is now **unconditional**: a homogeneous Q4_K expert trio always routes to the generic dispatch. ~~`DS4_GLM_GENERIC_MOE_Q4K`, added to gate it while the port was being written, **removed 2026-09-20**~~ | a spike shortcut to exercise Q4_K on CUDA before the kernel port exists; its own comment said remove or make unconditional once the result is measured. Both paths were then measured against each other on the pair — speed and output — and the shortcut turned out to be **2.7× faster** (258.9 vs 95.3 t/s prefill, identical output), so it was promoted rather than dropped; see §13 | Q4_K now defaults to the generic dispatch; the GLM-specific Q4_K kernels stay reachable via `DS4_CUDA_GLM_MOE_TYPES` and are still covered by `make test-glm53-moe-q4k` |

Changes 1–3 are the fix that makes GLM 5.3 pipeline mode work at all; change 4 is
required for the `~/bin` deployment convention; change 5 unblocks distributed
checkpoints; change 6 is an experiment, not part of the design.

**Not in the repository** (host-level, staged outside the tree): the Spark
thermal-protection scripts/units (§7) and the launchd tunnel job (§7). Both are
backed up on the Mac and documented in §7.

`docs/custom/` is tracked as of commit `413d331`; the user's pre-existing
`README.md` modification and an uncommitted `.gitignore` change remain local.

---

## 3. Chronological log

### Phase A — Analysis (before any change)

| # | Action | Evidence / result |
| --- | --- | --- |
| A1 | Read the technical-analysis doc, `docs/DISTRIBUTED.md`, `DGX_SPARK.md`, `MODELS.md`, `NETWORK_SETUP.md`, QA gates, and the distributed/GLM code paths | `ds4_distributed.c` is model-agnostic; GLM has a slice branch; whole pipeline mode exists |
| A2 | Parsed the GGUF directly (metadata + tensor table, 1412 tensors) | `general.architecture = glm5-next`, 46 blocks, `nextn=1` ⇒ 45 executable; `q4_k` = 163.27 GiB in **129 tensors = 43 MoE layers × gate/up/down**; everything else BF16/Q8_0/F32; per-layer bytes exact |
| A3 | `./ds4 --inspect` on the checkpoint | engine binds it as `GLM-5.3-Flash`, 320.76 B params, 177.77 GiB |
| A4 | Memory-admission probes per slice (`DS4_GLM_MEMORY_GUARD_REPORT=1`) | 3-layer slice: graph 2.80/2.88/2.97 GiB at 8K/32K/64K ctx ⇒ graph cost is ~layer-count independent; measured budget 115.19 GiB |
| A5 | Verified the Mac↔Spark 10GbE direct link | `en0` 10Gbase-T active, `192.168.2.1`; Spark `enP7s7` 10000Mb/s; RTT 0.94–1.21 ms; single-stream TCP **0.46 GiB/s**; payload need ≈26 MB/s |

### Phase B — Defect discovery, fix, validation

| # | Action | Evidence / result |
| --- | --- | --- |
| B1 | Traced the slice payload width | the GLM 5.3 tape reads/writes `n_tokens × N_EMBD × N_HC` (64 KiB/token) in all three forward paths (`ds4.c:52361`, `:53646`, `:55619` and readbacks), while `ds4_engine_hidden_f32_values` returned `N_EMBD` (16 KiB) |
| B2 | Confirmed by history | `git log -L` on that function: the `GLM_DSA → N_EMBD` special case arrived with **GLM 5.2** (n_hc = 0); the `glm53 ? N_HC : 1` multiplier arrived later with **GLM 5.3** — the wire contract was never updated |
| B3 | Patched the three sites (§2 #1–3) | — |
| B4 | Built Mac (Metal) and Spark (CUDA `sm_121`) | Spark `ds4` 47.6 MB; Mac binaries rebuilt |
| B5 | **Loopback parity test** — two processes on the Mac, Q2, `--layers 0:23` + `--layers 24:output`, 48 greedy tokens vs a single-host Q2 run | **byte-identical output** (`diff` clean). Any HC-carry error diverges immediately, so this validates the fix end to end |
| B6 | Probed long-context slice memory | coordinator (0:23) ctx 524288: 94.88 / 115.19 GiB; worker (24:output): 92.07 / 103.63 GiB — the pair holds all 173.74 GiB of executable weights resident |

### Phase C — Cross-machine work, and the Q4 blocker

| # | Action | Evidence / result |
| --- | --- | --- |
| C1 | First cross-machine attempt: Mac coordinator `--listen 192.168.2.1`, Spark worker | route never established; worker's HELLO sat unaccepted (`CLOSE_WAIT`, no owner, `accept()` never returned) |
| C2 | Isolated the cause | loopback `--listen 127.0.0.1` accepts normally; a **20-line `cc`-built listener fails identically** from the same shell while Apple-signed `/usr/bin/nc` works on the same address/port ⇒ macOS Application Firewall (confirmed enabled: `Firewall is enabled. (State = 1)`), not a ds4 bug |
| C3 | Workaround: `ssh -R 9911:127.0.0.1:9911` tunnel; Mac coordinator on loopback, Spark worker dialing its own loopback | route established |
| C4 | **Q2 pipeline measured** at 32 768 ctx | 383.48 t/s prefill, 12.82 t/s decode, first token 136 ms |
| C5 | Spark-as-coordinator Q4 attempt | `ds4: glm routed moe: unsupported types 12/12/12` |
| C6 | Read the CUDA dispatch | `ds4_cuda.cu:32103`: `if (gate_type != 10u || up_type != 10u || down_type != 10u) … return 0;` — type 10 = Q2_K; the decode path delegates to the same batch function |
| C7 | CUDA coordinator-side slice prefill | `CUDA tensor read failed: unspecified launch failure` for chunks ≥ 512 rows (512/2048/4096/8192 all fail; a 26-token slice works). **Reproduced on pristine `8db1d1d`** ⇒ pre-existing, not caused by §2 #1–3 |
| C8 | Stale-worker incident | an unpatched worker left running caused `input hidden-state size does not match token span`; diagnosed with temporary instrumentation that printed got/want/tokens/hc/bits. Instrumentation **reverted**; only the three functional hunks remain |

### Phase D — Q4 measured where it can run

| # | Action | Evidence / result |
| --- | --- | --- |
| D1 | Mac alone, Q4 + `--ssd-streaming`, 32 768 ctx | 84.23 t/s prefill, 8.84 t/s decode; plan 97.09 GiB; **5 435 / 12 384 experts** cached (71.65 GiB) |
| D2 | Same at 262 144 ctx | **82.47 t/s prefill (53 min cold ingest), 8.00 t/s decode**; plan 99.84 GiB; expert cache unchanged; no thermal warning logged by macOS |
| D3 | Derived streaming baseline | per-token routed-expert traffic 4.43 GiB; ~40 % misses ⇒ ~1.78 GiB/token from SSD |

### Phase E — Spark thermal protection

| # | Action | Evidence / result |
| --- | --- | --- |
| E1 | Spark went offline mid-run | Mac `en0` lost media (`status: inactive`), `spike.local` mDNS failed, `192.168.2.2` unreachable. User restarted it |
| E2 | User attributed it to thermal protection; confirmed on the web | NVIDIA field diagnostic fails such units on **PowerStress, `020000600139`**; death state logged as `[HW_THERMAL_SLOWDOWN] · GPU 88→90 °C · CPU zones 97→98 °C`, ACPI trip 104 °C; silent hard-lock with no panic/OOM is the documented symptom. **This corrects my earlier "memory exhaustion" hypothesis** |
| E3 | Measured the Spark uncapped under pipeline prefill | board peak **90 °C** (10/60 samples ≥ 85 °C), `HW Thermal Slowdown` accumulating, `SW Power Capping` 69 s; GPU die ~10 °C cooler than the board zone |
| E4 | Built the protection kit (guard + CPU cap + units + installers + README) | staged `/tmp/gb10-thermal-protect`, pushed to Spark `~/thermal-protect`; **backed up to Mac `~/thermal-protect`** |
| E5 | Install hang diagnosed | not the script: `is-system-running = starting`, `plymouth-quit-wait.service` `activating` 17 min (headless box booting to `graphical.target`), so every start job ordered after `multi-user.target` queued behind it. Unit reordered to `After=local-fs.target`; `set-default multi-user.target` applied by the user |
| E6 | Guard bug found and fixed | `board: parameter not set` → `set -u` exit 2 on the *first* `set_cap` (board not yet sampled); slow-down field parse was off by one; initial cap logged `board=0C`. All three fixed and **verified with a stub `nvidia-smi` that succeeds at `-lgc`** (the unprivileged dry-run never entered that branch, which is why the bug escaped) |
| E7 | Installed and verified | `gb10-cpu-cap.service` active (`scaling_max_freq = 2400000` on all policies, hw max 2.81 GHz); `gb10-thermal-guard.service` active; GPU clock range locked ≤2100 MHz; board 47–50 °C idle |
| E8 | **Capped sweep** (bounded 32K → 64K → 128K, guard armed) | see §4; peak board **81 °C**, **0** throttle-active samples over 269 guard samples, 0 guard interventions |

### Phase F — Deployment

| # | Action | Evidence / result |
| --- | --- | --- |
| F1 | `ds4_metal.m` executable-relative lookup (§2 #4) | `cd /tmp && ~/bin/ds4 …` compiled its Metal library and answered `O(log n)`; `/tmp` has no `metal/` so this can only come from the new path |
| F2 | Installed 5 binaries to `~/bin` on both hosts (+ `metal/` on the Mac) | checksums MATCH on all; replaced stale builds (Mac `ds4-server` Sep 17, Spark Sep 13) |
| F3 | Found the pre-existing CWD workaround in `llm_config.json` | V4.1 entry uses `/Users/xun/dev/ds4/ds4-server --chdir /Users/xun/dev/ds4`; V4 Flash uses the stable `~/bin/ds4-server`. With F1 the V4.1 entry can drop `--chdir` and the repo path |
| F4 | Built `~/ds4-tunnel` (daemon + agent plists, installers, README) | address pinned to the literal IPv4 (an mDNS failure is what killed the first tunnel); `KeepAlive`, `TCPKeepAlive`, `ServerAliveInterval`, `BatchMode`, explicit `-i`/known_hosts, `ExitOnForwardFailure`, logs to `~/Library/Logs` |
| F5 | Verified the exact plist command line before install | Spark loopback listener up; worker connected; coordinator (`~/bin/ds4`, run from `/tmp`) generated correct output |
| F6 | Could not install the launchd job myself | my shell is an SSH/Background session: `gui/501` unreachable (125), `user/501` rejects (EIO), legacy `launchctl load -w` silently ineffective. Installers now fail loudly with that explanation |
| F7 | User installed the daemon | `state = running`, never exited; runs as `xun` |
| F8 | **Acceptance tests on the installed daemon** | listener on Spark loopback ✓; **KeepAlive**: killed pid 45807 → launchd started 45914 within ~2 s, listener back ✓ |
| F9 | End-to-end through the daemon, installed binaries | **Q2: works** (route ready, correct output, 21.3 t/s prefill / 17.4 t/s decode on a 16-token prompt). **Q4: fails as designed** — coordinator `prompt processing failed: cuda GLM layer-slice evaluation failed at pos 0`, worker `glm routed moe: unsupported types 12/12/12` ⇒ the gate bites from the **worker side** too, confirming no contiguous cut avoids it |

### Phase G — Plan

| # | Action | Evidence / result |
| --- | --- | --- |
| G1 | Wrote `ds4-glm53-q4-split-design.md` | goal + acceptance criteria; measured state; the three gaps; design (template the ten GLM MoE kernels, dispatch mirroring Metal, escape hatch); 11 workstreams with effort; layered validation; risks; rollout; fallbacks; open questions |
| G2 | Updated it with the tunnel/snapshot constraint (§4.2) | acceptance criterion 6 (snapshot round-trip) needs a direct connection or a HELLO addition carrying the worker's reachable host |

### Phase H — Repository organisation

| # | Action | Evidence / result |
| --- | --- | --- |
| H1 | Renamed `docs/src/` → `docs/custom/` at the owner's request, so everything in that directory is known to be owner/agent-authored rather than upstream | directory is untracked in git (`?? docs/custom/`), so a plain `mv` was correct; updated the three references that named the old path (this log's artifact table and working-tree note, and `ds4-v41-split-design.md` §related-analysis); verified no `docs/src` or `src/ds4-` reference remains in the tree, and none in the out-of-tree kits |

### Phase I — Fork and remote workflow

| # | Action | Evidence / result |
| --- | --- | --- |
| I1 | Adopted the fork as the working remote: `origin` = `kunle12/ds4`, `upstream` = `antirez/ds4` (the existing `origin` was renamed) | `git remote -v` shows both; forks tracked this way make "pull from upstream, push to fork" the default |
| I2 | Created branch `customisation` from upstream `8db1d1d` | fork's `main` and the local `main` were both already at that commit, so the branch starts exactly at the fork point |
| I3 | Committed the work as five focused commits rather than one blob | `b2e9c39` wire-width fix, `cf077c1` Metal source lookup, `2bdb60c` gitignore, `3343e3c` network doc + README link, `413d331` custom docs. The two code fixes can be offered upstream independently, and the network doc (LAN addresses, MACs) is isolated so it can be dropped |
| I4 | Added `.venv/` and `venv/` to `.gitignore` | a 41 MB local Python virtualenv was untracked and one `git add -A` away from being committed |
| I5 | Set the commit identity from the account rather than guessing | `Xun <6646691+kunle12@users.noreply.github.com>` — the numeric-ID noreply form, derived from `api.github.com/users/kunle12`; set repo-local, then all five commits rewritten with `--reset-author` |
| I6 | Push credential path | SSH is **not** registered on the account (`git@github.com: Permission denied (publickey)`); `gh auth login` (HTTPS, scopes `gist, read:org, repo, workflow`) is the working path. Wired **repo-locally** via `credential.https://github.com.helper = !gh auth git-credential` so the global config is untouched |
| I7 | Pushed and verified | `origin/customisation = 413d331 =` local HEAD; fork `main` still `8db1d1d`; `main` tracks `upstream/main`; branch is 0 behind / 5 ahead of upstream. Push printed `failed to store: -25308` — that is the osxkeychain helper declining to cache the token from a non-interactive session, harmless because the gh helper supplies credentials on each push |

**Remote workflow from here:** `git fetch upstream && git merge upstream/main` (or rebase) to take antirez's changes; `git push` publishes to the fork. Nothing in this phase changes the code, the installed services, or the Spark.

### Phase J — Wedge, recovery, and distributed snapshot verification

| # | Action | Evidence / result |
| --- | --- | --- |
| J1 | Started a second worker on the Spark while a leftover from the earlier spike was still resident | the route registered from the **leftover** (`data_port=39215` ephemeral, `ctx=33792`), not the new worker, which never finished loading |
| J2 | The Spark wedged | `sshd` completed TCP handshakes and never wrote a banner on any established connection (60 s budget); only `:22` open; ping 0.5 ms, 0 % loss |
| J3 | Eliminated the network and the guard as causes | `en0` 0.1 KB/s in / 0.4 KB/s out, every socket to the Spark with empty queues, and new logins bypass the tunnel entirely; every `nvidia-smi` in the guard is wrapped in `timeout 5`; the CPU cap is a 14 % cut |
| J4 | User power-cycled the box | up in 1 min; 118 GiB free, zero `ds4`; guard and CPU cap re-armed by themselves; board 55 °C |
| J5 | Tunnel self-healed | `KeepAlive` re-established the session and the Mac held `127.0.0.1:55911` again; the plan gained a mandatory clean-slate precondition (§4.3) |
| J6 | Worker pinned with `--listen 127.0.0.1 55911` | `data_listen=127.0.0.1:55911`; the coordinator logged `data_port=55911` (not the ephemeral 39215) and `ctx=4096` |
| J7 | First snapshot attempt failed before any data connection | `kv cache skipped tokens=819 reason=cold because KV payload staging failed: distributed KV shard tensor size overflow` — the KDA guard of §2 #5 |
| J8 | Fixed, rebuilt, re-ran | save wrote `bf072cd0…kv` (165.08 MiB, `save=18.2 ms`), data sockets observed on 55911, and the load reported `cached_tokens: 819` with `cache_write_tokens: 0` — details in §11. (The completion text was *not* compared; see §6.1 #7.) |
| J9 | Rebuilt **all five** binaries on both hosts from byte-identical sources | `ds4.c` md5 `98c92891b4a880429dd1937c58ce4b67` on both; Spark build `make -j20 cuda-spark` (`sm_121`) |

### Phase K — Rebuild, verification, and publication

| # | Action | Evidence / result |
| --- | --- | --- |
| K1 | Rebuilt all five binaries on both hosts from byte-identical sources and installed them | Mac `make` 17:46 (`~/bin` + 26 Metal kernels), Spark `make -j20 cuda-spark` (`sm_121`) 17:49 → `~/bin` 17:50; both builds reported zero errors and zero warnings; Mac smoke test from `/tmp`: `--inspect` binds `glm5-next`, all five answer `--help` |
| K2 | Stopped the stale pair, then re-verified on the rebuilt binaries | worker `data_port=55911` + `ctx=4096`, route ready; cold save `size=164.89 MiB save=18.1 ms` with six data-socket observations on `127.0.0.1:55911`, then `cache hit … load=126.8 ms` and `cached_tokens: 819` |
| K3 | Committed as three focused commits and pushed to the fork | `6e1f447` KDA sizing fix, `5881ac0` env-gated experiment, `1b82376` docs; `origin/customisation = 1b82376`, 9 commits ahead of upstream `8db1d1d`; the working tree's `ds4.c` md5 equals the compiled one, so the binaries match the commits |

### Phase L — Workstream 1: GLM routed-MoE type predicate

First implementation step of the plan (§4.1, WS 1), started 2026-09-19 after the
owner asked to begin.

| # | Action | Evidence / result |
| --- | --- | --- |
| L1 | Read the CUDA entry and the dispatch beneath it | `ds4_gpu_glm_routed_moe_batch_tensor` gated on `gate/up/down_type == 10`; activations are already quantized to Q8_K on every path, and the streaming lookahead is expert-count based (`256 * expert_bytes`), so both are type-independent exactly as §4.1 predicted |
| L2 | Added `glm_moe_types_allowed` + `glm_moe_type_name` + the `DS4_CUDA_GLM_MOE_TYPES` hatch (default `q2k,q4k`) | `ds4_cuda.cu`. `ds4.c`'s `DS4_TENSOR_*` enum is file-local, so this uses literals with comments, matching the file's existing `8u /* DS4_TENSOR_Q8_0 */` convention |
| L3 | Kept a loud interim guard for a known-but-unimplemented type | a Q4_K trio fails with its own message rather than reaching Q2_K arithmetic and producing plausible-looking garbage |
| L4 | Verified on the pair, all three paths | **A** Q4_K default → `glm routed moe: q4_K expert kernels are not implemented in this build`, coordinator `prompt processing failed … at pos 0` — no freeze, no garbage; **B** Q2_K default → `route ready` and a correct generation, unchanged; **C** `DS4_CUDA_GLM_MOE_TYPES=q2k` → `unsupported types q4_K/q4_K/q4_K (12/12/12)`, so the hatch narrows the set and the old numeric signature survives in the logs |
| L5 | Incremental CUDA build + install on the Spark | 96 s, no warnings; the Mac is unaffected because the file is CUDA-only (Metal's Q4_K MoE already works — which is why the coordinator's half loaded for tests A and C) |

### Phase M — Workstream 2: Q4_K routed-MoE kernels

First working Q4_K pair path, with a measured prefill and one open defect.

| # | Action | Evidence / result |
| --- | --- | --- |
| M1 | Surveyed what already existed | The Q4_K×q8_K dots were already in `ds4_cuda.cu` (`dev_dot_q4_K_q8_K_block`, `_vec`, `_block8`) and are exercised by the other routed-MoE dispatcher's `q4k_path`; what was Q2_K-only was the GLM dispatch and its kernels |
| M2 | Templated the four GLM kernels on the block type | tile8 gate/up and down-terms (prefill), warp gate/up and down (decode/small-batch). Block stride and dot now come from `sizeof(block_t)` and two overloads (`glm_moe_dot`, `glm_moe_dot8`), so no `84u` remains in them. The reduce kernel was already type-agnostic, and `ds4_gpu_glm_routed_moe_one_tensor` forwards to the batch entry, so no caller was left half-migrated |
| M3 | Kept the unported paths refusing by name | expert-major, scalar and MTP tok2 return `glm_moe_unsupported_path(...)` for a Q4_K trio instead of reaching Q2_K arithmetic |
| M4 | Ran a Q4_K pair end to end | first working Q4_K pair run: coherent output, `prefill 103.45 t/s, generation 11.36 t/s` |
| M5 | Compared logits against the Metal reference, with a Q2_K calibration | `--dump-logits` on `{pair, Mac-alone} × {Q2_K, Q4_K}` after the same 1091-token prefill: Q2_K pair vs Metal mean \|Δ\| 0.196 / max 1.48 / top-16 15-of-16; Q4_K pair vs Metal 1.100 / 9.09 / 9-of-16. Argmax and top-3 order agreed in both, so the divergence was in degree |
| M6 | Localised it to the tile8 path | the same CUDA path through the warp kernels instead matched Metal at mean \|Δ\| **0.085** / max 0.50 / top-16 15-of-16. For Q2_K the two paths came out **bit-identical** (max 0.0000), which I read at the time as the tile8 machinery being sound for Q2_K and the defect being specific to its Q4_K instantiation. **That inference was unsound** and it sent the next several hours into the type-dependent code: a bit-identity establishes nothing unless the two runs are known to have taken *different* paths, and I never checked which path either took. It is also the step that was wrong — the fault was shared code (Phase N). Bisecting with `DS4_GLM_MOE_NO_DOWN_TILE8_EXACT` attributed it to the tile8 **gate/up** kernel (tile8-gate/up + warp-down still deviated at 1.148) |
| M7 | Ruled out the arithmetic, nondeterminism and staging | a unit test on the Spark over random blocks, built from the real helpers extracted verbatim (`/tmp/q4dots_test.cu`; sizes confirmed 144 / 84 / 292), shows `dev_dot_q4_K_q8_K_block8` is **bit-identical** to n single-block calls for n=1..8 — and substituting the single-block expression inside the tile8 kernel left the logits dump **byte-identical**. A repeat run is byte-identical, and disabling the local batch IO staging changes nothing |
| M8 | Temporarily routed Q4_K around tile8 | a guard making `use_expert_tile8` require Q2_K, verified byte-identical to the warp run — **reverted in Phase N**, where the real cause turned out to be the expert count rather than the kernel |

**Measured, same 1091-token prompt, greedy, ctx 8192:**

| Configuration | prefill | generation | vs Metal (mean \|Δ\| over vocab) |
| --- | --- | --- | --- |
| Mac alone, Q4_K + SSD streaming (**the baseline to beat**) | 40.80 t/s | 5.73 t/s | reference |
| Pair, Q4_K, tile8 with the 256-expert fault | 103.45 t/s | 11.36 t/s | 1.100 — **wrong, not usable** |
| Pair, Q4_K, warp kernels (correct, but not the intended path) | 56.48 t/s | 11.37 t/s | 0.085 — correct |
| Pair, Q4_K, **tile8 after Phase N** | **100.57 t/s** | **11.43 t/s** | correct — logits byte-identical to the warp run's |

The corrected path is **2.47× the streaming prefill and 1.99× its generation**,
so the plan's ≥2× prefill gate passes — and it now passes on a computation that
matches the Metal reference, which the 103.45 t/s figure never did.

**Resolved — see Phase N.** The tile8 Q4_K instantiation produced different
`mid` values than the warp path because the expert map was bounded by a
hardcoded 256 while GLM 5.3 Flash has 288 experts: the pairs routed to experts
256–287 were never mapped, so no `mid` row was written for them and the down
projection read uninitialised data. Every exclusion recorded above still holds
— the dots, the block stride, the tile mapping, the staging and determinism were
genuinely innocent, which is exactly why the fault looked like it was inside the
kernel rather than in which pairs were fed to it. Phase M's guard was reverted
once the real cause was fixed.

### Phase N — A pre-existing expert-count bug, found by the WS 2 logit comparison

| # | Action | Evidence / result |
| --- | --- | --- |
| N1 | Traced the Phase M deviation to the expert map rather than the kernel | `ds4_gpu_glm_routed_moe_batch_tensor` accepted `n_total_expert` and discarded it (`(void)n_total_expert;`), then used a literal `256u` for the expert-map bound, the counts and lists sizing, both tile-builder launches and the three weight lookaheads |
| N2 | The model has 288 experts, not 256 | read straight from both GGUFs: `glm5-next.expert_count = 288`, `expert_used_count = 8`, `expert_feed_forward_length = 2048`. `glm_moe_expert_map_kernel` skips `e >= n_total_expert`, so every pair routed to experts 256–287 was dropped: no `mid` row was written for it, and the down projection then read uninitialised data — a wrong answer that looked plausible, never a crash |
| N3 | Fixed all nine sites to use the model's count | counts and lists sizing, both expert-map launches, both tile-builder launches and the three weight lookaheads; the parameter is validated (`n_total_expert == 0 \|\| > 65536` rejected) instead of discarded. The lookahead matters on its own: with 256 it under-covered the last 32 experts for the streaming resolver |
| N4 | Verified with tile8 enabled on both models | Q4_K pair: **byte-identical to the warp-path run**, top-16 15-of-16 against the Metal reference, mean \|Δ\| 0.109 over the reference's top-16, argmax agreeing. Phase M's guard was reverted as unnecessary — the tile8 path was never wrong, it was being fed the wrong set of pairs |
| N5 | Bounded the exposure — **and withdrew my first attempt at this row** | `n_total_expert` is type-independent, so any run that reached the expert map was subject to the 256 bound, for either model. What I cannot support is what I first wrote here: that the Q2_K control runs "never routed to an expert ≥ 256". That is statistically impossible — about 22 layers × 1091 tokens × 8 slots ≈ 192k selections over 288 experts — and I never verified it. The evidence I offered for it, that the Q2_K tile8 and warp dumps were byte-identical, proves nothing either way unless the runs took *different* paths, which I had not checked. The path trace added in Phase Q shows this pipeline hands the worker's MoE all 1091 tokens at once and takes tile8 for Q4_K, so the Q2_K run should have taken it too and the byte-identity is left **unexplained**. The likeliest cause is my own harness: the same command form failed to propagate `DS4_GLM_MOE_TRACE` to one worker, so the `DS4_GLM_MOE_NO_EXPERT_TILE8` flag that defined the warp comparison may never have reached it, which would make the comparison vacuous |

**Consequence.** WS 2's gate is met on the corrected path (100.57 t/s against
the 40.80 t/s streaming baseline), and the WS 3–4 shapes inherit the fix. The
lesson worth keeping: every kernel-level check pointed at the kernel, and the
fault was in the routing data feeding it. What caught it was comparing logits
against a different implementation — the thing the plan's parity harnesses
exist for.

### Phase O — Workstream 3 (expert-major), and three more instances of the same hardcoding

| # | Action | Evidence / result |
| --- | --- | --- |
| O1 | Templated the expert-major gate/up and down kernels for Q4_K | same recipe as the tile8 and warp kernels (`template <typename block_t>`, `glm_moe_dot`, `sizeof(block_t)`); the guard that refused Q4_K on this path is gone |
| O2 | The first verification failed, with the Phase N signature | forcing the worker onto expert-major gave Q4_K logits deviating by mean 1.485 against the Metal reference (max 3.88, top-16 11-of-16) — the same shape of error as the pre-fix tile8 path, which is what made it recognisable |
| O3 | Cause: the expert index is the grid's y dimension and it was a literal 256 | `dim3 ge1(..., 256u, 1)` and `dim3 ge2(..., 256u, 1)` launch one block row per expert, so experts 256–287 were never launched at all. `tile_capacity`'s slack was a literal `256u` too, but that one was **sufficient rather than buggy**, and it is worth being precise rather than lumping it in: the tile count is at most one per used expert, so Σ ceil(count_e/8) ≤ (n_pairs + 7·288)/8 = ceil(n_pairs/8) + 252, against a capacity of ceil(n_pairs/8) + 256. Four to spare. It is now `+ n_total_expert` so the bound is right by construction rather than by arithmetic that happens to work, but no overflow was possible |
| O4 | Fixed and re-verified | the grids span `n_total_expert` and the slack is `+ n_total_expert`. Expert-major: mean \|Δ\| **0.152** against the Metal reference (max 0.27, top-16 14-of-16), down from 1.485, and 0.132 against the warp path — the residual is the float-atomic accumulation order this path uses by design, not an error. Tile8 re-checked in the same pass: byte-identical to the previous run, so the slack change is behaviour-preserving |
| O5 | Swept for the rest of the class | no `dim3` uses 256 as an expert dimension, no `256 * {gate,up,down}_expert_bytes` / `256 * sizeof(int32_t)` / `256 * cap` sizing remains, and the dispatch's remaining 256s are weight-block sizes (`expert_in_dim / 256`), thread counts and grid rounding. Those three were the last instances |

**Still guarded, loudly and by name**: the MTP tok2 path and the scalar debug
path. tok2 is a small mechanical instantiation by the same recipe, but
verifying it needs a GLM MTP support model and there is none on this
workstation, so it keeps refusing Q4_K rather than running unverified —
speculative decoding must not be silently wrong.

### Phase P — WS 5, first slice: the swiglu clamp, and a parity test that passes

| # | Action | Evidence / result |
| --- | --- | --- |
| P1 | Found that the CUDA GLM MoE ignored the model's swiglu clamp | `ds4_gpu.h` and the only caller both pass `swiglu_clamp`, but the CUDA *definition* was missing that parameter (and a trailing `force_resident`). It worked at all only by ABI accident: on AArch64 the extra `float` lands in the FP argument registers and the trailing `bool` beyond the integer parameters, so the integer arguments stayed aligned. `ds4_metal.m` defines the same symbol with the full signature and only one backend is linked into a given binary, which is why nothing complained |
| P2 | Confirmed the clamp is live for this model | `DS4_SHAPE_GLM53.swiglu_clamp_exp = 10.0f`, and both `ds4.c`'s `swiglu()` and the Metal kernels clamp gate above and up on both sides. The CUDA routed-MoE kernels did a plain `silu(g)*u` at four sites, plus two in the scalar and tok2 variants |
| P3 | Fixed the signature and plumbed the clamp | `float swiglu_clamp` before `layer_index` and `bool force_resident` at the end, matching the header; a `glm_moe_swiglu` helper mirroring the CPU reference exactly; wired into all six gate/up kernels and passed at every launch site; `one_tensor` forwards both |
| P4 | Verified on real weights | plumbing the clamp changed the pair's next-token logits and moved them **closer** to the Metal reference: mean \|Δ\| **0.085 → 0.0727**, max **0.50 → 0.197**, top-16 **15/16 → 16/16**. The clamp had been binding, and the earlier agreement was hiding it |
| P5 | Landed the first slice of the WS-5 harness | `tests/test_glm53_moe_q4k.c` with `make test-glm53-moe-q4k`: a synthetic 288-expert MoE built through the public `ds4_gpu_*` API, compared against a host mirror of the q8_K quantizer, the Q4_K block dot and the MoE math. **PASS, bit-exact (`worst rel = 0.000e+00`)** for 8 tokens (warp / small-batch) and 128 tokens (tile8) |
| P6 | The test is built around the regressions rather than the fix | `mid` is filled with NaN before the call and the test asserts that no selected pair is left unwritten, with a `selected` that deliberately uses expert **287**, expert **256** (the first a 256-expert bound drops), expert **255**, a negative slot, and an expert nobody selects. It runs the warp/small-batch shape, the tile8 prefill shape, the expert-major shape (forced, since tile8 wins whenever it is eligible) and the tile8-off shape |
| P7 | Validated that it detects the fault | with the expert-major grid temporarily restored to `256u`, the test **fails**: `57 selected pairs had no mid row written` plus the explicit expert-287/256 checks. With the fix back it passes. Fails pre-fix, passes post-fix — which is the only thing that makes a regression test worth having |

**Scope note.** The value comparison is Q4_K only: the Q2_K device dot spans more
than the single 84-byte block per row that keeps the host mirror readable, and
Q2_K point arithmetic is already covered by `make q4k-dot-test` and the Q2_K pair
runs. The coverage checks are the type-independent part and the Q4_K case proves
them. The clamp is verified on real weights (P4) rather than synthetically -
making it bind on purpose needs weights whose dot deterministically exceeds it,
and that construction is worth doing separately.

### Phase Q — Self-audit: what I got wrong, and the tool that would have prevented it

Prompted by a direct question about fallacies in my own reasoning. The *fixes*
hold up — each was verified, and the expert-major grid one is now validated to
fail pre-fix and pass post-fix. The *reasoning* around them did not, in three
places, and the tooling failed often enough to matter.

**The load-bearing fallacy.** I reasoned "the tile8 and warp kernels are
templated over the weight type, therefore the difference must be in a
type-dependent piece — the block dot or the block stride". That silently assumes
both paths compute *the same work*, i.e. that shared code means shared inputs.
They do not: tile8 goes through the expert map, warp reads `selected` directly. A
defect in *shared* code — the 256 expert bound — could therefore hit one path and
not the other, and I had ruled that out by construction. Every check I then ran
(a bit-exact dot unit test, substituting one dot form for the other, bisection,
staging, determinism) was sound and aimed at the wrong layer. What would have
shortcut it is comparing the inputs the two paths *see*, or simply asking which
experts each visited — and I had built the logits harness before ever pointing it
at path coverage.

| # | Finding |
| --- | --- |
| Q1 | **Unsound inference from a bit-identity** (Phase M, row M6). "Q2_K tile8 ≡ Q2_K warp, byte-identical" was treated as proof that the tile8 machinery is sound for Q2_K. A bit-identity is evidence of nothing unless the two runs are known to have taken *different* paths, and I never established that. It was the pivotal wrong step. |
| Q2 | **An impossible explanation written as fact** (Phase N, row N5). "The Q2_K controls never routed to an expert ≥ 256" — over ≈192k selections, so P ≈ e^−22600. I should have noticed the number instead of writing the sentence. Withdrawn. |
| Q3 | **A cause asserted but never measured.** The 64-token greedy divergence is described as a near-tie. Plausible, but unverified — and the Q2_K calibration (identical text with *noisier* logits, 0.196 against 0.085) argues against the simple noise story rather than settling it, which is how I described it. The gap at the divergence point was never measured. |
| Q4 | **Mixed measurement bases.** The clamp fix is reported as "mean \|Δ\| 0.085 → 0.0727", comparing the pre-fix *vocabulary* mean with the post-fix *top-16* mean. Like-for-like (top-16): 0.1085 → 0.0727, max 0.3416 → 0.1972. The conclusion holds; the arithmetic as stated did not. |
| Q5 | **A performance regression I introduced and then rationalised.** After my own substitution experiment showed the two dot forms are bit-identical, I left the Q4_K `glm_moe_dot8` as n single-block calls with a comment framing it as the safer choice. It is the slower of two proven-equivalent expressions. Reverted. |
| Q6 | **Two more instances of the same ABI fault, one of them called.** Extending the signature check found the `direct_scalar_q4` stub missing `swiglu_clamp` — and unlike the batch entry, `ds4.c` calls it. Harmless in effect (the stub returns 0 either way), but it is a public API. The CUDA GLM MoE surface had **three** signature mismatches, all masked by AAPCS64 passing floats in the FP register file. |
| Q7 | **An over-claim already corrected** (Phase O): a tile-capacity slack called a latent overflow when it was sufficient — \`ceil(n_pairs/8) + 252\` against a capacity of \`+ 256\`. |
| Q8 | **Tooling carelessness, repeatedly**: `rsync -e "$SSH"` with the host included (twice), a `sed` that corrupted a script, and a Python heredoc with an unterminated string — so that a "measurement" ran against the *unfixed* binary and printed a meaningless IDENTICAL — plus reading a post-run log tail as if it were the run. Individually trivial; together they cost a substantial part of a session and produced one result I briefly took seriously. |

**Added: a path trace.** `DS4_GLM_MOE_TRACE=1` makes the GLM MoE dispatch print
one line per call — `type`, `tokens`, `experts`, `used`, `path`. It is what
finally answered the question above: this pipeline hands the worker 1,091 tokens
unchunked, so prefill runs **tile8** (21 calls) and decode runs warp (63 calls).
Q1, Q2 and the unexplained byte-identity would all have fallen out of that one
line at the start, which is the lesson worth keeping: when two paths disagree,
first establish that they took different paths.

**Also checked while here**, so that the negative results are recorded too: the
router is *not* bounded to 256 (it accepts up to 384 experts), and Metal's clamp
form matches my `glm_moe_swiglu` mirror exactly (`gate = min(gate, limit)`, `up =
clamp(up, ±limit)`) — so the clamp fix's arithmetic is right, not merely closer.

**Resolved — and it invalidates the Q2_K half of my reasoning.** The Q2_K model
never enters the GLM routed-MoE dispatch at all. Two independent pieces of
evidence settle it: its expert tensors are IQ2_XXS for gate/up and Q2_K for down
(types 16 and 10, read straight from the GGUF), which is exactly the signature of
the *other* dispatcher — `ds4_gpu_routed_moe_batch_tensor`'s `iq2_path` — not this
one; and a worker run with `DS4_GLM_MOE_TRACE=1` *confirmed present in its own
process* (the startup banner added in this phase printed it, which is why that
banner exists) produced this dispatch's per-call trace **zero** times while the
pair ran to completion.

The control is symmetric, which is what makes this conclusive rather than merely
suggestive: the identical script, identical `env` form and identical worker
binary run against the Q4_K model under the same switch give the banner *and*
42 trace lines. So the environment reaches both workers and the difference is
the model's dispatcher routing — not the harness, which is where I spent the
previous attempt.

So the Q2_K tile8 and warp dumps were byte-identical because **neither path ran
in either run**: the flag gated code that model never reaches. That is worse than
vacuous, and it retracts the Q2_K-based reasoning wholesale — "the tile8 machinery
is sound for Q2_K" (Phase M, row M6) never had a basis, and the 0.196 "Q2_K
calibration" is a valid whole-pipeline observation but says nothing about these
kernels.

What survives is everything measured on the Q4_K model, where the trace confirms
the dispatch runs — tile8 for prefill (21 calls at 1091 tokens), warp for decode
(63 calls at 1 token). The parity checks, the clamp improvement and the three
regressions all rest on that model. Which is also the point of the port: the GLM
dispatch appears to be exercised only by a model whose experts are Q4_K, and its
Q2_K paths are effectively dead for the files on hand — including the two that
still refuse Q4_K.

---

## 4. Measurements

### 4.1 Q2 pipeline, Mac coordinator (0:23) + Spark worker (24:output), over the tunnel

| ctx | tokens prefilled in that step | prefill | decode | first token | conditions |
| ---: | ---: | ---: | ---: | ---: | --- |
| 32 768 | 32 768 | 383.48 t/s | 12.82 t/s | 136 ms | uncapped clocks |
| 32 768 | 32 768 | 380.55 t/s | 12.48 t/s | 135 ms | **capped** (GPU ≤2100 MHz, CPU 2.4 GHz) |
| 65 536 | +32 768 | 364.41 t/s | 12.02 t/s | 134 ms | capped |
| 131 072 | +65 536 | 360.75 t/s | 11.42 t/s | 123 ms | capped |
| 403 351 | cold 403K ingest | **346.41 t/s** | **9.26 t/s** | — | capped, `--prompt-file`, ~19.5 min ingest |

Prefill is **depth-robust** (346 t/s at 403K against 380 at 32K), while decode falls
with depth (12.48 → 9.26 t/s) — the `max(stage)` / `sum(stage)` structure the plan
predicted. Worth noting for anyone reading intermediate progress lines: two ad-hoc
rate estimates taken during that ingest (≈90 t/s and ≈171 t/s) were artefacts of
imprecisely bounded sampling intervals; only the program's own reported figure
counts.

### 4.1b Topology comparison: pair vs one machine (same model, prompt, ctx)

Measured 2026-09-19 with the identical 403,350-token prompt at ctx 524288, so no
quantisation or depth confound:

| topology | prefill | decode |
| --- | ---: | ---: |
| Mac `0:23` + Spark `24:output` | **346.41 t/s** | **9.26 t/s** |
| Mac alone (whole model resident) | **186.89 t/s** | **19.71 t/s** |
| Spark alone (whole model resident) | *wedged* — see below | — |

This is the cleanest result of the session: the split is **1.85× on prefill and
0.47× on decode**, which is `max(stage)` and `sum(stage)` measured rather than
argued. Two consequences the earlier estimates missed:

* **The split's decode is halved, not merely "near the harmonic mean".** So its
  decode benefit over the *streaming* route comes only from streaming's own cache
  penalty. Redone with the measured factor: Q4 single-machine resident decode
  ≈ 19.71 / 1.76 (weight bits) ≈ 11.2 t/s, so pair-Q4 ≈ **5.3 t/s** against
  streaming's measured 4.63–5.02 t/s — **a wash**. My earlier "~1.3× decode" estimate
  did not include the topology penalty and was wrong.
* **The split is a throughput-versus-latency trade**, not a free speedup: better
  prefill, worse decode. For Q2 at 500K the single Mac is the better configuration
  outright (19.71 vs 9.26 decode); the pair is only compelling where no single
  machine can hold the model and the alternative is streaming.

**Whole-model Q2 at 512K freezes the Spark, and it reboots itself.** The previous
boot's own records give the proximate cause, which is the **GPU driver, not heat**:

```
Sep 19 22:03:51-22:03:58 spike kernel: NVRM: GPU0 nvCheckOkFailedNoLog:
  Check failed: Out of memory [NV_ERR_NO_MEMORY] (0x00000051)
  returned from _memdescAllocInternal(...)          ← repeated, ~20+ lines
Sep 19 22:03:58 spike kernel: NVRM: ... [last line in that boot's journal]
```

The engine's own guard had admitted the run — `required=98.88 GiB budget=115.19
GiB` on the Mac — so **the guard's budget is not sufficient for a whole-model CUDA
run on this box**: it is derived from system RAM, while NVRM reserves its own
overhead and refused the allocations outright. Sequence: cool idle board
(`board=55C gpu=50,9.54W slowdown=Not Active` at 22:03:58) → the driver failed to
allocate → **the journal stops dead at 22:03:58** → kernel still answering `ping`
0.4 ms and accepting TCP on `:22`, but no `sshd` banner at 60-second budgets from
22:14 → **abrupt reset at 22:23** (boot record; no shutdown entry at all).

What is *excluded*: the **watchdog** (SBSA `state: inactive`, `timeout: 10`,
raw `bootstatus = 0`, and `CARDRESET = 0` — it was never armed), a logged panic or
oops (none), any NVIDIA Xid (none — only the allocation failures), and the guard's
own abort (0 ABORTs).

What is *not* established: which unlogged mechanism performed the reset at 22:23 —
19 minutes after the freeze. The `acpitz` zones all carry a **104 °C critical trip**
that powers off by design with no log and no shutdown record, which fits both the
silence and the delay (`CARDRESET = 0` rules the SBSA card-reset path out); a
platform/firmware reset after the wedged allocation loop, and a power-delivery
protection event, are equally consistent. The box behaves exactly as its documented
failure mode describes — *"no panic, no OOM, no shutdown record"* — now with the
trigger visible for the first time.

**Both of the evening's freezes carry this signature.** The earlier one (boot -2,
journal ends 17:21:10, previously put down to "memory pressure from two resident
workers") has **six** `NV_ERR_NO_MEMORY` lines in its final seconds, exactly like
this one's twenty-four. So both outages are the **GPU driver refusing allocations
under over-subscription** — two resident workers then (≈86 GiB Q4 + ≈45 GiB Q2
against 121 GiB), one whole model at 512K now (~99 GiB planned, still refused) — and
neither is thermal. That is the failure to design around, and it is why the fix is
the split-slice shape rather than a tighter thermal band: the box has survived 92.07
GiB slices at depth and has not survived either whole-model arrangement.

Consequences for the plan: §9's "Q2 on one Spark resident" fallback is **not valid
at 512K**; the guard's admission number is **necessary but not sufficient** for
whole-model runs on this box; and the **split slices are the supported shape**
(94.88 / 92.07 GiB, which ran a 403K ingest at 67 °C with the machine responsive).
The Mac ran the identical workload to completion (186.89 / 19.71).

Cap cost at 32K: **−0.8 % prefill, −2.7 % decode** for ~20 °C of board margin.

### 4.2 Q4_K on the Mac alone (`--ssd-streaming`)

| ctx | prefill | decode | plan | expert cache |
| ---: | ---: | ---: | ---: | --- |
| 32 768 | 84.23 t/s | 8.84 t/s | 97.09 GiB | 5 435 / 12 384 (71.65 GiB) |
| 262 144 | 82.47 t/s (53 min ingest) | 8.00 t/s | 99.84 GiB | unchanged |

### 4.3 Spark thermals

| Condition | Board peak | Throttle evidence |
| --- | ---: | --- |
| uncapped, pipeline prefill | **90 °C** (10/60 samples ≥ 85 °C) | `HW Thermal Slowdown` accumulating; `SW Power Capping` 69 s |
| capped + guard, 32K→128K sweep | **81 °C** over 269 samples | **0** slowdown-active samples, 0 guard interventions |
| idle after install | 47–50 °C | GPU clock 1495 MHz (within the [1500, 2100] lock) |

### 4.4 Memory admission (measured with the engine's own guard)

| Slice | ctx | required | budget |
| --- | ---: | ---: | ---: |
| Mac coordinator `0:23` | 262 144 | 93.29 GiB | 115.19 |
| Mac coordinator `0:23` | 524 288 | 94.88 GiB | 115.19 |
| Spark worker `24:output` | 262 144 | 90.74 GiB | 103.63 |
| Spark worker `24:output` | 524 288 | 92.07 GiB | 103.63 |

### 4.5 Link

`en0` 10Gbase-T active; `enP7s7` 10000Mb/s full duplex; RTT 0.94–1.21 ms; single-stream TCP 4 GiB in 8.94 s ⇒ **0.46 GiB/s**; wire need ≈26 MB/s at 400 t/s. **Not the constraint.**

### 4.6 Q4_K at 512K single-machine: viability, depth cost, and MTP

Measured 2026-09-19 on this Mac, after the owner stated that **500K context is a
normal session build-up** — which makes the single-machine alternative to the
split a live question rather than a hypothetical one.

**It fits.** The memory guard admits 512K comfortably, because streaming keeps only
the active layer window resident:

```
GLM memory guard ctx=524288 required=13.09 GiB budget=115.19 GiB (model 4.08, graph 9.01, transient 0.00)
GLM memory guard ctx=524288 required=102.00 GiB budget=115.19 GiB (… transient 88.90)   ← prefill phase
compact DSA cache rows=524288 logical_ctx=524288 kv_layers=45 indexer_layers=11 f16 5.84 GiB
SSD streaming cache target 78.50 GiB = 70.90 GiB dynamic cache (5378 experts)
```

The whole 512K KV is only **5.84 GiB** (compact DSA), and the expert cache is
barely smaller than at 262K (5378 vs 5435 experts) — so depth costs almost nothing
in capacity. The tight number is the **prefill transient: 102.00 GiB of a 115.19
GiB budget, i.e. ~13 GiB of slack.**

**Depth costs decode, though — this is the finding that matters:**

| ctx | decode | note |
| ---: | ---: | --- |
| 262 144 | 8.00 t/s | §4.2, cached run |
| 524 288 | **4.63 / 4.90 / 5.02 t/s** | three runs, same prompt |

Streaming is cache-bound, so decode falls ~40 % from 262K to 512K with the cache
held at the same size.

**MTP does not rescue it.** At 512K, bracketed on the same prompt (plain before and
after, MTP in the middle, so cache warmth cannot explain it):

| run | decode |
| --- | ---: |
| plain (before) | 4.90 t/s |
| `--mtp --mtp-timing` | **3.88 t/s** |
| plain (after) | 5.02 t/s |

The draft is rejected in the visible cycles (`verify2 399.2 ms, head+draft
71.9 ms, reject (draft 25 ':' vs true 55798 '**')`): acceptance collapses at
depth while the verify cost remains, so MTP is a **21 % loss**, not a gain. This
matches the repo's own warning — "poor acceptance can make it slower. Measure your
workload rather than assuming" — and it removes the earlier hypothesis that MTP
would let the single-machine path outrun the pair at depth.

**Consequence for the split decision.** At 512K the single-machine route delivers
Q4 quality at 4.6–4.9 t/s decode and a ~100-minute cold 500K ingest (~80 t/s),
with no lever left to improve it. The pair's estimates are ~6.5 t/s decode and
~180–220 t/s prefill (`[INFERENCE]`: the Q2 pair at 131K scaled by weight bits),
i.e. **~1.3–1.4× decode and ~2× ingest**, with capacity measured at 524K
(94.88 + 92.07 GiB, §4.4) and no dependence on which experts happen to be cached.

### 4.7 Quality: Q2 vs Q4_K on the GLM 5.3 Flash 100-case fixture

Run 2026-09-19 on this Mac (M4 Max, Metal), one checkpoint at a time, same build,
same fixture (`gguf-tools/quality-testing/data/glm53-flash-openrouter-zai-fp8-100`),
greedy, ctx 4096 — Q2 resident, Q4_K with `--ssd-streaming` because 177.8 GiB cannot
be resident on 128 GiB. Scored with the repo's own `score_official`, compared with
its `compare_scores.py`.

```sh
./gguf-tools/quality-testing/score_official /Users/xun/mlmodels/glm/GLM-5.3-Flash-Q2.gguf \
  gguf-tools/quality-testing/data/glm53-flash-openrouter-zai-fp8-100/manifest.tsv /tmp/glm53-q2.tsv 4096
./gguf-tools/quality-testing/score_official /Users/xun/mlmodels/glm/GLM-5.3-Flash-Q4_K.gguf \
  gguf-tools/quality-testing/data/glm53-flash-openrouter-zai-fp8-100/manifest.tsv /tmp/glm53-q4.tsv 4096 --ssd-streaming
```

| Run | cases | avg NLL | first-token match | avg greedy lcp | wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q2, resident | 100 | 0.458177271 | 90 | 7.390 | ≈65 min |
| Q4_K, `--ssd-streaming` | 100 | **0.300477636** | **90** | **9.480** | ≈95 min |
| *published Q2 reference* | 100 | 0.458030488 | 89 | 7.37 | — |
| *published Q4 reference* | 100 | 0.299917952 | 90 | 9.66 | — |
| *M3 Ultra Metal reference, Q4 layout* | 100 | 0.300804038 | 90 | 9.48 | — |

Paired, case by case (`compare_scores.py /tmp/glm53-q2.tsv /tmp/glm53-q4.tsv`):

```
cases                        100
tokens                     11559
old_avg_nll           0.458177271
new_avg_nll           0.300477636
relative_nll_change       -34.419%
case_wins_new_old_ties  98    2    0
first_token_matches_old_new  90   90
avg_greedy_lcp_old_new  7.390  9.480
```

Reading:

* Both checkpoints reproduce their published bands on this host, and the Q4
  layout's Metal reference (M3 Ultra) is matched to three decimals on the greedy
  prefix — so the local pipeline is faithful and the comparison is neither
  backend- nor build-specific.
* **Q4_K is better on 98 of 100 cases**, with **34.4 % lower NLL** and the *same*
  first-token match: the gain is in the continuation, not the first token.
* **The gap does not require the split.** It is available single-machine with
  `--ssd-streaming` today, so it argues for Q4 as the target quantisation rather
  than for the two-machine pipeline. What the split adds is Q4 at 262K–500K fully
  resident, without dependence on the expert cache, at the ingest speed the Q2
  pipeline already demonstrates.
* The repo's SSD-streaming gate ("the summary should stay in the same quality
  band") is satisfied here: streaming changes no quality metric.

**Cost of the streaming gate on this host.** The first ~15 cases run at
≈2.5 min/case while the expert cache warms, then settle at ≈25–30 s/case —
measured over two windows (12 cases/300 s and 6 cases/180 s), not sampled once.
Total ≈95 min for the fixture against ≈65 min resident for Q2: a ~1.2× cost
concentrated in the cold ramp, not a per-case penalty. Two claims I made from a
single cold-window sample ("flat rate", "≈9× penalty") were wrong and this is the
corrected measurement.

---

## 5. Defects found, and their disposition

| Defect | Evidence | Disposition |
| --- | --- | --- |
| GLM 5.3 slice wire width (`N_EMBD` vs `N_HC×N_EMBD`) | three tape sites + git history | **fixed** (§2 #1–3), bit-exact parity |
| CUDA GLM routed MoE is Q2_K-only | `ds4_cuda.cu:32103`; `unsupported types 12/12/12` from both roles | **open** — the port in the plan |
| CUDA coordinator-side slice prefill crashes ≥512-row chunks | reproduced on pristine `8db1d1d` | **open, out of scope** while the Mac leads; workstream 10 |
| macOS ALF blocks inbound for adhoc-signed binaries | `cc` listener vs Apple `nc`; loopback exempt | **worked around** (launchd tunnel); ALF allow is the alternative |
| Metal kernel sources resolved only from CWD | `metal/…`, `./metal/…` candidates | **fixed** (§2 #4) |
| Spark hard-locks under sustained load | board 90 °C, `HW_THERMAL_SLOWDOWN`, unreachable; NVIDIA field-diagnostic PowerStress failure | **mitigated** (caps + governor); RMA declined 2026-09-19 — revisit only if a caps-armed run trips, which the plan's endurance gate is designed to surface |
| Guard: `set -u` exit on first `set_cap`; slow-down parse; initial `board=0C` | journal `board: parameter not set`, status `2` | **fixed** and stub-verified |
| Install hang: plymouth boot stall | `is-system-running = starting` for 17 min | **fixed** (unit ordering) + `set-default multi-user.target` |
| Snapshot over the tunnel | coordinator derives worker address from the socket = `127.0.0.1` | **fixed** — second forward `-L 55911:127.0.0.1:55911` plus a worker pinned with `--listen 127.0.0.1 55911`; save verified and the load path exercised, data sockets observed on 55911 (§11) |
| GLM 5.3 KDA layers made any containing slice unsizeable | `staging failed: distributed KV shard tensor size overflow`; `glm_layer_payload_tensor_bytes` rejected the uniform `compact_live` the header carries for every layer | **fixed** in `ds4.c` (guard removed, span is conv+recurrent state); snapshot round-trip then verified — see §11 |

---

## 6. Corrections to earlier conclusions

1. **Spark offline was thermal, not memory exhaustion.** My first hypothesis was
   unified-memory exhaustion from a second worker; the log and the public record
   say thermal trip. Corrected in Phase E2.
2. **"`ds4-server` lacks `--chdir`" was wrong.** It is parsed (undocumented in
   `--help`): `--chdir` alone → `missing value for --chdir`; with a value it is
   accepted. The user's V4.1 entry works as intended.
3. **"Spark needs ufw rules" was wrong for this direction.** Verified Mac→Spark
   inbound on a fresh port (9777) with `ufw` active — it passes. Only relevant if
   the Spark leads.
4. **"`gb10-thermal-status.sh` has a defect" was wrong** — the missing section
   was my `sed` range truncating the output, not the script.
5. **The tunnel was not failing from data volume.** During the 2026-09-19 wedge
   the working hypothesis was too much traffic through the tunnel. Measured:
   `en0` carried 0.1 KB/s in / 0.4 KB/s out, every socket to the Spark had empty
   queues, and the failing SSH logins never traverse the tunnel at all. The
   Spark's `sshd` completed TCP handshakes and never wrote a banner — a wedged
   userland, which needed a physical power cycle. Both machines are fine; the
   tunnel re-established itself with `KeepAlive` once the box was back.
6. **The thermal guard was eliminated, not assumed, as a wedge cause.** Every
   `nvidia-smi` call in `gb10-thermal-guard.sh` is already wrapped in `timeout 5`,
   so the sampler cannot accumulate stuck processes; the CPU cap is a ~14 %
   frequency cut. Two resident workers (a leftover plus a new one, ~45 GiB each
   over a 95 GiB mmap) remain the plausible trigger, which is why the plan now
   carries a clean-slate precondition.

### 6.1 Corrections from the self-audit (2026-09-19, after the fact)

7. **"Byte-identical output" was inferred, not measured.** The load was reported
   as reproducing the cold run's completion text. It was not: both requests were
   sent with `temperature 0`, so equal text was *expected*, and the observation
   recorded was only the usage block (`cached_tokens: 819`,
   `cache_write_tokens: 0`, `load=126.8 ms`). The claim appeared in three places
   and is now removed from all of them. Comparing the two completions belongs in
   the fresh-pair restore test, which is where it now sits.

8. **Criterion 6 was overstated as "verified".** The load ran on a *live* pair
   whose worker still held the same KV, so an identical result cannot distinguish
   "restore applied" from "restore was a no-op". What the run establishes is:
   the **save** is solid (165 MiB written, the worker's shard fetched over the
   data forward, no error), and the **load path was exercised** and reported a
   hit. Equivalence needs the restore on a fresh pair — already required by the
   plan's §6 item 5, and now the only thing standing between this and a verified
   round-trip.

9. **"KDA layers fall inside the coordinator's 0:23 slice" was wrong.**
   `ds4_glm53_layer_is_kda` is `il % 4u != 3u && il + N_NEXTN_PREDICT < N_LAYER`,
   so three of every four layers are KDA and they appear in **both** slices. The
   fix was still necessary and sufficient, but for a different reason: the
   **coordinator** sizes and parses *every* shard (its own and each worker's),
   while the worker only *writes* spans and never calls the sizing helper. So the
   worker's older binary was unaffected because of which code it runs, not
   because of which layers it holds. The conclusion survived; the stated reason
   did not.

10. **The wedge's memory arithmetic was wrong, and the thermal question is now
    settled by evidence rather than by elimination.** The leftover worker's HELLO
    said `quant=Q4` — and a Q2 model reports `quant=Q2` in the same field — so
    the leftover was a **Q4_K** worker holding ≈86 GiB, not a Q2 worker holding
    ≈45 GiB. With the new Q2 worker at ≈45 GiB against 121 GiB total, exhaustion
    is the mechanism, not a plausible candidate. The guard's own journal settles
    the rest: at 17:19–17:21 the board was **54 °C, GPU 52 °C, 8 W, 0 % util,
    `slowdown=Not Active`**; its 3-second loop then **stopped for 12 minutes**
    (17:22–17:33) while the kernel still answered ping and TCP; there were **zero
    ABORTs** and the evening's peak board was **74 °C**. That is userland
    starvation with a live kernel, and not heat — evidence that was on the box
    the whole time and should have been read during the incident, not after it.

    **The mechanism is now identified at the driver level.** That boot's own
    journal ends with **six** `NVRM: GPU0 … Out of memory [NV_ERR_NO_MEMORY]
    (0x00000051) … _memdescAllocInternal` lines at 17:21:07–17:21:10, immediately
    before logging stops — the same signature as the 22:03 freeze (24 such lines,
    log §4.1b). Both outages are therefore the **GPU driver refusing allocations
    under over-subscription**, not thermal events and not generic host-memory
    pressure: two resident workers the first time (≈86 GiB Q4 + ≈45 GiB Q2 against
    121 GiB), one whole model at 512K the second (~99 GiB planned, still refused,
    despite the engine's own guard admitting it).

11. **Two comparisons used numbers that are not comparable.** The plan's §10
    answer on MTP set "the capped pair decodes 12.48 t/s at 32K" against "8.00 t/s
    for single-Mac Q4 with SSD streaming at 262K". That is Q2 versus Q4 *and* 32K
    versus 262K, so it cannot support a statement about the Q4 split's decode at
    depth. The decision (no MTP) is unaffected — it rests on MTP being excluded
    under a layer split — but the evidence cited for it was confounded. The
    honest state: **the Q4 pair's decode and prefill at 262K/524K do not exist
    yet**, blocked behind WS 1–2, and the plan's §6.5/§6.6 runs are what will
    produce them. The same confusion is latent in §9's fallback table, which pits
    Q2-pair numbers against Q4-single numbers.

12. **The ALF conclusion was slightly overclaimed.** What is established: the
    per-app allow did not take effect for an adhoc/linker-signed binary, and the
    blocking layer sits above `socketfilterfw` (the app was listed as *Allow
    incoming connections* while SYNs still dropped, and Apple-signed `nc` passed
    on the same address). What was *not* established: that no GUI route exists —
    Local Network privacy was inferred, not tested, because the tunnel already
    worked and is the better answer anyway (an Apple-signed `sshd` owns the
    socket). The docs now say "not pursued to a conclusion" rather than
    "unavailable".

13. **The SSD-streaming cost was mis-stated from one cold-window sample.** I
    reported "a flat ≈2.5 min/case, the cache is not accelerating it" and, from
    that, a "≈4 h" run. Measured across two later windows it is 25–30 s/case once
    the cache warms, ≈95 min for the fixture, against ≈65 min resident for Q2
    (§4.6). This is the same failure as #7 — generalising from a single
    observation — and it is the second time in this work, which is why the
    measurement convention now says: sample twice before stating a rate.

---

## 7. Artifacts and host configuration

### On the Mac

| Path | What |
| --- | --- |
| `~/dev/ds4/` | source tree, branch `customisation` (fork point `8db1d1d`); §2 #5–6 were uncommitted when the binaries were rebuilt |
| `~/dev/ds4/docs/custom/ds4-glm53-q4-split-design.md` | the plan |
| `~/dev/ds4/docs/custom/ds4-glm53-q4-implementation-log.md` | this log |
| `~/bin/` | `ds4`, `ds4-server`, `ds4-agent`, `ds4-bench`, `ds4-eval` (rebuilt 2026-09-19 17:46) + `metal/` (26 kernels) |
| `~/ds4-tunnel/` | tunnel kit: daemon + agent plists, `install.sh`, `install-agent.sh`, `uninstall.sh`, `README.md` |
| `~/ds4-deploy.sh` | rebuild + install to `~/bin` on this host or the peer; `check` verifies source parity (`ds4.c` md5) and prints both hosts' binaries and process state |
| `~/thermal-protect/` | backup copy of the Spark protection kit |
| `/Library/LaunchDaemons/com.local.ds4-tunnel.plist` | installed by the user; job `com.local.ds4-tunnel`, runs as `xun` |
| `~/Library/Logs/ds4-tunnel.log` | tunnel log; the `banner exchange` retries from the 2026-09-19 wedge are expected, and it is silent while the peer answers |

### On the Spark (`192.168.2.2`)

| Path | What |
| --- | --- |
| `~/dev/ds4/` | source tree content-matched to the Mac's (`ds4.c` md5 `98c92891…` on both), on branch `main` at `8db1d1d` |
| `~/bin/` | the same five binaries, rebuilt 2026-09-19 from that tree with `make -j20 cuda-spark` (CUDA `sm_121`) |
| `~/thermal-protect/` | kit source, README synced with the Mac backup |
| `~/ds4-deploy.sh` | same script (md5 identical to the Mac's); `local` here rebuilds `cuda-spark` |
| `/usr/local/bin/gb10-thermal-guard.sh`, `gb10-cpu-cap.sh`, `gb10-thermal-status.sh` | installed |
| `/etc/systemd/system/gb10-thermal-guard.service`, `gb10-cpu-cap.service`, `gb10-cpu-cap.timer` | installed, enabled |
| `~/mlmodels/glm/` | Q2 + Q4 GGUF + vision encoder |

Services: `gb10-thermal-guard.service` (active, `Restart=always`),
`gb10-cpu-cap.service` + `.timer` (re-applies the CPU cap every 5 min).
Default target changed to `multi-user.target` (headless).

---

## 8. Open decisions

1. ~~**IQ2_XXS in the same port pass, or Q4_K only?**~~ Answered 2026-09-19:
   **Q4_K only.** IQ2_XXS remains WS 11, after the Q4_K milestone clears QA; its
   mixed trio (IQ2_XXS gate/up with a Q2_K down) is why it is not a free
   addition to the same pass.
2. ~~**Snapshots:** accept the tunnel limitation, or take the ALF-allow route with
   explicit binds, or add the worker's reachable host to HELLO (protocol change)?~~
   Answered 2026-09-19: none of those. The tunnel carries the data connection too
   (`-L 55911:127.0.0.1:55911`) with the worker pinned (`--listen 127.0.0.1 55911`),
   and save plus load are verified (§11). The ALF route is closed (plan §3.3) and
   no protocol change is needed.
3. ~~**`make install`** target so rebuild → `~/bin` is one command?~~ Answered
   2026-09-19: solved **outside** the Makefile, as `~/ds4-deploy.sh` (host-level,
   like the tunnel and thermal kits). It syncs the source to the peer, picks the
   right backend per host (Metal here, `cuda-spark` there), and refuses to replace
   a running binary without `STOP=1`. Upstream's Makefile stays untouched, so a
   rebase against `antirez/ds4` cannot conflict on it.
4. ~~Confirm 500K must hold **without** MTP (excluded in distributed mode by
   `ds4_engine_has_mtp`), i.e. decode stays at ~11–14 t/s.~~ Answered 2026-09-19:
   **yes** — decode stays ~11–14 t/s and MTP remains excluded under the split;
   enabling it there is design work (taking the head and its routing across a
   slice boundary), not a flag.
5. ~~**The `DS4_GLM_GENERIC_MOE_Q4K` experiment (§2 #6):** drop it now that the
   result is measured, or keep it until WS 2 lands as a comparison point?~~
   Decided 2026-09-19: **keep until WS 2 lands**, then remove or promote
   deliberately. It is inert unless the variable is set, it is documented as an
   experiment rather than a path, and it is the only artefact of the Q4_K-on-CUDA
   comparison — so the cost of keeping it briefly is lower than the cost of
   destroying the comparison before the kernels that supersede it exist.

---

## 9. Next steps (from the plan)

| WS | Work | Effort |
| --- | --- | --- |
| 1 | type traits + dispatch predicate for GLM MoE (`{Q2_K, Q4_K}`), loud failure otherwise, `DS4_CUDA_GLM_MOE_TYPES=q2k` escape hatch | 0.5–1 d |
| 2 | Q4_K prefill instantiations (tile8 gate/up, down terms + reduce) — **first measurable milestone** | 2–3 d |
| 3 | Q4_K expert-major gate/up + down | 1–2 d |
| 4 | Q4_K decode instantiations (warp-per-pair, tok2-reuse, down warp, small-batch) | 2–3 d |
| 5 | CPU/GPU parity harness for the GLM MoE Q4_K path | 1–2 d |
| 6 | cross-machine oracle (pipeline vs single-host, logits + `--dist-replay-check`) | 1–2 d |
| 7 | boundary gates (2 048→2 056, 4 096→4 100), snapshot round-trip | 1–2 d |
| 8 | long-context endurance (262K ingest, 524K alloc) with peak board logged per frontier | 1–2 d |
| 9 | docs + release gates | 0.5–1 d |
| 10 | *(optional)* CUDA coordinator-side slice prefill fix | 1–3 d |
| 11 | *(optional)* IQ2_XXS instantiations | 1–2 d |

Critical path WS 1 → 2 → 5 → 6 ≈ 1–1.5 weeks; full set with QA and docs ≈ 2–3 weeks.

---

## 10. Quick acceptance commands

```sh
# tunnel (Mac)
launchctl print system/com.local.ds4-tunnel | grep -E "state|pid|last exit"
ssh 192.168.2.2 'ss -tln | grep 9911'

# thermal protection (Spark)
ssh 192.168.2.2 'systemctl is-active gb10-thermal-guard.service gb10-cpu-cap.service'
ssh 192.168.2.2 '/usr/local/bin/gb10-thermal-status.sh | head -20'
ssh 192.168.2.2 'journalctl -u gb10-thermal-guard -n 5 --no-pager'

# binaries work from any directory
cd /tmp && ~/bin/ds4 --inspect -m ~/mlmodels/glm/GLM-5.3-Flash-Q2.gguf | head -3

# working cross-machine configuration (Q2)
ssh 192.168.2.2 'pkill -x ds4; (setsid nohup ~/bin/ds4 --cuda -m ~/mlmodels/glm/GLM-5.3-Flash-Q2.gguf \
  --role worker --layers 24:output --coordinator 127.0.0.1 9911 --listen 127.0.0.1 55911 --ctx 524288 >/tmp/w.log 2>&1 </dev/null &)'
cd /tmp && ~/bin/ds4 -m ~/mlmodels/glm/GLM-5.3-Flash-Q2.gguf --role coordinator --layers 0:23 \
  --listen 127.0.0.1 9911 --ctx 32768 --temp 0 -n 32 -p "your prompt"

# long one-shot runs: detach from the terminal
#   a `nohup ... &` started from an agent/SSH PTY can leave ds4 blocked on
#   /dev/ttys000 with the model unloaded and the log frozen (seen 2026-09-19 with
#   --prompt-file). Feeding the prompt on stdin from a file works, as does
#   `ssh -T localhost` or a launchd job. To tell a working run from a wedged one,
#   check that the LOG IS GROWING - `ps -o pcpu` reads 0.0% while prefilling
#   normally, because the work is on the GPU.

# snapshot acceptance (distributed checkpoint save + load, §11)
~/bin/ds4-server -m ~/mlmodels/glm/GLM-5.3-Flash-Q2.gguf --role coordinator --layers 0:23 \
  --listen 127.0.0.1 9911 --ctx 4096 --host 127.0.0.1 --port 18080 \
  --kv-disk-dir /tmp/ds4kv --kv-cache-min-tokens 512
#   a cold prompt above min-tokens forces a save; re-sending it must report
#   cached_tokens == prompt_tokens and no cache_write_tokens
curl -s http://127.0.0.1:18080/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.2","prompt":"<prompt above 512 tokens>","max_tokens":4,"temperature":0}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["usage"])'
ls -la /tmp/ds4kv/          # one .kv entry per cold checkpoint
```

---

## 11. Distributed snapshot round-trip (2026-09-19)

Acceptance criterion 6 is **half met** for the `0:23` / `24:output` split: the
save is verified, the load path was exercised, and equivalence is not yet shown
(§6.1 #8).

**Run used.** Mac: `ds4-server --role coordinator --layers 0:23 --listen
127.0.0.1 9911 --ctx 4096 --host 127.0.0.1 --port 18080 --kv-disk-dir /tmp/ds4kv
--kv-cache-min-tokens 512`. Spark: `ds4 --cuda --role worker --layers 24:output
--coordinator 127.0.0.1 9911 --listen 127.0.0.1 55911 --ctx 4096`.

**Evidence**

* The worker registered with the pinned port — `registered worker
  127.0.0.1:60159 data_port=55911 … ctx=4096`. Before `--listen` was passed it
  advertised an ephemeral port (39215), which no static forward can reach.
* Cold save: `kv cache stored tokens=649 trimmed=0 reason=cold size=161.02 MiB
  save=18.2 ms`, writing `bf072cd0…kv` (173,096,280 bytes).
* The data path was genuinely used: sampling `netstat` every 10 ms during the
  save caught `127.0.0.1.55911 ↔ 127.0.0.1.60168/60170 ESTABLISHED` — the
  tunnel's `-L` listener, so the worker's slice crossed the link.
* Load: re-sending a cached prompt reported `cached_tokens: 819,
  cache_write_tokens: 0` and `cache hit … load=126.8 ms`. The **completion text
  was not compared** — the two requests shared a prompt and `temperature 0`, so
  equal text was expected and never checked. That comparison belongs in the
  fresh-pair restore (§6.1 #7).

**Defect cleared to get there.** The first attempt failed before opening any data
connection: `kv cache skipped tokens=819 reason=cold because KV payload staging
failed: distributed KV shard tensor size overflow`. Root cause: a layer-payload
header carries **one uniform `compact_live` for every layer of the slice**, and
`glm_layer_payload_tensor_bytes` rejected exactly that combination for KDA layers
— so any slice containing a KDA layer became unsizeable as soon as a context
existed. A KDA layer holds no KV rows: its span is the conv state plus the
recurrent state, independent of those counts. The guard is removed and `*out != 0`
is asserted instead. This also affected `ds4_session_layer_payload_bytes`, i.e.
the single-node sizing path, not just distributed snapshots.

**Rebuilt since.** Both hosts now run binaries built from this fix (§3 J9), so the
roles-swapped restore in plan §6 item 5 no longer needs a separate build, and the
round-trip was re-verified on those binaries: cold save `size=164.89 MiB
save=18.1 ms`, six data-socket observations on `127.0.0.1:55911`, then `cache hit
… load=126.8 ms` with `cached_tokens: 819`. What remains untested is the restore
on a **fresh** pair and with the roles swapped — this section covers save and load
on a live pair with the `0:23` / `24:output` split, which is the configuration the
goal uses.

One log line to read correctly when it appears: `kv cache skipped tokens=818
reason=evict … session has no valid checkpoint to stage` is a benign eviction
no-op after a hit, not the defect — the defect's message is
`distributed KV shard tensor size overflow`.

---

## 12. Q2 standalone regression check (baseline comparison)

The question this answers: did any of the GLM Q4_K work break the **existing** Q2
paths on a single machine? Q2 is the shipped sparse recipe and shares a binary
with the Q4_K work, so it had to be checked, not assumed.

**Where the changed code actually lives.** All 67 hunks in `ds4_cuda.cu` are
inside GLM-named functions — `ds4_gpu_glm_routed_moe_batch_tensor`,
`..._one_tensor`, `..._direct_scalar_q4_tensor` and their `glm_routed_moe_*` /
`glm_moe_*` kernels — with **zero** non-GLM hunks. Those entry points have
exactly one non-test caller each: `ds4.c:48508`, `48647`, `48678`, all inside the
GLM branch of the layer-MoE dispatch. `ds4.c`, `ds4_distributed.c`,
`ds4_metal.m`, `ds4_tp.c`, `ds4_gpu.h`, `ds4_agent.c` and `ds4_server.c` are
**byte-identical to baseline**. The Makefile change only adds a test target;
`ds4_cli.c` adds the env banner, which writes to stderr only and only when one of
the nine switches is set.

**Why Q2 never reaches the changed code.** The dispatch decides by tensor type:

```c
/* IQ2_XXS gate/up with a Q2_K down is the shipped sparse recipe and has
 * always been served by the generic routed-MoE dispatch. */
if (l->ffn_gate_exps->type == DS4_TENSOR_IQ2_XXS) return true;
```

`glm_graph_layer_uses_generic_routed_moe` returns true for that recipe, and the
caller then returns `ds4_gpu_routed_moe_batch_tensor` — the generic dispatcher,
untouched by this work. Only a homogeneous **Q4_K** trio (type 12) falls through
to the GLM-specific path that was fixed. That is also the mechanical explanation
for the zero trace lines in the Q2_K runs discussed above.

**Measured, not argued.** A baseline binary built from `8655de7` (the commit
before this work, verified to differ from the current build) was run against the
current binary — same model, same prompt, greedy decode, one machine with no
`--role`, i.e. the standalone path — comparing generated text byte-for-byte:

| model | baseline | current | result |
| --- | --- | --- | --- |
| GLM 5.3 Flash Q2 | 348 bytes, exit 0 | 348 bytes, exit 0 | identical |
| DeepSeek V4 Flash Q2 | 347 bytes, exit 0 | 347 bytes, exit 0 | identical |

`--dump-tokens` is *not* a valid check here: it tokenizes the prompt and exits
before decoding, so it exercises none of this path. The comparison above is on
generated text, which runs the decode path.

**V4.1 is not measured, and here is exactly what covers it.** No V4.1 language
model exists on either machine: `/Users/xun/mlmodels/deepseekv4/` holds only the
970 MB `DeepSeek-V4.1-Flash-Vision.gguf` encoder, and the Spark holds no V4.1
file at all. So a V4.1 Q2 run was not possible. What makes it safe is that every
file implementing V4.1 behaviour (`ds4_engine_is_deepseek41` in `ds4.c`, the
DSML4.1 syntax in `ds4_agent.c`, the model id in `ds4_server.c`, `ds4_tp.c`) is
byte-identical to baseline, and a V4.1 Q2 model routes by the same type-based
rule to the same untouched dispatcher. That is inference from identical inputs,
not a measurement — if a V4.1 Q2 GGUF appears, repeat the check above against it.

---

## 13. Which Q4_K dispatch is the default, and why (2026-09-20)

**The short version: the ported GLM-specific Q4_K kernels are correct but slow,
and the pre-existing generic dispatch is 2.7× faster at prefill. Q4_K now routes
there by default, and criterion 3's prefill bar is met for the first time.**

**Why both paths had to be measured before removing the spike.** Plan §8 said to
"drop it when [WS 1–2] land, or promote it deliberately — do not leave two
dispatch paths unexamined", and §8's own note had already argued *against*
destroying the comparison early, because the spike was "the only artefact of the
Q4_K-on-CUDA comparison". So the gate came out only after the two were run against
each other. That was the right order, because the result contradicted the
workstreams' premise.

**The measurement.** Same pair, same 1091-token prompt, greedy `-n 64`, the only
difference being which dispatch each host's layers take. The GLM dispatch prints a
per-call trace, so the trace doubles as the discriminator that the intended path
was really taken — 0 lines means the generic dispatch ran, and a non-zero count
means the GLM one did. That matters: an earlier comparison in this project was
void precisely because a switch silently failed to apply.

| config | Mac | Spark | Spark trace | prefill | decode | 64-token output |
| --- | --- | --- | ---: | ---: | ---: | --- |
| A | GLM-specific (Metal) | GLM-specific (CUDA) | 1344 | **95.25 t/s** | 11.18 t/s | reference |
| B | GLM-specific (Metal) | **generic** (CUDA) | **0** | **257.42 t/s** | 11.26 t/s | identical |
| D | **generic** (Metal) | **generic** (CUDA) | **0** | **258.92 t/s** | 11.37 t/s | identical |

All three outputs are **byte-identical over 64 greedy tokens** (300 bytes each).
D is the configuration an unconditional rule produces, which is why it was run: the
predicate lives in shared `ds4.c`, so promoting the spike also moves the **Metal**
side, and that side is what this project spent all its time validating against.

**Why the generic path wins, mechanically.** Not tuning luck — tensor cores. The
generic dispatch's Q4_K path selects `moe_gate_up_mid_q4K_tile16_mma_kernel<512>`
and `moe_down_q4K_tile16_mma_kernel<512>` (`ds4_cuda.cu` ~25236 / ~25711, gated by
`use_q4_mma_tiles16`). The WS 2/3 port instantiated `glm_routed_moe_*_tile8_*`
kernels, which do not use MMA. On a GB10 that is the whole difference.

**What changed.** `glm_graph_layer_uses_generic_routed_moe` now returns true for a
homogeneous Q4_K trio unconditionally; the `DS4_GLM_GENERIC_MOE_Q4K` gate is gone.
The GLM-specific Q4_K kernels remain reachable through `DS4_CUDA_GLM_MOE_TYPES`
and are still covered directly by `make test-glm53-moe-q4k`, so the port is not
dead code — it is a reference implementation and a fallback, not the default.

**Verified on the promoted default, with no routing variable set anywhere:**

| check | result |
| --- | --- |
| Spark GLM-dispatch trace | **0** — the generic dispatch is now the default |
| Mac startup banner | names no routing switch (proves nothing was set) |
| prefill / decode | **257.71 t/s** / 11.34 t/s (was 95.25 / 11.18) |
| 64-token output | byte-identical to run A |
| `make test-glm53-moe-q4k` | PASS, `worst rel = 0.000e+00`, exit 0 |

**Consequences, stated plainly.**

1. **Criterion 3's prefill bar is met.** It required ≥ 150 t/s at 32K on the pair;
   the ported path measured 100.57 t/s and could not reach it, and the promoted
   default measures 257.71 t/s at ctx 8192. The 32K-context confirmation is the
   remaining formality.
2. **WS 2/3's premise was wrong, and the log should say so rather than bury it.**
   Those workstreams existed to give the GLM-specific dispatch Q4_K support so it
   could supersede the spike. The kernels are correct and now covered by a parity
   test, but the dispatch was never the bottleneck the plan treated it as: a
   cheaper path already existed and is faster. The plan's judgement that the spike
   was a stopgap to be superseded was reasonable on the evidence available, and
   measuring both is what settled it.
3. **The next lever is load balance, not the Mac.** B (257.42) and D (258.92) are
   within noise of each other, so the Mac's routing choice does not move
   throughput at all — the Spark's stage is the critical path. Prefill on a
   pipeline is `max(stage)`, and the split currently gives the Mac 24 layers and
   the Spark 22; giving the Spark fewer should raise the maximum. That is
   measurable with the existing harness and is the obvious next experiment.

---

## 14. Criterion 3 measured at its stated scale (2026-09-20)

Criterion 3 asks for "**≥ 150 t/s prefill and ≥ 10 t/s decode at 32K on the
pair**". Everything measured up to this point was a 1091-token prompt at ctx
8192, where pipeline fill/drain dominates the average and the number understates
what the pair does on a real ingest. This is the measurement the criterion
actually asks for: **ctx 32768, a 28 657-token prompt**, greedy `-n 16`.

| run | order | flags | prefill | decode |
| --- | --- | --- | ---: | ---: |
| base | 1st (cold) | — | 340.49 t/s | 10.13 t/s |
| bits16 | 2nd | `--dist-activation-bits 16` | 389.03 t/s | 10.21 t/s |
| chunk4096 | 3rd | `--dist-prefill-chunk 4096` | 389.21 t/s | 10.29 t/s |
| window8 | 4th | `--dist-prefill-window 8` | 389.28 t/s | 10.35 t/s |
| **base (control)** | **5th** | **—** | **389.04 t/s** | 10.21 t/s |

**Criterion 3 is met, with margin: 389.0 t/s against a 150 t/s bar (2.6×).**

**The three tuning flags do nothing, and the control is what proves it.** The
first four runs make it look as though `--dist-activation-bits 16` and the two
chunking flags each buy ~14 %. They do not: base run **last, with no flags at
all**, lands at 389.04 t/s — indistinguishable from the three "tuned" runs. The
spread is run order: the first run after an idle period is ~14 % slower (cold
caches, clocks still ramping) and everything thereafter converges to 389.0 ± 0.3
t/s. Without that fifth run I would have reported three gains that do not exist,
and the plan would have carried a tuning recommendation that is noise.

**Output is unaffected by all four configurations** — byte-identical across base,
bits16, chunk4096 and window8. That matters for `--dist-activation-bits 16`
specifically, since it *does* change wire numerics by design; on this prompt it
did not change the greedy output. It is still not worth enabling: no measured
gain, and a documented numerical change is a cost with no benefit here.

**Measurement caveat worth carrying forward.** The warm-up spread is ~14 %, which
is larger than any effect the tuning knobs might have. Single-run comparisons
below ~15 % are not resolvable on this pair; anything claiming a smaller win needs
repeats, and the first run after an idle period should be discarded or repeated.

**Decode is met but thin.** 10.21–10.35 t/s against a ≥ 10 t/s bar is a 2–3 %
margin, and decode falls with depth (the plan's own single-Mac figures drop from
8.00 t/s at 262K to 4.63–5.02 at 512K). This measurement is at 32K, so the bar is
met *at the scale the criterion states* — but it should not be read as headroom at
262K or 524K, where criterion 4's runs will land.

**What actually fixed it.** Not tuning: the routing. The ported GLM-specific Q4_K
kernels were the default and are 2.7× slower per layer (§13); promoting the generic
dispatch took the same configuration from ~95 t/s to 389 t/s at 32K. The 32K
prompt then removed the fill/drain effect that made the 1091-token number look
worse still.

---

## 15. Load balance: the lever is real, and the first test of it was invalidated by a NIC fault (2026-09-20)

**Per-stage telemetry, ctx 32768, 28 657-token prompt** (`--debug`, coordinator's
view of the worker; the worker's own lines are not emitted, so the Mac's stage is
derived). Seven prefill chunks of 4096 (one 4081):

```
request=1 hop=0 layers=24:44 pos=0     tokens=4096 eval=5189.744ms input=256.02MiB
request=2 hop=0 layers=24:44 pos=4096  tokens=4096 eval=5466.542ms input=256.02MiB
...                                                       (chunks 3-6: 5498-5551ms)
request=7 hop=0 layers=24:44 pos=24576 tokens=4081 eval=5603.423ms input=255.08MiB
request=8 hop=0 layers=24:44 pos=28657 tokens=1    eval=59.221ms  input=0.06MiB   <- decode
```

| stage | layers | per chunk | per layer |
| --- | ---: | ---: | ---: |
| Spark (CUDA worker) | 21 | ~5.50 s | ~0.262 s |
| Mac (implied, coordinator) | 24 | ~9.75 s | ~0.406 s |

Total 28657 / 389.05 = 73.7 s for 7 chunks; the worker accounts for 43.4 s of
evals, so the coordinator's stage is the slower one and prefill is bound by it.

**Two consequences.**

1. **The wire is not a factor, which independently explains §14's null result.**
   Each chunk carries 256.02 MiB of activations (~0.23 s at 10GbE) against a
   5.5 s stage — about 2 %. So `--dist-activation-bits 16` *could not* have shown a
   measurable gain, and the telemetry says so independently of the run that
   measured nothing.
2. **Rebalancing should pay ~17–25 %** — in the direction of moving layers *from*
   the Mac *to* the Spark (21/25 and 18/28 were the predicted ~8 % / ~17 % / ~25 %
   steps), and that is also the memory-safe direction for the Mac.

**What actually stopped the run was the network, not the memory — and my first
account of this was wrong.** I originally attributed it to the 25-layer memory
footprint exhausting the box, and said so. The evidence contradicts that, and the
correction matters because it changes both the cause and the conclusion.

The previous boot's kernel journal (recovered after the reset; the application log
was in tmpfs and was lost) says:

```
13:08:05 kernel: r8127: enP7s7: link down
13:08:10 kernel: r8127: enP7s7: link up
13:08:15 r8127: enP7s7: link down   / systemd-networkd: enP7s7: Lost carrier
13:08:19 r8127: enP7s7: link up     / Gained carrier      (5 cycles, ~32 s)
13:08:33 r8127: enP7s7: link down
13:08:37 r8127: enP7s7: link up     <- last link event of the boot
```

- **No OOM kill was ever invoked that boot** (`oom mentions: 0`).
- **The box was never wedged.** It stayed alive and healthy throughout: cron ran at
  13:15:01 and 13:17:01, `systemd-resolved` logged continuously, the thermal guard
  kept reporting `board=50C gpu=48, 2093MHz, 7.80W, 0% slowdown` — idle and cool,
  which also rules out a thermal trip and a compute crash.
- What broke was the **direct 10GbE link**: `enP7s7`, driver `r8127`, carrying
  `192.168.2.2/24`, flapped five times in ~32 seconds starting at **13:08:05**. It
  recovered at 13:08:37, but the host's network state did not — NetworkManager went
  to `CONNECTED_SITE` at 13:11:33 and DNS to 192.168.0.1 degraded on a loop — so the
  box stayed off the network until the reset. `ping` failing while the LAN gateway
  answered in 0.86 ms is exactly what a link-layer fault at this end looks like; I
  read it as "the box is gone" when it was "the box is up and off the network".

**So the rebalance failure is explained without invoking memory at all.** The b21
worker started at 13:08:04 (the last successful ssh login logged), inside the flap
window, and the coordinator never saw its slice — `distributed route incomplete:
missing layer 21` is what a worker that cannot hold a link to the coordinator
produces. The 25-layer memory question was never actually exercised.

**The bound I stated is withdrawn.** "The Spark cannot hold more than ~22 layers,
so the balance lever is closed" does not follow from this evidence and is not
established — the experiment that would have tested it was invalidated by the NIC.
The lever is therefore **still open**, and the prediction above (Mac→Spark
rebalancing worth ~17–25 %, since the Mac's 0.406 s/layer against the Spark's
0.262 binds prefill) has not been tested. It should be re-run now that the link is
healthy — with the memory budget checked *before* the run, which is the part of the
original caution that still stands.

**Re-run, and the lever is confirmed: +13.2 %.** With the NIC fault ruled out (link
events constant at 1 across every run) and the Spark's memory watched live through
each run, the same split comparisons now complete:

| config | Mac layers | Spark layers | prefill | min free RAM on the Spark |
| --- | ---: | ---: | ---: | ---: |
| base | 24 | 22 | **389.16 t/s** | 24 GiB |
| **21 / 25** | 21 | **25** | **440.42 t/s** | 12 GiB |
| base (control, run last) | 24 | 22 | **389.16 t/s** | 24 GiB |

**+13.2 %** — the telemetry predicted ~13 % from `Mac 21 × 0.406 s = 8.53 s/chunk`
against base's 9.75, and that is what it delivers. So prefill on this pair is
**440 t/s**, 2.9× criterion 3's 150 t/s bar, by moving three layers from the Mac to
the Spark.

**This also refines §14's caveat, which was too pessimistic.** The two base runs are
identical to **0.01 %** (389.16 both). The ~14 % spread seen earlier was the
cold-first-run effect specifically, not general run-to-run noise: a warm baseline
reproduces that tightly. So small differences *are* resolvable here — which makes
the tuning-flags null result *stronger*, not weaker: `--dist-activation-bits 16`,
`--dist-prefill-chunk 4096` and `--dist-prefill-window 8` each landed within 0.1 %
of the base control, i.e. they were measured at high precision and genuinely do
nothing.

**Memory is now the binding constraint, and that is what stops the sweep.** At 25
layers the Spark has **12 GiB** free (24 GiB at 22 layers) — roughly one more layer
of headroom, not three. The remaining predicted step (19/27, ~487 t/s) would leave
low single-digit GiB and was deliberately **not** run: the last attempt to grow this
slice already cost a hard reset, and the gain left on the table (~+5–10 %) does not
justify repeating that risk. The recommendation is therefore **Mac `0:20` / Spark
`21:output`** — the measured 440 t/s with 12 GiB to spare — and any further gain has
to come from making the Mac faster per layer, not from moving layers onto a machine
that is already at 90 % of its RAM.

**A harness bug of mine aborted the first re-run and is worth recording**, because it
looked exactly like a system failure: I had added `missing layer` to the early-abort
pattern, but `waiting for distributed route: distributed route incomplete: missing
layer N` is a *normal transient* while the worker registers. All three runs
(including the base configuration that had just worked) were killed by my own script
before the route completed — the coordinator's log ended with `distributed route
ready` and the worker's with `coordinator disconnected`. The evidence that cleared
the system was the instrumentation added after the earlier misdiagnosis: link events
stayed at 1 (so not the NIC) and the worker had loaded `resident model 86.32 GiB =
89.39 GiB planned` (so not memory).

**A hardware item worth watching.** `r8127` (Realtek 10GbE) flapping under
sustained load is a known class of fault — cable, connector, 10GBASE-T thermal
behaviour or EEE/ASPM. It happened once, at the start of a heavy transfer, and has
not recurred since the reset (1 link event per boot, the boot-time one;
`Speed: 10000Mb/s, Duplex: Full, Link detected: yes`, and the count held at 1
through every run above). If a future run loses the peer, check
`journalctl -k | grep r8127` **before** assuming a wedge: this incident cost a hard
reset and a wrong root cause because I diagnosed from reachability instead of from
the kernel log.

---

## 16. The tunnel leaves the operational path (2026-09-20)

`~/ds4-tunnel` existed for one reason, measured on 2026-09-19 and written up in its
own README: macOS would not let an adhoc/linker-signed binary accept connections on
a non-loopback address, and the Application Firewall did not change that, so the
worker's connection had to be carried through `sshd` over loopback.

**That no longer reproduces on macOS 26.7 (build 25G229).** Tested directly rather
than assumed:

- A `ds4-server` bound to `0.0.0.0` was fetched from the Spark over the direct link:
  `curl http://192.168.2.1:8099/v1/models` returned the model list, exit 0. On
  2026-09-19 the same shape of connection was dropped.
- The **full pipeline with no tunnel anywhere**: worker on the Spark dialling
  `--coordinator 192.168.2.1 9911` with `--listen 192.168.2.2 55911`, coordinator on
  the Mac with `--listen 192.168.2.1 9911`. The control connection is a real socket
  on the link (`ESTAB 192.168.2.2:55652 -> 192.168.2.1:9911`), and a real completion
  returned exactly `NO TUNNEL OK` (`finish=stop`, 43 tokens), with the coordinator
  logging `chat ctx=21..69:48 gen=48 THINKING decoding … avg=12.46 t/s`.

So the tunnel is out of the operational path: the GLM 5.3 Flash entry in
`~/bin/llm_config.json` and the example in `docs/DISTRIBUTED.md` now use the link
addresses directly, and the tunnel's README leads with a NOT REQUIRED status.

**Kept as a fallback, not deleted.** The blocking behaviour was real and measured,
so it is OS-version dependent and can return — the README says exactly what to
re-enable and which loopback form to use if the Spark can no longer reach the Mac's
listener. Removing the mechanism outright would trade a one-file fallback for a
debugging session the next time macOS changes its mind.

**Also worth noting for the address question:** the tunnel README already recorded
that the tunnel must use the literal IPv4 of the direct link and never a hostname,
because an mDNS resolution failure killed an early session-scoped tunnel mid-ingest.
That is independent corroboration of the later finding that the name `spike` resolves
to the Mac's own LAN address and must not be used for this link.
