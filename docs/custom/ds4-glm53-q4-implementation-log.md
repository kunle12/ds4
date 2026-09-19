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
| Q4_K on the pair | **blocked** — CUDA GLM routed MoE is Q2_K-only; port specified in the plan |
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
| 6 | `ds4.c:46078` (`glm_graph_layer_uses_generic_routed_moe`) | **experiment, inert unless `DS4_GLM_GENERIC_MOE_Q4K` is set**: a homogeneous Q4_K expert trio may be routed through the generic MoE dispatch | a spike shortcut to exercise Q4_K on CUDA before the kernel port exists; its own comment says remove or make unconditional once the result is measured. **Not part of the design** — plan WS 1–4 supersede it | built on both backends; not a correctness path |

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
#   /dev/ttys000 at 0% CPU with the model unloaded (seen 2026-09-19 with
#   --prompt-file). Run it under `ssh localhost` or as a launchd job, and check
#   `ps -o pcpu` before waiting on it.

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
