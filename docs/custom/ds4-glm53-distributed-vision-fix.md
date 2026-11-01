# GLM 5.3 distributed vision: blank images (2026-11-01)

## Symptom

The titan (Mac Studio) coordinator `0:20` plus spike (DGX Spark) worker `21:44`
pair served text normally, but every image request was answered as if the
attachment were blank or missing. On the live 2-slot pair
(`--ctx 262144 --batched-session 2 --dist-prefill-chunk 2048`):

```sh
DS4_VISION_ENDPOINT=http://titan:8081/v1/chat/completions \
DS4_VISION_MODEL=glm-5.3-flash \
python3 tests/run_glm53_vision_quality.py
```

reported `0/6 passed`. The server logged `vision embedding encoded` and then
prefilled the prompt with the image placeholder tokens, but the answers said the
image was empty. The single-host path already forwarded the spans; the
distributed layer-slice path did not.

## Root cause

The distributed coordinator prefills its layer slice through
`ds4_session_eval_layer_slice()` (`ds4_distributed.c` ->
`dist_coordinator_eval_span()` -> `ds4_session_eval_layer_slice()`). That path
called the GLM batch prefill helpers with `NULL, 0` for the vision spans:

```c
glm_graph_forward_tokens(..., NULL, 0, ...);
glm_graph_forward_indexed_tokens(..., NULL, 0, ...);
```

The ordinary session path (`ds4_session_sync_internal()`) already passed
`s->sync_images, s->sync_image_count`; the layer-slice path was written later
and never picked them up. The encoder therefore ran and the prompt contained
the placeholder IDs (`154854` image, `154830`/`154831` start/end), but
`glm_vision_overlay_prepare()` never saw the embeddings, so the language model
read bare placeholders — a blank image.

This also explains why the coordinator-only `--vision` flag looked harmless:
the encoder was loaded and executed; only the row splice was missing.

## Fix

`ds4_session_eval_layer_slice()` now forwards `s->sync_images` to both batch
prefill calls. The splice is owned by the rank that embeds the tokens, so it is
suppressed when `input_hc` is present (the worker receives hidden states and
must not repeat it). A single-token decode step is never taken inside an image
span, and a span that cannot reach a batch path now fails explicitly instead of
silently dropping the rows.

Commit: `glm53: splice vision rows in distributed layer-slice prefill`.

## Validation

Built on titan with `make -j16 ds4-server` (no warnings). Restarted the 2-slot
pair with titan as coordinator and spike as worker. Same command as the
reproduction now reports **6/6 passed**:

| case | result |
| --- | --- |
| diagram | INPUT / ENCODER / SUMMARY, SUMMARY follows ENCODER |
| ocr | NORTH HARBOR, 482, C7, 19:45, 14A, MINT-731 |
| spatial | blue circle upper-left, red triangle upper-right, green square lower-center |
| screenshot | deploy.sh selected, build failed, missing CUDA_HOME, exit status 2 |
| photo | Earth, Africa, Madagascar |
| unrelated | rejects the train-ticket premise, describes the three shapes |

A multi-turn check (diagram, then "What color is the SUMMARY box?") answered
"orange", so the image survives the live-cache follow-up prefill. A text-only
request still completed normally.

The worker did not need a rebuild: it never owns the encoder or the splice.

## Deployment note

The GLM entries in titan's `~/bin/llm_config.json` launch
`/Users/xun/dev/ds4/ds4-server`, which is the rebuilt binary. The copies under
`~/bin` were not replaced; they are used by the other model entries.

## Regression checks

The change only enters the GLM distributed layer-slice path; DeepSeek does not
call it and the solo GLM path uses `ds4_session_sync_internal()` instead. Both
were still exercised end to end:

- **DeepSeek V4.1 Flash** (original Q2 + vision, `--metal --ssd-streaming`):
  text returned `4`; `earth.jpg` returned Earth / Africa.
- **Solo GLM 5.3 Flash Q2** (`--ctx 32768 --vision`, no `--role`): text returned
  `4`; `earth.jpg` returned Earth / Africa. The temporary manager entry used for
  this run was removed and `llm_config.json` restored byte-for-byte.
- **Restored 2-slot pair**: the Earth / Africa image smoke test passed.
