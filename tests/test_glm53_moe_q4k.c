/* GLM 5.3 routed-MoE dispatch: CUDA vs a host reference, for Q2_K and Q4_K.
 *
 * First slice of the WS-5 parity harness from the split plan. It builds a
 * synthetic 288-expert MoE through the public ds4_gpu_* API and compares the
 * dispatch's output against a straightforward host implementation of the same
 * arithmetic.
 *
 * It exists because three separate regressions during the Q4_K port were all a
 * literal 256 where GLM 5.3 Flash has 288 experts, and all three silently
 * dropped every pair routed to experts 256..287:
 *   - the expert-map bound, so those pairs never got a `mid` row written;
 *   - the expert-major grid's y extent, so those experts were never launched;
 *   - the tile-capacity slack, which under-sized the tile arrays.
 * The `mid` buffer is filled with NaN before each call, so a pair that is never
 * written shows up as NaN rather than as plausible-looking arithmetic - which
 * is exactly what made the original fault hard to see.
 *
 * The swiglu clamp is not exercised here, deliberately. GLM 5.3 sets
 * swiglu_clamp_exp = 10.0; the CPU reference and the Metal kernels both clamp
 * gate above and up on both sides, while the CUDA GLM MoE used to ignore the
 * value - it could not even receive it, because its definition was missing a
 * parameter that ds4_gpu.h and its only caller both pass. That is verified
 * against real weights instead: plumbing the clamp through changed the pair's
 * next-token logits and *improved* their agreement with the Metal reference
 * (mean |delta| 0.085 -> 0.073, max 0.50 -> 0.20, top-16 15 -> 16 of 16), which
 * no synthetic case here would show as convincingly. Making the clamp bind on
 * purpose needs weights whose dot deterministically exceeds it, and getting
 * that construction right is worth doing separately.
 *
 * Run: make test-glm53-moe-q4k   (needs a CUDA device)
 */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_gpu.h"

#define QK_K 256
#define N_EXPERT_TOTAL   288u   /* GLM 5.3 Flash, per glm5-next.expert_count */
#define N_EXPERT_USED      4u
#define EXPERT_IN_DIM    256u   /* one weight block per row keeps the reference readable */
#define EXPERT_MID_DIM   256u
#define OUT_DIM          256u

typedef struct { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[QK_K / 2]; } block_q4_K;
typedef struct { uint8_t scales[QK_K / 16]; uint8_t qs[QK_K / 4]; uint16_t d, dmin; } block_q2_K;
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K / 16]; } block_q8_K;

static int failures;
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); failures++; } } while (0)

static uint32_t rng_state = 0x5eed1234u;
static uint32_t rng(void) { rng_state = rng_state * 1664525u + 1013904223u; return rng_state; }

static uint32_t block_bytes(uint32_t type) { return type == 12u ? 144u : 84u; }

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const int32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t frac = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) bits = sign;
    else if (exp == 31) bits = sign | 0x7F800000u | (frac << 13);
    else bits = sign | ((uint32_t)(exp - 15 + 127) << 23) | (frac << 13);
    float out;
    memcpy(&out, &bits, sizeof out);
    return out;
}

/* ---- host mirrors of the device quantizer and the two block dots ---- */

static void ref_q8_K(const float *x, block_q8_K *out) {
    float amax = 0.0f, maxv = 0.0f;
    for (uint32_t i = 0; i < QK_K; i++) {
        const float a = fabsf(x[i]);
        if (a > amax) { amax = a; maxv = x[i]; }   /* strict compare: lowest index wins */
    }
    if (amax == 0.0f) {
        out->d = 0.0f;
        memset(out->qs, 0, sizeof out->qs);
        memset(out->bsums, 0, sizeof out->bsums);
        return;
    }
    const float iscale = -127.0f / maxv;
    for (uint32_t i = 0; i < QK_K; i++) {
        int qv = (int)lrintf(iscale * x[i]);
        if (qv > 127) qv = 127;
        if (qv < -128) qv = -128;
        out->qs[i] = (int8_t)qv;
    }
    for (uint32_t i = 0; i < QK_K / 16; i++) {
        int sum = 0;
        for (uint32_t j = 0; j < 16; j++) sum += out->qs[i * 16 + j];
        out->bsums[i] = (int16_t)sum;
    }
    out->d = 1.0f / iscale;
}

static void q4_K_scale_min(uint32_t j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4u) { *d = q[j] & 63u; *m = q[j + 4u] & 63u; }
    else {
        *d = (uint8_t)((q[j + 4u] & 0x0Fu) | ((q[j - 4u] >> 6u) << 4u));
        *m = (uint8_t)((q[j + 4u] >> 4u) | ((q[j] >> 6u) << 4u));
    }
}

static float ref_dot_q4_K(const block_q4_K *x, const block_q8_K *y) {
    const float xd = f16_to_f32(x->d);
    const float xmin = f16_to_f32(x->dmin);
    int32_t isum = 0, summs = 0;
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        q4_K_scale_min(j, x->scales, &sc, &m);
        summs += (int32_t)m * (int32_t)(y->bsums[2u * j] + y->bsums[2u * j + 1u]);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int shift = (j & 1u) ? 4 : 0;
        int32_t part = 0;
        for (uint32_t i = 0; i < 32u; i++) {
            const int32_t v = ((int32_t)x->qs[byte_off + i] >> shift) & 0x0F;
            part += v * (int32_t)y->qs[j * 32u + i];
        }
        isum += (int32_t)sc * part;
    }
    return y->d * xd * (float)isum - y->d * xmin * (float)summs;
}

static int32_t dot_q2_16(const uint8_t *q2, const int8_t *q8, int shift) {
    int32_t sum = 0;
    for (uint32_t i = 0; i < 16u; i++) {
        const int32_t v = ((int32_t)q2[i] >> shift) & 0x03;
        sum += v * (int32_t)q8[i];
    }
    return sum;
}

static float ref_dot_q2_K(const block_q2_K *x, const block_q8_K *y) {
    const uint8_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    const uint8_t *sc = x->scales;
    int32_t summs = 0;
    for (int j = 0; j < 16; j++) summs += y->bsums[j] * (sc[j] >> 4);
    const float dall = y->d * f16_to_f32(x->d);
    const float dmin = y->d * f16_to_f32(x->dmin);
    int32_t isum = 0;
    int is = 0;
    for (int k = 0; k < (int)(QK_K / 128); k++) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            int d = sc[is++] & 0x0F;
            isum += d * dot_q2_16(q2, q8, shift);
            d = sc[is++] & 0x0F;
            isum += d * dot_q2_16(q2 + 16, q8 + 16, shift);
            shift += 2;
            q8 += 32;
        }
        q2 += 32;
    }
    return dall * (float)isum - dmin * (float)summs;
}

static float ref_dot(uint32_t type, const void *w, const block_q8_K *y) {
    return type == 12u ? ref_dot_q4_K((const block_q4_K *)w, y)
                       : ref_dot_q2_K((const block_q2_K *)w, y);
}

static float swiglu_ref(float g, float u, float clamp) {
    if (clamp > 1.0e-6f) {
        if (g > clamp) g = clamp;
        if (u > clamp) u = clamp;
        if (u < -clamp) u = -clamp;
    }
    return g / (1.0f + expf(-g)) * u;
}

/* ---- the MoE reference ---- */

static uint64_t expert_bytes(uint32_t type) { return EXPERT_IN_DIM / QK_K * block_bytes(type); }
static uint64_t down_bytes(uint32_t type) { return EXPERT_MID_DIM / QK_K * block_bytes(type); }

static void reference(uint32_t type, const unsigned char *blob,
                      uint64_t gate_off, uint64_t up_off, uint64_t down_off,
                      uint32_t n_tokens, const int32_t *selected,
                      const float *weights, const float *x, float clamp,
                      float *out) {
    block_q8_K *xq = calloc(n_tokens, sizeof(block_q8_K));
    float *mid = calloc((size_t)n_tokens * N_EXPERT_USED * EXPERT_MID_DIM, sizeof(float));
    CHECK(xq && mid);
    if (!xq || !mid) { free(xq); free(mid); return; }

    for (uint32_t t = 0; t < n_tokens; t++) ref_q8_K(x + (size_t)t * EXPERT_IN_DIM, &xq[t]);
    memset(out, 0, (size_t)n_tokens * OUT_DIM * sizeof(float));

    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t s = 0; s < N_EXPERT_USED; s++) {
            const uint32_t pair = t * N_EXPERT_USED + s;
            const int32_t e = selected[pair];
            if (e < 0) continue;
            const uint64_t eb = expert_bytes(type), db = down_bytes(type);
            for (uint32_t r = 0; r < EXPERT_MID_DIM; r++) {
                const unsigned char *gw = blob + gate_off + (size_t)e * eb + (size_t)r * eb;
                const unsigned char *uw = blob + up_off + (size_t)e * eb + (size_t)r * eb;
                mid[(size_t)pair * EXPERT_MID_DIM + r] =
                    swiglu_ref(ref_dot(type, gw, &xq[t]), ref_dot(type, uw, &xq[t]), clamp);
            }
            block_q8_K mq;
            ref_q8_K(&mid[(size_t)pair * EXPERT_MID_DIM], &mq);
            for (uint32_t r = 0; r < OUT_DIM; r++) {
                const unsigned char *dw = blob + down_off + (size_t)e * db + (size_t)r * db;
                out[(size_t)t * OUT_DIM + r] += weights[pair] * ref_dot(type, dw, &mq);
            }
        }
    }
    free(xq);
    free(mid);
}

/* ---- one case: build the blob, run the dispatch, compare ---- */

static void run_case(uint32_t type, uint32_t n_tokens, float x_scale) {
    const float clamp = 10.0f;   /* GLM 5.3's swiglu_clamp_exp */
    const uint64_t eb = expert_bytes(type), db = down_bytes(type);
    const uint64_t gate_off = 0;
    const uint64_t up_off = gate_off + (uint64_t)N_EXPERT_TOTAL * eb;
    const uint64_t down_off = up_off + (uint64_t)N_EXPERT_TOTAL * eb;
    const uint64_t blob_bytes = down_off + (uint64_t)N_EXPERT_TOTAL * db;

    unsigned char *blob = malloc(blob_bytes);
    int32_t *selected = malloc((size_t)n_tokens * N_EXPERT_USED * sizeof(int32_t));
    float *weights = malloc((size_t)n_tokens * N_EXPERT_USED * sizeof(float));
    float *x = malloc((size_t)n_tokens * EXPERT_IN_DIM * sizeof(float));
    float *out_ref = malloc((size_t)n_tokens * OUT_DIM * sizeof(float));
    float *out_gpu = malloc((size_t)n_tokens * OUT_DIM * sizeof(float));
    float *mid_gpu = malloc((size_t)n_tokens * N_EXPERT_USED * EXPERT_MID_DIM * sizeof(float));
    CHECK(blob && selected && weights && x && out_ref && out_gpu && mid_gpu);
    if (!blob || !selected || !weights || !x || !out_ref || !out_gpu || !mid_gpu) goto done;

    /* Random-but-bounded weight blocks. The super-scale is fixed at a benign
     * fp16 value and dmin at zero on purpose: random super-scales drive the
     * integer dot into int32 overflow (undefined behaviour, so host and device
     * would diverge for reasons unrelated to the dispatch), and a non-zero dmin
     * makes the dot the difference of two large terms. */
    const uint16_t half_one = 0x3800u;   /* 0.5 */
    const uint16_t half_zero = 0x0000u;  /* 0.0 */
    for (uint64_t i = 0; i < blob_bytes; i++) blob[i] = (unsigned char)(rng() >> 24);
    for (uint32_t e = 0; e < N_EXPERT_TOTAL; e++) {
        const uint64_t rows[3] = { gate_off + (uint64_t)e * eb,
                                   up_off + (uint64_t)e * eb,
                                   down_off + (uint64_t)e * db };
        for (uint32_t k = 0; k < 3u; k++) {
            memcpy(blob + rows[k], &half_one, 2);
            memcpy(blob + rows[k] + 2, &half_zero, 2);
        }
    }

    /* Selection: the first token deliberately uses the top expert index (287),
     * the boundary expert (256), the last low expert (255), and nothing else -
     * so a 256-expert bound shows up immediately. Slot 2 of token 0 is then
     * forced negative, and expert 257 is never selected by anyone. */
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t s = 0; s < N_EXPERT_USED; s++) {
            selected[t * N_EXPERT_USED + s] = (int32_t)((t * 7u + s * 61u) % N_EXPERT_TOTAL);
            weights[t * N_EXPERT_USED + s] = 0.25f + 0.5f * (float)((t + s) % 3u);
        }
    }
    selected[0] = 287;   /* the highest expert in the model */
    selected[1] = 256;   /* the first expert a 256 bound would drop */
    selected[2] = 255;   /* the last expert such a bound keeps */
    selected[3] = -1;    /* an empty slot: must contribute nothing */

    for (uint32_t i = 0; i < n_tokens * EXPERT_IN_DIM; i++) {
        x[i] = x_scale * (1.0f - 2.0f * (float)(rng() >> 16) / 32768.0f);
    }

    CHECK(ds4_gpu_set_model_map(blob, blob_bytes));

    ds4_gpu_tensor *t_selected = ds4_gpu_tensor_alloc((size_t)n_tokens * N_EXPERT_USED * sizeof(int32_t));
    ds4_gpu_tensor *t_weights = ds4_gpu_tensor_alloc((size_t)n_tokens * N_EXPERT_USED * sizeof(float));
    ds4_gpu_tensor *t_x = ds4_gpu_tensor_alloc((size_t)n_tokens * EXPERT_IN_DIM * sizeof(float));
    ds4_gpu_tensor *t_mid = ds4_gpu_tensor_alloc((size_t)n_tokens * N_EXPERT_USED * EXPERT_MID_DIM * sizeof(float));
    ds4_gpu_tensor *t_out = ds4_gpu_tensor_alloc((size_t)n_tokens * OUT_DIM * sizeof(float));
    CHECK(t_selected && t_weights && t_x && t_mid && t_out);
    if (!t_selected || !t_weights || !t_x || !t_mid || !t_out) goto done;

    CHECK(ds4_gpu_tensor_write(t_selected, 0, selected, (size_t)n_tokens * N_EXPERT_USED * sizeof(int32_t)));
    CHECK(ds4_gpu_tensor_write(t_weights, 0, weights, (size_t)n_tokens * N_EXPERT_USED * sizeof(float)));
    CHECK(ds4_gpu_tensor_write(t_x, 0, x, (size_t)n_tokens * EXPERT_IN_DIM * sizeof(float)));
    /* NaN sentinel: anything the dispatch fails to write stays NaN. */
    CHECK(ds4_gpu_tensor_fill_f32(t_mid, NAN, (size_t)n_tokens * N_EXPERT_USED * EXPERT_MID_DIM));
    CHECK(ds4_gpu_tensor_fill_f32(t_out, NAN, (size_t)n_tokens * OUT_DIM));

    const int ok = ds4_gpu_glm_routed_moe_batch_tensor(
            t_out, t_mid, blob, blob_bytes, gate_off, up_off, down_off,
            type, type, type,
            eb, eb, eb, eb, db, db,
            EXPERT_IN_DIM, EXPERT_MID_DIM, OUT_DIM,
            t_selected, t_weights, N_EXPERT_TOTAL, N_EXPERT_USED,
            clamp, 0u, t_x, n_tokens, N_EXPERT_USED * EXPERT_MID_DIM, false);
    CHECK(ok);
    if (!ok) goto done;

    CHECK(ds4_gpu_tensor_read(t_mid, 0, mid_gpu,
                              (size_t)n_tokens * N_EXPERT_USED * EXPERT_MID_DIM * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(t_out, 0, out_gpu, (size_t)n_tokens * OUT_DIM * sizeof(float)));

    reference(type, blob, gate_off, up_off, down_off, n_tokens, selected, weights, x, clamp, out_ref);

    /* 1. every selected pair must have been written, including experts >= 256 */
    uint32_t nan_pairs = 0;
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t s = 0; s < N_EXPERT_USED; s++) {
            const uint32_t pair = t * N_EXPERT_USED + s;
            if (selected[pair] < 0) continue;
            for (uint32_t r = 0; r < EXPERT_MID_DIM; r++) {
                if (isnan(mid_gpu[(size_t)pair * EXPERT_MID_DIM + r])) { nan_pairs++; break; }
            }
        }
    }
    CHECK(nan_pairs == 0);
    if (nan_pairs) {
        fprintf(stderr, "  type=%u tokens=%u: %u selected pairs had no mid row written\n",
                type, n_tokens, nan_pairs);
    }

    /* 2. the specific high-expert pairs that a 256-expert bound would drop */
    for (uint32_t s = 0; s < 3u; s++) {
        const uint32_t pair = s;
        bool any = false;
        for (uint32_t r = 0; r < EXPERT_MID_DIM; r++) if (!isnan(mid_gpu[(size_t)pair * EXPERT_MID_DIM + r])) any = true;
        CHECK(any);
    }

    /* 3. values against the reference */
    float worst = 0.0f;
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t r = 0; r < OUT_DIM; r++) {
            const float a = out_gpu[(size_t)t * OUT_DIM + r];
            const float b = out_ref[(size_t)t * OUT_DIM + r];
            const float rel = fabsf(a - b) / (1.0f + fabsf(b));
            if (rel > worst) worst = rel;
        }
    }
    CHECK(worst < 2.0e-3f);
    printf("  type=%2u tokens=%3u clamp=%5.1f: worst rel=%.3e  %s\n",
           type, n_tokens, clamp, worst, worst < 2.0e-3f ? "ok" : "MISMATCH");

done:
    free(blob); free(selected); free(weights); free(x);
    free(out_ref); free(out_gpu); free(mid_gpu);
}

int main(void) {
    if (!ds4_gpu_init()) {
        fprintf(stderr, "ds4_gpu_init failed; this test needs a CUDA device\n");
        return 0;   /* not a failure: no device here */
    }

    printf("GLM 5.3 routed-MoE parity (CUDA vs host reference, %u experts)\n", N_EXPERT_TOTAL);

    /* Q4_K only. The regressions this guards were type-independent - the same
     * literal 256 bounded both types' expert maps - so one type proves them.
     * Q2_K's device dot spans more than the single 84-byte block per row that
     * keeps this reference readable, so a Q2_K case would need a multi-block
     * row and its own host mirror; its point arithmetic is covered by
     * `make q4k-dot-test` and by the Q2_K pair runs. */
    const uint32_t type = 12u;
    run_case(type, 8u, 0.05f);      /* warp / small-batch path */
    run_case(type, 128u, 0.05f);    /* tile8 prefill path      */

    if (failures) {
        printf("%d check(s) FAILED\n", failures);
        return 1;
    }
    printf("PASS\n");
    return 0;
}
