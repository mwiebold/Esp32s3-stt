#ifdef __FAST_MATH__
#error "tinyasr quantization requires strict floating-point arithmetic; disable -ffast-math"
#endif
// tasr_nemo: utterance-level engine for NVIDIA NeMo Conformer-CTC small (rel-pos MHSA, full context), int8.
// Mirrors train/nemo_small.py + train/nemo_eval.py (--bits 8 --att8) arithmetic.
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "kernels.h"
#include "tasr_nemo.h"
#include "tinyasr.h"
#include "tinyasr_lm.h"

#define NMEL 80
#define NBIN 257
#define WIN 400
#define HOP 160
#define NW 2
#define RB 64  // row block for position-wise sublayers

#ifdef TASR_PROFILE
#ifdef ESP_PLATFORM
#include "esp_timer.h"
static inline uint64_t nts(void) { return (uint64_t)esp_timer_get_time(); }
#else
#include <time.h>
static inline uint64_t nts(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + t.tv_nsec; }
#endif
enum { N_FEAT, N_CONV0, N_IM2COL, N_CONV2, N_SUB, N_LN, N_GEMM, N_QUANT, N_ACT, N_QKV8, N_POS, N_ATT, N_DW, N_HEAD, N_LNQ, N_ACTQ, N_RES, N_GLU, N_NP };
static const char *nprof_names[N_NP] = {"features", "conv0", "im2col", "gemm_fe", "gemm_k704", "layernorm", "gemm", "quant",
                                        "act", "qkv_int8", "pos", "attention", "dwconv", "head+dec", "norm+quant", "act+quant", "residual", "glu"};
static uint64_t nprof[N_NP];
void tasr_nemo_profile_reset(void) { memset(nprof, 0, sizeof(nprof)); }
#define NB(v) uint64_t v = nts()
#define NE(v, c) nprof[c] += nts() - v
const char *tasr_nemo_profile_name(int i) { return i < N_NP ? nprof_names[i] : 0; }
uint64_t tasr_nemo_profile_value(int i) { return i < N_NP ? nprof[i] : 0; }
#else
#define NB(v)
#define NE(v, c)
void tasr_nemo_profile_reset(void) {}
const char *tasr_nemo_profile_name(int i) { (void)i; return 0; }
uint64_t tasr_nemo_profile_value(int i) { (void)i; return 0; }
#endif

typedef struct {
    const float *g, *b;
} nln_t;

typedef struct {
    nln_t n_ff1, n_att, n_conv, n_ff2, n_out;
    tasr_qlin_t ff1_1, ff1_2, qkv, out, pos, pw1, pw2, ff2_1, ff2_2;
    const float *pbu, *pbv, *dw_w, *dw_b;
    float *dw_wt;  // [k][d]
} nlayer_t;

struct tasr_nemo {
    int d, h, dh, dhp, ff, k, nl, sc, V, f1, f2;
    const float *window;
    int mel_start[NMEL], mel_len[NMEL];
    float *mel_w;
    const float *c0_w, *c0_b;
    tasr_qlin_t c2, sub, head;
    int rnnt;                                              // transducer model (RNN-T head instead of CTC)
    tasr_qlin_t r_emb, r_ih, r_hh, r_jenc, r_jpred, r_jout;  // prediction net (embedding, LSTM) + joint net
    nlayer_t *L;
    const char **tok;
    uint8_t *tok_len;
    float tw_re[256], tw_im[256];
    int16_t bitrev[256];
    size_t weight_bytes, blob_bytes;
};

// ------------------------------------------------------------------ loading
typedef struct {
    const uint8_t *p, *end;
    int err;
} ncur_t;
static void nalign(ncur_t *c) { c->p = (const uint8_t *)(((uintptr_t)c->p + 15) & ~(uintptr_t)15); }
static const float *nf32(ncur_t *c, size_t n, size_t *acc)
{
    nalign(c);
    const float *r = (const float *)c->p;
    c->p += n * 4;
    if (c->p > c->end) c->err = 1;
    if (acc) *acc += n * 4;
    return r;
}
static tasr_qlin_t nload_qlin(ncur_t *c, int n, int k, int bits, size_t *acc)
{
    tasr_qlin_t L;
    memset(&L, 0, sizeof(L));
    L.blocked = bits == 4 && n % 16 == 0;
    int blk = (bits == 8 || L.blocked) ? 16 : 32;
    L.n = n; L.k = k; L.bits = bits;
    L.kp = (k + blk - 1) / blk * blk;
    nalign(c);
    L.w = (const int8_t *)c->p;
    size_t wb = (size_t)n * (bits == 8 ? L.kp : L.kp / 2);
    c->p += wb;
    *acc += wb;
    L.s = nf32(c, n, acc);
    L.b = nf32(c, n, acc);
    return L;
}
static nln_t nln(ncur_t *c, int d, size_t *acc)
{
    nln_t l;
    l.g = nf32(c, d, acc);
    l.b = nf32(c, d, acc);
    return l;
}

tasr_nemo_t *tasr_nemo_load(const uint8_t *blob, size_t size)
{
    if (size < 64 || memcmp(blob, "TNM1", 4)) return NULL;
    uint32_t hd[15];
    memcpy(hd, blob + 4, sizeof(hd));
    tasr_nemo_t *m = (tasr_nemo_t *)tasr_alloc(sizeof(tasr_nemo_t), 1);
    m->d = hd[1]; m->h = hd[2]; m->ff = hd[3]; m->k = hd[4]; m->nl = hd[5]; m->sc = hd[6]; m->V = hd[7];
    const int bits = hd[8];
    m->dh = m->d / m->h;
    m->dhp = (m->dh + 15) & ~15;
    m->f1 = (NMEL + 2 - 3) / 2 + 1;
    m->f2 = (m->f1 + 2 - 3) / 2 + 1;
    ncur_t c = {blob + 64, blob + size, 0};
    size_t acc = 0;
    const int d = m->d;
    m->window = nf32(&c, WIN, &acc);
    const float *fb = nf32(&c, NBIN * NMEL, NULL);
    int tot = 0;
    for (int j = 0; j < NMEL; j++) {
        int s = -1, e = -1;
        for (int b = 0; b < NBIN; b++)
            if (fb[b * NMEL + j] != 0.f) { if (s < 0) s = b; e = b; }
        m->mel_start[j] = s < 0 ? 0 : s;
        m->mel_len[j] = s < 0 ? 0 : e - s + 1;
        tot += m->mel_len[j];
    }
    m->mel_w = (float *)tasr_alloc(sizeof(float) * (tot + 1), 1);
    for (int j = 0, o = 0; j < NMEL; j++)
        for (int b = 0; b < m->mel_len[j]; b++) m->mel_w[o++] = fb[(m->mel_start[j] + b) * NMEL + j];
    m->c0_w = nf32(&c, m->sc * 9, &acc);
    m->c0_b = nf32(&c, m->sc, &acc);
    m->c2 = nload_qlin(&c, m->sc, m->sc * 9, 8, &acc);
    m->sub = nload_qlin(&c, d, m->sc * m->f2, 8, &acc);
    m->L = (nlayer_t *)tasr_alloc(sizeof(nlayer_t) * m->nl, 1);
    for (int i = 0; i < m->nl; i++) {
        nlayer_t *L = &m->L[i];
        L->n_ff1 = nln(&c, d, &acc);
        L->ff1_1 = nload_qlin(&c, m->ff, d, bits, &acc);
        L->ff1_2 = nload_qlin(&c, d, m->ff, bits, &acc);
        L->n_att = nln(&c, d, &acc);
        L->qkv = nload_qlin(&c, 3 * d, d, bits, &acc);
        L->out = nload_qlin(&c, d, d, bits, &acc);
        L->pos = nload_qlin(&c, d, d, 8, &acc);
        L->pbu = nf32(&c, d, &acc);
        L->pbv = nf32(&c, d, &acc);
        L->n_conv = nln(&c, d, &acc);
        L->pw1 = nload_qlin(&c, 2 * d, d, bits, &acc);
        L->dw_w = nf32(&c, d * m->k, &acc);
        L->dw_b = nf32(&c, d, &acc);
        L->pw2 = nload_qlin(&c, d, d, bits, &acc);
        L->n_ff2 = nln(&c, d, &acc);
        L->ff2_1 = nload_qlin(&c, m->ff, d, bits, &acc);
        L->ff2_2 = nload_qlin(&c, d, m->ff, bits, &acc);
        L->n_out = nln(&c, d, &acc);
        float *wt = (float *)tasr_alloc(sizeof(float) * m->k * d, 0);
        for (int ch = 0; ch < d; ch++)
            for (int j = 0; j < m->k; j++) wt[j * d + ch] = L->dw_w[ch * m->k + j];
        L->dw_wt = wt;
    }
    if (hd[9] == 1) {  // RNN-T: embedding (V+1 x 320), LSTM 320 (4 gates), joint (enc 176->320, pred 320->320, 320->V+1)
        const int P = 320;
        m->rnnt = 1;
        m->r_emb = nload_qlin(&c, m->V + 1, P, 8, &acc);
        m->r_ih = nload_qlin(&c, 4 * P, P, 8, &acc);
        m->r_hh = nload_qlin(&c, 4 * P, P, 8, &acc);
        m->r_jenc = nload_qlin(&c, P, d, 8, &acc);
        m->r_jpred = nload_qlin(&c, P, P, 8, &acc);
        m->r_jout = nload_qlin(&c, m->V + 1, P, 8, &acc);
    } else {
        m->head = nload_qlin(&c, m->V + 1, d, 8, &acc);
    }
    nalign(&c);
    m->tok = (const char **)tasr_alloc(sizeof(char *) * m->V, 1);
    m->tok_len = (uint8_t *)tasr_alloc(m->V, 1);
    for (int i = 0; i < m->V; i++) {
        m->tok_len[i] = *c.p++;
        m->tok[i] = (const char *)c.p;
        c.p += m->tok_len[i];
    }
    if (c.err || c.p > c.end) { tasr_nemo_free(m); return NULL; }
    m->blob_bytes = (size_t)(c.p - blob);
    m->weight_bytes = acc;
    for (int i = 0; i < 256; i++) {
        m->tw_re[i] = (float)cos(-2.0 * M_PI * i / 512);
        m->tw_im[i] = (float)sin(-2.0 * M_PI * i / 512);
    }
    for (int i = 0; i < 256; i++) {
        int r = 0;
        for (int b = 0; b < 8; b++) r |= ((i >> b) & 1) << (7 - b);
        m->bitrev[i] = (int16_t)r;
    }
    return m;
}

void tasr_nemo_free(tasr_nemo_t *m)
{
    if (!m) return;
    if (m->L)
        for (int i = 0; i < m->nl; i++) tasr_free(m->L[i].dw_wt);
    tasr_free(m->mel_w); tasr_free(m->L); tasr_free((void *)m->tok); tasr_free(m->tok_len);
    tasr_free(m);
}
size_t tasr_nemo_weight_bytes(const tasr_nemo_t *m) { return m->weight_bytes; }
size_t tasr_nemo_blob_bytes(const tasr_nemo_t *m) { return m->blob_bytes; }

static size_t nmove(tasr_qlin_t *L, size_t *left)
{
    size_t wb = (size_t)L->n * (L->bits == 8 ? L->kp : L->kp / 2);
    if (wb > *left) return 0;
    int8_t *p = (int8_t *)tasr_alloc(wb, 0);
    if (!p) return 0;
    memcpy(p, L->w, wb);
    L->w = p;
    *left -= wb;
    return wb;
}
size_t tasr_nemo_place_weights(tasr_nemo_t *m, size_t budget)
{
    // front end first (re-read every few frames), then the layers
    size_t left = budget, moved = nmove(&m->c2, &left) + nmove(&m->sub, &left);
    if (m->rnnt)  // joint output and LSTM weights are re-read every frame / every token
        moved += nmove(&m->r_jout, &left) + nmove(&m->r_jpred, &left) + nmove(&m->r_ih, &left) + nmove(&m->r_hh, &left) +
                 nmove(&m->r_jenc, &left);
    else
        moved += nmove(&m->head, &left);
    for (int i = 0; i < m->nl; i++) {
        nlayer_t *L = &m->L[i];
        tasr_qlin_t *q[9] = {&L->qkv, &L->pos, &L->out, &L->pw1, &L->pw2, &L->ff1_1, &L->ff1_2, &L->ff2_1, &L->ff2_2};
        for (int j = 0; j < 9; j++) moved += nmove(q[j], &left);
    }
    return moved;
}

// ------------------------------------------------------------------ helpers
static float nsig_tab[385], nsig_slope[384];
static int sig_ok;
static void nsig_init(void)
{
    if (sig_ok) return;
    for (int i = 0; i <= 384; i++) nsig_tab[i] = (float)(1.0 / (1.0 + exp(-(-12.0 + i / 16.0))));
    for (int i = 0; i < 384; i++) nsig_slope[i] = nsig_tab[i + 1] - nsig_tab[i];
    sig_ok = 1;
}
static inline float nsig(float x)
{
    float u = (x + 12.0f) * 16.0f;
    if (u <= 0.f) return 0.f;
    if (u >= 384.f) return 1.f;
    int i = (int)u;
    return nsig_tab[i] + (u - (float)i) * nsig_slope[i];
}
// exp(-i/64) for the attention softmax; nearest-entry lookup (error <= 0.8%, below the int8 probability step)
#define NEXP_STEPS 64
#define NEXP_N (20 * NEXP_STEPS)
static float nexp_tab[NEXP_N];
static void nexp_init(void)
{
    static int done;
    if (done) return;
    for (int i = 0; i < NEXP_N; i++) nexp_tab[i] = (float)exp(-(double)i / NEXP_STEPS);
    done = 1;
}
static inline float qvec(const float *x, int n, int8_t *q, int np)
{
    float mx = 0.f;
    for (int i = 0; i < n; i++) { float a = fabsf(x[i]); mx = a > mx ? a : mx; }
    if (mx < 1e-30f) mx = 1e-30f;
    const float inv = 127.0f / mx;
    for (int i = 0; i < n; i++) { float y = x[i] * inv + 12582912.0f; q[i] = (int8_t)(int)(y - 12582912.0f); }
    for (int i = n; i < np; i++) q[i] = 0;
    return mx / 127.0f;
}
static void layernorm_row(const float *r, int d, nln_t p, float *o)
{
    float mean = 0.f;
    for (int i = 0; i < d; i++) mean += r[i];
    mean /= (float)d;
    float var = 0.f;
    for (int i = 0; i < d; i++) { float v = r[i] - mean; var += v * v; }
    var /= (float)d;
    const float inv = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < d; i++) o[i] = (r[i] - mean) * inv * p.g[i] + p.b[i];
}

// ------------------------------------------------------------------ work context
typedef struct {
    const tasr_nemo_t *m;
    tasr_nemo_workspace_t *workspace;
    int T;                       // encoder frames
    int ldq;
    int8_t *xq;                  // [RB][ldq]
    float *xs;                   // [RB]
    int8_t *wtmp[NW];
    int32_t *acc[NW];
    // attention
    int8_t *qu, *qv, *k8, *vt, *p8;  // [h][T][dhp], [h][T][dhp], [h][T][dhp], [h][dh][Tp], [h][2T-1][dhp]
    float *squ, *sqv, *sk, *sv, *sp; // scales
    int Tp;
    float *sc[NW];
    int8_t *pq[NW];
    int32_t *iacc[NW];
    int iacc_n;  // stride of the second (positional) score buffer in iacc[w]
    float *att;                  // [T][d]
} nctx_t;

// A workspace belongs to one caller and one model. No global allocator swapping.
// Large buffers alias only when their phase lifetimes do not overlap; hot buffers
// stay individually allocated so the firmware's internal-RAM size policy still applies.
enum {
    WS_XQ,
    WS_XS,
    WS_WTMP0,
    WS_WTMP1,
    WS_ACC0,
    WS_ACC1,
    WS_X,
    WS_F,
    WS_XQ_SUB,
    WS_RING,
    WS_C2OUT,
    WS_FR,
    WS_XS_SUB,
    WS_COL,
    WS_CMAXR,
    WS_CMW,
    WS_PE,
    WS_PEQ,
    WS_PES,
    WS_QU,
    WS_QV,
    WS_K8,
    WS_VT,
    WS_P8,
    WS_SQU,
    WS_SQV,
    WS_SK,
    WS_SV,
    WS_SP,
    WS_SC0,
    WS_SC1,
    WS_PQ0,
    WS_PQ1,
    WS_IACC0,
    WS_IACC1,
    WS_HB,
    WS_H2,
    WS_GLB,
    WS_CV,
    WS_PROW,
    WS_LG,
    WS_LP,
    WS_RFJ,
    WS_RZ,
    WS_RLO,
    WS_RGI,
    WS_RGH,
    WS_RST,
    WS_RE,
    WS_COUNT
};
struct tasr_nemo_workspace {
    const tasr_nemo_t *model;
    int max_samples, use_decoder;
    void *ptr[WS_COUNT];
    uint8_t owned[WS_COUNT];
    void *arena;
    size_t bytes;
};

typedef struct { size_t size, off; int kind; unsigned phases; } nwbuf_t;

void tasr_nemo_workspace_free(tasr_nemo_workspace_t *ws)
{
    if (!ws) return;
    for (int i = 0; i < WS_COUNT; i++) if (ws->owned[i]) tasr_free(ws->ptr[i]);
    tasr_free(ws->arena);
    tasr_free(ws);
}
size_t tasr_nemo_workspace_bytes(const tasr_nemo_workspace_t *ws) { return ws ? ws->bytes : 0; }

tasr_nemo_workspace_t *tasr_nemo_workspace_create(const tasr_nemo_t *m, int max_samples, int use_decoder)
{
    if (!m || max_samples < 2 * HOP) return NULL;
    const int T0 = max_samples / HOP + 1, T1 = (T0 + 1) / 2, T = (T1 + 1) / 2;
    const int d = m->d, H = m->h, dh = m->dh, dhp = m->dhp, sc = m->sc, f1 = m->f1, f2 = m->f2;
    const int NP = 2 * T - 1, Tp = (T + 15) & ~15, halo = m->k / 2;
    const int ldq = m->ff > 2 * d ? m->ff : 2 * d;
    const int w2 = 3 * d > m->ff ? 3 * d : m->ff;
    const int C2B = RB / f2 < 1 ? 1 : RB / f2;
    const int unpack = m->L[0].ff1_1.bits == 4;
    nwbuf_t plan[WS_COUNT] = {
        [WS_XQ] = {(size_t)RB * ldq, 0, 1, 15},
        [WS_XS] = {sizeof(float) * RB, 0, 1, 15},
        [WS_WTMP0] = {unpack ? 16 * (size_t)ldq + 16 : 0, 0, 1, 15},
        [WS_WTMP1] = {unpack ? 16 * (size_t)ldq + 16 : 0, 0, 1, 15},
        [WS_ACC0] = {sizeof(int32_t) * RB * 16, 0, 1, 15},
        [WS_ACC1] = {sizeof(int32_t) * RB * 16, 0, 1, 15},
        [WS_X] = {sizeof(float) * (size_t)T * d, 0, 0, 15},
        [WS_F] = {sizeof(float) * (size_t)T0 * NMEL, 0, 0, 1},
        [WS_XQ_SUB] = {(size_t)RB * m->sub.kp, 0, 0, 1},
        [WS_RING] = {sizeof(float) * 3 * (size_t)sc * (f1 + 2), 0, 0, 1},
        [WS_C2OUT] = {sizeof(float) * C2B * (size_t)f2 * sc, 0, 0, 1},
        [WS_FR] = {sizeof(float) * (size_t)sc * f2, 0, 0, 1},
        [WS_XS_SUB] = {sizeof(float) * RB, 0, 1, 1},
        [WS_COL] = {(size_t)C2B * f2 * m->c2.kp, 0, 0, 1},
        [WS_CMAXR] = {sizeof(float) * 3 * (f1 + 2), 0, 1, 1},
        [WS_CMW] = {sizeof(float) * NW * (f1 + 2), 0, 1, 1},
        [WS_PE] = {sizeof(float) * (size_t)NP * d, 0, 0, 2},
        [WS_PEQ] = {(size_t)NP * m->L[0].pos.kp, 0, 0, 6},
        [WS_PES] = {sizeof(float) * NP, 0, 0, 6},
        [WS_QU] = {(size_t)H * T * dhp, 0, 0, 4},
        [WS_QV] = {(size_t)H * T * dhp, 0, 0, 4},
        [WS_K8] = {(size_t)H * T * dhp, 0, 0, 4},
        [WS_VT] = {(size_t)H * dh * Tp, 0, 0, 4},
        [WS_P8] = {(size_t)H * NP * dhp, 0, 0, 4},
        [WS_SQU] = {sizeof(float) * (size_t)H * T, 0, 0, 4},
        [WS_SQV] = {sizeof(float) * (size_t)H * T, 0, 0, 4},
        [WS_SK] = {sizeof(float) * (size_t)H * T, 0, 0, 4},
        [WS_SV] = {sizeof(float) * (size_t)H * T, 0, 0, 4},
        [WS_SP] = {sizeof(float) * (size_t)H * NP, 0, 0, 4},
        [WS_SC0] = {sizeof(float) * Tp, 0, 1, 4},
        [WS_SC1] = {sizeof(float) * Tp, 0, 1, 4},
        [WS_PQ0] = {(size_t)Tp, 0, 1, 4},
        [WS_PQ1] = {(size_t)Tp, 0, 1, 4},
        [WS_IACC0] = {sizeof(int32_t) * 2 * (Tp > RB ? Tp : RB), 0, 1, 4},
        [WS_IACC1] = {sizeof(int32_t) * 2 * (Tp > RB ? Tp : RB), 0, 1, 4},
        [WS_HB] = {sizeof(float) * RB * d, 0, 1, 4},
        [WS_H2] = {sizeof(float) * RB * w2, 0, 0, 4},
        [WS_GLB] = {sizeof(float) * (size_t)(T + 2 * halo) * d, 0, 0, 4},
        [WS_CV] = {sizeof(float) * (size_t)T * d, 0, 0, 4},
        [WS_PROW] = {sizeof(float) * RB * d, 0, 0, 4},
        [WS_LG] = {m->rnnt ? 0 : sizeof(float) * RB * (size_t)(m->V + 1), 0, 0, 8},
        [WS_LP] = {use_decoder && !m->rnnt ? sizeof(float) * (size_t)(m->V + 1) : 0, 0, 0, 8},
        [WS_RFJ] = {m->rnnt ? sizeof(float) * (size_t)T * 320 : 0, 0, 0, 8},
        [WS_RZ] = {m->rnnt ? sizeof(float) * 8 * 320 : 0, 0, 0, 8},
        [WS_RLO] = {m->rnnt ? sizeof(float) * 8 * (size_t)(m->V + 1) : 0, 0, 0, 8},
        [WS_RGI] = {m->rnnt ? sizeof(float) * 4 * 320 : 0, 0, 0, 8},
        [WS_RGH] = {m->rnnt ? sizeof(float) * 4 * 320 : 0, 0, 0, 8},
        [WS_RST] = {m->rnnt ? sizeof(float) * 3 * 320 : 0, 0, 0, 8},
        [WS_RE] = {m->rnnt ? sizeof(float) * 320 : 0, 0, 0, 8},
    };
    tasr_nemo_workspace_t *ws = tasr_alloc(sizeof(*ws), 1);
    if (!ws) return NULL;
    memset(ws, 0, sizeof(*ws));
    ws->model = m; ws->max_samples = max_samples; ws->use_decoder = use_decoder;
    ws->bytes = sizeof(*ws);
    size_t arena_size = 0;
    for (int i = 0; i < WS_COUNT; i++) {
        if (!plan[i].size) continue;
        if (plan[i].size > SIZE_MAX - 15) goto fail;
        plan[i].size = (plan[i].size + 15) & ~(size_t)15;
        if (plan[i].kind) {
            ws->ptr[i] = tasr_alloc(plan[i].size, 1);
            if (!ws->ptr[i]) goto fail;
            ws->owned[i] = 1;
            ws->bytes += plan[i].size;
            continue;
        }
        // First-fit interval packing; at most WS_COUNT objects, only at creation.
        size_t off = 0;
        for (int j = 0; j < i;) {
            if (plan[j].kind || !(plan[i].phases & plan[j].phases) || !plan[j].size) { j++; continue; }
            if (off > SIZE_MAX - plan[i].size) goto fail;
            if (off < plan[j].off + plan[j].size && plan[j].off < off + plan[i].size) {
                off = plan[j].off + plan[j].size;
                j = 0;
            } else j++;
        }
        if (off > SIZE_MAX - plan[i].size) goto fail;
        plan[i].off = off;
        if (off + plan[i].size > arena_size) arena_size = off + plan[i].size;
    }
    ws->arena = tasr_alloc(arena_size, 0);
    if (!ws->arena) goto fail;
    ws->bytes += arena_size;
    for (int i = 0; i < WS_COUNT; i++)
        if (!plan[i].kind && plan[i].size) ws->ptr[i] = (uint8_t *)ws->arena + plan[i].off;
    return ws;
fail:
    tasr_nemo_workspace_free(ws);
    return NULL;
}

typedef struct {
    nctx_t *c;
    const tasr_qlin_t *L;
    int T;
    float *y;
    int ldy;
} ngemm_t;
static void ngemm_job(void *p, int b, int e, int w)
{
    ngemm_t *j = (ngemm_t *)p;
    const int u = j->L->blocked ? 16 : 1;
    tasr_qlin_range(j->L, j->c->xq, j->c->xs, j->T, j->c->ldq, j->y, j->ldy, j->c->wtmp[w], j->c->acc[w], b * u, e * u);
}
// y[t][:] = L(x[t][:]) for T <= RB rows (x rows already quantized into c->xq / c->xs)
static void nqlin(nctx_t *c, const tasr_qlin_t *L, int T, float *y, int ldy)
{
    NB(t0);
    ngemm_t j = {c, L, T, y, ldy};
    tasr_parallel_work(ngemm_job, &j, L->blocked ? L->n / 16 : L->n, (size_t)T * L->n * L->k);
    NE(t0, L->kp > 1024 ? N_CONV2 : L->kp > 256 ? N_SUB : N_GEMM);  // front-end GEMMs / long-K layers / K <= 256
}
static void nq_job(void *p, int b, int e, int w);
static void nquant(nctx_t *c, const float *x, int T, int K, int ldx, int kp)
{
    NB(t0);
    typedef struct { const float *x; int K, ldx, kp; nctx_t *c; } nqj2_t;
    nqj2_t j = {x, K, ldx, kp, c};
    tasr_parallel_work(nq_job, &j, T, (size_t)T * K * 3);
    NE(t0, N_QUANT);
}

// ------------------------------------------------------------------ attention job (per head)
typedef struct {
    nctx_t *c;
} natt_t;
static void natt_job(void *p, int b, int e, int w)
{
    nctx_t *c = ((natt_t *)p)->c;
    const tasr_nemo_t *m = c->m;
    const int T = c->T, dh = m->dh, dhp = m->dhp, d = m->d;
    const float scale = 1.0f / sqrtf((float)dh);
    float *sc = c->sc[w];
    int8_t *pq = c->pq[w];
    int32_t *ia = c->iacc[w];
    for (int hh = b; hh < e; hh++) {
        const int8_t *K = c->k8 + (size_t)hh * T * dhp;
        const int8_t *P = c->p8 + (size_t)hh * (2 * T - 1) * dhp;
        const int8_t *VT = c->vt + (size_t)hh * dh * c->Tp;
        const float *sk = c->sk + (size_t)hh * T, *sv = c->sv + (size_t)hh * T, *sp = c->sp + (size_t)hh * (2 * T - 1);
        for (int i = 0; i < T; i++) {
            const int8_t *qu = c->qu + ((size_t)hh * T + i) * dhp, *qv = c->qv + ((size_t)hh * T + i) * dhp;
            const float squ = c->squ[hh * T + i], sqv = c->sqv[hh * T + i];
            int32_t *ib = ia + c->iacc_n;
            const int m0 = T - 1 - i;  // score(i, j) uses pos row m0 + j
            if (dhp == 48) {
                tasr_dot48_rows(qu, K, 48, T, ia);
                tasr_dot48_rows(qv, P + (size_t)m0 * 48, 48, T, ib);
            } else {
                tasr_dot_rows_s8(qu, K, dhp, T, dhp, ia);
                tasr_dot_rows_s8(qv, P + (size_t)m0 * dhp, dhp, T, dhp, ib);
            }
            const float A = squ * scale, B = sqv * scale;
            const float *spm = sp + m0;
            float mx = -1e30f;
            for (int j = 0; j < T; j++) {
                float a = (float)ia[j] * A * sk[j] + (float)ib[j] * B * spm[j];
                sc[j] = a;
                mx = a > mx ? a : mx;
            }
            float sum = 0.f, pm = 0.f;
            const float mx64 = mx * NEXP_STEPS + 0.5f;
            for (int j = 0; j < T; j++) {
                const int ix = (int)(mx64 - sc[j] * NEXP_STEPS);  // >= 0: round((mx - s) * 64)
                const float ex = ix < NEXP_N ? nexp_tab[ix] : 0.f;
                sum += ex;
                float p2 = ex * sv[j];
                sc[j] = p2;
                pm = p2 > pm ? p2 : pm;
            }
            if (pm < 1e-30f) pm = 1e-30f;
            const float inv = 127.0f / pm;
            for (int j = 0; j < T; j++) { float y = sc[j] * inv + 12582912.0f; pq[j] = (int8_t)(int)(y - 12582912.0f); }
            for (int j = T; j < c->Tp; j++) pq[j] = 0;
            tasr_dot_rows_s8(pq, VT, c->Tp, dh, c->Tp, ia);
            const float os = pm / 127.0f / sum;
            float *y = c->att + (size_t)i * d + hh * dh;
            for (int e2 = 0; e2 < dh; e2++) y[e2] = (float)ia[e2] * os;
        }
    }
}

// depthwise conv (k, symmetric padding) + swish over all frames: job over channel ranges
typedef struct {
    const float *in;  // [T][d]
    float *out;       // [T][d]
    const nlayer_t *L;
    int T, d, k;
} ndw_t;
static void ndw_job(void *p, int b, int e, int w)
{
    (void)w;
    ndw_t *j = (ndw_t *)p;
    const int d = j->d;
    int t = b;
    for (; t + 1 < e; t += 2) {  // two frames per pass: each weight load feeds both (same summation order per output)
        float *o = j->out + (size_t)t * d, *o2 = o + d;
        const float *xin = j->in + (size_t)t * d;
        for (int ch = 0; ch < d; ch += 4) {
            float a0 = j->L->dw_b[ch], a1 = j->L->dw_b[ch + 1], a2 = j->L->dw_b[ch + 2], a3 = j->L->dw_b[ch + 3];
            float c0 = a0, c1 = a1, c2 = a2, c3 = a3;
            const float *wt = j->L->dw_wt + ch, *x = xin + ch;
            for (int kk = 0; kk < j->k; kk++, wt += d, x += d) {
                const float w0 = wt[0], w1 = wt[1], w2 = wt[2], w3 = wt[3];
                a0 += w0 * x[0]; a1 += w1 * x[1]; a2 += w2 * x[2]; a3 += w3 * x[3];
                c0 += w0 * x[d]; c1 += w1 * x[d + 1]; c2 += w2 * x[d + 2]; c3 += w3 * x[d + 3];
            }
            o[ch] = a0; o[ch + 1] = a1; o[ch + 2] = a2; o[ch + 3] = a3;
            o2[ch] = c0; o2[ch + 1] = c1; o2[ch + 2] = c2; o2[ch + 3] = c3;
        }
        for (int ch = 0; ch < d; ch++) o[ch] = o[ch] * nsig(o[ch]);
        for (int ch = 0; ch < d; ch++) o2[ch] = o2[ch] * nsig(o2[ch]);
    }
    for (; t < e; t++) {  // j->in has k/2 zero rows before frame 0 and after frame T-1
        float *o = j->out + (size_t)t * d;
        const float *xin = j->in + (size_t)t * d;
        for (int ch = 0; ch < d; ch += 4) {  // d % 4 == 0; 4 accumulators stay in registers across the k taps
            float a0 = j->L->dw_b[ch], a1 = j->L->dw_b[ch + 1], a2 = j->L->dw_b[ch + 2], a3 = j->L->dw_b[ch + 3];
            const float *wt = j->L->dw_wt + ch, *x = xin + ch;
            for (int kk = 0; kk < j->k; kk++, wt += d, x += d) {
                a0 += wt[0] * x[0]; a1 += wt[1] * x[1]; a2 += wt[2] * x[2]; a3 += wt[3] * x[3];
            }
            o[ch] = a0; o[ch + 1] = a1; o[ch + 2] = a2; o[ch + 3] = a3;
        }
        for (int ch = 0; ch < d; ch++) o[ch] = o[ch] * nsig(o[ch]);
    }
}

// ------------------------------------------------------------------ front end
static void power_spectrum512(const tasr_nemo_t *m, const float *x, float *re, float *im, float *pw)
{
    const int N2 = 256;
    for (int n = 0; n < N2; n++) {
        int j = m->bitrev[n];
        re[j] = x[2 * n];
        im[j] = x[2 * n + 1];
    }
    for (int len = 2; len <= N2; len <<= 1) {
        const int half = len >> 1, step = 2 * (N2 / len);
        for (int i = 0; i < N2; i += len)
            for (int j = 0; j < half; j++) {
                const float wr = m->tw_re[j * step], wi = m->tw_im[j * step];
                float *ar = &re[i + j], *ai = &im[i + j], *br = &re[i + j + half], *bi = &im[i + j + half];
                const float tr = *br * wr - *bi * wi, ti = *br * wi + *bi * wr;
                *br = *ar - tr; *bi = *ai - ti;
                *ar += tr; *ai += ti;
            }
    }
    pw[0] = (re[0] + im[0]) * (re[0] + im[0]);
    pw[N2] = (re[0] - im[0]) * (re[0] - im[0]);
    for (int k = 1; k < N2; k++) {
        const float zr = re[k], zi = im[k], cr = re[N2 - k], ci = -im[N2 - k];
        const float er = 0.5f * (zr + cr), ei = 0.5f * (zi + ci), dr = 0.5f * (zr - cr), di = 0.5f * (zi - ci);
        const float wr = m->tw_re[k], wi = m->tw_im[k];
        const float o_r = er + (wr * di + wi * dr), oi = ei - (wr * dr - wi * di);
        pw[k] = o_r * o_r + oi * oi;
    }
}

// features: preemph -> centered STFT -> mel -> log -> per-feature normalization; returns T = n/160 + 1 frames
typedef struct {
    const tasr_nemo_t *m;
    const int16_t *pcm;
    int n;
    float *F;
} nfeat_t;
static void nfeat_job(void *p, int b, int e, int w)
{
    (void)w;
    nfeat_t *j = (nfeat_t *)p;
    const tasr_nemo_t *m = j->m;
    float xw[512], re[256], im[256], pw[NBIN];
    for (int t = b; t < e; t++) {
        // frame t covers original samples [160t - 200, 160t + 200) (window offset inside the 512 frame does not
        // change |X|); samples outside [0, n) are zero (center=True, zero padding)
        for (int i = 0; i < WIN; i++) {
            const int si = HOP * t - 200 + i;
            float v = 0.f;
            if (si >= 0 && si < j->n) {
                const float x0 = j->pcm[si] / 32768.0f;
                v = si == 0 ? x0 : x0 - 0.97f * (j->pcm[si - 1] / 32768.0f);
            }
            xw[i] = v * m->window[i];
        }
        for (int i = WIN; i < 512; i++) xw[i] = 0.f;
        power_spectrum512(m, xw, re, im, pw);
        float *o = j->F + (size_t)t * NMEL;
        const float *wm = m->mel_w;
        for (int k = 0; k < NMEL; k++) {
            float acc = 0.f;
            const float *pp = pw + m->mel_start[k];
            for (int q = 0; q < m->mel_len[k]; q++) acc += pp[q] * wm[q];
            wm += m->mel_len[k];
            o[k] = logf(acc + 5.9604644775390625e-08f);
        }
    }
}
static float *nemo_features(const tasr_nemo_t *m, const int16_t *pcm, int n, int *T_out, float *F)
{
    const int nvalid = n / HOP, T = nvalid + 1;
    nfeat_t fj = {m, pcm, n, F};
    tasr_parallel_work(nfeat_job, &fj, T, (size_t)T * 2048);
    // Visit contiguous rows; each channel still accumulates frames in the original order.
#ifdef TASR_NEMO_FLOAT_STATS
    typedef float stat_t;  // experimental: opt-in, changes numerical results
#else
    typedef double stat_t;
#endif
    stat_t mean[NMEL] = {0}, var[NMEL] = {0};
    float sd[NMEL];
    for (int t = 0; t < nvalid; t++)
        for (int j = 0; j < NMEL; j++) mean[j] += F[(size_t)t * NMEL + j];
    for (int j = 0; j < NMEL; j++) mean[j] /= nvalid;
    for (int t = 0; t < nvalid; t++)
        for (int j = 0; j < NMEL; j++) {
            stat_t v = F[(size_t)t * NMEL + j] - mean[j];
            var[j] += v * v;
        }
    for (int j = 0; j < NMEL; j++) {
#ifdef TASR_NEMO_FLOAT_STATS
        sd[j] = sqrtf(var[j] / (nvalid > 1 ? nvalid - 1 : 1)) + 1e-5f;
#else
        sd[j] = (float)sqrt(var[j] / (nvalid > 1 ? nvalid - 1 : 1)) + 1e-5f;
#endif
    }
    for (int t = 0; t < nvalid; t++)
        for (int j = 0; j < NMEL; j++)
            F[(size_t)t * NMEL + j] = (F[(size_t)t * NMEL + j] - (float)mean[j]) / sd[j];
    memset(F + (size_t)nvalid * NMEL, 0, sizeof(float) * NMEL);
    *T_out = T;
    return F;
}

typedef struct {
    const tasr_nemo_t *m;
    const float (*P)[NMEL + 2];  // 3 zero-padded mel rows
    float *out;                  // [sc][f1 + 2] (zero column on both sides for the next conv)
    float *cmw;                  // [NW][f1 + 2] per-worker max over its channels of each output column
} nc0_t;
static void nc0_job(void *p, int b, int e, int w)
{
    nc0_t *j = (nc0_t *)p;
    const tasr_nemo_t *m = j->m;
    const int f1 = m->f1;
    float *cm = j->cmw + (size_t)w * (f1 + 2);
    for (int f = 0; f < f1 + 2; f++) cm[f] = 0.f;
    for (int ch = b; ch < e; ch++) {
        const float *wt = m->c0_w + ch * 9;
        const float w0 = wt[0], w1 = wt[1], w2 = wt[2], w3 = wt[3], w4 = wt[4], w5 = wt[5], w6 = wt[6], w7 = wt[7], w8 = wt[8];
        const float bias = m->c0_b[ch];
        float *o = j->out + (size_t)ch * (f1 + 2);
        o[0] = 0.f;
        o[f1 + 1] = 0.f;
        for (int f = 0; f < f1; f++) {
            const float *p0 = &j->P[0][2 * f], *p1 = &j->P[1][2 * f], *p2 = &j->P[2][2 * f];
            float acc = bias + w0 * p0[0] + w1 * p0[1] + w2 * p0[2] + w3 * p1[0] + w4 * p1[1] + w5 * p1[2] + w6 * p2[0] +
                        w7 * p2[1] + w8 * p2[2];
            const float v = acc > 0.f ? acc : 0.f;
            o[1 + f] = v;
            cm[1 + f] = v > cm[1 + f] ? v : cm[1 + f];
        }
    }
}

// im2col for 20 output positions of one frame + per-row int8 quantization (job over positions). conv0 outputs are
// ReLU'd (>= 0), so each patch row's max is the max of 3 x 3 per-(row, column) channel maxima (cmax, computed once per
// conv0 row); patches are then quantized while gathering, with exactly the arithmetic of tasr_quant_rows.
typedef struct {
    const tasr_nemo_t *m;
    const float *rows[3];  // conv0 rows 2t2-1..2t2+1 (padded layout) or NULL
    const float *cmax[3];  // [f1 + 2] max over channels of each padded column of those rows
    int8_t *col;           // [f2][kp]
    float *xs;
} nim_t;
static void nim_job(void *p, int b, int e, int w)
{
    (void)w;
    nim_t *j = (nim_t *)p;
    const tasr_nemo_t *m = j->m;
    const int sc = m->sc, f1p = m->f1 + 2, K = m->c2.k, kp = m->c2.kp;
    for (int f = b; f < e; f++) {
        float mx = 0.f;
        for (int dt = 0; dt < 3; dt++)
            if (j->rows[dt])
                for (int df = 0; df < 3; df++) {
                    const float v = j->cmax[dt][2 * f + df];
                    mx = v > mx ? v : mx;
                }
        if (mx < 1e-30f) mx = 1e-30f;
        const float inv = 127.0f / mx;
        int8_t *q = j->col + (size_t)f * kp;
        for (int dt = 0; dt < 3; dt++) {
            const float *r = j->rows[dt];
            int8_t *qd = q + dt * 3;
            if (!r) {
                for (int ch = 0; ch < sc; ch++) qd[ch * 9] = qd[ch * 9 + 1] = qd[ch * 9 + 2] = 0;
                continue;
            }
            const float *x = r + 2 * f;
            for (int ch = 0; ch < sc; ch++, x += f1p, qd += 9) {
                qd[0] = (int8_t)(int)((x[0] * inv + 12582912.0f) - 12582912.0f);
                qd[1] = (int8_t)(int)((x[1] * inv + 12582912.0f) - 12582912.0f);
                qd[2] = (int8_t)(int)((x[2] * inv + 12582912.0f) - 12582912.0f);
            }
        }
        for (int k = K; k < kp; k++) q[k] = 0;
        j->xs[f] = mx / 127.0f;
    }
}

// generic row-parallel helpers for the encoder
typedef struct {
    const float *x;
    float *y;
    int d, ldx, ldy;
    nln_t p;
} nlnj_t;
static void nln_job(void *p, int b, int e, int w)
{
    (void)w;
    nlnj_t *j = (nlnj_t *)p;
    for (int t = b; t < e; t++) layernorm_row(j->x + (size_t)t * j->ldx, j->d, j->p, j->y + (size_t)t * j->ldy);
}
static void nln_rows(const float *x, int T, int d, nln_t p, float *y)
{
    NB(t0);
    nlnj_t j = {x, y, d, d, d, p};
    tasr_parallel(nln_job, &j, T);
    NE(t0, N_LN);
}
static void nsilu_job(void *p, int b, int e, int w)
{
    (void)w;
    float *x = (float *)p;
    for (int i = b; i < e; i++) x[i] = x[i] * nsig(x[i]);
}
static void nsilu(float *x, int n)
{
    NB(t0);
    tasr_parallel(nsilu_job, x, n);
    NE(t0, N_ACT);
}
typedef struct {
    const float *x;
    int K, ldx, kp;
    nctx_t *c;
} nqj_t;
static void nq_job(void *p, int b, int e, int w)
{
    (void)w;
    nqj_t *j = (nqj_t *)p;
    tasr_quant_rows(j->x + (size_t)b * j->ldx, e - b, j->K, j->ldx, j->c->xq + (size_t)b * j->c->ldq, j->c->ldq, j->kp,
                    j->c->xs + b);
}

// Fused row jobs keep pointwise intermediates local and use one dispatch per stage.
typedef struct {
    nctx_t *c;
    const float *x;
    int K, kp;
    nln_t norm;
} nlnqj_t;
static void nlnq_job(void *p, int b, int e, int w)
{
    (void)w;
    nlnqj_t *j = p;
    float row[j->K];
    for (int t = b; t < e; t++) {
        layernorm_row(j->x + (size_t)t * j->K, j->K, j->norm, row);
        tasr_quant_rows(row, 1, j->K, j->K, j->c->xq + (size_t)t * j->c->ldq,
                        j->c->ldq, j->kp, j->c->xs + t);
    }
}
static void nln_quant(nctx_t *c, const float *x, int T, int K, nln_t norm, int kp)
{
    NB(t0);
    nlnqj_t j = {c, x, K, kp, norm};
    tasr_parallel_work(nlnq_job, &j, T, (size_t)T * K * 6);
    NE(t0, N_LNQ);
}
typedef struct { nctx_t *c; float *x; int K, kp; } nactqj_t;
static void nactq_job(void *p, int b, int e, int w)
{
    (void)w;
    nactqj_t *j = p;
    for (int t = b; t < e; t++) {
        float *row = j->x + (size_t)t * j->K;
        for (int k = 0; k < j->K; k++) row[k] = row[k] * nsig(row[k]);
        tasr_quant_rows(row, 1, j->K, j->K, j->c->xq + (size_t)t * j->c->ldq,
                        j->c->ldq, j->kp, j->c->xs + t);
    }
}
static void nact_quant(nctx_t *c, float *x, int T, int K, int kp)
{
    NB(t0);
    nactqj_t j = {c, x, K, kp};
    tasr_parallel_work(nactq_job, &j, T, (size_t)T * K * 4);
    NE(t0, N_ACTQ);
}
typedef struct { float *x; const float *y; int d; float scale; const nln_t *norm; } nresj_t;
static void nres_job(void *p, int b, int e, int w)
{
    (void)w;
    nresj_t *j = p;
    for (int t = b; t < e; t++) {
        float *row = j->x + (size_t)t * j->d;
        for (int k = 0; k < j->d; k++) row[k] += j->scale * j->y[(size_t)t * j->d + k];
        if (j->norm) layernorm_row(row, j->d, *j->norm, row);
    }
}
static void nresidual(float *x, const float *y, int T, int d, float scale, const nln_t *norm)
{
    NB(t0);
    nresj_t j = {x, y, d, scale, norm};
    tasr_parallel_work(nres_job, &j, T, (size_t)T * d * (norm ? 6 : 2));
    NE(t0, N_RES);
}
typedef struct { const float *x; float *out; int d; } ngluj_t;
static void nglu_job(void *p, int b, int e, int w)
{
    (void)w;
    ngluj_t *j = p;
    for (int t = b; t < e; t++) {
        const float *g = j->x + (size_t)t * 2 * j->d;
        float *o = j->out + (size_t)t * j->d;
        for (int ch = 0; ch < j->d; ch++) o[ch] = g[ch] * nsig(g[j->d + ch]);
    }
}
typedef struct { nctx_t *c; const nlayer_t *L; const float *h2; int t0, tn; } nqkvj_t;
static void nqkv_job(void *p, int b, int e, int w)
{
    (void)w;
    nqkvj_t *j = p;
    nctx_t *c = j->c;
    const int T = c->T, d = c->m->d, dh = c->m->dh, dhp = c->m->dhp;
    float tmp[64];
    int8_t vtmp[64];
    // A worker owns whole heads, including the transposed V output.
    for (int hh = b; hh < e; hh++) for (int t = 0; t < j->tn; t++) {
        const int ti = j->t0 + t;
        const float *q = j->h2 + (size_t)t * 3 * d, *k = q + d, *v = q + 2 * d;
        for (int k2 = 0; k2 < dh; k2++) tmp[k2] = q[hh * dh + k2] + j->L->pbu[hh * dh + k2];
        c->squ[hh * T + ti] = qvec(tmp, dh, c->qu + ((size_t)hh * T + ti) * dhp, dhp);
        for (int k2 = 0; k2 < dh; k2++) tmp[k2] = q[hh * dh + k2] + j->L->pbv[hh * dh + k2];
        c->sqv[hh * T + ti] = qvec(tmp, dh, c->qv + ((size_t)hh * T + ti) * dhp, dhp);
        c->sk[hh * T + ti] = qvec(k + hh * dh, dh, c->k8 + ((size_t)hh * T + ti) * dhp, dhp);
        c->sv[hh * T + ti] = qvec(v + hh * dh, dh, vtmp, dh);
        int8_t *vt = c->vt + (size_t)hh * dh * c->Tp + ti;
        for (int k2 = 0; k2 < dh; k2++) vt[(size_t)k2 * c->Tp] = vtmp[k2];
    }
}

// ------------------------------------------------------------------ RNN-T greedy decoding
// Exactly NeMo's greedy transducer search (up to 5 symbols per frame), with int8 GEMVs for the LSTM and joint network.
#define RNNT_P 320
#define RNNT_MAXSYM 5
#define RNNT_B 8  // joint-network frames evaluated per pass (see rnnt_greedy)
static float nsigm(float x) { return 1.0f / (1.0f + expf(-x)); }
typedef struct {
    float h[RNNT_P], c[RNNT_P], gp[RNNT_P];
} rnnt_state_t;
static void rnnt_pred_step(const tasr_nemo_t *m, nctx_t *C, const float *xin, rnnt_state_t *st)
{
    const int P = RNNT_P;
    float *gi = C->workspace->ptr[WS_RGI], *gh = C->workspace->ptr[WS_RGH];
    nquant(C, xin, 1, P, P, m->r_ih.kp);
    nqlin(C, &m->r_ih, 1, gi, 4 * P);
    nquant(C, st->h, 1, P, P, m->r_hh.kp);
    nqlin(C, &m->r_hh, 1, gh, 4 * P);
    for (int j = 0; j < P; j++) {  // PyTorch gate order i, f, g, o
        const float i = nsigm(gi[j] + gh[j]), f = nsigm(gi[P + j] + gh[P + j]);
        const float g = tanhf(gi[2 * P + j] + gh[2 * P + j]), o = nsigm(gi[3 * P + j] + gh[3 * P + j]);
        st->c[j] = f * st->c[j] + i * g;
        st->h[j] = o * tanhf(st->c[j]);
    }
    nquant(C, st->h, 1, P, P, m->r_jpred.kp);
    nqlin(C, &m->r_jpred, 1, st->gp, P);
}
static void rnnt_greedy(const tasr_nemo_t *m, nctx_t *C, const float *x, int T, char *text, int maxlen, int *len)
{
    const int P = RNNT_P, d = m->d, V1 = m->V + 1;
    float *fj = (float *)C->workspace->ptr[WS_RFJ];  // joint encoder projection, all frames
    for (int t0 = 0; t0 < T; t0 += RB) {
        const int tn = T - t0 < RB ? T - t0 : RB;
        nquant(C, x + (size_t)t0 * d, tn, d, d, m->r_jenc.kp);
        nqlin(C, &m->r_jenc, tn, fj + (size_t)t0 * P, P);
    }
    rnnt_state_t *state = C->workspace->ptr[WS_RST];
    float *e = C->workspace->ptr[WS_RE];
    float *z = (float *)C->workspace->ptr[WS_RZ];
    float *lo = (float *)C->workspace->ptr[WS_RLO];
    memset(state, 0, sizeof(*state));
    memset(e, 0, sizeof(float) * RNNT_P);
    rnnt_pred_step(m, C, e, state);  // start of sequence: zero input
    // Most frames emit only blank and leave the prediction state unchanged, so the joint network is evaluated for up to
    // RNNT_B frames at once with the current state (one pass over its 328 KB output matrix instead of one per frame).
    // Rows after the first frame that emits a token are discarded and recomputed, so the result is exactly greedy search.
    int t = 0;
    while (t < T) {
        const int nb = T - t < RNNT_B ? T - t : RNNT_B;
        for (int i = 0; i < nb; i++) {
            const float *f = fj + (size_t)(t + i) * P;
            float *zi = z + (size_t)i * P;
            for (int j = 0; j < P; j++) { const float v = f[j] + state->gp[j]; zi[j] = v > 0.f ? v : 0.f; }
        }
        nquant(C, z, nb, P, P, m->r_jout.kp);
        nqlin(C, &m->r_jout, nb, lo, V1);
        int i = 0, k = m->V;
        for (; i < nb; i++) {
            const float *l = lo + (size_t)i * V1;
            k = 0;
            for (int v = 1; v < V1; v++) if (l[v] > l[k]) k = v;
            if (k != m->V) break;
        }
        if (i == nb) { t += nb; continue; }  // whole window blank
        // frame t+i emits k: continue that frame symbol by symbol with the updated prediction state
        const float *f = fj + (size_t)(t + i) * P;
        for (int s = 0; s < RNNT_MAXSYM; s++) {
            if (s > 0) {
                for (int j = 0; j < P; j++) { const float v = f[j] + state->gp[j]; z[j] = v > 0.f ? v : 0.f; }
                nquant(C, z, 1, P, P, m->r_jout.kp);
                nqlin(C, &m->r_jout, 1, lo, V1);
                k = 0;
                for (int v = 1; v < V1; v++) if (lo[v] > lo[k]) k = v;
                if (k == m->V) break;
            }
            if (maxlen) {
                const int nl = m->tok_len[k];
                if (*len + nl + 1 < maxlen) { memcpy(text + *len, m->tok[k], nl); *len += nl; text[*len] = 0; }
            }
            const int8_t *er = m->r_emb.w + (size_t)k * m->r_emb.kp;  // embedding row, dequantized
            for (int j = 0; j < P; j++) e[j] = (float)er[j] * m->r_emb.s[k];
            rnnt_pred_step(m, C, e, state);
        }
        t += i + 1;
    }
}

// ------------------------------------------------------------------ main entry
int tasr_nemo_transcribe_with_workspace(tasr_nemo_workspace_t *ws, const int16_t *pcm, int n,
                                        tasr_decoder_t *dec, char *text, int maxlen,
                                        float *logit_sink, int max_frames, int *n_frames)
{
    if (n_frames) *n_frames = 0;
    if (text && maxlen > 0) text[0] = 0;
    if (!ws || !pcm || n < 0 || n > ws->max_samples || maxlen < 0 || max_frames < 0 ||
        (maxlen && !text) || (dec && !ws->use_decoder)) return -1;
    const tasr_nemo_t *m = ws->model;
    nsig_init();
    nexp_init();
    if (n < 2 * HOP) { if (maxlen) text[0] = 0; return 0; }
    const int d = m->d, H = m->h, dh = m->dh, dhp = m->dhp, sc = m->sc, f1 = m->f1, f2 = m->f2;
    int T0;
    NB(tf);
    float *F = nemo_features(m, pcm, n, &T0, ws->ptr[WS_F]);
    NE(tf, N_FEAT);
    const int T1 = (T0 - 1) / 2 + 1, T = (T1 - 1) / 2 + 1;
    nctx_t C;
    memset(&C, 0, sizeof(C));
    C.m = m; C.T = T; C.workspace = ws;
    C.ldq = m->ff > 2 * d ? m->ff : 2 * d;   // layer activations (K <= ff)
    C.xq = (int8_t *)ws->ptr[WS_XQ];
    C.xs = (float *)ws->ptr[WS_XS];
    int8_t *xq_sub = (int8_t *)ws->ptr[WS_XQ_SUB];   // front-end projection rows (K = 3520)
    for (int w = 0; w < NW; w++) {
        C.wtmp[w] = (int8_t *)ws->ptr[WS_WTMP0 + w];
        C.acc[w] = (int32_t *)ws->ptr[WS_ACC0 + w];
    }
    float *x = (float *)ws->ptr[WS_X];
    // ---- striding conv subsampling, pipelined over output frames
    {
        float *ring = (float *)ws->ptr[WS_RING];  // conv0 rows (r % 3), padded
        // conv2 runs as one GEMM per C2B output frames (C2B * f2 <= RB patch rows), so its weights are fetched once per
        // C2B frames instead of once per frame
        const int C2B = RB / f2 < 1 ? 1 : RB / f2;
        float *c2out = (float *)ws->ptr[WS_C2OUT];    // [C2B * f2][sc]
        float *fr = (float *)ws->ptr[WS_FR];       // one sub_out input row (float)
        float *xs_sub = (float *)ws->ptr[WS_XS_SUB];          // its per-row scales
        int8_t *col = (int8_t *)ws->ptr[WS_COL];
        int nbat = 0;
        float *cmaxr = (float *)ws->ptr[WS_CMAXR];  // per ring row: max over channels
        float *cmw = (float *)ws->ptr[WS_CMW];
        int have = -1, nflat = 0, t_flat0 = 0;
        for (int t2 = 0; t2 < T; t2++) {
            for (int r = 2 * t2 - 1; r <= 2 * t2 + 1; r++) {  // conv0 output rows needed
                if (r < 0 || r >= T1 || r <= have) continue;
                NB(tc0);
                float P[3][NMEL + 2];
                for (int dt = 0; dt < 3; dt++) {
                    const int ti = 2 * r - 1 + dt;
                    P[dt][0] = 0.f;
                    P[dt][NMEL + 1] = 0.f;
                    if (ti < 0 || ti >= T0) memset(&P[dt][1], 0, sizeof(float) * NMEL);
                    else memcpy(&P[dt][1], F + (size_t)ti * NMEL, sizeof(float) * NMEL);
                }
                nc0_t j = {m, (const float(*)[NMEL + 2])P, ring + (size_t)(r % 3) * sc * (f1 + 2), cmw};
                memset(cmw, 0, sizeof(float) * NW * (f1 + 2));  // a worker that gets no channels leaves zeros
                tasr_parallel_work(nc0_job, &j, sc, (size_t)sc * f1 * 9);
                {
                    float *cm = cmaxr + (size_t)(r % 3) * (f1 + 2);
                    for (int f = 0; f < f1 + 2; f++) {
                        float v = cmw[f];
                        for (int w = 1; w < NW; w++) v = cmw[(size_t)w * (f1 + 2) + f] > v ? cmw[(size_t)w * (f1 + 2) + f] : v;
                        cm[f] = v;
                    }
                }
                NE(tc0, N_CONV0);
                have = r;
            }
            NB(tim);
            {
                nim_t j;
                j.m = m;
                j.col = col + (size_t)nbat * f2 * m->c2.kp;
                j.xs = C.xs + nbat * f2;
                for (int dt = 0; dt < 3; dt++) {
                    const int r = 2 * t2 - 1 + dt;
                    j.rows[dt] = (r >= 0 && r < T1) ? ring + (size_t)(r % 3) * sc * (f1 + 2) : NULL;
                    j.cmax[dt] = cmaxr + (size_t)((r + 3) % 3) * (f1 + 2);
                }
                tasr_parallel_work(nim_job, &j, f2, (size_t)f2 * m->c2.k);
            }
            NE(tim, N_IM2COL);
            if (++nbat < C2B && t2 < T - 1) continue;
            {   // conv2 GEMM on nbat * f2 patch rows
                int8_t *save = C.xq;
                int ld = C.ldq;
                C.xq = col; C.ldq = m->c2.kp;
                nqlin(&C, &m->c2, nbat * f2, c2out, sc);
                C.xq = save; C.ldq = ld;
            }
            for (int fb = 0; fb < nbat; fb++) {
                const int tt = t2 - nbat + 1 + fb;
                const float *co = c2out + (size_t)fb * f2 * sc;
                for (int f = 0; f < f2; f++)
                    for (int ch = 0; ch < sc; ch++) {
                        const float v = co[f * sc + ch];
                        fr[ch * f2 + f] = v > 0.f ? v : 0.f;
                    }
                NB(tq);  // quantize the row now so only int8 rows are buffered
                tasr_quant_rows(fr, 1, sc * f2, sc * f2, xq_sub + (size_t)nflat * m->sub.kp, m->sub.kp, m->sub.kp,
                                xs_sub + nflat);
                NE(tq, N_QUANT);
                nflat++;
                if (nflat == RB || tt == T - 1) {
                    int8_t *save = C.xq;
                    float *save_s = C.xs;
                    int ld = C.ldq;
                    C.xq = xq_sub; C.ldq = m->sub.kp; C.xs = xs_sub;
                    nqlin(&C, &m->sub, nflat, x + (size_t)t_flat0 * d, d);
                    C.xq = save; C.ldq = ld; C.xs = save_s;
                    t_flat0 += nflat;
                    nflat = 0;
                }
            }
            nbat = 0;
        }

    }

    const float xscale = sqrtf((float)d);
    for (int i = 0; i < T * d; i++) x[i] *= xscale;
    // ---- relative positional encodings (int8, shared by all layers before linear_pos)
    const int NP = 2 * T - 1;
    C.Tp = (T + 15) & ~15;
    float *pe = (float *)ws->ptr[WS_PE];
    NB(tpe);
    for (int i = 0; i < d / 2; i++) {
        // row r holds position (T-1-r): step the angle by -div with a rotation, re-anchored every 64 rows
        const double div = exp((2.0 * i) * -(log(10000.0) / d));
        const float cd = (float)cos(div), sd = (float)sin(div);
        float sn = 0.f, cs = 1.f;
        for (int r = 0; r < NP; r++) {
            if ((r & 63) == 0) {
                const double a = fmod((double)(T - 1 - r) * div, 2.0 * M_PI);
                sn = (float)sin(a); cs = (float)cos(a);
            }
            pe[(size_t)r * d + 2 * i] = sn;
            pe[(size_t)r * d + 2 * i + 1] = cs;
            const float s2 = sn * cd - cs * sd, c2 = cs * cd + sn * sd;  // angle - div
            sn = s2; cs = c2;
        }
    }
    // the positional rows are the same input to every layer's linear_pos: quantize them to int8 once
    const int kpp = m->L[0].pos.kp;
    int8_t *peq = (int8_t *)ws->ptr[WS_PEQ];
    float *pes = (float *)ws->ptr[WS_PES];
    tasr_quant_rows(pe, NP, d, d, peq, kpp, kpp, pes);

    NE(tpe, N_POS);
    C.qu = (int8_t *)ws->ptr[WS_QU];
    C.qv = (int8_t *)ws->ptr[WS_QV];
    C.k8 = (int8_t *)ws->ptr[WS_K8];
    C.vt = (int8_t *)ws->ptr[WS_VT];
    C.p8 = (int8_t *)ws->ptr[WS_P8];
    C.squ = (float *)ws->ptr[WS_SQU];
    C.sqv = (float *)ws->ptr[WS_SQV];
    C.sk = (float *)ws->ptr[WS_SK];
    C.sv = (float *)ws->ptr[WS_SV];
    C.sp = (float *)ws->ptr[WS_SP];
    for (int w = 0; w < NW; w++) {
        C.sc[w] = (float *)ws->ptr[WS_SC0 + w];
        C.pq[w] = (int8_t *)ws->ptr[WS_PQ0 + w];
        C.iacc_n = C.Tp > 64 ? C.Tp : 64;
        C.iacc[w] = (int32_t *)ws->ptr[WS_IACC0 + w];
    }
    float *hb = (float *)ws->ptr[WS_HB];           // LN outputs (row block)
    float *h2 = (float *)ws->ptr[WS_H2];
    const int halo = m->k / 2;
    float *glb = (float *)ws->ptr[WS_GLB];  // GLU outputs with zero halo
    // Phase-shared storage is not globally cleared between utterances.
    // Only the depthwise convolution's leading/trailing halo requires zeros.
    memset(glb, 0, sizeof(float) * (size_t)halo * d);
    memset(glb + (size_t)(halo + T) * d, 0, sizeof(float) * (size_t)halo * d);
    float *gl = glb + (size_t)halo * d;
    float *cv = (float *)ws->ptr[WS_CV];
    C.att = cv;  // attention output is consumed by linear_out before the conv module writes cv
    float *prow = (float *)ws->ptr[WS_PROW];
    for (int li = 0; li < m->nl; li++) {
        const nlayer_t *L = &m->L[li];
        // FF1 (half step), row blocks
        for (int t0 = 0; t0 < T; t0 += RB) {
            const int tn = T - t0 < RB ? T - t0 : RB;
            nln_quant(&C, x + (size_t)t0 * d, tn, d, L->n_ff1, L->ff1_1.kp);
            nqlin(&C, &L->ff1_1, tn, h2, m->ff);
            nact_quant(&C, h2, tn, m->ff, L->ff1_2.kp);
            nqlin(&C, &L->ff1_2, tn, hb, d);
            nresidual(x + (size_t)t0 * d, hb, tn, d, 0.5f, NULL);
        }
        // MHSA: q/k/v for all frames (int8 per head), positional rows, then attention per head
        for (int t0 = 0; t0 < T; t0 += RB) {
            const int tn = T - t0 < RB ? T - t0 : RB;
            nln_quant(&C, x + (size_t)t0 * d, tn, d, L->n_att, L->qkv.kp);
            nqlin(&C, &L->qkv, tn, h2, 3 * d);
            NB(tq8);
            nqkvj_t qj = {&C, L, h2, t0, tn};
            tasr_parallel_work(nqkv_job, &qj, H, (size_t)tn * d * 8);
            NE(tq8, N_QKV8);
        }
        for (int hh = 0; hh < H; hh++)
            for (int e = 0; e < dh; e++)
                for (int t = T; t < C.Tp; t++) C.vt[((size_t)hh * dh + e) * C.Tp + t] = 0;
        for (int r0 = 0; r0 < NP; r0 += RB) {
            const int rn = NP - r0 < RB ? NP - r0 : RB;
            {
                int8_t *save = C.xq;
                float *save_s = C.xs;
                int ld = C.ldq;
                C.xq = peq + (size_t)r0 * kpp; C.ldq = kpp; C.xs = pes + r0;
                nqlin(&C, &L->pos, rn, prow, d);
                C.xq = save; C.ldq = ld; C.xs = save_s;
            }
            NB(tpq);
            for (int r = 0; r < rn; r++)
                for (int hh = 0; hh < H; hh++)
                    C.sp[hh * NP + r0 + r] = qvec(prow + (size_t)r * d + hh * dh, dh,
                                                  C.p8 + ((size_t)hh * NP + r0 + r) * dhp, dhp);
            NE(tpq, N_POS);
        }
        {
            NB(ta);
            natt_t aj = {&C};
            tasr_parallel_work(natt_job, &aj, H, (size_t)H * T * T * dh);
            NE(ta, N_ATT);
        }
        for (int t0 = 0; t0 < T; t0 += RB) {
            const int tn = T - t0 < RB ? T - t0 : RB;
            nquant(&C, C.att + (size_t)t0 * d, tn, d, d, L->out.kp);
            nqlin(&C, &L->out, tn, hb, d);
            nresidual(x + (size_t)t0 * d, hb, tn, d, 1.0f, NULL);
        }
        // conv module: LN -> pw1 -> GLU (all frames) -> dw k=31 (+BN) -> swish -> pw2
        for (int t0 = 0; t0 < T; t0 += RB) {
            const int tn = T - t0 < RB ? T - t0 : RB;
            nln_quant(&C, x + (size_t)t0 * d, tn, d, L->n_conv, L->pw1.kp);
            nqlin(&C, &L->pw1, tn, h2, 2 * d);
            NB(tgl);
            ngluj_t gj = {h2, gl + (size_t)t0 * d, d};
            tasr_parallel_work(nglu_job, &gj, tn, (size_t)tn * d * 4);
            NE(tgl, N_GLU);
        }
        {
            NB(td);
            ndw_t j = {glb, cv, L, T, d, m->k};
            tasr_parallel_work(ndw_job, &j, T, (size_t)T * d * m->k);
            NE(td, N_DW);
        }
        for (int t0 = 0; t0 < T; t0 += RB) {
            const int tn = T - t0 < RB ? T - t0 : RB;
            nquant(&C, cv + (size_t)t0 * d, tn, d, d, L->pw2.kp);
            nqlin(&C, &L->pw2, tn, hb, d);
            nresidual(x + (size_t)t0 * d, hb, tn, d, 1.0f, NULL);
        }
        // FF2 (half step) + final LN
        for (int t0 = 0; t0 < T; t0 += RB) {
            const int tn = T - t0 < RB ? T - t0 : RB;
            nln_quant(&C, x + (size_t)t0 * d, tn, d, L->n_ff2, L->ff2_1.kp);
            nqlin(&C, &L->ff2_1, tn, h2, m->ff);
            nact_quant(&C, h2, tn, m->ff, L->ff2_2.kp);
            nqlin(&C, &L->ff2_2, tn, hb, d);
            nresidual(x + (size_t)t0 * d, hb, tn, d, 0.5f, &L->n_out);
        }
    }
    if (m->rnnt) {
        NB(thd);
#ifdef TASR_PROFILE
        const uint64_t nested0 = nprof[N_QUANT] + nprof[N_GEMM] + nprof[N_SUB] + nprof[N_CONV2];
#endif
        int len = 0;
        if (maxlen) text[0] = 0;
        rnnt_greedy(m, &C, x, T, text, maxlen, &len);
        NE(thd, N_HEAD);
#ifdef TASR_PROFILE
        nprof[N_HEAD] -= nprof[N_QUANT] + nprof[N_GEMM] + nprof[N_SUB] + nprof[N_CONV2] - nested0;
#endif
        if (n_frames) *n_frames = T;
        if (maxlen && text[0] == ' ') memmove(text, text + 1, strlen(text));

        return T;
    }
    // ---- CTC head (blank = V) -> greedy or beam decoder (which expects blank at index 0)
    const int V1 = m->V + 1;
    float *lg = (float *)ws->ptr[WS_LG];
    float *lp = (float *)ws->ptr[WS_LP];
    int prev = -1, len = 0, nf = 0;
    if (dec) tasr_decoder_reset(dec);
    if (maxlen) text[0] = 0;
    for (int t0 = 0; t0 < T; t0 += RB) {
        const int tn = T - t0 < RB ? T - t0 : RB;
        nquant(&C, x + (size_t)t0 * d, tn, d, d, m->head.kp);
        nqlin(&C, &m->head, tn, lg, V1);
        for (int t = 0; t < tn; t++) {
            NB(tdec);
            const float *l = lg + (size_t)t * V1;
            if (logit_sink && nf < max_frames) memcpy(logit_sink + (size_t)nf * V1, l, sizeof(float) * V1);
            nf++;
            int best = 0;
            float bv = l[0];
            for (int v = 1; v < V1; v++) if (l[v] > bv) { bv = l[v]; best = v; }
            if (dec) {
                float z = 0.f;
                for (int v = 0; v < V1; v++) z += expf(l[v] - bv);
                const float lz = bv + logf(z);
                lp[0] = l[m->V] - lz;
                for (int v = 0; v < m->V; v++) lp[v + 1] = l[v] - lz;
                tasr_decoder_step(dec, lp);
                if (tasr_decoder_overflowed(dec)) { if (maxlen) text[0] = 0; return -1; }
            } else if (best != prev && best != m->V && maxlen) {
                const int nl = m->tok_len[best];
                if (len + nl + 1 < maxlen) { memcpy(text + len, m->tok[best], nl); len += nl; text[len] = 0; }
            }
            prev = best;
            NE(tdec, N_HEAD);
        }
    }
    NB(tfinish);
    if (dec && maxlen) {
        int toks[2048];
        const int nt = tasr_decoder_best(dec, toks, 2048);
        for (int i = 0; i < nt; i++) {
            const int nl = m->tok_len[toks[i]];
            if (len + nl + 1 < maxlen) { memcpy(text + len, m->tok[toks[i]], nl); len += nl; text[len] = 0; }
        }
    }
    NE(tfinish, N_HEAD);
    if (n_frames) *n_frames = nf;
    if (maxlen && text[0] == ' ') memmove(text, text + 1, strlen(text));

    return T;
}

// Compatibility wrapper: callers doing repeated inference should retain a workspace.
int tasr_nemo_transcribe(const tasr_nemo_t *m, const int16_t *pcm, int n, tasr_decoder_t *dec, char *text, int maxlen,
                         float *logit_sink, int max_frames, int *n_frames)
{
    if (n_frames) *n_frames = 0;
    if (text && maxlen > 0) text[0] = 0;
    if (!m || !pcm || n < 0 || maxlen < 0 || max_frames < 0 || (maxlen && !text)) return -1;
    if (n < 2 * HOP) return 0;
    tasr_nemo_workspace_t *ws = tasr_nemo_workspace_create(m, n, dec != NULL);
    if (!ws) return -1;
    const int rc = tasr_nemo_transcribe_with_workspace(ws, pcm, n, dec, text, maxlen, logit_sink, max_frames, n_frames);
    tasr_nemo_workspace_free(ws);
    return rc;
}
