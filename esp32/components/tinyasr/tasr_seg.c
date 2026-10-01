#include "tasr_seg.h"
#include <math.h>
#include <string.h>

void tasr_seg_init(tasr_seg_t *s, int16_t *buf, int cap)
{
    memset(s, 0, sizeof(*s));
    s->gain = 4.0f;
    s->buf = buf;
    s->cap = cap;
}

void tasr_seg_next(tasr_seg_t *s)
{
    s->n = 0;
    s->speech = 0;
    s->silence = 0;
    s->voiced = 0;
}

int tasr_seg_vad(tasr_seg_t *s, const int16_t *in, int16_t *out, int k)
{
    float e = 0.f;
    for (int i = 0; i < k; i++) {
        const float x = in[i] / 32768.f;
        e += x * x;
    }
    e /= (k > 0 ? k : 1);
    // noise floor: follows dips quickly, creeps up slowly (+0.4 dB/s), so speech never teaches it its own level
    if (!s->init) { s->noise_floor = e > 1e-9f ? e : 1e-9f; s->init = 1; }
    if (e < s->noise_floor) s->noise_floor = 0.9f * s->noise_floor + 0.1f * e;
    else s->noise_floor *= 1.002f;
    if (s->noise_floor < 1e-9f) s->noise_floor = 1e-9f;
    // hysteresis: 9 dB above the floor to start speech, 4.8 dB to stay in it
    const int voiced = e > 1e-8f && e > (s->speech ? 3.0f : 8.0f) * s->noise_floor;
    if (voiced) {  // slow AGC towards ~-20 dBFS during speech
        const float rms = sqrtf(e) * s->gain;
        if (rms > 1e-5f) s->gain *= powf(0.1f / rms, 0.02f);
        if (s->gain > 64.f) s->gain = 64.f;
        if (s->gain < 0.25f) s->gain = 0.25f;
    }
    for (int i = 0; i < k; i++) {
        const float y = in[i] * s->gain;
        out[i] = (int16_t)(y > 32767.f ? 32767.f : (y < -32768.f ? -32768.f : y));
    }
    return voiced;
}

int tasr_seg_feed(tasr_seg_t *s, const int16_t *in, int k)
{
    if (k <= 0) return 0;
    if (k > 320) k = 320;
    int16_t blk[320];
    const int voiced = tasr_seg_vad(s, in, blk, k);
    if (voiced) {
        s->speech = 1;
        s->silence = 0;
        s->voiced++;
    } else {
        s->silence++;
    }
    if (!s->speech) {  // idle: keep a short pre-roll so the first phoneme is not cut
        if (s->n + k > TASR_SEG_PREROLL) {
            const int drop = s->n + k - TASR_SEG_PREROLL;
            memmove(s->buf, s->buf + drop, sizeof(int16_t) * (s->n - drop));
            s->n -= drop;
        }
        memcpy(s->buf + s->n, blk, sizeof(int16_t) * k);
        s->n += k;
        return 0;
    }
    const int room = s->cap - s->n;
    const int take = k < room ? k : room;
    memcpy(s->buf + s->n, blk, sizeof(int16_t) * take);
    s->n += take;
    if (s->silence > TASR_SEG_HANG_BLOCKS || s->n >= s->cap) {
        if (s->voiced >= TASR_SEG_MIN_VOICED) return s->n;
        s->speech = s->silence = s->voiced = 0;  // too short to be speech: back to idle, keep the tail as pre-roll
        if (s->n > TASR_SEG_PREROLL) {
            memmove(s->buf, s->buf + s->n - TASR_SEG_PREROLL, sizeof(int16_t) * TASR_SEG_PREROLL);
            s->n = TASR_SEG_PREROLL;
        }
    }
    return 0;
}
