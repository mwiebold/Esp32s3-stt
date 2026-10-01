// Utterance segmenter for live audio: slow AGC + energy VAD against a tracked noise floor (falls fast, rises slowly),
// with on/off hysteresis. Portable (the firmware mic task and host tests share it). Feed 16 kHz PCM in blocks of 320
// samples; an utterance is complete after TASR_SEG_HANG_BLOCKS non-speech blocks (0.8 s) or when the buffer is full.
// Bursts with fewer than TASR_SEG_MIN_VOICED speech blocks (clicks, bumps) are dropped.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TASR_SEG_PREROLL 3200   // samples of audio kept from before the speech onset (0.2 s)
#define TASR_SEG_HANG_BLOCKS 40
#define TASR_SEG_MIN_VOICED 10

typedef struct {
    float gain, noise_floor;
    int16_t *buf;  // utterance buffer (caller-owned)
    int cap, n, speech, silence, voiced, init;
    int last_voiced;  // end sample of the latest voiced block in the utterance
} tasr_seg_t;

void tasr_seg_init(tasr_seg_t *s, int16_t *buf, int cap);
// returns the utterance length in samples (in s->buf) when one is complete, else 0; call tasr_seg_next() after use
int tasr_seg_feed(tasr_seg_t *s, const int16_t *in, int k);
void tasr_seg_next(tasr_seg_t *s);
// -1 retains all endpointing silence (default). Nonnegative tail length is experimental.
// This changes only the inference slice, never the endpoint decision or stored audio.
int tasr_seg_audio_samples(const tasr_seg_t *s, int tail_samples);
// lower level: VAD decision for one block (uses s->speech for hysteresis) + AGC applied into out; returns voiced
int tasr_seg_vad(tasr_seg_t *s, const int16_t *in, int16_t *out, int k);

#ifdef __cplusplus
}
#endif
