// Utterance-level engine for NVIDIA NeMo Conformer-CTC small (13 M params, int8) — higher-accuracy mode.
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tasr_nemo tasr_nemo_t;
struct tasr_decoder;

tasr_nemo_t *tasr_nemo_load(const uint8_t *blob, size_t size);
void tasr_nemo_free(tasr_nemo_t *m);
size_t tasr_nemo_weight_bytes(const tasr_nemo_t *m);
size_t tasr_nemo_blob_bytes(const tasr_nemo_t *m);
// relocate weight matrices to tasr_alloc(kind 0) memory up to budget bytes (front end first)
size_t tasr_nemo_place_weights(tasr_nemo_t *m, size_t budget);

// Transcribe one utterance (16 kHz PCM). dec: optional CTC beam decoder (LM over the model's 1024 tokens), or NULL
// for greedy. logit_sink (optional): raw logits [frames][1025] (blank last). Returns encoder frames.
int tasr_nemo_transcribe(const tasr_nemo_t *m, const int16_t *pcm, int n, struct tasr_decoder *dec, char *text,
                         int maxlen, float *logit_sink, int max_frames, int *n_frames);

// profiling (TASR_PROFILE builds): per-stage cycle totals
const char *tasr_nemo_profile_name(int i);
uint64_t tasr_nemo_profile_value(int i);

#ifdef __cplusplus
}
#endif
