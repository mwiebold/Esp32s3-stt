// Utterance-level engine for NVIDIA NeMo Conformer-CTC small (13 M params, int8) — higher-accuracy mode.
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tasr_nemo tasr_nemo_t;
typedef struct tasr_nemo_workspace tasr_nemo_workspace_t;
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

// Reusable scratch for a fixed model and sample capacity. Allocate BEFORE weight caching.
// The model must outlive its workspace. One caller per workspace; no concurrent calls.
// use_decoder reserves optional CTC beam scratch. Returns NULL on allocation failure.
tasr_nemo_workspace_t *tasr_nemo_workspace_create(const tasr_nemo_t *m, int max_samples, int use_decoder);
void tasr_nemo_workspace_free(tasr_nemo_workspace_t *ws);
size_t tasr_nemo_workspace_bytes(const tasr_nemo_workspace_t *ws);
// No heap allocations during inference. -1 = invalid input/capacity or decoder overflow.
int tasr_nemo_transcribe_with_workspace(tasr_nemo_workspace_t *ws, const int16_t *pcm, int n,
                                        struct tasr_decoder *dec, char *text, int maxlen,
                                        float *logit_sink, int max_frames, int *n_frames);

// Exclusive profiling (TASR_PROFILE): microseconds on ESP32, nanoseconds on host.
void tasr_nemo_profile_reset(void);
const char *tasr_nemo_profile_name(int i);
uint64_t tasr_nemo_profile_value(int i);

#ifdef __cplusplus
}
#endif
