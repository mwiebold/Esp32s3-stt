#pragma once
#include <stddef.h>
#include <stdint.h>
#include "tinyasr.h"

typedef struct tasr_lm tasr_lm_t;
typedef struct tasr_decoder tasr_decoder_t;

tasr_lm_t *tasr_lm_load(const uint8_t *blob, size_t size);
void tasr_lm_free(tasr_lm_t *lm);
// copy LM weight matrices into tasr_alloc(kind 0) memory (PSRAM on ESP32); returns bytes moved
size_t tasr_lm_to_ram(tasr_lm_t *lm);
// lm may be NULL (plain CTC prefix beam search)
tasr_decoder_t *tasr_decoder_create(const tasr_lm_t *lm, int V1, int beam, int topk, float lm_weight, float token_bonus);
void tasr_decoder_free(tasr_decoder_t *d);
void tasr_decoder_reset(tasr_decoder_t *d);
void tasr_decoder_step(tasr_decoder_t *d, const float *logp);
int tasr_decoder_best(tasr_decoder_t *d, int *toks, int max);
