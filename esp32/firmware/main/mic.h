#pragma once
#include "tasr_nemo.h"
#include "tinyasr.h"

// Reserve audio queues/buffers before the remaining PSRAM is used for weights.
int mic_nemo_prepare(void);
void run_mic_nemo(tasr_nemo_workspace_t *ws, struct tasr_decoder *dec);
void run_mic(tasr_stream_t *s);
