// I2S capture -> whole-frame queue -> continuous segmentation -> bounded utterance queue.
// Inference never owns the capture/segmentation task. Overflow drops a whole frame
// or utterance and is counted; sequence gaps discard partial speech rather than splice it.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdatomic.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "driver/i2s_std.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "sdkconfig.h"
#include "mic.h"
#include "tasr_seg.h"
#include "oled.h"

#ifdef CONFIG_TASR_MODE_MIC
#define MIC_CHECK(expr) do { if (!(expr)) { ESP_LOGE("mic", "failed: %s", #expr); abort(); } } while (0)
#define BLOCK_SAMPLES 320
#define MIC_QUEUE_BLOCKS 100  // two seconds; the consumer runs throughout inference
#define UTT_SAMPLES (16000 * CONFIG_TASR_MAX_UTTERANCE_SECONDS)
#define POOL_SIZE CONFIG_TASR_UTTERANCE_BUFFERS

typedef struct {
    int64_t end_us;
    uint32_t sequence;
    int16_t pcm[BLOCK_SAMPLES];
} mic_block_t;
typedef struct {
    int16_t *pcm;
    int samples;
    int64_t end_us;
} utterance_t;

static QueueHandle_t g_audio, g_free, g_ready;
static i2s_chan_handle_t g_rx;
static int16_t *g_pool[POOL_SIZE];
static atomic_uint g_dropped_samples, g_dropped_utterances, g_gaps, g_io_errors, g_queue_peak;

static int audio_queue_prepare(void)
{
    if (g_audio) return 1;
    static StaticQueue_t cb;
    uint8_t *storage = heap_caps_malloc(MIC_QUEUE_BLOCKS * sizeof(mic_block_t), MALLOC_CAP_SPIRAM);
    if (!storage) return 0;
    g_audio = xQueueCreateStatic(MIC_QUEUE_BLOCKS, sizeof(mic_block_t), storage, &cb);
    if (!g_audio) { heap_caps_free(storage); return 0; }
    return 1;
}

int mic_nemo_prepare(void)
{
    if (g_ready) return 1;
    if (!audio_queue_prepare()) return 0;
    g_free = xQueueCreate(POOL_SIZE, sizeof(int16_t *));
    g_ready = xQueueCreate(POOL_SIZE - 1, sizeof(utterance_t));
    if (!g_free || !g_ready) goto fail;
    for (int i = 0; i < POOL_SIZE; i++) {
        g_pool[i] = heap_caps_malloc(sizeof(int16_t) * UTT_SAMPLES, MALLOC_CAP_SPIRAM);
        if (!g_pool[i]) goto fail;
        if (xQueueSend(g_free, &g_pool[i], 0) != pdTRUE) goto fail;
    }
    return 1;
fail:
    for (int i = 0; i < POOL_SIZE; i++) { heap_caps_free(g_pool[i]); g_pool[i] = NULL; }
    if (g_free) vQueueDelete(g_free);
    if (g_ready) vQueueDelete(g_ready);
    g_free = g_ready = NULL;
    return 0;
}

static void capture_task(void *arg)
{
    (void)arg;
    int32_t raw[BLOCK_SAMPLES];
    mic_block_t block;
    uint32_t sequence = 0;
    float dc = 0.f;
    for (;;) {
        size_t got = 0;
        const esp_err_t rc = i2s_channel_read(g_rx, raw, sizeof(raw), &got, portMAX_DELAY);
        block.sequence = sequence++;
        block.end_us = esp_timer_get_time();
        if (rc != ESP_OK || got != sizeof(raw)) {
            atomic_fetch_add_explicit(&g_io_errors, 1, memory_order_relaxed);
            atomic_fetch_add_explicit(&g_dropped_samples, got / sizeof(raw[0]), memory_order_relaxed);
            vTaskDelay(1);  // do not spin at maximum priority after a driver error
            continue;
        }
        for (int i = 0; i < BLOCK_SAMPLES; i++) {
            const float x = (float)(raw[i] >> 8) / 256.0f;
            dc += 0.001f * (x - dc);
            const float y = x - dc;
            block.pcm[i] = (int16_t)(y > 32767.f ? 32767.f : (y < -32768.f ? -32768.f : y));
        }
        if (xQueueSend(g_audio, &block, 0) != pdTRUE)
            atomic_fetch_add_explicit(&g_dropped_samples, BLOCK_SAMPLES, memory_order_relaxed);
    }
}

static void mic_start(void)
{
    static int started;
    if (started) return;
    MIC_CHECK(audio_queue_prepare());
    i2s_chan_config_t cc = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    cc.dma_desc_num = 8;
    cc.dma_frame_num = BLOCK_SAMPLES;
    ESP_ERROR_CHECK(i2s_new_channel(&cc, NULL, &g_rx));
    i2s_std_config_t sc = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(16000),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {.mclk = I2S_GPIO_UNUSED, .bclk = CONFIG_TASR_I2S_BCLK, .ws = CONFIG_TASR_I2S_WS,
                     .dout = I2S_GPIO_UNUSED, .din = CONFIG_TASR_I2S_DIN},
    };
    sc.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(g_rx, &sc));
    ESP_ERROR_CHECK(i2s_channel_enable(g_rx));
    MIC_CHECK(xTaskCreatePinnedToCore(capture_task, "mic_cap", 4096, NULL,
                                        configMAX_PRIORITIES - 1, NULL, 0) == pdPASS);
    started = 1;
    ESP_LOGI("mic", "listening (BCLK %d WS %d DIN %d)", CONFIG_TASR_I2S_BCLK, CONFIG_TASR_I2S_WS, CONFIG_TASR_I2S_DIN);
}

static void report_overflow(void)
{
    static int64_t last_us;
    static unsigned prev_samples, prev_utts, prev_errors;
    const int64_t now = esp_timer_get_time();
    if (now - last_us < 5000000) return;
    last_us = now;
    const unsigned samples = atomic_load_explicit(&g_dropped_samples, memory_order_relaxed);
    const unsigned utts = atomic_load_explicit(&g_dropped_utterances, memory_order_relaxed);
    const unsigned errors = atomic_load_explicit(&g_io_errors, memory_order_relaxed);
    if (samples != prev_samples || utts != prev_utts || errors != prev_errors) {
        ESP_LOGW("mic", "dropped samples %u, utterances %u, sequence gaps %u, I2S errors %u, queue peak %u",
                 samples, utts, atomic_load_explicit(&g_gaps, memory_order_relaxed), errors,
                 atomic_load_explicit(&g_queue_peak, memory_order_relaxed));
        prev_samples = samples; prev_utts = utts; prev_errors = errors;
    }
}

static void segment_task(void *arg)
{
    (void)arg;
    int16_t *recording = NULL;
    MIC_CHECK(xQueueReceive(g_free, &recording, portMAX_DELAY) == pdTRUE);
    tasr_seg_t seg;
    tasr_seg_init(&seg, recording, UTT_SAMPLES);
    mic_block_t block;
    uint32_t expected = 0;
    int have_sequence = 0;
    for (;;) {
        if (xQueueReceive(g_audio, &block, portMAX_DELAY) != pdTRUE) continue;
        if (have_sequence && block.sequence != expected) {
            if (seg.speech) atomic_fetch_add_explicit(&g_dropped_utterances, 1, memory_order_relaxed);
            atomic_fetch_add_explicit(&g_gaps, 1, memory_order_relaxed);
            tasr_seg_next(&seg);
        }
        expected = block.sequence + 1;
        have_sequence = 1;
        if (!tasr_seg_feed(&seg, block.pcm, BLOCK_SAMPLES)) continue;
        const int tail_ms = CONFIG_TASR_INFERENCE_TAIL_MS;
        utterance_t utt = {recording, tasr_seg_audio_samples(&seg, tail_ms < 0 ? -1 : tail_ms * 16), block.end_us};
        int16_t *next = NULL;
        // Keep one buffer for recording. If no free buffer remains, discard this
        // completed utterance, never block segmentation or overwrite queued audio.
        if (xQueueReceive(g_free, &next, 0) == pdTRUE) {
            if (xQueueSend(g_ready, &utt, 0) == pdTRUE) {
                recording = next;
                const unsigned queued = uxQueueMessagesWaiting(g_ready);
                if (queued > atomic_load_explicit(&g_queue_peak, memory_order_relaxed))
                    atomic_store_explicit(&g_queue_peak, queued, memory_order_relaxed);
            } else {
                MIC_CHECK(xQueueSend(g_free, &next, 0) == pdTRUE);
                atomic_fetch_add_explicit(&g_dropped_utterances, 1, memory_order_relaxed);
            }
        } else atomic_fetch_add_explicit(&g_dropped_utterances, 1, memory_order_relaxed);
        tasr_seg_next(&seg);
        seg.buf = recording;
    }
}

void run_mic_nemo(tasr_nemo_workspace_t *ws, struct tasr_decoder *dec)
{
    MIC_CHECK(ws && mic_nemo_prepare());
    oled_start();
    oled_status("* listening");
    MIC_CHECK(xTaskCreatePinnedToCore(segment_task, "mic_seg", 8192, NULL,
                                        configMAX_PRIORITIES - 2, NULL, 0) == pdPASS);
    mic_start();
    static char text[4096];
    utterance_t utt;
    printf("ready: speak, the transcript prints after each pause\n");
    for (;;) {
        if (xQueueReceive(g_ready, &utt, pdMS_TO_TICKS(1000)) != pdTRUE) { report_overflow(); continue; }
        oled_status("transcribing...");
        const int64_t t0 = esp_timer_get_time();
        const int rc = tasr_nemo_transcribe_with_workspace(ws, utt.pcm, utt.samples, dec, text, sizeof(text), NULL, 0, NULL);
        const int64_t done = esp_timer_get_time();
        MIC_CHECK(xQueueSend(g_free, &utt.pcm, 0) == pdTRUE);
        if (rc < 0) {
            ESP_LOGE("mic", "inference failed; utterance discarded");
            oled_status("inference error");
        } else {
            const double dt = (done - t0) / 1e6, audio_s = utt.samples / 16000.0;
            printf("%s   [%.2f s input, %.3f s compute, RTF %.3f, queue %.3f s, endpoint-to-text %.3f s]\n",
                   text, audio_s, dt, dt / audio_s, (t0 - utt.end_us) / 1e6, (done - utt.end_us) / 1e6);
            oled_push(text[0] ? text : "(nothing recognized)");
            oled_status("* listening");
        }
        report_overflow();
    }
}

void run_mic(tasr_stream_t *s)
{
    mic_start();
    int16_t pcm[BLOCK_SAMPLES];
    tasr_seg_t vad;
    tasr_seg_init(&vad, NULL, 0);
    oled_start();
    oled_status("* listening (stream)");
    mic_block_t block;
    uint32_t expected = 0;
    int have_sequence = 0;
    for (;;) {
        if (xQueueReceive(g_audio, &block, portMAX_DELAY) != pdTRUE) continue;
        if (have_sequence && block.sequence != expected) {
            tasr_stream_reset(s);
            tasr_seg_next(&vad);
            atomic_fetch_add_explicit(&g_gaps, 1, memory_order_relaxed);
        }
        expected = block.sequence + 1; have_sequence = 1;
        const int voiced = tasr_seg_vad(&vad, block.pcm, pcm, BLOCK_SAMPLES);
        if (voiced) { vad.speech = 1; vad.silence = 0; }
        else vad.silence++;
        if (!vad.speech) { report_overflow(); continue; }
        if (tasr_stream_feed(s, pcm, BLOCK_SAMPLES) > 0) {
            printf("\r%s", tasr_stream_text(s));
            fflush(stdout);
            oled_partial(tasr_stream_text(s));
        }
        if (vad.silence > TASR_SEG_HANG_BLOCKS) {
            tasr_stream_finish(s);
            printf("\r%s\n", tasr_stream_text(s));
            oled_push(tasr_stream_text(s));
            tasr_stream_reset(s);
            vad.speech = 0;
        }
        report_overflow();
    }
}
#else
int mic_nemo_prepare(void) { return 1; }
void run_mic_nemo(tasr_nemo_workspace_t *ws, struct tasr_decoder *dec) { (void)ws; (void)dec; }
void run_mic(tasr_stream_t *s) { (void)s; }
#endif
