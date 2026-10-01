// Optional SSD1306 128x64 I2C display showing the live transcript (menuconfig -> tinyasr -> OLED display).
// All functions are no-ops when CONFIG_TASR_OLED is off, so callers need no #ifdefs.
#pragma once
#include <stdint.h>

#define OLED_W 128
#define OLED_H 64

void oled_start(void);                  // I2C + panel init and the render task (once)
void oled_status(const char *s);        // top line, e.g. "listening" / "transcribing..."
void oled_push(const char *text);       // append a finished utterance to the history
void oled_partial(const char *text);    // replace the in-progress utterance (streaming model)

// portable: draw status + '\n'-separated history into a 1024-byte SSD1306 page buffer
void oled_render(uint8_t *fb, const char *status, const char *hist);
