// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdatomic.h>

// One atomic word couples readiness, arming, and generation. A pause, USB bus
// event, or readiness change invalidates every queued audio/text session.
typedef struct { atomic_uint state; } hid_gate_t;
void hid_gate_init(hid_gate_t *g);
void hid_gate_link(hid_gate_t *g, bool connected);
void hid_gate_ready(hid_gate_t *g, bool ready);
void hid_gate_pause(hid_gate_t *g);
bool hid_gate_toggle(hid_gate_t *g);
uint32_t hid_gate_session(const hid_gate_t *g);
unsigned hid_gate_flags(const hid_gate_t *g);
enum { HID_GATE_ARMED = 1, HID_GATE_LINK = 2, HID_GATE_READY = 4 };

// Debounced active-low button. A held button at startup never arms dictation;
// one stable release is required before the first press can be accepted.
typedef struct {
    uint32_t changed_ms;
    bool initialized, raw, stable, released;
} hid_button_t;
bool hid_button_update(hid_button_t *b, bool pressed, uint32_t now_ms);

// US ANSI keyboard mapping. Only printable ASCII is representable. The text
// preparer collapses whitespace, rejects controls/Unicode, and appends one space.
bool hid_us_key(unsigned char ch, bool caps_lock, uint8_t *modifier, uint8_t *key);
bool hid_text_prepare(const char *src, char *dst, size_t capacity, size_t *length);

typedef struct { uint8_t modifier, reserved, keys[6]; } hid_key_report_t;
typedef bool (*hid_report_send_fn)(void *ctx, const hid_key_report_t *report);
typedef struct {
    const char *text;
    size_t length, position;
    uint32_t session, last_ms;
    unsigned interval_ms;
    bool active, release_pending, paced;
} hid_tx_t;
void hid_tx_init(hid_tx_t *tx, unsigned interval_ms);
bool hid_tx_busy(const hid_tx_t *tx);
bool hid_tx_begin(hid_tx_t *tx, const char *text, size_t length, uint32_t session);
void hid_tx_cancel(hid_tx_t *tx);
// send() must return false unless a report was accepted. Rejected transfers are
// retried without advancing text. Releases are sent even after a session ends.
bool hid_tx_step(hid_tx_t *tx, uint32_t live_session, bool caps_lock,
                 uint32_t now_ms, hid_report_send_fn send, void *ctx);
