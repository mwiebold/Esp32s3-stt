// SPDX-License-Identifier: GPL-3.0-only
#include "hid_dictation_core.h"
#include <string.h>

#define FLAGS 7u
static unsigned next_state(unsigned old, unsigned flags)
{
    return ((old + 8u) & ~FLAGS) | (flags & FLAGS);
}
void hid_gate_init(hid_gate_t *g) { atomic_init(&g->state, 0); }
static void gate_update(hid_gate_t *g, unsigned mask, unsigned value)
{
    unsigned old = atomic_load(&g->state), next;
    do {
        const unsigned flags = ((old & FLAGS) & ~(mask | HID_GATE_ARMED)) | value;
        next = next_state(old, flags);
    } while (!atomic_compare_exchange_weak(&g->state, &old, next));
}
void hid_gate_link(hid_gate_t *g, bool connected)
{
    gate_update(g, HID_GATE_LINK, connected ? HID_GATE_LINK : 0);
}
void hid_gate_ready(hid_gate_t *g, bool ready)
{
    gate_update(g, HID_GATE_READY, ready ? HID_GATE_READY : 0);
}
void hid_gate_pause(hid_gate_t *g) { gate_update(g, 0, 0); }
bool hid_gate_toggle(hid_gate_t *g)
{
    unsigned old = atomic_load(&g->state), next;
    do {
        if ((old & (HID_GATE_LINK | HID_GATE_READY)) != (HID_GATE_LINK | HID_GATE_READY)) return false;
        next = next_state(old, (old & FLAGS) ^ HID_GATE_ARMED);
    } while (!atomic_compare_exchange_weak(&g->state, &old, next));
    return true;
}
uint32_t hid_gate_session(const hid_gate_t *g)
{
    const unsigned state = atomic_load(&g->state);
    return (state & FLAGS) == FLAGS ? state : 0;
}
unsigned hid_gate_flags(const hid_gate_t *g) { return atomic_load(&g->state) & FLAGS; }

bool hid_button_update(hid_button_t *b, bool pressed, uint32_t now_ms)
{
    if (!b->initialized) {
        b->initialized = true;
        b->raw = b->stable = pressed;
        b->changed_ms = now_ms;
        return false;
    }
    if (pressed != b->raw) { b->raw = pressed; b->changed_ms = now_ms; }
    if ((uint32_t)(now_ms - b->changed_ms) < 30) return false;
    if (!pressed) b->released = true;
    if (b->stable == pressed) return false;
    b->stable = pressed;
    if (pressed && b->released) { b->released = false; return true; }
    return false;
}

bool hid_us_key(unsigned char ch, bool caps_lock, uint8_t *modifier, uint8_t *key)
{
    if (!modifier || !key) return false;
    *modifier = *key = 0;
    bool shift = false;
    if (ch >= 'a' && ch <= 'z') { *key = (uint8_t)(4 + ch - 'a'); shift = caps_lock; }
    else if (ch >= 'A' && ch <= 'Z') { *key = (uint8_t)(4 + ch - 'A'); shift = !caps_lock; }
    else if (ch >= '1' && ch <= '9') *key = (uint8_t)(30 + ch - '1');
    else if (ch == '0') *key = 39;
    else {
        // Keyboard Usage Page (0x07), US ANSI; never Enter/Tab/control chords.
        static const char plain[]   = " -=[]\\;'`,./";
        static const char shifted[] = " _+{}|:\"~<>?";
        static const uint8_t codes[] = {44,45,46,47,48,49,51,52,53,54,55,56};
        const char *p = ch ? strchr(plain, ch) : NULL;
        if (p) *key = codes[p - plain];
        else if (ch && (p = strchr(shifted, ch))) { *key = codes[p - shifted]; shift = true; }
        else {
            static const char digits[] = "!@#$%^&*()";
            p = ch ? strchr(digits, ch) : NULL;
            if (!p) return false;
            *key = (uint8_t)(30 + (p - digits)); shift = true;
        }
    }
    *modifier = shift ? 0x02 : 0;  // left shift only
    return true;
}

bool hid_text_prepare(const char *src, char *dst, size_t capacity, size_t *length)
{
    if (length) *length = 0;
    if (!src || !dst || capacity < 2) return false;
    dst[0] = 0;
    size_t n = 0;
    bool space = false;
    // Bound the source scan too: overlong transcripts are rejected, not truncated.
    for (size_t i = 0; i < capacity; i++) {
        const unsigned char ch = (unsigned char)src[i];
        if (!ch) {
            if (!n || n + 1 >= capacity) return false;
            dst[n++] = ' '; dst[n] = 0;
            if (length) *length = n;
            return true;
        }
        if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n') { space = n != 0; continue; }
        if (ch < 32 || ch > 126) { dst[0] = 0; return false; }
        if (n + (space ? 1u : 0u) + 2u >= capacity) { dst[0] = 0; return false; }
        if (space) dst[n++] = ' ';
        dst[n++] = (char)ch; space = false;
    }
    dst[0] = 0;
    return false;
}

void hid_tx_init(hid_tx_t *tx, unsigned interval_ms)
{
    memset(tx, 0, sizeof(*tx));
    tx->interval_ms = interval_ms;
    tx->release_pending = true;  // neutral report on initial enumeration
}
bool hid_tx_busy(const hid_tx_t *tx) { return tx->active || tx->release_pending; }
bool hid_tx_begin(hid_tx_t *tx, const char *text, size_t length, uint32_t session)
{
    if (hid_tx_busy(tx) || !text || !length || !session) return false;
    tx->text = text; tx->length = length; tx->position = 0; tx->session = session;
    tx->active = true;
    return true;
}
void hid_tx_cancel(hid_tx_t *tx)
{
    tx->active = false;
    tx->text = NULL;
    tx->release_pending = true;
}
bool hid_tx_step(hid_tx_t *tx, uint32_t live_session, bool caps_lock,
                 uint32_t now_ms, hid_report_send_fn send, void *ctx)
{
    if (tx->active && (!live_session || live_session != tx->session)) {
        tx->active = false; tx->text = NULL;
    }
    if (!send || (tx->paced && (uint32_t)(now_ms - tx->last_ms) < tx->interval_ms)) return false;
    hid_key_report_t report = {0};
    if (tx->release_pending) {
        if (!send(ctx, &report)) return false;
        tx->release_pending = false;
        if (tx->position >= tx->length) tx->active = false;
    } else if (tx->active) {
        if (!hid_us_key((unsigned char)tx->text[tx->position], caps_lock, &report.modifier, &report.keys[0])) {
            hid_tx_cancel(tx); return false;
        }
        if (!send(ctx, &report)) return false;
        tx->position++;
        tx->release_pending = true;
    } else return false;
    tx->last_ms = now_ms; tx->paced = true;
    return true;
}
