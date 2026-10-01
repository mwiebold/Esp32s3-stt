// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include <stdbool.h>
#include <stdint.h>
#include "esp_err.h"

// Initialize before weight caching so USB storage is included in the memory budget.
esp_err_t usb_hid_init(void);
void usb_hid_input_ready(void);
// Zero means paused/not connected/not ready; otherwise an opaque generation.
// With USB disabled these return session 1 and preserve serial-only behavior.
uint32_t usb_hid_session(void);
bool usb_hid_submit(const char *text, uint32_t captured_session);
const char *usb_hid_status(void);
