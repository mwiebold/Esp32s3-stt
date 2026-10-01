// SPDX-License-Identifier: GPL-3.0-only
// USB keyboard output only. No host audio driver, synthetic shortcuts or Enter.
#include "usb_hid.h"
#include "sdkconfig.h"

#ifdef CONFIG_TASR_USB_HID
#include <stdio.h>
#include <string.h>
#include "hid_dictation_core.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "esp_timer.h"
#include "tinyusb.h"
#include "tusb.h"

#if !defined(CONFIG_TASR_MODE_MIC) || !defined(CONFIG_IDF_TARGET_ESP32S3)
#error USB dictation requires microphone mode on an ESP32-S3.
#endif
#if CONFIG_TINYUSB_HID_COUNT != 1
#error USB dictation requires CONFIG_TINYUSB_HID_COUNT=1 (use sdkconfig.usb_hid).
#endif
#if defined(CONFIG_TINYUSB_NO_DEFAULT_TASK) || CONFIG_TINYUSB_TASK_PRIORITY <= (configMAX_PRIORITIES - 2)
#error The default TinyUSB task must outrank ASR; use sdkconfig.usb_hid.
#endif
#if defined(CONFIG_TINYUSB_CDC_ENABLED) || defined(CONFIG_TINYUSB_MSC_ENABLED) || CONFIG_TINYUSB_MIDI_COUNT || CONFIG_TINYUSB_VENDOR_COUNT
#error This configuration descriptor is keyboard-only; disable other USB classes.
#endif

#define TEXT_CAPACITY 4096
#define TEXT_QUEUE_DEPTH 2
_Static_assert(sizeof(hid_key_report_t) == 8, "boot keyboard report must be 8 bytes");
static const char *TAG = "usb_hid";
typedef struct { uint32_t session; size_t length; char text[TEXT_CAPACITY]; } hid_message_t;
static QueueHandle_t s_queue;
static hid_message_t *s_message;
static hid_gate_t s_gate;
static atomic_uchar s_leds;
static atomic_uint s_bus_epoch;
static hid_key_report_t s_report;
static portMUX_TYPE s_report_lock = portMUX_INITIALIZER_UNLOCKED;
static char s_serial[13];

static const uint8_t s_hid_descriptor[] = { TUD_HID_REPORT_DESC_KEYBOARD() };
static const uint8_t s_config_descriptor[] = {
    // Bus powered, up to 500 mA; no remote wakeup, one boot-keyboard interface.
    TUD_CONFIG_DESCRIPTOR(1, 1, 0, TUD_CONFIG_DESC_LEN + TUD_HID_DESC_LEN, 0, 500),
    TUD_HID_DESCRIPTOR(0, 4, HID_ITF_PROTOCOL_KEYBOARD, sizeof(s_hid_descriptor), 0x81, 8, 10),
};
static const char *s_strings[] = {
    (const char[]){0x09, 0x04}, "Esp32s3-stt", "ESP32-S3 Dictation Keyboard", s_serial, "Dictation keyboard"
};

uint8_t const *tud_hid_descriptor_report_cb(uint8_t instance)
{
    (void)instance;
    return s_hid_descriptor;
}
uint16_t tud_hid_get_report_cb(uint8_t instance, uint8_t report_id, hid_report_type_t type,
                              uint8_t *buffer, uint16_t length)
{
    if (instance || report_id || !buffer) return 0;
    if (type == HID_REPORT_TYPE_OUTPUT && length) { buffer[0] = atomic_load(&s_leds); return 1; }
    if (type != HID_REPORT_TYPE_INPUT) return 0;
    const uint16_t n = length < sizeof(s_report) ? length : sizeof(s_report);
    portENTER_CRITICAL(&s_report_lock);
    memcpy(buffer, &s_report, n);
    portEXIT_CRITICAL(&s_report_lock);
    return n;
}
void tud_hid_set_report_cb(uint8_t instance, uint8_t report_id, hid_report_type_t type,
                          uint8_t const *buffer, uint16_t length)
{
    if (!instance && !report_id && type == HID_REPORT_TYPE_OUTPUT && length && buffer)
        atomic_store(&s_leds, buffer[0]);
}
static void bus_changed(bool connected)
{
    hid_gate_link(&s_gate, connected);  // always disarms and invalidates old generations
    atomic_fetch_add(&s_bus_epoch, 1);
    portENTER_CRITICAL(&s_report_lock);
    memset(&s_report, 0, sizeof(s_report));
    portEXIT_CRITICAL(&s_report_lock);
}
void tud_mount_cb(void) { atomic_store(&s_leds, 0); bus_changed(true); }
void tud_umount_cb(void) { bus_changed(false); }
void tud_suspend_cb(bool remote_wakeup_en) { (void)remote_wakeup_en; bus_changed(false); }
void tud_resume_cb(void) { bus_changed(true); }  // stays paused until BOOT is pressed again

uint32_t usb_hid_session(void) { return hid_gate_session(&s_gate); }
const char *usb_hid_status(void)
{
    const unsigned flags = hid_gate_flags(&s_gate);
    if (!(flags & HID_GATE_READY)) return "USB: loading model";
    if (!(flags & HID_GATE_LINK)) return "USB: connect / wake host";
    return (flags & HID_GATE_ARMED) ? "USB: dictation ON" : "USB: paused (BOOT=on)";
}
void usb_hid_input_ready(void) { hid_gate_ready(&s_gate, true); }
bool usb_hid_submit(const char *text, uint32_t captured_session)
{
    if (!s_queue || !captured_session || captured_session != usb_hid_session()) return false;
    hid_message_t message = {.session = captured_session};
    if (!hid_text_prepare(text, message.text, sizeof(message.text), &message.length)) {
        if (text && text[0]) ESP_LOGW(TAG, "text rejected: empty, overlong, non-ASCII, or control character");
        return false;
    }
    if (xQueueSend(s_queue, &message, 0) != pdTRUE) {
        ESP_LOGW(TAG, "keyboard queue full: dropping newest transcript");
        return false;
    }
    return true;
}

static bool send_report(void *ctx, const hid_key_report_t *report)
{
    const hid_tx_t *tx = (const hid_tx_t *)ctx;
    if (report->keys[0] && (!usb_hid_session() || usb_hid_session() != tx->session)) return false;
    if (!tud_mounted() || tud_suspended() || !tud_hid_ready()) return false;
    // Endpoint busy status acknowledges the previous report before a new one is
    // accepted. tx_step never advances on failed submission and always releases.
    if (!tud_hid_keyboard_report(0, report->modifier, report->keys)) return false;
    portENTER_CRITICAL(&s_report_lock);
    s_report = *report;
    portEXIT_CRITICAL(&s_report_lock);
    return true;
}

static void keyboard_task(void *arg)
{
    (void)arg;
    hid_tx_t tx;
    hid_tx_init(&tx, CONFIG_TASR_USB_REPORT_INTERVAL_MS);
    hid_button_t button = {0};
    unsigned last_epoch = atomic_load(&s_bus_epoch);
    unsigned last_flags = ~0u;
    uint32_t last_progress = (uint32_t)(esp_timer_get_time() / 1000);
    bool stalled = false;
    for (;;) {
        const uint32_t now = (uint32_t)(esp_timer_get_time() / 1000);
        if (hid_button_update(&button, !gpio_get_level(CONFIG_TASR_USB_BUTTON_GPIO), now))
            hid_gate_toggle(&s_gate);
        const unsigned epoch = atomic_load(&s_bus_epoch);
        if (epoch != last_epoch) { hid_tx_cancel(&tx); last_epoch = epoch; }
        const unsigned flags = hid_gate_flags(&s_gate);
        if (flags != last_flags) {
#if CONFIG_TASR_USB_LED_GPIO >= 0
            gpio_set_level(CONFIG_TASR_USB_LED_GPIO, (flags & 7u) == 7u);
#endif
            ESP_LOGI(TAG, "%s", usb_hid_status());
            last_flags = flags;
        }
        if (!hid_tx_busy(&tx)) {
            // Bound draining per pass so repeated stale submissions cannot starve
            // the button/USB service loop. Only this task owns s_message.
            for (int i = 0; i < TEXT_QUEUE_DEPTH; i++) {
                if (xQueueReceive(s_queue, s_message, 0) != pdTRUE) break;
                if (s_message->session == usb_hid_session() && s_message->session) {
                    hid_tx_begin(&tx, s_message->text, s_message->length, s_message->session);
                    break;
                }
            }
            last_progress = now; stalled = false;
        }
        if (hid_tx_step(&tx, usb_hid_session(), (atomic_load(&s_leds) & KEYBOARD_LED_CAPSLOCK) != 0,
                        now, send_report, &tx)) { last_progress = now; stalled = false; }
        if (!stalled && hid_tx_busy(&tx) && (uint32_t)(now - last_progress) >= 2000) {
            hid_gate_pause(&s_gate);
            hid_tx_cancel(&tx);
            ESP_LOGW(TAG, "USB stalled: paused and discarded pending text");
            stalled = true;
        }
        vTaskDelay(pdMS_TO_TICKS(2) ? pdMS_TO_TICKS(2) : 1);
    }
}

static bool pin_reserved(int pin)
{
    return pin == 19 || pin == 20 || (pin >= 26 && pin <= 37) ||
           pin == CONFIG_TASR_I2S_BCLK || pin == CONFIG_TASR_I2S_WS || pin == CONFIG_TASR_I2S_DIN
#ifdef CONFIG_TASR_OLED
           || pin == CONFIG_TASR_OLED_SDA || pin == CONFIG_TASR_OLED_SCL
#endif
           ;
}
esp_err_t usb_hid_init(void)
{
    if (s_queue) return ESP_OK;
    const int occupied[] = {CONFIG_TASR_I2S_BCLK, CONFIG_TASR_I2S_WS, CONFIG_TASR_I2S_DIN,
#ifdef CONFIG_TASR_OLED
        CONFIG_TASR_OLED_SDA, CONFIG_TASR_OLED_SCL,
#endif
    };
    for (size_t i = 0; i < sizeof(occupied) / sizeof(occupied[0]); i++)
        if (occupied[i] == 19 || occupied[i] == 20) return ESP_ERR_INVALID_ARG;
    const int button = CONFIG_TASR_USB_BUTTON_GPIO;
    if (!GPIO_IS_VALID_GPIO(button) || pin_reserved(button) || button == 46) return ESP_ERR_INVALID_ARG;
#if CONFIG_TASR_USB_LED_GPIO >= 0
    const int led = CONFIG_TASR_USB_LED_GPIO;
    if (!GPIO_IS_VALID_OUTPUT_GPIO(led) || pin_reserved(led) || led == button) return ESP_ERR_INVALID_ARG;
    const gpio_config_t led_config = {.pin_bit_mask = 1ULL << led, .mode = GPIO_MODE_OUTPUT};
    esp_err_t led_err = gpio_config(&led_config);
    if (led_err != ESP_OK) return led_err;
    gpio_set_level(led, 0);
#endif
    const gpio_config_t button_config = {
        .pin_bit_mask = 1ULL << button, .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE, .pull_down_en = GPIO_PULLDOWN_DISABLE, .intr_type = GPIO_INTR_DISABLE,
    };
    esp_err_t err = gpio_config(&button_config);
    if (err != ESP_OK) return err;
    hid_gate_init(&s_gate);
    uint8_t mac[6];
    err = esp_efuse_mac_get_default(mac);
    if (err != ESP_OK) return err;
    snprintf(s_serial, sizeof(s_serial), "%02X%02X%02X%02X%02X%02X", mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
    static StaticQueue_t queue_control;
    uint8_t *storage = heap_caps_malloc(TEXT_QUEUE_DEPTH * sizeof(hid_message_t), MALLOC_CAP_SPIRAM);
    s_message = heap_caps_malloc(sizeof(*s_message), MALLOC_CAP_SPIRAM);
    if (!storage || !s_message) { heap_caps_free(storage); heap_caps_free(s_message); s_message = NULL; return ESP_ERR_NO_MEM; }
    s_queue = xQueueCreateStatic(TEXT_QUEUE_DEPTH, sizeof(hid_message_t), storage, &queue_control);
    if (!s_queue) { heap_caps_free(storage); heap_caps_free(s_message); s_message = NULL; return ESP_ERR_NO_MEM; }
    const tinyusb_config_t config = {
        .device_descriptor = NULL,  // Espressif development VID/PID; not a commercial USB identity
        .string_descriptor = s_strings, .string_descriptor_count = sizeof(s_strings) / sizeof(s_strings[0]),
        .external_phy = false, .configuration_descriptor = s_config_descriptor,
    };
    err = tinyusb_driver_install(&config);
    if (err != ESP_OK) {
        vQueueDelete(s_queue); s_queue = NULL; heap_caps_free(storage); heap_caps_free(s_message); s_message = NULL;
        return err;
    }
    if (xTaskCreatePinnedToCore(keyboard_task, "dictation_hid", 4096, NULL, configMAX_PRIORITIES - 3, NULL, 0) != pdPASS) {
        tinyusb_driver_uninstall();
        vQueueDelete(s_queue); s_queue = NULL; heap_caps_free(storage); heap_caps_free(s_message); s_message = NULL;
        return ESP_ERR_NO_MEM;
    }
    ESP_LOGI(TAG, "keyboard ready; native USB only, US layout, BOOT toggles dictation after model load");
    return ESP_OK;
}
#else
esp_err_t usb_hid_init(void) { return ESP_OK; }
void usb_hid_input_ready(void) {}
uint32_t usb_hid_session(void) { return 1; }
bool usb_hid_submit(const char *text, uint32_t captured_session) { (void)text; (void)captured_session; return true; }
const char *usb_hid_status(void) { return "* listening"; }
#endif
