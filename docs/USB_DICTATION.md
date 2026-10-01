# USB keyboard dictation

An optional ESP32-S3 firmware mode turns recognized speech into standard USB HID
keyboard reports. Recognition stays on the ESP32. The computer needs no speech
service or companion application. This is a keyboard, **not a USB audio input**.

## Hardware

- ESP32-S3 with **16 MB flash and 8 MB octal PSRAM**, e.g. ESP32-S3-DevKitC-1 N16R8.
- **INMP441 I2S microphone breakout** (not a bare surface-mount microphone).
- A USB **data** cable connected to the board's native USB/OTG port.
- Optional SSD1306 display; optional ordinary LED and series resistor on GPIO7.

The default microphone connections are:

| INMP441 breakout | ESP32-S3 |
|---|---|
| VDD / VCC | 3V3, not 5V |
| GND | GND |
| SCK / BCLK | GPIO4 |
| WS / LRCL | GPIO5 |
| SD / DOUT | GPIO6 |
| L/R / SEL | GND (left channel) |

An enable pin, if exposed by the breakout, must be asserted as its manufacturer
specifies. Keep the microphone's acoustic port open and the I2S wires short. Start
with the microphone roughly 10–20 cm from your mouth and test placement; do not
expect one omnidirectional microphone to reject a noisy room like a close-talk headset.

The INMP441 is the existing engine's supported interface. TDK lists the chip as
"Production (NRND)": reasonable for this prototype, not the choice to freeze into
a new commercial hardware design without checking lifecycle and alternatives.

Native USB D- is GPIO19 and D+ is GPIO20. The USB-to-UART bridge port (CP210x/CH340,
etc.) cannot enumerate as this keyboard. On the official dual-port DevKitC, use the
port marked **USB**, not **UART**. Keep GPIO19/20 unused by other peripherals.
An ordinary ESP32/WROOM-32 and the USB-Serial/JTAG-only ESP32-C3 are not substitutes.

This descriptor is bus-powered. For normal use, power it from the host's native
USB cable alone. Flash through UART, disconnect that cable, and connect native USB.
A separately powered design requires VBUS monitoring; do not assume this bus-powered
configuration is a compliant self-powered USB implementation.

## Build and flash

Use ESP-IDF **5.5** and the existing flash script from the repository root:

```bash
# Replace the serial port with your UART flashing port.
TASR_USB_HID=1 esp32/tools/flash.sh /dev/ttyUSB0 models/nemo8.tnm

# Smaller int4 model, no language model:
TASR_USB_HID=1 NO_LM=1 esp32/tools/flash.sh /dev/ttyUSB0 models/nemo4.tnm

# Optional status/transcript display (SDA 8, SCL 9):
TASR_USB_HID=1 TASR_OLED=1 esp32/tools/flash.sh /dev/ttyUSB0 models/nemo8.tnm
```

On macOS the flashing port is commonly `/dev/cu.usbserial-*`; on Windows it is
usually `COMx`. Select the actual port, not the illustrative name above. Build
directories/configuration names include `_hid`, so serial-only builds stay separate.
The registry dependencies are pinned to `esp_tinyusb 1.7.6` and `tinyusb 0.18.0~4`.
The first build needs access to the Espressif component registry.

Manual build:

```bash
cd esp32/firmware
idf.py -B build_hid -D SDKCONFIG=sdkconfig_hid \
  -D 'SDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.nemo16lm;sdkconfig.mic;sdkconfig.usb_hid' build
```

This compiles the application; the model and optional LM partitions must also be
flashed. `tools/flash.sh` handles those images. Bare `idf.py flash` alone does not
install the model. Do not flash a microphone image over a different partition layout
without also installing the matching model image.

## Use

1. Connect native USB. Set the host input source/layout to **US ANSI**. macOS may
   present Keyboard Setup Assistant; choose ANSI/US or dismiss it and select the
   US input source. The device cannot supply the physical-key identification step.
2. Focus a plain text editor for the first test. Wait for model loading. The serial
   log/OLED will say `USB: paused (BOOT=on)` when ready.
3. Tap **BOOT** (GPIO0) to arm. This is a toggle, not push-to-talk. Wait a moment
   after arming (the capture path discards 200 ms of pre-existing DMA audio).
4. Speak a phrase, then pause. After endpoint detection and inference, the keyboard
   types the recognized phrase followed by one space.
5. Wait until typing finishes, then tap BOOT to pause. Pausing cancels queued or
   currently transcribing text; it is not a "finish and send" action.

The computer receives `ESP32-S3 Dictation Keyboard`. The optional GPIO7 LED is on
only while dictation is armed, the model is ready, and USB is connected/awake. To
use it, set `CONFIG_TASR_USB_LED_GPIO=7` in that build's menuconfig and connect an
ordinary LED through an appropriate series resistor. Do not use this setting for
the DevKit's addressable RGB LED.

Holding BOOT during reset invokes the ESP32 ROM download mode rather than the
application. A button already held when the application starts cannot arm it; a
stable release and a new press are required.

**Text goes wherever the cursor is.** HID cannot identify the focused application,
password fields, terminal prompts, or other keyboards' held Ctrl/Alt/Command keys.
Do not change focus or hold shortcut modifiers while it is typing. BOOT pauses;
the USB cable disconnects it. It never sends Enter, Tab, Backspace, or modifier
shortcuts, and it never interprets spoken words as operating-system commands.

## Timing and limitations

- Output is still **utterance-based**, not word-by-word streaming. The existing
  segmenter waits approximately 0.82 seconds of silence; inference and queue time
  come afterward. No hardware latency or new accuracy claim is made here.
- The default keyboard pace is one report every 10 ms, with separate key-down and
  key-up reports: at most about 50 characters/second before host/RTOS overhead.
- English recognition and printable ASCII on a **US host layout**. Caps Lock state
  is honored. Tabs/newlines in text are normalized to spaces; other controls and
  non-ASCII/overlong transcripts are rejected as a whole rather than partly typed.
- No automatic punctuation/capitalization, spoken editing commands, or Unicode
  composition shortcuts were added. Recognition errors are typed as recognized.
- Two queued text messages plus one active message are bounded. If full, the newest
  transcript is dropped with a log warning; inference/capture never blocks on HID.
- Pausing, USB disconnect, suspend, resume, or a USB transfer stall invalidates old
  text. Resume/reconnect starts paused. A key already accepted by the host cannot
  be retracted; its release is sent as soon as the endpoint can accept it.
- During USB disconnect/suspend the device does not attempt to wake the computer.
- The code shares the same USB PHY as USB-Serial/JTAG; diagnostics remain on UART.
- Development uses TinyUSB's Espressif VID/PID defaults. Obtain an appropriate USB
  identity before commercial distribution; these are not unique product IDs.

## Implementation

`hid_dictation_core.c` is hardware-independent: ASCII mapping, whitespace filtering,
button debounce, an atomic session gate, and a retry-safe press/release transmitter.
`usb_hid.c` implements the boot-keyboard descriptors, LED/control-report callbacks,
USB lifecycle, bounded text queue, and button task. USB servicing priority 24 is
above the ASR worker; the keyboard task runs on core 0 above the inference caller.
Without that priority adjustment, ASR can starve the default low-priority USB task.

Audio blocks carry the capture session through segmentation and inference. Paused
samples still calibrate the energy-VAD noise floor but are not retained as pre-roll.
An arming transition drains 200 ms, exceeding the 160 ms I2S descriptor capacity.
Session checks at capture, segmentation, inference dequeue, submission, and every
key press discard old work even when BOOT is pressed during a long inference.

Enabling USB does not change model weights, quantization, beam search, or VAD
endpoint parameters. Serial-only mode uses no-op USB hooks. Recognition-performance
experiments remain disabled by default.

## Validation

```bash
bash tests/run_hid.sh
bash tests/run_host.sh
```

HID native tests cover all 95 printable ASCII characters, shifted punctuation,
Caps Lock, whitespace/control/Unicode/length handling, generation changes on
pause/reconnect/readiness, button bounce/startup holds, blocked report submission,
repeated keys, guaranteed scheduled releases, stale-text cancellation, and timer
wrap. The suite also runs with AddressSanitizer and UndefinedBehaviorSanitizer.

GitHub Actions builds the HID firmware with ESP-IDF 5.5, checks the resolved USB
settings, and retains application/bootloader/partition artifacts. Build success is
not an electrical enumeration, FreeRTOS timing, sleep/resume, or host-OS test.
Before regular use, test in a text editor on the actual board: repeated letters,
Caps Lock, pausing during inference/typing, rearming, unplug/replug, host sleep,
long utterances, and back-to-back phrases. Confirm no stale text appears.

## Primary references

- Espressif ESP-IDF 5.5 USB device documentation:
  https://docs.espressif.com/projects/esp-idf/en/v5.5/esp32s3/api-reference/peripherals/usb_device.html
- Espressif HID example:
  https://github.com/espressif/esp-idf/tree/v5.5/examples/peripherals/usb/device/tusb_hid
- TDK INMP441 specification and lifecycle:
  https://product.tdk.com/en/search/sw_piezo/mic/mems-mic/info?part_no=INMP441
