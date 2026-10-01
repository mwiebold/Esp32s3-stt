# ESP32-S3 performance changes

Based on lokutor-ai/oido commit `d3de03f28364251651cc832f291197d0fab46d92`.
Code and model licenses are unchanged. Upstream history is retained in this repository.
The destination already existed, so this is an imported derivative, not a GitHub fork-network record.

## Default-path changes

- Whole-utterance firmware RTF uses the 64-bit microsecond timer. The wrapping
  32-bit cycle delta is explicitly a diagnostic; emulator scripts validate their
  QEMU calibration before applying a counter-to-instruction conversion.
- NeMo stage profiling uses 64-bit timestamps, can be reset, and reports exclusive
  stage totals. ESP32 units are microseconds; host units are nanoseconds.
- A reusable, capacity-bounded workspace is allocated before weight caching.
  Large scratch buffers share storage only across disjoint front-end, positional,
  encoder, and decoder phases. Hot buffers remain individually allocated so the
  existing internal-RAM placement policy still applies. Int8 models do not reserve
  int4 unpacking buffers. The legacy transcription API remains as an allocating wrapper.
- Reused inference does not allocate or clear the entire workspace. The depthwise
  convolution's zero halos are explicitly reset on every call. Capacity violations
  and decoder overflow return an error rather than a silently partial result.
- Feature normalization traverses frame-major storage contiguously, retaining the
  original per-channel accumulation order and default double-precision arithmetic.
- Layer-normalization/quantization and activation/quantization are fused by row;
  residual/final normalization, GLU, and QKV quantization use independent jobs.
  QKV workers own whole attention heads, including their transposed V output.
- Beam candidate scores are computed once after merging. Candidate scratch,
  state pools, and the bounded decoder's node/hash capacities follow configuration.
  At beam four and 501 frames, node/hash storage falls from 384 KiB to 48 KiB.
- Live NeMo audio uses separate capture and segmentation tasks. A bounded pool
  transfers ownership of complete utterances to inference without stopping
  segmentation. No free buffer means the newly completed utterance is dropped.
  Full capture queues drop complete frames; sequence gaps reset the partial segment.
  Overflow counters are reported outside the high-priority capture task.

## Configuration and tradeoffs

| Setting | Default | Meaning |
|---|---:|---|
| `CONFIG_TASR_MAX_UTTERANCE_SECONDS` | 20 | Actual workspace/decoder capacity; firmware range 1–20 s. |
| `CONFIG_TASR_UTTERANCE_BUFFERS` | 3 | Recording, queued, and inference ownership; range 2–4. |
| `CONFIG_TASR_INFERENCE_TAIL_MS` | -1 | Preserve the full endpointing tail. Nonnegative values opt into trimming. |
| `CONFIG_TASR_NEMO_FLOAT_STATS` | off | Opt into float rather than double feature statistics. |
| `CONFIG_TASR_PAR_MIN_WORK` | 0 | Preserve all parallel dispatches; tune the work threshold on hardware. |

The microphone queue holds two seconds of whole frames; it is not the backlog
for long inference. The utterance pool is that backlog. More utterance buffers
reduce available weight-cache memory and may increase inference time. No bounded
queue can absorb indefinite sustained overload.

The endpoint decision is unchanged: with 320-sample blocks, the strict `>40`
silence condition is 0.82 seconds. Tail trimming changes only the audio slice sent
to inference, not that decision. It changes utterance normalization and requires
recognition-accuracy testing. Float statistics likewise require WER validation.
Neither experiment is enabled by default. Global `-ffast-math` is rejected because
it invalidates the quantizer's rounding arithmetic.

The microphone's `endpoint-to-text` metric starts at the endpoint decision, not
at the last spoken phoneme. Add the endpointing delay for post-speech latency.
`queue` measures time from the captured endpoint block to inference start.

## Reusable API

```c
// The model and decoder must outlive each inference; one caller per workspace.
tasr_nemo_workspace_t *ws = tasr_nemo_workspace_create(model, 16000 * 20, decoder != NULL);
if (!ws) { /* insufficient memory */ }
// Allocate microphone buffers next, then spend remaining PSRAM on cached weights.
int frames = tasr_nemo_transcribe_with_workspace(ws, pcm, samples, decoder,
                                                text, sizeof(text), NULL, 0, NULL);
if (frames < 0) { /* invalid input, capacity exceeded, or decoder overflow */ }
// Reuse ws for subsequent utterances, then:
tasr_nemo_workspace_free(ws);
```

Workspace allocation and allocation-failure cleanup are explicit. No global
allocator replacement is used by the implementation. Existing applications may
continue using `tasr_nemo_transcribe`; they will not receive the persistent-workspace
allocation savings until they adopt the reusable API.

## Reproducible validation

```bash
make -C esp32/host
bash tests/run_host.sh
# Build a baseline from the pinned upstream revision in a separate worktree:
git worktree add --detach /tmp/oido-baseline d3de03f28364251651cc832f291197d0fab46d92
make -C /tmp/oido-baseline/esp32/host \
  CFLAGS='-O2 -D_DEFAULT_SOURCE -std=c11 -ffp-contract=off -I../components/tinyasr/include -I../components/tinyasr -Wall -Wno-unused-function'
python tests/compare_upstream.py /tmp/oido-baseline/esp32/host/libtinyasr.dylib
# Optional longer clips and repeated reuse after a maximum-length clip:
python tests/compare_upstream.py /tmp/oido-baseline/esp32/host/libtinyasr.dylib --full
```

Native tests cover poisoned backing storage, zero allocations during reused
inference, every workspace-allocation failure point, decoder bounds and allocation
failures, serial/two-worker equivalence, dispatch-threshold fallback, endpoint/tail
semantics, timer-wrap handling, and rejection of unsafe math flags. The same native
suite runs under AddressSanitizer and UndefinedBehaviorSanitizer.

The upstream comparator checks raw float32 logit bits and transcripts with int8
and int4 models, greedy decoding, plain beam search, and LM decoding. Its synthetic
signals test numerical equivalence and buffer boundaries; they are **not a WER
benchmark or representative microphone-quality evaluation**.

Before relying on realtime operation, measure physical ESP32-S3 hardware using
2/4/10/20-second clips and back-to-back live speech. Record RTF, post-speech latency,
queue depth/drops, internal versus PSRAM allocation, stack high-water marks, and
cold/warm timing. Validate the two experimental options on held-out speech and noise.
No silicon speedup or new word-error-rate result is claimed by this change set.

## Local validation record (2026-10-01)

The full native comparison suite passed **70** cases against the pinned upstream
engine: both int8 and int4, synthetic inputs from 320 samples through 20 seconds,
greedy, beam-four, and language-model decoding (10- and 20-second cases use greedy),
and reuse after long inputs. All compared logits and transcripts were bit-identical.
These synthetic inputs are regression fixtures, **not a word-error-rate corpus**.
Native serial/pthread parity, allocation-failure injection, poisoned-workspace
reuse, zero-allocation inference, ASan/UBSan/leak checks, segmenter checks, six timing
parser tests, and unsafe-fast-math rejection also passed. Hardware throughput,
FreeRTOS scheduling behavior, real-microphone accuracy, and experimental-option WER
remain unmeasured. See GitHub Actions for the separate ESP-IDF compile results.
