// Host regression tests: ownership, allocation failure, bounded decoding, segmentation,
// serial/parallel equivalence, and a reused workspace with poisoned backing storage.
#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "tasr_nemo.h"
#include "tasr_seg.h"
#include "tinyasr.h"
#include "tinyasr_lm.h"

static void *(*saved_alloc)(size_t, int);
static void (*saved_free)(void *);
static int alloc_calls, fail_at = -1, live;
static size_t live_bytes;
static struct { void *p; size_t n; } allocations[1024];
static void *checked_alloc(size_t n, int kind)
{
    (void)kind;
    const int call = alloc_calls++;
    if (call == fail_at) return NULL;
    void *p = NULL;
    assert(posix_memalign(&p, 16, n ? n : 16) == 0);
    memset(p, 0xa5, n);  // expose implicit dependence on allocator-wide zeroing
    for (int i = 0; i < 1024; i++) if (!allocations[i].p) {
        allocations[i].p = p; allocations[i].n = n; live++; live_bytes += n; return p;
    }
    abort();
}
static void checked_free(void *p)
{
    if (!p) return;
    for (int i = 0; i < 1024; i++) if (allocations[i].p == p) {
        live--; live_bytes -= allocations[i].n; allocations[i].p = NULL; free(p); return;
    }
    abort();  // double free or freeing a non-owned/phase-aliased pointer
}
static void check_hooks(void)
{
    saved_alloc = tasr_alloc; saved_free = tasr_free;
    tasr_alloc = checked_alloc; tasr_free = checked_free;
}
static void restore_hooks(void)
{
    assert(live == 0 && live_bytes == 0);
    tasr_alloc = saved_alloc; tasr_free = saved_free;
}
static uint8_t *read_model(const char *path, size_t *size)
{
    FILE *f = fopen(path, "rb"); assert(f);
    assert(fseek(f, 0, SEEK_END) == 0); long n = ftell(f); assert(n > 0); rewind(f);
    uint8_t *p = NULL; assert(posix_memalign((void **)&p, 16, (size_t)n + 16) == 0);
    assert(fread(p, 1, n, f) == (size_t)n); fclose(f); *size = n; return p;
}

typedef struct { tasr_job_fn fn; void *ctx; int b, e; } job_t;
static int parallel_calls;
static void *worker(void *p) { job_t *j = p; j->fn(j->ctx, j->b, j->e, 1); return NULL; }
static void parallel(tasr_job_fn fn, void *ctx, int n)
{
    if (n < 2) { fn(ctx, 0, n, 0); return; }
    parallel_calls++;
    job_t j = {fn, ctx, n / 2, n}; pthread_t thread;
    assert(pthread_create(&thread, NULL, worker, &j) == 0);
    fn(ctx, 0, n / 2, 0);
    assert(pthread_join(thread, NULL) == 0);
}
static void serial(tasr_job_fn fn, void *ctx, int n) { fn(ctx, 0, n, 0); }

static void test_workspace(const char *path)
{
    size_t bytes; uint8_t *blob = read_model(path, &bytes);
    tasr_nemo_t *m = tasr_nemo_load(blob, bytes); assert(m);
    check_hooks();
    alloc_calls = 0;
    tasr_nemo_workspace_t *ws = tasr_nemo_workspace_create(m, 6400, 1); assert(ws);
    const int count = alloc_calls;
    assert(tasr_nemo_workspace_bytes(ws) == live_bytes);
    printf("workspace %s: %zu bytes, %d initial allocations\n", path, live_bytes, count);
    int16_t pcm[6401];
    for (int i = 0; i < 6401; i++) pcm[i] = (int16_t)(6000 * sin(i * .11) + 900 * cos(i * .019));
    float a[11 * 1025], b[11 * 1025];
    char ta[4096], tb[4096]; int na, nb;
    alloc_calls = 0;
    tasr_parallel = serial;
    int rc = tasr_nemo_transcribe_with_workspace(ws, pcm, 6400, NULL, ta, sizeof(ta), a, 11, &na);
    assert(rc == 11 && na == 11 && alloc_calls == 0);
    tasr_parallel = parallel;
    rc = tasr_nemo_transcribe_with_workspace(ws, pcm, 6400, NULL, tb, sizeof(tb), b, 11, &nb);
    assert(rc == 11 && na == nb && !strcmp(ta, tb) && !memcmp(a, b, sizeof(a)) && alloc_calls == 0);
    assert(parallel_calls > 0);
    parallel_calls = 0; tasr_parallel_min_work = SIZE_MAX;
    rc = tasr_nemo_transcribe_with_workspace(ws, pcm, 6400, NULL, tb, sizeof(tb), b, 11, &nb);
    assert(rc == 11 && !memcmp(a, b, sizeof(a)) && parallel_calls == 0 && alloc_calls == 0);
    tasr_parallel_min_work = 0; tasr_parallel = serial;
    assert(tasr_nemo_transcribe_with_workspace(ws, pcm, 641, NULL, tb, sizeof(tb), b, 11, &nb) == 2);
    assert(tasr_nemo_transcribe_with_workspace(ws, pcm, 0, NULL, tb, sizeof(tb), b, 11, &nb) == 0 && nb == 0 && tb[0] == 0);
    assert(tasr_nemo_transcribe_with_workspace(ws, pcm, 6401, NULL, tb, sizeof(tb), b, 11, &nb) == -1 && nb == 0 && tb[0] == 0);
    assert(alloc_calls == 0);
    tasr_nemo_workspace_free(ws); assert(!live);
    for (int i = 0; i < count; i++) {
        fail_at = i; alloc_calls = 0;
        ws = tasr_nemo_workspace_create(m, 6400, 1);
        assert(!ws && live == 0 && live_bytes == 0);
    }
    fail_at = -1;
    assert(!tasr_nemo_workspace_create(NULL, 6400, 0));
    assert(!tasr_nemo_workspace_create(m, 10, 0));
    restore_hooks();
    tasr_nemo_free(m); free(blob);
    printf("PASS workspace reuse, strict equivalence, zero inference allocations, %d failure points\n", count);
}

static void test_decoder(void)
{
    check_hooks(); alloc_calls = 0;
    tasr_decoder_t *d = tasr_decoder_create_bounded(NULL, 8, 4, 6, .6f, 2.5f, 3); assert(d);
    int count = alloc_calls;
    float lp[8]; for (int i = 0; i < 8; i++) lp[i] = logf(1.f / 8);
    for (int i = 0; i < 3; i++) { tasr_decoder_step(d, lp); assert(!tasr_decoder_overflowed(d)); }
    tasr_decoder_step(d, lp); assert(tasr_decoder_overflowed(d));
    tasr_decoder_reset(d); assert(!tasr_decoder_overflowed(d));
    tasr_decoder_free(d);
    for (int i = 0; i < count; i++) {
        fail_at = i; alloc_calls = 0;
        assert(!tasr_decoder_create_bounded(NULL, 8, 4, 6, .6f, 2.5f, 3));
        assert(!live);
    }
    fail_at = -1;
    assert(!tasr_decoder_create(NULL, 8, 0, 6, .6f, 2.5f));
    assert(!tasr_decoder_create(NULL, 8, 4, 0, .6f, 2.5f));
    restore_hooks();
    puts("PASS bounded decoder and allocation failures");
}

static void test_segmenter(void)
{
    int16_t buf[32000], silence[320] = {0}, voice[320]; tasr_seg_t s;
    for (int i = 0; i < 320; i++) voice[i] = (i & 1) ? 5000 : -5000;
    tasr_seg_init(&s, buf, 32000);
    for (int i = 0; i < 20; i++) assert(!tasr_seg_feed(&s, silence, 320));
    for (int i = 0; i < 20; i++) assert(!tasr_seg_feed(&s, voice, 320));
    for (int i = 0; i < 40; i++) assert(!tasr_seg_feed(&s, silence, 320));
    assert(tasr_seg_feed(&s, silence, 320) == 22720);  // unchanged strict >40 endpoint
    assert(tasr_seg_audio_samples(&s, -1) == 22720);
    assert(tasr_seg_audio_samples(&s, 0) == 9600);
    assert(tasr_seg_audio_samples(&s, 3200) == 12800);
    assert(tasr_seg_audio_samples(&s, 1000000) == 22720);
    assert(s.n == 22720); // slicing does not mutate the stored segment
    tasr_seg_next(&s); assert(!s.n && !s.last_voiced);
    puts("PASS endpoint timing and opt-in tail slicing");
}

int main(void)
{
    test_segmenter(); test_decoder();
    test_workspace("models/nemo8.tnm"); test_workspace("models/nemo4.tnm");
    puts("PASS all native regression tests");
    return 0;
}
