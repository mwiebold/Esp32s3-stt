// seg_test: run a long recording through the firmware's live-mic path on the host: 20 ms blocks -> AGC/VAD utterance
// segmenter (tasr_seg.c, same code as the mic task) -> NeMo engine (+ optional LM). Prints one line per utterance.
// usage: seg_test model.tnm [--lm lm.tlm] long.wav
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "tasr_nemo.h"
#include "tasr_seg.h"
#include "tinyasr_lm.h"

static uint8_t *read_file(const char *path, size_t *n)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = NULL;
    if (posix_memalign((void **)&buf, 16, sz + 16)) { fclose(f); return NULL; }
    *n = fread(buf, 1, sz, f);
    fclose(f);
    return buf;
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s model.tnm [--lm lm.tlm] long.wav\n", argv[0]); return 1; }
    const char *lm_path = NULL, *wav = argv[argc - 1];
    for (int i = 2; i < argc - 1; i++)
        if (!strcmp(argv[i], "--lm") && i + 1 < argc - 1) lm_path = argv[++i];
    size_t msz, wsz;
    uint8_t *mb = read_file(argv[1], &msz);
    tasr_nemo_t *m = mb ? tasr_nemo_load(mb, msz) : NULL;
    if (!m) { fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
    tasr_decoder_t *dec = NULL;
    if (lm_path) {
        size_t lsz;
        uint8_t *lb = read_file(lm_path, &lsz);
        tasr_lm_t *lm = lb ? tasr_lm_load(lb, lsz) : NULL;
        if (!lm) { fprintf(stderr, "cannot load LM %s\n", lm_path); return 1; }
        dec = tasr_decoder_create(lm, 1025, 4, 6, 0.3f, 0.5f);
    }
    uint8_t *wb = read_file(wav, &wsz);
    if (!wb || wsz < 44) { fprintf(stderr, "cannot read %s\n", wav); return 1; }
    const int16_t *pcm = (const int16_t *)(wb + 44);  // canonical 44-byte header, 16 kHz mono PCM16
    const int ns = (int)((wsz - 44) / 2);
    const int cap = 16000 * 20;
    int16_t *utt = malloc(sizeof(int16_t) * cap);
    static char text[16384];
    tasr_seg_t seg;
    tasr_seg_init(&seg, utt, cap);
    int k = 0;
    for (int i = 0; i + 320 <= ns; i += 320) {
        const int n = tasr_seg_feed(&seg, pcm + i, 320);
        if (!n) continue;
        tasr_nemo_transcribe(m, utt, n, dec, text, sizeof(text), NULL, 0, NULL);
        printf("UTT %d end %.2fs len %.2fs\t%s\n", k++, (i + 320) / 16000.0, n / 16000.0, text);
        tasr_seg_next(&seg);
    }
    if (seg.speech && seg.n) {  // flush at end of file
        tasr_nemo_transcribe(m, utt, seg.n, dec, text, sizeof(text), NULL, 0, NULL);
        printf("UTT %d end %.2fs len %.2fs\t%s\n", k++, ns / 16000.0, seg.n / 16000.0, text);
    }
    return 0;
}
