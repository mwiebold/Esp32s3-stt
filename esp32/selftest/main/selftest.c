#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "esp_cpu.h"
#include "kernels.h"

static int8_t tile[512 * 16] __attribute__((aligned(16)));
static int8_t x[8 * 512] __attribute__((aligned(16)));
static int32_t out[8 * 16], ref[8 * 16];

void app_main(void)
{
    srand(1);
    int kps[] = {16, 48, 176, 352, 512};
    int bad_total = 0;
    for (int ci = 0; ci < 5; ci++) {
        int kp = kps[ci], T = 5;
        for (int i = 0; i < kp * 16; i++) tile[i] = (int8_t)((rand() % 15) - 7);
        for (int i = 0; i < T * kp; i++) x[i] = (int8_t)((rand() % 255) - 127);
        if (ci == 4) for (int i = 0; i < T * kp; i++) x[i] = 127, tile[i % (kp*16)] = 7;  // worst case
        if (ci == 4) for (int i = 0; i < kp * 16; i++) tile[i] = 7;
        for (int t = 0; t < T; t++)
            for (int j = 0; j < 16; j++) {
                int32_t a = 0;
                for (int k = 0; k < kp; k++) a += (int32_t)x[t * kp + k] * tile[k * 16 + j];
                ref[t * 16 + j] = a;
            }
        uint32_t c0 = esp_cpu_get_cycle_count();
        tasr_gemm_blk16(tile, kp, x, kp, T, out);
        uint32_t dc = esp_cpu_get_cycle_count() - c0;
        int bad = 0;
        for (int i = 0; i < T * 16; i++) if (out[i] != ref[i]) bad++;
        printf("kp %d: %d/%d mismatches (ccount %u)  out0 %ld ref0 %ld out17 %ld ref17 %ld\n", kp, bad, T * 16,
               (unsigned)dc, (long)out[0], (long)ref[0], (long)out[17], (long)ref[17]);
        bad_total += bad;
        if (ci == 1) {
            printf("OUT:"); for (int j = 0; j < 16; j++) printf(" %ld", (long)out[j]); printf("\n");
            printf("REF:"); for (int j = 0; j < 16; j++) printf(" %ld", (long)ref[j]); printf("\n");
        }
    }
    // int4 unpack check (blocked layout: byte j = lo w[k][j] | hi w[k+1][j])
    static uint8_t packed[64] __attribute__((aligned(16)));
    static int8_t un[128] __attribute__((aligned(16)));
    for (int i = 0; i < 64; i++) packed[i] = (uint8_t)(rand() & 0xff);
    tasr_unpack_s4(packed, un, 128);
    int ub = 0;
    for (int b = 0; b < 4; b++)
        for (int j = 0; j < 16; j++) {
            if (un[b * 32 + j] != (int8_t)((packed[b * 16 + j] & 15) - 8)) ub++;
            if (un[b * 32 + 16 + j] != (int8_t)((packed[b * 16 + j] >> 4) - 8)) ub++;
        }
    printf("unpack mismatches %d\n", ub);
    printf(bad_total + ub ? "SELFTEST FAIL\n" : "SELFTEST PASS\n");
    printf("DONE\n");
}
