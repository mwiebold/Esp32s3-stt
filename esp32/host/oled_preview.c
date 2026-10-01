// oled_preview: render the SSD1306 transcript screen on the host (same renderer as the firmware) to a PGM image.
// usage: oled_preview out.pgm "status" "utterance 1" ["utterance 2" ...]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "oled.h"

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s out.pgm status [utterance ...]\n", argv[0]); return 1; }
    char hist[4096] = "";
    for (int i = 3; i < argc; i++) {
        if (hist[0]) strcat(hist, "\n");
        strncat(hist, argv[i], sizeof(hist) - strlen(hist) - 1);
    }
    static uint8_t fb[OLED_W * OLED_H / 8];
    oled_render(fb, argv[2], hist);
    const int S = 4;  // scale up for viewing
    FILE *f = fopen(argv[1], "wb");
    fprintf(f, "P5 %d %d 255\n", OLED_W * S, OLED_H * S);
    for (int y = 0; y < OLED_H * S; y++)
        for (int x = 0; x < OLED_W * S; x++) {
            const int px = x / S, py = y / S;
            fputc((fb[(py / 8) * OLED_W + px] >> (py % 8)) & 1 ? 230 : 20, f);
        }
    fclose(f);
    return 0;
}
