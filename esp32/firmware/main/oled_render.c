// Portable SSD1306 128x64 frame renderer (no ESP-IDF dependencies; also built on the host by oled_preview).
// Row 0: inverted status line. Rows 1-7: transcript history, word-wrapped, newest at the bottom, each utterance
// starting with "> ".
#include "oled.h"
#include <string.h>
#include "font5x7.h"

#define COLS 21  // 6-pixel cells on 128 px
#define ROWS 8   // 8-pixel pages on 64 px

static void put_line(uint8_t *fb, int row, const char *s, int n, int invert)
{
    uint8_t *p = fb + row * OLED_W;
    memset(p, invert ? 0xff : 0x00, OLED_W);
    for (int i = 0; i < n && i < COLS; i++) {
        unsigned char ch = (unsigned char)s[i];
        const uint8_t *g = (ch >= 32 && ch < 127) ? font5x7[ch - 32] : font5x7[0];
        for (int c = 0; c < 5; c++) p[1 + i * 6 + c] = invert ? (uint8_t)~g[c] : g[c];
    }
}

// greedy word wrap of one utterance into lines of <= COLS chars; returns number of lines written (<= max)
static int wrap(const char *t, char lines[][COLS + 1], int max)
{
    int n = 0, len = 0;
    char cur[COLS + 1];
    strcpy(cur, "> ");
    len = 2;
    while (*t && n < max) {
        while (*t == ' ') t++;
        if (!*t) break;
        int w = 0;
        while (t[w] && t[w] != ' ') w++;
        if (len + (len > 2 ? 1 : 0) + w > COLS && len > 2) {  // flush the current line
            memcpy(lines[n], cur, len); lines[n][len] = 0; n++;
            strcpy(cur, "  ");
            len = 2;
            if (n >= max) break;
        }
        if (len > 2) cur[len++] = ' ';
        for (int i = 0; i < w; i++) {  // words longer than a line are split
            if (len == COLS) { memcpy(lines[n], cur, len); lines[n][len] = 0; if (++n >= max) return n; strcpy(cur, "  "); len = 2; }
            cur[len++] = t[i];
        }
        t += w;
    }
    if (n < max && len > 2) { memcpy(lines[n], cur, len); lines[n][len] = 0; n++; }
    return n;
}

void oled_render(uint8_t *fb, const char *status, const char *hist)
{
    static char all[64][COLS + 1];
    int n = 0;
    const char *u = hist;
    while (u && *u && n < 64) {  // utterances are separated by '\n'
        const char *e = strchr(u, '\n');
        char buf[512];
        int l = e ? (int)(e - u) : (int)strlen(u);
        if (l > (int)sizeof(buf) - 1) l = sizeof(buf) - 1;
        memcpy(buf, u, l);
        buf[l] = 0;
        n += wrap(buf, all + n, 64 - n);
        u = e ? e + 1 : NULL;
    }
    put_line(fb, 0, status, (int)strlen(status), 1);
    const int first = n > ROWS - 1 ? n - (ROWS - 1) : 0;
    for (int r = 1; r < ROWS; r++) {
        const int i = first + r - 1;
        if (i < n) put_line(fb, r, all[i], (int)strlen(all[i]), 0);
        else put_line(fb, r, "", 0, 0);
    }
}
