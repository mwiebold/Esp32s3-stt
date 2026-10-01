// SPDX-License-Identifier: GPL-3.0-only
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include "hid_dictation_core.h"

static void test_mapping(void)
{
    uint8_t mod, key;
    for (int c = 0; c < 256; c++) {
        bool ok = hid_us_key((unsigned char)c, false, &mod, &key);
        assert(ok == (c >= 32 && c <= 126));
        if (ok) { assert(key >= 4 && key <= 56); assert(key != 40 && key != 41 && key != 42 && key != 43); assert(mod == 0 || mod == 2); }
    }
    assert(hid_us_key('a', false, &mod, &key) && key == 4 && mod == 0);
    assert(hid_us_key('a', true, &mod, &key) && key == 4 && mod == 2);
    assert(hid_us_key('A', true, &mod, &key) && key == 4 && mod == 0);
    assert(hid_us_key('A', false, &mod, &key) && key == 4 && mod == 2);
    assert(hid_us_key('!', true, &mod, &key) && key == 30 && mod == 2);
    assert(hid_us_key('0', false, &mod, &key) && key == 39 && mod == 0);
    assert(hid_us_key(')', false, &mod, &key) && key == 39 && mod == 2);
    const char *plain="-=[]\\;'`,./", *shifted="_+{}|:\"~<>?";
    const uint8_t keys[]={45,46,47,48,49,51,52,53,54,55,56};
    for (size_t i=0; i<strlen(plain); i++) {
        assert(hid_us_key(plain[i], false, &mod, &key) && mod == 0 && key == keys[i]);
        assert(hid_us_key(shifted[i], false, &mod, &key) && mod == 2 && key == keys[i]);
    }
}
static void test_text(void)
{
    char out[64]; size_t n;
    assert(hid_text_prepare("  hello\tthere\r\nworld  ",out,sizeof(out),&n));
    assert(!strcmp(out,"hello there world ") && n==18);
    assert(hid_text_prepare("don't type Enter!",out,sizeof(out),&n));
    assert(!strcmp(out,"don't type Enter! "));
    assert(!hid_text_prepare("",out,sizeof(out),&n) && n==0);
    assert(!hid_text_prepare(" \n\t",out,sizeof(out),&n));
    assert(!hid_text_prepare("a\033b",out,sizeof(out),&n));
    assert(!hid_text_prepare("caf\xc3\xa9",out,sizeof(out),&n));
    assert(!hid_text_prepare("a\177b",out,sizeof(out),&n));
    assert(hid_text_prepare("a",out,3,&n) && n==2 && !strcmp(out,"a "));
    assert(!hid_text_prepare("ab",out,3,&n));
    char too_long[64]; memset(too_long,'a',sizeof(too_long));
    assert(!hid_text_prepare(too_long,out,sizeof(out),&n));
    assert(!hid_text_prepare(NULL,out,sizeof(out),&n));
}
static void test_gate(void)
{
    hid_gate_t g; hid_gate_init(&g);
    assert(!hid_gate_session(&g) && !hid_gate_toggle(&g));
    hid_gate_link(&g,true); assert(!hid_gate_toggle(&g)); // model not loaded
    hid_gate_ready(&g,true); assert(!hid_gate_session(&g));
    assert(hid_gate_toggle(&g)); uint32_t a=hid_gate_session(&g); assert(a);
    assert(hid_gate_toggle(&g) && !hid_gate_session(&g));
    assert(hid_gate_toggle(&g)); uint32_t b=hid_gate_session(&g); assert(b && a!=b);
    hid_gate_link(&g,false); assert(!hid_gate_session(&g) && !hid_gate_toggle(&g));
    hid_gate_link(&g,true); assert(!hid_gate_session(&g)); // resume never rearms
    assert(hid_gate_toggle(&g)); assert(hid_gate_session(&g)!=b);
    hid_gate_pause(&g); assert(!hid_gate_session(&g));
    hid_gate_ready(&g,false); assert(!hid_gate_toggle(&g));
    // Generation wrap remains well-defined; an armed session is never zero.
    atomic_store(&g.state, UINT_MAX - 1u);
    assert(hid_gate_toggle(&g)); assert(hid_gate_session(&g) == 7u);
}
static void test_button(void)
{
    hid_button_t b={0};
    assert(!hid_button_update(&b,true,0));
    assert(!hid_button_update(&b,true,1000)); // held at startup
    assert(!hid_button_update(&b,false,1010));
    assert(!hid_button_update(&b,false,1040));
    assert(!hid_button_update(&b,true,1050));
    assert(!hid_button_update(&b,false,1060)); // bounce
    assert(!hid_button_update(&b,true,1070));
    assert(!hid_button_update(&b,true,1099));
    assert(hid_button_update(&b,true,1100));
    assert(!hid_button_update(&b,true,3000)); // no repeats
    assert(!hid_button_update(&b,false,3010));
    assert(!hid_button_update(&b,false,3040));
    assert(!hid_button_update(&b,true,3050));
    assert(hid_button_update(&b,true,3080));
    b=(hid_button_t){0};
    assert(!hid_button_update(&b,false,UINT_MAX-40));
    assert(!hid_button_update(&b,false,UINT_MAX-10));
    assert(!hid_button_update(&b,true,UINT_MAX-5));
    assert(hid_button_update(&b,true,25));
}
typedef struct { hid_key_report_t reports[64]; int n; bool ready; } host_t;
static bool send_mock(void *ctx, const hid_key_report_t *r)
{
    host_t *h=ctx;
    if (!h->ready) return false;
    assert(h->n < 64); h->reports[h->n++]=*r;
    return true;
}
static void test_transport(void)
{
    host_t h={.ready=true}; hid_tx_t tx;
    hid_tx_init(&tx,10);
    assert(!hid_tx_begin(&tx,"aa ",3,7)); // first send neutral
    assert(hid_tx_step(&tx,0,false,0,send_mock,&h));
    assert(hid_tx_begin(&tx,"aa ",3,7));
    assert(!hid_tx_step(&tx,7,false,9,send_mock,&h));
    h.ready=false; assert(!hid_tx_step(&tx,7,false,10,send_mock,&h)); assert(tx.position==0);
    h.ready=true;
    for (uint32_t t=10;t<=60;t+=10) assert(hid_tx_step(&tx,7,false,t,send_mock,&h));
    assert(!hid_tx_busy(&tx) && h.n==7);
    const uint8_t expect[]={0,4,0,4,0,44,0};
    for (int i=0;i<7;i++) assert(h.reports[i].keys[0]==expect[i]);
    assert(hid_tx_begin(&tx,"abc",3,15));
    assert(hid_tx_step(&tx,15,true,70,send_mock,&h)); assert(h.reports[7].modifier==2);
    h.ready=false; assert(!hid_tx_step(&tx,0,false,80,send_mock,&h));
    assert(!tx.active && tx.release_pending); // canceled, release waits for host
    h.ready=true; assert(hid_tx_step(&tx,23,false,90,send_mock,&h));
    assert(h.reports[8].keys[0]==0 && h.reports[8].modifier==0 && !hid_tx_busy(&tx));
    assert(!hid_tx_step(&tx,23,false,100,send_mock,&h)); // never continues old text
    assert(hid_tx_begin(&tx,"stale",5,15));
    assert(!hid_tx_step(&tx,23,false,110,send_mock,&h)); assert(!hid_tx_busy(&tx));
    hid_tx_cancel(&tx);
    assert(hid_tx_step(&tx,0,false,120,send_mock,&h));
    assert(hid_tx_begin(&tx,"a",1,31));
    assert(hid_tx_step(&tx,31,false,UINT_MAX-5,send_mock,&h));
    assert(!hid_tx_step(&tx,31,false,3,send_mock,&h));
    assert(hid_tx_step(&tx,31,false,4,send_mock,&h));
}
int main(void)
{
    test_mapping(); test_text(); test_gate(); test_button(); test_transport();
    puts("PASS: all 95 ASCII keys, Caps Lock, text filtering, session invalidation, debounce, report backpressure/releases, timer wrap");
}
