#!/usr/bin/env python3
"""Compare strict-math host outputs with the pinned upstream shared library.

Usage: python tests/compare_upstream.py BASELINE_LIB [--full]
Synthetic signals exercise arithmetic and padding, not recognition accuracy/WER.
"""
import argparse
import ctypes as C
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]


def library(path, workspace=False):
    lib = C.CDLL(str(Path(path).resolve()))
    specs = {
        'tasr_nemo_load': (C.c_void_p, [C.c_void_p, C.c_size_t]),
        'tasr_nemo_free': (None, [C.c_void_p]),
        'tasr_nemo_transcribe': (C.c_int, [C.c_void_p,C.c_void_p,C.c_int,C.c_void_p,C.c_void_p,C.c_int,C.c_void_p,C.c_int,C.c_void_p]),
        'tasr_decoder_create': (C.c_void_p, [C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_float,C.c_float]),
        'tasr_decoder_free': (None, [C.c_void_p]),
        'tasr_lm_load': (C.c_void_p, [C.c_void_p,C.c_size_t]),
        'tasr_lm_free': (None, [C.c_void_p]),
    }
    if workspace:
        specs.update({
            'tasr_nemo_workspace_create': (C.c_void_p,[C.c_void_p,C.c_int,C.c_int]),
            'tasr_nemo_workspace_free': (None,[C.c_void_p]),
            'tasr_nemo_workspace_bytes': (C.c_size_t,[C.c_void_p]),
            'tasr_nemo_transcribe_with_workspace': (C.c_int,specs['tasr_nemo_transcribe'][1]),
            'tasr_decoder_create_bounded': (C.c_void_p,specs['tasr_decoder_create'][1]+[C.c_int]),
        })
    for name, (ret,args) in specs.items():
        getattr(lib,name).restype=ret;getattr(lib,name).argtypes=args
    return lib


def aligned_file(path):
    data=Path(path).read_bytes()
    arr=np.empty(len(data)+16,dtype=np.uint8)
    view=arr[(-arr.ctypes.data)%16:][:len(data)]
    view[:]=np.frombuffer(data,dtype=np.uint8)
    return arr,view


def transcribe(fn, ptr, pcm, dec):
    text=C.create_string_buffer(8192);n=C.c_int()
    logits=np.empty((len(pcm)//640+8,1025),dtype=np.float32)
    rc=fn(ptr,pcm.ctypes.data,len(pcm),dec,text,len(text),logits.ctypes.data,len(logits),C.byref(n))
    assert rc>=0, f'transcription returned {rc}'
    assert rc==n.value
    return text.value,logits[:n.value].copy()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('baseline');ap.add_argument('--full',action='store_true');a=ap.parse_args()
    base=library(a.baseline);new=library(ROOT/'esp32/host/libtinyasr.dylib',True)
    lengths=[320,639,640,641,1600,16000,32160]
    if a.full:lengths += [40960, 64000, 160000, 320000, 32160, 640]  # capacity and reuse after a long utterance
    rng=np.random.default_rng(20261001)
    lmbuf,lmarr=aligned_file(ROOT/'models/nemo_lm.tlm')
    lms=[lib.tasr_lm_load(lmarr.ctypes.data,lmarr.size) for lib in (base,new)]
    assert all(lms)
    cases=0
    try:
        for model in ('nemo8.tnm','nemo4.tnm'):
            buf,arr=aligned_file(ROOT/'models'/model)
            ms=[lib.tasr_nemo_load(arr.ctypes.data,arr.size) for lib in (base,new)]
            assert all(ms)
            ws=new.tasr_nemo_workspace_create(ms[1],max(lengths),1);assert ws
            print(model,'workspace bytes',new.tasr_nemo_workspace_bytes(ws),flush=True)
            try:
                for ns in lengths:
                    t=np.arange(ns,dtype=np.float64)/16000
                    signal=(1500*np.sin(2*np.pi*237*t)+650*np.sin(2*np.pi*523*t)+rng.normal(0,100,ns)).astype(np.int16)
                    for mode in ('greedy','beam','lm'):
                        if ns>64000 and mode!='greedy':continue
                        decs=[None,None]
                        if mode!='greedy':
                            decs[0]=base.tasr_decoder_create(lms[0] if mode=='lm' else None,1025,4,6,.6,2.5)
                            decs[1]=new.tasr_decoder_create_bounded(lms[1] if mode=='lm' else None,1025,4,6,.6,2.5,ns//640+1)
                            assert all(decs)
                        try:
                            expected=transcribe(base.tasr_nemo_transcribe,ms[0],signal,decs[0])
                            actual=transcribe(new.tasr_nemo_transcribe_with_workspace,ws,signal,decs[1])
                            assert expected[0]==actual[0],(model,ns,mode,expected[0],actual[0])
                            assert np.array_equal(expected[1].view('u4'),actual[1].view('u4')),(model,ns,mode,float(np.max(np.abs(expected[1]-actual[1]))))
                            cases+=1
                            print('PASS',model,ns,mode,'bit-exact logits and transcript',flush=True)
                        finally:
                            for lib,dec in zip((base,new),decs):
                                if dec:lib.tasr_decoder_free(dec)
            finally:
                new.tasr_nemo_workspace_free(ws)
                for lib,m in zip((base,new),ms):lib.tasr_nemo_free(m)
    finally:
        for lib,lm in zip((base,new),lms):lib.tasr_lm_free(lm)
    print(f'PASS: {cases} upstream comparisons',flush=True)


if __name__=='__main__':main()
