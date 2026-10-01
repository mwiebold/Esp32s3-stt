SRC = ../components/tinyasr/tinyasr.c ../components/tinyasr/kernels.c ../components/tinyasr/tinyasr_lm.c ../components/tinyasr/tasr_nemo.c ../components/tinyasr/tasr_seg.c
CFLAGS = -O2 -D_DEFAULT_SOURCE -std=c11 -ffp-contract=off -I../components/tinyasr/include -I../components/tinyasr -Wall -Wno-unused-function
all: libtinyasr.dylib tasr_cli seg_test
HDR = $(wildcard ../components/tinyasr/include/*.h ../components/tinyasr/*.h)
libtinyasr.dylib: $(SRC) $(HDR)
	$(CC) $(CFLAGS) -shared -fPIC -o $@ $(SRC) -lm
clean:
	rm -f libtinyasr.dylib tasr_cli seg_test

tasr_cli: tasr_cli.c $(SRC) $(HDR)
	$(CC) $(CFLAGS) -o $@ tasr_cli.c $(SRC) -lm

seg_test: seg_test.c $(SRC) $(HDR)
	$(CC) $(CFLAGS) -o $@ seg_test.c $(SRC) -lm
