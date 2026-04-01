CC = gcc
CFLAGS = -O2 -Wall
LDLIBS = -lOpenCL

TARGETS = findall_final findall_lmem findall_lmem_v2

.PHONY: all clean

all: $(TARGETS)

findall_final: findall_final.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

findall_lmem: findall_lmem.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

findall_lmem_v2: findall_lmem_v2.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f $(TARGETS)
