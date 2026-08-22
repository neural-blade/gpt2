CC ?= cc
BASE_CFLAGS = -Wall -Wextra -Wpedantic -std=c99
BASE_LDFLAGS = -lm

SRCS := $(wildcard src/*.c)
OBJS := $(SRCS:%.c=%.o)
ASMS := $(SRCS:%.c=%.s)

TARGET := gpt2

OPT_CFLAGS = -O3 -ffast-math
NATIVE_CFLAGS = -march=native
DEBUG_CFLAGS = -g -Og
OPT_LDFLAGS = -lmvec
ASM_FLAGS = -fverbose-asm
SAN_FLAGS = -fsanitize=address,undefined

.PHONY: all clean debug san asm

all: CFLAGS = $(BASE_CFLAGS) $(NATIVE_CFLAGS) $(OPT_CFLAGS)
all: LDFLAGS = $(BASE_LDFLAGS) $(OPT_LDFLAGS)
all: $(TARGET)

debug: CFLAGS = $(BASE_CFLAGS) $(DEBUG_CFLAGS)
debug: LDFLAGS = $(BASE_LDFLAGS)
debug: $(TARGET)

san: CFLAGS = $(BASE_CFLAGS) $(DEBUG_CFLAGS) $(SAN_FLAGS)
san: LDFLAGS = $(BASE_LDFLAGS) $(SAN_FLAGS)
san: $(TARGET)

asm: CFLAGS = $(BASE_CFLAGS) $(NATIVE_CFLAGS) $(OPT_CFLAGS) $(ASM_FLAGS)
asm: $(ASMS)

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) $^ -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.s: %.c
	$(CC) $(CFLAGS) -S $< -o $@

clean:
	rm -rf $(TARGET) $(OBJS) $(ASMS) *.out*
