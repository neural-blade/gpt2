CC ?= cc
NVCC ?= nvcc
BASE_CFLAGS = -Wall -Wextra -Wpedantic -std=c99 -D_POSIX_C_SOURCE=199309L

CUDA_ARCH ?= sm_89
CUDA_BASE_CFLAGS = -std=c++17 -arch=$(CUDA_ARCH)
CUDA_LDFLAGS = -lcudart

SRCS := $(wildcard src/*.c)
OBJS := $(SRCS:%.c=%.o)
ASMS := $(SRCS:%.c=%.s)

CPU_SRCS := $(wildcard src/backend/*.c)
CPU_OBJS := $(CPU_SRCS:%.c=%.o)
CPU_ASMS := $(CPU_SRCS:%.c=%.s)

CUDA_SRCS := $(wildcard src/backend/*.cu)
CUDA_OBJS := $(CUDA_SRCS:%.cu=%.o)

OPT_CFLAGS = -O3 -ffast-math
NATIVE_CFLAGS = -march=native
DEBUG_CFLAGS = -g -Og
ASM_FLAGS = -fverbose-asm
SAN_FLAGS = -fsanitize=address,undefined

CUDA_DEBUG_CFLAGS = -g -G
CPU_LDFLAGS = -lm
CPU_OPT_LDFLAGS = -lmvec

TARGET := gpt2

.PHONY: all clean debug san asm vgrind cuda cuda-debug cuda-target

all: CFLAGS = $(BASE_CFLAGS) $(NATIVE_CFLAGS) $(OPT_CFLAGS)
all: LDFLAGS = $(CPU_LDFLAGS) $(CPU_OPT_LDFLAGS)
all: $(TARGET)

debug: CFLAGS = $(BASE_CFLAGS) $(DEBUG_CFLAGS)
debug: LDFLAGS = $(CPU_LDFLAGS)
debug: $(TARGET)

san: CFLAGS = $(BASE_CFLAGS) $(DEBUG_CFLAGS) $(SAN_FLAGS)
san: LDFLAGS = $(CPU_LDFLAGS) $(SAN_FLAGS)
san: $(TARGET)

asm: CFLAGS = $(BASE_CFLAGS) $(NATIVE_CFLAGS) $(OPT_CFLAGS) $(ASM_FLAGS)
asm: $(ASMS) $(CPU_ASMS)

vgrind: CFLAGS = $(BASE_CFLAGS) -g $(NATIVE_CFLAGS) $(OPT_CFLAGS)
vgrind: LDFLAGS = $(CPU_LDFLAGS) $(CPU_OPT_LDFLAGS)
vgrind: $(TARGET)

cuda: CFLAGS = $(BASE_CFLAGS) $(NATIVE_CFLAGS) $(OPT_CFLAGS) -DCUDA_BACKEND
cuda: CUDA_CFLAGS = $(CUDA_BASE_CFLAGS)
cuda: cuda-target

cuda-debug: CFLAGS = $(BASE_CFLAGS) $(DEBUG_CFLAGS)
cuda-debug: CUDA_CFLAGS = $(CUDA_BASE_CFLAGS) $(CUDA_DEBUG_CFLAGS)
cuda-debug: cuda-target

$(TARGET): $(OBJS) $(CPU_OBJS)
	$(CC) $(LDFLAGS) $^ -o $@

cuda-target: $(OBJS) $(CUDA_OBJS)
	$(CC) $(LDFLAGS) $(CUDA_LDFLAGS) $^ -o $(TARGET)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(CUDA_CFLAGS) -c $< -o $@

%.s: %.c
	$(CC) $(CFLAGS) -S $< -o $@

clean:
	rm -rf $(TARGET) $(OBJS) $(CUDA_OBJS) $(CPU_OBJS) $(ASMS)
