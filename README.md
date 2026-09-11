GPT-2 inference engine. It loads a GPT-2 model in GGUF format, runs the forward
pass and generates text token by token.

The engine has two backends: a plain C implementation and a CUDA
implementation.

---

## Build

```
make           # CPU backend
make cuda      # CUDA backend
```

The default GPU architecture is `sm_89`. Override `CUDA_ARCH` to target a
different architecture.

## Download the model

```
hf download neural-blade/gpt2-fp32-gguf --local-dir models --include gpt2-small-fp32.gguf
```

## Run

```
./gpt2 --model <model.gguf> [options]
```

Options:

- `--model <file>`       model file (required)
- `--prompt-length <n>`  number of input tokens (default 10)
- `--max-tokens <n>`     maximum tokens to generate (default 100)
- `--show-gguf`          print GGUF metadata and exit
- `--show-stats`         print prefill/decode timings
- `--help`

Example:

```
./gpt2 --model models/gpt2-small-fp32.gguf --prompt-length 512 --max-tokens 128
```

## Project structure

```
src/
  gguf.c/.h                GGUF parsing
  model.c/.h               Model description and tensor loading
  transformer.c/.h         Forward pass and KV cache
  backend.h                Backend abstraction
  backend/backend_cpu.c    CPU backend
  backend/backend_cuda.cu  CUDA backend
  backend/kernels_cuda.cu  CUDA kernels
  main.c                   Command-line interface
```

## Algorithm Overview

The engine runs the GPT-2 forward pass:

1. **Embedding**: input tokens are mapped to token embeddings and combined
  with positional embeddings.

2. **Transformer blocks**, for each layer:
  - Layer normalization
  - Causal self-attention with cached Keys/Values
  - Residual connection
  - Feed-forward network: up projection, GELU, down projection
  - Residual connection

3. **Output**: final layer norm, projection onto the vocabulary through the
  token embedding matrix, greedy selection with argmax.

Generation has two phases:

- **Prefill**: the whole prompt is processed in one pass.
- **Decode**: one token at a time, printed and appended to the context.

## Optimizations

Throughput is measured on an RTX 4070 (Ada, sm_89) with gpt2-small
(always-batched prefill, greedy single-token decode).

| Step | Prefill (tok/s) | Decode (tok/s) |
|---|---|---|
| Initial CUDA port | 845.8 | 95.0 |
| Tiled GEMM + GEMV | 3610.7 | 200.2 |
| Packed add launches + shuffle LN | 4920.6 | 270.7 |

### v0 - Naive CUDA

One-to-one CUDA port of the CPU backend. The projections (gemm) dominate both
phases (88% of prefill, 69.7% of decode).

### v1 — Tiled GEMM + GEMV

- Replace the naive GEMM with a tiled implementation (16x16 tiles in shared
  memory) so all global loads are coalesced, and route vector-by-matrix
  multiplications to a dedicated `gemv_f32_kernel` when there is a single row
  (the decode phase and the final projection). This fixes the low occupancy and
  the uncoalesced traffic of the projections; prefill GEMM throughput goes from
  108 GFLOP/s to 1.2 TFLOP/s.
- `gemv_f32_kernel` is intrinsically memory bound, AI ~0.5 FLOP/byte, and already
  operates on the memory-bound slope of the roofline, hitting the bandwidth-limited
  performance ceiling of 135 GFLOP/s.

### v2 — Packed add launches + shuffle layer norm

- Packed add: all projection bias adds are merged into a single kernel launch,
  increasing the problem size enough to use the whole GPU.
- Layer-norm rewritten with warp-shuffle reductions and one block per token.
