# MoE expert cache on AMD — it runs on Vulkan (measured on RDNA4)

The llama.cpp MoE expert cache (GPU-resident LRU cache for host-offloaded MoE expert
weights) is labelled **CUDA-only** by every downstream fork. It is not. The upstream
implementation is backend-agnostic and **runs on Vulkan/RADV**, where it produces a
measured **1.8x decode speedup** on a Radeon RX 9070 XT (gfx1201).

This repo exists so a stranger can reproduce that in two commands.

## The claim, and the receipt

Environment: AMD RX 9070 XT (gfx1201, RADV), Ryzen 9 9950X3D, 32 GB RAM.
Model: a 30B-A3B MoE at Q4_K_M (18.5 GiB — **larger than the 16 GB card**, so the
experts live in host RAM; this is exactly the case the cache targets).
Build: llama.cpp + PR #27861 (`--moe-expert-cache`), Vulkan backend, single card.

Decode tokens/second, 128 generated tokens, prompt held identical (temp 0), n=3:

| config                                   | decode t/s (3 runs)      | mean |
|------------------------------------------|--------------------------|------|
| cache OFF (`--n-cpu-moe 99`)             | 30.97 / 30.59 / 30.90    | **30.8** |
| cache ON, 64 slots/layer                 | 56.47 / 55.31 / 56.22    | **56.0** |
| cache ON, 128 slots/layer                | 46.94 / 48.11 / 47.06    | 47.4 |

**64 slots/layer = 1.82x decode.** The win is real and reproducible within a couple
percent.

Why 128 slots is *slower* than 64: the cache allocates **48 layers x n_slots** slices of
the gate/up/down expert tensors. At 64 slots that is 8.5 GiB ("8500.8 MiB device memory"
in the log); at 128 slots it is **16.9 GiB — the entire card**. A cache that fills the
whole device starves the KV cache and the non-cached expert traffic, and the net falls
back to 47 t/s. The engine logs the allocation so you can see it:

```
I operator(): MoE expert cache enabled: 48 layers x 64 slots, 2 inserts/step, 8500.8 MiB device memory
```

## Reproduce

```bash
# 1. build llama.cpp with the patch applied
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git apply /path/to/patch/moe-expert-cache-vulkan.patch     # see patch/
cmake -B build -DGGML_VULKAN=ON -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j"$(nproc)"

# 2. measure (needs a MoE gguf that does NOT fit wholly in VRAM — that is the point)
./scripts/run_sweep.sh /path/to/model.gguf 0 128
```

`patch/moe-expert-cache-vulkan.patch` is PR #27861 rebased onto llama.cpp
`7049ff0cb` (2026-10-05). The only non-trivial hunk is `src/CMakeLists.txt`: upstream
refactored its source list into a `${LLAMA_CORE_SOURCES}` variable after the PR was
written, so the patch adds `llama-moecache.cpp` to that variable instead of to the
library's inline list. Every other file applies as-is.

## What this is, and is not

**Is:** a demonstration + a runnable patch that the expert cache works on a non-CUDA
backend, with numbers and the raw logs they came from (`results/`).

**Is not:** a claim the cache helps everyone. It only engages for single-token decode on
a layer whose experts are host-resident. If the model fits wholly on the device there is
nothing to cache and it is inert (and for our own box we found the bigger win is simply
*not offloading at all* — a model that fits wholly in the combined VRAM of two cards
needs no cache). The cache is for the case where the model does **not** fit.

## Raw logs

`results/` holds the unedited runs: `r_OFF_{1..3}`, `r_ON64_{1..3}`, `r_ON128_{1..3}`.
Each contains the load line (cache allocation size) and the `eval time` line the numbers
above are read from.

## Credit

Expert cache implementation: PR #27861 (csantiago78) on ggml-org/llama.cpp. This repo
adds no kernel code — it is a reproducibility report and a rebased patch for the AMD
/Vulkan path.

— Sophia Marie DeClue
