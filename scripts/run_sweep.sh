#!/usr/bin/env bash
# Reproduce the Vulkan MoE expert-cache measurement on llama.cpp + PR #27861.
# Usage: run_sweep.sh <model.gguf> [GPU_INDEX] [N_TOKENS]
set -euo pipefail
MODEL="${1:?path to a MoE gguf}"; GPU="${2:-0}"; N="${3:-128}"
BIN="${LLAMA_BIN:-./build/bin/llama-cli}"
export GGML_VK_VISIBLE_DEVICES="$GPU"
PROMPT="Explain in detail why the sky appears blue, covering Rayleigh scattering and the wavelength dependence of atmospheric scattering, then discuss why sunsets are red."
run() { local tag="$1"; shift
  "$BIN" -m "$MODEL" -ngl 99 --n-cpu-moe 99 "$@" -c 4096 -t 16 -n "$N" --temp 0 -st -v -p "$PROMPT" \
    | grep -a "eval time" | tail -1 | sed "s/^/$tag: /"; }
echo "--- cache OFF ---";  run OFF
echo "--- cache ON 64 ---"; run ON64  --moe-expert-cache 64
echo "--- cache ON 128 ---"; run ON128 --moe-expert-cache 128
