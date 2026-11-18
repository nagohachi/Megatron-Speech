#!/bin/bash
# Run a command inside the NGC PyTorch container with the repository venv active.
#
#   apptainer/run.sh python -m pytest tests
#   CUDA_VISIBLE_DEVICES=0,1 apptainer/run.sh torchrun --nproc-per-node 2 \
#       -m megatron_speech.training.train --config configs/qwen3_asr/finetune_librispeech.yaml
#   apptainer/run.sh --no-venv bash apptainer/setup_venv.sh   # build .venv
#
# The command runs with the repository root as the working directory.
#
# Environment variables (put host-specific ones in apptainer/env.local.sh, not tracked by git):
#   MEGATRON_SPEECH_SIF    container image (default: ~/containers/pytorch_26.05.sif)
#   MEGATRON_SPEECH_BIND   extra bind paths for `apptainer exec --bind`, comma separated
#   MEGATRON_SPEECH_RUNTIME  apptainer or singularity binary (default: whichever is installed)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$REPO_ROOT/apptainer/env.local.sh" ]]; then
    source "$REPO_ROOT/apptainer/env.local.sh"
fi
SIF="${MEGATRON_SPEECH_SIF:-$HOME/containers/pytorch_26.05.sif}"

USE_VENV=1
if [[ "${1:-}" == "--no-venv" ]]; then
    USE_VENV=0
    shift
fi

# Bind the file systems that hold the repository and the Hugging Face cache. Older
# Singularity releases do not mount them on their own.
BINDS=()
for path in "$REPO_ROOT" "${HF_HOME:-}"; do
    if [[ -n "$path" && -e "$path" ]]; then
        mount_point="$(df --output=target "$path" | tail -1)"
        if [[ "$mount_point" != "/" && "$mount_point" != "$HOME"* ]]; then
            BINDS+=("$mount_point")
        fi
    fi
done
if [[ -n "${MEGATRON_SPEECH_BIND:-}" ]]; then
    BINDS+=("$MEGATRON_SPEECH_BIND")
fi
BIND_ARGS=()
if [[ ${#BINDS[@]} -gt 0 ]]; then
    BIND_ARGS=(--bind "$(printf '%s\n' "${BINDS[@]}" | sort -u | paste -sd,)")
fi

# Some servers only have SingularityCE (same CLI and SIF format); fall back to it.
RUNTIME="${MEGATRON_SPEECH_RUNTIME:-$(type -P apptainer || type -P singularity || true)}"
if [[ -z "$RUNTIME" ]]; then
    echo "error: neither apptainer nor singularity is installed on $(hostname)" >&2
    exit 1
fi

exec "$RUNTIME" exec --nv "${BIND_ARGS[@]}" \
    --env "MS_REPO_ROOT=$REPO_ROOT,MS_USE_VENV=$USE_VENV" \
    "$SIF" bash "$REPO_ROOT/apptainer/entry.sh" "$@"
