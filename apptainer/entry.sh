#!/bin/bash
# Entry point inside the container (called by apptainer/run.sh).
set -eo pipefail
cd "$MS_REPO_ROOT"
if [[ "$MS_USE_VENV" == 1 ]]; then
    if [[ ! -f .venv/bin/activate ]]; then
        echo "error: .venv not found; run 'apptainer/run.sh --no-venv bash apptainer/setup_venv.sh'" >&2
        exit 1
    fi
    source .venv/bin/activate
fi
# With --nv, libcuda.so is mounted here; Triton does not find it on its own.
export TRITON_LIBCUDA_PATH=/.singularity.d/libs
# Megatron requires this for tensor-parallel communication overlap.
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
# Packed speech batches change shape every step; expandable segments limit fragmentation.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec "$@"
