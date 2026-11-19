#!/bin/bash
# Build .venv on top of the container's Python. torch, Transformer Engine and flash-attn
# stay the container's builds; the venv only adds packages the container lacks.
#
#   apptainer/run.sh --no-venv bash apptainer/setup_venv.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO_ROOT="$(pwd)"
MLM="$REPO_ROOT/third_party/Megatron-LM"

if [[ ! -f "$MLM/pyproject.toml" ]]; then
    git submodule update --init third_party/Megatron-LM
fi

# Apply patches/*.patch in file-name order; skip the ones already applied.
shopt -s nullglob
for patch in patches/*.patch; do
    if git -C "$MLM" apply --reverse --check "$REPO_ROOT/$patch" 2>/dev/null; then
        echo "patch already applied: $patch"
    else
        git -C "$MLM" apply "$REPO_ROOT/$patch"
        echo "patch applied: $patch"
    fi
done

# uv is usually installed in ~/.local/bin, which is not on PATH inside the container.
UV="$(type -P uv || echo "$HOME/.local/bin/uv")"
if [[ ! -x "$UV" ]]; then
    echo "error: uv not found; install it with 'curl -LsSf https://astral.sh/uv/install.sh | sh'" >&2
    exit 1
fi

if [[ ! -d .venv ]]; then
    "$UV" venv --system-site-packages --python /usr/bin/python3 .venv
fi
source .venv/bin/activate

# [tool.uv] exclude-dependencies in pyproject.toml keeps the container's torch and friends.
"$UV" pip install -e ".[dev,logging]"

# Megatron-LM is used from the submodule as is (megatron.core, megatron.training and
# the top-level helpers such as megatron.rank_log_setup).
SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$MLM" > "$SITE/megatron_lm.pth"

# Megatron-Bridge supplies the rules that convert weights between Hugging Face and Megatron.
# --no-deps: its dependencies include the megatron-core package, a second megatron.core.
MEGATRON_BRIDGE_COMMIT=16595bbdc5242a8d27505b19f3ea32e8ae2ed428
"$UV" pip install --no-deps \
    "megatron-bridge @ https://github.com/NVIDIA-NeMo/Megatron-Bridge/archive/$MEGATRON_BRIDGE_COMMIT.tar.gz"

python - <<'PY'
from importlib.metadata import version
import torch, transformer_engine, transformers, lhotse
import megatron.core, megatron.training
import megatron.bridge.models.conversion.param_mapping
print("torch", torch.__version__)
print("transformer_engine", transformer_engine.__version__)
print("megatron.core", megatron.core.__version__)
print("megatron.bridge", version("megatron-bridge"))
print("transformers", transformers.__version__)
print("lhotse", lhotse.__version__)
PY
