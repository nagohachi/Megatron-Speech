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

if [[ ! -d .venv ]]; then
    /usr/bin/python3 -m venv --system-site-packages .venv
fi
source .venv/bin/activate

# Pin the container's builds so the resolver never replaces them.
python - > .venv/container-constraints.txt <<'PY'
import importlib.metadata as md
for name in ["torch", "transformer_engine", "flash_attn", "triton", "numpy", "apex"]:
    try:
        print(f"{name}=={md.version(name)}")
    except md.PackageNotFoundError:
        pass
PY

# pip (not uv) because pip accepts the container's torch, installed outside the venv,
# as satisfying "torch" requirements of other packages.
python -m pip install --quiet --upgrade pip
python -m pip install --constraint .venv/container-constraints.txt -e ".[dev,logging]"

# Megatron-LM is used from the submodule as is (megatron.core, megatron.training and
# the top-level helpers such as megatron.rank_log_setup).
SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$MLM" > "$SITE/megatron_lm.pth"

# The container's /usr/local/bin/torchrun runs the system Python, which does not see the venv.
cat > .venv/bin/torchrun <<'SH'
#!/bin/bash
exec python -m torch.distributed.run "$@"
SH
chmod +x .venv/bin/torchrun

python - <<'PY'
import torch, transformer_engine, transformers, lhotse
import megatron.core, megatron.training
print("torch", torch.__version__)
print("transformer_engine", transformer_engine.__version__)
print("megatron.core", megatron.core.__version__)
print("transformers", transformers.__version__)
print("lhotse", lhotse.__version__)
PY
