import os

# Equivalence tests compare against HF in fp32; TF32 GEMMs (enabled by default in cuBLAS/cuDNN
# inside NGC containers) would add ~1e-3 relative error. Must be set before CUDA initializes.
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")

import pytest  # noqa: E402


@pytest.fixture(autouse=True, scope="session")
def _teardown_distributed():
    yield
    try:
        from megatron_speech import compat

        compat.destroy_model_parallel_for_tests()
    except Exception:
        pass
