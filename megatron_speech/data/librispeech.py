"""Reading LibriSpeech from the Hugging Face parquet release (``openslr/librispeech_asr``)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterator

import numpy as np

REPO_ID = "openslr/librispeech_asr"
SPLITS = {
    "train-clean-100": "all/train.clean.100",
    "train-clean-360": "all/train.clean.360",
    "train-other-500": "all/train.other.500",
    "dev-clean": "all/validation.clean",
    "dev-other": "all/validation.other",
    "test-clean": "all/test.clean",
    "test-other": "all/test.other",
}


def find_parquet_files(split: str) -> list[Path]:
    from huggingface_hub import snapshot_download

    pattern = f"{SPLITS[split]}/*.parquet"
    root = Path(snapshot_download(REPO_ID, repo_type="dataset", allow_patterns=[pattern]))
    files = sorted(root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no parquet files for {split} under {root}")
    return files


def iter_librispeech(split: str, limit: int | None = None) -> Iterator[tuple[str, np.ndarray, int, str]]:
    """Yield ``(utterance_id, waveform float32, sampling_rate, text)``."""
    import pyarrow.parquet as pq
    import soundfile as sf

    count = 0
    for path in find_parquet_files(split):
        pf = pq.ParquetFile(path)
        for group in range(pf.metadata.num_row_groups):
            for row in pf.read_row_group(group, columns=["id", "text", "audio"]).to_pylist():
                wav, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
                yield row["id"], wav, sr, row["text"]
                count += 1
                if limit is not None and count >= limit:
                    return
