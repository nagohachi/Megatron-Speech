"""Convert LibriSpeech (HF parquet release) into lhotse Shar shards.

    apptainer/run.sh python scripts/prepare_librispeech.py \
        --splits train-clean-100 dev-clean test-clean --out data/librispeech/shar

Each split becomes ``<out>/<split>/{cuts,recording}.NNNNNN.*`` with FLAC audio. Parquet files
are converted in parallel; file ``i`` writes shards numbered from ``i * 1000``.
"""

from __future__ import annotations

import argparse
import io
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def convert_parquet(path: str, out_dir: str, shard_offset: int, shard_size: int) -> int:
    import pyarrow.parquet as pq
    from lhotse import Recording, SupervisionSegment
    from lhotse.shar import SharWriter

    count = 0
    pf = pq.ParquetFile(path)
    with SharWriter(out_dir, fields={"recording": "flac"}, shard_size=shard_size, shard_offset=shard_offset) as writer:
        for group in range(pf.metadata.num_row_groups):
            for row in pf.read_row_group(group, columns=["id", "text", "speaker_id", "audio"]).to_pylist():
                recording = Recording.from_bytes(row["audio"]["bytes"], recording_id=row["id"])
                cut = recording.to_cut()
                cut.supervisions = [
                    SupervisionSegment(
                        id=row["id"],
                        recording_id=row["id"],
                        start=0.0,
                        duration=recording.duration,
                        channel=0,
                        text=row["text"],
                        language="en",
                        speaker=str(row["speaker_id"]),
                    )
                ]
                writer.write(cut)
                count += 1
    return count


def main() -> None:
    from megatron_speech.data.librispeech import SPLITS, find_parquet_files

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", default=["train-clean-100", "dev-clean", "test-clean"], choices=sorted(SPLITS))
    parser.add_argument("--out", default="data/librispeech/shar")
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    for split in args.splits:
        files = find_parquet_files(split)
        out_dir = Path(args.out) / split
        out_dir.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(args.workers) as pool:
            futures = [
                pool.submit(convert_parquet, str(f), str(out_dir), i * 1000, args.shard_size) for i, f in enumerate(files)
            ]
            total = sum(f.result() for f in futures)
        print(f"{split}: {total} utterances from {len(files)} parquet files -> {out_dir}")


if __name__ == "__main__":
    main()
