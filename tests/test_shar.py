from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from lhotse import CutSet

from megatron_speech.data.shar import main

# Utterances of a LibriSpeech part: two speakers, one of them in two chapters.
TRANSCRIPTS = {
    "19-198-0000": "NORTHANGER ABBEY",
    "19-198-0001": "CHAPTER ONE",
    "19-227-0000": "NO ONE WHO HAD EVER SEEN CATHERINE",
    "26-495-0000": "AND THE NIGHT WAS DARK",
    "26-495-0001": "SHE SAID",
}
SAMPLING_RATE = 16000
SHARD_SIZE = 2


@pytest.fixture(name="corpus_dir")
def build_corpus(tmp_path: Path) -> Path:
    """Build a LibriSpeech with the part ``dev-clean`` of noise, laid out as OpenSLR does."""
    corpus_dir = tmp_path / "LibriSpeech"
    rng = np.random.default_rng(0)
    for utterance_id, transcript in TRANSCRIPTS.items():
        speaker, chapter, _ = utterance_id.split("-")
        chapter_dir = corpus_dir / "dev-clean" / speaker / chapter
        chapter_dir.mkdir(parents=True, exist_ok=True)
        waveform = rng.standard_normal(rng.integers(8000, 24000)) * 0.1
        sf.write(chapter_dir / f"{utterance_id}.flac", waveform, SAMPLING_RATE)
        with open(chapter_dir / f"{speaker}-{chapter}.trans.txt", "a") as trans_file:
            trans_file.write(f"{utterance_id} {transcript}\n")
    return corpus_dir


def write_shar(corpus_dir: Path, output_dir: Path, seed: int) -> Path:
    """Write the corpus as Shar shards with the LibriSpeech recipe; return the part's directory."""
    main(
        [
            "librispeech",
            str(corpus_dir),
            str(output_dir),
            "--parts",
            "dev-clean",
            "--shard-size",
            str(SHARD_SIZE),
            "--seed",
            str(seed),
            "--num-jobs",
            "1",
        ]
    )
    return output_dir / "dev-clean"


def read_utterance_ids(shar_dir: Path) -> list[str]:
    """Read the ids of the utterances in the shards, in the order that the shards hold them."""
    return [
        cut.recording_id
        for path in sorted(shar_dir.glob("cuts.*.jsonl.gz"))
        for cut in CutSet.from_file(path)
    ]


@pytest.fixture(name="shar_dir")
def build_shar(corpus_dir: Path, tmp_path: Path) -> Path:
    """Write the corpus as Shar shards with seed 0."""
    return write_shar(corpus_dir, tmp_path / "shar", 0)


def test_shards_hold_every_utterance_once(shar_dir: Path) -> None:
    """Check that the shards hold each utterance of the corpus once."""
    assert sorted(read_utterance_ids(shar_dir)) == sorted(TRANSCRIPTS)


def test_shards_hold_normalized_transcripts(shar_dir: Path) -> None:
    """Check that each utterance in the shards has its transcript, in lowercase for LibriSpeech."""
    for cut in CutSet.from_shar(in_dir=shar_dir):
        assert cut.supervisions[0].text == TRANSCRIPTS[cut.recording_id].lower()


def test_shards_hold_original_audio(shar_dir: Path, corpus_dir: Path) -> None:
    """Check that each utterance in the shards has the samples of its FLAC file."""
    for cut in CutSet.from_shar(in_dir=shar_dir):
        speaker, chapter, _ = cut.recording_id.split("-")
        flac_path = corpus_dir / "dev-clean" / speaker / chapter / f"{cut.recording_id}.flac"
        waveform, _ = sf.read(flac_path, dtype="float32")
        np.testing.assert_array_equal(cut.load_audio()[0], waveform)


def test_utterance_order_depends_on_seed(corpus_dir: Path, tmp_path: Path) -> None:
    """Check that the utterance order is the same with the same seed and differs with another."""
    utterance_ids_seed0 = read_utterance_ids(write_shar(corpus_dir, tmp_path / "seed0", 0))
    utterance_ids_seed0_rerun = read_utterance_ids(
        write_shar(corpus_dir, tmp_path / "seed0-rerun", 0)
    )
    utterance_ids_seed1 = read_utterance_ids(write_shar(corpus_dir, tmp_path / "seed1", 1))
    assert utterance_ids_seed0_rerun == utterance_ids_seed0
    assert utterance_ids_seed1 != utterance_ids_seed0


def test_shards_hold_shard_size_utterances(shar_dir: Path) -> None:
    """Check that every shard but the last holds ``SHARD_SIZE`` utterances."""
    shard_sizes = [len(CutSet.from_file(path)) for path in sorted(shar_dir.glob("cuts.*.jsonl.gz"))]
    num_utterances = len(TRANSCRIPTS)
    assert shard_sizes == [
        min(SHARD_SIZE, num_utterances - start) for start in range(0, num_utterances, SHARD_SIZE)
    ]


@pytest.mark.parametrize("option", ["--shard-size", "--num-jobs"])
def test_rejects_counts_below_one(
    option: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Check that the writer is stopped with an error naming the option for a count below 1."""
    with pytest.raises(SystemExit):
        main(
            [
                "librispeech",
                str(tmp_path),
                str(tmp_path / "shar"),
                "--parts",
                "dev-clean",
                option,
                "0",
            ]
        )
    assert f"{option} must be at least 1" in capsys.readouterr().err
