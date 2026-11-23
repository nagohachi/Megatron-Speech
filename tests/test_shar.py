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


@pytest.fixture(name="shar_dir")
def build_shar(corpus_dir: Path, tmp_path: Path) -> Path:
    """Write the corpus as Shar shards with the LibriSpeech recipe; return the part's directory."""
    output_dir = tmp_path / "shar"
    main(
        [
            "librispeech",
            str(corpus_dir),
            str(output_dir),
            "--parts",
            "dev-clean",
            "--shard-size",
            str(SHARD_SIZE),
            "--num-jobs",
            "1",
        ]
    )
    return output_dir / "dev-clean"


def test_shards_hold_every_utterance(shar_dir: Path, corpus_dir: Path) -> None:
    """Check that the shards hold each utterance once, with its transcript and its audio."""
    cuts = list(CutSet.from_shar(in_dir=shar_dir))
    assert sorted(cut.recording_id for cut in cuts) == sorted(TRANSCRIPTS)
    for cut in cuts:
        assert cut.supervisions[0].text == TRANSCRIPTS[cut.recording_id]
        speaker, chapter, _ = cut.recording_id.split("-")
        flac_path = corpus_dir / "dev-clean" / speaker / chapter / f"{cut.recording_id}.flac"
        waveform, _ = sf.read(flac_path, dtype="float32")
        np.testing.assert_array_equal(cut.load_audio()[0], waveform)


def test_shards_hold_shard_size_utterances(shar_dir: Path) -> None:
    """Check that every shard but the last holds ``SHARD_SIZE`` utterances."""
    shard_sizes = [len(CutSet.from_file(path)) for path in sorted(shar_dir.glob("cuts.*.jsonl.gz"))]
    assert shard_sizes == [2, 2, 1]


@pytest.mark.parametrize("option", ["--shard-size", "--num-jobs"])
def test_rejects_counts_below_one(
    option: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Check that a count below 1 stops the writer, with a message that names the option."""
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
