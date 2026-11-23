#!/bin/bash
# Convert parts of LibriSpeech into lhotse Shar shards with FLAC audio.
#
#   apptainer/run.sh bash scripts/prepare_librispeech.sh <LibriSpeech dir> <output dir> <part>...
#
# <LibriSpeech dir> holds the parts as OpenSLR distributes them, such as train-clean-100 and
# dev-clean. The shards of each part go into <output dir>/<part>, and the lhotse manifests that
# list the audio files and transcripts into <output dir>/manifests. NUM_JOBS (default 4) is the
# number of processes.
set -euo pipefail

corpus_dir=$1
output_dir=$2
shift 2
num_jobs=${NUM_JOBS:-4}
manifest_dir=$output_dir/manifests

for part in "$@"; do
    lhotse prepare librispeech --dataset-parts "$part" --num-jobs "$num_jobs" \
        "$corpus_dir" "$manifest_dir"
    lhotse cut simple \
        --recording-manifest "$manifest_dir/librispeech_recordings_$part.jsonl.gz" \
        --supervision-manifest "$manifest_dir/librispeech_supervisions_$part.jsonl.gz" \
        "$manifest_dir/librispeech_cuts_$part.jsonl.gz"
    # LibriSpeech lists the utterances by speaker; shuffling first puts many speakers in a shard.
    lhotse shar export --audio flac --shuffle --num-jobs "$num_jobs" \
        "$manifest_dir/librispeech_cuts_$part.jsonl.gz" "$output_dir/$part"
done
