#!/bin/bash
# Convert the given parts of LibriSpeech into lhotse Shar shards with FLAC audio.
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
