"""Word error rate of a Hugging Face Qwen3-ASR model on LibriSpeech, with plain transformers.

    apptainer/run.sh python scripts/eval_hf_asr.py --model Qwen/Qwen3-ASR-1.7B-hf
    apptainer/run.sh python scripts/eval_hf_asr.py --model outputs/hf/run/merged
    apptainer/run.sh python scripts/eval_hf_asr.py --model Qwen/Qwen3-ASR-1.7B-hf --adapter outputs/hf/run/adapter

The language is forced to English (the processor prefills ``language English<asr_text>``).
Hypotheses and references are upper-cased, punctuation other than apostrophes is removed, and
WER = (substitutions + deletions + insertions) / reference words, summed over utterances.
"""

from __future__ import annotations

import argparse
import json
import re
import time

import torch


def normalize(text: str) -> list[str]:
    text = re.sub(r"[^A-Z0-9' ]", " ", text.upper())
    return text.split()


def calc_edit_distance(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def main() -> None:
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

    from megatron_speech.data.librispeech import iter_librispeech

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None, help="PEFT adapter directory")
    parser.add_argument("--split", default="test-clean")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", default=None, help="write per-utterance results (jsonl)")
    ns = parser.parse_args()

    processor = AutoProcessor.from_pretrained(ns.model)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(ns.model, dtype=torch.bfloat16).cuda().eval()
    if ns.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, ns.adapter).eval()

    utts = list(iter_librispeech(ns.split, limit=ns.limit))
    errors = words = 0
    results = []
    start = time.time()
    for i in range(0, len(utts), ns.batch_size):
        chunk = utts[i : i + ns.batch_size]
        inputs = processor.apply_transcription_request(
            [u[1] for u in chunk], language="English", sampling_rate=chunk[0][2]
        ).to("cuda")
        inputs["input_features"] = inputs["input_features"].to(torch.bfloat16)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        hyps = processor.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        for (uid, _, _, ref), hyp in zip(chunk, hyps):
            r, h = normalize(ref), normalize(hyp)
            e = calc_edit_distance(r, h)
            errors += e
            words += len(r)
            results.append({"id": uid, "ref": ref, "hyp": hyp, "errors": e, "words": len(r)})
    elapsed = time.time() - start
    print(f"model={ns.model} adapter={ns.adapter} split={ns.split} utterances={len(utts)} "
          f"WER={100 * errors / words:.2f}% ({errors}/{words}) time={elapsed:.0f}s")
    if ns.output:
        with open(ns.output, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
