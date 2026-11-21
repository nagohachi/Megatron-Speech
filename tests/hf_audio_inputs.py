import torch
from transformers import AutoProcessor

from megatron_speech.components.encoders.qwen3_asr import Qwen3ASREncoderConfig

# The released Qwen3-ASR that tests compare with.
HF_MODEL = "Qwen/Qwen3-ASR-1.7B-hf"
# Its processor, downloaded if it is not in the cache.
PROCESSOR = AutoProcessor.from_pretrained(HF_MODEL)
# Qwen3-ASR-1.7B's transcript of a LibriSpeech test-clean utterance: cased, with punctuation.
TRANSCRIPT = "Concord returned to its place amidst the tents."
# Lengths in samples, at 16 kHz, of the waveforms that tests compare with Hugging Face's
# Qwen3-ASR feature extractor and processor. Hugging Face pads audio shorter than 8000 samples
# (0.5 s) to 8000. They give 50 mel frames in one partial encoder chunk (the first two), 200 in
# two whole chunks, 230 with a partial last chunk, and 350 from audio that ends within a hop.
WAVEFORM_LENGTHS = {
    "shorter-than-min-length": 3000,
    "shorter-than-a-chunk": 8000,
    "whole-chunks": 32000,
    "partial-last-chunk": 36800,
    "ending-within-a-hop": 56080,
}


def build_hf_audio_inputs(
    mels: list[torch.Tensor], encoder_config: Qwen3ASREncoderConfig
) -> dict[str, torch.Tensor]:
    """Build Hugging Face's Qwen3-ASR audio inputs from ``[num_mel_bins, num_frames]`` log-mels.

    The log-mels are zero padded to a multiple of the chunk length;
    ``input_features_mask`` is 1 on the frames of each utterance.
    """
    mel_frames_per_chunk = encoder_config.mel_frames_per_chunk
    longest_length = max(mel.shape[1] for mel in mels)
    padded_length = -(-longest_length // mel_frames_per_chunk) * mel_frames_per_chunk
    features = torch.zeros(len(mels), encoder_config.num_mel_bins, padded_length)
    mask = torch.zeros(len(mels), padded_length, dtype=torch.long)
    for index, mel in enumerate(mels):
        features[index, :, : mel.shape[1]] = mel
        mask[index, : mel.shape[1]] = 1
    return {"input_features": features.cuda(), "input_features_mask": mask.cuda()}
