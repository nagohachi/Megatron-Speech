import torch

from megatron_speech.components.encoders.qwen3_asr import Qwen3ASREncoderConfig


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
