"""
Feature extractor for FunASR using FunASR's built-in frontend.
Extracts fbank features compatible with FunASR models.
"""

import numpy as np
import torch
from typing import List, Tuple


class FunASRFeatExtractorFromFrontend:
    """
    Feature extractor using FunASR's built-in frontend.
    This replaces kaldifeat-based extraction with FunASR's native extract_fbank.
    """

    def __init__(self, frontend, device: str = "cuda:0"):
        """
        Initialize with FunASR frontend.

        Args:
            frontend: FunASR frontend object from model kwargs
            device: Device string (e.g., "cuda:0")
        """
        self.frontend = frontend
        self.device = device

    def __call__(self, wavs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features using FunASR frontend.

        Args:
            wavs: List of waveform tensors (1D float tensors at 16kHz)

        Returns:
            Tuple of (features [B, T, D], lengths [B])
        """
        from funasr.utils.load_utils import extract_fbank

        # Ensure all wavs are proper tensors
        processed_wavs = []
        for wav in wavs:
            if isinstance(wav, np.ndarray):
                wav_tensor = torch.from_numpy(wav).float()
            elif isinstance(wav, torch.Tensor):
                wav_tensor = wav.float()
            else:
                raise TypeError("wav must be tensor or numpy array")

            if wav_tensor.ndim > 1:
                wav_tensor = wav_tensor.squeeze()

            processed_wavs.append(wav_tensor)

        speech, speech_lengths = extract_fbank(
            processed_wavs,
            frontend=self.frontend,
            is_final=True,
        )

        return speech.to(self.device), speech_lengths.to(self.device)


# Alias for backwards compatibility
FunASRFeatExtractor = FunASRFeatExtractorFromFrontend
