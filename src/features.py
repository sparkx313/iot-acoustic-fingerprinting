"""
Audio Feature Extraction for IoT Acoustic Fingerprinting.
"""

import numpy as np
from scipy import signal
import librosa
from typing import Tuple


def load_audio(filepath: str, sr: int = 48000) -> Tuple[np.ndarray, int]:
    """Load audio file, resample to target sample rate."""
    y, sr_out = librosa.load(filepath, sr=sr, mono=True)
    return y, sr_out


def compute_mel_spectrogram(
    y: np.ndarray,
    sr: int = 48000,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmin: float = 20.0,
    fmax: float = 24000.0,
) -> np.ndarray:
    """
    Compute log-mel spectrogram.
    Returns: (n_mels, n_frames) float32 array.
    """
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels,
        n_fft=n_fft, hop_length=hop_length,
        fmin=fmin, fmax=fmax,
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    # Normalize to [0, 1]
    normalized = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-9)
    return normalized.astype(np.float32)


def extract_spectral_peak_features(y: np.ndarray, sr: int = 48000) -> np.ndarray:
    """
    Extract dominant frequency peaks — key for coil whine / motor identification.
    Returns: (n_peaks, 3) array of [freq_hz, amplitude_db, bandwidth_hz]
    """
    freqs, times, Sxx = signal.spectrogram(y, fs=sr, nperseg=4096)
    mean_spec = Sxx.mean(axis=1)

    # Find local maxima
    peaks, props = signal.find_peaks(
        mean_spec,
        height=mean_spec.max() * 0.05,  # at least 5% of max
        distance=sr // 4096 * 10,        # min 10 Hz apart
    )

    features = []
    for p in peaks[:30]:  # top 30 peaks
        freq_hz = freqs[p]
        amplitude_db = 10 * np.log10(mean_spec[p] + 1e-12)
        # Estimate bandwidth
        half_height = mean_spec[p] * 0.5
        left = p
        right = p
        while left > 0 and mean_spec[left] > half_height:
            left -= 1
        while right < len(mean_spec) - 1 and mean_spec[right] > half_height:
            right += 1
        bandwidth = freqs[min(right, len(freqs)-1)] - freqs[max(left, 0)]
        features.append([freq_hz, amplitude_db, bandwidth])

    return np.array(features, dtype=np.float32) if features else np.zeros((1, 3), dtype=np.float32)


def extract_temporal_features(y: np.ndarray, sr: int = 48000) -> np.ndarray:
    """
    Extract temporal envelope features — identifies duty cycles (thermostat, motor cycling).
    Returns: 64-dim vector.
    """
    # RMS energy envelope
    frame_length = int(0.05 * sr)  # 50ms frames
    hop = int(0.025 * sr)          # 25ms hop
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop)[0]

    # Normalize and resample to fixed length (64 values)
    if len(rms) < 64:
        rms = np.pad(rms, (0, 64 - len(rms)))
    else:
        indices = np.linspace(0, len(rms) - 1, 64).astype(int)
        rms = rms[indices]

    rms = rms / (rms.max() + 1e-9)
    return rms.astype(np.float32)


def extract_full_features(
    y: np.ndarray, sr: int = 48000, n_mels: int = 128
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract all feature views for multi-modal fusion.
    Returns: (mel_spec, spectral_peaks, temporal_features)
    """
    mel = compute_mel_spectrogram(y, sr=sr, n_mels=n_mels)
    peaks = extract_spectral_peak_features(y, sr)
    temporal = extract_temporal_features(y, sr)
    return mel, peaks, temporal
