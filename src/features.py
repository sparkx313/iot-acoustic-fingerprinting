"""Fixed-size spectral and waveform features from local PCM WAV recordings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import welch

FEATURE_NAMES = [
    "rms",
    "peak_absolute",
    "zero_crossing_rate",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "spectral_rolloff_85_hz",
    "spectral_entropy",
    "dominant_frequency_hz",
    "dominant_power_share",
]


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, raw = wavfile.read(path)
    if sample_rate <= 0:
        raise ValueError(f"{path}: sample rate must be positive.")
    if raw.ndim not in (1, 2):
        raise ValueError(f"{path}: expected mono or stereo PCM audio.")
    if raw.dtype.kind == "u":
        midpoint = (np.iinfo(raw.dtype).max + 1) / 2
        audio = (raw.astype(np.float64) - midpoint) / midpoint
    elif raw.dtype.kind == "i":
        scale = float(max(abs(np.iinfo(raw.dtype).min), np.iinfo(raw.dtype).max))
        audio = raw.astype(np.float64) / scale
    elif raw.dtype.kind == "f":
        audio = raw.astype(np.float64)
    else:
        raise ValueError(f"{path}: unsupported WAV sample type {raw.dtype}.")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if len(audio) < 32:
        raise ValueError(f"{path}: recording must contain at least 32 audio samples.")
    if not np.isfinite(audio).all():
        raise ValueError(f"{path}: audio contains a non-finite sample.")
    return int(sample_rate), audio


def extract_features(path: Path) -> dict[str, float]:
    sample_rate, audio = read_wav(path)
    centered = audio - float(np.mean(audio))
    rms = float(np.sqrt(np.mean(centered**2)))
    peak = float(np.max(np.abs(centered)))
    zero_crossing_rate = float(
        np.mean(np.signbit(centered[1:]) != np.signbit(centered[:-1]))
    )

    frequencies, power = welch(
        centered, fs=sample_rate, nperseg=min(len(centered), 4096)
    )
    positive = frequencies > 0
    frequencies = frequencies[positive]
    power = power[positive]
    total_power = float(power.sum())
    if len(power) and total_power > 0:
        probability = power / total_power
        centroid = float(np.sum(frequencies * probability))
        bandwidth = float(np.sqrt(np.sum(((frequencies - centroid) ** 2) * probability)))
        cumulative = np.cumsum(probability)
        rolloff_index = min(int(np.searchsorted(cumulative, 0.85)), len(frequencies) - 1)
        peak_index = int(np.argmax(power))
        entropy = (
            float(-np.sum(probability * np.log(probability + 1e-15)) / np.log(len(probability)))
            if len(probability) > 1
            else 0.0
        )
        dominant_frequency = float(frequencies[peak_index])
        dominant_power_share = float(probability[peak_index])
        rolloff = float(frequencies[rolloff_index])
    else:
        centroid = bandwidth = entropy = dominant_frequency = dominant_power_share = rolloff = 0.0

    return {
        "rms": rms,
        "peak_absolute": peak,
        "zero_crossing_rate": zero_crossing_rate,
        "spectral_centroid_hz": centroid,
        "spectral_bandwidth_hz": bandwidth,
        "spectral_rolloff_85_hz": rolloff,
        "spectral_entropy": entropy,
        "dominant_frequency_hz": dominant_frequency,
        "dominant_power_share": dominant_power_share,
    }
