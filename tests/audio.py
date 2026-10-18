"""Synthetic test audio: two "voices" (harmonic tones with a pitch contour and a syllable rhythm) talking over each other, so the
tests need no recordings. Deterministic for a given seed."""

from __future__ import annotations

import numpy as np


def voice(seconds: float, rate: int, f0: float, syllables_per_second: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    pitch = f0 * (1 + 0.08 * np.sin(2 * np.pi * 0.7 * t + rng.uniform(0, 6)))
    phase = 2 * np.pi * np.cumsum(pitch) / rate
    tone = sum(np.sin(k * phase) / k for k in range(1, 12))
    envelope = np.clip(np.sin(np.pi * syllables_per_second * t + rng.uniform(0, 6)), 0, None) ** 2
    pauses = (np.sin(2 * np.pi * 0.23 * t + rng.uniform(0, 6)) > -0.6).astype(np.float64)
    return (0.25 * tone * envelope * pauses).astype(np.float32)


def mixture(seconds: float = 6.0, rate: int = 16000, channels: int = 1, seed: int = 0) -> np.ndarray:
    """[samples, channels] float32: a louder main voice and a quieter second voice; channels differ slightly."""
    main = voice(seconds, rate, 130.0, 4.0, seed)
    other = voice(seconds, rate, 215.0, 5.5, seed + 1)
    rng = np.random.default_rng(seed + 2)
    mono = main + 0.5 * other + 0.002 * rng.standard_normal(len(main)).astype(np.float32)
    return np.stack([mono * (1.0 - 0.1 * c) for c in range(channels)], axis=1).astype(np.float32)


def main_voice(seconds: float = 3.0, rate: int = 16000, seed: int = 0) -> np.ndarray:
    """[samples, 1] float32: the main voice alone (a voice sample)."""
    return voice(seconds, rate, 130.0, 4.0, seed)[:, None]
