"""Voice isolation with the official `voxlock` package (arbazsiddiqui/voxlock): keep one voice, remove the other voices.

The steps are those of the official command line (`voxlock call.wav [--sample voice.wav]`): every input is made mono and 16 kHz
the way `voxlock.load_audio` makes a file (mean of the channels, then soxr), and `VoxLock.isolate` keeps the voice of the sample -
by default the loudest 3 s in the first 15 s of the input - and removes the other voices. The model files come from a local
folder only (`VoxLock(model_dir=...)`); this module never calls the package's downloader.
"""

from __future__ import annotations

import os
import threading
import time

import numpy as np

PRECISIONS = ("int8", "int8-full", "fp16", "fp32")
OUTPUT_RATES = ("16000", "input")
_LOCK = threading.Lock()
_CACHE: dict = {}


def files(precision: str) -> tuple[str, str]:
    """The (enroll, chunk) ONNX files of a precision, as the package names them."""
    from voxlock.core import PRECISIONS as PACKAGE_PRECISIONS

    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    enroll, chunk = PACKAGE_PRECISIONS[precision]
    return f"{enroll}.onnx", f"{chunk}.onnx"


def load(model_dir: str, precision: str = "int8", threads: int = 0):
    """A `voxlock.VoxLock` on the files in model_dir (kept for the session; the models are small and run on the CPU)."""
    names = files(precision)
    missing = [n for n in names if not os.path.isfile(os.path.join(model_dir, n))]
    if missing:  # the package would quietly use the fp32 files instead of a missing chunk file
        raise FileNotFoundError(f"{', '.join(missing)} not found in {model_dir} (see the README for the download command)")
    key = (os.path.abspath(model_dir), precision, int(threads))
    with _LOCK:
        if key not in _CACHE:
            from voxlock import VoxLock

            _CACHE[key] = VoxLock(model_dir=model_dir, precision=precision, threads=int(threads))
        return _CACHE[key]


def release() -> None:
    with _LOCK:
        _CACHE.clear()


def to_model_input(channels_first, sample_rate: int) -> np.ndarray:
    """[channels, samples] float audio -> 16 kHz mono float32, exactly as `voxlock.load_audio` turns a file (which soundfile reads
    as [samples, channels]) into the model input."""
    import soxr
    from voxlock.core import SR

    x = np.ascontiguousarray(np.asarray(channels_first, dtype=np.float32).T)
    x = x.mean(1)
    return (soxr.resample(x, sample_rate, SR) if sample_rate != SR else x).astype(np.float32)


def _items(audio: dict, name: str) -> tuple[np.ndarray, int]:
    waveform = audio["waveform"]
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 3 or waveform.shape[-1] == 0:
        raise ValueError(f"{name}: expected an AUDIO waveform [batch, channels, samples], got shape {tuple(waveform.shape)}")
    return waveform.detach().float().cpu().numpy(), int(audio["sample_rate"])


class _CountedSession:
    """The package's chunk session with a callback before every run (one run per 100 frames = 0.8 s of audio). The runs are the
    package's own; the callback may raise to interrupt."""

    def __init__(self, session, callback):
        self._session, self._callback, self.calls = session, callback, 0

    def run(self, *args, **kwargs):
        self.calls += 1
        self._callback(self.calls)
        return self._session.run(*args, **kwargs)


def runs_for(samples_16k: int) -> int:
    """About how many chunk-model runs `VoxLock.isolate` makes for an input of this length (for a progress bar): one per 100
    STFT frames of 128 samples, plus one for the end of the input."""
    frames = 1 + samples_16k // 128
    return -(-frames // 100) + 1


def isolate(vl, audio: dict, sample: dict | None = None, output_rate: str = "16000", progress=None) -> tuple[dict, dict]:
    """Returns (AUDIO, report). Every batch item of `audio` is isolated on its own; `sample` (optional) is a recording of the voice
    to keep, one item for all or one per item. output_rate "16000" is the model's rate; "input" resamples back to the input rate
    with soxr. progress(n) is called before every chunk-model run with the number of runs so far over the whole batch (it may raise
    to interrupt)."""
    from voxlock.core import SR

    if output_rate not in OUTPUT_RATES:
        raise ValueError(f"output_rate must be one of {OUTPUT_RATES}")
    waves, rate = _items(audio, "audio")
    samples = None
    if sample is not None:
        samples, sample_rate = _items(sample, "voice_sample")
        if len(samples) not in (1, len(waves)):
            raise ValueError(f"voice_sample has {len(samples)} items; give one, or one per audio item ({len(waves)})")
    started = time.perf_counter()
    outputs, items, done = [], [], 0
    for i, wave in enumerate(waves):
        x = to_model_input(wave, rate)
        voice = None if samples is None else to_model_input(samples[i if len(samples) > 1 else 0], sample_rate)
        session = vl.chunk
        counted = _CountedSession(session, lambda n, before=done: progress(before + n) if progress else None)
        with _LOCK:  # the counting session is set on the shared model for this call only
            vl.chunk = counted
            try:
                y = vl.isolate(x, voice)
            finally:
                vl.chunk = session
        if output_rate == "input" and rate != SR:
            import soxr

            y = soxr.resample(y, SR, rate).astype(np.float32)
        outputs.append(y)
        done += counted.calls
        items.append({"input_seconds": round(len(x) / SR, 3), "chunk_runs": counted.calls,
                      "voice_sample": "auto: loudest 3 s in the first 15 s" if voice is None else f"given: {len(voice) / SR:.2f} s"})
    out_rate = rate if output_rate == "input" else SR
    waveform = np.stack(outputs)[:, None, :]
    import torch

    report = {"output_sample_rate": out_rate, "items": items, "seconds": round(time.perf_counter() - started, 3)}
    return {"waveform": torch.from_numpy(np.ascontiguousarray(waveform)), "sample_rate": out_rate}, report
