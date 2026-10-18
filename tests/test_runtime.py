"""The runtime against the official package's own steps (`voxlock.load_audio` on a file, then `VoxLock.isolate`), with the real
models on synthetic two-voice audio."""

import numpy as np
import pytest
import soundfile as sf
import soxr
import torch
import voxlock
from voxlock import VoxLock

import audio
from voxlock_comfy import runtime


@pytest.fixture(autouse=True)
def fresh():
    yield
    runtime.release()


def _comfy_audio(data: np.ndarray, rate: int) -> dict:
    """[samples, channels] -> a ComfyUI AUDIO with one batch item ([1, channels, samples])."""
    return {"waveform": torch.from_numpy(np.ascontiguousarray(data.T))[None], "sample_rate": rate}


def _official(onnx_dir, precision, path, sample_path=None):
    """What `voxlock <path> [--sample <sample_path>] --precision <precision>` computes, with the same model folder."""
    vl = VoxLock(model_dir=onnx_dir, precision=precision)
    return vl.isolate(voxlock.load_audio(path), voxlock.load_audio(sample_path) if sample_path else None)


@pytest.mark.parametrize(("rate", "channels"), [(16000, 1), (48000, 2), (44100, 3), (22050, 1)])
def test_model_input_is_load_audio(tmp_path, rate, channels):
    data = audio.mixture(2.0, rate, channels)
    path = tmp_path / "in.wav"
    sf.write(path, data, rate, subtype="FLOAT")
    assert np.array_equal(runtime.to_model_input(data.T, rate), voxlock.load_audio(path))


def test_precision_files_follow_the_package():
    from voxlock.core import PRECISIONS

    for precision in runtime.PRECISIONS:
        enroll, chunk = PRECISIONS[precision]
        assert runtime.files(precision) == (f"{enroll}.onnx", f"{chunk}.onnx")
    with pytest.raises(ValueError):
        runtime.files("int4")


def test_a_missing_file_is_an_error_not_a_fallback(tmp_path, onnx_dir):
    import shutil

    shutil.copyfile(f"{onnx_dir}/enroll.onnx", tmp_path / "enroll.onnx")
    shutil.copyfile(f"{onnx_dir}/chunk.onnx", tmp_path / "chunk.onnx")
    with pytest.raises(FileNotFoundError, match="chunk_int8.onnx"):
        runtime.load(str(tmp_path), "int8")  # the package itself would quietly use chunk.onnx (fp32) here
    assert runtime.load(str(tmp_path), "fp32") is not None


@pytest.mark.parametrize("precision", runtime.PRECISIONS)
def test_isolate_equals_the_official_steps(tmp_path, onnx_dir, precision):
    data = audio.mixture(6.0, 48000, 2)
    path = tmp_path / "mix.wav"
    sf.write(path, data, 48000, subtype="FLOAT")
    out, report = runtime.isolate(runtime.load(onnx_dir, precision), _comfy_audio(data, 48000))
    expected = _official(onnx_dir, precision, path)
    assert out["sample_rate"] == 16000 and out["waveform"].shape == (1, 1, len(expected)) and out["waveform"].dtype == torch.float32
    assert np.array_equal(out["waveform"][0, 0].numpy(), expected)
    assert report["items"][0]["voice_sample"].startswith("auto") and report["output_sample_rate"] == 16000


def test_a_voice_sample_equals_the_official_sample_option(tmp_path, onnx_dir):
    data, voice = audio.mixture(6.0, 16000, 1), audio.main_voice(3.0, 16000)
    sf.write(tmp_path / "mix.wav", data, 16000, subtype="FLOAT")
    sf.write(tmp_path / "voice.wav", voice, 16000, subtype="FLOAT")
    out, report = runtime.isolate(runtime.load(onnx_dir, "int8"), _comfy_audio(data, 16000), _comfy_audio(voice, 16000))
    assert np.array_equal(out["waveform"][0, 0].numpy(), _official(onnx_dir, "int8", tmp_path / "mix.wav", tmp_path / "voice.wav"))
    assert report["items"][0]["voice_sample"] == "given: 3.00 s"
    auto, _ = runtime.isolate(runtime.load(onnx_dir, "int8"), _comfy_audio(data, 16000))
    assert not torch.equal(auto["waveform"], out["waveform"])  # the sample matters


def test_batch_items_are_isolated_on_their_own(tmp_path, onnx_dir):
    a, b = audio.mixture(4.0, 16000, 1, seed=0), audio.mixture(4.0, 16000, 1, seed=5)
    batch = {"waveform": torch.from_numpy(np.stack([a.T, b.T])), "sample_rate": 16000}
    out, report = runtime.isolate(runtime.load(onnx_dir, "int8"), batch)
    for i, data in enumerate((a, b)):
        single, _ = runtime.isolate(runtime.load(onnx_dir, "int8"), _comfy_audio(data, 16000))
        assert torch.equal(out["waveform"][i], single["waveform"][0])
    assert len(report["items"]) == 2


def test_output_at_the_input_rate_is_the_resampled_model_output(tmp_path, onnx_dir):
    data = audio.mixture(3.0, 44100, 1)
    vl = runtime.load(onnx_dir, "int8")
    model_rate, _ = runtime.isolate(vl, _comfy_audio(data, 44100), output_rate="16000")
    input_rate, report = runtime.isolate(vl, _comfy_audio(data, 44100), output_rate="input")
    expected = soxr.resample(model_rate["waveform"][0, 0].numpy(), 16000, 44100).astype(np.float32)
    assert input_rate["sample_rate"] == 44100 and np.array_equal(input_rate["waveform"][0, 0].numpy(), expected)
    assert report["output_sample_rate"] == 44100
    with pytest.raises(ValueError):
        runtime.isolate(vl, _comfy_audio(data, 44100), output_rate="8000")


def test_progress_counts_the_chunk_runs_and_interrupt_restores_the_model(onnx_dir):
    vl = runtime.load(onnx_dir, "int8")
    session = vl.chunk
    data = audio.mixture(5.0, 16000, 1)
    seen = []
    first, report = runtime.isolate(vl, _comfy_audio(data, 16000), progress=seen.append)
    assert seen == list(range(1, report["items"][0]["chunk_runs"] + 1))
    assert abs(len(seen) - runtime.runs_for(len(data))) <= 1

    class Stop(Exception):
        pass

    def stop(n):
        if n == 2:
            raise Stop

    with pytest.raises(Stop):
        runtime.isolate(vl, _comfy_audio(data, 16000), progress=stop)
    assert vl.chunk is session  # the counting wrapper is gone
    again, _ = runtime.isolate(vl, _comfy_audio(data, 16000))
    assert torch.equal(first["waveform"], again["waveform"])


def test_bad_inputs(onnx_dir):
    vl = runtime.load(onnx_dir, "int8")
    data = audio.mixture(2.0, 16000, 1)
    two = {"waveform": torch.from_numpy(np.stack([data.T, data.T, data.T])), "sample_rate": 16000}
    sample = {"waveform": torch.zeros(2, 1, 16000), "sample_rate": 16000}
    with pytest.raises(ValueError, match="voice_sample has 2 items"):
        runtime.isolate(vl, two, sample)
    with pytest.raises(ValueError, match="expected an AUDIO waveform"):
        runtime.isolate(vl, {"waveform": torch.zeros(1, 1, 0), "sample_rate": 16000})


def test_one_model_per_folder_precision_and_threads(onnx_dir):
    a = runtime.load(onnx_dir, "int8")
    assert runtime.load(onnx_dir, "int8") is a
    assert runtime.load(onnx_dir, "fp16") is not a and runtime.load(onnx_dir, "int8", threads=2) is not a
