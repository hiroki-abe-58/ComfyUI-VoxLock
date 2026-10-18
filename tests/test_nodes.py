"""The node pack as ComfyUI loads it: registration, model discovery, ComfyUI's own validation of the shipped workflows, and node
execution on audio decoded by ComfyUI's own Load Audio, compared with the official package's steps on the same file."""

import asyncio
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import voxlock
from voxlock import VoxLock

import audio

ROOT = Path(__file__).resolve().parents[1]
NODE_IDS = {"VoxLockLoader", "VoxLockIsolateVoice"}


def pack_module(comfy_nodes, name):
    """A module of the pack as ComfyUI imported it (under a path-based package name, so not the `voxlock_comfy` the tests import)."""
    return importlib.import_module(comfy_nodes.NODE_CLASS_MAPPINGS["VoxLockIsolateVoice"].__module__.rsplit(".", 1)[0] + "." + name)


@pytest.fixture(scope="module")
def comfy_nodes(comfy_dir, models_dir, tmp_path_factory):
    import folder_paths
    import nodes

    base = tmp_path_factory.mktemp("comfy")
    folder_paths.set_output_directory(str(base / "output"))
    folder_paths.set_input_directory(str(base / "input"))
    (base / "input").mkdir()
    sf.write(base / "input" / "voice_mix.wav", audio.mixture(6.0, 44100, 2), 44100, subtype="FLOAT")  # the names the workflows use
    sf.write(base / "input" / "voice_sample.wav", audio.main_voice(3.0, 22050), 22050, subtype="FLOAT")
    folder_paths.add_model_folder_path("voxlock", str(models_dir), is_default=True)

    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
    assert asyncio.run(nodes.load_custom_node(str(ROOT))) is True
    yield nodes
    pack_module(nodes, "runtime").release()


def _load_audio(comfy_nodes, name):
    return comfy_nodes.NODE_CLASS_MAPPINGS["LoadAudio"].execute(name).args[0]


def _model(comfy_nodes, precision="int8"):
    return comfy_nodes.NODE_CLASS_MAPPINGS["VoxLockLoader"].execute(precision, 0).args[0]


def _isolate(comfy_nodes, precision="int8", output_rate="16000", sample=None):
    mix = _load_audio(comfy_nodes, "voice_mix.wav")
    voice = _load_audio(comfy_nodes, sample) if sample else None
    return comfy_nodes.NODE_CLASS_MAPPINGS["VoxLockIsolateVoice"].execute(_model(comfy_nodes, precision), mix, output_rate, voice).args


def test_registered(comfy_nodes):
    assert NODE_IDS <= set(comfy_nodes.NODE_CLASS_MAPPINGS)


@pytest.mark.parametrize("name", sorted(p.name for p in (ROOT / "workflows" / "api").glob("*.json")))
def test_api_workflows_pass_comfy_validation(comfy_nodes, name):
    import execution

    prompt = json.loads((ROOT / "workflows" / "api" / name).read_text(encoding="utf-8"))
    valid, error, outputs, node_errors = asyncio.run(execution.validate_prompt("t", prompt, None))
    assert valid, (error, node_errors)


@pytest.mark.parametrize("precision", ["int8", "fp16"])
def test_node_output_equals_the_official_steps_on_the_same_file(comfy_nodes, models_dir, precision):
    import folder_paths

    out, report = _isolate(comfy_nodes, precision)
    path = Path(folder_paths.get_input_directory()) / "voice_mix.wav"
    expected = VoxLock(model_dir=str(models_dir / "onnx"), precision=precision).isolate(voxlock.load_audio(path))
    assert out["sample_rate"] == 16000 and np.array_equal(out["waveform"][0, 0].numpy(), expected)
    parsed = json.loads(report)
    assert parsed["model"] == {"precision": precision, "threads": 0} and parsed["items"][0]["voice_sample"].startswith("auto")


def test_voice_sample_and_input_rate(comfy_nodes, models_dir):
    import folder_paths

    out, report = _isolate(comfy_nodes, "int8", "input", sample="voice_sample.wav")
    folder = Path(folder_paths.get_input_directory())
    expected = VoxLock(model_dir=str(models_dir / "onnx"), precision="int8").isolate(voxlock.load_audio(folder / "voice_mix.wav"),
                                                                                    voxlock.load_audio(folder / "voice_sample.wav"))
    import soxr

    assert out["sample_rate"] == 44100
    assert np.array_equal(out["waveform"][0, 0].numpy(), soxr.resample(expected, 16000, 44100).astype(np.float32))
    assert json.loads(report)["items"][0]["voice_sample"] == "given: 3.00 s"


def test_missing_model_files_are_reported(comfy_nodes, tmp_path, monkeypatch):
    paths = pack_module(comfy_nodes, "paths")
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx" / "enroll.onnx").write_bytes(b"")
    monkeypatch.setattr(paths, "_bases", lambda: [str(tmp_path)])
    with pytest.raises(FileNotFoundError, match="chunk_int8.onnx"):
        _model(comfy_nodes)


def test_interrupt_stops_between_model_runs(comfy_nodes, monkeypatch):
    import comfy.model_management as mm

    before, _ = _isolate(comfy_nodes)
    calls = {"n": 0}

    def interrupt():
        calls["n"] += 1
        if calls["n"] > 2:
            raise mm.InterruptProcessingException()

    monkeypatch.setattr(mm, "throw_exception_if_processing_interrupted", interrupt)
    with pytest.raises(mm.InterruptProcessingException):
        _isolate(comfy_nodes)
    assert calls["n"] == 3  # the check before starting, then one per model run
    monkeypatch.undo()
    after, _ = _isolate(comfy_nodes)
    assert np.array_equal(before["waveform"].numpy(), after["waveform"].numpy())


def test_shipped_gui_workflows_use_known_nodes():
    for path in sorted((ROOT / "workflows" / "gui").glob("*.json")):
        types = {n["type"] for n in json.loads(path.read_text(encoding="utf-8"))["nodes"]}
        assert types <= NODE_IDS | {"LoadAudio", "SaveAudioAdvanced", "PreviewAny", "Note", "MarkdownNote"}, path.name
