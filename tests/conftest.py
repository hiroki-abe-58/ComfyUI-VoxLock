"""Test setup. References come from environment variables, never from this repository:

VOXLOCK_MODELS_DIR   a models/voxlock folder with the files of tools/model_files.sha256 (tools/download_model.py fetches them)
VOXLOCK_COMFYUI_DIR  ComfyUI checkout (v0.38.0 or later)
VOXLOCK_REQUIRE_REFERENCES=1  fail instead of skip when one of them is missing (CI sets this)
The tests run the real VoxLock models (small ONNX files, CPU) on synthetic two-voice audio (tests/audio.py), and compare the pack
with the official package's own steps (`voxlock.load_audio` + `VoxLock.isolate`) on the same files.
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))
REQUIRE = os.environ.get("VOXLOCK_REQUIRE_REFERENCES") == "1"


def _path(var):
    p = os.environ.get(var)
    return p if p and Path(p).exists() else None


COMFY = _path("VOXLOCK_COMFYUI_DIR")
MODELS = _path("VOXLOCK_MODELS_DIR")
if COMFY:
    sys.path.insert(1, COMFY)
    import comfy.options

    comfy.options.enable_args_parsing(False)
    import comfy.cli_args

    comfy.cli_args.args.cpu = True


def _need(value, what):
    if value:
        return value
    if REQUIRE:
        pytest.fail(f"{what} is required (VOXLOCK_REQUIRE_REFERENCES=1)")
    pytest.skip(f"{what} not available")


@pytest.fixture(scope="session")
def comfy_dir():
    return _need(COMFY, "ComfyUI checkout (VOXLOCK_COMFYUI_DIR)")


@pytest.fixture(scope="session")
def models_dir():
    """The models/voxlock folder (holding onnx/)."""
    return Path(_need(MODELS, "VoxLock model files (VOXLOCK_MODELS_DIR)"))


@pytest.fixture(scope="session")
def onnx_dir(models_dir):
    return str(models_dir / "onnx")
