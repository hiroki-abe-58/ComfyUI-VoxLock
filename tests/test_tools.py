"""The download tool's manifest covers every file the nodes can ask for, and its verification reports what it finds."""

import hashlib
import importlib.util
from pathlib import Path

from voxlock_comfy import runtime

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("download_model", ROOT / "tools" / "download_model.py")
download_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download_model)


def test_manifest_covers_every_precision_and_matches_the_package():
    import voxlock.core

    files = download_model.expected_files()
    assert all(len(d) == 64 and int(d, 16) >= 0 for d in files.values())
    assert set(download_model.PRECISION_FILES) == set(runtime.PRECISIONS)
    for precision, pair in download_model.PRECISION_FILES.items():
        assert tuple(f"onnx/{n}" for n in runtime.files(precision)) == pair
        assert all(name in files for name in pair)
    assert download_model.REPO == (voxlock.core.REPO, voxlock.core.REVISION)  # the revision the package itself downloads


def test_selected_precisions_only(tmp_path):
    assert download_model.wanted(["fp16"]) == ["config.yaml", "LICENSE", "onnx/enroll_fp16.onnx", "onnx/chunk_fp16.onnx"]
    assert download_model.wanted(["int8", "fp32"]) == ["config.yaml", "LICENSE", "onnx/enroll.onnx", "onnx/chunk_int8.onnx",
                                                       "onnx/chunk.onnx"]


def test_verify_only_reports_missing_and_mismatching_files(tmp_path, capsys, monkeypatch):
    manifest = tmp_path / "model_files.sha256"
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "config.yaml").write_bytes(b"x")
    (tmp_path / "models" / "LICENSE").write_bytes(b"not the license")
    manifest.write_text(f"# test\n{hashlib.sha256(b'x').hexdigest()}  config.yaml\n{'0' * 64}  LICENSE\n{'1' * 64}  onnx/enroll_fp16.onnx\n"
                        f"{'2' * 64}  onnx/chunk_fp16.onnx\n", encoding="ascii")
    monkeypatch.setattr(download_model, "MANIFEST", manifest)
    assert download_model.main(["--models-dir", str(tmp_path / "models"), "--precisions", "fp16", "--verify-only"]) == 1
    out = capsys.readouterr().out
    assert "ok       config.yaml" in out and "MISMATCH LICENSE" in out and "MISSING  onnx/chunk_fp16.onnx" in out
