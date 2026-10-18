"""Download the official VoxLock model files into ComfyUI's models/voxlock, and verify them.

    python tools/download_model.py --models-dir <ComfyUI>/models/voxlock [--precisions int8 fp16 ...] [--verify-only]

Source (pinned to the revision the voxlock 1.1.1 package itself uses; Apache-2.0):
  https://huggingface.co/arbazsiddiqui/voxlock   about 275 MB for all four precisions (int8 79 MB, int8-full 70 MB, fp16 51 MB,
                                                  fp32 100 MB; int8, int8-full and fp32 share enroll.onnx)
Result:
  models/voxlock/config.yaml, LICENSE
  models/voxlock/onnx/enroll.onnx, enroll_fp16.onnx, chunk.onnx, chunk_fp16.onnx, chunk_int8.onnx, chunk_int8_full.onnx
Every file is checked against its SHA-256 in tools/model_files.sha256. Needs `huggingface_hub` (installed with voxlock). Nothing
is downloaded unless you run this script.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO = ("arbazsiddiqui/voxlock", "a50f4e1d109301d4b6f61dec57ebb1a61e7d8f03")
PRECISION_FILES = {  # the files voxlock.core.PRECISIONS names for each precision
    "int8": ("onnx/enroll.onnx", "onnx/chunk_int8.onnx"),
    "int8-full": ("onnx/enroll.onnx", "onnx/chunk_int8_full.onnx"),
    "fp16": ("onnx/enroll_fp16.onnx", "onnx/chunk_fp16.onnx"),
    "fp32": ("onnx/enroll.onnx", "onnx/chunk.onnx"),
}
ALWAYS = ("config.yaml", "LICENSE")
MANIFEST = Path(__file__).resolve().with_name("model_files.sha256")


def expected_files() -> dict[str, str]:
    """{path below models/voxlock: sha256} from tools/model_files.sha256."""
    out = {}
    for line in MANIFEST.read_text(encoding="ascii").splitlines():
        if line and not line.startswith("#"):
            digest, _, name = line.partition("  ")
            out[name] = digest
    return out


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wanted(precisions) -> list[str]:
    names = list(ALWAYS)
    for precision in precisions:
        names += [n for n in PRECISION_FILES[precision] if n not in names]
    return names


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", type=Path, required=True, help="ComfyUI's models/voxlock folder")
    parser.add_argument("--precisions", nargs="+", choices=list(PRECISION_FILES), default=list(PRECISION_FILES),
                        help="model files to fetch (default: all four)")
    parser.add_argument("--verify-only", action="store_true", help="check existing files, download nothing")
    args = parser.parse_args(argv)

    expected = expected_files()
    names = wanted(args.precisions)
    models = args.models_dir
    if not args.verify_only:
        from huggingface_hub import hf_hub_download

        for name in names:
            target = models / name
            if not (target.is_file() and sha256(target) == expected[name]):
                hf_hub_download(repo_id=REPO[0], filename=name, revision=REPO[1], local_dir=str(models))
    ok = True
    for name in names:
        path = models / name
        good = path.is_file() and sha256(path) == expected[name]
        ok &= good
        print(f"{'ok      ' if good else ('MISMATCH' if path.is_file() else 'MISSING ')} {name}", flush=True)
    print("all files verified" if ok else "verification FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
