"""Model location: `<ComfyUI models>/voxlock/onnx/` (plus any path added for `voxlock` in extra_model_paths.yaml), the layout of
the Hugging Face repository arbazsiddiqui/voxlock: `onnx/enroll.onnx`, `onnx/enroll_fp16.onnx`, `onnx/chunk.onnx`,
`onnx/chunk_fp16.onnx`, `onnx/chunk_int8.onnx`, `onnx/chunk_int8_full.onnx`. Nothing is downloaded at run time.
"""

from __future__ import annotations

import os

FOLDER = "voxlock"
ONNX = "onnx"


def register() -> None:
    import folder_paths

    folder_paths.add_model_folder_path(FOLDER, os.path.join(folder_paths.models_dir, FOLDER), is_default=False)


def _bases() -> list[str]:
    import folder_paths

    return [b for b in folder_paths.get_folder_paths(FOLDER) if os.path.isdir(b)]


def model_dir(files: tuple[str, ...]) -> str:
    """The `onnx` folder of the first registered voxlock folder that holds all of `files`."""
    for base in _bases():
        folder = os.path.join(base, ONNX)
        if all(os.path.isfile(os.path.join(folder, name)) for name in files):
            return folder
    raise FileNotFoundError(f"models/{FOLDER}/{ONNX}/ does not hold {', '.join(files)} (see the README for the download command)")
