"""Isolate voices in audio files through this pack's runtime (the code behind the nodes), in-process, and record hashes comparable
with the official package run on the same files.

    python scripts/run_jobs.py --models-dir <models/voxlock> --jobs jobs.json --out <dir>

jobs.json: [{"id", "audio": path[, "sample": path][, "precision": "int8"][, "output_rate": "16000"]}]. Files are read with soundfile
as ComfyUI's Load Audio would hand them to the node ([channels, samples] float32). Writes <out>/<id>.npy (float32 output) and
results.jsonl with the SHA-256 (first 16 hex digits) of the output's float32 bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from voxlock_comfy import runtime  # noqa: E402


def read(path: str) -> dict:
    import soundfile as sf
    import torch

    data, rate = sf.read(path, dtype="float32", always_2d=True)
    return {"waveform": torch.from_numpy(np.ascontiguousarray(data.T))[None], "sample_rate": rate}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", type=Path, required=True)
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for job in json.loads(args.jobs.read_text(encoding="utf-8")):
        vl = runtime.load(str(args.models_dir / "onnx"), job.get("precision", "int8"))
        out, report = runtime.isolate(vl, read(job["audio"]), read(job["sample"]) if job.get("sample") else None, job.get("output_rate", "16000"))
        y = out["waveform"][0, 0].numpy()
        np.save(args.out / f"{job['id']}.npy", y)
        row = {**job, "sample_rate": out["sample_rate"], "samples": len(y), "sha": hashlib.sha256(y.tobytes()).hexdigest()[:16], **report}
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("id", "sha", "samples", "sample_rate", "seconds")}), flush=True)
    with open(args.out / "results.jsonl", "w", encoding="utf-8") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
