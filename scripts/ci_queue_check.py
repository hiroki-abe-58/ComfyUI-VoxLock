"""CPU "real queue" check for CI and for a clean-install test: start a real ComfyUI server with this node pack installed, put two
synthetic recordings into its input folder, and run the shipped workflows through POST /prompt with the real VoxLock models.

    python scripts/ci_queue_check.py --comfyui <ComfyUI checkout> --models-dir <models/voxlock> [--workdir DIR] [--keep-pack]

VoxLock runs on the CPU, so CI runs the real nodes and the real models (tools/download_model.py fetches them). The check proves that
the pack installs into a stock ComfyUI, that Load Audio -> VoxLock Loader -> VoxLock Isolate Voice -> Save Audio (FLAC) executes
through the queue, that the saved FLAC equals the official package's output for the same file (to the 16 bits of the FLAC), with
and without a voice sample, that another precision gives other audio, that a repeated job gives the same audio, and that an invalid
precision is rejected. The server is started and stopped by this script only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import audio  # noqa: E402
import queue_client  # noqa: E402

NODE_IDS = {"VoxLockLoader", "VoxLockIsolateVoice"}
STEP = 1 / 32768  # one 16-bit step: ComfyUI's FLAC is 16 bits


def wait_up(server: str, proc: subprocess.Popen, timeout: float = 300) -> None:
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            raise SystemExit(f"ComfyUI exited early with code {proc.returncode}")
        try:
            urllib.request.urlopen(f"{server}/system_stats", timeout=3).read()
            return
        except Exception:
            time.sleep(2)
    raise SystemExit("ComfyUI did not start in time")


def main(argv=None) -> int:
    import soundfile as sf
    import soxr
    import voxlock
    from voxlock import VoxLock

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comfyui", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True, help="a models/voxlock folder (holding onnx/)")
    parser.add_argument("--port", type=int, default=8197)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--keep-pack", action="store_true", help="use the pack already installed in custom_nodes (clean-install test)")
    args = parser.parse_args(argv)

    comfy = args.comfyui.resolve()
    models = args.models_dir.resolve()
    work = (args.workdir or Path(tempfile.mkdtemp(prefix="voxlock-ci-"))).resolve()
    work.mkdir(parents=True, exist_ok=True)
    pack = comfy / "custom_nodes" / "ComfyUI-VoxLock"
    if pack.is_symlink() or getattr(os.path, "isjunction", lambda p: False)(pack):  # never delete through a link into a working copy
        raise SystemExit(f"{pack} is a link; use a ComfyUI checkout without it")
    if not args.keep_pack:
        if pack.exists():
            shutil.rmtree(pack)
        shutil.copytree(ROOT, pack, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", ".ruff_cache", "outputs"))
    elif not pack.is_dir():
        raise SystemExit(f"--keep-pack: {pack} is not installed")
    inputs, output = work / "input", work / "output"
    inputs.mkdir(exist_ok=True)
    sf.write(inputs / "voice_mix.wav", audio.mixture(8.0, 44100, 2), 44100, subtype="FLOAT")  # the names the workflows use
    sf.write(inputs / "voice_sample.wav", audio.main_voice(3.0, 22050), 22050, subtype="FLOAT")
    yaml = work / "extra_model_paths.yaml"
    yaml.write_text(f"voxlock_ci:\n  base_path: {models.parent.as_posix()}\n  voxlock: {models.name}\n", encoding="utf-8")

    onnx = str(models / "onnx")
    official = VoxLock(model_dir=onnx, precision="int8").isolate(voxlock.load_audio(inputs / "voice_mix.wav"))
    official_sample = soxr.resample(VoxLock(model_dir=onnx, precision="int8").isolate(
        voxlock.load_audio(inputs / "voice_mix.wav"), voxlock.load_audio(inputs / "voice_sample.wav")), 16000, 44100).astype(np.float32)

    log = (work / "comfyui.log").open("w", encoding="utf-8")
    server = f"http://127.0.0.1:{args.port}"
    proc = subprocess.Popen(
        [sys.executable, "main.py", "--cpu", "--listen", "127.0.0.1", "--port", str(args.port), "--disable-auto-launch", "--disable-api-nodes",
         "--extra-model-paths-config", str(yaml), "--input-directory", str(inputs), "--output-directory", str(output)],
        cwd=comfy, stdout=log, stderr=subprocess.STDOUT,
    )
    failures = []
    try:
        wait_up(server, proc)
        info = json.loads(urllib.request.urlopen(f"{server}/object_info", timeout=30).read())
        print("registered:", sorted(NODE_IDS & set(info)))
        if NODE_IDS - set(info):
            raise SystemExit(f"nodes missing from /object_info: {sorted(NODE_IDS - set(info))}")

        def run(workflow: str, prefix: str, expect_success: bool = True, **overrides) -> dict:
            wf = json.loads((ROOT / "workflows" / "api" / workflow).read_text(encoding="utf-8"))
            wf["4"]["inputs"]["filename_prefix"] = f"ci/{prefix}"
            for key, value in overrides.items():
                node, _, field = key.partition(".")
                wf[node]["inputs"][field] = value
            try:
                result = queue_client.run(server, wf, None, 600)
            except SystemExit as rejected:  # POST /prompt refused the workflow (validation error)
                result = {"status": "rejected", "wall_seconds": 0, "error": str(rejected)[:200]}
            error = result.get("error") or {}
            print(json.dumps({"run": prefix, "status": result["status"], "seconds": result["wall_seconds"], "files": result.get("files"),
                              "error": error.get("exception_message", "")[:200] if isinstance(error, dict) else error}))
            if (result["status"] == "success") != expect_success:
                failures.append(f"{prefix}: expected {'success' if expect_success else 'an error'}, got {result['status']} {error}")
            return result

        def saved(prefix: str):
            files = sorted((output / "ci").glob(f"{prefix}*.flac"))
            if len(files) != 1:
                failures.append(f"{prefix}: expected one saved .flac, found {files}")
                return None, None
            data, rate = sf.read(str(files[0]), dtype="float32")
            return data, rate

        def same(prefix: str, got, rate, expected, expected_rate):
            if got is None:
                return
            diff = float(np.abs(got - expected).max()) if got.shape == expected.shape else None
            print(json.dumps({"compare": prefix, "rate": rate, "samples": len(got), "max_diff_in_16bit_steps": None if diff is None else round(diff / STEP, 3)}))
            if rate != expected_rate or diff is None or diff > STEP:
                failures.append(f"{prefix}: saved FLAC differs from the official output by more than one 16-bit step ({diff}, rate {rate})")

        report = run("isolate_voice.json", "int8_a")
        a, rate = saved("int8_a")
        same("int8_a", a, rate, official, 16000)
        texts = [json.loads(v[0]) for v in (report.get("text") or {}).values() if v and v[0].lstrip().startswith("{")]
        if not texts or texts[0]["model"]["precision"] != "int8" or texts[0]["items"][0]["voice_sample"] != "auto: loudest 3 s in the first 15 s":
            failures.append(f"unexpected report {texts}")
        run("isolate_voice.json", "fp16", **{"2.precision": "fp16"})
        run("isolate_voice.json", "int8_b")  # after the fp16 job, so ComfyUI's cache no longer holds the int8 result
        b, _ = saved("int8_b")
        other, _ = saved("fp16")
        if a is not None and b is not None and other is not None:
            if not np.array_equal(a, b):
                failures.append("the same job gave different audio")
            if a.shape == other.shape and np.array_equal(a, other):
                failures.append("int8 and fp16 gave the same audio")
        run("isolate_with_sample.json", "with_sample")
        s, rate = saved("with_sample")
        same("with_sample", s, rate, official_sample, 44100)
        run("isolate_voice.json", "bad_precision", expect_success=False, **{"2.precision": "int4"})
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        print((work / "comfyui.log").read_text(encoding="utf-8", errors="replace")[-4000:])
        return 1
    print("queue check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
