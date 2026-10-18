# Verification

What was run to check this node pack, with the raw numbers.

## 1. Setup

- Windows 11 Pro (build 26200), Intel Core Ultra 9 285K (24 threads), 64 GB RAM. VoxLock runs on the CPU.
- ComfyUI v0.38.0 (commit `6b747c0428c343e1417219641db93a4fb7cb69ae`), started with `--listen 127.0.0.1 --disable-auto-launch
  --disable-api-nodes`.
- `voxlock` 1.1.1 from PyPI; model files from `arbazsiddiqui/voxlock` at `a50f4e1d109301d4b6f61dec57ebb1a61e7d8f03` (the revision the
  package's own downloader uses), SHA-256 checked (`tools/model_files.sha256`).
- Two Python environments:

  | | Reference ("ref") | ComfyUI ("comfy") |
  | --- | --- | --- |
  | Contents | `pip install voxlock==1.1.1` only | ComfyUI's requirements, torch 2.14.1+cu130, `voxlock==1.1.1` |
  | Python | 3.12.13 | 3.12.13 |
  | onnxruntime | 1.31.0 | 1.31.0 |
  | numpy | 2.5.4 | 2.5.2 |
  | soxr / soundfile | 1.1.0 / 0.14.0 | 1.1.0 / 0.14.0 |

- Audio: three demo mixtures from the VoxLock repository (`demo/audio/*_off.mp3` at `08ed37c`: two LibriSpeech test-clean speakers,
  CC BY 4.0, in a simulated room; 11-14 s, 16 kHz mono), decoded once and stored as 32-bit float WAV so that the official loader
  and ComfyUI's Load Audio read the same samples. Also a 48 kHz stereo copy of one of them (resampled with soxr, right channel at
  0.9) and a 3 s voice sample cut from the repository's isolated version of that clip.

## 2. Pack vs the official package (separate environments)

**Official side** (ref): `VoxLock(model_dir, precision).isolate(voxlock.load_audio(file), voxlock.load_audio(sample) or None)`,
i.e. what `voxlock file [--sample sample] --precision P` computes, before it writes a 16-bit WAV. **Pack side** (comfy):
`scripts/run_jobs.py`, the runtime behind the nodes, on the same files read as ComfyUI hands audio to a node. Compared: SHA-256 of
the float32 output.

| Job | Precision | Official | Pack | Official time |
| --- | --- | --- | --- | --- |
| bystander (14.0 s) | int8 | `e89889e70f60a0e3` | `e89889e70f60a0e3` | 1.78 s |
| bystander | int8-full | `14ebeca5f84957bb` | `14ebeca5f84957bb` | 1.59 s |
| bystander | fp16 | `6e2289e7dfd2f623` | `6e2289e7dfd2f623` | 2.18 s |
| bystander | fp32 | `6d97bc93e49bd42e` | `6d97bc93e49bd42e` | 2.15 s |
| tail_024 (12.0 s) | int8 | `494191b3b285c930` | `494191b3b285c930` | 1.50 s |
| tail_028 (11.4 s) | int8 | `857c0e8f4a556593` | `857c0e8f4a556593` | 1.68 s |
| bystander, 48 kHz stereo | int8 | `0f94916ae4045dea` | `0f94916ae4045dea` | 1.78 s |
| bystander, given 3 s voice sample | int8 | `04482b356dc22b81` | `04482b356dc22b81` | 1.64 s |

**8 of 8 identical.** Without a sample, both sides use the official default (the loudest 3 s in the first 15 s).

Polarity and level: the output is in phase with the input. Fitting `input ≈ a * output + rest` gives a = +0.969, +0.976 and +0.971
for the three clips (the package's changelog for 1.1.0: "Output is at the input's level"). The repository's own isolated demo
clip (`bystander_000_on.mp3`) is the same signal with the opposite polarity: correlation -0.9965 with the pack's output and
-0.755 with the input (likely made with 1.0.0, whose output the changelog describes as twice the level).

## 3. Real ComfyUI queue (comfy environment)

ComfyUI started from a checkout with a copy of this pack in `custom_nodes`, the models in an `extra_model_paths.yaml` folder, the
audio files as its input folder; jobs submitted through `POST /prompt` (`scripts/queue_client.py`). Load Audio → VoxLock Loader →
VoxLock Isolate Voice → Save Audio (Advanced, FLAC). The FLAC is 16-bit, so the decoded file is compared with the official float
output (section 2) in units of one 16-bit step.

| Job | Workflow, overrides | Saved | Largest difference |
| --- | --- | --- | --- |
| bystander, int8 | `isolate_voice` | 16 kHz, 224160 samples | 0.5 step |
| bystander, fp32 | `isolate_voice`, precision fp32 | 16 kHz | 0.5 step |
| tail_024, int8 | `isolate_voice` | 16 kHz | 0.5 step |
| bystander 48 kHz stereo | `isolate_voice`, output_sample_rate input | 48 kHz, 672480 samples (official output resampled with soxr) | 0.5 step |
| bystander + 3 s sample | `isolate_with_sample` | 16 kHz | 0.5 step |

Half a step is the rounding of the 16-bit FLAC. Both shipped workflows also ran unchanged (with the demo clips copied to the names
they use). Node time 1.7-2.2 s per job, wall time 2.0-4.7 s including loading the model on first use.

## 4. Speed, memory, threads

10 minutes of 16 kHz speech (the three clips repeated with 0.5 s gaps), pack runtime in the comfy environment, model loaded,
`threads` = 0:

| Precision | Time | Speed | Model runs (progress estimate) |
| --- | --- | --- | --- |
| int8 | 49.1 s | 12x real time | 751 (752) |
| int8-full | 45.9 s | 13x real time | 751 (752) |
| fp16 | 73.9 s | 8x real time | 751 (752) |
| fp32 | 72.6 s | 8x real time | 751 (752) |

Resident memory grew by 0.08-0.12 GB during a run.

Thread count, bystander clip, compared with `threads` = 0:

| Precision | 1 thread | 4 threads | 8 threads |
| --- | --- | --- | --- |
| int8 | max difference 0.0025 | 0.0021 | 0.0018 |
| fp32 | 2.4e-7 | identical | identical |

Repeated runs with the same settings are identical (tests and queue).

## 5. Tests and CI

- `tests/` (30 tests, CPU, the real models on synthetic two-voice audio from `tests/audio.py`):
  - the model input equals `voxlock.load_audio` for mono/stereo/3-channel input at 16, 22.05, 44.1 and 48 kHz;
  - the output equals the official steps on the same file for all four precisions, with and without a voice sample, for
    batches, and with output at the input rate;
  - progress, interrupt (the model keeps working and gives the same output), bad inputs, one model per folder / precision /
    threads;
  - a missing model file is an error instead of the package's silent fallback to the fp32 file;
  - the download manifest matches the package's revision and file names; the package scan for environment / network / process
    code, and no use of the package's downloader;
  - the nodes as ComfyUI loads them, on audio decoded by ComfyUI's Load Audio, including ComfyUI's own `validate_prompt` on the
    shipped workflows.
- `scripts/ci_queue_check.py`: a stock ComfyUI server with the pack and the real models, both shipped workflows through the
  queue; saved FLAC = official output within half a 16-bit step, fp16 differs from int8, a repeated job gives the same audio, an
  invalid precision is rejected.
- GitHub Actions (`.github/workflows/ci.yml`), CPU, ComfyUI v0.38.0 at its pinned commit, the models downloaded with
  `tools/download_model.py` (SHA-256 checked): Ubuntu 24.04 and Windows with Python 3.12 / torch 2.14.1, Ubuntu 24.04 with
  Python 3.10 / torch 2.8.0. Each job runs ruff, a scan for absolute local paths, the tests with the references required (nothing
  may be skipped) and the real-queue check. All three jobs passed on the first run (`352e50f`); on Linux, too, the queue's FLAC
  equals the official output within half a 16-bit step.

## 6. Clean install

Fresh ComfyUI v0.38.0 clone, fresh Python 3.12 virtual environment (torch 2.14.1+cu130, then ComfyUI's `requirements.txt`, then
this pack's `requirements.txt`), `git clone` of this repository into `custom_nodes`:

- Installing the pack's requirements added `voxlock` 1.1.1, `onnxruntime` 1.31.0, `soundfile`, `soxr` and their dependencies
  (`cffi`, `flatbuffers`, `protobuf`, `pycparser`); nothing that ComfyUI had installed was changed.
- `tools/download_model.py --models-dir ComfyUI/models/voxlock` downloaded all 8 files from Hugging Face (19 s) and verified them.
- `scripts/ci_queue_check.py --keep-pack` against the installed pack and the downloaded models: passed (section 5).

## ComfyUI v0.41.0

Since 2026-10-19 the CI also runs the test suite and the real-queue check against ComfyUI v0.41.0 (commit
`98cc0bf84b7cb1383654c5b402e7612d875d6d22`, Ubuntu 24.04, Python 3.12 / torch 2.14.1), next to v0.38.0, the release the
real-model checks above used. Both pass.
