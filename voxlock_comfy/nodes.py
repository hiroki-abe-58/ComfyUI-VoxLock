"""ComfyUI nodes for VoxLock voice isolation (runs in the ComfyUI process on the CPU; nothing is downloaded at run time)."""

from __future__ import annotations

import json

from comfy_api.latest import ComfyExtension, io

from . import paths, runtime

VoxLockModel = io.Custom("VOXLOCK_MODEL")


def _progress(total: int):
    import comfy.model_management as mm
    import comfy.utils

    bar = comfy.utils.ProgressBar(max(1, int(total)))

    def tick(done: int):
        mm.throw_exception_if_processing_interrupted()
        bar.update_absolute(min(done, total), total)

    return tick


class VoxLockLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VoxLockLoader",
            display_name="VoxLock Loader",
            category="loaders/voxlock",
            description=("Loads VoxLock (ONNX, CPU) from models/voxlock/onnx: a voice isolator that keeps one voice and removes the "
                         "other voices in the room."),
            inputs=[
                io.Combo.Input("precision", options=list(runtime.PRECISIONS), default="int8",
                               tooltip="Model files: int8 (the package default), int8-full (fastest), fp16, or fp32 (the benchmarked model)"),
                io.Int.Input("threads", default=0, min=0, max=256,
                             tooltip="CPU threads for ONNX Runtime; 0 lets ONNX Runtime choose (the official default). The thread count changes the int8 output slightly"),
            ],
            outputs=[VoxLockModel.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, precision, threads) -> io.NodeOutput:
        folder = paths.model_dir(runtime.files(precision))
        runtime.load(folder, precision, threads)  # fail here, not later, when a file is missing or broken
        return io.NodeOutput({"model_dir": folder, "precision": precision, "threads": int(threads)})


class VoxLockIsolateVoice(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VoxLockIsolateVoice",
            display_name="VoxLock Isolate Voice",
            category="audio/voxlock",
            description=("Keeps the voice of one speaker and removes the other voices. Without a voice "
                         "sample it locks onto the loudest 3 s in the first 15 s, as the official command line does. The model "
                         "works on 16 kHz mono speech; every batch item is processed on its own."),
            inputs=[
                VoxLockModel.Input("model"),
                io.Audio.Input("audio"),
                io.Combo.Input("output_sample_rate", options=list(runtime.OUTPUT_RATES), default="16000",
                               tooltip="16000: the model's output as is. input: resampled back to the input's rate (the content stays 16 kHz band-limited)"),
                io.Audio.Input("voice_sample", optional=True,
                               tooltip="Optional: a few seconds of the voice to keep (about 3 s works best); one item for all, or one per audio item"),
            ],
            outputs=[io.Audio.Output(display_name="audio"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, model, audio, output_sample_rate, voice_sample=None) -> io.NodeOutput:
        import comfy.model_management as mm
        from voxlock.core import SR

        mm.throw_exception_if_processing_interrupted()
        vl = runtime.load(model["model_dir"], model["precision"], model["threads"])
        rate = int(audio["sample_rate"])
        samples = audio["waveform"].shape[-1]
        total = runtime.runs_for(round(samples * SR / rate)) * int(audio["waveform"].shape[0] if audio["waveform"].ndim == 3 else 1)
        out, report = runtime.isolate(vl, audio, voice_sample, output_sample_rate, progress=_progress(total))
        report = json.dumps({"model": {"precision": model["precision"], "threads": model["threads"]}, **report}, indent=1)
        return io.NodeOutput(out, report)


class VoxLockExtension(ComfyExtension):
    async def get_node_list(self):
        return [VoxLockLoader, VoxLockIsolateVoice]


async def comfy_entrypoint() -> VoxLockExtension:
    paths.register()
    return VoxLockExtension()
