# GlassesRobot integration

`src/egodata_eval/get_depth.py` and the legacy batch entry point
`src/FoundationStereo/scripts/stereo2depth.py` both use this implementation.
The old `src/FoundationStereo` tree remains because the project also stores
SAM2 utilities and ZED calibration assets there; its stereo network is no
longer imported by either depth path.

## Checkpoint

The default checkpoint is NVIDIA's official
[`nvidia/c-fast-foundationstereo`](https://huggingface.co/nvidia/c-fast-foundationstereo)
release. It is placed at:

```
weights/c-fast-foundationstereo/model_best_bp2_serialize.pth
```

The checkpoint is intentionally gitignored. Its expected SHA-256 is
`7aee85948373da62b0503c2542507129a3e7cab9d97d10e6790d89512a7db214`.
The live pipeline defaults to four refinement iterations, maximum disparity
192, and the Triton cost-volume implementation. All can be overridden from
`src/egodata_eval/eval.py` CLI flags. The other upstream speed/accuracy
checkpoints from the Google Drive folder can also be selected with
`--stereo-checkpoint`.

## RTX 5090 environment note

This machine has a Blackwell (`sm_120`) GPU. The upstream README's PyTorch
2.6/CUDA 12.4 command targets older GPUs and cannot run kernels on this card.
Use a CUDA 12.8 build that supports `sm_120` (for example PyTorch 2.8 CUDA
12.8), then install `requirements.txt`. Keep the matching Torch/TorchVision
pair from the same PyTorch package index.

The first inference compiles kernels and is expected to be slower. Benchmark
only after warm-up. TensorRT export is optional; the PyTorch/Triton path is the
default because upstream still has open Blackwell TensorRT compatibility
issues.

## Camera-only rollout profiling

The profiler opens the ZED directly and never imports or initializes the robot
SDK. With no `--mask` or `--click`, click the object once in the live frame;
SAM and FoundationPose registration are reported as initialization and excluded
from steady-state averages.

```bash
python src/egodata_eval/profile_camera_rollout.py \
  --ckpt MBA/ckpt_deploy/policy_last.ckpt \
  --base-from-camera glasses_hardware/calib/T_base_cam_runtime.npy \
  --warmup 20 \
  --frames 100
```

The output directory contains `per_frame.csv` and `summary.json`. Timings use a
CUDA synchronization at every GPU-stage boundary, so the component breakdown
captures actual GPU execution rather than asynchronous launch time. The
`motion_planning` stage is the trajectory-to-pose-command conversion present in
`eval.py`; it does not include collision checking, robot SDK calls, or physical
motion because no such planner exists in the current pipeline.
