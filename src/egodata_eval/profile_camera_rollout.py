#!/usr/bin/env python3
"""Profile one camera-only perception/policy rollout without robot hardware.

The measured steady-state stages are camera capture, Fast-FoundationStereo,
FoundationPose tracking, policy input preparation, policy inference, policy
output decoding, and the robot-independent motion-plan post-processing used by
``eval.py``. SAM and FoundationPose registration are one-time initialization
costs and are reported separately from the rollout averages.
"""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch


_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[2]
_SRC_ROOT = _HERE.parents[1]
for _path in (_PROJECT_ROOT, _SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import MinkowskiEngine as ME  # type: ignore  # noqa: E402
from FoundationStereo.sam2_root.notebooks.get_mask import click_mask  # type: ignore  # noqa: E402
from egodata_eval.eval import TrajectoryPredictor  # type: ignore  # noqa: E402
from egodata_eval.eval_utils import (  # type: ignore  # noqa: E402
    _build_pose_mats,
    _denormalize_obj_traj,
    _import_zed_class,
)
from egodata_eval.get_depth import DepthEstimator  # type: ignore  # noqa: E402
from egodata_eval.get_pose import PoseEstimatorFP  # type: ignore  # noqa: E402


STAGES = (
    "camera_capture",
    "foundation_stereo",
    "foundation_pose",
    "policy_preprocess",
    "policy_inference",
    "policy_postprocess",
    "motion_planning",
    "rollout_total",
)


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(call: Callable[[], Any], synchronize_cuda: bool = True) -> Tuple[Any, float]:
    if synchronize_cuda:
        _cuda_sync()
    start = time.perf_counter()
    result = call()
    if synchronize_cuda:
        _cuda_sync()
    return result, (time.perf_counter() - start) * 1000.0


def _read_stereo_or_raise(camera: Any) -> Tuple[np.ndarray, np.ndarray]:
    frames = camera.read_stereo()
    if frames is None or frames[0] is None or frames[1] is None:
        raise RuntimeError("ZED failed to return a stereo pair")
    return frames


def _interactive_click(image_bgr: np.ndarray) -> Tuple[float, float]:
    window = "Select object for profiling"
    selected: List[Tuple[float, float]] = []

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            selected[:] = [(float(x), float(y))]

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    try:
        while not selected:
            cv2.imshow(window, image_bgr)
            key = cv2.waitKey(10) & 0xFF
            if key in (27, ord("q")):
                raise RuntimeError("Object selection cancelled")
    finally:
        cv2.destroyWindow(window)
    return selected[0]


def _initial_mask(
    image_bgr: np.ndarray,
    mask_path: Optional[Path],
    click_xy: Optional[Tuple[float, float]],
) -> Tuple[np.ndarray, float]:
    if mask_path is not None:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")
        if mask.shape != image_bgr.shape[:2]:
            mask = cv2.resize(
                mask,
                (image_bgr.shape[1], image_bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        return (mask > 0).astype(np.uint8) * 255, 0.0

    point = click_xy or _interactive_click(image_bgr)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask, elapsed_ms = _timed(
        lambda: click_mask(image_rgb, [point], labels=[1], multimask=True)
    )
    return mask, elapsed_ms


def _load_transform(path: Optional[Path]) -> np.ndarray:
    if path is None:
        print("[WARN] --base-from-camera not provided; policy receives camera-frame data.")
        return np.eye(4, dtype=np.float32)
    transform = np.load(str(path)).astype(np.float32)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform in {path}, got {transform.shape}")
    return transform


def _prepare_policy_inputs(
    predictor: TrajectoryPredictor,
    image_bgr: np.ndarray,
    depth_m: np.ndarray,
    K: np.ndarray,
    pose_cam_ob: np.ndarray,
    base_from_camera: np.ndarray,
) -> Tuple[Any, torch.Tensor]:
    pose_base_ob = base_from_camera @ pose_cam_ob
    feats, coords = predictor._make_sparse_input(
        image_bgr, depth_m, K, T_base_cam=base_from_camera
    )
    sparse_tensor = ME.SparseTensor(feats, coords)
    current_obj = predictor._current_obj_vec(pose_base_ob)
    current_obj_t = torch.from_numpy(current_obj[None]).to(predictor.device)
    return sparse_tensor, current_obj_t


def _policy_forward(
    predictor: TrajectoryPredictor,
    sparse_tensor: Any,
    current_obj: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    with torch.inference_mode():
        return predictor.model(
            sparse_tensor,
            actions=None,
            batch_size=1,
            current_obj=current_obj,
        )


def _decode_policy(outputs: Dict[str, torch.Tensor]) -> np.ndarray:
    if "obj_pred" not in outputs:
        raise RuntimeError("Policy output does not contain 'obj_pred'")
    normalized = outputs["obj_pred"].squeeze(0).detach().cpu().numpy()
    return _denormalize_obj_traj(normalized).astype(np.float32)


def _motion_plan_postprocess(
    trajectory: np.ndarray,
    pose_cam_ob: np.ndarray,
    base_from_camera: np.ndarray,
    current_tcp_pose: np.ndarray,
    steps: int,
) -> Dict[str, np.ndarray]:
    """Reproduce eval.py's robot-independent target construction.

    This is trajectory-to-command conversion, not collision-aware MoveIt-style
    planning; the repository currently has no separate collision planner.
    """
    pose_base_ob = base_from_camera @ pose_cam_ob
    pose_sequence = _build_pose_mats(trajectory[:, :3], trajectory[:, 3:9])
    count = min(int(steps), max(0, len(pose_sequence) - 1))
    relative_xyz = (
        pose_sequence[1 : 1 + count, :3, 3] - pose_base_ob[:3, 3][None]
    )
    target_xyz = current_tcp_pose[:3][None] + relative_xyz
    target_quat = np.repeat(current_tcp_pose[3:7][None], count, axis=0)
    target_pose7 = np.concatenate([target_xyz, target_quat], axis=1)
    gripper = trajectory[1 : 1 + count, 9] if trajectory.shape[1] > 9 else np.empty(0)
    return {
        "pose_sequence": pose_sequence,
        "target_pose7": target_pose7.astype(np.float32),
        "gripper": gripper.astype(np.float32),
    }


def _stats(values: List[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "std_ms": float(array.std()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
    }


def _print_summary(summary: Dict[str, Dict[str, float]], frame_count: int) -> None:
    print(f"\nSteady-state rollout timing ({frame_count} frames)")
    print(f"{'stage':24s} {'mean':>10s} {'p50':>10s} {'p95':>10s} {'std':>10s}")
    for stage in STAGES:
        values = summary[stage]
        print(
            f"{stage:24s} {values['mean_ms']:9.2f}ms "
            f"{values['p50_ms']:9.2f}ms {values['p95_ms']:9.2f}ms "
            f"{values['std_ms']:9.2f}ms"
        )
    for stage in ("policy_total", "pipeline_compute"):
        values = summary[stage]
        print(
            f"{stage:24s} {values['mean_ms']:9.2f}ms "
            f"{values['p50_ms']:9.2f}ms {values['p95_ms']:9.2f}ms "
            f"{values['std_ms']:9.2f}ms"
        )
    total_ms = summary["rollout_total"]["mean_ms"]
    print(f"effective sequential throughput: {1000.0 / total_ms:.2f} FPS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile camera -> stereo -> pose -> policy -> motion plan without a robot"
    )
    parser.add_argument("--ckpt", type=Path, required=True, help="RISE policy checkpoint")
    parser.add_argument(
        "--mesh",
        type=Path,
        default=_PROJECT_ROOT / "data" / "book" / "mesh.obj",
    )
    parser.add_argument("--mask", type=Path, default=None, help="Mask for the first live frame")
    parser.add_argument("--click", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--base-from-camera", type=Path, default=None, help="Optional 4x4 .npy transform")
    parser.add_argument("--frames", type=int, default=100, help="Measured rollout frames")
    parser.add_argument("--warmup", type=int, default=20, help="Unmeasured warm-up frames")
    parser.add_argument("--resolution", default="WVGA")
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--stereo-checkpoint", type=Path, default=None)
    parser.add_argument("--stereo-iters", type=int, default=4)
    parser.add_argument("--stereo-max-disp", type=int, default=192)
    parser.add_argument(
        "--stereo-volume-backend",
        choices=("triton", "pytorch1"),
        default="triton",
    )
    parser.add_argument("--pose-register-iters", type=int, default=5)
    parser.add_argument("--pose-track-iters", type=int, default=2)
    parser.add_argument("--motion-steps", type=int, default=3)
    parser.add_argument(
        "--current-tcp-pose",
        nargs=7,
        type=float,
        default=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_HERE.parent / "eval_output" / f"profile_{datetime.now():%Y%m%d_%H%M%S}",
    )
    args = parser.parse_args()
    if args.frames < 1 or args.warmup < 0:
        parser.error("--frames must be >= 1 and --warmup must be >= 0")
    if args.mask is not None and args.click is not None:
        parser.error("Use either --mask or --click, not both")
    return args


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This profiling pipeline requires CUDA")

    base_from_camera = _load_transform(args.base_from_camera)
    current_tcp_pose = np.asarray(args.current_tcp_pose, dtype=np.float32)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    initialization: Dict[str, float] = {}

    print("[INFO] Loading models (excluded from rollout timing)...")
    depth_estimator, initialization["foundation_stereo_load_ms"] = _timed(
        lambda: DepthEstimator(
            ckpt_path=args.stereo_checkpoint,
            valid_iters=args.stereo_iters,
            max_disp=args.stereo_max_disp,
            volume_backend=args.stereo_volume_backend,
        )
    )
    # K_ZED.txt is calibrated for the 640x360 frames returned by the project
    # ZED wrapper. Keep depth conversion and pose projection correct when a
    # different profiling size is requested.
    depth_estimator.K[0, :] *= args.width / 640.0
    depth_estimator.K[1, :] *= args.height / 360.0
    pose_estimator, initialization["foundation_pose_load_ms"] = _timed(
        lambda: PoseEstimatorFP(
            args.mesh,
            register_iterations=args.pose_register_iters,
            track_iterations=args.pose_track_iters,
        )
    )
    policy, initialization["policy_load_ms"] = _timed(
        lambda: TrajectoryPredictor(ckpt_path=args.ckpt)
    )

    ZEDCamera = _import_zed_class()
    camera, initialization["camera_open_ms"] = _timed(
        lambda: ZEDCamera(resolution=args.resolution, fps=args.camera_fps),
        synchronize_cuda=False,
    )
    samples: List[Dict[str, Any]] = []
    try:
        (left, right), initialization["camera_capture_ms"] = _timed(
            lambda: _read_stereo_or_raise(camera), synchronize_cuda=False
        )
        if left.shape[1] != args.width or left.shape[0] != args.height:
            left = cv2.resize(left, (args.width, args.height), interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, (args.width, args.height), interpolation=cv2.INTER_AREA)

        click_xy = tuple(args.click) if args.click is not None else None
        mask, initialization["sam_mask_ms"] = _initial_mask(left, args.mask, click_xy)
        depth, initialization["foundation_stereo_first_ms"] = _timed(
            lambda: depth_estimator.depth(left, right)
        )
        pose, initialization["foundation_pose_register_ms"] = _timed(
            lambda: pose_estimator.initialize(left, depth, mask, depth_estimator.K)
        )
        if pose is None:
            raise RuntimeError("FoundationPose registration failed")

        total_frames = args.warmup + args.frames
        print(f"[INFO] Running {args.warmup} warm-up + {args.frames} measured frames...")
        for index in range(total_frames):
            rollout_start = time.perf_counter()
            (left, right), camera_ms = _timed(
                lambda: _read_stereo_or_raise(camera), synchronize_cuda=False
            )
            if left.shape[1] != args.width or left.shape[0] != args.height:
                left = cv2.resize(left, (args.width, args.height), interpolation=cv2.INTER_AREA)
                right = cv2.resize(right, (args.width, args.height), interpolation=cv2.INTER_AREA)

            depth, stereo_ms = _timed(lambda: depth_estimator.depth(left, right))
            pose, pose_ms = _timed(
                lambda: pose_estimator.track(left, depth, depth_estimator.K)
            )
            if pose is None:
                raise RuntimeError(f"FoundationPose tracking failed at rollout frame {index}")

            prepared, policy_pre_ms = _timed(
                lambda: _prepare_policy_inputs(
                    policy,
                    left,
                    depth,
                    depth_estimator.K,
                    pose,
                    base_from_camera,
                )
            )
            sparse_tensor, current_obj = prepared
            outputs, policy_infer_ms = _timed(
                lambda: _policy_forward(policy, sparse_tensor, current_obj)
            )
            trajectory, policy_post_ms = _timed(lambda: _decode_policy(outputs))
            _, planning_ms = _timed(
                lambda: _motion_plan_postprocess(
                    trajectory,
                    pose,
                    base_from_camera,
                    current_tcp_pose,
                    args.motion_steps,
                )
            )
            _cuda_sync()
            rollout_ms = (time.perf_counter() - rollout_start) * 1000.0

            if index >= args.warmup:
                samples.append(
                    {
                        "frame": index - args.warmup,
                        "camera_capture": camera_ms,
                        "foundation_stereo": stereo_ms,
                        "foundation_pose": pose_ms,
                        "policy_preprocess": policy_pre_ms,
                        "policy_inference": policy_infer_ms,
                        "policy_postprocess": policy_post_ms,
                        "motion_planning": planning_ms,
                        "rollout_total": rollout_ms,
                    }
                )
    finally:
        camera.close()
        cv2.destroyAllWindows()

    summary = {
        stage: _stats([sample[stage] for sample in samples]) for stage in STAGES
    }
    summary["policy_total"] = _stats(
        [
            sample["policy_preprocess"]
            + sample["policy_inference"]
            + sample["policy_postprocess"]
            for sample in samples
        ]
    )
    summary["pipeline_compute"] = _stats(
        [sample["rollout_total"] - sample["camera_capture"] for sample in samples]
    )
    metadata = {
        "frames": args.frames,
        "warmup": args.warmup,
        "image_size": [args.width, args.height],
        "camera": {"resolution": args.resolution, "requested_fps": args.camera_fps},
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "stereo": {
            "checkpoint": str(depth_estimator.ckpt_path),
            "iters": args.stereo_iters,
            "max_disp": args.stereo_max_disp,
            "volume_backend": args.stereo_volume_backend,
        },
        "foundation_pose": {
            "register_iters": args.pose_register_iters,
            "track_iters": args.pose_track_iters,
        },
        "motion_planning_definition": "trajectory-to-pose7 conversion; no collision planner or robot I/O",
    }

    csv_path = args.output_dir / "per_frame.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", *STAGES])
        writer.writeheader()
        writer.writerows(samples)
    report_path = args.output_dir / "summary.json"
    report_path.write_text(
        json.dumps(
            {"metadata": metadata, "initialization": initialization, "summary": summary},
            indent=2,
        )
        + "\n"
    )

    print("\nOne-time initialization")
    for name, value in initialization.items():
        print(f"{name:36s} {value:9.2f}ms")
    _print_summary(summary, args.frames)
    print(f"raw samples: {csv_path}")
    print(f"summary:     {report_path}")


if __name__ == "__main__":
    run(parse_args())
