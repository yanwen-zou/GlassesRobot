#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glasses_hardware.hardware.my_device.zed import ZEDCamera  # type: ignore


def read_zed_intrinsics(zed: ZEDCamera) -> list[list[float]]:
    cam_info = zed._zed.get_camera_information()
    calib = cam_info.camera_configuration.calibration_parameters
    left = calib.left_cam
    return [
        [float(left.fx), 0.0, float(left.cx)],
        [0.0, float(left.fy), float(left.cy)],
        [0.0, 0.0, 1.0],
    ]


def read_zed_baseline(zed: ZEDCamera) -> float | None:
    cam_info = zed._zed.get_camera_information()
    calib = cam_info.camera_configuration.calibration_parameters
    for attr in ("baseline", "stereo_baseline", "camera_baseline"):
        if hasattr(calib, attr):
            try:
                return float(getattr(calib, attr))
            except (TypeError, ValueError):
                pass
    stereo = getattr(calib, "stereo_transform", None)
    if stereo is not None and hasattr(stereo, "get_translation"):
        try:
            translation = stereo.get_translation()
            if hasattr(translation, "get"):
                translation = translation.get()
            if translation is not None:
                return abs(float(translation[0]))
        except (TypeError, ValueError, IndexError):
            pass
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Read ZED intrinsics and print 3x3 K on one line (baseline on second line)."
    )
    ap.add_argument("--resolution", default="WVGA", help="ZED resolution: 2K/1080P/720P/WVGA")
    ap.add_argument("--fps", type=int, default=30, help="Target FPS")
    ap.add_argument(
        "--write-path",
        default=str(REPO_ROOT / "src" / "FoundationStereo" / "assets" / "K_ZED.txt"),
        help="Path to write K (first line) before printing.",
    )
    args = ap.parse_args()

    zed = None
    try:
        zed = ZEDCamera(resolution=args.resolution, fps=args.fps)
        K = read_zed_intrinsics(zed)
        baseline = read_zed_baseline(zed) / 1000
    finally:
        if zed is not None:
            zed.close()

    flat = [v for row in K for v in row]
    if baseline is None:
        baseline = 0.0
        print("[WARN] Failed to read ZED baseline; writing 0.0", file=sys.stderr)
    try:
        with open(args.write_path, "w", encoding="utf-8") as f:
            f.write(" ".join(f"{v:.6f}" for v in flat) + "\n")
            f.write(f"{baseline:.6f}\n")
    except OSError as exc:
        print(f"[WARN] Failed to write intrinsics to {args.write_path}: {exc}", file=sys.stderr)
    print(" ".join(f"{v:.6f}" for v in flat))
    print(f"{baseline:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
