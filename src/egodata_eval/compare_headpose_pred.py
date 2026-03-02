#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np


def _to_seq_list(arr: np.ndarray) -> List[Optional[np.ndarray]]:
    if arr.dtype == object:
        out: List[Optional[np.ndarray]] = []
        for item in arr:
            if item is None:
                out.append(None)
            else:
                x = np.asarray(item, dtype=np.float32)
                out.append(x if x.size > 0 else None)
        return out
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim == 3:
        return [x[i] for i in range(x.shape[0])]
    raise ValueError(f"Unsupported headpose array shape: {x.shape}, dtype={arr.dtype}")


def _load_train_headpose(path: Path) -> List[Optional[np.ndarray]]:
    arr = np.load(path, allow_pickle=True)
    return _to_seq_list(arr)


def _load_eval_headpose_from_records(path: Path, key: str) -> List[Optional[np.ndarray]]:
    recs = np.load(path, allow_pickle=True)
    out: List[Optional[np.ndarray]] = []
    for rec in recs:
        d = dict(rec)
        item = d.get(key, None)
        if item is None:
            out.append(None)
            continue
        x = np.asarray(item, dtype=np.float32)
        out.append(x if x.size > 0 else None)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visual-compare train headpose_pred.npy and eval robot_pose_records headpose_pred in rerun."
    )
    parser.add_argument(
        "--train-headpose",
        type=Path,
        default=Path("src/egodata_eval/train_output/episode/20260227_144020_20260301_205439/headpose_pred.npy"),
        help="Path to train headpose_pred.npy",
    )
    parser.add_argument(
        "--eval-records",
        type=Path,
        default=Path("src/egodata_eval/eval_output/20260302_140206/robot_pose_records.npy"),
        help="Path to eval robot_pose_records.npy",
    )
    parser.add_argument(
        "--eval-key",
        type=str,
        default="headpose_pred",
        help="Record field key to compare against train headpose_pred.npy",
    )
    parser.add_argument("--spawn", action="store_true", help="Spawn rerun viewer.")
    parser.add_argument("--point-radius", type=float, default=0.006, help="Point radius in rerun.")
    args = parser.parse_args()

    if not args.train_headpose.exists():
        raise FileNotFoundError(f"Train headpose file not found: {args.train_headpose}")
    if not args.eval_records.exists():
        raise FileNotFoundError(f"Eval records file not found: {args.eval_records}")

    train_seq = _load_train_headpose(args.train_headpose)
    eval_seq = _load_eval_headpose_from_records(args.eval_records, key=args.eval_key)

    n_train = len(train_seq)
    n_eval = len(eval_seq)
    n_cmp = min(n_train, n_eval)
    if n_cmp == 0:
        raise RuntimeError("No comparable frames.")

    try:
        import rerun as rr
    except Exception as exc:
        raise RuntimeError("rerun package required. `pip install rerun-sdk`.") from exc

    rr.init("headpose_pred_compare", spawn=args.spawn)
    rr.log("world", rr.ViewCoordinates.FRU)

    frame_mae: List[float] = []
    frame_max: List[float] = []
    compared_frames = 0
    skipped_frames = 0
    total_steps = 0

    for i in range(n_cmp):
        rr.set_time_sequence("frame", i)
        a = train_seq[i]
        b = eval_seq[i]
        if a is None or b is None:
            skipped_frames += 1
            rr.log("compare/train/points", rr.Clear(recursive=True))
            rr.log("compare/eval/points", rr.Clear(recursive=True))
            continue
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] < 9 or b.shape[1] < 9:
            skipped_frames += 1
            rr.log("compare/train/points", rr.Clear(recursive=True))
            rr.log("compare/eval/points", rr.Clear(recursive=True))
            continue
        steps = min(a.shape[0], b.shape[0])
        if steps <= 0:
            skipped_frames += 1
            rr.log("compare/train/points", rr.Clear(recursive=True))
            rr.log("compare/eval/points", rr.Clear(recursive=True))
            continue
        da = a[:steps, :9]
        db = b[:steps, :9]
        diff = np.abs(da - db)
        frame_mae.append(float(diff.mean()))
        frame_max.append(float(diff.max()))
        compared_frames += 1
        total_steps += steps

        train_pts = da[:, :3].astype(np.float32)
        eval_pts = db[:, :3].astype(np.float32)
        rr.log(
            "compare/train/points",
            rr.Points3D(
                positions=train_pts,
                colors=np.array([[255, 180, 0, 255]], dtype=np.uint8),
                radii=np.full(train_pts.shape[0], args.point_radius, dtype=np.float32),
            ),
        )
        rr.log(
            "compare/train/line",
            rr.LineStrips3D(
                strips=[train_pts],
                colors=np.array([[255, 180, 0, 255]], dtype=np.uint8),
                radii=np.array([args.point_radius * 0.55], dtype=np.float32),
            ),
        )
        rr.log(
            "compare/eval/points",
            rr.Points3D(
                positions=eval_pts,
                colors=np.array([[0, 220, 255, 255]], dtype=np.uint8),
                radii=np.full(eval_pts.shape[0], args.point_radius, dtype=np.float32),
            ),
        )
        rr.log(
            "compare/eval/line",
            rr.LineStrips3D(
                strips=[eval_pts],
                colors=np.array([[0, 220, 255, 255]], dtype=np.uint8),
                radii=np.array([args.point_radius * 0.55], dtype=np.float32),
            ),
        )

    print(f"train frames: {n_train}")
    print(f"eval frames: {n_eval}")
    print(f"compared frames (index-aligned): {compared_frames} / {n_cmp}")
    print(f"skipped frames: {skipped_frames}")
    print(f"total compared steps: {total_steps}")

    if compared_frames == 0:
        print("No valid frame pairs to compare.")
        return

    mae_arr = np.asarray(frame_mae, dtype=np.float32)
    max_arr = np.asarray(frame_max, dtype=np.float32)
    print(f"frame MAE mean: {mae_arr.mean():.6f}")
    print(f"frame MAE median: {np.median(mae_arr):.6f}")
    print(f"frame MAE max: {mae_arr.max():.6f}")
    print(f"frame abs-max mean: {max_arr.mean():.6f}")
    print(f"frame abs-max max: {max_arr.max():.6f}")


if __name__ == "__main__":
    main()
