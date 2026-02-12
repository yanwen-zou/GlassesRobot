#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np


TIMESTAMP_DIR_RE = re.compile(r"^\d{8}_\d{6}$")


def find_timestamp_dirs(root: Path) -> list[Path]:
    dirs = [p for p in root.rglob("*") if p.is_dir() and TIMESTAMP_DIR_RE.match(p.name)]
    return sorted(dirs)


def load_first_transform_from_npy(path: Path) -> np.ndarray:
    obj = np.load(path, allow_pickle=True)
    if obj.shape == () and obj.dtype == object:
        content = obj.item()
        if isinstance(content, dict) and "transforms" in content:
            arr = np.asarray(content["transforms"])
        else:
            raise ValueError(f"Unsupported npy object structure: {path}")
    else:
        arr = np.asarray(obj)

    if arr.ndim == 3 and arr.shape[1:] == (4, 4):
        return arr[0].astype(np.float32)
    if arr.ndim == 2 and arr.shape == (4, 4):
        return arr.astype(np.float32)
    raise ValueError(f"Unsupported npy transform shape {arr.shape}: {path}")


def load_first_transform_from_txt(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if parts[0] == "frame_id":
                continue
            if len(parts) < 13:
                continue
            vals = [float(x) for x in parts[1:13]]
            r = np.array(vals[:9], dtype=np.float32).reshape(3, 3)
            t = np.array(vals[9:], dtype=np.float32)
            tf = np.eye(4, dtype=np.float32)
            tf[:3, :3] = r
            tf[:3, 3] = t
            return tf
    raise ValueError(f"No valid transform row in {path}")


def load_first_transform(ts_dir: Path) -> tuple[np.ndarray, Path]:
    npy_path = ts_dir / "cam_to_base.npy"
    txt_path = ts_dir / "cam_to_base.txt"
    if npy_path.exists():
        return load_first_transform_from_npy(npy_path), npy_path
    if txt_path.exists():
        return load_first_transform_from_txt(txt_path), txt_path
    raise FileNotFoundError(f"No cam_to_base.npy/txt in {ts_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize initial cam_to_base poses in timestamp folders using Rerun."
    )
    parser.add_argument("data_dir", type=Path, help="Root directory containing timestamp-named folders")
    parser.add_argument("--axis-len", type=float, default=0.08, help="Coordinate frame axis length (meters)")
    parser.add_argument("--spawn", action="store_true", help="Spawn Rerun viewer window")
    parser.add_argument("--save", type=Path, default=None, help="Optional output .rrd file path")
    args = parser.parse_args()

    try:
        import rerun as rr
    except Exception as exc:
        raise RuntimeError("rerun-sdk is required. Install with `pip install rerun-sdk`.") from exc

    data_dir = args.data_dir.resolve()
    ts_dirs = find_timestamp_dirs(data_dir)
    if not ts_dirs:
        raise RuntimeError(f"No timestamp-named folders found under: {data_dir}")

    rr.init(f"Initial cam_to_base poses ({data_dir.name})", spawn=args.spawn)
    if args.save is not None:
        rr.save(str(args.save.resolve()))

    try:
        rr.log("world", rr.ViewCoordinates.FRU)
    except Exception:
        pass

    rr.log(
        "world/base/axes",
        rr.Arrows3D(
            origins=np.zeros((3, 3), dtype=np.float32),
            vectors=np.eye(3, dtype=np.float32) * args.axis_len,
            colors=np.array([[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]], dtype=np.uint8),
            radii=np.full(3, args.axis_len * 0.05, dtype=np.float32),
        ),
    )

    origins = []
    valid_count = 0
    for idx, ts_dir in enumerate(ts_dirs):
        try:
            tf, src = load_first_transform(ts_dir)
        except Exception:
            continue

        valid_count += 1
        name = ts_dir.name
        rr.set_time_sequence("sample_idx", idx)
        rr.log(
            f"world/cam_poses/{name}",
            rr.Transform3D(
                translation=tf[:3, 3],
                mat3x3=tf[:3, :3],
            ),
        )
        rr.log(
            f"world/cam_poses/{name}/axes",
            rr.Arrows3D(
                origins=np.zeros((3, 3), dtype=np.float32),
                vectors=np.eye(3, dtype=np.float32) * args.axis_len,
                colors=np.array([[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]], dtype=np.uint8),
                radii=np.full(3, args.axis_len * 0.05, dtype=np.float32),
            ),
        )
        rr.log(f"world/cam_poses/{name}/source", rr.TextLog(str(src)))
        origins.append(tf[:3, 3])

    if not origins:
        raise RuntimeError(f"No valid cam_to_base transforms found under: {data_dir}")

    origins_arr = np.asarray(origins, dtype=np.float32)
    rr.log(
        "world/cam_origins",
        rr.Points3D(
            positions=origins_arr,
            colors=np.tile(np.array([[255, 255, 0, 255]], dtype=np.uint8), (origins_arr.shape[0], 1)),
            radii=0.01,
        ),
    )

    print(f"data_dir: {data_dir}")
    print(f"timestamp_dirs: {len(ts_dirs)}")
    print(f"valid_initial_poses: {valid_count}")
    if args.save is not None:
        print(f"saved_rrd: {args.save.resolve()}")
    else:
        print("No .rrd file saved. Use --save to export.")


if __name__ == "__main__":
    main()
