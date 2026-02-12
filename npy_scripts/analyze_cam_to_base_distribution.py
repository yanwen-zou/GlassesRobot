#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
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
            raise ValueError(f"Unsupported object npy structure: {path}")
    else:
        arr = np.asarray(obj)

    if arr.ndim == 3 and arr.shape[1:] == (4, 4):
        return arr[0].astype(np.float64)
    if arr.ndim == 2 and arr.shape == (4, 4):
        return arr.astype(np.float64)
    raise ValueError(f"Unsupported npy transform shape {arr.shape} in {path}")


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
            r = np.array(vals[:9], dtype=np.float64).reshape(3, 3)
            t = np.array(vals[9:], dtype=np.float64)
            tf = np.eye(4, dtype=np.float64)
            tf[:3, :3] = r
            tf[:3, 3] = t
            return tf
    raise ValueError(f"No valid transform row in {path}")


def load_first_transform(ts_dir: Path) -> tuple[np.ndarray, str]:
    npy = ts_dir / "cam_to_base.npy"
    txt = ts_dir / "cam_to_base.txt"
    if npy.exists():
        return load_first_transform_from_npy(npy), str(npy)
    if txt.exists():
        return load_first_transform_from_txt(txt), str(txt)
    raise FileNotFoundError(f"No cam_to_base.npy/txt in {ts_dir}")


def rotation_angle_deg(r: np.ndarray) -> float:
    tr = np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))


def summarize(values: np.ndarray) -> dict:
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "max": float(values.max()),
    }


def mean_rotation_matrix(rotations: np.ndarray) -> np.ndarray:
    """Compute a valid mean rotation by projecting element-wise mean onto SO(3)."""
    r_mean_raw = rotations.mean(axis=0)
    u, _, vh = np.linalg.svd(r_mean_raw)
    r_mean = u @ vh
    if np.linalg.det(r_mean) < 0:
        u[:, -1] *= -1.0
        r_mean = u @ vh
    return r_mean


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze first cam_to_base transforms in timestamp-named folders."
    )
    parser.add_argument("data_dir", type=Path, help="Root data folder")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cam_to_base_distribution"),
        help="Directory to save plots and summary files",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ts_dirs = find_timestamp_dirs(data_dir)
    transforms = []
    records = []
    for d in ts_dirs:
        try:
            tf, src = load_first_transform(d)
            transforms.append(tf)
            records.append((d, src))
        except Exception:
            continue

    if not transforms:
        raise RuntimeError(f"No usable cam_to_base transforms found under {data_dir}")

    tfs = np.stack(transforms, axis=0)
    ref_tf = tfs[0]
    ref_inv = np.linalg.inv(ref_tf)

    rel_t_norm = []
    rel_r_deg = []
    abs_t = tfs[:, :3, 3]
    rel_t_xyz = []
    for tf in tfs:
        rel = ref_inv @ tf
        rel_t = rel[:3, 3]
        rel_r = rel[:3, :3]
        rel_t_xyz.append(rel_t)
        rel_t_norm.append(np.linalg.norm(rel_t))
        rel_r_deg.append(rotation_angle_deg(rel_r))

    rel_t_xyz = np.stack(rel_t_xyz, axis=0)
    rel_t_norm = np.asarray(rel_t_norm)
    rel_r_deg = np.asarray(rel_r_deg)
    abs_r = tfs[:, :3, :3]
    r_mean = mean_rotation_matrix(abs_r)
    r_std = abs_r.std(axis=0)
    t_mean = abs_t.mean(axis=0)
    t_std = abs_t.std(axis=0)
    tf_mean = np.eye(4, dtype=np.float64)
    tf_mean[:3, :3] = r_mean
    tf_mean[:3, 3] = t_mean
    tf_std = tfs.std(axis=0)

    summary = {
        "data_dir": str(data_dir),
        "num_timestamp_dirs_found": len(ts_dirs),
        "num_valid_transforms": int(len(transforms)),
        "reference_dir": str(records[0][0]),
        "reference_source": records[0][1],
        "relative_translation_norm_m": summarize(rel_t_norm),
        "relative_rotation_angle_deg": summarize(rel_r_deg),
        "absolute_translation_xyz_m": {
            "x": summarize(abs_t[:, 0]),
            "y": summarize(abs_t[:, 1]),
            "z": summarize(abs_t[:, 2]),
        },
        "absolute_rotation_matrix_mean": r_mean.tolist(),
        "absolute_rotation_matrix_std_elementwise": r_std.tolist(),
        "absolute_transform_4x4_mean": tf_mean.tolist(),
        "absolute_transform_4x4_std_elementwise": tf_std.tolist(),
        "absolute_translation_mean_xyz_m": t_mean.tolist(),
        "absolute_translation_std_xyz_m": t_std.tolist(),
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with (out_dir / "transforms_index.csv").open("w", encoding="utf-8") as f:
        f.write("index,timestamp_dir,source_path\n")
        for i, (d, src) in enumerate(records):
            f.write(f"{i},{d},{src}\n")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].hist(rel_r_deg, bins=30, color="#1f77b4", alpha=0.85)
    axes[0, 0].set_title("Relative Rotation Angle (deg)")
    axes[0, 0].set_xlabel("deg")
    axes[0, 0].set_ylabel("count")

    axes[0, 1].hist(rel_t_norm, bins=30, color="#ff7f0e", alpha=0.85)
    axes[0, 1].set_title("Relative Translation Norm (m)")
    axes[0, 1].set_xlabel("m")
    axes[0, 1].set_ylabel("count")

    axes[1, 0].scatter(abs_t[:, 0], abs_t[:, 1], c=abs_t[:, 2], s=18, cmap="viridis")
    axes[1, 0].set_title("Absolute Translation XY (color=Z)")
    axes[1, 0].set_xlabel("tx (m)")
    axes[1, 0].set_ylabel("ty (m)")

    axes[1, 1].boxplot(
        [rel_t_xyz[:, 0], rel_t_xyz[:, 1], rel_t_xyz[:, 2]],
        tick_labels=["dx", "dy", "dz"],
    )
    axes[1, 1].set_title("Relative Translation Components (m)")
    axes[1, 1].set_ylabel("m")

    fig.suptitle("cam_to_base First-Transform Distribution")
    fig.tight_layout()
    fig.savefig(out_dir / "distribution.png", dpi=180)
    plt.close(fig)

    print(f"Saved summary: {out_dir / 'summary.json'}")
    print(f"Saved index:   {out_dir / 'transforms_index.csv'}")
    print(f"Saved figure:  {out_dir / 'distribution.png'}")
    print(f"Valid transforms: {len(transforms)}")
    print(f"Reference: {records[0][0]}")


if __name__ == "__main__":
    main()
