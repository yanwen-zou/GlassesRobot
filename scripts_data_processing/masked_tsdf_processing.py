import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

try:
    from open3d.pipelines import integration as o3d_integration
except ImportError:  # pragma: no cover
    o3d_integration = o3d.integration  # type: ignore[attr-defined]


EPISODE_DIR = Path("data/20251112_142342")
RGB_DIR = EPISODE_DIR / "rgb"
DEPTH_DIR = EPISODE_DIR / "depth"
HEAD_POS_DIR = EPISODE_DIR / "head_pos"
MASK_DIR = EPISODE_DIR / "masks"
HAND_MASK_DIR = EPISODE_DIR / "mask_hand"
INTRINSIC_PATH = EPISODE_DIR / "cam_K.txt"
OUTPUT_DIR = EPISODE_DIR / "tsdf_reconstruction"

VOXEL_SIZE = 0.01  # meters
TRUNC_MARGIN = 0.04  # meters
DEPTH_SCALE = 1000.0
DEPTH_TRUNC = 3.0  # meters
FRAME_STRIDE = 5
HEAD_POSE_PRIOR_WEIGHT = 1000.0


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm == 0:
        return np.eye(3, dtype=np.float32)
    x, y, z, w = q / norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    rot = np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float32)
    return rot


def load_head_poses(head_dir: Path) -> dict[int, np.ndarray]:
    pose_files = sorted(head_dir.glob("*.txt"), key=lambda p: int(p.stem))
    if not pose_files:
        raise FileNotFoundError(f"No pose files in {head_dir}")
    pose_map: dict[int, np.ndarray] = {}
    for path in pose_files:
        values = np.loadtxt(path, dtype=np.float32).reshape(-1)
        if values.size < 7:
            raise ValueError(f"Pose file {path} must contain tx ty tz qx qy qz qw.")
        t = values[:3]
        q = values[3:7]
        rot = quaternion_to_matrix(q)
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = rot
        mat[:3, 3] = t
        pose_map[int(path.stem)] = mat
    first_key = min(pose_map.keys())
    ref_inv = np.linalg.inv(pose_map[first_key])
    return {k: ref_inv @ pose for k, pose in pose_map.items()}


def sorted_png_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing directory: {directory}")
    paths = sorted(
        [p for p in directory.iterdir() if p.suffix.lower() == ".png"],
        key=lambda p: int(p.stem)
    )
    if not paths:
        raise FileNotFoundError(f"No PNG files in {directory}")
    return paths


def create_rgbd_from_arrays(color_np: np.ndarray, depth_m: np.ndarray) -> o3d.geometry.RGBDImage:
    color_o3d = o3d.geometry.Image(color_np.astype(np.uint8))
    depth_o3d = o3d.geometry.Image(depth_m.astype(np.float32))
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1.0,
        depth_trunc=DEPTH_TRUNC,
        convert_rgb_to_intensity=False,
    )


def preprocess_frames(color_paths, depth_paths, masks, hand_masks, height, width):
    entries = []
    for idx, (color_path, depth_path) in enumerate(zip(color_paths, depth_paths)):
        if idx % FRAME_STRIDE != 0:
            continue
        frame_id = int(color_path.stem)

        color_img = Image.open(color_path).convert("RGB")
        depth_raw = np.array(Image.open(depth_path)).astype(np.float32)
        depth_m = depth_raw / DEPTH_SCALE

        mask = np.zeros((height, width), dtype=bool)
        mask_path = masks.get(frame_id)
        if mask_path is not None:
            mask |= np.array(Image.open(mask_path).convert("L")) > 0
        hand_mask_path = hand_masks.get(frame_id)
        if hand_mask_path is not None:
            mask |= np.array(Image.open(hand_mask_path).convert("L")) > 0

        depth_m[mask] = 0.0
        color_np = np.array(color_img)
        color_np[mask] = 0

        if not np.any(depth_m > 0):
            continue

        entries.append(
            {
                "frame_id": frame_id,
                "color": color_np,
                "depth": depth_m,
            }
        )
    return entries


def refine_poses_with_ba(entries, poses, intrinsic):
    if len(entries) < 2:
        return poses

    rgbd_images = [create_rgbd_from_arrays(e["color"], e["depth"]) for e in entries]
    frame_ids = [e["frame_id"] for e in entries]

    pose_graph = o3d.pipelines.registration.PoseGraph()
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(poses[frame_ids[0]])))

    odo_method = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
    odo_option = o3d.pipelines.odometry.OdometryOption()
    try:
        odo_option.min_depth = 0.1
        odo_option.max_depth = DEPTH_TRUNC
    except AttributeError:
        # older Open3D versions use depth_min/depth_max
        odo_option.depth_min = 0.1  # type: ignore[attr-defined]
        odo_option.depth_max = DEPTH_TRUNC  # type: ignore[attr-defined]

    prior_info = np.eye(6, dtype=np.float64) * HEAD_POSE_PRIOR_WEIGHT

    for i in range(1, len(entries)):
        curr_id = frame_ids[i]
        prev_id = frame_ids[i - 1]
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(poses[curr_id])))

        head_rel = np.linalg.inv(poses[prev_id]) @ poses[curr_id]
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                i - 1,
                i,
                head_rel,
                prior_info,
                uncertain=False,
            )
        )

        success, odo_trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
            rgbd_images[i - 1],
            rgbd_images[i],
            intrinsic,
            np.eye(4),
            odo_method,
            odo_option,
        )
        if success:
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    i - 1,
                    i,
                    odo_trans,
                    info,
                    uncertain=True,
                )
            )

    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=0.05,
        edge_prune_threshold=0.25,
        reference_node=0,
    )
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option,
    )

    refined = poses.copy()
    for node, frame_id in zip(pose_graph.nodes, frame_ids):
        refined[frame_id] = np.linalg.inv(node.pose)
    return refined


def main():
    color_paths = sorted_png_paths(RGB_DIR)
    depth_paths = sorted_png_paths(DEPTH_DIR)
    poses = load_head_poses(HEAD_POS_DIR)

    mask_map = {int(p.stem): p for p in sorted_png_paths(MASK_DIR)} if MASK_DIR.exists() else {}
    hand_mask_map = {int(p.stem): p for p in sorted_png_paths(HAND_MASK_DIR)} if HAND_MASK_DIR.exists() else {}

    with open(INTRINSIC_PATH, "r", encoding="utf-8") as f:
        intr_values = [list(map(float, line.split())) for line in f if line.strip()]
    K = np.array(intr_values, dtype=np.float32)

    sample_color = np.array(Image.open(color_paths[0]).convert("RGB"))
    height, width = sample_color.shape[:2]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, K[0, 0], K[1, 1], K[0, 2], K[1, 2])

    entries = preprocess_frames(color_paths, depth_paths, mask_map, hand_mask_map, height, width)
    if not entries:
        raise RuntimeError("No valid frames available after preprocessing.")

    refined_poses = refine_poses_with_ba(entries, poses, intrinsic)

    volume = o3d_integration.ScalableTSDFVolume(
        voxel_length=VOXEL_SIZE,
        sdf_trunc=TRUNC_MARGIN,
        color_type=o3d_integration.TSDFVolumeColorType.RGB8,
    )

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    used_frames: list[int] = []

    for entry in entries:
        frame_id = entry["frame_id"]
        rgbd = create_rgbd_from_arrays(entry["color"], entry["depth"])

        if frame_id not in refined_poses:
            print(f"[WARN] Missing refined pose for frame {frame_id}, skipping.")
            continue

        extrinsic = np.linalg.inv(refined_poses[frame_id])
        volume.integrate(rgbd, intrinsic, extrinsic)
        used_frames.append(frame_id)
        if len(used_frames) % 25 == 0:
            print(f"[INFO] Integrated {len(used_frames)} frames...")

    if not used_frames:
        raise RuntimeError("No frames were integrated; aborting TSDF extraction.")

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    mesh_path = OUTPUT_DIR / "tsdf_mesh.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)

    pcd = volume.extract_point_cloud()
    pcd_path = OUTPUT_DIR / "tsdf_pointcloud.ply"
    o3d.io.write_point_cloud(str(pcd_path), pcd)

    meta = {
        "episode": str(EPISODE_DIR),
        "num_frames": len(used_frames),
        "frames": sorted(used_frames),
        "voxel_size_m": VOXEL_SIZE,
        "trunc_margin_m": TRUNC_MARGIN,
        "depth_trunc_m": DEPTH_TRUNC,
        "depth_scale": DEPTH_SCALE,
        "frame_stride": FRAME_STRIDE,
        "ba_refined": True,
    }
    with open(OUTPUT_DIR / "tsdf_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[DONE] Saved TSDF mesh to {mesh_path}")
    print(f"[DONE] Saved TSDF point cloud to {pcd_path}")


if __name__ == "__main__":
    main()
