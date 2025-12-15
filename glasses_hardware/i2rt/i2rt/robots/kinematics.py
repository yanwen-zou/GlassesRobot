import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

# Ensure local PyRoki sources and snippets are importable when running as a script.
here = Path(__file__).resolve()
repo_root = here.parents[4]  # unity_comm/
gh_root = repo_root / "glasses_hardware" / "i2rt"
pyroki_src = gh_root / "pyroki" / "src"
snippets_root = gh_root / "pyroki_snippets"
for path in (pyroki_src, snippets_root, gh_root, repo_root):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import jax.numpy as jnp
import numpy as np
import pyroki as pk
import pyroki_snippets as pks
import viser
import yourdfpy
from scipy.spatial.transform import Rotation as R
from viser.extras import ViserUrdf


def _load_urdf_with_assets(xml_path: Path) -> yourdfpy.URDF:
    """Load URDF (xml->urdf) resolving package://assets to the local asset folder."""
    urdf_path = xml_path.with_suffix(".urdf")
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found for IK: {urdf_path}")
    assets_root = urdf_path.parent

    def filename_handler(fname: str, **kwargs) -> str:
        if fname.startswith("package://"):
            rel = fname[len("package://") :]
            resolved = (assets_root / rel).resolve()
            return str(resolved)
        return fname

    return yourdfpy.URDF.load(str(urdf_path), filename_handler=filename_handler)


class Kinematics:
    def __init__(self, xml_path: str, link_name: Optional[str]):
        """Initialize PyRoki-based kinematics for a target link."""
        self._xml_path = Path(xml_path)

        # Locate URDF next to the XML (yam.xml -> yam.urdf).
        urdf = _load_urdf_with_assets(self._xml_path)
        self._robot = pk.Robot.from_urdf(urdf)
        self._link_indices: Dict[str, int] = {name: idx for idx, name in enumerate(self._robot.links.names)}
        self._target_link = link_name or self._robot.links.names[-1]

        # Default joint configuration (mid-limits).
        self._default_q = np.asarray((self._robot.joints.lower_limits + self._robot.joints.upper_limits) / 2.0)

    @staticmethod
    def _wxyz_pos_to_matrix(wxyz: np.ndarray, pos: np.ndarray) -> np.ndarray:
        rot = R.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = rot.as_matrix().astype(np.float32)
        T[:3, 3] = pos.astype(np.float32)
        return T

    @staticmethod
    def _matrix_to_wxyz_pos(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rot = R.from_matrix(T[:3, :3])
        quat_xyzw = rot.as_quat()
        wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
        pos = T[:3, 3].astype(np.float32)
        return wxyz, pos

    def _fk_link(self, q: np.ndarray, link_name: str) -> np.ndarray:
        if link_name not in self._link_indices:
            raise ValueError(f"Unknown link '{link_name}'.")
        idx = self._link_indices[link_name]
        poses = self._robot.forward_kinematics(jnp.asarray(q, dtype=jnp.float32))
        wxyz_xyz = np.array(poses[idx])
        wxyz, pos = wxyz_xyz[:4], wxyz_xyz[4:]
        return self._wxyz_pos_to_matrix(wxyz, pos)

    def fk(self, q: np.ndarray, link_name: Optional[str] = None) -> np.ndarray:
        """Forward kinematics for the requested link (or default)."""
        target_link = link_name or self._target_link
        return self._fk_link(q, target_link)

    def ik(
        self,
        target_pose: np.ndarray,
        link_name: Optional[str],
        init_q: Optional[np.ndarray] = None,
        limits: Optional[List[object]] = None,  # retained for signature compatibility
        dt: float = 0.01,
        solver: str = "quadprog",
        pos_threshold: float = 1e-4,
        ori_threshold: float = 1e-4,
        damping: float = 1e-4,
        max_iters: int = 200,
        verbose: bool = False,
    ) -> Tuple[bool, np.ndarray]:
        """Inverse kinematics solved with PyRoki snippets for a target link."""
        del limits, dt, solver, damping, max_iters  # unused in PyRoki path
        target_link = link_name or self._target_link
        target_pos = np.asarray(target_pose[:3, 3], dtype=np.float32)
        target_rot = R.from_matrix(target_pose[:3, :3])
        quat_xyzw = target_rot.as_quat()
        target_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
        start_time = time.time()
        q_sol = pks.solve_ik(
            robot=self._robot,
            target_link_name=target_link,
            target_position=target_pos,
            target_wxyz=target_wxyz,
        )
        # Evaluate error at the solved pose.
        T_world_link_sol = self.fk(q_sol, target_link)
        pos_err = np.linalg.norm(T_world_link_sol[:3, 3] - target_pos)
        rot_current = R.from_matrix(T_world_link_sol[:3, :3])
        rot_err = target_rot.inv() * rot_current
        ori_err = np.linalg.norm(rot_err.as_rotvec())
        success = pos_err <= pos_threshold and ori_err <= ori_threshold
        if verbose:
            elapsed = time.time() - start_time
            print(
                f"[PyRoki IK] success={success}, pos_err={pos_err:.2e}, ori_err={ori_err:.2e}, "
                f"time={elapsed:.4f}s, q={q_sol}"
            )
        return success, q_sol


def main() -> None:
    """Launch interactive IK visualization."""
    from i2rt.robots.utils import YAM_XML_PATH

    xml_path = Path(YAM_XML_PATH)
    urdf = _load_urdf_with_assets(xml_path)
    target_link_name = "link_6"

    robot = pk.Robot.from_urdf(urdf)
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    ik_target = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.3, 0.0, 0.5), wxyz=(0, 0, 1, 0)
    )
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
    model = Kinematics(YAM_XML_PATH, "link_6")

    def _handle_to_pose(handle) -> np.ndarray:
        pos = np.array(handle.position, dtype=np.float32)
        wxyz = np.array(handle.wxyz, dtype=np.float32)
        rot = R.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = rot.as_matrix().astype(np.float32)
        T[:3, 3] = pos
        return T

    while True:
        start_time = time.time()
        target_pose = _handle_to_pose(ik_target)
        _, solution = model.ik(target_pose, "link_6", verbose=True)
        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)
        urdf_vis.update_cfg(solution)


if __name__ == "__main__":
    main()
