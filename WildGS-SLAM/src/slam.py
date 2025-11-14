import os
import torch
import numpy as np
import torch.multiprocessing as mp
from munch import munchify
from scipy.spatial.transform import Rotation

from src.depth_video import DepthVideo
from src.utils.common import setup_seed, update_cam
from src.utils.Printer import Printer, FontColor
from src.utils.eval_traj import kf_traj_eval
from src.utils.datasets import BaseDataset, RGB_NoPose
from src.mapper import Mapper
from src.utils.dyn_uncertainty.uncertainty_model import generate_uncertainty_mlp
from src.gui import gui_utils, slam_gui
from thirdparty.gaussian_splatting.scene.gaussian_model import GaussianModel

def matrix4x4_to_se3(w2c: torch.Tensor) -> torch.Tensor:
    """
    Convert a 4x4 world-to-camera matrix to the [tx, ty, tz, qx, qy, qz, qw] format
    expected by lietorch.SE3.
    """
    if w2c.shape != (4, 4):
        raise ValueError("Pose matrix must be 4x4.")

    rot = w2c[:3, :3].cpu().numpy()
    trans = w2c[:3, 3].cpu().numpy()
    quat = Rotation.from_matrix(rot).as_quat()  # x, y, z, w
    data = np.concatenate([trans, quat], axis=0)
    return torch.from_numpy(data).float()


class SLAM:
    def __init__(self, cfg, stream: BaseDataset):
        super(SLAM, self).__init__()
        self.cfg = cfg
        self.device = cfg["device"]
        self.verbose: bool = cfg["verbose"]
        self.logger = None
        self.save_dir = cfg["data"]["output"] + "/" + cfg["scene"]

        os.makedirs(self.save_dir, exist_ok=True)

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = update_cam(cfg)

        self.printer = Printer(len(stream))

        self.num_running_thread = torch.zeros((1)).int()
        self.num_running_thread.share_memory_()
        self.all_trigered = torch.zeros((1)).int()
        self.all_trigered.share_memory_()

        if self.cfg["mapping"]["uncertainty_params"]["activate"]:
            n_features = self.cfg["mapping"]["uncertainty_params"]["feature_dim"]
            self.uncer_network = generate_uncertainty_mlp(n_features)
            self.uncer_network.share_memory()
        else:
            self.uncer_network = None

        self.video = DepthVideo(cfg, self.printer, uncer_network=self.uncer_network)

        self.mapper: Mapper = None
        self.stream = stream
        self.pointcloud_dir = os.path.join(self.save_dir, "pointclouds")
        os.makedirs(self.pointcloud_dir, exist_ok=True)
        self.base_intrinsic = self._load_intrinsic_from_file()

    def _load_intrinsic_from_file(self) -> torch.Tensor:
        """
        Prefer per-sequence intrinsics stored in <data>/cam_K.txt.
        Falls back to dataset-provided intrinsics when the file is absent.
        """
        if hasattr(self.stream, "input_folder"):
            cam_file = os.path.join(self.stream.input_folder, "cam_K.txt")
            if os.path.exists(cam_file):
                K = np.loadtxt(cam_file).reshape(3, 3)
                vec = torch.tensor(
                    [K[0, 0], K[1, 1], K[0, 2], K[1, 2]],
                    dtype=torch.float32,
                    device=self.device,
                )
                return vec
            else:
                raise FileNotFoundError(f"Intrinsic file not found: {cam_file}")

        return self.stream.get_intrinsic().to(self.device)

    def _notify_mapper(self, pipe, timestamp, video_idx, just_initialized):
        pipe.send(
            {
                "is_keyframe": True,
                "video_idx": video_idx,
                "timestamp": timestamp,
                "just_initialized": just_initialized,
                "end": False,
            }
        )
        pipe.recv()

    def feed_sequence(self, pipe):
        """
        Feed the mapper with ground-truth pose and depth for every frame.
        """
        self.printer.print("Sequence feeder triggered.", FontColor.TRACKER)
        self.all_trigered += 1

        os.makedirs(f"{self.save_dir}/mono_priors/depths", exist_ok=True)
        os.makedirs(f"{self.save_dir}/mono_priors/features", exist_ok=True)

        while self.all_trigered < self.num_running_thread:
            pass
        self.printer.print("Sequence feeding starts!", FontColor.TRACKER)
        self.printer.pbar_ready()

        intrinsic = self.base_intrinsic

        for idx in range(len(self.stream)):
            timestamp, image, depth, pose = self.stream[idx]
            if pose is None:
                raise ValueError(
                    "Ground-truth stream requires pose information for every frame."
                )

            image_tensor = image.squeeze(0)
            depth_tensor = depth.to(self.device).float()
            w2c = torch.linalg.inv(pose).to(self.device)
            pose_vec = matrix4x4_to_se3(w2c).to(self.device)

            self.video.append(
                timestamp,
                image_tensor,
                pose_vec,
                None,
                depth_tensor,
                intrinsic.clone(),
            )

            video_idx = self.video.counter.value - 1
            self._notify_mapper(pipe, timestamp, video_idx, just_initialized=(idx == 0))
            self.printer.update_pbar()

        pipe.send(
            {
                "is_keyframe": True,
                "video_idx": None,
                "timestamp": None,
                "just_initialized": False,
                "end": True,
            }
        )
        self.printer.print("Sequence feeding done.", FontColor.TRACKER)

    def mapping(self, pipe, q_main2vis, q_vis2main):
        if self.cfg["mapping"]["uncertainty_params"]["activate"]:
            self.mapper = Mapper(self, pipe, self.uncer_network, q_main2vis, q_vis2main)
        else:
            self.mapper = Mapper(self, pipe, None, q_main2vis, q_vis2main)
        self.printer.print("Mapping Triggered!", FontColor.MAPPER)

        self.all_trigered += 1
        setup_seed(self.cfg["setup_seed"])

        while self.all_trigered < self.num_running_thread:
            pass
        self.printer.print("Mapping Starts!", FontColor.MAPPER)
        self.mapper.run()
        self.printer.print("Mapping Done!", FontColor.MAPPER)

        self.terminate()

    def terminate(self):
        """Finalize the run, dump assets, and evaluate metrics."""
        self.video.save_video(f"{self.save_dir}/video.npz")
        if not isinstance(self.stream, RGB_NoPose):
            try:
                kf_traj_eval(
                    f"{self.save_dir}/video.npz",
                    f"{self.save_dir}/traj",
                    "kf_traj",
                    self.stream,
                    self.logger,
                    self.printer,
                )
            except Exception as e:
                self.printer.print(e, FontColor.ERROR)

        if self.cfg["mapping"]["final_refine_iters"] > 0:
            self.mapper.final_refine(iters=self.cfg["mapping"]["final_refine_iters"])

        self.mapper.save_all_kf_figs(
            self.save_dir,
            iteration="after_refine",
        )

        self._export_pointclouds_from_mapper()

        self.mapper.gaussians.save_ply(f"{self.save_dir}/final_gs.ply")

        if self.cfg["mapping"]["uncertainty_params"]["activate"]:
            torch.save(
                self.mapper.uncer_network.state_dict(),
                self.save_dir + "/uncertainty_mlp_weight.pth",
            )

        self.printer.print("Metrics Evaluation Done!", FontColor.EVAL)

    def _export_pointclouds_from_mapper(self):
        """
        Export per-frame point clouds using the mapper's refined depth and pose.
        """
        if self.mapper is None or not hasattr(self.mapper, "cameras"):
            return

        fx, fy, cx, cy = self.base_intrinsic.detach().cpu().numpy()

        for video_idx in getattr(self.mapper, "video_idxs", []):
            viewpoint = self.mapper.cameras.get(video_idx)
            if viewpoint is None or viewpoint.depth is None:
                continue

            depth_np = np.asarray(viewpoint.depth)
            valid_mask = depth_np > 0
            if not np.any(valid_mask):
                continue

            h, w = depth_np.shape
            ys, xs = np.indices((h, w))

            z = depth_np[valid_mask]
            x = (xs[valid_mask] - cx) / fx * z
            y = (ys[valid_mask] - cy) / fy * z
            pts_cam = np.stack([x, y, z], axis=-1)

            color = getattr(viewpoint, "original_image", None)
            if isinstance(color, torch.Tensor) and color.numel() > 0:
                color_np = (
                    color.detach().cpu().permute(1, 2, 0).contiguous().numpy()
                )
                rgb = color_np[valid_mask].reshape(-1, 3)
                rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
            else:
                rgb = None

            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :3] = viewpoint.R.detach().cpu().numpy()
            w2c[:3, 3] = viewpoint.T.detach().cpu().numpy()
            c2w = np.linalg.inv(w2c)

            ones = np.ones((pts_cam.shape[0], 1), dtype=np.float32)
            pts_h = np.concatenate([pts_cam, ones], axis=1)
            pts_world = (c2w @ pts_h.T).T[:, :3].astype(np.float32)

            if rgb is not None and rgb.shape[0] == pts_world.shape[0]:
                cloud = np.concatenate([pts_world, rgb], axis=1)
            else:
                cloud = pts_world

            timestamp = int(self.video.timestamp[video_idx].item())
            out_path = os.path.join(self.pointcloud_dir, f"{timestamp:05d}.npy")
            np.save(out_path, cloud)

    def run(self):
        m_pipe, t_pipe = mp.Pipe()

        q_main2vis = mp.Queue() if self.cfg['gui'] else None
        q_vis2main = mp.Queue() if self.cfg['gui'] else None

        processes = [
            mp.Process(target=self.feed_sequence, args=(t_pipe,)),
            mp.Process(target=self.mapping, args=(m_pipe, q_main2vis, q_vis2main)),
        ]
        self.num_running_thread += len(processes)
        if self.cfg['gui']:
            self.num_running_thread += 1
        for p in processes:
            p.start()

        if self.cfg['gui']:
            pipeline_params = munchify(self.cfg["mapping"]["pipeline_params"])
            bg_color = [0, 0, 0]
            background = torch.tensor(
                bg_color, dtype=torch.float32, device=self.device
            )
            gaussians = GaussianModel(self.cfg['mapping']['model_params']['sh_degree'], config=self.cfg)

            params_gui = gui_utils.ParamsGUI(
                pipe=pipeline_params,
                background=background,
                gaussians=gaussians,
                q_main2vis=q_main2vis,
                q_vis2main=q_vis2main,
            )
            gui_process = mp.Process(target=slam_gui.run, args=(params_gui,))
            gui_process.start()
            self.all_trigered += 1


        for p in processes:
            p.join()

        self.printer.terminate()

        for process in mp.active_children():
            process.terminate()
            process.join()
