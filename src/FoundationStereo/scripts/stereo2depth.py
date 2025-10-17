import cv2, os
import numpy as np
import torch
from omegaconf import OmegaConf
import sys
import argparse
import logging
from typing import Iterator, List, Tuple

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

from core.foundation_stereo import FoundationStereo
from core.utils.utils import InputPadder
from Utils import *
import imageio

IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def load_model(ckpt_dir):
    cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'
    args = OmegaConf.create(cfg)
    model = FoundationStereo(args)
    ckpt = torch.load(ckpt_dir, map_location="cuda", weights_only=False)
    logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
    model.load_state_dict(ckpt['model'])
    model.cuda()
    model.eval()
    return model, args


def sorted_image_files(directory: str) -> List[str]:
    files = [os.path.join(directory, f) for f in os.listdir(directory)
             if os.path.splitext(f.lower())[1] in IMAGE_EXTS]
    def sort_key(path: str) -> Tuple[int, str]:
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            return (0, int(stem))
        except ValueError:
            return (1, stem)
    return sorted(files, key=sort_key)


def frame_source(left_path: str, right_path: str) -> Iterator[Tuple[str, np.ndarray, np.ndarray]]:
    if os.path.isdir(left_path) and os.path.isdir(right_path):
        left_files = sorted_image_files(left_path)
        right_files = sorted_image_files(right_path)
        if not left_files or not right_files:
            raise RuntimeError("Left/right directories must contain image files")
        if len(left_files) != len(right_files):
            logging.warning("Mismatch between left (%d) and right (%d) frame counts; using minimum",
                            len(left_files), len(right_files))
        for left_file, right_file in zip(left_files, right_files):
            left = cv2.imread(left_file, cv2.IMREAD_COLOR)
            right = cv2.imread(right_file, cv2.IMREAD_COLOR)
            if left is None or right is None:
                logging.warning("Failed to read pair (%s, %s); skipping", left_file, right_file)
                continue
            yield os.path.splitext(os.path.basename(left_file))[0], left, right
    else:
        cap_left = cv2.VideoCapture(left_path)
        cap_right = cv2.VideoCapture(right_path)
        if not cap_left.isOpened() or not cap_right.isOpened():
            raise RuntimeError(f"Failed to open videos {left_path} / {right_path}")
        frame_id = 0
        while True:
            retL, left = cap_left.read()
            retR, right = cap_right.read()
            if not (retL and retR):
                break
            yield f"{frame_id:06d}", left, right
            frame_id += 1
        cap_left.release()
        cap_right.release()


def main(left_file, right_file, out_dir):
    os.makedirs(f"{out_dir}/rgb", exist_ok=True)
    os.makedirs(f"{out_dir}/depth", exist_ok=True)
    os.makedirs(f"{out_dir}/depth_vis", exist_ok=True)

    scale = 1

    with open("assets/K_ZED.txt", "r") as f:
        lines = f.readlines()
        K = np.array(list(map(float, lines[0].split()))).reshape(3, 3)
        baseline = float(lines[1])
    K[:2] *= scale

    ckpt = "pretrained_models/23-51-11/model_best_bp2.pth"
    model, args = load_model(ckpt)

    processed = 0
    for frame_key, left, right in frame_source(left_file, right_file):
        print(f"Processing frame {frame_key}", flush=True)
        H, W = left.shape[:2]
        img0 = torch.as_tensor(left).cuda().float()[None].permute(0, 3, 1, 2)
        img1 = torch.as_tensor(right).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)

        with torch.cuda.amp.autocast(True):
            disp = model.forward(img0, img1, iters=32, test_mode=True)
        disp = padder.unpad(disp.float())
        disp = disp.data.cpu().numpy().reshape(H, W)

        depth = K[0, 0] * baseline / disp
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth_mm = (depth * 1000.0).astype(np.uint16)

        rgb_path = os.path.join(out_dir, "rgb", f"{frame_key}.png")
        depth_path = os.path.join(out_dir, "depth", f"{frame_key}.png")
        depth_vis_path = os.path.join(out_dir, "depth_vis", f"{frame_key}.png")

        cv2.imwrite(rgb_path, left)
        cv2.imwrite(depth_path, depth_mm)

        depth_vis = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = depth_vis.astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        cv2.imwrite(depth_vis_path, depth_color)

        processed += 1

    print(f"Saved {processed} frames to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--left_file', default='demo_data_for_pose/my_demo/left.mp4', type=str,
                        help="左相机输入（视频文件或帧目录）")
    parser.add_argument('--right_file', default='demo_data_for_pose/my_demo/right.mp4', type=str,
                        help="右相机输入（视频文件或帧目录）")
    parser.add_argument('--out_dir', default="demo_data_for_pose/my_demo", type=str, help="输出目录")
    args = parser.parse_args()

    main(args.left_file, args.right_file, args.out_dir)
