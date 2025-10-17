import os
import cv2
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from sam2.build_sam import build_sam2_video_predictor


# 鼠标框选回调
refPt = []
cropping = False
def click_and_crop(event, x, y, flags, param):
    global refPt, cropping
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append((x, y))
        if len(refPt) == 2:
            cv2.destroyAllWindows()

import matplotlib.pyplot as plt

def apply_mask(frame, mask, obj_id=None, random_color=False):
    """
    在OpenCV的frame上叠加半透明mask，类似show_mask逻辑
    """
    # squeeze 并 resize 一下保证对齐
    mask = np.squeeze(mask)
    if mask.shape != frame.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0])).astype(bool)

    # 生成颜色
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])  # RGBA, alpha=0.6

    # 拆开颜色
    overlay_color = (color[:3] * 255).astype(np.uint8)   # RGB 转 BGR
    alpha = color[3]

    # 创建 overlay
    overlay = frame.copy()
    overlay[mask] = overlay_color[::-1]  # RGB→BGR

    # 融合
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    return frame



def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def main(args):
    # 读取所有jpg序列
    frames = sorted([f for f in os.listdir(args.input) if f.lower().endswith(".jpg")])
    if not frames:
        print(f"❌ No jpg images found in {args.input}")
        return

    first_frame = cv2.imread(os.path.join(args.input, frames[0]))
    first_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

    # 框选
    clone = first_frame.copy()
    cv2.namedWindow("Select ROI (2 clicks)")
    cv2.setMouseCallback("Select ROI (2 clicks)", click_and_crop)
    while True:
        cv2.imshow("Select ROI (2 clicks)", clone)
        key = cv2.waitKey(1) & 0xFF
        if len(refPt) == 2:
            break
    cv2.destroyAllWindows()

    (x0, y0), (x1, y1) = refPt


    # 构建SAM2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = build_sam2_video_predictor(args.config, args.checkpoint, device=device)

    # 初始化
    inference_state = predictor.init_state(video_path=args.input)
    predictor.reset_state(inference_state)

    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

    points = np.array([refPt], dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array([1, 1], np.int32)

    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    # 遍历传播
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # 写视频
    h, w = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))

    for idx, fname in enumerate(frames):
        frame = cv2.imread(os.path.join(args.input, fname))
        mask_dict = video_segments.get(idx, {})
        for obj_id, mask in mask_dict.items():
            frame = apply_mask(frame, mask, obj_id=obj_id, random_color=False)

        out.write(frame)

    out.release()
    print(f"✅ Done! Saved masked video to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./demo_data_for_pose/cam_static/color", help="输入jpg序列目录")
    parser.add_argument("--output", type=str, default="masked.mp4", help="输出视频路径")
    parser.add_argument("--config", type=str, default="./configs/sam2.1/sam2.1_hiera_l.yaml", help="SAM2 config yaml")
    parser.add_argument("--checkpoint", type=str, default="./sam2/checkpoints/sam2.1_hiera_large.pt", help="SAM2 checkpoint路径")
    parser.add_argument("--fps", type=int, default=30, help="输出视频fps")
    args = parser.parse_args()
    main(args)
