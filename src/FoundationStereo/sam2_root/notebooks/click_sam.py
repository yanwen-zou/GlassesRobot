import os
import argparse
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_name", type=str, required=True, help="视频文件夹名")
    args = parser.parse_args()

    video_name = args.video_name

    # 修改数据和输出路径
    video_dir = f"demo_data_for_pose/{video_name}/color_jpg"
    output_video_path = f"demo_data_for_pose/{video_name}/sam_video.mp4"
    mask_dir = f"sam_output/mask_{video_name}"

    # 自动创建相关输出目录
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)

    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

    # 显示第一帧，鼠标点击选点
    plt.figure(figsize=(9, 6))
    plt.title("请用鼠标在图像上点击两个点（左键），用于SAM分割")
    img = Image.open(os.path.join(video_dir, frame_names[0]))
    plt.imshow(img)
    clicked_points = plt.ginput(2, timeout=0)  # 等待用户点击两个点
    plt.close()

    # 转换为 numpy 格式，注意 ginput 返回 (x, y)，而后续代码是 (y, x)
    points = np.array([[pt[0], pt[1]] for pt in clicked_points], dtype=np.float32)
    labels = np.array([1, 1], np.int32)  # 两个正样本

    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    # show the results on the current (interacted) frame
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {ann_frame_idx}")
    plt.imshow(img)
    show_points(points, labels, plt.gca())
    show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

    import cv2 

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
        return frame, mask  # 返回叠加后的frame和二值mask

    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        
    # 写视频
    first_frame_path = os.path.join(video_dir, frame_names[0])
    first_frame = cv2.imread(first_frame_path)
    h, w = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    out = cv2.VideoWriter(output_video_path, fourcc, 30, (w, h))

    for out_frame_idx in range(0, len(frame_names)):
        frame = cv2.imread(os.path.join(video_dir, frame_names[out_frame_idx]))
        mask_dict = video_segments.get(out_frame_idx, {})

        combined_mask = np.zeros((h, w), dtype=np.uint8)  # 多个obj叠加
        for obj_id, mask in mask_dict.items():
            frame, mask_bin = apply_mask(frame, mask, obj_id=obj_id, random_color=False)
            combined_mask = np.logical_or(combined_mask, mask_bin).astype(np.uint8)

        # 保存mask图像（0/255二值）
        mask_filename = os.path.join(mask_dir, f"{out_frame_idx:06d}.png")
        cv2.imwrite(mask_filename, combined_mask * 255)

        out.write(frame)

    out.release()
    print(f"✅ Done! Saved masked video to {output_video_path}")
    print(f"✅ Saved per-frame masks to {mask_dir}")
