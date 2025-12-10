#!/usr/bin/env python3
"""
Wrapper to run SAM segmentation on a timestamp directory that contains RGB frames
under `rgb/` without loading all frames to GPU at once. We process frames
sequentially using the SAM2 image predictor to avoid OOM.

Usage examples:
  python3 data_processing_scripts/run_sam_for_timestamp.py \
      --timestamp-dir data/20251030_111157

Notes:
- Converts `rgb/*.png` to `jpg/*.jpg` if needed, but keeps only one frame
  in memory/GPU at a time.
- Masks are written to `<timestamp>/masks`.
- Interactive for the first frame (click seeds), then applies same points to
  all subsequent frames, one-by-one, freeing memory each step.
"""

import argparse
import os
import re
import sys
from pathlib import Path


def _read_png(path: Path):
    import numpy as np
    # Try Pillow first for robust JPEG save
    try:
        from PIL import Image  # type: ignore
        im = Image.open(str(path)).convert("RGB")
        return np.array(im)
    except Exception:
        pass
    try:
        import cv2  # type: ignore
        im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if im is None:
            raise RuntimeError("cv2.imread failed")
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        elif im.shape[2] == 4:
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
        return im[:, :, ::-1]  # BGR->RGB
    except Exception:
        pass
    try:
        import imageio.v3 as iio  # type: ignore
        im = iio.imread(str(path))
        if im.ndim == 2:
            im = np.stack([im, im, im], axis=-1)
        elif im.shape[2] == 4:
            im = im[:, :, :3]
        return im
    except Exception as e:
        raise RuntimeError(f"无法读取图片: {path}: {e}")


def _save_jpg(path: Path, rgb_array) -> None:
    # Prefer Pillow for JPEG
    try:
        from PIL import Image  # type: ignore
        Image.fromarray(rgb_array).save(str(path), format="JPEG", quality=95)
        return
    except Exception:
        pass
    try:
        import cv2  # type: ignore
        bgr = rgb_array[:, :, ::-1]
        cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return
    except Exception:
        pass
    try:
        import imageio.v3 as iio  # type: ignore
        iio.imwrite(str(path), rgb_array, quality=95)
        return
    except Exception as e:
        raise RuntimeError(f"无法保存JPEG: {path}: {e}")


def ensure_color_converted(timestamp_dir: Path) -> Path:
    """Ensure frames exist for SAM without symlinks.

    - If `jpg/` exists with JPGs, use it directly.
    - Else if `color/` exists with PNGs, use it directly.
    - Else if `rgb/` exists with PNGs, convert them to JPGs and place under `color/`.
      For compatibility with batch_sam_segmentation.py, also mirror files to `jpg/`.
    Returns the frame directory to pass to SAM (we prefer `jpg/`).
    """
    rgb_dir = timestamp_dir / "rgb"
    color_dir = timestamp_dir / "color"
    jpg_dir = timestamp_dir / "jpg"

    if jpg_dir.is_dir():
        return jpg_dir
    if color_dir.is_dir():
        # If color contains PNGs, the downstream script will pick it up.
        return color_dir
    if rgb_dir.is_dir():
        color_dir.mkdir(parents=True, exist_ok=True)
        jpg_dir.mkdir(parents=True, exist_ok=True)

        # Clean jpg_dir: remove any files whose stems are not pure digits
        for f in list(jpg_dir.glob("*.jpg")) + list(jpg_dir.glob("*.jpeg")):
            stem = f.stem
            if not stem.isdigit():
                try:
                    f.unlink()
                except OSError:
                    pass

        pngs = sorted([p for p in rgb_dir.iterdir() if p.suffix.lower() == ".png"])
        if not pngs:
            raise FileNotFoundError(f"{rgb_dir} 中未找到 PNG 文件")
        print(f"Converting {len(pngs)} PNG(s) from rgb/ to JPEG in color/ and jpg/ ...")
        # Determine zero padding based on the longest trailing digit sequence
        pad = 6
        # Try infer pad from filenames like rgb_000123.png
        for p in pngs:
            m = re.search(r"(\d+)$", p.stem)
            if m:
                pad = max(pad, len(m.group(1)))
        idx = 1
        for p in pngs:
            arr = _read_png(p)
            # Prefer numeric stems; try to extract trailing digits, else assign sequential
            m = re.search(r"(\d+)$", p.stem)
            if m:
                stem_numeric = m.group(1).zfill(pad)
            else:
                stem_numeric = str(idx).zfill(pad)
                idx += 1
            out1 = color_dir / f"{stem_numeric}.jpg"
            out2 = jpg_dir / f"{stem_numeric}.jpg"
            _save_jpg(out1, arr)
            if out2 != out1:
                _save_jpg(out2, arr)
        # Downstream script prefers `jpg/`; return that
        return jpg_dir

    raise FileNotFoundError(
        f"未找到 {timestamp_dir}/rgb、{timestamp_dir}/color 或 {timestamp_dir}/jpg 目录"
    )


def main():
    repo_root = Path(__file__).resolve().parents[1]
    default_ts = repo_root / "data" / "20251030_113006"

    ap = argparse.ArgumentParser(description="Run SAM mask generation for a timestamp directory")
    ap.add_argument("--timestamp-dir", type=str, default=str(default_ts), help="Path to data/<timestamp> directory")
    args = ap.parse_args()

    ts_dir = Path(args.timestamp_dir).expanduser().resolve()
    if not ts_dir.exists():
        raise FileNotFoundError(f"未找到时间戳目录: {ts_dir}")

    # Ensure expected frame layout without symlinks (convert to JPGs if needed)
    frame_dir = ensure_color_converted(ts_dir)
    print(f"Using frames from: {frame_dir}")

    # Discover frames (prefer jpg/ else color/)
    if (ts_dir / "jpg").is_dir():
        frame_dir = ts_dir / "jpg"
        extensions = {".jpg", ".jpeg", ".JPG", ".JPEG"}
    elif (ts_dir / "color").is_dir():
        frame_dir = ts_dir / "color"
        extensions = {".png", ".PNG"}
    else:
        raise FileNotFoundError(f"未找到帧目录: {ts_dir}/jpg 或 {ts_dir}/color")

    frame_names = [p.name for p in frame_dir.iterdir() if p.suffix in extensions]
    try:
        frame_names.sort(key=lambda n: int(Path(n).stem))
    except ValueError:
        frame_names.sort()
    if not frame_names:
        raise FileNotFoundError(f"未在 {frame_dir} 找到图像帧")

    # Prompt on the first frame (single time; then track continuously)
    from PIL import Image  # lazy import
    import numpy as np
    import matplotlib.pyplot as plt

    def prompt_box(frame_path: Path):
        image = Image.open(frame_path)
        plt.figure(figsize=(9, 6))
        plt.title(f"框选目标：依次点击左上与右下角\n{frame_path}")
        plt.imshow(image)
        clicks = plt.ginput(n=2, timeout=0)
        plt.close()
        if clicks is None or len(clicks) < 2:
            return None
        (x0, y0), (x1, y1) = clicks[0], clicks[1]
        # normalize ordering to xyxy with top-left and bottom-right
        x_min, x_max = (x0, x1) if x0 <= x1 else (x1, x0)
        y_min, y_max = (y0, y1) if y0 <= y1 else (y1, y0)
        return float(x_min), float(y_min), float(x_max), float(y_max)

    first_path = frame_dir / frame_names[0]
    box_xyxy = prompt_box(first_path)
    if box_xyxy is None:
        print("⚠️ 未记录到框，退出。")
        return

    # Use SAM2 image predictor to process frames one-by-one
    sys.path.append(str((repo_root / "src" / "FoundationStereo" / "sam2_root").resolve()))
    from notebooks.get_mask import box_mask  # type: ignore

    masks_dir = ts_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    import torch  # type: ignore

    REBOX_INTERVAL = 50  # 每隔多少帧重新框选

    for idx, name in enumerate(frame_names):
        path = frame_dir / name
        # 每隔固定帧数，允许用户在当前帧重新框选
        if idx != 0 and (idx % REBOX_INTERVAL == 0):
            refreshed = prompt_box(path)
            if refreshed is not None:
                box_xyxy = refreshed
                print(f"在第 {idx} 帧重新框选: {box_xyxy}")
            else:
                print(f"未重新框选，继续使用之前的框（第 {idx} 帧）。")
        # Read as RGB np.uint8 without keeping reference
        img = Image.open(path).convert("RGB")
        arr = np.array(img, copy=True)
        del img

        mask = box_mask(arr, box_xyxy=box_xyxy, multimask=True)

        out_name = f"masks_{Path(name).stem}.png"
        out_path = masks_dir / out_name
        Image.fromarray(mask, mode="L").save(out_path)

        # Free per-frame tensors
        del arr, mask
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Progress log every 50 frames
        if (idx + 1) % 50 == 0 or (idx + 1) == len(frame_names):
            print(f"已处理 {idx+1}/{len(frame_names)} 帧 -> {out_path}")

    print(f"✅ 完成，mask 输出目录: {masks_dir}")


if __name__ == "__main__":
    main()
