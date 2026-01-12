#!/usr/bin/env python3
"""
Iterate images in scripts_data_processing/img, collect click prompts,
run SAM2 segmentation, and save RGBA output with alpha from the mask.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple


def _collect_clicks(img_path: Path) -> List[Tuple[float, float]]:
    import matplotlib.pyplot as plt
    from PIL import Image

    image = Image.open(str(img_path)).convert("RGB")
    plt.figure(figsize=(9, 6))
    plt.title("点击物体上的点(回车结束)，无点则跳过\n" + str(img_path.name))
    plt.imshow(image)
    clicks = plt.ginput(n=-1, timeout=0)
    plt.close()
    if not clicks:
        return []
    return [(float(x), float(y)) for x, y in clicks]


def _iter_images(img_dir: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])


def main() -> None:
    here = Path(__file__).resolve()
    repo_root = here.parents[1]
    img_dir = here.parent / "img"
    out_dir = here.parent / "img_alpha"

    ap = argparse.ArgumentParser(description="SAM alpha matte from click prompts")
    ap.add_argument("--img-dir", type=str, default=str(img_dir), help="Input image dir")
    ap.add_argument("--out-dir", type=str, default=str(out_dir), help="Output dir for RGBA PNGs")
    args = ap.parse_args()

    img_dir = Path(args.img_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not img_dir.is_dir():
        raise FileNotFoundError(f"未找到目录: {img_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure project root and src are importable, then import SAM click helper
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from FoundationStereo.sam2_root.notebooks.get_mask import click_mask  # type: ignore

    images = _iter_images(img_dir)
    if not images:
        print(f"未在 {img_dir} 找到图片")
        return

    from PIL import Image
    import numpy as np

    for idx, img_path in enumerate(images, start=1):
        print(f"[{idx}/{len(images)}] 处理 {img_path.name}")
        points = _collect_clicks(img_path)
        if not points:
            print("  - 未点击，跳过")
            continue

        rgb = np.array(Image.open(str(img_path)).convert("RGB"))
        mask = click_mask(rgb, points, labels=None, multimask=True)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        alpha = (mask.astype(np.uint8) > 0) * 255

        rgba = np.dstack([rgb, alpha.astype(np.uint8)])
        out_name = img_path.stem + "_alpha.png"
        out_path = out_dir / out_name
        Image.fromarray(rgba, mode="RGBA").save(str(out_path))
        print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
