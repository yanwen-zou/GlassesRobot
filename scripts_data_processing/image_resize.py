import shutil
from pathlib import Path

from PIL import Image


DATA_ROOT = Path("data/20251118_131652")
TARGET_DIRS = ["rgb_backup", "zed_left_backup", "zed_right_backup", "masks", "mask_hand"]
BACKUP_SUFFIX = "_backup"
TARGET_SIZE = (640, 360)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def copy_directory(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def resize_images_in_dir(directory: Path) -> None:
    for image_path in sorted(directory.glob("*")):
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        with Image.open(image_path) as img:
            resized = img.resize(TARGET_SIZE, Image.BILINEAR)
            resized.save(image_path)


def main():
    for name in TARGET_DIRS:
        src_dir = DATA_ROOT / name
        if not src_dir.exists():
            print(f"Skipping missing directory: {src_dir}")
            continue
        backup_dir = DATA_ROOT / f"{name}{BACKUP_SUFFIX}"
        print(f"Backing up {src_dir} -> {backup_dir}")
        copy_directory(src_dir, backup_dir)

        print(f"Resizing images in {src_dir} to {TARGET_SIZE}")
        resize_images_in_dir(src_dir)

    print("Done.")


if __name__ == "__main__":
    main()
