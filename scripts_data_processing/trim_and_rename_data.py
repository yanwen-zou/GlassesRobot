from pathlib import Path

ROOT = Path("data")   # 换成你的 data 也可以
TARGET_DIRS = ["head_pos", "zed_left", "zed_right"]
CUT_FRONT = 50
CUT_BACK = 40

for timestamp_dir in ROOT.iterdir():
    if not timestamp_dir.is_dir():
        continue

    print(f"Processing: {timestamp_dir.name}")

    for sub in TARGET_DIRS:
        d = timestamp_dir / sub
        if not d.exists():
            print(f"  Skip {sub}, not found")
            continue

        files = sorted(d.glob("*.*"))
        total = len(files)
        if total == 0:
            continue

        # 计算需要保留的文件列表
        keep = files[CUT_FRONT : total - CUT_BACK]

        # 删除前40和后20
        for f in files[:CUT_FRONT]:
            f.unlink()
        for f in files[total - CUT_BACK:]:
            f.unlink()

        # 对剩余文件重新编号
        keep = sorted(keep)  # 再排一次，确保顺序正确
        for i, f in enumerate(keep):
            new_name = d / f"{i:06d}{f.suffix}"
            f.rename(new_name)

        print(f"  {sub}: kept {len(keep)}, removed {total - len(keep)}")
