import os
import cv2
import argparse


def convert_png_to_jpg(video_name=None, input_dir=None, output_dir=None):
    if input_dir is None:
        if not video_name:
            raise ValueError("video_name 或 input_dir 必须提供其中一个")
        input_dir = f"demo_data_for_pose/{video_name}/rgb"

    if output_dir is None:
        if not video_name:
            raise ValueError("video_name 或 output_dir 必须提供其中一个")
        output_dir = f"demo_data_for_pose/{video_name}/color_jpg"

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"未找到输入目录: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".png")]
    if not files:
        print(f"⚠️ 输入目录 {input_dir} 中未找到 PNG 文件")
        return

    for filename in files:
        path_in = os.path.join(input_dir, filename)
        img = cv2.imread(path_in)
        if img is None:
            print(f"❌ 无法读取 {path_in}")
            continue

        out_name = os.path.splitext(filename)[0] + ".jpg"
        path_out = os.path.join(output_dir, out_name)
        print(f"转换 {path_in} -> {path_out}")
        cv2.imwrite(path_out, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        print(f"✅ {filename} -> {out_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_name", type=str, help="视频文件夹名")
    parser.add_argument("--input_dir", type=str, help="PNG 输入目录")
    parser.add_argument("--output_dir", type=str, help="JPG 输出目录")
    args = parser.parse_args()

    if not args.input_dir and not args.video_name:
        parser.error("需要提供 --video_name 或 --input_dir")

    convert_png_to_jpg(
        video_name=args.video_name,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
