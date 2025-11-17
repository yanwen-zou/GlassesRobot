import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from utils.track_utils import sample_points_from_masks


def parse_args():
    parser = argparse.ArgumentParser(description="Generate per-frame masks using Grounding DINO + SAM 2.")
    parser.add_argument(
        "--video_dir",
        required=True,
        help="Directory containing sequential .jpg frames to process.",
    )
    parser.add_argument(
        "--text",
        default="hand.",
        help="Grounding DINO text prompt (lowercase with trailing dot).",
    )
    parser.add_argument(
        "--output_dir",
        default="mask_hand",
        help="Directory to store generated mask images (default: mask_hand).",
    )
    parser.add_argument(
        "--prompt_type",
        choices=["point", "box", "mask"],
        default="box",
        help="SAM2 video predictor prompt type.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.video_dir):
        raise FileNotFoundError(f"Input directory {args.video_dir} does not exist.")

    # use bfloat16 for the entire script
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # init sam image predictor and video predictor model
    sam2_checkpoint = "src/FoundationStereo/sam2_root/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
    image_predictor = SAM2ImagePredictor(sam2_image_model)

    # init grounding dino model from huggingface
    model_id = "IDEA-Research/grounding-dino-tiny"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(args.video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    if not frame_names:
        raise RuntimeError(f"No JPEG frames found in {args.video_dir}")
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    inference_state = video_predictor.init_state(video_path=args.video_dir)

    ann_frame_idx = 0
    image = None
    results = None

    # search for first frame with detections
    while ann_frame_idx < len(frame_names):
        img_path = os.path.join(args.video_dir, frame_names[ann_frame_idx])
        image = Image.open(img_path)

        inputs = processor(images=image, text=args.text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = grounding_model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.25,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )

        if len(results) == 0 or results[0]["boxes"].shape[0] == 0:
            print(f"[INFO] No target found on frame index {ann_frame_idx}, skipping to next frame.")
            ann_frame_idx += 1
            continue
        break

    if results is None or len(results) == 0 or results[0]["boxes"].shape[0] == 0:
        raise RuntimeError("Grounding DINO did not detect any targets in the provided video frames.")

    # prompt SAM image predictor to get the mask for the object
    image_predictor.set_image(np.array(image.convert("RGB")))

    # process detection results
    input_boxes = results[0]["boxes"].cpu().numpy()

    # prompt SAM 2 image predictor to get the mask for the object
    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    # convert the mask shape to (n, H, W)
    if masks.ndim == 3:
        masks = masks[None]
    elif masks.ndim == 4:
        masks = masks.squeeze(1)

    # register prompts with video predictor
    if args.prompt_type == "point":
        all_sample_points = sample_points_from_masks(masks=masks, num_points=10)

        for object_id, points in enumerate(all_sample_points, start=1):
            labels = np.ones((points.shape[0]), dtype=np.int32)
            video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                points=points,
                labels=labels,
            )
    elif args.prompt_type == "box":
        for object_id, box in enumerate(input_boxes, start=1):
            video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                box=box,
            )
    elif args.prompt_type == "mask":
        for object_id, mask in enumerate(masks, start=1):
            labels = np.ones((1), dtype=np.int32)
            video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                mask=mask
            )
    else:
        raise NotImplementedError("Unsupported prompt type.")

    # propagate through the video
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    os.makedirs(args.output_dir, exist_ok=True)

    image_height = image.size[1]
    image_width = image.size[0]

    # write masks in zero-padded order
    for output_idx, frame_idx in enumerate(sorted(range(len(frame_names)))):
        segments = video_segments.get(frame_idx)
        if segments:
            masks_list = []
            for segment_mask in segments.values():
                if segment_mask.ndim == 3 and segment_mask.shape[0] == 1:
                    masks_list.append(segment_mask.squeeze(0))
                else:
                    masks_list.append(segment_mask)
            mask = np.any(np.stack(masks_list, axis=0), axis=0)
        else:
            mask = np.zeros((image_height, image_width), dtype=bool)

        mask_img = (mask.astype(np.uint8)) * 255
        output_path = os.path.join(args.output_dir, f"{output_idx:06d}.png")
        cv2.imwrite(output_path, mask_img)


if __name__ == "__main__":
    main()
