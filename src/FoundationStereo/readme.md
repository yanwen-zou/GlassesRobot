# FoundationStereo: Zero-Shot Stereo Matching

This is the official implementation of our paper accepted by CVPR 2025 Oral (**Best Paper Nomination**)

[[Website]](https://nvlabs.github.io/FoundationStereo/) [[Paper]](https://arxiv.org/abs/2501.09898) [[Video]](https://www.youtube.com/watch?v=R7RgHxEXB3o)

Authors: Bowen Wen, Matthew Trepte, Joseph Aribido, Jan Kautz, Orazio Gallo, Stan Birchfield

# Abstract
Tremendous progress has been made in deep stereo matching to excel on benchmark datasets through per-domain fine-tuning. However, achieving strong zero-shot generalization — a hallmark of foundation models in other computer vision tasks — remains challenging for stereo matching. We introduce FoundationStereo, a foundation model for stereo depth estimation designed to achieve strong zero-shot generalization. To this end, we first construct a large-scale (1M stereo pairs) synthetic training dataset featuring large diversity and high photorealism, followed by an automatic self-curation pipeline to remove ambiguous samples. We then design a number of network architecture components to enhance scalability, including a side-tuning feature backbone that adapts rich monocular priors from vision foundation models to mitigate the sim-to-real gap, and long-range context reasoning for effective cost volume filtering. Together, these components lead to strong robustness and accuracy across domains, establishing a new standard in zero-shot stereo depth estimation.

<p align="center">
  <img src="https://raw.githubusercontent.com/NVlabs/FoundationStereo/website/static/images/intro.jpg" width="800"/>
</p>


**TLDR**: Our method takes as input a pair of stereo images and outputs a dense disparity map, which can be converted to a metric-scale depth map or 3D point cloud.

<p align="center">
  <img src="./teaser/input_output.gif" width="600"/>
</p>

# Changelog
| Date       | Description                                                                                                         |
|------------|---------------------------------------------------------------------------------------------------------------------|
| 2025/08/05 | Our commercial model is available now at [here](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/foundationstereo)! |
| 2025/07/03 | Improve ONNX and TRT support. Add support for Jetson                                                                |


# Leaderboards 🏆
We obtained the 1st place on the world-wide [Middlebury leaderboard](https://vision.middlebury.edu/stereo/eval3/) and [ETH3D leaderboard](https://www.eth3d.net/low_res_two_view).

<p align="center">
  <img src="https://raw.githubusercontent.com/NVlabs/FoundationStereo/website/static/images/middlebury_leaderboard.jpg" width="700"/>
  <br>
  <img src="https://raw.githubusercontent.com/NVlabs/FoundationStereo/website/static/images/eth_leaderboard.png" width="700"/>
</p>


# Comparison with Monocular Depth Estimation
Our method outperforms existing approaches in zero-shot stereo matching tasks across different scenes.

<p align="center">
  <img src="https://raw.githubusercontent.com/NVlabs/FoundationStereo/website/static/images/mono_comparison.png" width="700"/>
</p>

# Installation

We've tested on Linux with GPU 3090, 4090, A100, V100, Jetson Orin. Other GPUs should also work, but make sure you have enough memory

```
conda env create -f environment.yml
conda run -n foundation_stereo pip install flash-attn
conda activate foundation_stereo
```

Note that `flash-attn` needs to be installed separately to avoid [errors during environment creation](https://github.com/NVlabs/FoundationStereo/issues/20).


# Model Weights
- Download the foundation model for zero-shot inference on your data. Put the entire folder (e.g. `23-51-11`) under `./pretrained_models/`.


| Model     | Description                                                                 |
|-----------|-----------------------------------------------------------------------------|
| [23-51-11](https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf?usp=sharing)  | Our best performing model for general use, based on Vit-large               |
| [11-33-40](https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf?usp=sharing)  | Slightly lower accuracy but faster inference, based on Vit-small            |
| [NVIDIA-TAO](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/foundationstereo)       | For commercial usage (adapted from Vit-small model)                 |

# Run demo
```
python scripts/run_demo.py --left_file ./assets/left.png --right_file ./assets/right.png --ckpt_dir ./pretrained_models/23-51-11/model_best_bp2.pth --out_dir ./test_outputs/
```
You can see output point cloud.

<p align="center">
  <img src="./teaser/output.jpg" width="700"/>
</p>

Tips:
- The input left and right images should be **rectified and undistorted**, which means there should not be fisheye kind of lens distortion and the epipolar lines are horizontal between the left/right images. If you obtain images from stereo cameras such as Zed, they usually have [handled this](https://github.com/stereolabs/zed-sdk/blob/3472a79fc635a9cee048e9c3e960cc48348415f0/recording/export/svo/python/svo_export.py#L124) for you.
- Do not swap left and right image. The left image should really be obtained from the left-side camera (objects will appear righter in the image).
- We recommend to use PNG files with no lossy compression
- Our method works best on stereo RGB images. However, we have also tested it on monochrome or IR stereo images (e.g. from RealSense D4XX series) and it works well too.
- For all options and instructions, check by `python scripts/run_demo.py --help`
- To get point cloud for your own data, you need to specify the intrinsics. In the intrinsic file in args, 1st line is the flattened 1x9 intrinsic matrix, 2nd line is the baseline (distance) between the left and right camera, unit in meters.
- For high-resolution image (>1000px), you can either (1) run with `--hiera 1` to enable hierarchical inference to get full resolution depth but slower; or (2) run with smaller scale, e.g. `--scale 0.5` to get downsized resolution depth but faster.
- For faster inference, you can reduce the input image resolution by e.g. `--scale 0.5`, and reduce refine iterations by e.g. `--valid_iters 16`.



# ONNX/TensorRT(TRT) Inference

We only support docker setup for ONNX/TRT version.

- Build docker (tested on NVIDIA Driver Version: 560.35.03, CUDA Version: 12.6)
```bash
export DIR=$(pwd)
cd docker && docker build --network host -t foundation_stereo .
bash run_container.sh
cd /
git clone https://github.com/onnx/onnx-tensorrt.git
cd onnx-tensorrt
python3 setup.py install
apt-get install -y libnvinfer-dispatch10 libnvinfer-bin tensorrt
cd $DIR
```


- Make ONNX:
```
XFORMERS_DISABLED=1 python scripts/make_onnx.py --save_path ./pretrained_models/foundation_stereo.onnx --ckpt_dir ./pretrained_models/23-51-11/model_best_bp2.pth --height 448 --width 672 --valid_iters 20
```

- Convert to TRT:
```
trtexec --onnx=pretrained_models/foundation_stereo.onnx --verbose --saveEngine=pretrained_models/foundation_stereo.plan --fp16
```

- Run TRT:
```
python scripts/run_demo_tensorrt.py \
        --left_img ${PWD}/assets/left.png \
        --right_img ${PWD}/assets/right.png \
        --save_path ${PWD}/output \
        --pretrained pretrained_models/foundation_stereo.plan \
        --height 448 \
        --width 672 \
        --pc \
        --z_far 100.0
```

We have observed 6X speed on the same GPU 3090 with TensorRT FP16. Although how much it speeds up depends on various factors, we recommend trying it out if you care about faster inference. Also remember to adjust the args setting based on your need.

# Running on Jetson
Please refer to [readme_jetson.md](readme_jetson.md).

# FSD Dataset
<p align="center">
  <img src="https://raw.githubusercontent.com/NVlabs/FoundationStereo/website/static/images/sdg_montage.jpg" width="800"/>
</p>

You can download the whole dataset [here](https://drive.google.com/drive/folders/1YdC2a0_KTZ9xix_HyqNMPCrClpm0-XFU?usp=sharing) (>1TB). We also provide a small [sample data](https://drive.google.com/file/d/1dJwK5x8xsaCazz5xPGJ2OKFIWrd9rQT5/view?usp=drive_link) (3GB) to peek. The whole dataset contains ~1M data points, where each consists of:
- Left and right images
- Ground-truth disparity

You can check how to read data by using our example with the sample data:
```
python scripts/vis_dataset.py --dataset_path ./DATA/sample/manipulation_v5_realistic_kitchen_2500_1/dataset/data/
```

It will produce:
<p align="center">
  <img src="./teaser/fsd_sample.png" width="800"/>
</p>

For dataset license, please check [this](https://github.com/NVlabs/FoundationStereo/blob/master/LICENSE).


# FAQ
- Q: Conda install does not work for me?<br>
  A: Check [this](https://github.com/NVlabs/FoundationStereo/issues/20)

- Q: I'm not getting point cloud or getting incomplete point cloud?<br>
  A: Check the flags in argparse about point cloud processing, such as `--z_far`, `--remove_invisible`, `--denoise_cloud`.

- Q: My GPU doesn't support Flash attention?<br>
  A: See [this](https://github.com/NVlabs/FoundationStereo/issues/13#issuecomment-2708791825)

- Q: RuntimeError: cuDNN error: CUDNN_STATUS_NOT_SUPPORTED. This error may appear if you passed in a non-contiguous input.<br>
  A: This may indicate OOM issue. Try reducing your image resolution or use a GPU with more memory.

- Q: How to run with RealSense?<br>
  A: See [this](https://github.com/NVlabs/FoundationStereo/issues/26) and [this](https://github.com/NVlabs/FoundationStereo/issues/80)

- Q: I have two or multiple RGB cameras, can I run this? <br>
  A: You can first rectify a pair of images using this [OpenCV function](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga617b1685d4059c6040827800e72ad2b6) into stereo image pair (now they don't have relative rotations), then feed into FoundationStereo.

- Q: Can I use it for commercial purpose? <br>
  A: We released a commercial version [here](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/foundationstereo).


# BibTeX
```
@article{wen2025stereo,
  title={FoundationStereo: Zero-Shot Stereo Matching},
  author={Bowen Wen and Matthew Trepte and Joseph Aribido and Jan Kautz and Orazio Gallo and Stan Birchfield},
  journal={CVPR},
  year={2025}
}
```

# Acknowledgement
We would like to thank Gordon Grigor, Jack Zhang, Karsten Patzwaldt, Hammad Mazhar and other NVIDIA Isaac team members for their tremendous engineering support and valuable discussions. Thanks to the authors of [DINOv2](https://github.com/facebookresearch/dinov2), [DepthAnything V2](https://github.com/DepthAnything/Depth-Anything-V2), [Selective-IGEV](https://github.com/Windsrain/Selective-Stereo) and [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) for their code release. Finally, thanks to CVPR reviewers and AC for their appreciation of this work and constructive feedback.


# Contact
For commercial inquiries, additional technical support, and other questions, please reach out to [Bowen Wen](https://wenbowen123.github.io/) (bowenw@nvidia.com).
