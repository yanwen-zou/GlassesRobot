# GlassesRobot

GlassesRobot is an egocentric robot learning and evaluation codebase built around:

- head pose streamed from AR glasses
- stereo capture from a ZED camera
- task-level calibration between the camera, robot, and object
- offline perception and pose processing
- robot-side replay / evaluation
- optional policy training baselines

The repository contains both the data collection stack and several downstream research pipelines. If you only need data collection or only need offline processing, you do not need to reproduce the entire repository.

## Overview

> Replace the placeholder figures below with your project images before publishing.

![System Overview](docs/figures/system_overview.png)

![Pipeline Overview](docs/figures/pipeline_overview.png)

### At a Glance

| Item | Description |
| --- | --- |
| Input | Glasses head pose stream + ZED stereo images |
| Calibration | Camera / robot / object calibration, including ball-based and task-specific transforms |
| Perception | Stereo depth, masking, object pose estimation |
| Execution | Robot replay and online task evaluation |
| Training | Optional policy training and baseline integrations |

## Repository Contents

| Path | Role |
| --- | --- |
| `src/egodata_record/` | ROS2 nodes for glasses pose ingestion, stereo recording, and hand-eye calibration |
| `src/egodata_eval/` | calibration, replay, evaluation, and online execution utilities |
| `scripts/` | user-facing scripts for calibration, preprocessing, pipelines, and visualization |
| `src/FoundationStereo/` | depth and mask related components |
| `foundationpose/` | object pose estimation dependency |
| `MBA/` | policy training code |
| `baseline/openpi/` | optional OpenPI baseline integration |
| `glasses_hardware/` | hardware wrappers, SDK adapters, and robot-specific utilities |

## Installation

### Recommended Software Stack

The current code structure is most naturally reproduced on:

- Ubuntu 22.04
- Python 3.10
- ROS2 Humble
- ZED SDK with Python bindings (`pyzed.sl`)
- CUDA-enabled PyTorch for training or heavy perception modules

`uv` is recommended for Python environment management, but `uv` does not replace ROS2 or the ZED SDK. Those still need to be installed at the system level.

### Environment Summary

| Layer | Recommended choice | Managed by |
| --- | --- | --- |
| OS | Ubuntu 22.04 | system |
| Python | 3.10 | `uv` |
| ROS | ROS2 Humble | system |
| Stereo SDK | ZED SDK + Python bindings | system |
| Deep learning | PyTorch with matching CUDA | `uv` + system CUDA |
| Robot SDKs | vendor-specific | system / local install |

### Environment Setup

#### 1. Clone the repository

```bash
git clone https://github.com/yanwen-zou/GlassesRobot.git
cd GlassesRobot
```

#### 2. Install system-level dependencies

These are not managed by `uv`:

##### ROS2 Humble

Install ROS2 Humble and source it:

```bash
source /opt/ros/humble/setup.bash
```

Required ROS2 Python packages used by this repo include:

- `rclpy`
- `geometry_msgs`
- `std_msgs`
- `launch`
- `launch_ros`

##### ZED SDK

Install the ZED SDK and make sure Python can import:

```python
import pyzed.sl as sl
```

`stereo_video_recorder.py` depends on the ZED Python bindings at runtime.

##### CUDA / PyTorch

You also need install CUDA toolkit and PyTorch according to
your own system.

#### 3. Create a Python environment with `uv`

Create a Python 3.10 virtual environment:

```bash
uv venv .venv --python 3.10
source .venv/bin/activate
```

Upgrade packaging tools:

```bash
uv pip install --upgrade pip setuptools wheel
```

#### 4. Install Python dependencies

If you only want the glasses + ZED data collection stack, start with:

```bash
uv pip install numpy opencv-contrib-python
```

For the broader offline processing / evaluation path, you will typically also need:

```bash
uv pip install scipy h5py pillow matplotlib
```

For training or additional research modules, you may further need:

```bash
uv pip install torch torchvision
```

Some heavy dependencies are intentionally not pinned here because they depend on your GPU / CUDA / SDK / robot setup.

### Dependency Profiles

| Use case | Python packages typically needed |
| --- | --- |
| Glasses + ZED recording only | `numpy`, `opencv-contrib-python` |
| Offline processing / evaluation | `numpy`, `opencv-contrib-python`, `scipy`, `h5py`, `pillow`, `matplotlib` |
| Training / heavier pipelines | above + `torch`, `torchvision` |

#### 5. Build the ROS2 package

The ROS2 package in this repository is `egodata_record`.

From the repository root:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select egodata_record
source install/setup.bash
```

When opening a new shell, the usual order is:

```bash
cd /path/to/GlassesRobot
source /opt/ros/humble/setup.bash
source .venv/bin/activate
source install/setup.bash
```

#### 6. Verify the core imports

Before running any pipelines, verify the minimum runtime imports:

```bash
python - <<'PY'
import numpy
import cv2
import rclpy
import pyzed.sl as sl
print("Environment check passed.")
PY
```

## Quick Start

### A. Glasses + ZED data collection

Launch the ROS2 recording pipeline:

```bash
ros2 launch egodata_record stereo_record.launch.py
```

This starts the main collection stack:

- `udp_listener`
- `headpos_listener`
- `stereo_video_recorder`

Main code paths:

- `src/egodata_record/egodata_record/headpos_listener.py`
- `src/egodata_record/egodata_record/udp_listener.py`
- `src/egodata_record/egodata_record/stereo_video_recorder.py`

Note that you should install the Unity package in your XReal Glasses first, for detail, see <TODO: Write a doc for XReal here.>

## Data Processing

The repository contains several data-processing steps after raw recording. A typical offline flow is:

1. Record stereo images and glasses head pose.
2. Run ball-based calibration to estimate base-related transforms.
3. Generate masks, depth, and object pose estimates.
4. Convert processed data into evaluation or training-ready formats.

After recording original stereo egocentric data, run:

`./scripts/pipelines/data_process.sh`

This will go through the next 3 steps as above.


## Training

Typical training entry points:

```bash
python MBA/train_obj.py --data_path data --ckpt_dir MBA/ckpt_delta --enable_mba
```

and for the OpenPI baseline:

```bash
cd baseline/openpi
uv run scripts/train.py pi05_realworld --exp-name=<your_exp_name>
```

## Hardware Notes

This project is hardware-dependent. You should expect to adapt the hardware configuration to your own reproduction setup.

The exact hardware used in your setup may differ from the setup used during development. Treat this repository as research code rather than a hardware-agnostic package.

### Hardware adaptation checklist

| Category | Typical changes needed |
| --- | --- |
| Robot | SDK paths, home poses, motion limits, gripper configuration |
| Camera | ZED serial, resolution, FPS, intrinsics paths |
| Glasses | UDP source, coordinate convention, topic / port mapping |
| Network | local IPs for UDP control and telemetry, you should modify these settings BOTH in python scripts and the Unity project.<TODO:detailed info here> |
| Calibration | output paths, mesh locations, task constants |
| Runtime safety | workspace bounds, initialization poses, command limits |

## Citation

If you use this repository in academic work, consider adding your paper or project citation here.

```bibtex
@misc{glassesrobot,
  title  = {GlassesRobot},
  author = {Yanwen Zou and collaborators},
  year   = {2026},
  note   = {GitHub repository}
}
```


## Acknowledgements

This repository integrates or builds on several external components, including:

- ROS2
- ZED SDK
- FoundationPose
- FoundationStereo-related components
- OpenPI baseline code

Please also check the corresponding subdirectories for their original licenses, setup instructions, and attribution requirements.

## Notes

- This repository contains research code and local hardware assumptions.
- Not every path is expected to work unchanged on a new machine.
- Some modules are optional and only needed for specific experiments.
- If you are preparing a public reproduction, start from the smallest runnable subset first.
