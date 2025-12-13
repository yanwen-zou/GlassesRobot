
# ----------------ZED Env---------------
# Recording data recording pipeline launching
ros2 launch egodata_record stereo_record.launch.py 

# Zed&Cam Calibration
ros2 launch egodata_record zed_handeye.launch.py \
  intrinsics:=/home/yanwen/research/unity_comm/src/FoundationStereo/assets/K_ZED.txt \
  intrinsic_scale:=1.0 \
  output:=/home/yanwen/research/unity_comm/glasses_hardware/calib/T_zed_tcp_calib.txt

# -------------foundation_stereo Env----------
# SAM2 For Mask -> FoundationStereo For Depth -> FoundationPose for Obj Pose Tracking
./cmd_book_pipeline.sh

# SAM2 For ball calib
./scripts_calib_balls/run_ball_pipeline.sh --data-dir data/20251127_175609

# -------------MBA Env------------------
# Train a obj pose prediction model

MPLBACKEND=Agg MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
python MBA/train_obj.py \
  --data_path data \
  --ckpt_dir MBA/ckpt_delta \
  --batch_size 8 \
  --num_epochs  500\
  --save_epochs 50 \
  --enable_mba \
  --obj_pose_mode delta

  We put all prediction at the coordinate of ball, as well as eval.

  CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --master_addr 127.0.0.1 --master_port 14524 \
    --nproc_per_node 2 --nnodes 1 --node_rank 0 \
    MBA/train_obj.py \
    --data_path /mnt/data/yanwen/glass_data/ \
    --ckpt_dir MBA/ckpt_1211 \
    --batch_size 128 --num_epochs 1000 --save_epochs 50 --num_workers 24 \
    --lr 4e-4 --seed 233 --enable_mba --obj_pose_mode delta
  
  # 3D Traj Preview

python MBA/dataset/traj_vis.py

# -------------AnyGrasp------------------
# This is amazing. It can literally grasp anything I want.
python glasses_hardware/hardware/anygrasp_debug.py --real_robot

python scripts_data_processing/realsense_record_second.py --keyboard-control

# -------------6D Pose prediction Inference ----------------
python src/egodata_eval/eval.py --ckpt MBA/ckpt_deploy/policy_last.ckpt 

# -------------PiPER ------------------------
cd glasses_hardware/piper_sdk/piper_sdk
bash can_activate.sh can0 1000000
 # Control
python glasses_hardware/hardware/my_device/piper.py

# -----------Visualize point cloud-----------
python src/egodata_eval/vis_pointcloud_sequence.py     --data_path data/moving --seq_index 0 --fps 30

# -----------VGGT Pointcloud Reconstruction-------------

 # IN foundation_stereo ENV
python vggt/glass_demo.py --episode-dir data/20251112_142342

python src/egodata_eval/visualize_scripts/vis_tsdf.py
