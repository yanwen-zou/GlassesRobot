
# ----------------ZED Env---------------
# Recording data recording pipeline launching
ros2 launch egodata_record stereo_record.launch.py 

# Zed&Cam Calibration
ros2 launch egodata_record zed_handeye.launch.py \
  intrinsics:=/home/yanwen/research/unity_comm/src/FoundationStereo/assets/K_ZED.txt \
  intrinsic_scale:=1.0 \
  output:=/home/yanwen/research/unity_comm/glasses_hardware/calib/T_zed_tcp_calib.txt

# after data processing, view:
./src/egodata_eval/visualize_scripts/vis_datset.sh data/train_teapot_2/ --spawn --episode-idx 2

# -------------foundation_stereo Env----------
# SAM2 For Mask -> FoundationStereo For Depth -> FoundationPose for Obj Pose Tracking
./cmd_book_pipeline.sh

./4070_pipeline.sh --data-root data/train_teapot/ --run-fp --mesh-name teapot

# then cleanup intermediate before upload to server.

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

  PYTHONPATH=$(pwd) CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --master_addr 127.0.0.1 --master_port 14524 \
    --nproc_per_node 2 --nnodes 1 --node_rank 0 \
    MBA/train_obj.py \
    --data_path /mnt/data/yanwen/glass_data/train_0207_teapot \
    --ckpt_dir /mnt/data/yanwen/glass_ckpt/0215_teapot \
    --batch_size 64 --num_epochs 1000 --save_epochs 100 --num_workers 24 \
    --lr 2e-4 --seed 233 --enable_mba --obj_pose_mode abs --num_action 10 \
    --resume_ckpt /mnt/data/yanwen/glass_ckpt/0215_teapot/policy_epoch_500_seed_233.ckpt --resume_epoch 500 

  # 3D Traj view

python src/egodata_eval/visualize_scripts/vis_eval.py --data-dir src/egodata_eval/eval_output/20260203_204815/ --spawn

# -------------AnyGrasp------------------
# This is amazing. It can literally grasp anything I want.
python glasses_hardware/hardware/anygrasp_debug.py --real_robot

python scripts_data_processing/realsense_record_second.py --keyboard-control

# -------------6D Pose prediction Inference ----------------

# in train dataset, offline
python src/egodata_eval/eval_dataset.py --data-path data/train_teapot/20260202_162226/ --ckpt ckpt/ckpt_0203_teapot_abs_wo_curr/policy_epoch_1000_seed_233.ckpt --add_curr_cond false --obj-pose-mode abs --task teapot --enable-headpose-head

# vis the output:
python src/egodata_eval/visualize_scripts/vis_train.py --data-dir src/egodata_eval/train_output/episode/20260202_162226_20260206_144411 --spawn

# -- eval
./src/egodata_eval/eval.sh --task book --ckpt ckpt/ckpt_0205_book_abs_wo_curr/policy_epoch_600_seed_233.ckpt --enable-headpose-head --obj-pose-mode abs

# openpi training
export HF_LEROBOT_HOME=/mnt/data/yanwen/glass_data

uv run scripts/compute_norm_stats.py --config-name pi05_realworld

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_realworld --exp-name=openpi_book_0217 --batch-size=64 --overwrite