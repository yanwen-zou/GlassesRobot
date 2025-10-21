
# ----------------ZED Env---------------
# Recording data recording pipeline launching
ros2 launch egodata_record stereo_record.launch.py 
# pose visualize
ros2 run egodata_record pose_visualizer --show-orientation glasses_pose_20250929_151440.txt

# -------------foundation_stereo Env----------
# SAM2 For Mask -> FoundationStereo For Depth -> FoundationPose for Obj Pose Tracking
./cmd_book_pipeline.sh


# -------------MBA Env------------------
# Train a obj pose prediction model

MPLBACKEND=Agg MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
python MBA/train_obj.py \
  --data_path data_lion \
  --ckpt_dir MBA/ckpt_delta \
  --batch_size 8 \
  --num_epochs  16\
  --save_epochs 2 \
  --enable_mba \
  --obj_pose_mode delta

  We put all prediction at the coordinate of the first frame as reference.
  When doing inference, we transform the traj back using the cam extrinsics.
  
  # Eval Preview
python MBA/dataset/vis_prediction.py --ckpt MBA/ckpt/policy_last.ckpt     --output_video outputs/vis_pose_1014.mp4 --demo_index 0 --full_episode --fps 20 --compare_mode trajectory

python MBA/dataset/vis_prediction.py --ckpt MBA/ckpt_delta/policy_epoch_1000_seed_233.ckpt     --output_video outputs/delta_eval.mp4 --demo_index 0 --full_episod
e --fps 20 --compare_mode trajectory --obj_pose_mode delta --data_path data