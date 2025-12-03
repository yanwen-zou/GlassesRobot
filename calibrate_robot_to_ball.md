# record data
```bash
python scripts_data_processing/realsense_record_second.py --keyboard-control
```

# SAM2 get mask of balls
```bash
# change data path
python scripts_calib_balls/compute_base_from_ball_centers.py \
        --ball-centers data/train/20251125_210453/ball_centers.txt \
        --npy-output data/train/20251125_210453/cam_to_base.npy
```

# calculate ball centers
```bash
# change data path
python scripts_calib_balls/calculate_ball_centers.py \
        --data-dir data/train/20251125_210453 \
        --output data/train/20251125_210453/ball_centers.txt
```

# calculate robot to base transform
```bash
# change data path
python scripts_calib_balls/calc_robot_to_base.py --cam_pose data/20251128_143254/robot_to_cam.npy --cam_to_base data/20251128_143254/cam_to_base.npy
```

# visualize
```bash
python src/egodata_eval/vis_sequence_robot_frames.py \ 
        --data-dir data/20251128_143254 \
        --robot-to-cam-npy data/20251128_143254/robot_to_cam.npy \ 
        --cam-to-base-npy data/20251128_143254/cam_to_base.npy
```