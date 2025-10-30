# --------record realsense video------------
python data_processing_scripts/realsense_record_second.py

# --------clip depth > 1000mm --------------
python /home/yuwenye/yanwen/GlassesRobot/data_processing_scripts/zero_depth_ge_threshold.py

# --------run sam, 50frames one prompt ------------
python /home/yuwenye/yanwen/GlassesRobot/data_processing_scripts/run_sam_for_timestamp.py

# --------then mv the episode to bundlesdf/data and follow its repo------