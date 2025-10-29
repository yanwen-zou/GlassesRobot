# 如何标定

## eih 标定

首先运行`collect_data1.py`（或者`collect_data1_replay.py`）
以 `collect_data1.py`为例，修改 robot ip, cam id, dir 后运行
采集 30 张不同姿势机械臂拍摄统一位置标定板的图片，机械臂需要有不同角度

然后运行`calibrate_extr_eih.py`，修改相机 intr，dirName，glob

## e2h 标定

首先运行`collect_data2.py`
修改 robot ip, cam id, dir 后运行
采集 30 张任意姿势机械臂拍摄不同标定板的图片，标定版位置需要稳定且丰富

然后运行`calibrate_extr_e2h.py`
