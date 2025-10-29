import importlib.util
import os

# 指定路径
so_path = os.path.join(os.path.dirname(__file__), "mycpp/build/mycpp.so")

# 加载模块
spec = importlib.util.spec_from_file_location("mycpp", so_path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

# 输出验证
print("✅ Loaded:", m.__file__)
print("✅ Functions:", dir(m))

# 测试调用
rot_grid = m.cluster_poses(30, 99999, 1, 1)  # 根据实际参数调整
print("✅ Rot Grid:", rot_grid)
