import os
import sys
import importlib.util

# 动态加载 mycpp.so 文件
build_dir = os.path.join(os.path.dirname(__file__), "build")
so_path = os.path.join(build_dir, "mycpp.so")

if os.path.exists(so_path):
    spec = importlib.util.spec_from_file_location("mycpp", so_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    globals().update(vars(m))  # 更新全局变量，确保可以使用 mycpp 中的函数
else:
    raise ImportError(f"❌ Cannot find compiled mycpp.so at {so_path}")
