cmake_minimum_required(VERSION 3.15)
project(mycpp LANGUAGES CXX)

# ==== 基础设置 ====
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3 -Wall")

# ==== 查找依赖 ====
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED COMPONENTS system program_options)
find_package(Eigen3 REQUIRED)

# ==== Python 与 pybind11 ====
# Python include 与 pybind11 header 路径（pip 安装的）
set(PYTHON_INCLUDE_DIR /home/yuwenye/miniconda3/envs/glasses/include/python3.10)
set(PYTHON_LIBRARY /home/yuwenye/miniconda3/envs/glasses/lib/libpython3.10.so)
set(PYBIND11_INCLUDE_DIR /home/yuwenye/miniconda3/envs/glasses/lib/python3.10/site-packages/pybind11/include)

include_directories(
    ${PYTHON_INCLUDE_DIR}
    /home/yuwenye/miniconda3/envs/glasses/include/python3.10  # 强制再补一次
    ${PYBIND11_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    include
)


link_directories(${Python_LIBRARY})

# ==== 源文件 ====
file(GLOB MY_SRC ${PROJECT_SOURCE_DIR}/src/*.cpp)

# ==== 构建 Python 扩展模块 ====
add_library(mycpp MODULE src/app/pybind_api.cpp ${MY_SRC})

target_link_libraries(mycpp
    PRIVATE
    ${Boost_LIBRARIES}
    OpenMP::OpenMP_CXX
    Eigen3::Eigen
    ${PYTHON_LIBRARY}  # 确保链接到正确的 Python 库
)

# ==== 设置输出为 Python 可识别的 .so 模块 ====
set_target_properties(mycpp PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)

# ==== 输出信息 ====
message(STATUS "✅ pybind11 include: ${PYBIND11_INCLUDE_DIR}")
message(STATUS "✅ Python include: ${PYTHON_INCLUDE_DIR}")
message(STATUS "✅ Eigen include: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "✅ Python library: ${PYTHON_LIBRARY}")
