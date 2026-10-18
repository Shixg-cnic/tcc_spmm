cmake_minimum_required(VERSION 3.22)
#cmake_policy(SET CMP0148 NEW)

# 项目信息
project(GNN VERSION 1.0.0 LANGUAGES CXX CUDA)

# 设置CUDA架构
set(CMAKE_CUDA_ARCHITECTURES "90")

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#set(CMAKE_BUILD_TYPE Debug)

# 编译选项
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -O0")

# torch 依赖
# import torch;print(torch.utils.cmake_prefix_path) 获取 CMAKE_PREFIX_PATH
#set(CMAKE_PREFIX_PATH /home/gaohy/usr/local/miniconda3/envs/gaohy_legion/lib/python3.9/site-packages/torch/share/cmake)
set(CMAKE_PREFIX_PATH /work/gaohy/usr/local/libtorch-shared-with-deps-2.4.1+cu124/share/cmake)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message(STATUS 
    "torch libs: ${TORCH_LIBRARIES}\n\n"
    "torch include: ${TORCH_INCLUDE_DIRS}\n\n"
    "torch cxx flags: ${TORCH_CXX_FLAGS}\n"
)
# python 依赖
# reference: https://stackoverflow.com/questions/11041299/python-h-no-such-file-or-directory
find_package(Python COMPONENTS Development)

# 包含头文件目录
# 实测即使不手动添加 ${TORCH_INCLUDE_DIRS}，只要添加 target_link_libraries 即可自动处理头文件依赖
message(STATUS "Include directories:\n ${PROJECT_SOURCE_DIR}/include \n${PROJECT_SOURCE_DIR}/../include \n${Python_INCLUDE_DIRS}")
include_directories(
    ${Python_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/3rd/include
    ${PROJECT_SOURCE_DIR}/3rd/ipcservice/build/include
    ${PROJECT_SOURCE_DIR}/3rd/spmm_cnic/build/include
)

# 源文件
# 查找所有源文件
message(STATUS "src directories: ${PROJECT_SOURCE_DIR}/src")
file(GLOB_RECURSE SOURCES
    ${PROJECT_SOURCE_DIR}/src/train.cu
)

# 生成可执行文件
add_executable(train ${SOURCES})

# 生成 worker 可执行文件
#file(GLOB_RECURSE WORKER_SOURCES
#    ${PROJECT_SOURCE_DIR}/src/worker_main.cu
#)
#add_executable(worker ${WORKER_SOURCES})

# 生成 worker_with_model 可执行文件
file(GLOB_RECURSE WORKER_SOURCES
    ${PROJECT_SOURCE_DIR}/src/worker_with_model.cu
)
add_executable(worker_with_model ${WORKER_SOURCES})

# 编译选项
#target_compile_options(train PRIVATE -g -G -O0)

# 链接选项
message(STATUS "link directories:\n ${TORCH_LIBRARIES} \n ${Python_LIBRARIES}")
target_link_libraries(train "${Python_LIBRARIES}")
target_link_libraries(train "${TORCH_LIBRARIES}")
target_link_libraries(train "/home/gaohy/Legions/TAC/training_backend/c++_frontend/3rd/ipcservice/build/lib/libipc_service.so")
target_link_libraries(train "/home/gaohy/Legions/TAC/training_backend/c++_frontend/3rd/spmm_cnic/build/lib/libspmm_cnic.so")
target_link_libraries(train nvToolsExt)

#target_link_libraries(worker "${Python_LIBRARIES}")
#target_link_libraries(worker "${TORCH_LIBRARIES}")
#target_link_libraries(worker "/home/gaohy/Legions/TAC/training_backend/c++_frontend/3rd/ipcservice/build/lib/libipc_service.so")
#target_link_libraries(worker "/home/gaohy/Legions/TAC/training_backend/c++_frontend/3rd/spmm_cnic/build/lib/libspmm_cnic.so")
#target_link_libraries(worker nvToolsExt)

target_link_libraries(worker_with_model "${Python_LIBRARIES}")
target_link_libraries(worker_with_model "${TORCH_LIBRARIES}")
target_link_libraries(worker_with_model "/home/gaohy/Legions/TAC/training_backend/c++_frontend/3rd/ipcservice/build/lib/libipc_service.so")
target_link_libraries(worker_with_model "/home/gaohy/Legions/TAC/training_backend/c++_frontend/3rd/spmm_cnic/build/lib/libspmm_cnic.so")
target_link_libraries(worker_with_model nvToolsExt)

# 安装头文件

# 包含测试子目录
