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
# 注意 torch 的相关头文件还需要隐式 include nccl 相关头文件，需要依赖 .bashrc 中的 nccl 配置项
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

# 生成可执行文件
file(GLOB_RECURSE SOURCES
    ${PROJECT_SOURCE_DIR}/src/train.cu
)
add_executable(train ${SOURCES})

# 生成 worker 可执行文件（无计算，仅完成 ipc 通信交互逻辑）
#file(GLOB_RECURSE WORKER_SOURCES
#    ${PROJECT_SOURCE_DIR}/src/worker_only_ipc.cu
#)
#add_executable(worker ${WORKER_SOURCES})

# 生成 worker 可执行文件（无模型，用于获取 denseC_size）
#file(GLOB_RECURSE WORKER_SOURCES
#    ${PROJECT_SOURCE_DIR}/src/worker_without_model.cu
#)
#add_executable(worker ${WORKER_SOURCES})

# 生成 worker_with_model 可执行文件
#file(GLOB_RECURSE WORKER_SOURCES
#    ${PROJECT_SOURCE_DIR}/src/worker_with_model_only_first_layer_ax.cu
#)
#add_executable(worker ${WORKER_SOURCES})

# 生成 worker_with_model 可执行文件
#file(GLOB_RECURSE WORKER_SOURCES
#    ${PROJECT_SOURCE_DIR}/src/worker_with_model.cu
#)
#add_executable(worker ${WORKER_SOURCES})

# 生成 gat 可执行文件
file(GLOB_RECURSE WORKER_SOURCES
    ${PROJECT_SOURCE_DIR}/src/worker_gat.cu
)
add_executable(worker ${WORKER_SOURCES})

# 编译选项
#target_compile_options(train PRIVATE -g -G -O0)

# 可执行程序公共链接库列表
set(COMMON_LIBS
    "${Python_LIBRARIES}"
    "${TORCH_LIBRARIES}"
    "/work/gaohy/Legions/TAC/training_backend/c++_frontend/3rd/ipcservice/build/lib/libipc_service.so"
    "/work/gaohy/Legions/TAC/training_backend/c++_frontend/3rd/spmm_cnic/build/lib/libspmm_cnic.so"
    nvToolsExt
)

# 所有目标文件列表
set(ALL_TARGETS
    train
    worker
)

## 为可执行文件设置 RPATH，避免 cuda lib 路径冲突（不管用，警告依然存在）
#set(CUDA_LIB_PATH "/home/gaohy/usr/local/cuda-12.4/lib64")
#foreach(TARGET IN ITEMS ${ALL_TARGETS})
#    # 明确设置运行时搜索路径 (RPATH)
#    set_target_properties(${TARGET} PROPERTIES
#        # 运行时搜索路径，优先确保 CUDA 库被找到
#        BUILD_RPATH ${CUDA_LIB_PATH}
#        INSTALL_RPATH ${CUDA_LIB_PATH}
#    )
#endforeach()

# 遍历所有目标，添加链接选项
foreach(TARGET IN ITEMS ${ALL_TARGETS})
    target_link_libraries(${TARGET} PRIVATE ${COMMON_LIBS})
endforeach()

## 链接选项
#message(STATUS "link directories:\n ${TORCH_LIBRARIES} \n ${Python_LIBRARIES}")
#target_link_libraries(train "${Python_LIBRARIES}")
#target_link_libraries(train "${TORCH_LIBRARIES}")
#target_link_libraries(train "/work/gaohy/Legions/TAC/training_backend/c++_frontend/3rd/ipcservice/build/lib/libipc_service.so")
#target_link_libraries(train "/work/gaohy/Legions/TAC/training_backend/c++_frontend/3rd/spmm_cnic/build/lib/libspmm_cnic.so")
#target_link_libraries(train nvToolsExt)
#
#target_link_libraries(worker_only_ipc "${Python_LIBRARIES}")
#target_link_libraries(worker_only_ipc "${TORCH_LIBRARIES}")
#target_link_libraries(worker_only_ipc "/work/gaohy/Legions/TAC/training_backend/c++_frontend/3rd/ipcservice/build/lib/libipc_service.so")
#target_link_libraries(worker_only_ipc "/work/gaohy/Legions/TAC/training_backend/c++_frontend/3rd/spmm_cnic/build/lib/libspmm_cnic.so")
#target_link_libraries(worker_only_ipc nvToolsExt)
#
##target_link_libraries(worker_without_model "${Python_LIBRARIES}")
##target_link_libraries(worker_without_model "${TORCH_LIBRARIES}")
##target_link_libraries(worker_without_model "/work/gaohy/Legions/TAC/training_backend/c++_frontend/3rd/ipcservice/build/lib/libipc_service.so")
##target_link_libraries(worker_without_model "/work/gaohy/Legions/TAC/training_backend/c++_frontend/3rd/spmm_cnic/build/lib/libspmm_cnic.so")
##target_link_libraries(worker_without_model nvToolsExt)
#
#target_link_libraries(worker_with_model_only_first_layer_ax "${Python_LIBRARIES}")
#target_link_libraries(worker_with_model_only_first_layer_ax "${TORCH_LIBRARIES}")
#target_link_libraries(worker_with_model_only_first_layer_ax "/work/gaohy/Legions/TAC/training_backend/c++_frontend/3rd/ipcservice/build/lib/libipc_service.so")
#target_link_libraries(worker_with_model_only_first_layer_ax "/work/gaohy/Legions/TAC/training_backend/c++_frontend/3rd/spmm_cnic/build/lib/libspmm_cnic.so")
#target_link_libraries(worker_with_model_only_first_layer_ax nvToolsExt)
#
#target_link_libraries(worker_with_model "${Python_LIBRARIES}")
#target_link_libraries(worker_with_model "${TORCH_LIBRARIES}")
#target_link_libraries(worker_with_model "/work/gaohy/Legions/TAC/training_backend/c++_frontend/3rd/ipcservice/build/lib/libipc_service.so")
#target_link_libraries(worker_with_model "/work/gaohy/Legions/TAC/training_backend/c++_frontend/3rd/spmm_cnic/build/lib/libspmm_cnic.so")
#target_link_libraries(worker_with_model nvToolsExt)
#
## 安装头文件
#
## 包含测试子目录
