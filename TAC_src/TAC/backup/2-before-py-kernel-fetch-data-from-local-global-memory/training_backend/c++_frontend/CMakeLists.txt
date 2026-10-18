cmake_minimum_required(VERSION 3.22)
#cmake_policy(SET CMP0148 NEW)

# 项目信息
project(GNN VERSION 1.0.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 编译选项
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -O0")

# torch 依赖
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message(STATUS "torch cxx flags: ${TORCH_CXX_FLAGS}")

# python 依赖
# reference: https://stackoverflow.com/questions/11041299/python-h-no-such-file-or-directory
find_package(Python COMPONENTS Development)

# 包含头文件目录
message(STATUS "Include directories:\n ${PROJECT_SOURCE_DIR}/include \n${PROJECT_SOURCE_DIR}/../include \n${Python_INCLUDE_DIRS}")
include_directories(
    ${Python_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/3rd/include
    ${PROJECT_SOURCE_DIR}/3rd/ipcservice/build/include
    ${PROJECT_SOURCE_DIR}/3rd/spmm_cnic/build/include
)

# 源文件
# 查找所有源文件
message(STATUS "src directories: ${PROJECT_SOURCE_DIR}/src")
file(GLOB_RECURSE SOURCES
    ${PROJECT_SOURCE_DIR}/src/main.cu
)

# 生成可执行文件
add_executable(train ${SOURCES})

# 生成 worker 可执行文件
file(GLOB_RECURSE WORKER_SOURCES
    ${PROJECT_SOURCE_DIR}/src/worker_main.cu
)
add_executable(worker ${WORKER_SOURCES})

# 编译选项
#target_compile_options(train PRIVATE -g -G -O0)

# 链接选项
message(STATUS "link directories:\n ${TORCH_LIBRARIES} \n ${Python_LIBRARIES}")
target_link_libraries(train "${Python_LIBRARIES}")
target_link_libraries(train "${TORCH_LIBRARIES}")
target_link_libraries(train "/home/gaohy/Legions/TAC/training_backend/c++_frontend/3rd/ipcservice/build/lib/libipc_service.so")
target_link_libraries(train "/home/gaohy/Legions/TAC/training_backend/c++_frontend/3rd/spmm_cnic/build/lib/libspmm_cnic.so")
target_link_libraries(train nvToolsExt)

target_link_libraries(worker "${Python_LIBRARIES}")
target_link_libraries(worker "${TORCH_LIBRARIES}")
target_link_libraries(worker "/home/gaohy/Legions/TAC/training_backend/c++_frontend/3rd/ipcservice/build/lib/libipc_service.so")
target_link_libraries(worker "/home/gaohy/Legions/TAC/training_backend/c++_frontend/3rd/spmm_cnic/build/lib/libspmm_cnic.so")
target_link_libraries(worker nvToolsExt)

# 安装头文件

# 包含测试子目录
