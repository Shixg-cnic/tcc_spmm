cmake_minimum_required(VERSION 3.22)
#cmake_policy(SET CMP0148 NEW)

# 项目信息
project(SpMM_CNIC VERSION 1.0.0 LANGUAGES CXX CUDA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Release)   # 尝试解决 ncu 提示以 debug 模式进行编译的情况

# 编译选项
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -O0")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

# torch 依赖
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message(STATUS "torch cxx flags: ${TORCH_CXX_FLAGS}")

# python 依赖
# reference: https://stackoverflow.com/questions/11041299/python-h-no-such-file-or-directory
# find_package(PythonLibs REQUIRED) # if policy CMP0148 is not set to NEW(https://cmake.org/cmake/help/latest/module/FindPythonLibs.html)
find_package(Python COMPONENTS Development) # https://cmake.org/cmake/help/latest/module/FindPython.html#module:FindPython

# 包含头文件目录
message(STATUS "Include directories:\n ${PROJECT_SOURCE_DIR}/include \n${PROJECT_SOURCE_DIR}/../include \n${Python_INCLUDE_DIRS}")
include_directories(
    ${Python_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/../include
)

# 源文件
# 查找所有源文件
message(STATUS "src directories: ${PROJECT_SOURCE_DIR}/src")
file(GLOB_RECURSE SOURCES
    ${PROJECT_SOURCE_DIR}/src/*.cpp
    ${PROJECT_SOURCE_DIR}/src/*.cu
)

# 生成静态库
message(STATUS "lib output directories: ${CMAKE_BINARY_DIR}/lib")
#add_library(IpcService STATIC ${SOURCES})
#set_target_properties(IpcService PROPERTIES 
#    OUTPUT_NAME "ipc_service"
#    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
#)

# 生成动态库
# 使用静态库时 main.cpp 最终链接存在 libpython 符号不存在的问题（尚不清楚原因）
add_library(SpMM_CNIC SHARED ${SOURCES})
set_target_properties(SpMM_CNIC PROPERTIES 
    OUTPUT_NAME "spmm_cnic"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# 编译选项
#target_compile_options(SpMM_CNIC PRIVATE -g -G -O0)

# 链接选项
target_link_libraries(SpMM_CNIC "${TORCH_LIBRARIES}")
target_link_libraries(SpMM_CNIC "${Python_LIBRARIES}")

# 安装目标
#install(TARGETS GraphAlgorithmLibStatic GraphAlgorithmLibShared
#        LIBRARY DESTINATION lib
#        ARCHIVE DESTINATION lib
#        RUNTIME DESTINATION bin)

# 安装头文件
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION ${CMAKE_BINARY_DIR}/include)

# 包含测试子目录
#add_subdirectory(test)
