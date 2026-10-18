################################################################################
# Makefile 配置说明
# 1. 目标：编译 CUDA/C++ 混合项目，生成可执行文件和共享库
# 2. 特性：支持自动依赖检测、多目录构建、增量编译
# 3. 使用方式：
#    - make bin      : 编译生成可执行文件到 build/bin/
#    - make shared   : 编译生成共享库到 build/lib/
#    - make clean    : 清理所有生成文件
################################################################################

default: bin

# 获取当前 Makefile 的绝对路径
HERE := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
ROOT := $(realpath $(HERE))

# 构建目录结构定义
# 构建根目录
BUILD_DIR := $(ROOT)/build
# 头文件目录（暂未使用）
INCLUDE_DIR := $(BUILD_DIR)/include
# 库文件输出目录
LIB_DIR := $(BUILD_DIR)/lib
# 可执行文件输出目录
BIN_DIR := $(BUILD_DIR)/bin

# 第三方库和源码目录
# 第三方依赖目录
THIRD_PARTY_DIR := $(ROOT)/3rd
# Python 相关目录
PYTHON_DIR := $(ROOT)/python
# 源码目录
SRC_DIR := $(ROOT)/src

# CUDA 工具链路径（通过 which nvcc 推断）
CUDA_HOME := $(dir $(shell which nvcc))/..

# 清理目标：删除整个 build 目录
.PHONY: clean
clean:
	@rm -rf $(BUILD_DIR)

################################## compiler #######################################

# CUDA 编译器配置
# -g: Generate debug information for host code
# -G: Generate debug information for device code
# 注意 -O0 和 -G 选项会造成 too many resources requested for launch 问题，仅在需要使用 cuda-gdb 时再添加
NVCXX := nvcc
NVCXXFLAGS := -std=c++17 -arch=sm_90 -Xcompiler -Wall -Dtf,transpose_ \
			  --extended-lambda \
			  -Xcompiler -fPIC -lpthread \
			  --expt-relaxed-constexpr \
			  -I $(SRC_DIR)/include \
			  -I $(SRC_DIR)/cache \
			  -I $(SRC_DIR)/engine \
			  -I $(SRC_DIR)/storage \
			  -I $(THIRD_PARTY_DIR) \
			  -I $(THIRD_PARTY_DIR)/hashmap \
			  -g 
			  

# C++ 编译器配置（暂未使用，保留备用）
CXX := g++
CXXFLAGS := -std=c++17 -fPIC -O2 -Dtf,transpose_ \
			-I $(SRC_DIR)/engine -g

################################## bin ########################################### 
.PHONY: bin
bin: core
	@mkdir -p $(BUILD_DIR)/built
	@mkdir -p $(INCLUDE_DIR)
	@mkdir -p $(LIB_DIR)
	@mkdir -p $(BIN_DIR)
	@echo $(CORE_H)
	$(NVCXX) $(NVCXXFLAGS) \
		-L $(CUDA_HOME)/lib64 -L $(BUILD_DIR)/lib \
		$(CORE_OBJ) $(THIRD_PARTY_OBJ) \
		-lcudart -lpthread -lnvToolsExt \
		-o $(BIN_DIR)/sampling_server -lrt


################################## shared ########################################### 
.PHONY: shared
shared: core
	@mkdir -p $(BUILD_DIR)/built
	@mkdir -p $(INCLUDE_DIR)
	@mkdir -p $(LIB_DIR)
	@mkdir -p $(BIN_DIR)
	@echo $(CORE_H)
	$(NVCXX) $(NVCXXFLAGS) -shared \
		-L $(CUDA_HOME)/lib64 \
		$(CORE_OBJ) $(THIRD_PARTY_OBJ) \
		-lcudart -lpthread -lnvToolsExt \
		-o $(LIB_DIR)/libserver.so -lrt

################################## core ########################################### 
# 核心模块定义
CORE_DIR := $(SRC_DIR)
# 存放编译中间文件
CORE_BUILT_DIR := $(BUILD_DIR)/built

# 自动发现源码结构（最大深度3层）
CORE_DIRS := $(shell find "src" -maxdepth 3 -type d)
CORE_H := $(foreach dir,$(CORE_DIRS),$(wildcard $(dir)/*.h))
CORE_CUH := $(foreach dir,$(CORE_DIRS),$(wildcard $(dir)/*.cuh))
CORE_CU := $(foreach dir,$(CORE_DIRS),$(wildcard $(dir)/*.cu))
# $(CORE_CU) 中的 *.cu 对应的 *.o，最终形成 $(CORE_BUILT_DIR)/xxx/%.o
CORE_OBJ := $(addprefix $(CORE_BUILT_DIR)/,$(patsubst %.cu,%.o,$(CORE_CU)))

THIRD_PARTY_DIRS := $(shell find "3rd" -maxdepth 3 -type d)
THIRD_PARTY_CU := $(filter-out ,$(foreach dir,$(THIRD_PARTY_DIRS),$(wildcard $(dir)/*.cu)))
THIRD_PARTY_OBJ := $(addprefix $(CORE_BUILT_DIR)/,$(patsubst %.cu,%.o,$(THIRD_PARTY_CU)))

# 通用构建规则：处理所有 .cu 文件
$(CORE_BUILT_DIR)/src/%.o:$(CORE_DIR)/%.cu $(CORE_H) $(CORE_CUH)
	@mkdir -p $(CORE_BUILT_DIR)
	@mkdir -p $(CORE_BUILT_DIR)/src
	$(NVCXX) $(NVCXXFLAGS) -c $< -o $@ -lrt
	
$(CORE_BUILT_DIR)/src/cache/%.o:$(CORE_DIR)/cache/%.cu $(CORE_H) $(CORE_CUH)
	@mkdir -p $(CORE_BUILT_DIR)/src/cache
	@echo $<
	$(NVCXX) $(NVCXXFLAGS) -c $< -o $@ -lrt

$(CORE_BUILT_DIR)/src/storage/%.o:$(CORE_DIR)/storage/%.cu $(CORE_H) $(CORE_CUH)
	@mkdir -p $(CORE_BUILT_DIR)/src/storage
	$(NVCXX) $(NVCXXFLAGS) -c $< -o $@ -lrt

$(CORE_BUILT_DIR)/src/engine/%.o:$(CORE_DIR)/engine/%.cu $(CORE_H) $(CORE_CUH)
	@mkdir -p $(CORE_BUILT_DIR)/src/engine
	$(NVCXX) $(NVCXXFLAGS) -c $< -o $@ -lrt

# 处理 third_party-acc_spmm
$(CORE_BUILT_DIR)/3rd/acc-spmm/src/%.o:$(THIRD_PARTY_DIR)/acc-spmm/src/%.cu
	@mkdir -p $(CORE_BUILT_DIR)/3rd/acc-spmm/src
	$(NVCXX) $(NVCXXFLAGS) -c $< -o $@ -lrt

# 声明核心构建目标
.PHONY: core
core: $(CORE_OBJ) $(THIRD_PARTY_OBJ)

.PHONY: debug
debug:
	@echo "SRC_DIR = $(SRC_DIR)"
	@echo
	@echo "CORE_DIR = $(CORE_DIR)"
	@echo
	@echo "$(CORE_DIR)/engine/%.cu"
	@echo
	@echo "CORE_DIRS = $(CORE_DIRS)"
	@echo
	@echo "THIRD_PARTY_DIRS = $(THIRD_PARTY_DIRS)"
	@echo
	@echo "CORE_BUILT_DIR = $(CORE_BUILT_DIR)"
	@echo
	@echo "CORE_CU = $(CORE_CU)"
	@echo
	@echo "THIRD_PARTY_CU = $(THIRD_PARTY_CU)"
	@echo
	@echo "CORE_OBJ = $(CORE_OBJ)"
	@echo
	@echo "THIRD_PARTY_OBJ = $(THIRD_PARTY_OBJ)"
	@echo
	