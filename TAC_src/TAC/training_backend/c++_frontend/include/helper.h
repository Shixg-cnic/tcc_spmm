#pragma once

#include <cstdio>
#include <vector>
#include <thread>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <chrono>

#include <nvToolsExt.h>

#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/profiler/orchestration/observer.h>

// 进程组管理的 API 由 torch 提供，将 ​​NCCL 的底层通信功能​​（如 ncclAllReduce）封装成 PyTorch 的分布式接口，运行时仍需要 NCCL 动态库支持
c10::intrusive_ptr<c10d::ProcessGroupNCCL> setup(int rank, int world_size) {
    // 设置主节点地址和端口
    std::string master_addr = "127.0.0.1";
    std::string master_port = "29500";
    
    // 创建 TCP store(基于 TCP 的分布式键值存储实现)
    auto store = c10::make_intrusive<c10d::TCPStore>(
        master_addr, 
        std::stoi(master_port),
        world_size,
        rank == 0,  // isServer
        std::chrono::seconds(30)
    );
    
    // 初始化 NCCL 进程组选项
    auto options = c10d::ProcessGroupNCCL::Options::create();
    options->global_ranks_in_group = std::vector<uint64_t>(world_size);
    for (int i = 0; i < world_size; ++i) {
        options->global_ranks_in_group[i] = i;
    }
    
    // 创建 NCCL 进程组
    c10::intrusive_ptr<c10d::ProcessGroupNCCL> pg = c10::make_intrusive<c10d::ProcessGroupNCCL>(
        store,
        rank,
        world_size,
        options
    );
    
    //c10d::initProcessGroup(pg);

    return pg;
}

void waitWork(
    c10::intrusive_ptr<c10d::ProcessGroupNCCL> pg,
    std::vector<c10::intrusive_ptr<c10d::Work>> works) {
    for (auto& work : works) {
      try {
        work->wait();
      } catch (const std::exception& ex) {
        std::cerr << "Exception received: " << ex.what() << std::endl;
        pg->abort();
      }
    }
}

inline void cleanup(c10::intrusive_ptr<c10d::ProcessGroupNCCL> pg) {
    // Difference between `abort()` and `shutdown()`:
    // 1. `abort()` will signal communicators to terminate all NCCL kernels
    // immediately.
    // 2. `shutdown()` will wait for all NCCL kernels to finish before destroying
    // communicators.

    // Destroy (shutdown) this backend -- normal exit.
    pg->shutdown();
}

// -----------------------------
// Helper types
// -----------------------------
struct SparseMatrixSize {
    int64_t bcsr_num_rows, bcsr_num_cols, bcsr_nnz, bcsr_num_TCBlocks, bcsr_num_rowWindows;
    int64_t bcsc_num_rows, bcsc_num_cols, bcsc_nnz, bcsc_num_TCBlocks, bcsc_num_colWindows;
};

struct SparseMatrixData {
    at::Tensor bcsr_rowWindowOffset, bcsr_tcOffset, bcsr_sparseA2B, bcsr_tcLocalBit;
    at::Tensor bcsc_colWindowOffset, bcsc_tcOffset, bcsc_sparseA2C, bcsc_tcLocalBit;
};
