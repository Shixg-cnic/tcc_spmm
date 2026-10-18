#pragma once

#include <vector>
#include <map>
#include <unordered_map>
#include <bitset>
#include <iostream>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include "config.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/unique.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include <cuda_profiler_api.h>

template <class dataType>
struct COO {
    vint rows, cols, nnz;
    vint *row, *col;

    COO(vint num_rows, vint num_cols, vint nnz, 
        vint *row_indices, vint *col_indices);
    ~COO();

    std::string toString() const;
};

template <typename dataType>
class BCSR {
public:
    vint num_rows, num_cols, nnz;
    vint num_TCBlocks;      // TC 块数量
    vint num_rowWindows;    // ROW_WINDOW 数量

    vint            *rowWindowOffset;
    vint            *tcOffset;
    vint            *sparseA2B;
    TCLOCAL_TYPE    *tcLocalBit;

    explicit BCSR(const COO<dataType>& coo, vint* rowWindowOffset, vint* tcOffset, vint* sparseA2B, TCLOCAL_TYPE* tcLocalBit, int pipe);

    std::string toString() const;

// 理论上应该属于 private，但是如果要在类的成员函数中使用扩展的 __device__ lambda，该成员函数必须是 ​public 访问权限
public:
    void COO2BCSR(thrust::device_vector<vint> &x, thrust::device_vector<vint> &y, int pipe);
};

template <class dataType>
class BCSC {
public:
    vint num_rows, num_cols, nnz;
    vint num_TCBlocks;  // TC 块数量
    vint num_colWindows;    // col_window 数量

    vint            *colWindowOffset;
    vint            *tcOffset;
    vint            *sparseA2C;   // 应该从 CSR 的列号转变为行号了，将非连续的行号进行连续化
    TCLOCAL_TYPE    *tcLocalBit;

    explicit BCSC(const COO<dataType>& coo, vint* colWindowOffset, vint* tcOffset, vint* sparseA2C, TCLOCAL_TYPE* tcLocalBit, int pipe);

    std::string toString() const;

// 同 BCSR
public:
    void COO2BCSC(thrust::device_vector<vint> &x, thrust::device_vector<vint> &y, int pipe);
};