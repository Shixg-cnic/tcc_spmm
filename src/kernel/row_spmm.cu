// Derived from Acc-SpMM tf32_mma_kernel, MIT (c) 2025 Haisha Zhao.
// Keep sparse-A/mapping double buffers, dense register prefetch and MMA order.
#include "row_format.cuh"
#include "mma_tf32.cuh"
#include "pipeline.cuh"
#include <algorithm>
#include <type_traits>

namespace {
using vint = std::uint32_t;
using MAT_PTR_TYPE = std::uint32_t;
using TCLOCAL_TYPE = std::uint64_t;
using MAT_VAL_TYPE = float;
constexpr int COL_WINDOW_R = 16;

template<bool Bounds>
__device__ __forceinline__ float loadFeature(const float* pointer, vint feature, vint features) {
    if constexpr (Bounds) { if (feature >= features) return 0.f; }
    return load_fp32_from_global(pointer);
}
template<bool Bounds>
__global__ void rowSpmmKernel(
    const TCLOCAL_TYPE* __restrict__    d_tcLocalBit,
    const vint*         __restrict__    d_sparseA2B,
    const MAT_VAL_TYPE* __restrict__    d_valueA,
    const MAT_PTR_TYPE* __restrict__    d_block2Idx,
    const MAT_PTR_TYPE* __restrict__    d_data2Idx,
    const MAT_VAL_TYPE* __restrict__    d_MatB,
    MAT_VAL_TYPE* d_MatC,
    const vint numNodes,
    const vint feature_dim, const vint rows
) {
    using DenseIndex = std::conditional_t<Bounds, std::size_t, vint>;
    using ARegisters = MAT_VAL_TYPE[2];
    using BRegisters = MAT_VAL_TYPE[4];
    using CRegisters = MAT_VAL_TYPE[2][4];
    ARegisters fragA;
    BRegisters fragB00;
    BRegisters fragB01;
    BRegisters fragB10;
    BRegisters fragB11;
    CRegisters fragC = {0.0};

    vint bid                  =   blockIdx.x;
    vint offY                 =   (blockIdx.y << 9);
    const vint laneid         =   31 & threadIdx.x;
    const vint warpSize       =   32;
    const vint tid            =   threadIdx.y * warpSize + laneid;
    const vint local_warpID   =   threadIdx.y;

    vint groupID         =   laneid >> 2;
    vint tID_in_group    =   3 & laneid;

    vint rowA            =   groupID;
    vint colA0           =   tID_in_group;
    vint colA1           =   tID_in_group + 4;

    vint colB02          =   offY + groupID + (local_warpID << 5);
    vint colB13          =   offY + groupID + 8 + (local_warpID << 5);
    vint row01           =   tID_in_group;
    vint row23           =   tID_in_group + 4;

    constexpr const int inst_k  = 8;
    constexpr const int inst_n  = 8;

    const vint mat_len = 64;
    const vint idx_len = 8;
    vint  local_idx    = 0;

    __shared__ MAT_VAL_TYPE d_sharedSparseA[2 * mat_len];
    __shared__ vint         d_sharedSparseA2B[2 * idx_len];

    vint saPtr = __cvta_generic_to_shared(d_sharedSparseA);
    vint siPtr = __cvta_generic_to_shared(d_sharedSparseA2B);

    MAT_PTR_TYPE start_blk_idx  = d_block2Idx[bid];
    MAT_PTR_TYPE end_blk_idx    = d_block2Idx[bid+1];


    if (start_blk_idx == end_blk_idx) {
        const vint width = min(feature_dim - offY, vint{512});
        for (vint i = tid; i < 8 * width; i += blockDim.y * 32) {
            vint row = bid * 8 + i / width;
            if (row < rows) d_MatC[static_cast<std::size_t>(row) * feature_dim + offY + i % width] = 0.f;
        }
        return;
    }

    /* === pre loop === */
    if(tid < mat_len) {
        TCLOCAL_TYPE present_local = d_tcLocalBit[start_blk_idx];
        vint start_dataIdx         = d_data2Idx[start_blk_idx];
        if(present_local & (1ULL << tid))
            local_idx = __popcll(present_local << (63 - tid));
        // prefetch 1 tc_block
        if(local_idx == 0) d_sharedSparseA[tid] = 0.0;
        else d_sharedSparseA[tid] = load_fp32_from_global2shared(d_valueA + start_dataIdx + local_idx - 1);
    }
    // prefetch A2B idx
    if(tid < inst_k) {
        d_sharedSparseA2B[tid] = load_int_from_global(d_sparseA2B + start_blk_idx * inst_k + tid);
        if(start_blk_idx + 1 < end_blk_idx) {
            d_sharedSparseA2B[tid + 8] = load_int_from_global(d_sparseA2B + (start_blk_idx + 1) * inst_k + tid);
        }
    }

    __syncthreads();

    vint dense_rowIdx01 = d_sharedSparseA2B[row01];
    vint dense_rowIdx23 = d_sharedSparseA2B[row23];
    if(dense_rowIdx01 >= numNodes) {
        fragB00[0] = 0.0; fragB00[1] = 0.0;
        fragB01[0] = 0.0; fragB01[1] = 0.0;
    } else {
        DenseIndex sourceIdx0 = static_cast<DenseIndex>(dense_rowIdx01) * feature_dim + colB02;
        DenseIndex sourceIdx1 = static_cast<DenseIndex>(dense_rowIdx01) * feature_dim + colB13;
        fragB00[0] = loadFeature<Bounds>(d_MatB + sourceIdx0, colB02, feature_dim);
        fragB00[1] = loadFeature<Bounds>(d_MatB + sourceIdx1, colB13, feature_dim);
        fragB01[0] = loadFeature<Bounds>(d_MatB + sourceIdx0 + COL_WINDOW_R, colB02 + COL_WINDOW_R, feature_dim);
        fragB01[1] = loadFeature<Bounds>(d_MatB + sourceIdx1 + COL_WINDOW_R, colB13 + COL_WINDOW_R, feature_dim);
    }
    if(dense_rowIdx23 >= numNodes) {
        fragB00[2] = 0.0; fragB00[3] = 0.0;
        fragB01[2] = 0.0; fragB01[3] = 0.0;
    } else {
        DenseIndex sourceIdx0 = static_cast<DenseIndex>(dense_rowIdx23) * feature_dim + colB02;
        DenseIndex sourceIdx1 = static_cast<DenseIndex>(dense_rowIdx23) * feature_dim + colB13;
        fragB00[2] = loadFeature<Bounds>(d_MatB + sourceIdx0, colB02, feature_dim);
        fragB00[3] = loadFeature<Bounds>(d_MatB + sourceIdx1, colB13, feature_dim);
        fragB01[2] = loadFeature<Bounds>(d_MatB + sourceIdx0 + COL_WINDOW_R, colB02 + COL_WINDOW_R, feature_dim);
        fragB01[3] = loadFeature<Bounds>(d_MatB + sourceIdx1 + COL_WINDOW_R, colB13 + COL_WINDOW_R, feature_dim);
    }

    /* === main loop === */
    for(vint tc_block = start_blk_idx + 1; tc_block < end_blk_idx; ++tc_block) {
        vint sel_shm       =   ((tc_block - start_blk_idx + 1) & 1) << 6;   // 0->iter1
        vint sel_shm_next  =   ((tc_block - start_blk_idx ) & 1) << 6;  // 1->iter1
        vint sel_idx_shm       =   ((tc_block - start_blk_idx + 1) & 1) << 3;
        vint sel_idx_shm_next  =   ((tc_block - start_blk_idx ) & 1) << 3;

        vint dense_rowIdx101 = d_sharedSparseA2B[sel_idx_shm_next + row01];
        vint dense_rowIdx123 = d_sharedSparseA2B[sel_idx_shm_next + row23];
        if(sel_shm_next) {
            if(dense_rowIdx101 >= numNodes) {
                fragB10[0] = 0.0; fragB10[1] = 0.0;
                fragB11[0] = 0.0; fragB11[1] = 0.0;
            } else {
                DenseIndex sourceIdx0 = static_cast<DenseIndex>(dense_rowIdx101) * feature_dim + colB02;
                DenseIndex sourceIdx1 = static_cast<DenseIndex>(dense_rowIdx101) * feature_dim + colB13;
                fragB10[0] = loadFeature<Bounds>(d_MatB + sourceIdx0, colB02, feature_dim);
                fragB10[1] = loadFeature<Bounds>(d_MatB + sourceIdx1, colB13, feature_dim);
                fragB11[0] = loadFeature<Bounds>(d_MatB + sourceIdx0 + COL_WINDOW_R, colB02 + COL_WINDOW_R, feature_dim);
                fragB11[1] = loadFeature<Bounds>(d_MatB + sourceIdx1 + COL_WINDOW_R, colB13 + COL_WINDOW_R, feature_dim);
            }
            if(dense_rowIdx123 >= numNodes) {
                fragB10[2] = 0.0; fragB10[3] = 0.0;
                fragB11[2] = 0.0; fragB11[3] = 0.0;
            } else {
                DenseIndex sourceIdx0 = static_cast<DenseIndex>(dense_rowIdx123) * feature_dim + colB02;
                DenseIndex sourceIdx1 = static_cast<DenseIndex>(dense_rowIdx123) * feature_dim + colB13;
                fragB10[2] = loadFeature<Bounds>(d_MatB + sourceIdx0, colB02, feature_dim);
                fragB10[3] = loadFeature<Bounds>(d_MatB + sourceIdx1, colB13, feature_dim);
                fragB11[2] = loadFeature<Bounds>(d_MatB + sourceIdx0 + COL_WINDOW_R, colB02 + COL_WINDOW_R, feature_dim);
                fragB11[3] = loadFeature<Bounds>(d_MatB + sourceIdx1 + COL_WINDOW_R, colB13 + COL_WINDOW_R, feature_dim);
            }
        } else {
            if(dense_rowIdx101 >= numNodes) {
                fragB00[0] = 0.0; fragB00[1] = 0.0;
                fragB01[0] = 0.0; fragB01[1] = 0.0;
            } else {
                DenseIndex sourceIdx0 = static_cast<DenseIndex>(dense_rowIdx101) * feature_dim + colB02;
                DenseIndex sourceIdx1 = static_cast<DenseIndex>(dense_rowIdx101) * feature_dim + colB13;
                fragB00[0] = loadFeature<Bounds>(d_MatB + sourceIdx0, colB02, feature_dim);
                fragB00[1] = loadFeature<Bounds>(d_MatB + sourceIdx1, colB13, feature_dim);
                fragB01[0] = loadFeature<Bounds>(d_MatB + sourceIdx0 + COL_WINDOW_R, colB02 + COL_WINDOW_R, feature_dim);
                fragB01[1] = loadFeature<Bounds>(d_MatB + sourceIdx1 + COL_WINDOW_R, colB13 + COL_WINDOW_R, feature_dim);
            }
            if(dense_rowIdx123 >= numNodes) {
                fragB00[2] = 0.0; fragB00[3] = 0.0;
                fragB01[2] = 0.0; fragB01[3] = 0.0;
            } else {
                DenseIndex sourceIdx0 = static_cast<DenseIndex>(dense_rowIdx123) * feature_dim + colB02;
                DenseIndex sourceIdx1 = static_cast<DenseIndex>(dense_rowIdx123) * feature_dim + colB13;
                fragB00[2] = loadFeature<Bounds>(d_MatB + sourceIdx0, colB02, feature_dim);
                fragB00[3] = loadFeature<Bounds>(d_MatB + sourceIdx1, colB13, feature_dim);
                fragB01[2] = loadFeature<Bounds>(d_MatB + sourceIdx0 + COL_WINDOW_R, colB02 + COL_WINDOW_R, feature_dim);
                fragB01[3] = loadFeature<Bounds>(d_MatB + sourceIdx1 + COL_WINDOW_R, colB13 + COL_WINDOW_R, feature_dim);
            }
        }

        /* === START ASYNC COPY === */
        local_idx = 0;
        if(tid < mat_len) {
            TCLOCAL_TYPE present_local = d_tcLocalBit[tc_block];
            vint         start_dataIdx = d_data2Idx[tc_block];
            if(present_local & (1ULL << tid))
                local_idx = __popcll(present_local << (63 - tid));
            if(local_idx == 0) d_sharedSparseA[sel_shm_next + tid] = 0.0;
            else async_copy(saPtr + ((sel_shm_next + tid) << 2), d_valueA + start_dataIdx + local_idx - 1);
        }
        if(tid < inst_k) {
            if(tc_block + 1 < end_blk_idx)
                async_copy_idx(siPtr + ((sel_idx_shm + tid) << 2), d_sparseA2B + (tc_block + 1) * inst_k + tid);
        }
        /* === END OF ASYNC COPY === */
        // fetch A
        fragA[0] = d_sharedSparseA[sel_shm + rowA * inst_n + colA0];
        fragA[1] = d_sharedSparseA[sel_shm + rowA * inst_n + colA1];

        if(sel_shm_next) {
            tf32_m16n8k8(fragB00, fragA, fragC[0]);
            tf32_m16n8k8(fragB01, fragA, fragC[1]);
        } else {
            tf32_m16n8k8(fragB10, fragA, fragC[0]);
            tf32_m16n8k8(fragB11, fragA, fragC[1]);
        }

        wait_group();
		__syncthreads();
    }

    /* === end loop === */
    vint smem_sel  = ((end_blk_idx - start_blk_idx + 1) & 1) << 6;
    fragA[0] = d_sharedSparseA[smem_sel + rowA * inst_n + colA0];
    fragA[1] = d_sharedSparseA[smem_sel + rowA * inst_n + colA1];

    if(!smem_sel) {
        tf32_m16n8k8(fragB00, fragA, fragC[0]);
        tf32_m16n8k8(fragB01, fragA, fragC[1]);
    } else {
        tf32_m16n8k8(fragB10, fragA, fragC[0]);
        tf32_m16n8k8(fragB11, fragA, fragC[1]);
    }

    vint colC  =  0;
    vint rowC  =  0;

    DenseIndex outOff = static_cast<DenseIndex>(bid << 3) * feature_dim + (local_warpID << 5) + offY;
    #pragma unroll
    for(vint i = 0; i < 4; ++i) {
        rowC = (tID_in_group << 1) + (i & 0x1);
        if(i < 2) colC = groupID;
        else colC = groupID + 8;
        if (!Bounds || (bid * 8 + rowC < rows && (local_warpID << 5) + offY + colC < feature_dim))
            store_fp32_to_global(d_MatC + outOff + rowC * feature_dim + colC, fragC[0][i]);
        if (!Bounds || (bid * 8 + rowC < rows && (local_warpID << 5) + offY + colC + COL_WINDOW_R < feature_dim))
            store_fp32_to_global(d_MatC + outOff + rowC * feature_dim + colC + COL_WINDOW_R, fragC[1][i]);
    }
}


}
cudaError_t launchRowSpMM(const std::uint32_t* window, const std::uint32_t* offset,
    const std::uint32_t* mapping, const std::uint64_t* bits, const float* values,
    const float* x, float* y, std::uint32_t rows, std::uint32_t cols,
    std::uint32_t features, cudaStream_t stream) {
    if (!rows || !cols || !features) return cudaErrorInvalidValue;
    dim3 block(32, std::min(16u, std::max(2u, (features + 31) / 32)));
    dim3 grid((rows + 7) / 8, (features + 511) / 512);
    bool bounded = rows % 8 || features < 64 || features % 32 || (features > 512 && features % 512) ||
                   std::uint64_t(cols) * features > UINT32_MAX || std::uint64_t(rows) * features > UINT32_MAX;
    if (bounded)
        rowSpmmKernel<true><<<grid, block, 0, stream>>>(bits, mapping, values, window, offset, x, y, cols, features, rows);
    else
        rowSpmmKernel<false><<<grid, block, 0, stream>>>(bits, mapping, values, window, offset, x, y, cols, features, rows);
    return cudaGetLastError();
}
cudaError_t rowKernelAttributes(int* out) {
    cudaFuncAttributes a;
    auto error = cudaFuncGetAttributes(&a, rowSpmmKernel<false>);
    if (error != cudaSuccess) return error;
    out[0]=a.numRegs; out[1]=a.sharedSizeBytes; out[2]=a.localSizeBytes;
    error = cudaFuncGetAttributes(&a, rowSpmmKernel<true>);
    out[3]=a.numRegs; out[4]=a.sharedSizeBytes; out[5]=a.localSizeBytes;
    return error;
}
