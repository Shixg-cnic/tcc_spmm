#include "format.hpp"
#include "int4_format.cuh"
#include "csr.cuh"
#include "row_format.cuh"
// Fused INT4 X load: given the linear FP32 index idx = row*features+col,
// read the packed nibble and dequantise q*scale[col].
__device__ __forceinline__ float int4X(const std::uint8_t* q, const float* scale,
    vint features, std::size_t idx) {
    vint row = vint(idx / features), col = vint(idx % features);
    return int4ToFloat(nibbleAt(q, packedByteIndex(row, col, features), col), scale[row]);
}
/**
 * @file mma_tf32.cuh
 * @author Haisha Zhao
 * @date 2025-04-02
 * 
 * @copyright MIT License (c) 2025 Haisha Zhao
*/

#pragma once
#include "common.hpp"
#include "ptx_tf32.hpp"

__global__
void tf32_mma_kernel_int4(
    const TCLOCAL_TYPE* __restrict__    d_tcLocalBit, 
    const vint*         __restrict__    d_sparseA2B,
    const MAT_VAL_TYPE* __restrict__    d_valueA,
    const MAT_PTR_TYPE* __restrict__    d_block2Idx,
    const MAT_PTR_TYPE* __restrict__    d_data2Idx,
    const std::uint8_t* __restrict__ d_q,
    const float*         __restrict__ d_scale, MAT_VAL_TYPE* d_MatC,
    const vint numNodes,
    const vint feature_dim
) {
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
    vint offY                 =   (blockIdx.y << 7);
    const vint laneid         =   31 & threadIdx.x;
    const vint warpSize       =   32;
    const vint tid            =   threadIdx.y * warpSize + laneid; 
    const vint local_warpID   =   threadIdx.y;

    vint groupID         =   laneid >> 2;
    vint tID_in_group    =   3 & laneid;

    vint rowA            =   groupID;
    vint colA0           =   tID_in_group;
    vint colA1           =   tID_in_group + 4;

    vint colB02          =   groupID + (local_warpID << 5);
    vint colB13          =   groupID + 8 + (local_warpID << 5);
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
    if(dense_rowIdx01 > numNodes) {
        fragB00[0] = 0.0; fragB00[1] = 0.0; 
        fragB01[0] = 0.0; fragB01[1] = 0.0;
    } else {
        vint sourceIdx0 = dense_rowIdx01 * feature_dim + colB02;
        vint sourceIdx1 = dense_rowIdx01 * feature_dim + colB13;
        fragB00[0] = int4X(d_q, d_scale, feature_dim, sourceIdx0);
        fragB00[1] = int4X(d_q, d_scale, feature_dim, sourceIdx1);
        fragB01[0] = int4X(d_q, d_scale, feature_dim, sourceIdx0 + COL_WINDOW_R);
        fragB01[1] = int4X(d_q, d_scale, feature_dim, sourceIdx1 + COL_WINDOW_R);
    }
    if(dense_rowIdx23 > numNodes) {
        fragB00[2] = 0.0; fragB00[3] = 0.0; 
        fragB01[2] = 0.0; fragB01[3] = 0.0;
    } else {
        vint sourceIdx0 = dense_rowIdx23 * feature_dim + colB02;
        vint sourceIdx1 = dense_rowIdx23 * feature_dim + colB13;
        fragB00[2] = int4X(d_q, d_scale, feature_dim, sourceIdx0);
        fragB00[3] = int4X(d_q, d_scale, feature_dim, sourceIdx1);
        fragB01[2] = int4X(d_q, d_scale, feature_dim, sourceIdx0 + COL_WINDOW_R);
        fragB01[3] = int4X(d_q, d_scale, feature_dim, sourceIdx1 + COL_WINDOW_R);
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
            if(dense_rowIdx101 > numNodes) {
                fragB10[0] = 0.0; fragB10[1] = 0.0; 
                fragB11[0] = 0.0; fragB11[1] = 0.0;
            } else {
                vint sourceIdx0 = dense_rowIdx101 * feature_dim + colB02;
                vint sourceIdx1 = dense_rowIdx101 * feature_dim + colB13;
                fragB10[0] = int4X(d_q, d_scale, feature_dim, sourceIdx0);
                fragB10[1] = int4X(d_q, d_scale, feature_dim, sourceIdx1);
                fragB11[0] = int4X(d_q, d_scale, feature_dim, sourceIdx0 + COL_WINDOW_R);
                fragB11[1] = int4X(d_q, d_scale, feature_dim, sourceIdx1 + COL_WINDOW_R);
            }
            if(dense_rowIdx123 > numNodes) {
                fragB10[2] = 0.0; fragB10[3] = 0.0; 
                fragB11[2] = 0.0; fragB11[3] = 0.0;
            } else {
                vint sourceIdx0 = dense_rowIdx123 * feature_dim + colB02;
                vint sourceIdx1 = dense_rowIdx123 * feature_dim + colB13;
                fragB10[2] = int4X(d_q, d_scale, feature_dim, sourceIdx0);
                fragB10[3] = int4X(d_q, d_scale, feature_dim, sourceIdx1);
                fragB11[2] = int4X(d_q, d_scale, feature_dim, sourceIdx0 + COL_WINDOW_R);
                fragB11[3] = int4X(d_q, d_scale, feature_dim, sourceIdx1 + COL_WINDOW_R);
            }
        } else {
            if(dense_rowIdx101 > numNodes) {
                fragB00[0] = 0.0; fragB00[1] = 0.0; 
                fragB01[0] = 0.0; fragB01[1] = 0.0;
            } else {
                vint sourceIdx0 = dense_rowIdx101 * feature_dim + colB02;
                vint sourceIdx1 = dense_rowIdx101 * feature_dim + colB13;
                fragB00[0] = int4X(d_q, d_scale, feature_dim, sourceIdx0);
                fragB00[1] = int4X(d_q, d_scale, feature_dim, sourceIdx1);
                fragB01[0] = int4X(d_q, d_scale, feature_dim, sourceIdx0 + COL_WINDOW_R);
                fragB01[1] = int4X(d_q, d_scale, feature_dim, sourceIdx1 + COL_WINDOW_R);
            }
            if(dense_rowIdx123 > numNodes) {
                fragB00[2] = 0.0; fragB00[3] = 0.0; 
                fragB01[2] = 0.0; fragB01[3] = 0.0;
            } else {
                vint sourceIdx0 = dense_rowIdx123 * feature_dim + colB02;
                vint sourceIdx1 = dense_rowIdx123 * feature_dim + colB13;
                fragB00[2] = int4X(d_q, d_scale, feature_dim, sourceIdx0);
                fragB00[3] = int4X(d_q, d_scale, feature_dim, sourceIdx1);
                fragB01[2] = int4X(d_q, d_scale, feature_dim, sourceIdx0 + COL_WINDOW_R);
                fragB01[3] = int4X(d_q, d_scale, feature_dim, sourceIdx1 + COL_WINDOW_R);
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
    
    vint outOff = (bid << 3) * feature_dim + (local_warpID << 5) + offY;
    #pragma unroll
    for(vint i = 0; i < 4; ++i) {
        rowC = (tID_in_group << 1) + (i & 0x1);
        if(i < 2) colC = groupID;
        else colC = groupID + 8;
        store_fp32_to_global(d_MatC + outOff + rowC * feature_dim + colC, fragC[0][i]);
        store_fp32_to_global(d_MatC + outOff + rowC * feature_dim + colC + COL_WINDOW_R, fragC[1][i]);
    }
}

/**
 * @file mma_tf32.cuh
 * @author Haisha Zhao
 * @date 2025-04-02
 * 
 * @copyright MIT License (c) 2025 Haisha Zhao
*/

#pragma once
#include "common.hpp"
#include "ptx_tf32.hpp"

__global__
void tf32_mma_kernel(
    const TCLOCAL_TYPE* __restrict__    d_tcLocalBit, 
    const vint*         __restrict__    d_sparseA2B,
    const MAT_VAL_TYPE* __restrict__    d_valueA,
    const MAT_PTR_TYPE* __restrict__    d_block2Idx,
    const MAT_PTR_TYPE* __restrict__    d_data2Idx,
    const MAT_VAL_TYPE* __restrict__    d_MatB, 
    MAT_VAL_TYPE* d_MatC,
    const vint numNodes,
    const vint feature_dim
) {
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
    vint offY                 =   (blockIdx.y << 7);
    const vint laneid         =   31 & threadIdx.x;
    const vint warpSize       =   32;
    const vint tid            =   threadIdx.y * warpSize + laneid; 
    const vint local_warpID   =   threadIdx.y;

    vint groupID         =   laneid >> 2;
    vint tID_in_group    =   3 & laneid;

    vint rowA            =   groupID;
    vint colA0           =   tID_in_group;
    vint colA1           =   tID_in_group + 4;

    vint colB02          =   groupID + (local_warpID << 5);
    vint colB13          =   groupID + 8 + (local_warpID << 5);
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
    if(dense_rowIdx01 > numNodes) {
        fragB00[0] = 0.0; fragB00[1] = 0.0; 
        fragB01[0] = 0.0; fragB01[1] = 0.0;
    } else {
        vint sourceIdx0 = dense_rowIdx01 * feature_dim + colB02;
        vint sourceIdx1 = dense_rowIdx01 * feature_dim + colB13;
        fragB00[0] = load_fp32_from_global(d_MatB + sourceIdx0);
        fragB00[1] = load_fp32_from_global(d_MatB + sourceIdx1);
        fragB01[0] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
        fragB01[1] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
    }
    if(dense_rowIdx23 > numNodes) {
        fragB00[2] = 0.0; fragB00[3] = 0.0; 
        fragB01[2] = 0.0; fragB01[3] = 0.0;
    } else {
        vint sourceIdx0 = dense_rowIdx23 * feature_dim + colB02;
        vint sourceIdx1 = dense_rowIdx23 * feature_dim + colB13;
        fragB00[2] = load_fp32_from_global(d_MatB + sourceIdx0);
        fragB00[3] = load_fp32_from_global(d_MatB + sourceIdx1);
        fragB01[2] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
        fragB01[3] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
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
            if(dense_rowIdx101 > numNodes) {
                fragB10[0] = 0.0; fragB10[1] = 0.0; 
                fragB11[0] = 0.0; fragB11[1] = 0.0;
            } else {
                vint sourceIdx0 = dense_rowIdx101 * feature_dim + colB02;
                vint sourceIdx1 = dense_rowIdx101 * feature_dim + colB13;
                fragB10[0] = load_fp32_from_global(d_MatB + sourceIdx0);
                fragB10[1] = load_fp32_from_global(d_MatB + sourceIdx1);
                fragB11[0] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
                fragB11[1] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
            }
            if(dense_rowIdx123 > numNodes) {
                fragB10[2] = 0.0; fragB10[3] = 0.0; 
                fragB11[2] = 0.0; fragB11[3] = 0.0;
            } else {
                vint sourceIdx0 = dense_rowIdx123 * feature_dim + colB02;
                vint sourceIdx1 = dense_rowIdx123 * feature_dim + colB13;
                fragB10[2] = load_fp32_from_global(d_MatB + sourceIdx0);
                fragB10[3] = load_fp32_from_global(d_MatB + sourceIdx1);
                fragB11[2] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
                fragB11[3] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
            }
        } else {
            if(dense_rowIdx101 > numNodes) {
                fragB00[0] = 0.0; fragB00[1] = 0.0; 
                fragB01[0] = 0.0; fragB01[1] = 0.0;
            } else {
                vint sourceIdx0 = dense_rowIdx101 * feature_dim + colB02;
                vint sourceIdx1 = dense_rowIdx101 * feature_dim + colB13;
                fragB00[0] = load_fp32_from_global(d_MatB + sourceIdx0);
                fragB00[1] = load_fp32_from_global(d_MatB + sourceIdx1);
                fragB01[0] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
                fragB01[1] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
            }
            if(dense_rowIdx123 > numNodes) {
                fragB00[2] = 0.0; fragB00[3] = 0.0; 
                fragB01[2] = 0.0; fragB01[3] = 0.0;
            } else {
                vint sourceIdx0 = dense_rowIdx123 * feature_dim + colB02;
                vint sourceIdx1 = dense_rowIdx123 * feature_dim + colB13;
                fragB00[2] = load_fp32_from_global(d_MatB + sourceIdx0);
                fragB00[3] = load_fp32_from_global(d_MatB + sourceIdx1);
                fragB01[2] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
                fragB01[3] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
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
    
    vint outOff = (bid << 3) * feature_dim + (local_warpID << 5) + offY;
    #pragma unroll
    for(vint i = 0; i < 4; ++i) {
        rowC = (tID_in_group << 1) + (i & 0x1);
        if(i < 2) colC = groupID;
        else colC = groupID + 8;
        store_fp32_to_global(d_MatC + outOff + rowC * feature_dim + colC, fragC[0][i]);
        store_fp32_to_global(d_MatC + outOff + rowC * feature_dim + colC + COL_WINDOW_R, fragC[1][i]);
    }
}


// ---- Acc converter + FP32/INT4 host launchers (single-TU, batch 4) ----
RowBitTCF convertAccRow(std::uint32_t rows, std::uint32_t cols, const CanonicalCSR& canonical) {
    CSR<float> csr;
    csr.rows = rows; csr.cols = cols; csr.nnz = canonical.src.size();
    csr.row_ptr = canonical.ptr; csr.col_idx = canonical.src; csr.data = canonical.values;
    METCFBit<float> acc;
    acc.convertFromCSR(csr);
    RowBitTCF out;
    out.rows = rows; out.cols = cols;
    out.rowWindowOffset = std::move(acc.rowWindowOffset);
    out.tcOffset = std::move(acc.tcOffset);
    out.sparseA2B = std::move(acc.sparseA2B);
    out.tcLocalBit = std::move(acc.tcLocalBit);
    out.values = std::move(acc.data);
    return out;
}

cudaError_t launchAccRowSpMM(const std::uint32_t* window, const std::uint32_t* offset,
    const std::uint32_t* mapping, const std::uint64_t* bits, const float* values,
    const float* x, float* y, std::uint32_t paddedRows, std::uint32_t cols,
    std::uint32_t features, cudaStream_t stream) {
    if (paddedRows % 8 || features < 64 || features > 512 || features % 32 ||
        std::uint64_t(cols) * features > UINT32_MAX ||
        std::uint64_t(paddedRows) * features > UINT32_MAX) return cudaErrorInvalidValue;
    tf32_mma_kernel<<<dim3(paddedRows / 8), dim3(32, features / 32), 0, stream>>>(
        bits, mapping, values, window, offset, x, y, cols, features);
    return cudaGetLastError();
}

cudaError_t launchAccRowInt4SpMM(const std::uint32_t* window, const std::uint32_t* offset,
    const std::uint32_t* mapping, const std::uint64_t* bits, const float* values,
    const std::uint8_t* xq, const float* xscale, float* y, std::uint32_t paddedRows,
    std::uint32_t cols, std::uint32_t features, cudaStream_t stream) {
    if (paddedRows % 8 || features < 64 || features > 512 || features % 32 ||
        std::uint64_t(cols) * features > UINT32_MAX ||
        std::uint64_t(paddedRows) * features > UINT32_MAX) return cudaErrorInvalidValue;
    tf32_mma_kernel_int4<<<dim3(paddedRows / 8), dim3(32, features / 32), 0, stream>>>(
        bits, mapping, values, window, offset, xq, xscale, y, cols, features);
    return cudaGetLastError();
}
