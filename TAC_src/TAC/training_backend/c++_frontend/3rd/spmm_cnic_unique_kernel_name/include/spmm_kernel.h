#pragma once

#include "ptx_tf32.h"

// ======================================= gnn first layer =======================================

__global__
void tf32_computeX128TransposePipe2_BCSR_forward_first_layer(
    const MAT_PTR_TYPE* __restrict__    d_block2Idx,
    const MAT_PTR_TYPE* __restrict__    d_data2Idx,
    const vint*         __restrict__    d_sparseA2B,
    const TCLOCAL_TYPE* __restrict__    d_tcLocalBit, 
    const MAT_VAL_TYPE* __restrict__    d_MatB, 
    MAT_VAL_TYPE* d_MatC,
    const vint numRows,
    const vint numCols,
    const vint feature_dim
);

__global__
void tf32_computeX128TransposePipe2_BCSC_forward_first_layer(
    const MAT_PTR_TYPE* __restrict__    d_block2Idx,
    const MAT_PTR_TYPE* __restrict__    d_data2Idx,
    const vint*         __restrict__    d_sparseA2C,
    const TCLOCAL_TYPE* __restrict__    d_tcLocalBit, 
    const MAT_VAL_TYPE* __restrict__    d_MatB, 
    MAT_VAL_TYPE* d_MatC,
    const vint numRows,
    const vint numCols,
    const vint feature_dim
);

// ======================================= gnn first layer =======================================

__global__
void tf32_computeX128TransposePipe2_BCSR_forward_second_layer(
    const MAT_PTR_TYPE* __restrict__    d_block2Idx,
    const MAT_PTR_TYPE* __restrict__    d_data2Idx,
    const vint*         __restrict__    d_sparseA2B,
    const TCLOCAL_TYPE* __restrict__    d_tcLocalBit, 
    const MAT_VAL_TYPE* __restrict__    d_MatB, 
    MAT_VAL_TYPE* d_MatC,
    const vint numRows,
    const vint numCols,
    const vint feature_dim
);

__global__
void tf32_computeX128TransposePipe2_BCSC_forward_second_layer(
    const MAT_PTR_TYPE* __restrict__    d_block2Idx,
    const MAT_PTR_TYPE* __restrict__    d_data2Idx,
    const vint*         __restrict__    d_sparseA2C,
    const TCLOCAL_TYPE* __restrict__    d_tcLocalBit, 
    const MAT_VAL_TYPE* __restrict__    d_MatB, 
    MAT_VAL_TYPE* d_MatC,
    const vint numRows,
    const vint numCols,
    const vint feature_dim
);

// ======================================= gnn first layer =======================================

__global__
void tf32_computeX128TransposePipe2_BCSR_backward(
    const MAT_PTR_TYPE* __restrict__    d_block2Idx,
    const MAT_PTR_TYPE* __restrict__    d_data2Idx,
    const vint*         __restrict__    d_sparseA2B,
    const TCLOCAL_TYPE* __restrict__    d_tcLocalBit, 
    const MAT_VAL_TYPE* __restrict__    d_MatB, 
    MAT_VAL_TYPE* d_MatC,
    const vint numRows,
    const vint numCols,
    const vint feature_dim
);

__global__
void tf32_computeX128TransposePipe2_BCSC_backward(
    const MAT_PTR_TYPE* __restrict__    d_block2Idx,
    const MAT_PTR_TYPE* __restrict__    d_data2Idx,
    const vint*         __restrict__    d_sparseA2C,
    const TCLOCAL_TYPE* __restrict__    d_tcLocalBit, 
    const MAT_VAL_TYPE* __restrict__    d_MatB, 
    MAT_VAL_TYPE* d_MatC,
    const vint numRows,
    const vint numCols,
    const vint feature_dim
);