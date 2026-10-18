#pragma once

#include "class.h"
#include "config.h"
#include "ptx_tf32.h"
#include "spmm_kernel.h"
#include "utils.h"

#include <cuda_runtime.h>


// ======================================= gnn first layer =======================================

__host__
float tf32_spmm_bcsr_forward_first_layer(
    const BCSR<MAT_VAL_TYPE>& bcsr, 
    const int32_t feature_dim,
    MAT_VAL_TYPE* d_DenseMatB,
    MAT_VAL_TYPE* &d_DenseMatC,
    cudaStream_t stream_hdl
);

__host__
float tf32_spmm_bcsc_forward_first_layer(
    const BCSC<MAT_VAL_TYPE>& bcsc, 
    const int32_t feature_dim,
    MAT_VAL_TYPE* d_DenseMatB,
    MAT_VAL_TYPE* &d_DenseMatC,
    cudaStream_t stream_hdl
);

// ======================================= gnn second layer =======================================

__host__
float tf32_spmm_bcsr_forward_second_layer(
    const BCSR<MAT_VAL_TYPE>& bcsr, 
    const int32_t feature_dim,
    MAT_VAL_TYPE* d_DenseMatB,
    MAT_VAL_TYPE* &d_DenseMatC,
    cudaStream_t stream_hdl
);

__host__
float tf32_spmm_bcsc_forward_second_layer(
    const BCSC<MAT_VAL_TYPE>& bcsc, 
    const int32_t feature_dim,
    MAT_VAL_TYPE* d_DenseMatB,
    MAT_VAL_TYPE* &d_DenseMatC,
    cudaStream_t stream_hdl
);

// ======================================= gnn backward layer =======================================

__host__
float tf32_spmm_bcsr_backward(
    const BCSR<MAT_VAL_TYPE>& bcsr, 
    const int32_t feature_dim,
    MAT_VAL_TYPE* d_DenseMatB,
    MAT_VAL_TYPE* &d_DenseMatC,
    cudaStream_t stream_hdl
);

__host__
float tf32_spmm_bcsc_backward(
    const BCSC<MAT_VAL_TYPE>& bcsc, 
    const int32_t feature_dim,
    MAT_VAL_TYPE* d_DenseMatB,
    MAT_VAL_TYPE* &d_DenseMatC,
    cudaStream_t stream_hdl
);