#pragma once

#include "class.h"
#include "config.h"
#include "ptx_tf32.h"
#include "spmm_kernel.h"
#include "utils.h"

#include <cuda_runtime.h>


// ======================================= BCSR =======================================

__host__
float tf32_spmm_bcsr(
    const BCSR<MAT_VAL_TYPE>& bcsr, 
    const int32_t feature_dim,
    MAT_VAL_TYPE* d_DenseMatB,
    MAT_VAL_TYPE* &d_DenseMatC,
    cudaStream_t stream_hdl
);

// ======================================= BCSC =======================================

__host__
float tf32_spmm_bcsc(
    const BCSC<MAT_VAL_TYPE>& bcsc, 
    const int32_t feature_dim,
    MAT_VAL_TYPE* d_DenseMatB,
    MAT_VAL_TYPE* &d_DenseMatC,
    cudaStream_t stream_hdl
);