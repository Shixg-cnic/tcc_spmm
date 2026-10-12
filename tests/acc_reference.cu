// Original Acc kernels and converter are compiled unmodified for comparison.
#include "acc_reference.cuh"
#include "../third_party/Acc-SpMM/include/format.hpp"
#include "../third_party/Acc-SpMM/include/mma_tf32.cuh"

RowBitTCF convertAccRow(std::uint32_t rows, std::uint32_t cols, const CanonicalCSR& canonical) {
    CSR<float> csr;
    csr.rows=rows; csr.cols=cols; csr.nnz=canonical.src.size();
    csr.row_ptr=canonical.ptr; csr.col_idx=canonical.src; csr.data=canonical.values;
    METCFBit<float> acc;
    acc.convertFromCSR(csr);
    RowBitTCF out;
    out.rows=rows; out.cols=cols;
    out.rowWindowOffset=std::move(acc.rowWindowOffset);
    out.tcOffset=std::move(acc.tcOffset);
    out.sparseA2B=std::move(acc.sparseA2B);
    out.tcLocalBit=std::move(acc.tcLocalBit);
    out.values=std::move(acc.data);
    return out;
}

cudaError_t launchAccRowSpMM(const std::uint32_t* window, const std::uint32_t* offset,
    const std::uint32_t* mapping, const std::uint64_t* bits, const float* values,
    const float* x, float* y, std::uint32_t paddedRows, std::uint32_t cols,
    std::uint32_t features, cudaStream_t stream) {
    if (paddedRows % 8 || features < 64 || features > 512 || features % 32 ||
        std::uint64_t(cols) * features > UINT32_MAX ||
        std::uint64_t(paddedRows) * features > UINT32_MAX) return cudaErrorInvalidValue;
    tf32_mma_kernel<<<dim3(paddedRows/8), dim3(32,features/32), 0, stream>>>(
        bits, mapping, values, window, offset, x, y, cols, features);
    return cudaGetLastError();
}
