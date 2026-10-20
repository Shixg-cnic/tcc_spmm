#!/usr/bin/env python3
# Build bench/acc_int4_fused.cu: a single-TU Acc unit containing
#   (a) the ORIGINAL first kernel tf32_mma_kernel (FP32 X, untouched text),
#   (b) a mechanical copy tf32_mma_kernel_int4 with the X loads replaced by
#       fused INT4 dequantisation (d_MatB -> packed d_q + d_scale),
#   (c) the Acc converter + FP32/INT4 host launchers.
# The third_party file is NOT modified; tests/acc_reference.cu is not linked
# into this target (common.hpp's host helpers are compiled exactly once here).
import re, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "third_party/Acc-SpMM/include/mma_tf32.cuh"
OUT = ROOT / "bench/acc_int4_fused.cu"

src = open(SRC).read()

# location of the second kernel: everything after it is dropped (the Acc Basic
# baseline launches exactly tf32_mma_kernel; g512/balance have own d_MatB)
m = re.search(r"\n__global__\s*\n\s*void tf32_mma_g512_kernel\s*\(", src)
assert m, "g512 kernel not found"
orig_first = src[:m.start()]          # original first kernel (FP32)

# ---- transformed INT4 copy ----
ts = orig_first
assert ts.count("void tf32_mma_kernel(") == 1
ts = ts.replace("void tf32_mma_kernel(", "void tf32_mma_kernel_int4(")
msig = re.search(r"const MAT_VAL_TYPE\*\s*__restrict__\s+d_MatB,\s*", ts)
assert msig, "signature not found"
ts = ts[:msig.start()] + ("const std::uint8_t* __restrict__ d_q,\n"
    "    const float*         __restrict__ d_scale, ") + ts[msig.end():]
n = ts.count("load_fp32_from_global(d_MatB + ")
ts = ts.replace("load_fp32_from_global(d_MatB + ", "int4X(d_q, d_scale, feature_dim, ")
assert n == 24, f"unexpected X load count: {n}"
print("replaced X load sites:", n)

# ---- assemble: includes + int4X + INT4 kernel + FP32 kernel + launchers ----
out = ('#include "format.hpp"\n#include "int4_format.cuh"\n#include "csr.cuh"\n#include "row_format.cuh"\n'
       '// Fused INT4 X load: given the linear FP32 index idx = row*features+col,\n'
       '// read the packed nibble and dequantise q*scale[row].\n'
       '__device__ __forceinline__ float int4X(const std::uint8_t* q, const float* scale,\n'
       '    vint features, std::size_t idx) {\n'
       '    vint row = vint(idx / features), col = vint(idx % features);\n'
       '    return int4ToFloat(nibbleAt(q, packedByteIndex(row, col, features), col), scale[row]);\n'
       '}\n') + ts + "\n" + orig_first + r'''

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
'''

with open(OUT, "w") as f:
    f.write(out)
print("wrote", OUT, os.path.getsize(OUT), "bytes")
