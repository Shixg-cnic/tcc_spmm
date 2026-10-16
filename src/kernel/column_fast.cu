// Fast Column kernels (task-book batch 2). Each CTA owns one Column Window
// (8 input columns), loads X once per feature tile and reuses it across the
// CW's tiles; output rows come DIRECTLY from ColumnBaseFormat::tileRows
// (no SW-local slot indirection). Variants:
//   V1     : direct tileRows mapping + scalar atomics, single tile in flight
//   V2A    : V1 + hoisted per-row address computation (V2A == WIDE=1 + the
//            natural address hoist of the direct-row kernel; kept as a
//            separate label for the ablation table)
//   V2B    : warp reshuffle so one lane holds consecutive features of one
//            output row -> float2 (WIDE=2) / float4 (WIDE=4) wide atomics
//   V3     : tile pipeline with cp.async (STAGES=2/4), scalar atomics
//   V4     : internal feature tile FTILE in {32,64,128,256}
#include "column_fast.cuh"
#include "mma_tf32.cuh"
#include "pipeline.cuh"
#include <cstdint>
#include <cstring>

namespace {
using U = std::uint32_t;

__device__ void fastLoadX(const float* x, U w, U cols, U f, U features, float b[2][4]) {
    U lane = threadIdx.x % 32, group = lane / 4, k = lane % 4;
    #pragma unroll
    for (int part = 0; part < 2; ++part)
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            U col = w * 8 + k + (i >= 2 ? 4 : 0);
            U feature = f + group + (i % 2) * 8 + part * 16;
            b[part][i] = col < cols && feature < features ? x[std::size_t(col) * features + feature] : 0.f;
        }
}

// Fused INT4 X loader (batch 4): same indexing as fastLoadX but reads the
// packed stream and dequantises q*scale in registers.
__device__ void fastLoadXInt4(const DeviceInt4X* x, U w, U cols, U f, U features, float b[2][4]) {
    U lane = threadIdx.x % 32, group = lane / 4, k = lane % 4;
    #pragma unroll
    for (int part = 0; part < 2; ++part)
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            U col = w * 8 + k + (i >= 2 ? 4 : 0);
            U feature = f + group + (i % 2) * 8 + part * 16;
            float v = 0.f;
            if (col < cols && feature < features)
                v = int4ToFloat(nibbleAt(x->q, packedByteIndex(col, feature, features), feature), x->scale[col]);
            b[part][i] = v;
        }
}

// Emit the atomic updates of one MMA part (16 features x 8 rows, per-lane
// fragment c[4]). WIDE=1: 4 scalar REDG. WIDE=2: reshuffle to float2 pairs.
// WIDE=4: reshuffle to float4 quads (one per lane).
template<int WIDE>
__device__ __forceinline__ void emitFast(const float c[4], const U* rows, float* y,
    U fBase, U group, U k, U features, int part) {
    if constexpr (WIDE == 1) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            U row = rows[k * 2 + i % 2];
            U feat = fBase + group + (i >= 2 ? 8 : 0) + part * 16;
            if (row != UINT32_MAX && feat < features)
                atomicAdd(y + std::size_t(row) * features + feat, c[i]);
        }
    } else if constexpr (WIDE == 2) {
        // Fragment (verified against C0's correct mapping): lane (g,k) holds
        // c0=(row 2k, feat g), c1=(row 2k+1, feat g), c2=(row 2k, feat g+8),
        // c3=(row 2k+1, feat g+8). Destination lane (g',k) emits float2
        // (row 2k, feat 2g'+part*16..+1) and (row 2k+1, ...).
        // Feature q of row 2k lives in c0 (q<8) or c2 (q>=8) of lane (q&7, k);
        // of row 2k+1 in c1/c3 of the same lane. Shuffle both candidates per
        // feature and select by the DESTINATION feature value.
        U row0 = rows[k * 2], row1 = rows[k * 2 + 1];
        const U mask = 0xffffffffu;
        U srcA = ((2 * group) & 7) * 4 + k;
        U srcB = ((2 * group + 1) & 7) * 4 + k;
        float c0a = __shfl_sync(mask, c[0], srcA), c2a = __shfl_sync(mask, c[2], srcA);
        float c0b = __shfl_sync(mask, c[0], srcB), c2b = __shfl_sync(mask, c[2], srcB);
        float v0 = (2 * group < 8) ? c0a : c2a;
        float v1 = (2 * group + 1 < 8) ? c0b : c2b;
        float c1a = __shfl_sync(mask, c[1], srcA), c3a = __shfl_sync(mask, c[3], srcA);
        float c1b = __shfl_sync(mask, c[1], srcB), c3b = __shfl_sync(mask, c[3], srcB);
        float v2 = (2 * group < 8) ? c1a : c3a;
        float v3 = (2 * group + 1 < 8) ? c1b : c3b;
        U feat = fBase + 2 * group + part * 16;
        if (row0 != UINT32_MAX && feat + 1 < features)
            atomicAdd(reinterpret_cast<float2*>(y + std::size_t(row0) * features + feat), make_float2(v0, v1));
        if (row1 != UINT32_MAX && feat + 1 < features)
            atomicAdd(reinterpret_cast<float2*>(y + std::size_t(row1) * features + feat), make_float2(v2, v3));
    } else {
        // Lane (g,k): row = 2k + (g>=4 ? 1 : 0), feature quad = 4*(g&3).
        // Feature q of row 2k lives in c0 (q<8) or c2 (q>=8) of lane (q&7, k);
        // of row 2k+1 in c1/c3. Shuffle c0/c2 (or c1/c3) and select by q.
        U row = rows[k * 2 + (group >= 4 ? 1 : 0)];
        U quad = 4 * (group & 3);
        const U mask = 0xffffffffu;
        bool row0 = group < 4;
        float v[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            U q = quad + j;
            U src = (q & 7) * 4 + k;
            float cA0 = __shfl_sync(mask, c[0], src); // (row 2k, feat q&7)
            float cA1 = __shfl_sync(mask, c[1], src); // (row 2k+1, feat q&7)
            float cA2 = __shfl_sync(mask, c[2], src); // (row 2k, feat (q&7)+8)
            float cA3 = __shfl_sync(mask, c[3], src); // (row 2k+1, feat (q&7)+8)
            v[j] = row0 ? ((q < 8) ? cA0 : cA2) : ((q < 8) ? cA1 : cA3);
        }
        U feat = fBase + quad + part * 16;
        if (row != UINT32_MAX && feat + 3 < features)
            atomicAdd(reinterpret_cast<float4*>(y + std::size_t(row) * features + feat),
                make_float4(v[0], v[1], v[2], v[3]));
    }
}

// Core fast kernel: WIDE in {1,2,4}, internal feature tile FTILE.
// Int4: X loaded fused from the packed INT4 stream (see fastLoadXInt4).
// Ordered: CW id comes from the task-position -> CW permutation (offline
// task-order experiment, experiment/cw-order-only). All other computation is
// unchanged. Ordered=false callers pass taskOrder=nullptr (unused).
template<int WIDE, int FTILE, bool Int4 = false, bool Ordered = false>
__global__ void fastKernel(DeviceColumnFast m, const void* x, float* y, U features,
    const U* taskOrder) {
    constexpr int WARPS = FTILE / 32;
    __shared__ float a[64];
    __shared__ U rows[8];
    const U windows = (std::uint64_t(m.cols) + 7) / 8;
    U w;
    if constexpr (Ordered) w = taskOrder[blockIdx.x];
    else w = blockIdx.x;
    const U fBase = blockIdx.y * FTILE + (threadIdx.x / 32) * 32;
    const U lane = threadIdx.x % 32, group = lane / 4, k = lane % 4;
    float b[2][4];
    if constexpr (Int4)
        fastLoadXInt4(static_cast<const DeviceInt4X*>(x), w, m.cols, fBase, features, b);
    else
        fastLoadX(static_cast<const float*>(x), w, m.cols, fBase, features, b);
    const U first = m.colWindowOffset[w], end = m.colWindowOffset[w + 1];
    for (U tile = first; tile < end; ++tile) {
        for (U i = threadIdx.x; i < 64; i += blockDim.x) {
            auto bit = m.tileLocalBit[tile];
            a[i] = (bit & (1ULL << i)) ?
                m.values[m.tileValueOffset[tile] + __popcll(bit & ((1ULL << i) - 1))] : 0.f;
        }
        if (threadIdx.x < 8) rows[threadIdx.x] = m.tileRows[std::size_t(tile) * 8 + threadIdx.x];
        __syncthreads();
        float sparse[2] = {a[group * 8 + k], a[group * 8 + k + 4]};
        #pragma unroll
        for (int part = 0; part < 2; ++part) {
            float c[4] = {0.f, 0.f, 0.f, 0.f};
            tf32_m16n8k8(b[part], sparse, c);
            emitFast<WIDE>(c, rows, y, fBase, group, k, features, part);
        }
        __syncthreads();
    }
}

// V3: tile pipeline with cp.async (values staged asynchronously, rows via
// plain smem stores). Stages = 2 or 4. WIDE selects the atomic width.
template<int Stages, int WIDE = 1, bool Int4 = false>
__global__ void fastPipeKernel(DeviceColumnFast m, const void* x, float* y, U features) {
    __shared__ float a[Stages][64];
    __shared__ U rows[Stages][8];
    const U windows = (std::uint64_t(m.cols) + 7) / 8;
    const U w = blockIdx.x;
    const U fBase = blockIdx.y * 128 + (threadIdx.x / 32) * 32;
    const U lane = threadIdx.x % 32, group = lane / 4, k = lane % 4;
    float b[2][4];
    if constexpr (Int4)
        fastLoadXInt4(static_cast<const DeviceInt4X*>(x), w, m.cols, fBase, features, b);
    else
        fastLoadX(static_cast<const float*>(x), w, m.cols, fBase, features, b);
    const U first = m.colWindowOffset[w], end = m.colWindowOffset[w + 1];
    const U count = end - first;
    auto stage = [&](U tile, int buf) {
        if (threadIdx.x < 64) {
            auto bit = m.tileLocalBit[tile];
            if (bit & (1ULL << threadIdx.x)) {
                auto idx = m.tileValueOffset[tile] + __popcll(bit & ((1ULL << threadIdx.x) - 1));
                async_copy(static_cast<U>(__cvta_generic_to_shared(a[buf] + threadIdx.x)), m.values + idx);
            } else a[buf][threadIdx.x] = 0.f;
        }
        if (threadIdx.x < 8) rows[buf][threadIdx.x] = m.tileRows[std::size_t(tile) * 8 + threadIdx.x];
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    for (int s = 0; s < Stages && s < int(count); ++s) stage(first + s, s);
    for (U t = 0; t < count; ++t) {
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        const int buf = t % Stages;
        float sparse[2] = {a[buf][group * 8 + k], a[buf][group * 8 + k + 4]};
        #pragma unroll
        for (int part = 0; part < 2; ++part) {
            float c[4] = {0.f, 0.f, 0.f, 0.f};
            tf32_m16n8k8(b[part], sparse, c);
            emitFast<WIDE>(c, rows[buf], y, fBase, group, k, features, part);
        }
        __syncthreads();
        // Buffer (t+Stages)%Stages is now free (all warps passed the barrier).
        if (t + Stages < count) stage(first + t + Stages, (t + Stages) % Stages);
    }
}

template<int WIDE, int FTILE, bool Int4 = false, bool Ordered = false>
void launchFast(dim3 grid, DeviceColumnFast m, const void* x, float* y,
    U features, cudaStream_t stream, const U* taskOrder = nullptr) {
    fastKernel<WIDE, FTILE, Int4, Ordered><<<grid, FTILE / 32 * 32, 0, stream>>>(
        m, x, y, features, taskOrder);
}

template<bool Int4>
cudaError_t launchImpl(const DeviceColumnFast& m, const void* x, float* y,
    U features, FastVariant variant, U ftile, U stages, cudaStream_t stream, bool clear) {
    const U windows = (std::uint64_t(m.cols) + 7) / 8;
    if (clear) {
        auto status = cudaMemsetAsync(y, 0, std::size_t(m.rows) * features * sizeof(float), stream);
        if (status != cudaSuccess) return status;
    }
    if (features % ftile) return cudaErrorInvalidValue;
    dim3 grid(windows, features / ftile);
    switch (variant) {
        case FAST_V1:
            if (ftile == 128) launchFast<1, 128, Int4, false>(grid, m, x, y, features, stream);
            else if (ftile == 256) launchFast<1, 256, Int4, false>(grid, m, x, y, features, stream);
            else if (ftile == 64) launchFast<1, 64, Int4, false>(grid, m, x, y, features, stream);
            else launchFast<1, 32, Int4, false>(grid, m, x, y, features, stream);
            break;
        case FAST_V2A:
            launchFast<1, 128, Int4, false>(grid, m, x, y, features, stream);
            break;
        case FAST_V2B:
            launchFast<2, 128, Int4, false>(grid, m, x, y, features, stream);
            break;
        case FAST_V2C:
            if (ftile == 64) launchFast<4, 64, Int4, false>(grid, m, x, y, features, stream);
            else if (ftile == 32) launchFast<4, 32, Int4, false>(grid, m, x, y, features, stream);
            else if (ftile == 256) launchFast<4, 256, Int4, false>(grid, m, x, y, features, stream);
            else launchFast<4, 128, Int4, false>(grid, m, x, y, features, stream);
            break;
        case FAST_V4:
            if (ftile == 256) launchFast<1, 256, Int4, false>(grid, m, x, y, features, stream);
            else if (ftile == 64) launchFast<1, 64, Int4, false>(grid, m, x, y, features, stream);
            else if (ftile == 32) launchFast<1, 32, Int4, false>(grid, m, x, y, features, stream);
            else launchFast<1, 128, Int4, false>(grid, m, x, y, features, stream);
            break;
        case FAST_V3W:
            if (stages == 4) fastPipeKernel<4, 2, Int4><<<grid, 128, 0, stream>>>(m, x, y, features);
            else fastPipeKernel<2, 2, Int4><<<grid, 128, 0, stream>>>(m, x, y, features);
            break;
        case FAST_V3:
            if (stages == 4) fastPipeKernel<4, 1, Int4><<<grid, 128, 0, stream>>>(m, x, y, features);
            else fastPipeKernel<2, 1, Int4><<<grid, 128, 0, stream>>>(m, x, y, features);
            break;
        default:
            return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}
} // namespace

cudaError_t launchColumnFast(const DeviceColumnFast& matrix, const float* x, float* y,
    std::uint32_t features, FastVariant variant, std::uint32_t ftile,
    std::uint32_t stages, cudaStream_t stream, bool clearOutput) {
    return launchImpl<false>(matrix, x, y, features, variant, ftile, stages, stream, clearOutput);
}

cudaError_t launchColumnFastInt4(const DeviceColumnFast& matrix, const DeviceInt4X& x, float* y,
    std::uint32_t features, FastVariant variant, std::uint32_t ftile,
    std::uint32_t stages, cudaStream_t stream, bool clearOutput) {
    return launchImpl<true>(matrix, &x, y, features, variant, ftile, stages, stream, clearOutput);
}

namespace {
template<bool Int4>
cudaError_t columnFastAttributesImpl(FastVariant variant, std::uint32_t ftile,
    std::uint32_t stages, ColumnFastInfo* info) {
    if (!info) return cudaErrorInvalidValue;
    auto fill = [&](const void* kernel, U threads, U dyn) {
        cudaFuncAttributes a;
        auto status = cudaFuncGetAttributes(&a, kernel);
        if (status != cudaSuccess) return status;
        info->registers = a.numRegs; info->sharedBytes = a.sharedSizeBytes + dyn;
        info->localBytes = a.localSizeBytes;
        return cudaOccupancyMaxActiveBlocksPerMultiprocessor(&info->activeBlocksPerSM, kernel, threads, dyn);
    };
    switch (variant) {
        case FAST_V1:
            if (ftile == 256) return fill((const void*)fastKernel<1, 256, Int4, false>, 256, 0);
            if (ftile == 64) return fill((const void*)fastKernel<1, 64, Int4, false>, 64, 0);
            if (ftile == 32) return fill((const void*)fastKernel<1, 32, Int4, false>, 32, 0);
            return fill((const void*)fastKernel<1, 128, Int4, false>, 128, 0);
        case FAST_V2A: return fill((const void*)fastKernel<1, 128, Int4, false>, 128, 0);
        case FAST_V2B: return fill((const void*)fastKernel<2, 128, Int4, false>, 128, 0);
        case FAST_V2C:
            if (ftile == 256) return fill((const void*)fastKernel<4, 256, Int4, false>, 256, 0);
            if (ftile == 64) return fill((const void*)fastKernel<4, 64, Int4, false>, 64, 0);
            if (ftile == 32) return fill((const void*)fastKernel<4, 32, Int4, false>, 32, 0);
            return fill((const void*)fastKernel<4, 128, Int4, false>, 128, 0);
        case FAST_V4:
            if (ftile == 256) return fill((const void*)fastKernel<1, 256, Int4, false>, 256, 0);
            if (ftile == 64) return fill((const void*)fastKernel<1, 64, Int4, false>, 64, 0);
            if (ftile == 32) return fill((const void*)fastKernel<1, 32, Int4, false>, 32, 0);
            return fill((const void*)fastKernel<1, 128, Int4, false>, 128, 0);
        case FAST_V3W:
            if (stages == 4) return fill((const void*)fastPipeKernel<4, 2, Int4>, 128, 0);
            return fill((const void*)fastPipeKernel<2, 2, Int4>, 128, 0);
        case FAST_V3:
            if (stages == 4) return fill((const void*)fastPipeKernel<4, 1, Int4>, 128, 0);
            return fill((const void*)fastPipeKernel<2, 1, Int4>, 128, 0);
        default: return cudaErrorInvalidValue;
    }
}
} // namespace

cudaError_t columnFastAttributes(FastVariant variant, std::uint32_t ftile,
    std::uint32_t stages, ColumnFastInfo* info) {
    return columnFastAttributesImpl<false>(variant, ftile, stages, info);
}

cudaError_t columnFastInt4Attributes(FastVariant variant, std::uint32_t ftile,
    std::uint32_t stages, ColumnFastInfo* info) {
    return columnFastAttributesImpl<true>(variant, ftile, stages, info);
}

// Ordered V2C (experiment/cw-order-only): task position blockIdx.x maps to CW
// w = dTaskOrder[blockIdx.x]. Fixed WIDE=4 / FTILE=128 / FP32; only
// features 128 or 256 supported. All computation identical to the original
// V2C; only the CW id source changes. Y is zeroed once before the kernel
// when clearOutput is true (never per-CW, no cross-CTA synchronization).
cudaError_t launchColumnFastOrdered(const DeviceColumnFast& matrix, const float* x, float* y,
    const std::uint32_t* dTaskOrder, std::uint32_t features,
    cudaStream_t stream, bool clearOutput) {
    const U windows = (std::uint64_t(matrix.cols) + 7) / 8;
    if (features != 128 && features != 256) return cudaErrorInvalidValue;
    if (!dTaskOrder) return cudaErrorInvalidValue;
    if (clearOutput) {
        auto status = cudaMemsetAsync(y, 0, std::size_t(matrix.rows) * features * sizeof(float), stream);
        if (status != cudaSuccess) return status;
    }
    dim3 grid(windows, features / 128);
    fastKernel<4, 128, false, true><<<grid, 128, 0, stream>>>(
        matrix, x, y, features, dTaskOrder);
    return cudaGetLastError();
}
