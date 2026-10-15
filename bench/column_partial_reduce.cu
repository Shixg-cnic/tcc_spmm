// Batch 3, scheme A: bounded partial results + per-row reduction (two-phase).
// Pass 1: the Column kernel computes each (CW, tile) once (full 8x8 tile MMA,
// X loaded once per feature block) and writes the (CW,row) partial feature
// vectors NON-atomically into per-row buckets (each (CW,row) pair is written
// exactly once, so no RMW is needed). Pass 2: per-row sequential reduction.
//
// The per-row bucket layout is precomputed offline: count[r] = number of
// (CW,row) pairs of row r, base[r] = prefix sum, and the slot of the pair
// (CW=w, row=r) is base[r] + ordinal(w among r's CWs). The partial buffer
// holds FB floats per pair (feature-blocked; the full F is processed as
// F/FB sequential blocks reusing one buffer).
#include "matrix_io.cuh"
#include "column_schedule.cuh"
#include "column_fast.cuh"
#include "mma_tf32.cuh"
#include "column_test_utils.cuh"
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <map>
#include <set>

using namespace test;
namespace {
using U = std::uint32_t;

struct PartialLayout {
    std::vector<U> count, base;         // per row: #(CW,row) pairs, prefix sum
    std::vector<U> tileSlot;            // per tile row slot: base[r]+ordinal
    std::size_t pairs = 0;
};

// Compute count[r] (distinct CWs of row r) and the per-(CW,row) ordinal.
PartialLayout buildLayout(const CanonicalCSR& csr, U rows, const ColumnBaseFormat& base) {
    PartialLayout L;
    L.count.assign(rows, 0);
    std::vector<std::vector<U>> cws(rows);
    for (U r = 0; r < rows; ++r) {
        std::set<U> seen;
        for (auto e = csr.ptr[r]; e < csr.ptr[r + 1]; ++e) seen.insert(csr.src[e] / 8);
        cws[r].assign(seen.begin(), seen.end());
        L.count[r] = U(cws[r].size());
    }
    L.base.assign(rows + 1, 0);
    for (U r = 0; r < rows; ++r) L.base[r + 1] = L.base[r] + L.count[r];
    L.pairs = L.base[rows];
    // tileSlot[tile*8+i] = base[row] + ordinal(CW-of-tile among row's CWs)
    L.tileSlot.assign(base.tileRows.size(), UINT32_MAX);
    for (U w = 0; w + 1 < base.colWindowOffset.size(); ++w) {
        for (U tile = base.colWindowOffset[w]; tile < base.colWindowOffset[w + 1]; ++tile) {
            for (U i = 0; i < 8; ++i) {
                U r = base.tileRows[std::size_t(tile) * 8 + i];
                if (r == UINT32_MAX) continue;
                auto it = std::lower_bound(cws[r].begin(), cws[r].end(), w);
                require(it != cws[r].end() && *it == w, "window not in row CW list");
                L.tileSlot[std::size_t(tile) * 8 + i] = L.base[r] + U(it - cws[r].begin());
            }
        }
    }
    return L;
}

__device__ void partialLoadX(const float* x, U w, U cols, U f, U features, float b[2][4]) {
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

// Pass 1: one CW per CTA, FB features per block (grid.y = F/FB).
// Emits float4 non-atomic stores of each (CW,row) partial into its bucket.
// When StoreMode==false the stores are skipped (diagnosis: isolate the
// compute/staging cost from the scattered-store cost). When Wide==false the
// stores are scalar (diagnosis: float4-scatter vs scalar-scatter cost).
template<int FB, bool StoreMode = true, bool Wide = true>
__global__ void partialPass1(DeviceColumnFast m, const U* tileSlot, const float* x,
    float* partial, U features, U fbBlock) {
    static_assert(FB % 16 == 0, "FB must be a multiple of 16 (float4 stores)");
    constexpr int WARPS = FB / 32 < 1 ? 1 : FB / 32;
    constexpr int PARTS = FB / 16;
    __shared__ float a[64];
    __shared__ U rows[8];
    __shared__ U slots[8];
    const U windows = (std::uint64_t(m.cols) + 7) / 8;
    const U w = blockIdx.x;
    const U fBase = fbBlock * FB + (threadIdx.x / 32) * 32;
    const U lane = threadIdx.x % 32, group = lane / 4, k = lane % 4;
    float b[2][4];
    partialLoadX(x, w, m.cols, fBase, features, b);
    const U first = m.colWindowOffset[w], end = m.colWindowOffset[w + 1];
    for (U tile = first; tile < end; ++tile) {
        for (U i = threadIdx.x; i < 64; i += blockDim.x) {
            auto bit = m.tileLocalBit[tile];
            a[i] = (bit & (1ULL << i)) ?
                m.values[m.tileValueOffset[tile] + __popcll(bit & ((1ULL << i) - 1))] : 0.f;
        }
        if (threadIdx.x < 8) {
            rows[threadIdx.x] = m.tileRows[std::size_t(tile) * 8 + threadIdx.x];
            slots[threadIdx.x] = tileSlot[std::size_t(tile) * 8 + threadIdx.x];
        }
        __syncthreads();
        float sparse[2] = {a[group * 8 + k], a[group * 8 + k + 4]};
        #pragma unroll
        for (int part = 0; part < PARTS; ++part) {
            float c[4] = {0.f, 0.f, 0.f, 0.f};
            tf32_m16n8k8(b[part], sparse, c);
            // Lane (g,k) fragment: c0=(row 2k, feat g), c1=(row 2k+1, feat g),
            // c2=(row 2k, feat g+8), c3=(row 2k+1, feat g+8). Shuffle so each
            // lane holds a float4 of one (row, feat quad) and store.
            const U mask = 0xffffffffu;
            U row = rows[k * 2 + (group >= 4 ? 1 : 0)];
            U quad = 4 * (group & 3);
            U slot = slots[k * 2 + (group >= 4 ? 1 : 0)];
            bool row0 = group < 4;
            float v[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                U q = quad + j;
                U src = (q & 7) * 4 + k;
                float cA0 = __shfl_sync(mask, c[0], src);
                float cA1 = __shfl_sync(mask, c[1], src);
                float cA2 = __shfl_sync(mask, c[2], src);
                float cA3 = __shfl_sync(mask, c[3], src);
                v[j] = row0 ? ((q < 8) ? cA0 : cA2) : ((q < 8) ? cA1 : cA3);
            }
            U feat = quad + part * 16; // local feature within the FB block
            if constexpr (StoreMode) {
                if constexpr (Wide) {
                    if (row != UINT32_MAX && slot != UINT32_MAX && feat + 3 < FB)
                        *reinterpret_cast<float4*>(partial + std::size_t(slot) * FB + feat) =
                            make_float4(v[0], v[1], v[2], v[3]);
                } else {
                    // scalar: D1b-style, no shuffle; 4 scalar stores per lane
                    U slot0 = slots[k * 2], slot1 = slots[k * 2 + 1];
                    U f0 = group + part * 16, f1 = group + 8 + part * 16;
                    float* p0 = partial + std::size_t(slot0) * FB;
                    float* p1 = partial + std::size_t(slot1) * FB;
                    if (rows[k * 2] != UINT32_MAX && f0 < FB && f1 < FB) {
                        p0[f0] = c[0]; p1[f0] = c[1]; p0[f1] = c[2]; p1[f1] = c[3];
                    }
                }
            } else {
                if (row != UINT32_MAX && slot != UINT32_MAX)
                    (void)partial; // keep the address chain live, skip the store
            }
        }
        __syncthreads();
    }
}

// Pass 2: per-row reduction. One CTA per (row-block, FB-block); each row's
// bucket is read sequentially and summed into Y[r][fb].
template<int FB>
__global__ void partialPass2(const U* count, const U* base, const float* partial,
    float* y, U rows, U features, U fb) {
    const U row = blockIdx.x * 128 + threadIdx.x;
    if (row >= rows) return;
    const U n = count[row];
    const float* bucket = partial + std::size_t(base[row]) * FB;
    float acc[FB / 4][4];
    #pragma unroll
    for (int j = 0; j < FB / 4; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    #pragma unroll 4
    for (U p = 0; p < n; ++p) {
        const float4* q = reinterpret_cast<const float4*>(bucket + std::size_t(p) * FB);
        #pragma unroll
        for (int j = 0; j < FB / 4; ++j) {
            acc[j][0] += q[j].x; acc[j][1] += q[j].y; acc[j][2] += q[j].z; acc[j][3] += q[j].w;
        }
    }
    float* out = y + std::size_t(row) * features + fb * FB;
    #pragma unroll
    for (int j = 0; j < FB / 4; ++j)
        reinterpret_cast<float4*>(out)[j] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || std::string(argv[1]) == "--help") {
            std::cout << "Usage: column_partial_reduce matrix.csr [F=128|256] [device=0] [X.bin] [--bench] [FB=16|32]\n";
            return 0;
        }
        const bool bench = std::find(argv, argv + argc, std::string("--bench")) != argv + argc;
        U FB = 16;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a.rfind("FB=", 0) == 0) FB = U(std::stoi(a.substr(3)));
        }
        if (FB != 16 && FB != 32) throw std::runtime_error("FB must be 16 or 32");
        check(cudaSetDevice(argc > 3 ? std::stoi(argv[3]) : 0));
        cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        auto m = readMatrix(argv[1]);
        const U features = U(argc > 2 ? std::stoi(argv[2]) : 128);
        auto base = buildColumnBase(m.rows, m.cols, m.csr);
        auto x = readFeatures(argc > 4 ? argv[4] : nullptr, m.cols, features);
        Buffer<float> input(x), y(std::size_t(m.rows) * features);
        Buffer<std::uint32_t> win(base.colWindowOffset), rows(base.tileRows), off(base.tileValueOffset);
        Buffer<std::uint64_t> bits(base.tileLocalBit);
        Buffer<float> vals(base.values);
        DeviceColumnFast view{base.rows, base.cols, win.data, rows.data, off.data, bits.data, vals.data};
        auto ref = reference(m, x, features, true);

        PartialLayout L = buildLayout(m.csr, m.rows, base);
        std::cout << "# layout rows=" << m.rows << " pairs=" << L.pairs
                  << " partial_bytes=" << (double(L.pairs) * FB * 4) / 1e9 << "GB\n";
        Buffer<std::uint32_t> dCount(L.count), dBase(L.base), dSlot(L.tileSlot);
        Buffer<float> partial(L.pairs * FB);

        auto execute = [&](auto fbc) {
            const bool nostore = std::find(argv, argv + argc, std::string("NOSTORE")) != argv + argc;
            const bool scalar = std::find(argv, argv + argc, std::string("SCALAR")) != argv + argc;
            const U fbBlocks = (features + fbc - 1) / fbc;
            check(cudaMemsetAsync(y.data, 0, std::size_t(m.rows) * features * sizeof(float), stream));
            auto launchP1 = [&](cudaStream_t s, U fb) {
                dim3 grid((std::uint64_t(m.cols) + 7) / 8);
                if (nostore) partialPass1<fbc, false><<<grid, 32, 0, s>>>(view, dSlot.data, input.data, partial.data, features, fb);
                else if (scalar) partialPass1<fbc, true, false><<<grid, 32, 0, s>>>(view, dSlot.data, input.data, partial.data, features, fb);
                else partialPass1<fbc, true, true><<<grid, 32, 0, s>>>(view, dSlot.data, input.data, partial.data, features, fb);
            };
            auto runAll = [&](cudaStream_t s) {
                for (U fb = 0; fb < fbBlocks; ++fb) {
                    launchP1(s, fb);
                    partialPass2<fbc><<<(m.rows + 127) / 128, 128, 0, s>>>(dCount.data, dBase.data,
                        partial.data, y.data, m.rows, features, fb);
                }
            };
            auto runP1 = [&](cudaStream_t s) {
                for (U fb = 0; fb < fbBlocks; ++fb) launchP1(s, fb);
            };
            auto runP2 = [&](cudaStream_t s) {
                for (U fb = 0; fb < fbBlocks; ++fb) {
                    partialPass2<fbc><<<(m.rows + 127) / 128, 128, 0, s>>>(dCount.data, dBase.data,
                        partial.data, y.data, m.rows, features, fb);
                }
            };
            // correctness (full Y vs TF32 reference)
            check(cudaMemsetAsync(partial.data, 0, L.pairs * fbc * sizeof(float), stream));
            runAll(stream);
            check(cudaStreamSynchronize(stream));
            auto out = y.host(stream);
            Error e = error(out, ref);
            std::cout << "correctness: rmse=" << e.rmse << " maxAbs=" << e.maxAbsolute
                      << " maxRef=" << e.maxReference << " -> " << (e.rmse < 2e-4 ? "OK" : "FAIL") << '\n';
            if (bench) {
                auto timeIt = [&](const std::function<void(cudaStream_t)>& op, const char* name) {
                    for (int i = 0; i < 5; ++i) op(stream);
                    check(cudaStreamSynchronize(stream));
                    cudaGraph_t g; cudaGraphExec_t exec;
                    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                    for (int r = 0; r < 5; ++r) op(stream);
                    check(cudaStreamEndCapture(stream, &g));
                    check(cudaGraphInstantiate(&exec, g, nullptr, nullptr, 0));
                    cudaEvent_t be, en; check(cudaEventCreate(&be)); check(cudaEventCreate(&en));
                    std::vector<float> samples;
                    for (int l = 0; l < 9; ++l) {
                        check(cudaEventRecord(be, stream)); check(cudaGraphLaunch(exec, stream));
                        check(cudaEventRecord(en, stream)); check(cudaEventSynchronize(en));
                        float ms; check(cudaEventElapsedTime(&ms, be, en)); samples.push_back(ms / 5);
                    }
                    std::sort(samples.begin(), samples.end());
                    cudaEventDestroy(be); cudaEventDestroy(en); cudaGraphExecDestroy(exec); cudaGraphDestroy(g);
                    std::cout << name << ',' << std::setprecision(9) << samples[samples.size() / 2]
                              << ',' << samples.front() << ',' << samples.back() << '\n';
                };
                timeIt(runP1, "pass1_ms");
                timeIt(runP2, "pass2_ms");
                timeIt(runAll, "total_ms");
                std::cout << "dataset,M,K,nnz,F,scheme,partial_gb\n";
                std::cout << argv[1] << ',' << m.rows << ',' << m.cols << ',' << m.csr.src.size() << ','
                          << features << ",partial_reduce_fb" << fbc << ','
                          << (double(L.pairs) * fbc * 4) / 1e9 << '\n';
            }
        };
        if (FB == 16) execute(std::integral_constant<U, 16>{});
        else execute(std::integral_constant<U, 32>{});
        check(cudaStreamDestroy(stream));
    } catch (const std::exception& ex) { std::cerr << "Error: " << ex.what() << '\n'; return 1; }
    return 0;
}
