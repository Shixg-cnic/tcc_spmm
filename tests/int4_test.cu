// INT4 fused-kernel correctness:
//  - GPU quantize/dequantize vs the CPU reference (byte-exact q, scale exact);
//  - fused INT4 kernels (fixed C0..C3 and adaptive, cache on/off) vs the
//    TF32 CPU reference on the dequantised X-hat (same math as FP32);
//  - the independent baseline (GPU dequantize + FP32 C1) vs the same reference;
//  - the quantization delta (output vs the FP32 reference on the ORIGINAL X)
//    is reported, not asserted.
// Matrices: weighted / dense-CW / zero / repeated-row / negative-weight;
// F = 1,17,32,64,100,128,256,512 (odd F exercises nibble padding).
#include "column_schedule.cuh"
#include "int4_format.cuh"
#include "column_test_utils.cuh"
#include <cstdio>

namespace test {
void quantizeCPU(const std::vector<float>& x, std::vector<std::uint8_t>& q,
                 std::vector<float>& scale, unsigned nodes, unsigned features) {
    const unsigned pairs = (features + 1) / 2;
    q.assign(std::size_t(nodes) * pairs, 0);
    scale.assign(nodes, 0.f);
    for (unsigned i = 0; i < nodes; ++i) {
        float m = 0.f;
        for (unsigned j = 0; j < features; ++j) m = std::max(m, std::fabs(x[std::size_t(i)*features+j]));
        const float s = m == 0.f ? 1.f : m / 7.f;
        scale[i] = s;
        for (unsigned j = 0; j < features; ++j) {
            int r = int(std::lround(x[std::size_t(i)*features+j] / s));
            r = std::max(-7, std::min(7, r));
            std::uint8_t nib = std::uint8_t(r) & 0xF;
            if (j % 2 == 0) q[std::size_t(i)*pairs + j/2] = nib;
            else q[std::size_t(i)*pairs + j/2] |= std::uint8_t(nib << 4);
        }
    }
}
std::vector<float> dequantCPU(const std::vector<std::uint8_t>& q, const std::vector<float>& scale,
                              unsigned nodes, unsigned features) {
    const unsigned pairs = (features + 1) / 2;
    std::vector<float> xh(std::size_t(nodes) * features, 0.f);
    for (unsigned i = 0; i < nodes; ++i)
        for (unsigned j = 0; j < features; ++j) {
            std::uint8_t nib = (j % 2 == 0) ? (q[std::size_t(i)*pairs + j/2] & 0xF)
                                            : (q[std::size_t(i)*pairs + j/2] >> 4);
            int qi = (nib & 8) ? int(nib) - 16 : int(nib);
            xh[std::size_t(i)*features+j] = float(qi) * scale[i];
        }
    return xh;
}

struct Int4Buffers {
    Buffer<std::uint8_t> q;
    Buffer<float> scale;
    DeviceInt4X view;
    explicit Int4Buffers(const std::vector<std::uint8_t>& qh, const std::vector<float>& sh)
        : q(qh), scale(sh), view{q.data, scale.data} {}
};

void run(const char* name, const Matrix& matrix, const std::vector<float>& x, unsigned f) {
    const unsigned nodes = matrix.cols;
    std::vector<std::uint8_t> qr;
    std::vector<float> sr;
    quantizeCPU(x, qr, sr, nodes, f);
    Int4Buffers gpu(qr, sr);
    const unsigned pairs = (f + 1) / 2;
    // GPU quantize from the raw X must reproduce qr/sr.
    {
        Buffer<std::uint8_t> qg(std::size_t(nodes) * pairs);
        Buffer<float> sg(nodes);
        Buffer<float> xg(x);
        check(quantizeXInt4(xg.data, qg.data, sg.data, nodes, f));
        auto qh = qg.host(nullptr);
        auto sh = sg.host(nullptr);
        for (std::size_t i = 0; i < qr.size(); ++i) require(qh[i] == qr[i], "quantize q mismatch");
        for (unsigned i = 0; i < nodes; ++i)
            require(std::fabs(sh[i] - sr[i]) <= 1e-6f * std::max(1.f, std::fabs(sr[i])), "quantize scale mismatch");
    }
    auto xhat = dequantCPU(qr, sr, nodes, f);
    // GPU dequantize baseline matches the CPU X-hat.
    {
        Buffer<float> xg(std::size_t(nodes) * f);
        check(dequantizeXInt4(gpu.q.data, gpu.scale.data, xg.data, nodes, f));
        auto xh = xg.host(nullptr);
        bool bad = false;
        for (std::size_t i = 0; i < xhat.size(); ++i)
            if (std::fabs(xh[i] - xhat[i]) > 1e-6f) { bad = true; break; }
        if (bad) {
            std::size_t first = 0;
            for (std::size_t i = 0; i < xhat.size(); ++i)
                if (std::fabs(xh[i] - xhat[i]) > 1e-6f) { first = i; break; }
            std::cerr << "debug F=" << f << " nodes=" << nodes << " pairs=" << (f+1)/2
                      << " first=" << first << " node=" << first/f << " j=" << first%f
                      << " byte=" << unsigned(gpu.q.host(nullptr)[(first/f)*((f+1)/2)+(first%f)/2])
                      << " gpu=" << xh[first] << " cpu=" << xhat[first] << '\n';
            throw std::runtime_error("dequantize mismatch");
        }
    }
    // CPU references: A*X-hat with TF32 math (GPU expectation) and A*X in FP32
    // (quantization-free truth, reported only).
    auto ref1 = reference(matrix, xhat, f, true);
    auto ref2 = reference(matrix, x, f, false);
    // Formats.
    auto col = convertColumn(matrix.rows, matrix.cols, matrix.csr);
    GPUColumn gc(col);
    auto base = buildColumnBase(matrix.rows, matrix.cols, matrix.csr);
    auto sched = buildAdaptiveSchedule(base, ScheduleConfig{});
    auto fmt = buildAdaptiveColumn(base, sched);
    Buffer<std::uint32_t> win(fmt.colWindowOffset), slots(fmt.tileRowSlot), off(fmt.tileValueOffset),
        swOff(fmt.swRowOffset), rows(fmt.swRows), cacheSlots(fmt.swRowCacheSlot),
        cachedRows(fmt.swCachedRowSlot), counts(fmt.swCachedCount);
    Buffer<std::uint64_t> bits(fmt.tileLocalBit);
    Buffer<float> vals(fmt.values);
    Buffer<SWTask> tasks(fmt.tasks);
    DeviceColumnAdaptive view{fmt.rows, fmt.cols, std::uint32_t(fmt.tasks.size()),
        win.data, slots.data, off.data, bits.data, vals.data,
        swOff.data, rows.data, cacheSlots.data, cachedRows.data, counts.data, tasks.data};
    Buffer<float> y(std::size_t(matrix.rows) * f);
    double qdelta = 0, bestErr = 1;
    // Fused fixed kernels.
    for (int v = 0; v < 4; ++v) {
        check(cudaMemsetAsync(y.data, 0, y.size * sizeof(float)));
        check(launchColumnInt4SpMM(gc.view, gpu.view, y.data, f, ColumnVersion(v)));
        auto e = error(y.host(nullptr), ref1);
        verify(e);
        auto eq = error(y.host(nullptr), ref2); qdelta = std::max(qdelta, eq.rmse);
        bestErr = std::min(bestErr, e.rmse);
        std::printf("  %-10s C%d fused err=%g\n", name, v, e.rmse);
    }
    // Independent baseline: GPU dequantize + FP32 C1.
    {
        Buffer<float> xg(std::size_t(nodes) * f);
        check(dequantizeXInt4(gpu.q.data, gpu.scale.data, xg.data, nodes, f));
        check(cudaMemsetAsync(y.data, 0, y.size * sizeof(float)));
        check(launchColumnSpMM(gc.view, xg.data, y.data, f, ColumnVersion::C1));
        auto e = error(y.host(nullptr), ref1);
        verify(e);
        std::printf("  %-10s baseline deq+C1 err=%g\n", name, e.rmse);
    }
    // Fused adaptive (cache off / on).
    for (int c = 0; c < 2; ++c) {
        check(cudaMemsetAsync(y.data, 0, y.size * sizeof(float)));
        check(launchColumnAdaptiveInt4SpMM(view, gpu.view, y.data, f, c == 1));
        auto e = error(y.host(nullptr), ref1);
        verify(e);
        auto eq = error(y.host(nullptr), ref2); qdelta = std::max(qdelta, eq.rmse);
        bestErr = std::min(bestErr, e.rmse);
        std::printf("  %-10s dyn%d  fused err=%g\n", name, c, e.rmse);
    }
    std::printf("  %-10s F=%3u tasks=%zu quantization-delta-rmse(max)=%g best-tf32-err=%g\n",
                name, f, fmt.tasks.size(), qdelta, bestErr);
}
} // namespace test

int main() {
    using namespace test;
    Matrix weighted = makeMatrix(19, 269, [](unsigned r, unsigned c) { return (r * 17 + c * 7) % 29 == 0 ? float((r + c) % 5) - 2.f : 0.f; });
    Matrix dense = makeMatrix(65, 401, [](unsigned, unsigned c) { return c < 8 || c >= 393 ? 1.f : 0.f; });
    Matrix zero = makeMatrix(9, 259, [](unsigned, unsigned) { return 0.f; });
    Matrix repeated = makeMatrix(65, 265, [](unsigned r, unsigned c) { return (r % 7) == (c % 11) ? 1.f : 0.f; });
    Matrix negative = makeMatrix(17, 273, [](unsigned r, unsigned c) { return (r * 3 + c * 5) % 13 == 0 ? -float((r + c) % 3) : 0.f; });
    std::vector<Matrix> ms{weighted, dense, zero, repeated, negative};
    const char* names[] = {"weighted", "dense", "zero", "repeated", "negative"};
    std::vector<unsigned> fs{1, 17, 32, 64, 100, 128, 256, 512};
    int failed = 0;
    try {
        for (std::size_t mi = 0; mi < ms.size(); ++mi)
            for (unsigned f : fs) {
                auto x = features(ms[mi].cols, f);
                run(names[mi], ms[mi], x, f);
            }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        failed = 1;
    }
    if (!failed) std::cout << "int4_test: all configurations passed\n";
    return failed;
}
