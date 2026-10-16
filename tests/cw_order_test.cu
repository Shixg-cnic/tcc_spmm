// Correctness tests for the CW execution-order experiment
// (experiment/cw-order-only).
//
// Default mode (no args): built-in small graph covering every required case,
// all four strategies x F=128/256, checked against the TF32-rounded CPU
// reference, plus order-validity and determinism checks.
//
// Full-graph mode: cw_order_test matrix.csr [F] [device] [X.bin] compares the
// FULL output of V2C Original and of every ordered variant against the full
// Acc output (same tolerance), no sampling.
#include "matrix_io.cuh"
#include "acc_reference.cuh"
#include "column_schedule.cuh"
#include "column_fast.cuh"
#include "cw_order.cuh"

using namespace test;
namespace {
using U = std::uint32_t;

// 19 x 269 rectangular graph covering the required small-graph cases:
//   - positive and negative FP32 weights
//   - rectangular (19 rows, 269 cols -> 34 CWs, last CW partial)
//   - a completely empty CW (CW5, cols 40..47, referenced by nobody)
//   - an output row with no neighbours (row 18)
//   - one output row contributed by multiple CWs (row 0: CW0 and CW3)
//   - a high-degree CW (CW0: rows 0..3 with all 8 columns)
//   - duplicate CSR edges (row 2 col 10 appears twice; canonicalCSR sums them)
//   - row count not divisible by 8 (19)
test::Matrix smallGraph() {
    struct E { U r, c; float v; };
    const E es[] = {
        // row 0: CW0 (cols 0,3,5) and CW3 (cols 24,27)
        {0, 0, 1.0f}, {0, 3, -0.5f}, {0, 5, 2.0f}, {0, 24, 1.5f}, {0, 27, -1.0f},
        // row 1: high degree, all 8 columns of CW0 + a far column 264 (partial CW33)
        {1, 0, 0.5f}, {1, 1, 1.0f}, {1, 2, -1.0f}, {1, 3, 0.25f}, {1, 4, 2.0f},
        {1, 5, -2.0f}, {1, 6, 1.0f}, {1, 7, -0.75f}, {1, 264, 3.0f},
        // row 2: duplicate coordinate (2,10) 1.5 + (-0.5) = 1.0 after canonical
        {2, 10, 1.5f}, {2, 10, -0.5f}, {2, 11, 0.75f},
        // row 3: negative weights, CW1 + CW4
        {3, 8, -2.5f}, {3, 9, -1.25f}, {3, 33, 0.5f},
        // rows 4..7: CW1/CW2 scattered
        {4, 8, 1.0f}, {5, 9, -1.0f}, {6, 15, 2.0f}, {7, 16, -0.5f}, {7, 23, 1.25f},
        // rows 8..17: cover CW2/CW6/CW8 and an isolated-neighbour band
        {8, 16, 0.5f}, {9, 17, -1.5f}, {10, 18, 2.5f}, {11, 19, -0.25f},
        {12, 20, 1.0f}, {13, 21, -2.0f}, {14, 22, 0.75f}, {15, 23, -0.5f},
        {16, 48, 1.0f}, {16, 55, -1.0f}, {17, 49, 2.0f}, {17, 50, -2.0f},
        // row 18: no edges (isolated output row)
        // CW5 (cols 40..47): deliberately untouched -> empty CW
    };
    CanonicalCSR raw;
    std::vector<std::vector<std::pair<U, float>>> buckets(19);
    for (auto& e : es) buckets[e.r].emplace_back(e.c, e.v);
    for (auto& b : buckets) {
        std::sort(b.begin(), b.end()); // ascending col; canonical will merge dups
        for (auto& p : b) { raw.src.push_back(p.first); raw.values.push_back(p.second); }
        raw.ptr.push_back(raw.src.size());
    }
    return {19, 269, canonicalCSR(19, 269, raw.src.size(), raw.ptr.data(), raw.src.data(), raw.values.data())};
}

const char* strategyName(CWOrderStrategy s) {
    switch (s) {
        case CWOrderStrategy::Identity: return "Identity";
        case CWOrderStrategy::SeededRandom: return "SeededRandom";
        case CWOrderStrategy::HotAnchor: return "HotAnchor";
        case CWOrderStrategy::BandAnchor: return "BandAnchor";
    }
    return "?";
}

// Acc writes paddedRows*F rows; all methods share a padded Y buffer and the
// comparison always truncates to the real m.rows*F values.
inline std::vector<float> head(std::vector<float> v, std::size_t n) { v.resize(n); return v; }

// Run every ordered variant on the given (already uploaded) view and check
// the FULL output against the reference. Also returns the V2C Original and
// Acc outputs for the caller to compare.
void runSmallCase(const test::Matrix& m, U f, cudaStream_t stream) {
    auto base = buildColumnBase(m.rows, m.cols, m.csr);
    const U W = U(base.colWindowOffset.size() - 1);
    const U paddedRows = U((std::uint64_t(m.rows) + 7) / 8 * 8);
    auto x = features(m.cols, f);
    Buffer<float> input(x), y(std::size_t(paddedRows) * f);
    Buffer<std::uint32_t> win(base.colWindowOffset), rows(base.tileRows), off(base.tileValueOffset);
    Buffer<std::uint64_t> bits(base.tileLocalBit);
    Buffer<float> vals(base.values);
    DeviceColumnFast view{base.rows, base.cols, win.data, rows.data, off.data, bits.data, vals.data};
    auto ref = reference(m, x, f, true);
    const std::size_t real = std::size_t(m.rows) * f;

    // Acc sanity (the reference itself)
    {
        auto accF = convertAccRow(m.rows, m.cols, m.csr);
        Buffer<std::uint64_t> accBits(accF.tcLocalBit);
        Buffer<std::uint32_t> accWindow(accF.rowWindowOffset), accOffset(accF.tcOffset),
            accMapping(accF.sparseA2B);
        Buffer<float> accValues(accF.values);
        check(cudaMemsetAsync(y.data, 0, std::size_t(paddedRows) * f * sizeof(float), stream));
        check(launchAccRowSpMM(accWindow.data, accOffset.data, accMapping.data, accBits.data,
            accValues.data, input.data, y.data, paddedRows, m.cols, f, stream));
        auto out = y.host(stream);
        verify(error(head(out, real), ref));
    }
    // V2C Original
    {
        check(launchColumnFast(view, input.data, y.data, f, FAST_V2C, 128, 1, stream, true));
        auto out = y.host(stream);
        verify(error(head(out, real), ref));
    }
    // the four strategies, small bandRows=8 (19 rows -> 3 bands)
    const CWOrderStrategy strategies[] = {
        CWOrderStrategy::Identity, CWOrderStrategy::SeededRandom,
        CWOrderStrategy::HotAnchor, CWOrderStrategy::BandAnchor};
    const U bandRows = 8;
    for (auto s : strategies) {
        CWOrderResult r1 = buildCWOrder(base, CWOrderConfig{s, bandRows, 20261016});
        CWOrderResult r2 = buildCWOrder(base, CWOrderConfig{s, bandRows, 20261016});
        require(r1.order == r2.order, (std::string(strategyName(s)) + " order not deterministic").c_str());
        validateCWOrder(r1.order, W);
        require(r1.order.size() == W, (std::string(strategyName(s)) + " size != W").c_str());
        Buffer<U> d(r1.order);
        check(cudaMemcpy(d.data, r1.order.data(), W * 4, cudaMemcpyHostToDevice));
        // first run
        check(launchColumnFastOrdered(view, input.data, y.data, d.data, f, stream, true));
        auto out1 = y.host(stream);
        // second run (no re-upload; guards against uninitialized state)
        check(launchColumnFastOrdered(view, input.data, y.data, d.data, f, stream, true));
        auto out2 = y.host(stream);
        require(head(out1, real) == head(out2, real), (std::string(strategyName(s)) + " run-to-run mismatch").c_str());
        Error e = error(head(out1, real), ref);
        std::cout << "small " << strategyName(s) << " F=" << f
                  << " W=" << W << " rmse=" << e.rmse
                  << " maxAbs=" << e.maxAbsolute << '\n';
        verify(e);
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
            std::cout << "Usage: cw_order_test            (built-in small graph)\n"
                      << "       cw_order_test matrix.csr [F] [device] [X.bin]  (full-graph vs Acc)\n";
            return 0;
        }
        check(cudaSetDevice(argc > 3 ? std::stoi(argv[3]) : 0));
        cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        if (argc < 2) {
            // ---- small-graph mode ----
            auto m = smallGraph();
            runSmallCase(m, 128, stream);
            runSmallCase(m, 256, stream);
            std::cout << "small-graph correctness OK\n";
        } else {
            // ---- full-graph mode: FULL output vs full Acc output ----
            const U features = U(argc > 2 ? std::stoi(argv[2]) : 128);
            require(features == 128 || features == 256, "F must be 128 or 256");
            auto m = readMatrix(argv[1]);
            auto base = buildColumnBase(m.rows, m.cols, m.csr);
            const U W = U(base.colWindowOffset.size() - 1);
            const U paddedRows = U((std::uint64_t(m.rows) + 7) / 8 * 8);
            auto x = readFeatures(argc > 4 ? argv[4] : nullptr, m.cols, features);
            const std::size_t real = std::size_t(m.rows) * features;
            Buffer<float> input(x), y(std::size_t(paddedRows) * features);
            Buffer<std::uint32_t> win(base.colWindowOffset), rows(base.tileRows), off(base.tileValueOffset);
            Buffer<std::uint64_t> bits(base.tileLocalBit);
            Buffer<float> vals(base.values);
            DeviceColumnFast view{base.rows, base.cols, win.data, rows.data, off.data, bits.data, vals.data};
            auto accF = convertAccRow(m.rows, m.cols, m.csr);
            Buffer<std::uint64_t> accBits(accF.tcLocalBit);
            Buffer<std::uint32_t> accWindow(accF.rowWindowOffset), accOffset(accF.tcOffset),
                accMapping(accF.sparseA2B);
            Buffer<float> accValues(accF.values);
            auto accOut = [&]() {
                check(cudaMemsetAsync(y.data, 0, std::size_t(paddedRows) * features * sizeof(float), stream));
                check(launchAccRowSpMM(accWindow.data, accOffset.data, accMapping.data, accBits.data,
                    accValues.data, input.data, y.data, paddedRows, m.cols, features, stream));
                check(cudaStreamSynchronize(stream));
                return y.host(stream);
            }();

            auto vsAcc = [&](const char* name, const std::function<void()>& launch) {
                launch();
                auto out = y.host(stream);
                Error e = error(head(out, real), head(accOut, real));
                std::cout << "full " << name << " F=" << features << " M=" << m.rows
                          << " rmse=" << e.rmse << " maxAbs=" << e.maxAbsolute << '\n';
                require(e.rmse < 2e-4 && e.maxAbsolute < 2e-4 + e.maxReference * 5e-4,
                    (std::string(name) + " vs Acc mismatch").c_str());
            };
            vsAcc("V2C_Original", [&]() {
                check(launchColumnFast(view, input.data, y.data, features, FAST_V2C, 128, 1, stream, true));
            });
            const CWOrderStrategy strategies[] = {
                CWOrderStrategy::Identity, CWOrderStrategy::SeededRandom,
                CWOrderStrategy::HotAnchor, CWOrderStrategy::BandAnchor};
            for (auto s : strategies) {
                CWOrderResult r = buildCWOrder(base, CWOrderConfig{s, 16384, 20261016});
                validateCWOrder(r.order, W);
                Buffer<U> d(r.order);
                check(cudaMemcpy(d.data, r.order.data(), W * 4, cudaMemcpyHostToDevice));
                vsAcc(strategyName(s), [&]() {
                    check(launchColumnFastOrdered(view, input.data, y.data, d.data, features, stream, true));
                });
            }
            std::cout << "full-graph correctness OK\n";
        }
        check(cudaStreamDestroy(stream));
    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << '\n'; return 1; }
    return 0;
}
