// Offline CW-order structure statistics (task-book section 5).
// Pure CPU: reads the CSR, builds the ColumnBaseFormat once, then for every
// strategy x bandRows x group-size emits one reuse row to the given CSV.
//
//   analyze_cw_order matrix.csr out.csv
//
// Columns follow task-book 5.4; y-bytes are reported for both F=128 and F=256
// (task-book 5.3). dominant_band_concentration is only meaningful for
// BandAnchor (0 for the other strategies).
#include "matrix_io.cuh"
#include "column_schedule.cuh"
#include "cw_order.cuh"

#include <iomanip>

using namespace test;

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::cout << "Usage: analyze_cw_order matrix.csr out.csv\n";
            return 0;
        }
        auto m = readMatrix(argv[1]);
        auto base = buildColumnBase(m.rows, m.cols, m.csr);
        const std::uint32_t W = base.colWindowOffset.size() - 1;
        const std::uint32_t groupSizes[3] = {32, 64, 128};

        std::cout << std::setprecision(9);
        std::cout << "strategy,band_rows,F,num_windows,total_tiles,total_row_occurrences,group_size,"
                     "group_reuse_ratio,group_unique_rows_mean,group_unique_rows_p95,group_unique_rows_max,"
                     "group_y_bytes_mean,group_y_bytes_p95,group_y_bytes_max,dominant_band_concentration,"
                     "preprocess_ms,task_order_bytes\n";

        auto emit = [&](const char* name, CWOrderStrategy s, std::uint32_t bandRows) {
            auto r = buildCWOrder(base, CWOrderConfig{s, bandRows, 20261016});
            validateCWOrder(r.order, W);
            auto st = analyzeCWOrder(base, r.order, bandRows, groupSizes);
            const double conc = (s == CWOrderStrategy::BandAnchor) ? st.dominantBandConcentration : 0;
            for (int gi = 0; gi < 3; ++gi) {
                const auto& rs = st.reuse[gi];
                for (std::uint32_t f : {128u, 256u}) {
                    const double mult = double(f) / 128.0;
                    std::cout << name << ',' << bandRows << ',' << f << ','
                              << st.numWindows << ',' << st.totalTiles << ',' << st.totalRowOccurrences << ','
                              << rs.groupSize << ',' << rs.reuseRatio << ','
                              << rs.uniqueRowsMean << ',' << rs.uniqueRowsP95 << ',' << rs.uniqueRowsMax << ','
                              << rs.yBytesMean * mult << ',' << rs.yBytesP95 * mult << ',' << rs.yBytesMax * mult << ','
                              << conc << ',' << r.buildMs << ',' << std::size_t(W) * 4 << '\n';
                }
            }
        };
        emit("Identity", CWOrderStrategy::Identity, 0);
        emit("SeededRandom", CWOrderStrategy::SeededRandom, 0);
        emit("HotAnchor", CWOrderStrategy::HotAnchor, 0);
        for (std::uint32_t band : {8192u, 16384u, 32768u})
            emit("BandAnchor", CWOrderStrategy::BandAnchor, band);
    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << '\n'; return 1; }
    return 0;
}
