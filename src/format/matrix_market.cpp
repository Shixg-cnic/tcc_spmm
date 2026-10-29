#include "format/csr.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
std::string lowercase(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

bool nextDataLine(std::ifstream& input, std::string& line) {
    while (std::getline(input, line)) {
        auto first = line.find_first_not_of(" \t\r");
        if (first != std::string::npos && line[first] != '%') return true;
    }
    return false;
}
}

MtxMatrix loadMatrixMarket(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open Matrix Market file: " + path);
    std::string line, banner, object, storage, field, symmetry;
    std::getline(input, line);
    std::istringstream header(line);
    header >> banner >> object >> storage >> field >> symmetry;
    banner = lowercase(banner); object = lowercase(object); storage = lowercase(storage);
    field = lowercase(field); symmetry = lowercase(symmetry);
    if (banner != "%%matrixmarket" || object != "matrix" || storage != "coordinate" ||
        (field != "real" && field != "integer" && field != "pattern") ||
        (symmetry != "general" && symmetry != "symmetric"))
        throw std::runtime_error("expected a coordinate real/integer/pattern general/symmetric Matrix Market file");
    if (!nextDataLine(input, line)) throw std::runtime_error("missing Matrix Market dimensions");
    std::uint64_t rows64, cols64, entries64;
    std::istringstream dimensions(line);
    constexpr auto max = std::numeric_limits<std::uint32_t>::max();
    if (!(dimensions >> rows64 >> cols64 >> entries64) || !rows64 || !cols64 ||
        rows64 > max || cols64 > max || entries64 > max ||
        (symmetry == "symmetric" && rows64 != cols64))
        throw std::runtime_error("invalid or unsupported Matrix Market dimensions");
    const auto rows = static_cast<std::uint32_t>(rows64);
    const auto cols = static_cast<std::uint32_t>(cols64);
    const auto start = input.tellg();
    std::vector<std::uint32_t> offsets(std::size_t(rows) + 1, 0);

    // First count entries in each row; symmetric files store one triangle.
    auto readEntries = [&](auto consume) {
        for (std::uint64_t i = 0; i < entries64; ++i) {
            if (!nextDataLine(input, line)) throw std::runtime_error("truncated Matrix Market file");
            std::uint64_t row, col;
            float value = 1.0f;
            std::istringstream record(line);
            if (!(record >> row >> col) || !row || row > rows || !col || col > cols ||
                (field != "pattern" && !(record >> value)) || !std::isfinite(value))
                throw std::runtime_error("invalid Matrix Market entry");
            consume(static_cast<std::uint32_t>(row - 1), static_cast<std::uint32_t>(col - 1), value);
            if (symmetry == "symmetric" && row != col)
                consume(static_cast<std::uint32_t>(col - 1), static_cast<std::uint32_t>(row - 1), value);
        }
    };
    std::uint64_t expanded = 0;
    readEntries([&](std::uint32_t row, std::uint32_t, float) {
        if (++expanded > max) throw std::overflow_error("Matrix Market matrix exceeds uint32 CSR offsets");
        ++offsets[std::size_t(row) + 1];
    });
    for (std::size_t row = 1; row < offsets.size(); ++row) offsets[row] += offsets[row - 1];
    std::vector<std::uint32_t> columns(expanded), cursor(offsets.begin(), offsets.end() - 1);
    std::vector<float> values(expanded);
    if (entries64) {
        input.clear();
        input.seekg(start);
        if (!input) throw std::runtime_error("cannot seek Matrix Market file: " + path);
        readEntries([&](std::uint32_t row, std::uint32_t col, float value) {
            auto index = cursor[row]++;
            columns[index] = col;
            values[index] = value;
        });
    }
    auto csr = makeCanonicalCsr(rows, cols, static_cast<std::uint32_t>(expanded),
                                offsets.data(), columns.data(), values.data());
    return {rows, cols, std::move(csr)};
}
