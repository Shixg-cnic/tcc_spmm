#!/usr/bin/env bash
# Built binary required. Use sudo only if GPU counters require administrator access.
set -euo pipefail
cd "$(dirname "$0")/.."
NCU="${NCU:-/usr/local/cuda-13.0/bin/ncu}"
DEVICE="${DEVICE:-0}"
OUT="${OUT:-$PWD/results/acc_dequant/nsight}"
mkdir -p "$OUT" "$PWD/build/tmp"
export TMPDIR="$PWD/build/tmp"
mode="${1:-all}"
case "$mode" in all|original|int4) ;; *) echo 'Usage: bash bench/profile_acc_dequant.sh [all|original|int4]' >&2; exit 2;; esac
sections=(SpeedOfLight ComputeWorkloadAnalysis MemoryWorkloadAnalysis
          MemoryWorkloadAnalysis_Chart MemoryWorkloadAnalysis_Tables
          LaunchStats Occupancy SchedulerStats WarpStateStats InstructionStats
          SpeedOfLight_HierarchicalTensorRooflineChart SourceCounters)
args=()
for section in "${sections[@]}"; do args+=(--section "$section"); done
for method in original int4; do
    if [[ "$mode" != all && "$mode" != "$method" ]]; then continue; fi
    profile_method=0
    if [[ "$method" == int4 ]]; then profile_method=2; fi
    prefix="$OUT/products_f256_$method"
    "$NCU" --profile-from-start off --launch-count 1 --replay-mode kernel \
        --cache-control all --clock-control none \
        --section-folder /usr/local/cuda-13.0/nsight-compute-2025.3.0/sections \
        "${args[@]}" --force-overwrite --export "$prefix" \
        "$PWD/build/benchmark_acc_dequant" \
        "$PWD/data/ogbn_products_full.csr" "$PWD/data/native_features/ogbn-products.x.bin" \
        256 "$DEVICE" "$prefix" "$profile_method" 2>&1 | tee "$prefix.capture.log"
    test -s "$prefix.ncu-rep"
    "$NCU" --import "$prefix.ncu-rep" --page details > "$prefix.details.txt"
    "$NCU" --import "$prefix.ncu-rep" --page raw --csv > "$prefix.metrics.csv"
done
