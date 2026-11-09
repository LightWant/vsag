#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 || ${1:-} == --help ]]; then
    echo "Usage: bash examples/mci_repro/run.sh [dataset.hdf5]"
    echo "Without a dataset argument, generate a small synthetic HDF5 dataset."
    exit 0
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/../.." && pwd)
output_dir=${MCI_OUTPUT_DIR:-"$script_dir/_output"}
mkdir -p -- "$output_dir"
output_dir=$(cd -- "$output_dir" && pwd)
threads=${THREADS:-16}
base_limit=${BASE_LIMIT:-20000}
query_limit=${QUERY_LIMIT:-128}

# Keep dataset/dependency downloads off host proxy interfaces.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
git_config_count=${GIT_CONFIG_COUNT:-0}
export "GIT_CONFIG_KEY_${git_config_count}=http.proxy"
export "GIT_CONFIG_VALUE_${git_config_count}="
export GIT_CONFIG_COUNT=$((git_config_count + 1))
export OPENBLAS_NUM_THREADS=1

library=${VSAG_LIBRARY:-}
if [[ -z "$library" ]]; then
    make -C "$repo_dir" release COMPILE_JOBS="$threads" \
        RELEASE_BUILD_DIR="$output_dir/vsag-build" VSAG_ENABLE_LIBAIO=OFF \
        2>&1 | tee "$output_dir/vsag-build.log"
    library="$output_dir/vsag-build/src/libvsag.so"
fi
if [[ ! -f "$library" ]]; then
    echo "VSAG shared library not found: $library" >&2
    exit 1
fi

cmake -S "$script_dir" -B "$output_dir/example-build" \
    -DCMAKE_BUILD_TYPE=Release -DVSAG_ROOT="$repo_dir" -DVSAG_LIBRARY="$library"
cmake --build "$output_dir/example-build" --parallel "$threads"
binary="$output_dir/example-build/mci_hdf5_e2e"
dataset=${1:-"$output_dir/synthetic.hdf5"}
if [[ $# == 0 && ! -f "$dataset" ]]; then
    "$binary" --mode prepare --hdf5 "$dataset" --base-limit "$base_limit" \
        --query-limit "$query_limit" --dim 128 --seed 20261109 \
        | tee "$output_dir/prepare.log"
fi

args=(--hdf5 "$dataset" --index "$output_dir/mci.hgraph" --metric l2
      --base-limit "$base_limit" --query-limit "$query_limit" --k 10
      --threads "$threads" --build-threads "$threads" --repeats 3
      --retentions 0.1 --route-threshold 1 --mci-ef 120 --hgraph-ef 60
      --seed-count 64 --coverage 0 --seed-cap 0 --activation-cap 256
      --seed 20261109)
mode=build-search
if [[ -f "$output_dir/mci.hgraph" ]]; then
    mode=load
fi
"$binary" --mode "$mode" "${args[@]}" | tee "$output_dir/search.log"
# A second process verifies that the serialized index can be reused.
"$binary" --mode load "${args[@]}" | tee "$output_dir/reload.log"
echo "Dataset, index, parameters and metrics: $output_dir"
