# Minimal MCI example

Build an FP32 MCI index, save it, reload it, run filtered search, and report
Recall@10, QPS, and mean/p50/p95/p99 latency. The example computes exact filtered
ground truth over the vectors that satisfy each query's predicate.

## Run

Use Linux with a C++20 compiler for VSAG (GCC 13 or newer), CMake, Make, and the
HDF5 and OpenBLAS development libraries. For example, on Ubuntu:

```bash
sudo apt-get install build-essential gfortran cmake ninja-build libhdf5-dev libopenblas-dev liblapacke-dev
git clone --branch codex/mci-minimal-artifact https://github.com/LightWant/vsag.git
cd vsag
bash examples/mci_repro/run.sh
```

The script builds VSAG in Release mode and generates a small synthetic HDF5
dataset: 20,000 base vectors, 128 independent query vectors, 128 dimensions,
eight Gaussian clusters, and seed `20261109`. Data generation uses the HDF5 C
library and C++ random generators; Python packages and dataset downloads are
not needed. The seed is stored in the HDF5 file. The same executable and seed
reproduce the dataset.

The workload uses random 10% filters, `k=10`, MCI width 120, 64 sampled seeds,
coverage zero, and 16 concurrent query workers. Search uses MCI directly, with
no HGraph routing or scan threshold. Query-internal parallelism is one and the
index stays read-only during search. The base graph used for clique construction
is FP32 NSW with degree 48 and construction width 600; MCI uses `mcs=32`,
`clique_min_size=8`, and initial `alpha=1.2`. These small construction settings
keep the synthetic example quick to build.

Outputs are under `examples/mci_repro/_output/`, including the dataset,
serialized index, build configuration, and `search.log`/`reload.log`. Each log
contains a CSV metrics table. QPS is measured over the concurrent batch wall
time; latency is measured per query. Exact ground truth and predicate creation
are outside the search timer. The script runs three timed passes after warmup
and loads the saved index again in a separate process.

## Use an existing dataset

An HDF5 file must contain floating-point matrices `train[N, dim]` and
`test[nq, dim]`. For example:

```bash
BASE_LIMIT=100000 QUERY_LIMIT=128 MCI_OUTPUT_DIR=/tmp/mci-real \
    bash examples/mci_repro/run.sh /path/to/dataset.hdf5
```

`BASE_LIMIT=0` and `QUERY_LIMIT=0` read all rows. This command uses squared L2
distance; the executable also supports `--metric angular` to normalize vectors
before indexing. Original HDF5 unfiltered neighbors are never used as filtered
ground truth. Choose a new output directory when changing the input dataset.

If VSAG is already built from this checkout, pass its shared library:

```bash
VSAG_LIBRARY="$PWD/build-release/src/libvsag.so" \
    bash examples/mci_repro/run.sh
```

To generate only a dataset, after compiling the example:

```bash
examples/mci_repro/_output/example-build/mci_hdf5_e2e \
    --mode prepare --hdf5 /tmp/mci-data.hdf5 \
    --base-limit 20000 --query-limit 128 --dim 128 --seed 20261109
```

Run `mci_hdf5_e2e --help` for all options. All downloads made by the runner bypass
proxy environment variables and Git's HTTP proxy setting for that invocation.
