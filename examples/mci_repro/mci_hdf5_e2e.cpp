// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
// Standalone example for VSAG main containing PR #3179 (bcb0417 or newer).
// Dependencies: VSAG shared library, HDF5 C library, OpenMP, C++17.
// HDF5 datasets: train [N, dim], test [nq, dim]. Labels are zero-based row IDs.
// Filtered ground truth is recomputed exactly; unfiltered HDF5 neighbors are NOT reused.

#include <hdf5.h>
#include <omp.h>
#include <vsag/filter.h>
#include <vsag/vsag.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

double
seconds(Clock::time_point begin) {
    return std::chrono::duration<double>(Clock::now() - begin).count();
}

void
require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename T>
void
check(const T& result, const std::string& operation) {
    if (!result.has_value()) {
        throw std::runtime_error(operation + ": " + result.error().message);
    }
}

template <typename Function>
void
parallel_for(uint64_t count, int threads, Function&& function) {
    std::exception_ptr error;
    std::mutex error_mutex;
    std::atomic<bool> failed{false};
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (uint64_t i = 0; i < count; ++i) {
        if (!failed.load(std::memory_order_relaxed)) {
            try {
                function(i);
            } catch (...) {
                std::lock_guard<std::mutex> guard(error_mutex);
                if (!error) {
                    error = std::current_exception();
                }
                failed.store(true, std::memory_order_relaxed);
            }
        }
    }
    if (error) {
        std::rethrow_exception(error);
    }
}

struct Options {
    std::string hdf5;
    std::string index = "mci.hgraph";
    std::string mode = "build-search";
    std::string metric = "l2";
    uint64_t base_limit = 20000;
    uint64_t dim = 128;
    uint64_t query_limit = 128;
    uint64_t k = 10;
    int threads = 16;
    int build_threads = 16;
    int repeats = 3;
    uint64_t mci_ef = 120;
    uint64_t hgraph_ef = 60;
    uint64_t seed_count = 64;
    uint64_t activation_cap = 256;
    double route_threshold = 1.0;
    double coverage = 0.0;
    uint64_t seed_cap = 0;
    uint64_t seed = 20261109;
    std::vector<double> retentions{0.1};
};

void
help() {
    std::cout << R"(Usage: mci_hdf5_e2e --hdf5 FILE [options]
  --index FILE            Output/input index (default mci.hgraph)
  --mode prepare|build-search|load  Generate data; build/save/reload; or only load
  --metric angular|l2     angular: normalize vectors, then use squared L2
  --base-limit N          Read first N train rows (default 20000; 0 = all)
  --query-limit N         Read first N test rows (default 128; 0 = all)
  --dim N                Synthetic dataset dimension (default 128)
  --k K                  Recall@K (default 10)
  --threads N            Concurrent queries and exact-GT workers (default 16)
  --build-threads N      Index construction threads (default 16)
  --repeats N            Timed full-query passes after warmup (default 3)
  --retentions LIST      Rate (default 0.1) or mixed rates, e.g. 0.005,0.01
  --mci-ef N             MCI ef_search (default 120)
  --hgraph-ef N          High-retention ef_search (default 60)
  --route-threshold R    Native MCI/HGraph routing threshold (default 1)
  --seed-count N         sqrt(N_base)*ratio floor target (default 64)
  --coverage R           mci_seed_coverage (default 0)
  --seed-cap N           Coverage budget cap (default 0; unlimited)
  --activation-cap N     Per-node clique activation cap (default 256; 0 = unlimited)
  --seed N               Random workload seed (default 20261109)

Build settings: FP32, NSW degree=48, ef_construction=600, MCS=32,
clique_min_size=8 (parameter mci_clique_max), alpha=1.2, KNNG source=hgraph.
Search: search_no_lock=true, parallelism=1, use_hybrid_traversal=false,
brute_force_threshold=0, packed filter + batch distance, hops_limit=9.
The index is immutable during all searches. Query and index owners outlive workers.
Exact filtered GT and predicate materialization are outside search timers.
Overall QPS = completed queries / concurrent batch wall time, NOT threads/mean latency.
Group QPS is omitted because groups execute together in the mixed workload.
Index configuration and preprocessing metadata are saved beside the index.
)";
}

Options
parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help" || key == "-h") {
            help();
            std::exit(0);
        }
        require(i + 1 < argc, "missing value for " + key);
        const std::string value = argv[++i];
        if (key == "--hdf5")
            o.hdf5 = value;
        else if (key == "--index")
            o.index = value;
        else if (key == "--mode")
            o.mode = value;
        else if (key == "--metric")
            o.metric = value;
        else if (key == "--base-limit")
            o.base_limit = std::stoull(value);
        else if (key == "--query-limit")
            o.query_limit = std::stoull(value);
        else if (key == "--dim")
            o.dim = std::stoull(value);
        else if (key == "--k")
            o.k = std::stoull(value);
        else if (key == "--threads")
            o.threads = std::stoi(value);
        else if (key == "--build-threads")
            o.build_threads = std::stoi(value);
        else if (key == "--repeats")
            o.repeats = std::stoi(value);
        else if (key == "--mci-ef")
            o.mci_ef = std::stoull(value);
        else if (key == "--hgraph-ef")
            o.hgraph_ef = std::stoull(value);
        else if (key == "--seed-count")
            o.seed_count = std::stoull(value);
        else if (key == "--activation-cap")
            o.activation_cap = std::stoull(value);
        else if (key == "--route-threshold")
            o.route_threshold = std::stod(value);
        else if (key == "--coverage")
            o.coverage = std::stod(value);
        else if (key == "--seed-cap")
            o.seed_cap = std::stoull(value);
        else if (key == "--seed")
            o.seed = std::stoull(value);
        else if (key == "--retentions") {
            o.retentions.clear();
            std::istringstream input(value);
            std::string part;
            while (std::getline(input, part, ',')) o.retentions.push_back(std::stod(part));
        } else {
            throw std::runtime_error("unknown option: " + key);
        }
    }
    require(!o.hdf5.empty(), "--hdf5 is required (see --help)");
    require(o.mode == "prepare" || o.mode == "build-search" || o.mode == "load", "invalid --mode");
    require(o.metric == "angular" || o.metric == "l2", "invalid --metric");
    require(o.k > 0 && o.k <= 100000, "invalid k");
    require(o.threads > 0 && o.build_threads > 0 && o.repeats > 0, "invalid thread/pass count");
    require(o.mci_ef >= o.k && o.hgraph_ef >= o.k, "both ef values must be >= k");
    require(std::isfinite(o.coverage) && o.coverage >= 0 && o.coverage <= 1,
            "coverage must be in [0,1]");
    require(std::isfinite(o.route_threshold) && o.route_threshold >= 0 && o.route_threshold <= 1,
            "route threshold must be in [0,1]");
    require(!o.retentions.empty(), "empty retention list");
    for (double r : o.retentions) {
        require(std::isfinite(r) && r > 0 && r <= 1, "retention must be in (0,1]");
    }
    return o;
}

struct H5Handle {
    hid_t id;
    herr_t (*close)(hid_t);
    H5Handle(hid_t value, herr_t (*closer)(hid_t)) : id(value), close(closer) {
        require(id >= 0, "HDF5 open/create failed");
    }
    ~H5Handle() {
        close(id);
    }
    H5Handle(const H5Handle&) = delete;
    H5Handle&
    operator=(const H5Handle&) = delete;
};

struct Matrix {
    uint64_t n;
    uint64_t dim;
    std::vector<float> values;
    const float*
    row(uint64_t i) const {
        return values.data() + i * dim;
    }
};

Matrix
read_hdf5(const std::string& path, const char* key, uint64_t limit) {
    H5Handle file(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
    H5Handle data(H5Dopen2(file.id, key, H5P_DEFAULT), H5Dclose);
    H5Handle space(H5Dget_space(data.id), H5Sclose);
    require(H5Sget_simple_extent_ndims(space.id) == 2, "HDF5 dataset must be a matrix");
    hsize_t shape[2];
    require(H5Sget_simple_extent_dims(space.id, shape, nullptr) >= 0, "HDF5 shape failed");
    if (limit != 0)
        shape[0] = std::min<hsize_t>(shape[0], limit);
    require(
        shape[0] > 0 && shape[1] > 0 && shape[0] <= std::numeric_limits<uint64_t>::max() / shape[1],
        "invalid HDF5 matrix shape");
    Matrix out{shape[0], shape[1], std::vector<float>(shape[0] * shape[1])};
    const hsize_t offset[2]{0, 0};
    require(H5Sselect_hyperslab(space.id, H5S_SELECT_SET, offset, nullptr, shape, nullptr) >= 0,
            "HDF5 hyperslab failed");
    H5Handle memory(H5Screate_simple(2, shape, nullptr), H5Sclose);
    require(
        H5Dread(data.id, H5T_NATIVE_FLOAT, memory.id, space.id, H5P_DEFAULT, out.values.data()) >=
            0,
        "HDF5 read failed");
    return out;
}

// Make a small, independent base/query dataset using only the HDF5 C library.
void
generate_hdf5(const Options& o) {
    require(!std::filesystem::exists(o.hdf5), "dataset exists; choose another output path");
    require(o.base_limit >= 1000 && o.query_limit > 0 && o.dim > 0 && o.dim <= 4096,
            "prepare requires base-limit >= 1000, query-limit > 0, and dim in [1,4096]");
    require(o.base_limit <= std::numeric_limits<uint64_t>::max() / o.dim,
            "synthetic matrix is too large");
    const auto parent = std::filesystem::path(o.hdf5).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    std::mt19937_64 rng(o.seed);
    std::uniform_real_distribution<float> center(-2.0F, 2.0F);
    std::normal_distribution<float> noise(0.0F, 0.5F);
    constexpr uint64_t clusters = 8;
    std::vector<float> centers(clusters * o.dim);
    for (auto& value : centers) {
        value = center(rng);
    }
    H5Handle file(H5Fcreate(o.hdf5.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT), H5Fclose);
    auto write = [&](const char* name, uint64_t rows) {
        require(rows <= std::numeric_limits<uint64_t>::max() / o.dim,
                "synthetic matrix is too large");
        std::vector<float> values(rows * o.dim);
        for (uint64_t i = 0; i < rows; ++i) {
            const auto c = std::uniform_int_distribution<uint64_t>(0, clusters - 1)(rng);
            for (uint64_t d = 0; d < o.dim; ++d) {
                values[i * o.dim + d] = centers[c * o.dim + d] + noise(rng);
            }
        }
        const hsize_t shape[2]{rows, o.dim};
        H5Handle space(H5Screate_simple(2, shape, nullptr), H5Sclose);
        H5Handle data(
            H5Dcreate2(
                file.id, name, H5T_IEEE_F32LE, space.id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
            H5Dclose);
        require(
            H5Dwrite(data.id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) >= 0,
            "HDF5 dataset write failed");
    };
    write("train", o.base_limit);
    write("test", o.query_limit);
    H5Handle scalar(H5Screate(H5S_SCALAR), H5Sclose);
    H5Handle seed(H5Acreate2(file.id, "seed", H5T_STD_U64LE, scalar.id, H5P_DEFAULT, H5P_DEFAULT),
                  H5Aclose);
    require(H5Awrite(seed.id, H5T_NATIVE_UINT64, &o.seed) >= 0, "seed attribute write failed");
    std::cout << "generated_hdf5=" << o.hdf5 << " base=" << o.base_limit
              << " queries=" << o.query_limit << " dim=" << o.dim << " clusters=" << clusters
              << " seed=" << o.seed << std::endl;
}

void
preprocess(Matrix& matrix, const Options& o) {
    parallel_for(matrix.n, o.threads, [&](uint64_t i) {
        auto* row = matrix.values.data() + i * matrix.dim;
        double norm = 0;
        for (uint64_t d = 0; d < matrix.dim; ++d) {
            require(std::isfinite(row[d]), "nonfinite input vector");
            norm += static_cast<double>(row[d]) * row[d];
        }
        if (o.metric == "angular") {
            require(norm > 0, "angular distance needs nonzero input vectors");
            const double scale = 1.0 / std::sqrt(norm);
            for (uint64_t d = 0; d < matrix.dim; ++d) row[d] *= scale;
        }
    });
}

class Allowed final : public vsag::Filter {
public:
    std::vector<uint8_t> bitmap;
    std::vector<int64_t> ids;
    Allowed(uint64_t n, uint64_t count, uint64_t seed) : bitmap(n, 0) {
        ids.reserve(count);
        std::mt19937_64 rng(seed);
        // Floyd sampling: exact uniform sample without replacement, O(count) work.
        for (uint64_t j = n - count; j < n; ++j) {
            const uint64_t trial = std::uniform_int_distribution<uint64_t>(0, j)(rng);
            const uint64_t chosen = bitmap[trial] ? j : trial;
            bitmap[chosen] = 1;
            ids.push_back(static_cast<int64_t>(chosen));
        }
        std::sort(ids.begin(), ids.end());
    }
    bool
    CheckValid(int64_t id) const override {
        return id >= 0 && static_cast<uint64_t>(id) < bitmap.size() && bitmap[id] != 0;
    }
    float
    ValidRatio() const override {
        return static_cast<float>(ids.size()) / static_cast<float>(bitmap.size());
    }
    void
    GetValidIds(const int64_t** out, int64_t& count) const override {
        *out = ids.data();
        count = static_cast<int64_t>(ids.size());
    }
    const uint8_t*
    GetValidBitmap(uint64_t* size) const override {
        if (size != nullptr)
            *size = bitmap.size();
        return bitmap.data();
    }
};

std::vector<int64_t>
exact_topk(const Matrix& base, const float* query, const Allowed& filter, uint64_t k) {
    using Item = std::pair<double, int64_t>;
    std::priority_queue<Item> heap;
    for (int64_t id : filter.ids) {
        double distance = 0;
        const float* vector = base.row(id);
#pragma omp simd reduction(+ : distance)
        for (uint64_t d = 0; d < base.dim; ++d) {
            const double diff = static_cast<double>(query[d]) - vector[d];
            distance += diff * diff;
        }
        const Item item{distance, id};
        if (heap.size() < k)
            heap.push(item);
        else if (item < heap.top()) {
            heap.pop();
            heap.push(item);
        }
    }
    std::vector<int64_t> result(k);
    for (uint64_t i = k; i > 0; --i) {
        result[i - 1] = heap.top().second;
        heap.pop();
    }
    return result;
}

std::string
build_params(uint64_t dim, const Options& o) {
    std::ostringstream s;
    s << R"({"dtype":"float32","metric_type":"l2","dim":)" << dim
      << R"(,"index_param":{"base_quantization_type":"fp32",)"
      << R"("precise_quantization_type":"fp32","base_io_type":"block_memory_io",)"
      << R"("store_raw_vector":true,"use_reorder":false,"graph_type":"nsw","max_degree":48,)"
      << R"("ef_construction":600,"build_thread_count":)" << o.build_threads
      << R"(,"use_mci":true,"mci_mcs":32,"mci_clique_max":8,)"
      << R"("mci_alpha":1.2,"mci_knng_source":"hgraph"}})";
    return s.str();
}

std::string
search_params(uint64_t n, const Options& o, bool high) {
    // Subtract a fraction to make ceil(sqrt(N) * float(ratio)) equal seed_count.
    // The coverage term may raise it; seed_count is NOT a hard cap on all seed sources.
    const double ratio = o.seed_count == 0 ? 0 : (o.seed_count - 0.25) / std::sqrt(n);
    std::ostringstream s;
    s << std::setprecision(17) << R"({"hgraph":{"ef_search":)" << (high ? o.hgraph_ef : o.mci_ef)
      << R"(,"use_mci":true,"use_hybrid_traversal":false,"search_no_lock":true,)"
      << R"("parallelism":1,"brute_force_threshold":0,"hgraph_valid_ratio_threshold":)"
      << o.route_threshold << R"(,"mci_seed_ratio":)" << ratio << R"(,"mci_seed_coverage":)"
      << o.coverage << R"(,"mci_seed_max_count":)" << o.seed_cap << R"(,"mci_activation_cap":)"
      << o.activation_cap
      << R"(,"mci_packed_filter":true,"mci_batch_distance":true,"mci_hops_limit":9,)"
      << R"("mci_expansion_idle_window":0,"mci_expansion_lower_bound":false}})";
    return s.str();
}

std::string
read_text(const std::string& path) {
    std::ifstream input(path);
    require(bool(input), "cannot open " + path);
    return std::string(std::istreambuf_iterator<char>(input), {});
}

void
write_text(const std::string& path, const std::string& text) {
    std::ofstream output(path);
    output << text << '\n';
    output.close();
    require(bool(output), "cannot write " + path);
}

vsag::IndexPtr
save_and_load(vsag::Engine& engine, Matrix& base, const Options& o) {
    const std::string config_file = o.index + ".build.json";
    const std::string meta_file = o.index + ".input.txt";
    std::string config;
    if (o.mode == "build-search") {
        require(!std::filesystem::exists(o.index), "index already exists; use --mode load");
        const auto parent = std::filesystem::path(o.index).parent_path();
        if (!parent.empty())
            std::filesystem::create_directories(parent);
        config = build_params(base.dim, o);
        auto made = engine.CreateIndex("hgraph", config);
        check(made, "CreateIndex");
        auto index = made.value();
        std::vector<int64_t> ids(base.n);
        std::iota(ids.begin(), ids.end(), 0);
        auto dataset = vsag::Dataset::Make()
                           ->NumElements(base.n)
                           ->Dim(base.dim)
                           ->Ids(ids.data())
                           ->Float32Vectors(base.values.data())
                           ->Owner(false);
        auto started = Clock::now();
        auto built = index->Build(dataset);
        check(built, "Build");
        require(built.value().empty(), "Build returned failed IDs");
        std::cout << "build_seconds=" << seconds(started) << '\n';
        started = Clock::now();
        std::ofstream output(o.index + ".tmp", std::ios::binary);
        require(bool(output), "cannot create index output");
        auto saved = index->Serialize(output);
        check(saved, "Serialize");
        output.close();
        require(bool(output), "index write/close failed");
        write_text(config_file, config);
        write_text(meta_file,
                   o.metric + " " + std::to_string(base.n) + " " + std::to_string(base.dim));
        std::filesystem::rename(o.index + ".tmp", o.index);
        std::cout << "serialize_seconds=" << seconds(started)
                  << " index_bytes=" << std::filesystem::file_size(o.index) << '\n';
        // Both this index and its Dataset/label owners leave scope before Deserialize below.
    } else {
        config = read_text(config_file);
        std::istringstream meta(read_text(meta_file));
        std::string metric;
        uint64_t n = 0, dim = 0;
        meta >> metric >> n >> dim;
        require(metric == o.metric && n == base.n && dim == base.dim,
                "loaded HDF5 prefix/metric differs from saved preprocessing metadata");
    }
    auto created = engine.CreateIndex("hgraph", config);
    check(created, "CreateIndex for Deserialize");
    auto loaded = created.value();
    std::ifstream input(o.index, std::ios::binary);
    require(bool(input), "cannot read saved index");
    const auto started = Clock::now();
    auto restored = loaded->Deserialize(input);
    check(restored, "Deserialize");
    require(loaded->GetNumElements() == static_cast<int64_t>(base.n), "index count mismatch");
    std::cout << "deserialize_seconds=" << seconds(started)
              << " loaded_vectors=" << loaded->GetNumElements() << '\n';
    return loaded;
}

double
percentile(std::vector<double> values, double fraction) {
    require(!values.empty(), "empty latency sample");
    const uint64_t rank = static_cast<uint64_t>(std::ceil(fraction * values.size())) - 1;
    std::nth_element(values.begin(), values.begin() + rank, values.end());
    return values[rank];
}

void
run(vsag::Engine& engine, const Options& o) {
    auto started = Clock::now();
    auto base = read_hdf5(o.hdf5, "train", o.base_limit);
    auto query = read_hdf5(o.hdf5, "test", o.query_limit);
    require(base.dim == query.dim && o.k <= base.n, "invalid base/query dimensions or k");
    preprocess(base, o);
    preprocess(query, o);
    std::cout << "vsag_version=" << vsag::version() << '\n'
              << "base=" << base.n << " queries=" << query.n << " dim=" << base.dim
              << " metric=" << o.metric << " load_preprocess_seconds=" << seconds(started)
              << std::endl;
    auto index = save_and_load(engine, base, o);
    std::vector<std::shared_ptr<Allowed>> filters(query.n);
    std::vector<double> rates(query.n);
    std::mt19937_64 assignment(o.seed);
    for (uint64_t q = 0; q < query.n; ++q) {
        rates[q] = o.retentions[std::uniform_int_distribution<uint64_t>(
            0, o.retentions.size() - 1)(assignment)];
        const uint64_t count = std::min(
            base.n, static_cast<uint64_t>(std::llround(static_cast<double>(base.n) * rates[q])));
        require(count >= o.k, "retention has fewer than k valid IDs; increase base/rate");
        filters[q] = std::make_shared<Allowed>(base.n, count, o.seed + 100000 + q);
    }
    std::vector<std::vector<int64_t>> truth(query.n);
    started = Clock::now();
    parallel_for(query.n, o.threads, [&](uint64_t q) {
        truth[q] = exact_topk(base, query.row(q), *filters[q], o.k);
    });
    std::cout << "exact_filtered_gt_seconds=" << seconds(started) << std::endl;

    std::vector<vsag::DatasetPtr> queries(query.n);
    for (uint64_t q = 0; q < query.n; ++q) {
        queries[q] = vsag::Dataset::Make()
                         ->NumElements(1)
                         ->Dim(query.dim)
                         ->Float32Vectors(query.row(q))
                         ->Owner(false);
    }
    const auto low = search_params(base.n, o, false);
    const auto high = search_params(base.n, o, true);
    std::cout << "mci_parameters=" << low << '\n' << "hgraph_parameters=" << high << '\n';
    auto search = [&](uint64_t q, std::vector<int64_t>& out) {
        // Both ef selection and the native route decision are inside the query timer.
        const bool graph = filters[q]->ValidRatio() >= static_cast<float>(o.route_threshold);
        auto result = index->KnnSearch(queries[q], o.k, graph ? high : low, filters[q]);
        check(result, "KnnSearch");
        auto data = result.value();
        require(data->GetDim() <= static_cast<int64_t>(o.k), "result exceeds k");
        std::copy_n(data->GetIds(), data->GetDim(), out.data() + q * o.k);
    };
    std::vector<int64_t> reference(query.n * o.k, -1);
    parallel_for(query.n, o.threads, [&](uint64_t q) { search(q, reference); });  // Warmup.
    std::vector<double> recalls(query.n);
    for (uint64_t q = 0; q < query.n; ++q) {
        uint64_t hits = 0;
        for (uint64_t j = 0; j < o.k; ++j) {
            const auto id = reference[q * o.k + j];
            if (id == -1)
                continue;  // Missing results count as recall misses.
            require(filters[q]->CheckValid(id), "result violates predicate");
            const auto begin = reference.begin() + q * o.k;
            require(std::find(begin, begin + j, id) == begin + j, "duplicate result ID");
            hits += std::find(truth[q].begin(), truth[q].end(), id) != truth[q].end();
        }
        recalls[q] = static_cast<double>(hits) / o.k;
    }
    std::vector<double> latency(query.n * o.repeats);
    double wall_seconds = 0;
    for (int pass = 0; pass < o.repeats; ++pass) {
        std::vector<int64_t> ids(query.n * o.k, -1);
        started = Clock::now();
        parallel_for(query.n, o.threads, [&](uint64_t q) {
            const auto begin = Clock::now();
            search(q, ids);
            latency[pass * query.n + q] = seconds(begin) * 1000;
        });
        wall_seconds += seconds(started);
        require(ids == reference, "result IDs changed between identical query passes");
    }
    std::cout << "\n# Search-only metrics; percentiles use all recorded per-query samples.\n"
              << "scope,retention,queries,k,recall,qps,mean_ms,p50_ms,p95_ms,p99_ms\n";
    auto print = [&](const std::string& scope, double rate, const std::vector<uint64_t>& qs) {
        std::vector<double> samples;
        double recall = 0;
        for (uint64_t q : qs) {
            recall += recalls[q];
            for (int pass = 0; pass < o.repeats; ++pass)
                samples.push_back(latency[pass * query.n + q]);
        }
        std::cout << std::fixed << std::setprecision(6) << scope << ',';
        if (scope == "group")
            std::cout << rate;
        std::cout << ',' << qs.size() << ',' << o.k << ',' << recall / qs.size() << ',';
        if (scope == "overall")
            std::cout << query.n * o.repeats / wall_seconds;
        std::cout << ',' << std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size()
                  << ',' << percentile(samples, 0.5) << ',' << percentile(samples, 0.95) << ','
                  << percentile(samples, 0.99) << '\n';
    };
    std::vector<uint64_t> all(query.n);
    std::iota(all.begin(), all.end(), 0);
    print("overall", 0, all);
    std::map<double, std::vector<uint64_t>> groups;
    for (uint64_t q = 0; q < query.n; ++q) groups[rates[q]].push_back(q);
    for (const auto& [rate, qs] : groups) print("group", rate, qs);
    std::cout << "measured_queries=" << query.n * o.repeats
              << " batch_wall_seconds=" << wall_seconds << std::endl;
}
}  // namespace

int
main(int argc, char** argv) try {
    const auto options = parse(argc, argv);
    if (options.mode == "prepare") {
        generate_hdf5(options);
        return 0;
    }
    omp_set_dynamic(0);
    vsag::init();
    vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
    vsag::Engine engine(&resource);
    run(engine, options);  // Index, filters, datasets and vector buffers die before Shutdown.
    engine.Shutdown();
    return 0;
} catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << std::endl;
    return 1;
}
