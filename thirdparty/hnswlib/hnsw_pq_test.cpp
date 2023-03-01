#include <boost/program_options.hpp>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <unordered_set>

#include "diskann/aux_utils.h"
#include "diskann/partition_and_pq.h"
#include "hdf5_operator.h"
#include "hnswlib/hnswalg.h"
#include "knowhere/comp/knowhere_config.h"
const static std::string graph_index_fname = "hnsw.bin";
const static std::string pq_pivots_fname = "pq_pivots_path";
const static std::string pq_compressed_vectors_fname = "pq_compressed_vectors_path";
namespace po = boost::program_options;
void
normalize(float* data, uint64_t npts, uint64_t ndims) {
#pragma omp parallel for schedule(dynamic)
    for (_s64 i = 0; i < (_s64)npts; i++) {
        float pt_norm = 0;
        for (_u32 d = 0; d < ndims; d++) pt_norm += data[i * ndims + d] * data[i * ndims + d];
        pt_norm = std::sqrt(pt_norm);
        for (_u32 d = 0; d < ndims; d++) data[i * ndims + d] = data[i * ndims + d] / pt_norm;
    }
}
template <typename T>
void
write_bin_file(const std::string& data_path, const T* data, const uint32_t row, const uint32_t col) {
    std::ofstream writer(data_path.c_str(), std::ios::binary);
    writer.write((char*)&row, sizeof(uint32_t));
    writer.write((char*)&col, sizeof(uint32_t));
    writer.write((char*)data, sizeof(T) * row * col);
    writer.close();
    return;
}
inline float
CalcKNNRecall(const int32_t* gt_ids, const int64_t* ids, int64_t nq, int32_t gt_k, int32_t k) {
    int64_t min_k = std::min(gt_k, k);
    int64_t hit = 0;
    for (auto i = 0; i < nq; i++) {
        std::unordered_set<int32_t> ground(gt_ids + i * gt_k, gt_ids + i * gt_k + min_k);
        for (auto j = 0; j < min_k; j++) {
            auto id = ids[i * k + j];
            if (ground.count(id) > 0) {
                hit++;
            }
        }
    }
    return (hit * 1.0f / (nq * min_k)) * 100;
}
void
build_and_save(const std::string& data_path, const std::string& index_path, const std::string& pq_path,
               const std::string& metric, const int sub_dim) {
    HDF5_file::HDF5Reader data_reader(data_path);
    int32_t ndim, nb;
    void* data = nullptr;
    size_t m, ef_construction;
    m = 30;
    ef_construction = 360;

    data_reader.GetBaseData(nb, ndim, data);
    float* tensor = static_cast<float*>(data);
    if (metric == "IP") {
        normalize(tensor, uint64_t(nb), uint64_t(ndim));
    }

    // construct
    hnswlib::SpaceInterface<float>* space = nullptr;
    if (metric == "L2") {
        space = new hnswlib::L2Space(ndim);
    } else if (metric == "IP") {
        space = new (std::nothrow) hnswlib::InnerProductSpace(ndim);
    }
    auto index = new hnswlib::HierarchicalNSW<float>(space, nb, m, ef_construction);

    // add data

    index->addPoint(tensor, 0);

#pragma omp parallel for
    for (int i = 1; i < nb; ++i) {
        index->addPoint((static_cast<const float*>(tensor) + ndim * i), i);
    }

    // construct pq
    size_t train_size;
    float* train_data = nullptr;
    double p_val = ((double)diskann::MAX_PQ_TRAINING_SET_SIZE / (double)nb);
    size_t num_pq_chunks = std::ceil(ndim * 1.0 / sub_dim);
    bool make_zero_mean = true;
    std::string data_file_to_use = "raw_data.bin";
    // if (metric == "IP") {
    //     make_zero_mean = false;
    // }
    auto pq_s = std::chrono::high_resolution_clock::now();
    const float* raw_data = static_cast<const float*>(tensor);
    write_bin_file<float>(data_file_to_use, raw_data, nb, ndim);

    gen_random_slice(raw_data, nb, ndim, p_val, train_data, train_size);

    generate_pq_pivots(train_data, train_size, (uint32_t)ndim, 256, (uint32_t)num_pq_chunks, diskann::NUM_KMEANS_REPS,
                       pq_path + pq_pivots_fname, make_zero_mean);
    std::cout << "generate_pq_pivots" << std::endl;

    generate_pq_data_from_pivots<float>(data_file_to_use.c_str(), 256, (uint32_t)num_pq_chunks,
                                        pq_path + pq_pivots_fname, pq_path + pq_compressed_vectors_fname);
    std::cout << "generate_pq_data_from_pivots" << std::endl;
    auto pq_e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> pq_diff = pq_e - pq_s;
    std::cout << "Training PQ codes cost: " << pq_diff.count() << "s";

    // save index
    index->saveIndex(index_path);
    if (train_data != nullptr) {
        delete[] train_data;
    }
    if (data != nullptr) {
        delete[] (char*)data;
    }
}

void
load_and_Search(const std::string& data_path, const std::string& index_path, const std::string& pq_path,
                const std::string& metric, const int ef) {
    HDF5_file::HDF5Reader data_reader(data_path);
    // get query vector
    int32_t ndim, nq;
    void* query = nullptr;
    data_reader.GetQueryData(nq, ndim, query);
    float* xq = static_cast<float*>(query);
    if (metric == "IP") {
        normalize(xq, uint64_t(nq), uint64_t(ndim));
    }
    // get gt
    int32_t gt_k, nq_tmp;
    int32_t* gt_ids = nullptr;
    float* gt_dis = nullptr;
    data_reader.GetKNNGT(gt_k, nq_tmp, gt_ids, gt_dis);

    hnswlib::SpaceInterface<float>* space = nullptr;

    if (metric == "L2") {
        space = new hnswlib::L2Space(ndim);
    } else if (metric == "IP") {
        space = new hnswlib::InnerProductSpace(ndim);
    }

    auto index = new (std::nothrow) hnswlib::HierarchicalNSW<float>(space);
    std::string pq_compressed = pq_path + pq_compressed_vectors_fname;
    std::string pq_pivot = pq_path + pq_pivots_fname;
    index->loadIndex(index_path, space, pq_pivot, pq_compressed);
    bool transform = false;
    if (metric == "IP")
        transform = true;
    size_t k = 10;

    hnswlib::SearchParam param{(size_t)ef, 1.0};

    auto p_id = new int64_t[k * nq];
    auto p_dist = new float[k * nq];
    auto search_s = std::chrono::high_resolution_clock::now();
    // #pragma omp parallel for
    for (int i = 0; i < nq; ++i) {
        auto single_query = xq + i * ndim;
        auto rst = index->searchKnn(single_query, k, nullptr, &param, nullptr);
        size_t rst_size = rst.size();
        auto p_single_dis = p_dist + i * k;
        auto p_single_id = p_id + i * k;
        for (size_t idx = 0; idx < rst_size; ++idx) {
            const auto& [dist, id] = rst[idx];
            p_single_dis[idx] = transform ? (1 - dist) : dist;
            p_single_id[idx] = id;
        }
        for (size_t idx = rst_size; idx < (size_t)k; idx++) {
            p_single_dis[idx] = float(1.0 / 0.0);
            p_single_id[idx] = -1;
        }
    }
    auto search_e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = search_e - search_s;
    std::cout << "PQS: " << nq / diff.count();

    std::cout << "recall :" << CalcKNNRecall(gt_ids, p_id, nq, gt_k, k) << std::endl;
}

int
main(int argc, char** argv) {
    knowhere::KnowhereConfig::SetSimdType(knowhere::KnowhereConfig::SimdType::SSE4_2);
    // std::string data_path = "/data/knowhere-2.0/knowhere/cqy-benchmark/tests/data/gist-960-euclidean.hdf5";
    // std::string index_path = "/data/knowhere-2.0/knowhere/thirdparty/hnswlib/index/hnsw.bin";
    // std::string pq_path = "/data/knowhere-2.0/knowhere/thirdparty/hnswlib/index/";
    std::string data_path;
    std::string path;
    std::string index_path;
    std::string pq_path;
    std::string metric;
    std::string fun;
    int sub_dim, ef;
    po::options_description desc{"Arguments"};
    try {
        desc.add_options()("help,h", "Print information on arguments.");
        desc.add_options()("fun, -f", po::value<std::string>(&fun)->required(), "Runner type: [build, search].");
        desc.add_options()("data, -d", po::value<std::string>(&data_path)->required(), "data_path.");
        desc.add_options()("index, -i", po::value<std::string>(&path)->required(), "index_path.");
        desc.add_options()("metric, -m", po::value<std::string>(&metric)->required(), "metric, L2 or IP.");
        desc.add_options()("sub_dim, -s", po::value<int>(&sub_dim)->default_value(2), "sub dim.");

        desc.add_options()("ef, -e", po::value<int>(&ef)->default_value(10), "ef.");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    index_path = path + graph_index_fname;
    pq_path = path;
    if (fun == "build") {
        build_and_save(data_path, index_path, pq_path, metric, sub_dim);
    } else {
        load_and_Search(data_path, index_path, pq_path, metric, ef);
    }
};