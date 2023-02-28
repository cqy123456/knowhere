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
const static std::string pq_pivots_path = "pq_pivots_path";
const static std::string pq_compressed_vectors_path = "pq_compressed_vectors_path";
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
               const std::string& metric) {
    HDF5_file::HDF5Reader data_reader(data_path);
    int32_t ndim, nb;
    void* data = nullptr;
    size_t m, ef_construction, sub_dim;
    m = 30;
    ef_construction = 360;
    sub_dim = 4;
    data_reader.GetBaseData(nb, ndim, data);

    // construct
    hnswlib::SpaceInterface<float>* space = nullptr;
    if (metric == "IP") {
        space = new hnswlib::L2Space(ndim);
    } else if (metric == "L2") {
        space = new (std::nothrow) hnswlib::InnerProductSpace(ndim);
    }
    auto index = new hnswlib::HierarchicalNSW<float>(space, nb, m, ef_construction);

    // add data
    float* tensor = static_cast<float*>(data);
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
    if (metric == "IP") {
        make_zero_mean = false;
    }
    auto pq_s = std::chrono::high_resolution_clock::now();
    const float* raw_data = static_cast<const float*>(tensor);
    write_bin_file<float>(data_file_to_use, raw_data, nb, ndim);

    gen_random_slice(raw_data, nb, ndim, p_val, train_data, train_size);

    generate_pq_pivots(train_data, train_size, (uint32_t)ndim, 256, (uint32_t)num_pq_chunks, diskann::NUM_KMEANS_REPS,
                       pq_path + pq_pivots_path, make_zero_mean);
    std::cout << "generate_pq_pivots" << std::endl;

    generate_pq_data_from_pivots<float>(data_file_to_use.c_str(), 256, (uint32_t)num_pq_chunks,
                                        pq_path + pq_pivots_path, pq_path + pq_compressed_vectors_path);
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
                const std::string& metric) {
    HDF5_file::HDF5Reader data_reader(data_path);
    // get query vector
    int32_t ndim, nq;
    void* query = nullptr;
    data_reader.GetQueryData(nq, ndim, query);
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
    std::string pq_compressed = pq_path + pq_compressed_vectors_path;
    std::string pq_pivot = pq_path + pq_pivots_path;
    index->loadIndex(index_path, space, pq_pivot, pq_compressed);
    bool transform = false;
    if (metric == "IP")
        transform = true;
    size_t k = 10;
    size_t ef = 30;
    hnswlib::SearchParam param{(size_t)ef};
    float* xq = static_cast<float*>(query);
    auto p_id = new int64_t[k * nq];
    auto p_dist = new float[k * nq];
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
    std::cout << "recall :" << CalcKNNRecall(gt_ids, p_id, nq, gt_k, k) << std::endl;
}

int
main() {
    std::string data_path = "/data/knowhere-2.0/knowhere/cqy-benchmark/tests/data/sift-128-euclidean.hdf5";
    std::string index_path = "/data/knowhere-2.0/knowhere/thirdparty/hnswlib/index/hnsw.bin";
    std::string pq_path = "/data/knowhere-2.0/knowhere/thirdparty/hnswlib/index/";
    std::string metric = "L2";
    // build_and_save(data_path, index_path, pq_path, metric);
    load_and_Search(data_path, index_path, pq_path, metric);
};