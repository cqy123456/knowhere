#include <omp.h>
#include <sys/time.h>

#include <cassert>
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "faiss/IndexIVFPQFastScan.h"
#include "faiss/IndexRefine.h"
#include "faiss/index_factory.h"
#include "faiss/index_io.h"
#include "faiss_hdf5_handler.h"
#include "util/evaluate.h"

using namespace std::chrono;
namespace scann_test {
class Timer {
 public:
    typedef high_resolution_clock Clock;
    Timer() {
        start();
    }
    void
    start() {
        epoch = Clock::now();
    }
    double
    time_elapsed_ms() const {
        auto cost = Clock::now() - epoch;
        return duration_cast<microseconds>(cost).count() / 1000;
    }
    double
    time_elapsed_s() const {
        auto cost = Clock::now() - epoch;
        return duration_cast<seconds>(cost).count();
    }

 private:
    Clock::time_point epoch;
};

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
}  // namespace scann_test

void
build_and_save(const std::string& fname, const std::string& index_file) {
    scann_test::HDF5Reader data_reader(fname);
    int32_t ndim, nb;
    void* data = nullptr;
    data_reader.GetBaseData(nb, ndim, data);
    std::cout << "ndim, nb: " << ndim << " " << nb << std::endl;

    int nlist = 2000;
    int subdim = 4;
    int m = ndim / subdim;
    std::stringstream namestream;
    std::string index_name;
    namestream << "IVF" << nlist << ",";
    namestream << "PQ" << m << "x4fs";
    namestream >> index_name;
    std::cout << index_name << std::endl;

    // build index
    faiss::Index* index;
    index = faiss::index_factory(ndim, index_name.c_str());
    // if metric == l2, compute_residual = false;
    // if metric == ip, compute_residual = true
    auto fs_index = dynamic_cast<faiss::IndexIVFPQFastScan*>(index);
    fs_index->by_residual = false;
    scann_test::Timer time;
    float train_time, add_time;
    time.start();
    fs_index->train(nb, static_cast<float*>(data));
    train_time = time.time_elapsed_s();
    std::cout << "train cost : " << train_time << std::endl;
    time.start();
    fs_index->add(nb, static_cast<float*>(data));
    add_time = time.time_elapsed_s();
    std::cout << "add cost: " << add_time << std::endl;
    std::cout << "build time: " << train_time + add_time << std::endl;

    faiss::write_index(fs_index, index_file.c_str());
    if (data != nullptr) {
        delete[] (char*)data;
    }
    return;
}

void
load_and_search(const std::string& data_file, const std::string& index_file) {
    scann_test::HDF5Reader data_reader(data_file);
    int32_t ndim, nb;
    void* data = nullptr;
    data_reader.GetBaseData(nb, ndim, data);
    auto start_mem = benchmark::getCurrentRSS();
    auto index = dynamic_cast<faiss::IndexIVFPQFastScan*>(faiss::read_index(index_file.c_str()));
    auto load_mem_usage = benchmark::getCurrentRSS() - start_mem;
    std::cout << "load ivf fast scann, index, mem usage:(MB) " << load_mem_usage << std::endl;

    auto refine_index = faiss::IndexRefineFlat(index, static_cast<float*>(data));

    load_mem_usage = benchmark::getCurrentRSS() - start_mem;

    std::cout << "load refine index, mem usage:(MB) " << load_mem_usage << std::endl;
    // get query vector
    int32_t ndim1, nq;
    void* query = nullptr;
    data_reader.GetQueryData(nq, ndim1, query);
    assert(ndim1 == ndim);
    // get gt
    int32_t k, nq_tmp;
    int32_t* gt_ids = nullptr;
    float* gt_dis = nullptr;
    data_reader.GetKNNGT(k, nq_tmp, gt_ids, gt_dis);

    int search_k = 10;
    auto nprobe_list = std::vector<int>({10, 20, 24, 32, 40, 48, 56, 64, 72, 81, 90, 100});
    auto factor_list = std::vector<float>({5, 10, 20, 30, 40, 50, 60, 70, 80});
    auto ids = new int64_t[nq * search_k];
    auto dis = new float[nq * search_k];
    auto q = static_cast<float*>(query);

    float search_time;
    start_mem = benchmark::getCurrentRSS();
    std::cout << "before search index" << start_mem << std::endl;
    for (auto& k_factor : factor_list) {
        for (auto& nprobe : nprobe_list) {
            refine_index.k_factor = k_factor;
            index->nprobe = nprobe;
            scann_test::Timer time;
            time.start();
            refine_index.search(nq, q, search_k, dis, ids);
            search_time = time.time_elapsed_ms();
            auto mem_usage = benchmark::getPeakRSS() - start_mem;
            std::cout << "peak mem " << benchmark::getPeakRSS() << std::endl;
            // get recall
            std::cout << "nprobe: " << nprobe << " k_factor: " << k_factor << std::endl;

            std::cout << "recall :" << scann_test::CalcKNNRecall(gt_ids, ids, nq, k, search_k)
                      << " qps: " << nq / (search_time / 1000.0) << " extra mem use" << mem_usage << std::endl;
        }
    }
    if (ids != nullptr) {
        delete[] ids;
    }
    if (dis != nullptr) {
        delete[] dis;
    }
    if (gt_ids != nullptr) {
        delete[] gt_ids;
    }
    if (data != nullptr) {
        delete[] (char*)data;
    }
}
int
main() {
    omp_set_num_threads(12);
    const std::string file_name = "/data/knowhere-2.0/knowhere/cqy-benchmark/tests/data/sift-128-euclidean.hdf5";
    const std::string index_file = "fast_scann_sift1m.bin";
    // build_and_save(file_name, index_file);
    load_and_search(file_name, index_file);

    return 0;
}