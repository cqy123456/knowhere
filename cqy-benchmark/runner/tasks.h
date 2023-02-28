#include <knowhere/comp/local_file_manager.h>

#include <algorithm>
#include <cstdio>
#include <iomanip>
#include <random>
#include <sstream>
#include <thread>

#include "cqy-benchmark/file_handler/bin_file_handler.h"
#include "cqy-benchmark/file_handler/hdf5_handler.h"
#include "cqy-benchmark/file_handler/json_handler.h"
#include "faiss/IndexIVFPQFastScan.h"
#include "faiss/IndexRefine.h"
#include "faiss/index_factory.h"
#include "faiss/index_io.h"
#include "knowhere/dataset.h"
#include "knowhere/factory.h"
#include "knowhere/object.h"
#include "tests/ut/utils.h"
#include "util/evaluate.h"
#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
error "Missing the <filesystem> header."
#endif
#include <fstream>

namespace benchmark {
namespace {
constexpr int kSerialSearchRound = 100;
constexpr int kThreadMaxQueryNum = 10000;
constexpr int kThreadTotalQueryNum = 300;
inline bool
IsBinaryType(const std::string& index_type) {
    return index_type.find("BIN") != std::string::npos;
}
template <bool is_binary>
std::unique_ptr<knowhere::DataSet>
GenRandomDataSet(int rows, int dim, int seed = 42) {
    std::mt19937 rng(seed);

    auto ds = std::make_unique<knowhere::DataSet>();
    ds->SetRows(rows);
    ds->SetDim(dim);
    if (!is_binary) {
        std::uniform_real_distribution<> distrib(0.0, 100.0);
        float* ts = new float[rows * dim];
        for (int i = 0; i < rows * dim; ++i) ts[i] = distrib(rng);
        ds->SetTensor(ts);
    } else {
        std::uniform_int_distribution<> distrib(0, 1);
        uint8_t* ts = new uint8_t[rows * dim];
        for (int i = 0; i < rows * dim; ++i) ts[i] = distrib(rng);
        ds->SetTensor(ts);
    }
    return ds;
}
knowhere::Index<knowhere::IndexNode>
LoadIndex(const std::string& index_type, const std::string& index_path, const std::string& data_path,
          const std::string& metric_type) {
    HDF5Reader data_reader(data_path);
    std::shared_ptr<knowhere::FileManager> file_manager = std::make_shared<knowhere::LocalFileManager>();
    auto diskann_index_pack = knowhere::Pack(file_manager);
    knowhere::Index<knowhere::IndexNode> idx =
        knowhere::IndexFactory::Instance().Create(index_type, diskann_index_pack);
    if (index_type == "DISKANN") {
        int32_t ndim, nb;
        data_reader.GetDataShape<false>(ndim, nb);

        auto prepare_json = ReadIndexParams(index_type, PARAMS_JSON::LOAD_JSON);
        prepare_json["index_prefix"] = index_path;
        prepare_json["search_cache_budget_gb"] =
            prepare_json["search_cache_budget_gb_rate"].get<float>() * sizeof(float) * ndim * nb / 1024 / 1024 / 1024;

        knowhere::DataSetPtr ds = data_reader.GetQueryData<false>();
        idx.Search(*ds, prepare_json, nullptr);

    } else if (index_type == knowhere::IndexEnum::INDEX_FAISS_IVFFLAT) {
        knowhere::BinarySet binset;
        read_index_file(index_path, binset);
        auto base = data_reader.GetBaseData<false>();
        auto data_ptr = (float*)base->GetTensor();
        knowhere::BinaryPtr bptr = std::make_shared<knowhere::Binary>();
        bptr->data = std::shared_ptr<uint8_t[]>((uint8_t*)data_ptr, [&](uint8_t*) {});
        bptr->size = base->GetDim() * base->GetRows() * sizeof(float);
        binset.Append("RAW_DATA", bptr);
        idx.Deserialize(binset);
        binset.clear();
    } else {
        knowhere::BinarySet binset;
        read_index_file(index_path, binset);
        idx.Deserialize(binset);
    }
    return idx;
};

struct ThreadMeta {
    knowhere::Index<knowhere::IndexNode>& index;
    double& latancy;
    knowhere::Json search_json;
    uint32_t nq;
    uint32_t ndim;
    uint32_t thread_id;
};

void
ThreadTask(ThreadMeta thread_meta) {
    bool is_binary = IsBinaryType(thread_meta.index.Type());

    auto total_qs = is_binary ? GenRandomDataSet<true>(kThreadMaxQueryNum, thread_meta.ndim, thread_meta.thread_id)
                              : GenRandomDataSet<false>(kThreadMaxQueryNum, thread_meta.ndim, thread_meta.thread_id);
    uint32_t round = 100;
    auto qs = std::make_shared<knowhere::DataSet>();
    qs->SetDim(thread_meta.ndim);
    qs->SetRows(thread_meta.nq);
    qs->SetIsOwner(false);

    double time = 0;
    Timer timer;

    for (auto i = 0; i < round; i++) {
        if (is_binary) {
            for (auto i = 0; i < round; i++) {
                qs->SetTensor((uint8_t*)total_qs->GetTensor() +
                              (i * thread_meta.nq) % kThreadMaxQueryNum * thread_meta.ndim);
                timer.start();
                thread_meta.index.Search(*qs, thread_meta.search_json, nullptr);
                time += timer.time_elapsed_ms();
            }
        } else {
            for (auto i = 0; i < round; i++) {
                qs->SetTensor((float*)total_qs->GetTensor() +
                              (i * thread_meta.nq) % kThreadMaxQueryNum * thread_meta.ndim);
                timer.start();
                thread_meta.index.Search(*qs, thread_meta.search_json, nullptr);
                time += timer.time_elapsed_ms();
            }
        }
    }

    thread_meta.latancy = time / (double)round;
}

void
ParallelSearchTask(knowhere::Index<knowhere::IndexNode>& index, const std::string& data_path,
                   const std::string& index_path, const std::string& metric_type, uint64_t nq, uint64_t client_num,
                   std::ofstream& result_out) {
    result_out << "[Parallel_Search]" << std::endl;
    HDF5Reader data_reader(data_path);
    int32_t ndim, nrows;
    if (IsBinaryType(index.Type())) {
        data_reader.GetDataShape<true>(ndim, nrows);
    } else {
        data_reader.GetDataShape<false>(ndim, nrows);
    }
    auto search_json = ReadIndexParams(index.Type(), PARAMS_JSON::PERF_JSON);
    auto k = search_json["k"].get<int>();
    search_json["dim"] = ndim;
    search_json["metric_type"] = metric_type;
    search_json["index_prefix"] = index_path;

    std::vector<double> latancy(client_num);
    std::vector<std::thread> threads;
    for (auto t_id = 0; t_id < client_num; t_id++) {
        knowhere::Json thread_json = search_json;
        ThreadMeta threadmeta{index,        std::ref(latancy[t_id]), search_json,
                              (uint32_t)nq, (uint32_t)index.Dim(),   (uint32_t)t_id};
        threads.push_back(std::thread(ThreadTask, threadmeta));
    }
    for (auto& thread : threads) {
        thread.join();
    }
    auto aver_and_tp99 = GetAverageAndTP99Latancy(latancy);
    result_out << std::setw(6) << "NQ" << std::setw(12) << "TOPK" << std::setw(30) << "average latancy:(ms)"
               << std::setw(16) << "tp99(ms)" << std::endl;
    result_out << std::setw(6) << nq << std::setw(12) << k << std::setw(30) << aver_and_tp99.first << std::setw(16)
               << aver_and_tp99.second << std::endl;
}

void
SerialSearchTask(knowhere::Index<knowhere::IndexNode>& index, const std::string& data_path,
                 const std::string& index_path, const std::string& metric_type, uint64_t nq,
                 std::ofstream& result_out) {
    result_out << "[Serial_Search] " << std::endl;
    HDF5Reader data_reader(data_path);
    bool is_binary = IsBinaryType(index.Type());
    auto total_qs = is_binary ? data_reader.GetQueryData<true>() : data_reader.GetQueryData<false>();
    auto search_json = ReadIndexParams(index.Type(), PARAMS_JSON::PERF_JSON);
    auto dim = total_qs->GetDim();
    auto rows = total_qs->GetRows();
    auto k = search_json["k"].get<int>();
    search_json["dim"] = dim;
    search_json["metric_type"] = metric_type;
    search_json["index_prefix"] = index_path;
    auto qs = std::make_shared<knowhere::DataSet>();
    qs->SetDim((int64_t)dim);
    qs->SetRows((int64_t)nq);
    qs->SetIsOwner(false);
    std::vector<double> latancy(kSerialSearchRound);

    Timer timer;
    if (is_binary) {
        for (auto i = 0; i < kSerialSearchRound; i++) {
            qs->SetTensor((uint8_t*)total_qs->GetTensor() + (i * nq) % rows * (dim / 8));
            timer.start();
            index.Search(*qs, search_json, nullptr);
            latancy[i] = timer.time_elapsed_ms();
        }
    } else {
        for (auto i = 0; i < kSerialSearchRound; i++) {
            qs->SetTensor((float*)total_qs->GetTensor() + (i * nq) % rows);
            timer.start();
            index.Search(*qs, search_json, nullptr);
            latancy[i] = timer.time_elapsed_ms();
        }
    }
    auto aver_and_tp99 = GetAverageAndTP99Latancy(latancy);
    result_out << std::setw(6) << "NQ" << std::setw(12) << "TOPK" << std::setw(30) << "average latancy:(ms)"
               << std::setw(16) << "tp99(ms)" << std::endl;
    result_out << std::setw(6) << nq << std::setw(12) << k << std::setw(30) << aver_and_tp99.first << std::setw(16)
               << aver_and_tp99.second << std::endl;
}
};  // namespace
void
BuildIndexTask(const std::string& index_type, const std::string& data_path, const std::string& index_path,
               const std::string& metric_type, std::ofstream& result_out) {
    result_out << "==============================================================="
                  "======================================================="
               << std::endl;
    result_out << "Build_Performance " << std::endl;
    result_out << "==============================================================="
                  "======================================================="
               << std::endl;
    size_t start_mem, build_mem_usage;
    start_mem = getCurrentRSS();

    knowhere::Json build_json = ReadIndexParams(index_type, PARAMS_JSON::BUILD_JSON);
    HDF5Reader data_reader(data_path);
    Timer timer;
    size_t peak_mem = 0;
    std::shared_ptr<knowhere::FileManager> file_manager = std::make_shared<knowhere::LocalFileManager>();
    auto diskann_index_pack = knowhere::Pack(file_manager);
    knowhere::Index<knowhere::IndexNode> idx =
        knowhere::IndexFactory::Instance().Create(index_type, diskann_index_pack);
    if (index_type == "DISKANN") {
        // prepare data
        std::string raw_data_path = index_path + "raw_data.fbin";
        data_reader.SaveBaseBinFile(raw_data_path);
        // prepare config
        uint32_t ndim, nb;
        read_bin_meta(raw_data_path, nb, ndim);
        build_json["data_path"] = raw_data_path;
        build_json["index_prefix"] = index_path + "/";
        build_json["dim"] = (int)ndim;
        build_json["metric_type"] = metric_type;
        build_json["pq_code_budget_gb"] =
            build_json["pq_code_budget_gb_rate"].get<float>() * sizeof(float) * ndim * nb / 1024 / 1024 / 1024;

        knowhere::DataSetPtr ds = nullptr;
        idx.Build(*ds, build_json);
        build_mem_usage = getPeakRSS() - start_mem;
        std::remove(raw_data_path.c_str());
    } else if (index_type == "FASTSCANN") {
        auto ds = IsBinaryType(index_type) ? data_reader.GetBaseData<true>() : data_reader.GetBaseData<false>();
        auto base = (float*)ds->GetTensor();
        std::string metric;
        if (metric_type == knowhere::metric::IP) {
            metric = faiss::METRIC_INNER_PRODUCT;
        } else {
            metric = faiss::METRIC_L2;
        }
        auto nlist = build_json["nlist"].get<int>();
        auto m = std::ceil((int)ds->GetDim() / build_json["sub_dim"].get<int>());
        auto by_residual = build_json["by_residual"].get<bool>();
        std::stringstream namestream;
        std::string index_name;
        namestream << "IVF" << nlist << ",";
        namestream << "PQ" << m << "x4fs";
        namestream >> index_name;
        std::cout << "building " << index_name << std::endl;

        faiss::Index* index;
        index = faiss::index_factory((int)ds->GetDim(), index_name.c_str());
        auto fs_index = dynamic_cast<faiss::IndexIVFPQFastScan*>(index);
        fs_index->by_residual = by_residual;
        fs_index->train(ds->GetRows(), base);
        fs_index->add(ds->GetRows(), base);
        build_mem_usage = getPeakRSS() - start_mem;
        faiss::write_index(fs_index, index_path.c_str());
    } else {
        // create memory index
        start_mem = getCurrentRSS();
        auto ds = IsBinaryType(index_type) ? data_reader.GetBaseData<true>() : data_reader.GetBaseData<false>();
        build_json["dim"] = (int)ds->GetDim();
        build_json["metric_type"] = metric_type;
        knowhere::Json json = knowhere::Json::parse(build_json.dump());
        idx.Build(*ds, json);

        build_mem_usage = getPeakRSS() - start_mem;

        knowhere::BinarySet bs;
        idx.Serialize(bs);
        write_index_file(index_path, bs);
    }
    auto build_time = timer.time_elapsed_s();
    result_out << "build time:(s) " << build_time << std::endl;
    result_out << "mem usage:(MB) " << build_mem_usage << std::endl;
}

// void
// FastScannSearchAccuracyTask((const std::string& index_type, const std::string& data_path, const std::string&
// index_path,
//                    const std::string& metric_type, std::ofstream& result_out) {
//     result_out << "==============================================================="
//                   "======================================================="
//                << std::endl;
//     result_out << "SearchAccuracyTask " << std::endl;
//     result_out << "==============================================================="
//                   "======================================================="
//                << std::endl;
//     HDF5Reader data_reader(data_path);
//     auto start_mem = getCurrentRSS();
//     // load raw data and construct a refine ivf_pq_scann index
//     auto ds = IsBinaryType(index_type) ? data_reader.GetBaseData<true>() : data_reader.GetBaseData<false>();
//     auto data = static_cast<float*>(ds->GetTensor());
//     auto index = dynamic_cast<faiss::IndexIVFPQFastScan*>(faiss::read_index(index_path.c_str()));
//     auto refine_index = faiss::IndexRefineFlat(index, static_cast<float*>(data));
//     auto load_mem_usage = getCurrentRSS() - start_mem;
//     result_out << "After loading the index, memory usage(MB): " << load_mem_usage << std::endl;

//     // load query and gt
//     auto qs = IsBinaryType(index_type) ? data_reader.GetQueryData<true>() : data_reader.GetQueryData<false>();
//     auto gt = data_reader.GetKNNGT();

//     auto search_json_list = ReadIndexParams(index_type, PARAMS_JSON::ACC_JSON);

//     for (knowhere::Json::iterator it = search_json_list.begin(); it != search_json_list.end(); ++it) {
//         auto search_json = *it;
//         result_out << "Search Parameters: " << search_json.dump() << std::endl;

//         auto ids = new int64_t[nq * search_k];
//         auto dis = new float[nq * search_k];
//     }
// }

void
SearchAccuracyTask(const std::string& index_type, const std::string& data_path, const std::string& index_path,
                   const std::string& metric_type, std::ofstream& result_out) {
    result_out << "==============================================================="
                  "======================================================="
               << std::endl;
    result_out << "SearchAccuracyTask " << std::endl;
    result_out << "==============================================================="
                  "======================================================="
               << std::endl;
    // load index from index file
    auto start_mem = getCurrentRSS();
    auto idx = LoadIndex(index_type, index_path, data_path, metric_type);
    auto load_mem_usage = getCurrentRSS() - start_mem;
    result_out << "After loading the index, memory usage(MB): " << load_mem_usage << std::endl;
    HDF5Reader data_reader(data_path);
    auto qs = IsBinaryType(index_type) ? data_reader.GetQueryData<true>() : data_reader.GetQueryData<false>();
    auto gt = data_reader.GetKNNGT();

    auto search_json_list = ReadIndexParams(index_type, PARAMS_JSON::ACC_JSON);

    for (knowhere::Json::iterator it = search_json_list.begin(); it != search_json_list.end(); ++it) {
        auto search_json = *it;
        result_out << "Search Parameters: " << search_json.dump() << std::endl;
        search_json["index_prefix"] = index_path;
        search_json["metric_type"] = metric_type;
        search_json["dim"] = (int)qs->GetDim();

        Timer timer;
        auto search_stat = idx.Search(*qs, search_json, nullptr);
        assert(search_stat.has_value());

        auto res = search_stat.value();
        auto latancy = timer.time_elapsed_ms();
        auto nq = qs->GetRows();
        auto recall = CalcKNNRecall(gt->GetIds(), res->GetIds(), nq, gt->GetDim(), search_json["k"]);
        auto qps = nq / latancy * 1000;
        result_out << std::setw(6) << "recall" << std::setw(20) << "latancy of nq:" << nq << "(ms)" << std::setw(16)
                   << "qps" << std::endl;
        result_out << std::setw(6) << recall << std::setw(20) << latancy << std::setw(16) << qps << std::endl;
    }
}

void
SearchPerformanceTask(const std::string& index_type, const std::string& data_path, const std::string& index_path,
                      const std::string& metric_type, std::ofstream& result_out) {
    if (index_type == "FASTSCANN") {
        assert(false && "fastscann not support performance test");
    }
    result_out << "==============================================================="
                  "======================================================="
               << std::endl;
    result_out << "SearchPerformance" << std::endl;
    result_out << "==============================================================="
                  "======================================================="
               << std::endl;
    auto start_mem = getCurrentRSS();
    auto index = LoadIndex(index_type, index_path, data_path, metric_type);
    auto load_mem_usage = getCurrentRSS() - start_mem;
    result_out << "Memory usage after loading the index(MB): " << load_mem_usage << std::endl;
    auto search_json = ReadTaskCfg(PARAMS_JSON::PERF_JSON);
    auto& nq_list = search_json.at("query_num");
    for (auto i = 0; i < nq_list.size(); i++) {
        int nq = nq_list.at(i).get<int>();
        SerialSearchTask(index, data_path, index_path, metric_type, nq, result_out);
        ParallelSearchTask(index, data_path, index_path, metric_type, nq, search_json.at("client_num").get<int>(),
                           result_out);
    }
}
};  // namespace benchmark