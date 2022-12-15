#include <algorithm>
#include <cstdio>
#include <thread>
#include "knowhere/index/IndexType.h"
#include "knowhere/index/VecIndexFactory.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"
#include "knowhere/index/vector_index/IndexDiskANN.h"
#include "knowhere/index/vector_index/IndexDiskANNConfig.h"
#include "cqy-benchmark/util/evaluate.h"
#include "cqy-benchmark/file_handler/bin_file_handler.h"
#include "cqy-benchmark/file_handler/hdf5_handler.h"
#include "cqy-benchmark/file_handler/json_handler.h"
#include "unittest/LocalFileManager.h"
#include "common/Dataset.h"
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
    constexpr int kSerialSearchRound = 1000;
    constexpr int kThreadMaxQueryNum = 10000;
    inline bool IsBinaryType(const std::string& index_type) {
        return index_type.find("bin") != std::string::npos;
    }
    inline knowhere::DatasetPtr
    GenRandomDataSet(int rows, int dim, int seed = 42) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<> distrib(0.0, 100.0);

        auto ds = std::make_shared<knowhere::Dataset>();
        knowhere::SetDatasetRows(ds, (int64_t)rows);
        knowhere::SetDatasetDim(ds, (int64_t)dim);
        
        float* ts = new float[rows * dim];
        for (int i = 0; i < rows * dim; ++i) ts[i] = distrib(rng);
        knowhere::SetDatasetTensor(ds, ts);
        return ds;
    }
    knowhere::VecIndexPtr
    LoadIndex(const std::string& index_type, const std::string& index_path, const std::string& data_path, const std::string& metric_type) {
        HDF5Reader data_reader(data_path);
        knowhere::VecIndexPtr idx = nullptr;
        if (index_type == knowhere::IndexEnum::INDEX_DISKANN) {
            std::shared_ptr<knowhere::FileManager> file_manager = std::make_shared<knowhere::LocalFileManager>();
            idx = std::make_unique<knowhere::IndexDiskANN<float>>(index_path, metric_type,
                                                                   std::make_unique<knowhere::LocalFileManager>());
            int32_t ndim, nb;
            data_reader.GetDataShape<false>(ndim, nb);
            auto prepare_json = ReadIndexParams(index_type, PARAMS_JSON::LOAD_JSON);
            prepare_json["index_prefix"] = index_path;
            prepare_json["search_cache_budget_gb"] =  prepare_json["search_cache_budget_gb_rate"].get<float>() * sizeof(float) * ndim * nb / 1024 / 1024 / 1024;
            idx->Prepare(prepare_json);
        } else if (index_type == knowhere::IndexEnum::INDEX_FAISS_IVFFLAT) {
            auto idx = knowhere::VecIndexFactory::GetInstance().CreateVecIndex(index_type, knowhere::IndexMode::MODE_CPU);
            knowhere::BinarySet binset;
            read_index_file(index_path, binset); 
            auto base = data_reader.GetBaseData<false>();
            knowhere::BinaryPtr bptr = std::make_shared<knowhere::Binary>();
            bptr->data = std::shared_ptr<uint8_t[]>((uint8_t*)knowhere::GetDatasetTensor(base), [&](uint8_t*) {});
            bptr->size = knowhere::GetDatasetDim(base) *knowhere::GetDatasetRows(base) * sizeof(float);
            binset.Append("RAW_DATA", bptr);
            idx->Load(binset);
        }  else  {
            auto idx = knowhere::VecIndexFactory::GetInstance().CreateVecIndex(index_type, knowhere::IndexMode::MODE_CPU);
            knowhere::BinarySet binset;
            read_index_file(index_path, binset); 
            idx->Load(binset);
        }
       return idx;
    };
    
    struct ThreadMeta {
        const knowhere::VecIndexPtr index;
        double& latancy;
        knowhere::Config search_json;
        uint32_t nq;
        uint32_t ndim;
        uint32_t thread_id;
    };
    
    void 
   ThreadTask(ThreadMeta thread_meta) {
        auto total_qs = GenRandomDataSet(kThreadMaxQueryNum, thread_meta.ndim, thread_meta.thread_id);
        uint32_t query_num = 900;
        uint32_t round = query_num / thread_meta.nq;
        auto qs = std::make_shared<knowhere::Dataset>();
        knowhere::SetDatasetDim(qs, thread_meta.ndim);
        knowhere::SetDatasetRows(qs, thread_meta.nq);
        double time;
        Timer timer;
        auto query_beg = (float*)knowhere::GetDatasetTensor(total_qs);
        for (auto i = 0; i < round; i++) {
            knowhere::SetDatasetTensor(qs, (query_beg + (i * thread_meta.nq) % kThreadMaxQueryNum * thread_meta.ndim));
            timer.start();
            thread_meta.index->Query(qs, thread_meta.search_json, nullptr);
            time += timer.time_elapsed_ms();
        }
        thread_meta.latancy = time / (double)round;
   }

   void 
   ParallelSearchTask(const knowhere::VecIndexPtr& index, const std::string& index_path, const std::string& metric_type, uint64_t nq, uint64_t client_num, std::ofstream& result_out) {
        result_out << "[ParallelSearchTask]:"<< std::endl;
        auto search_json = ReadIndexParams(metric_type, PARAMS_JSON::PERF_JSON);
        search_json["dim"] = index->Dim();
        search_json["metric_type"] = metric_type;
        search_json["index_prefix"] = index_path;
        std::vector<double> latancy(client_num);
        std::vector<std::thread> threads;
        for(auto t_id = 0; t_id < client_num; t_id++) {
            knowhere::Config thread_json = search_json;
            ThreadMeta threadmeta{index, std::ref(latancy[t_id]), search_json, (uint32_t)nq, (uint32_t)index->Dim(), (uint32_t)t_id};
            threads.push_back(std::thread(ThreadTask, threadmeta) );
        }
        for(auto& thread : threads) {
            thread.join();
        }
        auto aver_and_tp99 = GetAverageAndTP99Latancy(latancy.data(), client_num); 
        result_out << "[PARAMS]:"<<search_json.dump()<<std::endl;
        result_out <<"[RESULT]: nq: "<<nq<<", average latancy:(ms) "<<aver_and_tp99.first<<", tp99: "<<aver_and_tp99.second<<std::endl;
   }

    void 
    SerialSearchTask(const knowhere::VecIndexPtr& index, const std::string& data_path, const std::string& index_path, const std::string& metric_type, uint64_t nq, std::ofstream& result_out) {
        result_out << "[SerialSearchTask]: "<< std::endl;
        HDF5Reader data_reader(data_path);
        auto total_qs = IsBinaryType(index->index_type())? data_reader.GetQueryData<true>():data_reader.GetQueryData<false>();
        knowhere::SetDatasetRows(total_qs, nq);
        auto search_json = ReadIndexParams(index->index_type(), PARAMS_JSON::PERF_JSON);
        auto dim = knowhere::GetDatasetDim(total_qs);
        auto rows = knowhere::GetDatasetRows(total_qs);
        search_json["dim"] = dim;
        search_json["metric_type"] = metric_type;
        search_json["index_prefix"] = index_path;
    
        std::vector<double> latancy(kSerialSearchRound);
        Timer timer;
        for (auto i = 0; i < kSerialSearchRound; i++) {
            auto qs = std::make_shared<knowhere::Dataset>();
            knowhere::SetDatasetDim(qs, (int64_t)dim);
            knowhere::SetDatasetRows(qs,(int64_t)nq);
            knowhere::SetDatasetTensor(qs, ((float*)knowhere::GetDatasetTensor(total_qs) + (i * nq) % rows * dim));
            timer.start();
            index->Query(qs, search_json, nullptr);
            latancy[i] = timer.time_elapsed_ms();
        }
        auto aver_and_tp99 = GetAverageAndTP99Latancy(latancy.data(), kSerialSearchRound);
        result_out << "[PARAMS]:"<<search_json.dump()<<std::endl;
        result_out <<"[RESULT]: nq: "<<nq<<", average latancy: (ms) "<<aver_and_tp99.first<<", tp99: "<<aver_and_tp99.second<<std::endl;
    }
};
    void 
    BuildIndexTask(const std::string& index_type, const std::string& data_path, const std::string& index_path, const std::string& metric_type, std::ofstream& result_out) {
        result_out << "[BuildandSaveIndexTask]: "<< std::endl;
        size_t start_mem, build_mem_usage;
        start_mem = getCurrentRSS();
        
        auto build_json = ReadIndexParams(index_type, PARAMS_JSON::BUILD_JSON);
        HDF5Reader data_reader (data_path);
        Timer timer;
        size_t peak_mem = 0;
        if (index_type == knowhere::IndexEnum::INDEX_DISKANN) {
            fs::create_directory(index_path);
            // prepare data 
            std::string raw_data_path = index_path + "/raw_data.fbin";
            data_reader.SaveBaseBinFile(raw_data_path);
            // prepare config
            uint32_t ndim, nb;
            read_bin_meta(raw_data_path, nb, ndim);
            build_json["data_path"] = raw_data_path;
            build_json["index_prefix"] = index_path + "/";
            build_json["dim"] = (int)ndim;
            build_json["metric_type"] = metric_type;
            build_json["pq_code_budget_gb"] = build_json["pq_code_budget_gb_rate"].get<float>() * sizeof(float) * ndim * nb / 1024 / 1024 / 1024;
            std::cout<<build_json["pq_code_budget_gb"].get<float>()<<std::endl;
            std::shared_ptr<knowhere::FileManager> file_manager = std::make_shared<knowhere::LocalFileManager>();
            auto idx = std::make_shared<knowhere::IndexDiskANN<float>>(std::string(build_json["index_prefix"]), metric_type,
                                                                   std::make_unique<knowhere::LocalFileManager>());
            auto json = knowhere::Config::parse(build_json.dump());
            idx->BuildAll(nullptr, build_json);
            build_mem_usage = getPeakRSS() - start_mem;
            std::remove(raw_data_path.c_str());
        } else {
            // create memory index
            auto idx = knowhere::VecIndexFactory::GetInstance().CreateVecIndex(index_type, knowhere::IndexMode::MODE_CPU);
            auto ds = IsBinaryType(index_type) ?  data_reader.GetBaseData<true>():data_reader.GetBaseData<false>();
            build_json["dim"] = (int)knowhere::GetDatasetDim(ds);
            build_json["metric_type"] = metric_type;
            knowhere::Config json = knowhere::Config::parse(build_json.dump());
            idx->BuildAll(ds, json);
            build_mem_usage = getPeakRSS() - start_mem;
            knowhere::BinarySet bs;
            idx->Serialize(json);
            write_index_file(index_path, bs);
        } 
        auto build_time = timer.time_elapsed_s();
        result_out << "[PARAMS]:"<<build_json.dump()<<std::endl;
        result_out << "[RESULT]:build time:(s)" << build_time<<" mem usage:(MB) "<<std::endl;
    }

    void 
    SearchAccuracyTask(const std::string& index_type, const std::string& data_path, const std::string& index_path, const std::string& metric_type, std::ofstream& result_out) {
        result_out << "[SearchAccuracyTask]: "<< std::endl;
        // load index from index file 
        auto start_mem = getCurrentRSS();
        auto idx = LoadIndex(index_type, index_path, data_path, metric_type);
        auto load_mem_usage = getCurrentRSS() - start_mem;
        result_out << "After loading the index, memory usage(MB): "<<load_mem_usage<<std::endl;
        HDF5Reader data_reader(data_path);
        auto qs = IsBinaryType(index_type)? data_reader.GetQueryData<true>():data_reader.GetQueryData<false>();
        auto gt = data_reader.GetKNNGT();

        auto search_json_list = ReadIndexParams(index_type, PARAMS_JSON::ACC_JSON);

        for (knowhere::Config::iterator it = search_json_list.begin(); it != search_json_list.end(); ++it){
            auto search_json = *it;
            std::cout << *it <<std::endl;  
            search_json["index_prefix"] = index_path;
            search_json["dim"] = (int)knowhere::GetDatasetDim(qs);
            search_json["metric_type"] = metric_type;
            Timer timer;
            auto res = idx->Query(qs, search_json, nullptr);
            auto latancy = timer.time_elapsed_ms();
            auto nq = knowhere::GetDatasetRows(res);
            auto recall = CalcKNNRecall(knowhere::GetDatasetIDs(gt), knowhere::GetDatasetIDs(res), nq, knowhere::GetDatasetDim(gt), knowhere::GetDatasetDim(res));
            auto qps = nq / latancy * 1000;
            result_out <<"[PARAMS]:"<< search_json.dump()<<std::endl;
            result_out <<"[RESULT]:recall: "<<recall<<", latancy:"<<latancy<<", qps: "<<qps << std::endl;
        }    
    }
    
    void 
    SearchPerformanceTask(const std::string& index_type, const std::string& data_path, const std::string& index_path, const std::string& metric_type, std::ofstream& result_out) {
        result_out << "[SearchAccuracyTask]: "<< std::endl;
        auto start_mem = getCurrentRSS();
        auto index = LoadIndex(index_type, index_path, data_path, metric_type);
        auto load_mem_usage = getCurrentRSS() - start_mem;
        result_out << "After loading the index, memory usage(MB): "<<load_mem_usage<<std::endl;
        auto search_json = ReadTaskCfg(PARAMS_JSON::PERF_JSON);
        std::cout<<search_json<<std::endl;
        auto& nq_list = search_json.at("query_num");
        for (auto i = 0; i < nq_list.size(); i++) {
            int nq = nq_list.at(i).get<int>();
            SerialSearchTask(index, data_path, index_path, metric_type, nq, result_out); 
            ParallelSearchTask(index, index_path, metric_type, nq, search_json.at("client_num").get<int>(), result_out);
        }
    }
};