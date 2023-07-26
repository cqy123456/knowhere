// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include <gtest/gtest.h>
#include <random>
#include <vector>
#include <functional>

#include "BitsetView.h"
#include "knowhere/common/Config.h"
#include "knowhere/feder/HNSW.h"
#include "knowhere/index/vector_index/ConfAdapterMgr.h"
#include "knowhere/index/vector_index/IndexHNSW.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"
#include "knowhere/index/vector_index/helpers/IndexParameter.h"
#include "unittest/Helper.h"
#include "unittest/range_utils.h"
#include "unittest/utils.h"
#include "evaluate.h"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

class HNSWTest : public DataGen, public TestWithParam<std::string> {
 protected:
    void
    SetUp() override {
        Init_with_default();
        conf_ = ParamGenerator::GetInstance().Gen(index_type_);
        index_ = std::make_shared<knowhere::IndexHNSW>();
    }

 protected:
    knowhere::Config conf_;
    knowhere::IndexMode index_mode_ = knowhere::IndexMode::MODE_CPU;
    knowhere::IndexType index_type_ = knowhere::IndexEnum::INDEX_HNSW;
    std::shared_ptr<knowhere::IndexHNSW> index_ = nullptr;
};

INSTANTIATE_TEST_CASE_P(HNSWParameters, HNSWTest, Values("HNSW"));

// TEST_P(HNSWTest, HNSW_basic) {
//     assert(!xb.empty());

//     // null faiss index
//     /*
//     {
//         ASSERT_ANY_THROW(index_->Serialize());
//         ASSERT_ANY_THROW(index_->Query(query_dataset, conf_));
//         ASSERT_ANY_THROW(index_->AddWithoutIds(nullptr, conf_));
//         ASSERT_ANY_THROW(index_->Count());
//         ASSERT_ANY_THROW(index_->Dim());
//     }
//     */

//     index_->BuildAll(base_dataset, conf_);
//     EXPECT_EQ(index_->Count(), nb);
//     EXPECT_EQ(index_->Dim(), dim);
//     ASSERT_GT(index_->Size(), 0);

//     GET_TENSOR_DATA(base_dataset)

//     // Serialize and Load before Query
//     knowhere::BinarySet bs = index_->Serialize(conf_);
//     knowhere::BinaryPtr bptr = std::make_shared<knowhere::Binary>();
//     bptr->data = std::shared_ptr<uint8_t[]>((uint8_t*)p_data, [&](uint8_t*) {});
//     bptr->size = dim * rows * sizeof(float);
//     bs.Append(RAW_DATA, bptr);

//     index_->Load(bs);

//     ASSERT_TRUE(index_->HasRawData(knowhere::GetMetaMetricType(conf_)));
//     auto result = index_->GetVectorById(id_dataset, conf_);
//     AssertVec(result, base_dataset, id_dataset, nq, dim);

//     std::vector<int64_t> ids_invalid(nq, nb);
//     auto id_dataset_invalid = knowhere::GenDatasetWithIds(nq, dim, ids_invalid.data());
//     ASSERT_ANY_THROW(index_->GetVectorById(id_dataset_invalid, conf_));

//     auto adapter = knowhere::AdapterMgr::GetInstance().GetAdapter(index_type_);
//     ASSERT_TRUE(adapter->CheckSearch(conf_, index_type_, index_mode_));

//     auto result1 = index_->Query(query_dataset, conf_, nullptr);
//     AssertAnns(result1, nq, k);

//     auto result2 = index_->Query(query_dataset, conf_, *bitset);
//     AssertAnns(result2, nq, k, CheckMode::CHECK_NOT_EQUAL);

//     // case: k > nb
//     const int64_t new_rows = 6;
//     knowhere::SetDatasetRows(base_dataset, new_rows);
//     index_->BuildAll(base_dataset, conf_);
//     auto result3 = index_->Query(query_dataset, conf_, nullptr);
//     auto res_ids = knowhere::GetDatasetIDs(result3);
//     for (int64_t i = 0; i < nq; i++) {
//         for (int64_t j = new_rows; j < k; j++) {
//             ASSERT_EQ(res_ids[i * k + j], -1);
//         }
//     }
// }

TEST_P(HNSWTest, HNSW_serialize) {
    auto serialize = [](const std::string& filename, knowhere::BinaryPtr& bin) {
        {
            FileIOWriter writer(filename);
            writer(static_cast<void*>(bin->data.get()), bin->size);
        }
    };
    int64_t bin_size;
    std::string filename = std::string("/tmp/HNSW_test_serialize.bin");
    std::cout<<"build and save"<<std::endl;
    // build process
    // {
    //     index_->BuildAll(base_dataset, conf_);
    //     auto binaryset = index_->Serialize(conf_);
    //     auto bin = binaryset.GetByName("HNSW");
        
    //     serialize(filename, bin);
    //     bin_size = bin->size;
    //     std::cout<<"build index size"<<bin_size<<std::endl;
    // }

    // load process 
    std::cout<<"load and search"<<std::endl;
    {           
        std::cout<<"before case:"<<benchmark::getCurrentRSS()<<std::endl;
        FileIOReader reader(filename);  
        std::cout<<"read binary from file"<<reader.size()<<std::endl;
        bin_size = reader.size();
        auto load_data = std::shared_ptr<uint8_t []>(new uint8_t[bin_size]);
        reader(static_cast<void*>(load_data.get()), bin_size);

        knowhere::BinarySet binaryset;
        std::cout<<"append binary set"<<bin_size<<std::endl;
        binaryset.Append("HNSW", load_data, bin_size);
        auto bin = binaryset.GetByName("HNSW");
        std::cout<<"search index size"<<bin->size/1024/1024<<std::endl;
        std::cout<<"before case:"<<benchmark::getCurrentRSS()<<std::endl;
        index_->Load(binaryset);
        auto peak_men =benchmark::getPeakRSS();
        std::cout<<"peak mem cost "<<peak_men<<std::endl;
        EXPECT_EQ(index_->Count(), nb);
        EXPECT_EQ(index_->Dim(), dim);
        auto result = index_->Query(query_dataset, conf_, nullptr);
        AssertAnns(result, nq, k);
    }
}

// TEST_P(HNSWTest, hnsw_slice) {
//     // serialize index
//     index_->BuildAll(base_dataset, conf_);
//     auto binaryset = index_->Serialize(knowhere::Config());
//     index_->Load(binaryset);
//     ASSERT_EQ(index_->Count(), nb);
//     ASSERT_EQ(index_->Dim(), dim);
//     auto result = index_->Query(query_dataset, conf_, nullptr);
//     AssertAnns(result, nq, knowhere::GetMetaTopk(conf_));
// }

// TEST_P(HNSWTest, hnsw_range_search_l2) {
//     knowhere::MetricType metric_type = knowhere::metric::L2;
//     knowhere::SetMetaMetricType(conf_, metric_type);

//     index_->BuildAll(base_dataset, conf_);

//     auto qd = knowhere::GenDataset(nq, dim, xq.data());

//     auto test_range_search_l2 = [&](const float range_filter, const float radius, const faiss::BitsetView bitset) {
//         std::vector<int64_t> golden_labels;
//         std::vector<float> golden_distances;
//         std::vector<size_t> golden_lims;
//         RunFloatRangeSearchBF(golden_labels, golden_distances, golden_lims, knowhere::metric::L2,
//                               xb.data(), nb, xq.data(), nq, dim, radius, range_filter, bitset);

//         auto adapter = knowhere::AdapterMgr::GetInstance().GetAdapter(index_type_);
//         ASSERT_TRUE(adapter->CheckRangeSearch(conf_, index_type_, index_mode_));

//         auto result = index_->QueryByRange(qd, conf_, bitset);
//         CheckRangeSearchResult(result, metric_type, nq, radius, range_filter,
//                                golden_labels.data(), golden_lims.data(), false, bitset);
//     };

//     for (std::pair<float, float> range: {
//              std::make_pair<float, float>(0, 16.81f),
//              std::make_pair<float, float>(16.81f, 17.64f),
//              std::make_pair<float, float>(17.64f, 18.49f)}) {
//         knowhere::SetMetaRangeFilter(conf_, range.first);
//         knowhere::SetMetaRadius(conf_, range.second);
//         test_range_search_l2(range.first, range.second, nullptr);
//         test_range_search_l2(range.first, range.second, *bitset);
//     }
// }

// TEST_P(HNSWTest, hnsw_range_search_ip) {
//     knowhere::MetricType metric_type = knowhere::metric::IP;
//     knowhere::SetMetaMetricType(conf_, metric_type);

//     normalize(xb.data(), nb, dim);
//     normalize(xq.data(), nq, dim);

//     index_->BuildAll(base_dataset, conf_);

//     auto qd = knowhere::GenDataset(nq, dim, xq.data());

//     auto test_range_search_ip = [&](const float range_filter, const float radius, const faiss::BitsetView bitset) {
//         std::vector<int64_t> golden_labels;
//         std::vector<float> golden_distances;
//         std::vector<size_t> golden_lims;
//         RunFloatRangeSearchBF(golden_labels, golden_distances, golden_lims, knowhere::metric::IP,
//                               xb.data(), nb, xq.data(), nq, dim, radius, range_filter, bitset);

//         auto adapter = knowhere::AdapterMgr::GetInstance().GetAdapter(index_type_);
//         ASSERT_TRUE(adapter->CheckRangeSearch(conf_, index_type_, index_mode_));

//         auto result = index_->QueryByRange(qd, conf_, bitset);
//         CheckRangeSearchResult(result, metric_type, nq, radius, range_filter,
//                                golden_labels.data(), golden_lims.data(), false, bitset);
//     };

//     for (std::pair<float, float> range: {
//         std::make_pair<float, float>(1.01f, 0.80f),
//         std::make_pair<float, float>(0.80f, 0.75f),
//         std::make_pair<float, float>(0.75f, 0.70f)}) {
//         knowhere::SetMetaRangeFilter(conf_, range.first);
//         knowhere::SetMetaRadius(conf_, range.second);
//         test_range_search_ip(range.first, range.second, nullptr);
//         test_range_search_ip(range.first, range.second, *bitset);
//     }
// }

// TEST_P(HNSWTest, HNSW_get_meta) {
//     assert(!xb.empty());

//     index_->BuildAll(base_dataset, conf_);

//     knowhere::SetIndexParamOverviewLevels(conf_, 2);
//     auto result = index_->GetIndexMeta(conf_);

//     auto json_info = knowhere::GetDatasetJsonInfo(result);
//     auto json_id_set = knowhere::GetDatasetJsonIdSet(result);
//     //std::cout << json_info << std::endl;
//     std::cout << "json_info size = " << json_info.size() << std::endl;
//     std::cout << "json_id_set size = " << json_id_set.size() << std::endl;

//     // check HNSWMeta
//     knowhere::feder::hnsw::HNSWMeta meta;
//     knowhere::Config j1 = nlohmann::json::parse(json_info);
//     ASSERT_NO_THROW(nlohmann::from_json(j1, meta));

//     ASSERT_EQ(meta.GetEfConstruction(), knowhere::GetIndexParamEfConstruction(conf_));
//     ASSERT_EQ(meta.GetM(), knowhere::GetIndexParamHNSWM(conf_));
//     ASSERT_EQ(meta.GetNumElem(), nb);

//     auto& hier_graph = meta.GetOverviewHierGraph();
//     for (auto& graph : hier_graph) {
//         auto& nodes = graph.GetNodes();
//         for (auto& node : nodes) {
//             ASSERT_GE(node.id_, 0);
//             ASSERT_LT(node.id_, nb);
//             for (auto n : node.neighbors_) {
//                 ASSERT_GE(n, 0);
//                 ASSERT_LT(n, nb);
//             }
//         }
//     }

//     // check IDSet
//     std::unordered_set<int64_t> id_set;
//     knowhere::Config j2 = nlohmann::json::parse(json_id_set);
//     ASSERT_NO_THROW(nlohmann::from_json(j2, id_set));
//     std::cout << "id_set num = " << id_set.size() << std::endl;
//     for (auto id : id_set) {
//         ASSERT_GE(id, 0);
//         ASSERT_LT(id, nb);
//     }
// }

// void
// CheckFederResult(const knowhere::DatasetPtr result, int64_t nb) {
//     auto json_info = knowhere::GetDatasetJsonInfo(result);
//     auto json_id_set = knowhere::GetDatasetJsonIdSet(result);
//     //std::cout << json_info << std::endl;
//     std::cout << "json_info size = " << json_info.size() << std::endl;
//     std::cout << "json_id_set size = " << json_id_set.size() << std::endl;

//     // check HNSWVisitInfo
//     knowhere::feder::hnsw::HNSWVisitInfo visit_info;
//     knowhere::Config j1 = nlohmann::json::parse(json_info);
//     ASSERT_NO_THROW(nlohmann::from_json(j1, visit_info));

//     for (auto& level_visit_record : visit_info.GetInfos()) {
//         auto& records = level_visit_record.GetRecords();
//         for (auto& record : records) {
//             auto id_from = std::get<0>(record);
//             auto id_to = std::get<1>(record);
//             auto dist = std::get<2>(record);
//             ASSERT_GE(id_from, 0);
//             ASSERT_GE(id_to, 0);
//             ASSERT_LT(id_from, nb);
//             ASSERT_LT(id_to, nb);
//             ASSERT_TRUE(dist >= 0.0 || dist == -1.0);
//         }
//     }

//     // check IDSet
//     std::unordered_set<int64_t> id_set;
//     knowhere::Config j2 = nlohmann::json::parse(json_id_set);
//     ASSERT_NO_THROW(nlohmann::from_json(j2, id_set));
//     std::cout << "id_set num = " << id_set.size() << std::endl;
//     for (auto id : id_set) {
//         ASSERT_GE(id, 0);
//         ASSERT_LT(id, nb);
//     }
// }

// TEST_P(HNSWTest, HNSW_trace_visit) {
//     assert(!xb.empty());

//     index_->BuildAll(base_dataset, conf_);

//     knowhere::SetMetaTraceVisit(conf_, true);
//     ASSERT_ANY_THROW(index_->Query(query_dataset, conf_, nullptr));

//     auto qd = knowhere::GenDataset(1, dim, xq.data());
//     auto result = index_->Query(qd, conf_, nullptr);

//     CheckFederResult(result, nb);
// }

// TEST_P(HNSWTest, HNSW_range_trace_visit) {
//     index_->BuildAll(base_dataset, conf_);

//     knowhere::SetMetaRadius(conf_, 4.1f);

//     knowhere::SetMetaTraceVisit(conf_, true);
//     ASSERT_ANY_THROW(index_->QueryByRange(query_dataset, conf_, nullptr));

//     auto qd = knowhere::GenDataset(1, dim, xq.data());
//     auto result = index_->QueryByRange(qd, conf_, nullptr);

//     CheckFederResult(result, nb);
// }

// TEST_P(HNSWTest, hnsw_data_overflow) {
//     auto data_p = (float*)knowhere::GetDatasetTensor(base_dataset);
//     auto dim = knowhere::GetDatasetDim(base_dataset);
//     auto rows = knowhere::GetDatasetRows(base_dataset);
//     for (auto i = 0; i < dim * rows; i++) {
//         data_p[i] = std::numeric_limits<float>::max() * data_p[i];
//         if (!std::isnormal(data_p[i])) {
//             data_p[i] = 1.0;
//         }
//     }
   
//     index_->BuildAll(base_dataset, conf_);
//     ASSERT_EQ(index_->Count(), nb);
//     ASSERT_EQ(index_->Dim(), dim);

//     auto result = index_->Query(base_dataset, conf_, nullptr);
// }

// namespace {
//     constexpr float kKnnRecallThreshold = 0.8f;
//     constexpr float kBruteForceRecallThreshold = 0.99f;
// }

// TEST_P(HNSWTest, hnsw_bitset) {
//     index_->BuildAll(base_dataset, conf_);
//     const auto metric = knowhere::GetMetaMetricType(conf_);
//     std::vector<std::function<std::vector<uint8_t>(size_t, size_t)>> gen_bitset_funcs = {
//         GenerateBitsetWithFirstTbitsSet, GenerateBitsetWithRandomTbitsSet};
//     const float threshold = hnswlib::kHnswBruteForceFilterRate;
//     const auto bitset_percentages = {0.4f, 0.98f};
//     for (const float percentage : bitset_percentages) {
//         for (const auto& gen_func : gen_bitset_funcs) {
//             auto bitset_data = gen_func(nb, percentage * nb);
//             faiss::BitsetView bs(bitset_data.data(), nb);
//             float* data_p = (float*)knowhere::GetDatasetTensor(base_dataset);
//             auto result = index_->Query(query_dataset, conf_, bs);
//             float* query_p = (float*)knowhere::GetDatasetTensor(query_dataset);
//             auto gt = GenGroundTruth(data_p, query_p, metric, nb, dim, nq, k, bs);
//             auto result_p = knowhere::GetDatasetIDs(result);
//             float recall = CheckTopKRecall(gt, result_p, k, nq);
//             if (percentage > threshold) {
//                 ASSERT_TRUE(recall > kBruteForceRecallThreshold);
//             } else {
//                 ASSERT_TRUE(recall > kKnnRecallThreshold);
//             }
//         }
//     }
// }