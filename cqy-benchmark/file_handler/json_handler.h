#pragma once
#include <string>
namespace benchmark {
std::string PARAMS_FILE = "params.json";
enum PARAMS_JSON {
    BUILD_JSON  = 0,
    LOAD_JSON   = 1,
    ACC_JSON    = 2,
    PERF_JSON   = 3
};

inline
knowhere::Config
ReadIndexParams(const std::string& index_type, const PARAMS_JSON json_type) {
    std::ifstream json_file(PARAMS_FILE);
    knowhere::Config all_config = knowhere::Config::parse(json_file);
    std::string json_type_str;
    switch (json_type) {
        case BUILD_JSON: json_type_str = "build";      break;
        case LOAD_JSON: json_type_str = "load";        break;
        case ACC_JSON: json_type_str = "accuracy_search";       break;
        case PERF_JSON: json_type_str = "performance_search"; break;
    };
    if(index_type.rfind("IVF", 0) == 0) {
        return all_config[json_type_str]["IVF"];
    } else {
        return all_config[json_type_str][index_type];
    }
}

inline 
knowhere::Config
ReadTaskCfg(const PARAMS_JSON json_type) {
    std::ifstream json_file(PARAMS_FILE);
    knowhere::Config all_config = knowhere::Config::parse(json_file);
    std::string json_type_str;
    switch (json_type) {
        case BUILD_JSON: json_type_str = "build";      break;
        case LOAD_JSON: json_type_str = "init";        break;
        case ACC_JSON: json_type_str = "accuracy_search";       break;
        case PERF_JSON: json_type_str = "performance_search"; break;
    };
    return all_config[json_type_str]["base"];
}
};