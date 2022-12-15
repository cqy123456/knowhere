#include <cassert>
#include <unordered_map>
#include <functional>
#include "cqy-benchmark/runner/tasks.h"

namespace benchmark {  
using TaskFunction = std::function<void(const std::string&, const std::string&, const std::string&, const std::string&, std::ofstream& result_out)>;
class Runner {
    public:
        explicit Runner() {
            mapping_runner_function();
        }
        
        void 
        Run(std::string task_name, const std::string& index_type, const std::string& data_path, const std::string& index_path, const std::string& metric_type, std::ofstream& result_out) {
            assert(task_mapper.find(task_name) != task_mapper.end());
            task_mapper[task_name](index_type, data_path, index_path, metric_type, result_out);
        }
        
    private:
        void mapping_runner_function() {
            task_mapper["build"] = BuildIndexTask;
            task_mapper["accuracy"] = SearchAccuracyTask;
            task_mapper["performance"] = SearchPerformanceTask;
        }
        std::unordered_map<std::string, TaskFunction> task_mapper;
};
};