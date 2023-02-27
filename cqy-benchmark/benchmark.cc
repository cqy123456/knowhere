#include <boost/program_options.hpp>
#include <fstream>
#include <iostream>

#include "file_handler/hdf5_handler.h"
#include "knowhere/dataset.h"
#include "runner/runner.h"
#include "util/evaluate.h"
namespace po = boost::program_options;

int
main(int argc, char** argv) {
    std::string index_type;
    std::string runner_type;
    std::string data_path;
    std::string index_path;
    std::string log_path;
    std::string metric_type;
    po::options_description desc{"Arguments"};
    try {
        desc.add_options()("help,h", "Print information on arguments.");
        desc.add_options()("runner_type,r", po::value<std::string>(&runner_type)->required(),
                           "Runner type: [build, accuracy, performance].");
        desc.add_options()("index_type,i", po::value<std::string>(&index_type)->required(),
                           "Index type: [FLAT, IVFFLAT, IVFPQ, IVFSQ8, DISKANN].");
        desc.add_options()("data_path", po::value<std::string>(&data_path)->required(),
                           "The file path of the HDF5 dataset.");
        desc.add_options()("index_path", po::value<std::string>(&index_path)->required(),
                           "The path to load or store the index.");
        desc.add_options()("metric_type", po::value<std::string>(&metric_type)->required(),
                           "The compute function: [L2, IP] for float, [jaccard] for binary");
        desc.add_options()("log_path", po::value<std::string>(&log_path)->required(),
                           "The path to load or store the index.");
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
    std::ofstream result_out(log_path.c_str(), std::ios::out);
    benchmark::Runner task_runner;
    task_runner.Run(runner_type, index_type, data_path, index_path, metric_type, result_out);
    result_out.close();
    return 0;
}