#pragma once
#include <string>

#include "knowhere/binaryset.h"
#include "knowhere/dataset.h"
namespace benchmark {
namespace {
struct IndexIOWriter {
    std::fstream fs;
    std::string name;

    explicit IndexIOWriter(const std::string& fname) {
        name = fname;
        fs = std::fstream(name, std::ios::out | std::ios::binary);
    }

    ~IndexIOWriter() {
        fs.close();
    }

    size_t
    operator()(void* ptr, size_t size) {
        fs.write(reinterpret_cast<char*>(ptr), size);
        return size;
    }
};

struct IndexIOReader {
    std::fstream fs;
    std::string name;

    explicit IndexIOReader(const std::string& fname) {
        name = fname;
        fs = std::fstream(name, std::ios::in | std::ios::binary);
    }

    ~IndexIOReader() {
        fs.close();
    }

    size_t
    operator()(void* ptr, size_t size) {
        fs.read(reinterpret_cast<char*>(ptr), size);
        return size;
    }

    size_t
    size() {
        fs.seekg(0, fs.end);
        size_t len = fs.tellg();
        fs.seekg(0, fs.beg);
        return len;
    }
};
};  // namespace
/* memory index bin file handler */
void
write_index_file(const std::string& filename, const knowhere::BinarySet& binary_set) {
    IndexIOWriter writer(filename);

    const auto& m = binary_set.binary_map_;
    for (auto it = m.begin(); it != m.end(); ++it) {
        const std::string& name = it->first;
        size_t name_size = name.length();
        const knowhere::BinaryPtr data = it->second;
        size_t data_size = data->size;

        writer(&name_size, sizeof(name_size));
        writer(&data_size, sizeof(data_size));
        writer((void*)name.c_str(), name_size);
        writer(data->data.get(), data_size);
    }
}

void
read_index_file(const std::string& filename, knowhere::BinarySet& binary_set) {
    binary_set.clear();

    IndexIOReader reader(filename);

    int64_t offset = 0;
    auto file_size = reader.size();
    while (offset < file_size) {
        size_t name_size, data_size;
        reader(&name_size, sizeof(size_t));
        offset += sizeof(size_t);
        reader(&data_size, sizeof(size_t));
        offset += sizeof(size_t);

        std::string name;
        name.resize(name_size);
        reader(name.data(), name_size);
        offset += name_size;
        auto data = new uint8_t[data_size];
        reader(data, data_size);
        offset += data_size;

        std::shared_ptr<uint8_t[]> data_ptr(data);
        binary_set.Append(name, data_ptr, data_size);
    }
}

/* raw bin file handler of diskann */
template <typename T>
T*
read_bin_file(const std::string& data_path, uint32_t& row, uint32_t& col) {
    std::ifstream reader(data_path.c_str(), std::ios::binary);
    reader.read((char*)&row, sizeof(uint32_t));
    reader.read((char*)&col, sizeof(uint32_t));
    auto data = new T[row * col];
    reader.read((char*)data, sizeof(T) * col * row);
    return data;
}

void
read_bin_meta(const std::string& data_path, uint32_t& row, uint32_t& col) {
    std::ifstream reader(data_path.c_str(), std::ios::binary);
    reader.read((char*)&row, sizeof(uint32_t));
    reader.read((char*)&col, sizeof(uint32_t));
    reader.close();
    return;
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
};  // namespace benchmark
