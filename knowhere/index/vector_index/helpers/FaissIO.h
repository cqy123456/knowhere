// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#pragma once

#include <faiss/impl/io.h>

namespace knowhere {

struct MemoryIOWriter : public faiss::IOWriter {
    uint8_t* data_ = nullptr;
    size_t total = 0;
    size_t rp = 0;

    size_t
    operator()(const void* ptr, size_t size, size_t nitems) override;

    template <typename T>
    size_t
    write(T* ptr, size_t size, size_t nitems = 1) {
        return operator()((const void*)ptr, size, nitems);
    }
};

struct MemoryIOReader : public faiss::IOReader {
    uint8_t* data_;
    size_t rp = 0;
    size_t total = 0;

    size_t
    operator()(void* ptr, size_t size, size_t nitems) override;

    template <typename T>
    size_t
    read(T* ptr, size_t size, size_t nitems = 1) {
        return operator()((void*)ptr, size, nitems);
    }
};

struct MemoryMapper {
    std::shared_ptr<uint8_t[]>  data_;
    size_t rp = 0;
    size_t total = 0;

    size_t
    operator()(void* ptr, size_t size, bool zero_copy, size_t nitems);
    template <typename T>
    size_t 
    read(T* ptr, size_t size, bool zero_copy = false, size_t nitems = 1) {
        return operator()(ptr, size, zero_copy, nitems);
    }

    template <typename T>
    size_t pin(T*& ptr, size_t size, bool zero_copy, size_t nitems = 1) {
        if (rp >= total) {
            return 0;
        }
        size_t nremain = (total - rp) / size;
        if (nremain < nitems) {
            nitems = nremain;
        }
        ptr = (T*)(data_.get() + rp);
        rp += size * nitems;
        return nitems;
    }
    std::shared_ptr<uint8_t[]> 
    getData() {return std::move(data_);}
};

}  // namespace knowhere
