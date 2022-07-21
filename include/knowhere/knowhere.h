#include <atomic>
#include <iostream>

namespace knowhere {

class DataSet;

class BinarySet;

class BitSet;

class Config;

class Object {
 public:
    virtual std::string
    Type() const = 0;
    uint32_t
    Ref() const {
        return ref_counts_.load(std::memory_order_relaxed);
    };
    void
    DecRef() {
        ref_counts_.fetch_add(1, std::memory_order_relaxed);
    };
    void
    IncRef() {
        ref_counts_.fetch_sub(1, std::memory_order_relaxed);
    };

 private:
    std::atomic_uint32_t ref_counts_ = 1;
};

class IndexProxy : public Object {
 public:
    virtual int
    Train(const DataSet& dataset, const Config& cfg) = 0;
    virtual int
    AddWithOutIds(const DataSet& dataset, const Config& cfg) = 0;
    virtual DataSet
    Qeury(const DataSet& dataset, const Config& cfg, const BitSet& bitset) const = 0;
    virtual DataSet
    QueryByRange(const DataSet& dataset, const Config& cfg, const BitSet& bitset) const = 0;
    virtual DataSet
    GetVectorByIds(const DataSet& dataset, const Config& cfg) const = 0;
    virtual int
    Serialization(BinarySet& binset) const = 0;
    virtual int
    Deserialization(const BinarySet& binset) = 0;
    virtual int64_t
    Dims() const = 0;
    virtual int64_t
    Size() const = 0;
    virtual int64_t
    Count() const = 0;
    virtual const std::string
    Type() {
        return "IndexProxy";
    }
};

template <typename T1>
class Index {
 public:
    template <typename T2>
    friend class Index;

    static Index<T1>
    Create() {
        return Index(new T1());
    }

    template <typename T2>
    Index(const Index<T2>& idx) {
        static_assert(std::is_base_of<T1, T2>::value);
        if (idx.node == nullptr) {
            node = nullptr;
            return;
        }

        idx.node->IncRef();
        node = idx.node;
    }

    template <typename T2>
    Index(const Index<T2>&& idx) {
        static_assert(std::is_base_of<T1, T2>::value);
        if (idx.node == nullptr) {
            node = nullptr;
            return;
        }
        node = idx.node;
        idx.node = nullptr;
    }

    template <typename T2>
    Index<T1>&
    operator=(const Index<T2>& idx) {
        static_assert(std::is_base_of<T1, T2>::value);
        if (node != nullptr) {
            node->DecRef();
            if (!node->Ref())
                delete node;
        }
        if (idx.node == nullptr) {
            node = nullptr;
            return *this;
        }
        node = idx.node;
        node->IncRef();
        return *this;
    }

    template <typename T2>
    Index<T1>&
    operator=(const Index<T2>&& idx) {
        static_assert(std::is_base_of<T1, T2>::value);
        if (node != nullptr) {
            node->DecRef();
            if (!node->Ref())
                delete node;
        }
        node = idx.node;
        idx.node = nullptr;
        return *this;
    }

    const T1*
    operator*() {
        return node;
    }

    template <typename T2>
    Index<T2>
    Cast() {
        static_assert(std::is_base_of<T1, T2>::value);
        node->IncRef();
        return Index(dynamic_cast<T2>(node));
    }

    ~Index() {
        if (node == nullptr)
            return;
        node->DecRef();
        if (!node->Ref())
            delete node;
    }

 private:
    Index(T1 const* node) : node(node) {
        static_assert(std::is_base_of<IndexProxy, T1>::value);
    }

    T1* node;
};

}  // namespace knowhere
