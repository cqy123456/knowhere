#pragma once
#include "hdf5.h"
// #include "knowhere/config.h"
#include "cqy-benchmark/file_handler/bin_file_handler.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"
#include "common/Dataset.h"
namespace benchmark {
namespace {
static const char* HDF5_DATASET_TRAIN = "train";
static const char* HDF5_DATASET_TEST = "test";
static const char* HDF5_DATASET_NEIGHBORS = "neighbors";
static const char* HDF5_DATASET_DISTANCES = "distances";
static const char* HDF5_DATASET_LIMS = "lims";
static const char* HDF5_DATASET_RADIUS = "radius";
void
hdf5_read_meta(const std::string& file_name, const std::string& dataset_name, H5T_class_t dataset_class, int32_t& d_out,
            int32_t& n_out) {
    hid_t file, dataset, datatype, dataspace, memspace;
    H5T_class_t t_class;      /* data type class */
    hsize_t dimsm[3];         /* memory space dimensions */
    hsize_t dims_out[2];      /* dataset dimensions */
    hsize_t count[2];         /* size of the hyperslab in the file */
    hsize_t offset[2];        /* hyperslab offset in the file */
    hsize_t count_out[3];     /* size of the hyperslab in memory */
    hsize_t offset_out[3];    /* hyperslab offset in memory */
    void* data_out = nullptr; /* output buffer */

    /* Open the file and the dataset. */
    file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    dataset = H5Dopen2(file, dataset_name.c_str(), H5P_DEFAULT);

    /* Get datatype and dataspace handles and then query
        * dataset class, order, size, rank and dimensions. */
    datatype = H5Dget_type(dataset); /* datatype handle */
    t_class = H5Tget_class(datatype);
    assert(t_class == dataset_class || !"Illegal dataset class type");

    dataspace = H5Dget_space(dataset); /* dataspace handle */
    H5Sget_simple_extent_dims(dataspace, dims_out, nullptr);
    n_out = dims_out[0];
    d_out = dims_out[1];
    return; 
}

void*
hdf5_read(const std::string& file_name, const std::string& dataset_name, H5T_class_t dataset_class, int32_t& d_out,
            int32_t& n_out) {
    hid_t file, dataset, datatype, dataspace, memspace;
    H5T_class_t t_class;      /* data type class */
    hsize_t dimsm[3];         /* memory space dimensions */
    hsize_t dims_out[2];      /* dataset dimensions */
    hsize_t count[2];         /* size of the hyperslab in the file */
    hsize_t offset[2];        /* hyperslab offset in the file */
    hsize_t count_out[3];     /* size of the hyperslab in memory */
    hsize_t offset_out[3];    /* hyperslab offset in memory */
    void* data_out = nullptr; /* output buffer */

    /* Open the file and the dataset. */
    file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    dataset = H5Dopen2(file, dataset_name.c_str(), H5P_DEFAULT);

    /* Get datatype and dataspace handles and then query
        * dataset class, order, size, rank and dimensions. */
    datatype = H5Dget_type(dataset); /* datatype handle */
    t_class = H5Tget_class(datatype);
    assert(t_class == dataset_class || !"Illegal dataset class type");

    dataspace = H5Dget_space(dataset); /* dataspace handle */
    H5Sget_simple_extent_dims(dataspace, dims_out, nullptr);
    n_out = dims_out[0];
    d_out = dims_out[1];

    /* Define hyperslab in the dataset. */
    offset[0] = offset[1] = 0;
    count[0] = dims_out[0];
    count[1] = dims_out[1];
    H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, nullptr, count, nullptr);

    /* Define the memory dataspace. */
    dimsm[0] = dims_out[0];
    dimsm[1] = dims_out[1];
    dimsm[2] = 1;
    memspace = H5Screate_simple(3, dimsm, nullptr);

    /* Define memory hyperslab. */
    offset_out[0] = offset_out[1] = offset_out[2] = 0;
    count_out[0] = dims_out[0];
    count_out[1] = dims_out[1];
    count_out[2] = 1;
    H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offset_out, nullptr, count_out, nullptr);

    /* Read data from hyperslab in the file into the hyperslab in memory and display. */
    switch (t_class) {
        case H5T_INTEGER:
            data_out = new int32_t[dims_out[0] * dims_out[1]];
            H5Dread(dataset, H5T_NATIVE_INT32, memspace, dataspace, H5P_DEFAULT, data_out);
            break;
        case H5T_FLOAT:
            data_out = new float[dims_out[0] * dims_out[1]];
            H5Dread(dataset, H5T_NATIVE_FLOAT, memspace, dataspace, H5P_DEFAULT, data_out);
            break;
        default:
            printf("Illegal dataset class type\n");
            break;
    }

    /* Close/release resources. */
    H5Tclose(datatype);
    H5Dclose(dataset);
    H5Sclose(dataspace);
    H5Sclose(memspace);
    H5Fclose(file);
    
    return data_out;
}

// For binary vector, dim should be divided by 32, since we use int32 to store binary vector data */
template <bool is_binary>
void
hdf5_write(const char* file_name, const int32_t dim, const int32_t k, const void* xb, const int32_t nb,
            const void* xq, const int32_t nq, const void* g_ids, const void* g_dist) {
    /* Open the file and the dataset. */
    hid_t file = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    auto write_hdf5_dataset = [](hid_t file, const char* dataset_name, hid_t type_id, int32_t rows, int32_t cols,
                                    const void* data) {
        hsize_t dims[2];
        dims[0] = rows;
        dims[1] = cols;
        auto dataspace = H5Screate_simple(2, dims, NULL);
        auto dataset = H5Dcreate2(file, dataset_name, type_id, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        auto err = H5Dwrite(dataset, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        assert(err == 0);
        H5Dclose(dataset);
        H5Sclose(dataspace);
    };

    /* write train dataset */
    if (!is_binary) {
        write_hdf5_dataset(file, HDF5_DATASET_TRAIN, H5T_NATIVE_FLOAT, nb, dim, xb);
    } else {
        write_hdf5_dataset(file, HDF5_DATASET_TRAIN, H5T_NATIVE_INT32, nb, dim, xb);
    }

    /* write test dataset */
    if (!is_binary) {
        write_hdf5_dataset(file, HDF5_DATASET_TEST, H5T_NATIVE_FLOAT, nq, dim, xq);
    } else {
        write_hdf5_dataset(file, HDF5_DATASET_TEST, H5T_NATIVE_INT32, nq, dim, xq);
    }

    /* write ground-truth labels dataset */
    write_hdf5_dataset(file, HDF5_DATASET_NEIGHBORS, H5T_NATIVE_INT32, nq, k, g_ids);

    /* write ground-truth distance dataset */
    write_hdf5_dataset(file, HDF5_DATASET_DISTANCES, H5T_NATIVE_FLOAT, nq, k, g_dist);

    /* Close/release resources. */
    H5Fclose(file);
}

// For binary vector, dim should be divided by 32, since we use int32 to store binary vector data */
// Write HDF5 file with following dataset:
//    HDF5_DATASET_RADIUS    - H5T_NATIVE_FLOAT, [1, 1]
//    HDF5_DATASET_LIMS      - H5T_NATIVE_INT32, [1, nq+1]
//    HDF5_DATASET_NEIGHBORS - H5T_NATIVE_INT32, [1, lims[nq]]
//    HDF5_DATASET_DISTANCES - H5T_NATIVE_FLOAT, [1, lims[nq]]
template <bool is_binary>
void
hdf5_write_range(const char* file_name, const int32_t dim, const void* xb, const int32_t nb, const void* xq,
                    const int32_t nq, const float radius, const void* g_lims, const void* g_ids, const void* g_dist) {
    /* Open the file and the dataset. */
    hid_t file = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    auto write_hdf5_dataset = [](hid_t file, const char* dataset_name, hid_t type_id, int32_t rows, int32_t cols,
                                    const void* data) {
        hsize_t dims[2];
        dims[0] = rows;
        dims[1] = cols;
        auto dataspace = H5Screate_simple(2, dims, NULL);
        auto dataset = H5Dcreate2(file, dataset_name, type_id, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        auto err = H5Dwrite(dataset, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        assert(err == 0);
        H5Dclose(dataset);
        H5Sclose(dataspace);
    };

    /* write train dataset */
    if (!is_binary) {
        write_hdf5_dataset(file, HDF5_DATASET_TRAIN, H5T_NATIVE_FLOAT, nb, dim, xb);
    } else {
        write_hdf5_dataset(file, HDF5_DATASET_TRAIN, H5T_NATIVE_INT32, nb, dim, xb);
    }

    /* write test dataset */
    if (!is_binary) {
        write_hdf5_dataset(file, HDF5_DATASET_TEST, H5T_NATIVE_FLOAT, nq, dim, xq);
    } else {
        write_hdf5_dataset(file, HDF5_DATASET_TEST, H5T_NATIVE_INT32, nq, dim, xq);
    }

    /* write ground-truth radius */
    write_hdf5_dataset(file, HDF5_DATASET_RADIUS, H5T_NATIVE_FLOAT, 1, 1, &radius);

    /* write ground-truth lims dataset */
    write_hdf5_dataset(file, HDF5_DATASET_LIMS, H5T_NATIVE_INT32, 1, nq + 1, g_lims);

    /* write ground-truth labels dataset */
    write_hdf5_dataset(file, HDF5_DATASET_NEIGHBORS, H5T_NATIVE_INT32, 1, ((int32_t*)g_lims)[nq], g_ids);

    /* write ground-truth distance dataset */
    write_hdf5_dataset(file, HDF5_DATASET_DISTANCES, H5T_NATIVE_FLOAT, 1, ((int32_t*)g_lims)[nq], g_dist);

    /* Close/release resources. */
    H5Fclose(file);
}

// For binary vector, dim should be divided by 32, since we use int32 to store binary vector data */
// Write HDF5 file with following dataset:
//    HDF5_DATASET_RADIUS    - H5T_NATIVE_FLOAT, [1, nq]
//    HDF5_DATASET_LIMS      - H5T_NATIVE_INT32, [1, nq+1]
//    HDF5_DATASET_NEIGHBORS - H5T_NATIVE_INT32, [1, lims[nq]]
//    HDF5_DATASET_DISTANCES - H5T_NATIVE_FLOAT, [1, lims[nq]]
template <bool is_binary>
void
hdf5_write_range(const char* file_name, const int32_t dim, const void* xb, const int32_t nb, const void* xq,
                    const int32_t nq, const float* g_radius, const void* g_lims, const void* g_ids,
                    const void* g_dist) {
    /* Open the file and the dataset. */
    hid_t file = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    auto write_hdf5_dataset = [](hid_t file, const char* dataset_name, hid_t type_id, int32_t rows, int32_t cols,
                                    const void* data) {
        hsize_t dims[2];
        dims[0] = rows;
        dims[1] = cols;
        auto dataspace = H5Screate_simple(2, dims, NULL);
        auto dataset = H5Dcreate2(file, dataset_name, type_id, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        auto err = H5Dwrite(dataset, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        assert(err == 0);
        H5Dclose(dataset);
        H5Sclose(dataspace);
    };

    /* write train dataset */
    if (!is_binary) {
        write_hdf5_dataset(file, HDF5_DATASET_TRAIN, H5T_NATIVE_FLOAT, nb, dim, xb);
    } else {
        write_hdf5_dataset(file, HDF5_DATASET_TRAIN, H5T_NATIVE_INT32, nb, dim, xb);
    }

    /* write test dataset */
    if (!is_binary) {
        write_hdf5_dataset(file, HDF5_DATASET_TEST, H5T_NATIVE_FLOAT, nq, dim, xq);
    } else {
        write_hdf5_dataset(file, HDF5_DATASET_TEST, H5T_NATIVE_INT32, nq, dim, xq);
    }

    /* write ground-truth radius */
    write_hdf5_dataset(file, HDF5_DATASET_RADIUS, H5T_NATIVE_FLOAT, 1, nq, g_radius);

    /* write ground-truth lims dataset */
    write_hdf5_dataset(file, HDF5_DATASET_LIMS, H5T_NATIVE_INT32, 1, nq + 1, g_lims);

    /* write ground-truth labels dataset */
    write_hdf5_dataset(file, HDF5_DATASET_NEIGHBORS, H5T_NATIVE_INT32, 1, ((int32_t*)g_lims)[nq], g_ids);

    /* write ground-truth distance dataset */
    write_hdf5_dataset(file, HDF5_DATASET_DISTANCES, H5T_NATIVE_FLOAT, 1, ((int32_t*)g_lims)[nq], g_dist);

    /* Close/release resources. */
    H5Fclose(file);
}
};

class HDF5Reader {
public:
    explicit HDF5Reader(const std::string& fname) {
        file_name_ = fname;
    }

    template <bool is_binary>
    knowhere::DatasetPtr
    GetBaseData() {
        int32_t nb, ndim;
        void* data = nullptr; 
        if (!is_binary) {
            data = hdf5_read(file_name_, HDF5_DATASET_TRAIN, H5T_FLOAT, ndim, nb);
        } else {
            data = hdf5_read(file_name_, HDF5_DATASET_TRAIN, H5T_INTEGER, ndim, nb);
            ndim = ndim * 32;
        }
        auto bs = std::make_shared<knowhere::Dataset>();
        knowhere::SetDatasetRows(bs, nb);
        knowhere::SetDatasetDim(bs, ndim);
        knowhere::SetDatasetTensor(bs, data);
        return bs;
    }

    void 
    SaveBaseBinFile(const std::string& out_file) {
        auto bs = GetBaseData<false>();

        auto base = (float*)knowhere::GetDatasetTensor(bs);
        auto nb = (uint32_t)knowhere::GetDatasetRows(bs);
        auto ndim = (uint32_t)knowhere::GetDatasetDim(bs);
        std::cout<<"writting nv, ndim"<<nb<<" "<<ndim<<" "<<out_file<<std::endl;
        write_bin_file<float>(out_file, base, nb, ndim);
        return ;
    }

    template <bool is_binary> 
    void 
    GetDataShape(int32_t ndim, int32_t row) {
        if (!is_binary) {
            hdf5_read_meta(file_name_, HDF5_DATASET_TEST, H5T_FLOAT, ndim, row);
        } else {
            hdf5_read_meta(file_name_, HDF5_DATASET_TEST, H5T_INTEGER, ndim, row);
            ndim = ndim * 32;
        }
    }
    
    template <bool is_binary>
    knowhere::DatasetPtr
    GetQueryData() {
        int32_t nq, ndim;
        void* query = nullptr; 
        if (!is_binary) {
            query = hdf5_read(file_name_, HDF5_DATASET_TEST, H5T_FLOAT, ndim, nq);
        } else {
            query = hdf5_read(file_name_, HDF5_DATASET_TEST, H5T_INTEGER, ndim, nq);
            ndim = ndim * 32;
        }
        auto bs = std::make_shared<knowhere::Dataset>();
        knowhere::SetDatasetRows(bs, nq);
        knowhere::SetDatasetDim(bs, ndim);
        knowhere::SetDatasetTensor(bs, query);
        return bs;
    }

    knowhere::DatasetPtr
    GetKNNGT() {
        int32_t topk, nq;
        auto gt_ids_int32 = reinterpret_cast<int32_t*>(hdf5_read(file_name_, HDF5_DATASET_NEIGHBORS, H5T_INTEGER, topk, nq));
        int32_t topk_tmp, nq_tmp; 
        auto gt_dis = reinterpret_cast<float*>(hdf5_read(file_name_, HDF5_DATASET_DISTANCES, H5T_FLOAT, topk_tmp, nq_tmp));
        assert(topk == topk_tmp && nq == nq_tmp);
        int64_t* gt_ids_int64 = new int64_t[topk * nq];
        for (auto i = 0; i < topk * nq; i++) {
            gt_ids_int64[i] = gt_ids_int32[i];
        }
        auto gt = std::make_shared<knowhere::Dataset>();
        knowhere::SetDatasetRows(gt, nq);
        knowhere::SetDatasetDim(gt, topk);
        knowhere::SetDatasetIDs(gt, gt_ids_int64);
        knowhere::SetDatasetDistance(gt, gt_dis);
        delete [] gt_ids_int32;
        return gt;
    }

    knowhere::DatasetPtr 
    GetRadius() {
        int32_t nradius, ndim;
        auto radius_list = (float*)hdf5_read(file_name_, HDF5_DATASET_RADIUS, H5T_FLOAT, nradius, ndim);
        assert((ndim == 1) || !"incorrect ground truth radius");
        auto gt = std::make_shared<knowhere::Dataset>();
        knowhere::SetDatasetRows(gt, nradius);
        knowhere::SetDatasetDim(gt, 1);
        knowhere::SetDatasetTensor(gt, radius_list);
        return gt;
    }

    knowhere::DatasetPtr
    GetRangeSearchGT() {
        int32_t nq, cols, rows;
        auto gt_lims_int32 = (int32_t*)hdf5_read(file_name_, HDF5_DATASET_LIMS, H5T_INTEGER, cols, rows);
        assert((rows == 1) || !"incorrect dims of ground truth lims");
        nq = cols - 1;
        size_t* gt_lims = new size_t [cols * rows]; 
        for (auto i = 0; i < cols * rows; i++) {
            gt_lims[i] = gt_lims_int32[i];
        }

        auto gt_ids_int32 = (int32_t*)hdf5_read(file_name_, HDF5_DATASET_NEIGHBORS, H5T_INTEGER, cols, rows);
        assert((cols == gt_lims[nq] && rows == 1) || !"incorrect dims of ground truth labels");
        int64_t* gt_ids_int64 = new int64_t [cols * rows];
        for (auto i = 0; i < cols * rows; i++) {
            gt_ids_int64[i] = gt_ids_int32[i];
        }
        delete [] gt_ids_int32;
        
        auto gt_dist = (float*)hdf5_read(file_name_, HDF5_DATASET_DISTANCES, H5T_FLOAT, cols, rows);
        assert((cols == gt_lims[nq] && rows == 1) || !"incorrect dims of ground truth distances");

        auto gt = std::make_shared<knowhere::Dataset>();
        knowhere::SetDatasetRows(gt, nq);
        knowhere::SetDatasetIDs(gt, gt_ids_int64);
        knowhere::SetDatasetDistance(gt, gt_dist);
        knowhere::SetDatasetLims(gt, gt_lims);
        return gt;
        }

private:
    std::string file_name_;
};
};