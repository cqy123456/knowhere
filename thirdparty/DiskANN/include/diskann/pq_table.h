// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "utils.h"
#include "immintrin.h"

#define NUM_PQ_CENTROIDS 256

namespace diskann {
  class FixedChunkPQTable {
    // data_dim = n_chunks * chunk_size;
    float* tables =
        nullptr;  // pq_tables = float* [[2^8 * [chunk_size]] * n_chunks]
    //    _u64   n_chunks;    // n_chunks = # of chunks ndims is split into
    //    _u64   chunk_size;  // chunk_size = chunk size of each dimension chunk
    _u64   ndims = 0;  // ndims = chunk_size * n_chunks
    _u64   n_chunks = 0;
    _u32*  chunk_offsets = nullptr;
    _u32*  rearrangement = nullptr;
    float* centroid = nullptr;
    float* tables_T = nullptr;  // same as pq_tables, but col-major
    float* part1 = nullptr;
    float* part2 = nullptr;

   public:
    FixedChunkPQTable() {
    }

    virtual ~FixedChunkPQTable() {
#ifndef EXEC_ENV_OLS
      if (tables != nullptr)
        delete[] tables;
      if (tables_T != nullptr)
        delete[] tables_T;
      if (rearrangement != nullptr)
        delete[] rearrangement;
      if (chunk_offsets != nullptr)
        delete[] chunk_offsets;
      if (centroid != nullptr)
        delete[] centroid;
      if (part1 != nullptr)
        delete[] part1;
      if (part2 != nullptr)
        delete[] part2;
#endif
    }

#ifdef EXEC_ENV_OLS
    void load_pq_centroid_bin(MemoryMappedFiles& files,
                              const char* pq_table_file, size_t num_chunks){
#else
    void load_pq_centroid_bin(const char* pq_table_file, size_t num_chunks) {
#endif
        std::string rearrangement_file =
            get_pq_rearrangement_perm_filename(std::string(pq_table_file));
    std::string chunk_offset_file =
        get_pq_chunk_offsets_filename(std::string(pq_table_file));
    std::string centroid_file =
        get_pq_centroid_filename(std::string(pq_table_file));

    // bin structure: [256][ndims][ndims(float)]
    uint64_t numr, numc;
    size_t   npts_u64, ndims_u64;
#ifdef EXEC_ENV_OLS
    diskann::load_bin<float>(files, pq_table_file, tables, npts_u64, ndims_u64);
#else
      diskann::load_bin<float>(pq_table_file, tables, npts_u64, ndims_u64);
#endif
    this->ndims = ndims_u64;

    if (file_exists(chunk_offset_file)) {
#ifdef EXEC_ENV_OLS
      diskann::load_bin<_u32>(files, rearrangement_file, rearrangement, numr,
                              numc);
#else
        diskann::load_bin<_u32>(rearrangement_file, rearrangement, numr, numc);
#endif
      if (numr != ndims_u64 || numc != 1) {
        diskann::cerr << "Error loading rearrangement file" << std::endl;
        throw diskann::ANNException("Error loading rearrangement file", -1,
                                    __FUNCSIG__, __FILE__, __LINE__);
      }

#ifdef EXEC_ENV_OLS
      diskann::load_bin<_u32>(files, chunk_offset_file, chunk_offsets, numr,
                              numc);
#else
        diskann::load_bin<_u32>(chunk_offset_file, chunk_offsets, numr, numc);
#endif
      if (numc != 1 || (numr != num_chunks + 1 && num_chunks != 0)) {
        LOG(ERROR) << "Error loading chunk offsets file. numc: " << numc
                   << " (should be 1). numr: " << numr << " (should be "
                   << num_chunks + 1 << ")";
        throw diskann::ANNException("Error loading chunk offsets file", -1,
                                    __FUNCSIG__, __FILE__, __LINE__);
      }
      LOG(DEBUG) << "PQ data has " << numr - 1 << " bytes per point.";
      this->n_chunks = numr - 1;

#ifdef EXEC_ENV_OLS
      diskann::load_bin<float>(files, centroid_file, centroid, numr, numc);
#else
        diskann::load_bin<float>(centroid_file, centroid, numr, numc);
#endif
      if (numc != 1 || numr != ndims_u64) {
        LOG(ERROR) << "Error loading centroid file";
        throw diskann::ANNException("Error loading centroid file", -1,
                                    __FUNCSIG__, __FILE__, __LINE__);
      }
    } else {
      this->n_chunks = num_chunks;
      rearrangement = new uint32_t[ndims];

      uint64_t chunk_size = DIV_ROUND_UP(ndims, num_chunks);
      for (uint32_t d = 0; d < ndims; d++)
        rearrangement[d] = d;
      chunk_offsets = new uint32_t[num_chunks + 1];
      for (uint32_t d = 0; d <= num_chunks; d++)
        chunk_offsets[d] = (_u32) (std::min)(ndims, d * chunk_size);
      centroid = new float[ndims];
      std::memset(centroid, 0, ndims * sizeof(float));
    }

    LOG_KNOWHERE_INFO_ << "PQ Pivots: #ctrs: " << npts_u64
                       << ", #dims: " << ndims_u64 << ", #chunks: " << n_chunks;
    //      assert((_u64) ndims_u32 == n_chunks * chunk_size);
    // alloc and compute transpose
    tables_T = new float[256 * ndims_u64];
    for (_u64 i = 0; i < 256; i++) {
      for (_u64 j = 0; j < ndims_u64; j++) {
        tables_T[j * 256 + i] = tables[i * ndims_u64 + j];
      }
    }

    part1 = new float[256 * n_chunks];
    part2 = new float[256 * ndims_u64];

    memset(part1, 0, 256 * n_chunks * sizeof(float));
    for (_u64 chunk = 0; chunk < n_chunks; chunk++) {
      // sum (q-c)^2 for the dimensions associated with this chunk
      float* part1_dists = part1 + (256 * chunk);
      for (_u64 j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++) {
        _u64         permuted_dim_in_query = rearrangement[j];
        const float* centers_dim_vec = tables_T + (256 * j);
        float*       part2_dists = part2 + (256 * j);
        auto         c = centroid[permuted_dim_in_query];
        for (_u64 idx = 0; idx < 256; idx++) {
          part1_dists[idx] += centers_dim_vec[idx] * centers_dim_vec[idx] +
                              c * c + 2 * centers_dim_vec[idx] * c;
          part2_dists[idx] = -2 * (centers_dim_vec[idx] + c);
        }
      }
    }
    std::cout << "end of loading pq" << std::endl;
  }

  _u32
  get_num_chunks() {
    return static_cast<_u32>(n_chunks);
  }
  void populate_chunk_distances(const float* query_vec, float* dist_vec) {
    // memset(dist_vec, 0, 256 * n_chunks * sizeof(float));
    // // chunk wise distance computation
    // for (_u64 chunk = 0; chunk < n_chunks; chunk++) {
    //   // sum (q-c)^2 for the dimensions associated with this chunk
    //   float* chunk_dists = dist_vec + (256 * chunk);
    //   for (_u64 j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++)
    //   {
    //     _u64         permuted_dim_in_query = rearrangement[j];
    //     const float* centers_dim_vec = tables_T + (256 * j);
    //     for (_u64 idx = 0; idx < 256; idx++) {
    //       double diff =
    //           centers_dim_vec[idx] - (query_vec[permuted_dim_in_query] -
    //                                   centroid[permuted_dim_in_query]);
    //       chunk_dists[idx] += (float) (diff * diff);
    //     }
    //   }
    // }

    auto q = new float[this->ndims];
    for (auto idx = 0; idx < this->ndims; idx++) {
      q[idx] = query_vec[rearrangement[idx]];
    }
    for (_u64 chunk = 0; chunk < n_chunks; chunk++) {
      float* chunk_dists = dist_vec + (256 * chunk);
      float* sum_part = part1 + (256 * chunk);
      auto   beg_dim = chunk_offsets[chunk];
      auto   end_dim = chunk_offsets[chunk + 1];

      // use avx2
      size_t offest = 0;
      __m256 msum1, msum2, msum3, msum4;

      while (offest < 256) {
        const float* mul = part2 + (256 * beg_dim) + offest;
        msum1 = _mm256_loadu_ps(sum_part);
        msum2 = _mm256_loadu_ps(sum_part + 8);
        msum3 = _mm256_loadu_ps(sum_part + 16);
        msum4 = _mm256_loadu_ps(sum_part + 24);
        size_t dim_idx = beg_dim;
        while (dim_idx < end_dim) {
          __m256 mx = _mm256_set1_ps(*(q + dim_idx));
          __m256 my1 = _mm256_loadu_ps(mul);
          __m256 my2 = _mm256_loadu_ps(mul + 8);
          __m256 my3 = _mm256_loadu_ps(mul + 16);
          __m256 my4 = _mm256_loadu_ps(mul + 24);
          msum1 = _mm256_fmadd_ps(mx, my1, msum1);
          msum2 = _mm256_fmadd_ps(mx, my2, msum2);
          msum3 = _mm256_fmadd_ps(mx, my3, msum3);
          msum4 = _mm256_fmadd_ps(mx, my4, msum4);
          mul += 256;
          dim_idx++;
        }
        _mm256_storeu_ps(chunk_dists, msum1);
        _mm256_storeu_ps(chunk_dists + 8, msum2);
        _mm256_storeu_ps(chunk_dists + 16, msum3);
        _mm256_storeu_ps(chunk_dists + 24, msum4);
        offest += 32;
        chunk_dists += 32;
        sum_part += 32;
      }
    }
    delete[] q;
  }

  float l2_distance(const float* query_vec, _u8* base_vec) {
    float res = 0;
    for (_u64 chunk = 0; chunk < n_chunks; chunk++) {
      for (_u64 j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++) {
        _u64         permuted_dim_in_query = rearrangement[j];
        const float* centers_dim_vec = tables_T + (256 * j);
        float        diff = centers_dim_vec[base_vec[chunk]] -
                     (query_vec[permuted_dim_in_query] -
                      centroid[permuted_dim_in_query]);
        res += diff * diff;
      }
    }
    return res;
  }

  float inner_product(const float* query_vec, _u8* base_vec) {
    float res = 0;
    for (_u64 chunk = 0; chunk < n_chunks; chunk++) {
      for (_u64 j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++) {
        _u64         permuted_dim_in_query = rearrangement[j];
        const float* centers_dim_vec = tables_T + (256 * j);
        float        diff =
            centers_dim_vec[base_vec[chunk]] *
            query_vec[permuted_dim_in_query];  // assumes centroid is 0 to
                                               // prevent translation errors
        res += diff;
      }
    }
    return -res;  // returns negative value to simulate distances (max -> min
                  // conversion)
  }

  void inflate_vector(_u8* base_vec, float* out_vec) {
    for (_u64 chunk = 0; chunk < n_chunks; chunk++) {
      for (_u64 j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++) {
        _u64         original_dim = rearrangement[j];
        const float* centers_dim_vec = tables_T + (256 * j);
        out_vec[original_dim] =
            centers_dim_vec[base_vec[chunk]] + centroid[original_dim];
      }
    }
  }

  void populate_chunk_inner_products(const float* query_vec, float* dist_vec) {
    memset(dist_vec, 0, 256 * n_chunks * sizeof(float));
    // chunk wise distance computation
    for (_u64 chunk = 0; chunk < n_chunks; chunk++) {
      // sum (q-c)^2 for the dimensions associated with this chunk
      float* chunk_dists = dist_vec + (256 * chunk);
      for (_u64 j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++) {
        _u64         permuted_dim_in_query = rearrangement[j];
        const float* centers_dim_vec = tables_T + (256 * j);
        for (_u64 idx = 0; idx < 256; idx++) {
          double prod =
              centers_dim_vec[idx] *
              query_vec[permuted_dim_in_query];  // assumes that we are not
                                                 // shifting the vectors to mean
                                                 // zero, i.e., centroid array
                                                 // should be all zeros
          chunk_dists[idx] -=
              (float) prod;  // returning negative to keep the search code clean
                             // (max inner product vs min distance)
        }
      }
    }
  }
};  // namespace diskann
}  // namespace diskann
