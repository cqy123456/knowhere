cmake_minimum_required (VERSION 2.6)
project(hnsw_lib
    LANGUAGES CXX)

add_library(hnswlib INTERFACE)
target_include_directories(hnswlib INTERFACE .) 

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  find_package(Boost COMPONENTS program_options)
  if (Boost_FOUND)
      include_directories (${Boost_INCLUDE_DIR})
      link_directories (${Boost_LIBRARY_DIR})
      message (STATUS "Found Boost.")
      message (STATUS "Include Path: ${Boost_INCLUDE_DIRS}")
      message (STATUS "Library Path: ${Boost_LIBRARY_DIRS}")
      message (STATUS "Library: ${Boost_LIBRARIES}")
  else()
      message (FATAL_ERROR "Could not find Boost 1.67!")
  endif()
    set(CMAKE_CXX_STANDARD 17)

    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      SET( CMAKE_CXX_FLAGS  "-Ofast -DNDEBUG -std=c++17 -DHAVE_CXX0X -openmp -march=native -fpic -ftree-vectorize")
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -DNDEBUG -std=c++17 -DHAVE_CXX0X -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -DNDEBUG -std=c++17 -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize" )
    endif()

    include_directories(/usr/local/hdf5/include)
    link_directories(/usr/local/hdf5/lib)

    include_directories(../DiskANN/include)
    include_directories(../easyloggingpp/src)
    include_directories(../faiss/faiss/src)
    include_directories(../faiss/)
    include_directories(../../include)
    include_directories(../../src)
    include_directories(../)
    include_directories(../../cqy_benchmark)

    link_directories(/data/knowhere-2.0/knowhere/build)
    # add_executable(test_updates examples/updates_test.cpp)
    # target_link_libraries(test_updates hnswlib)

    # add_executable(searchKnnCloserFirst_test examples/searchKnnCloserFirst_test.cpp)
    # target_link_libraries(searchKnnCloserFirst_test hnswlib)

    add_executable(hnsw_pq_test hnsw_pq_test.cpp )
    target_link_libraries(hnsw_pq_test hnswlib knowhere faiss hdf5 Boost::program_options )

endif()
