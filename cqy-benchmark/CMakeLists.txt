set(CMAKE_CXX_STANDARD 17)

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

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/thirdparty/faiss)
include_directories(${CMAKE_SOURCE_DIR}/thirdparty/faiss/faiss)
include_directories(${CMAKE_SOURCE_DIR}/cqy-benchmark)

include_directories(/usr/local/hdf5/include)
link_directories(/usr/local/hdf5/lib)

include_directories(${KNOWHERE_SOURCE_DIR}/knowhere)

message("${CMAKE_SOURCE_DIR}/thirdparty/faiss")
add_executable(knowhere-benchmark benchmark.cc)
target_link_libraries(knowhere-benchmark faiss knowhere Boost::program_options hdf5 stdc++fs)

add_executable(ivfpqfast_sift1m ivfpqfast_sift1m.cpp)
target_link_libraries(ivfpqfast_sift1m PRIVATE faiss knowhere hdf5)