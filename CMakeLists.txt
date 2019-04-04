cmake_minimum_required(VERSION 3.7)

file(GLOB TORCH_TVM_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

set(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -w")


link_directories($ENV{PYTORCH_DIR}/lib)
add_subdirectory(pybind11)

set(DMLC_USE_GLOG ON)
set(USE_BLAS none)
set(USE_LLVM $ENV{USE_LLVM})
add_subdirectory(tvm)

pybind11_add_module(torch_tvm SHARED ${TORCH_TVM_SRCS})
target_link_libraries(torch_tvm PUBLIC torch caffe2 c10 pybind11 tvm tvm_topi)

target_include_directories(torch_tvm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(torch_tvm PUBLIC
    tvm/3rdparty/dlpack/include)
target_include_directories(torch_tvm PUBLIC
    tvm/3rdparty/HalideIR/src)
target_include_directories(torch_tvm PUBLIC
    tvm/3rdparty/dmlc-core/include)
target_include_directories(torch_tvm PUBLIC
    tvm/include)
target_include_directories(torch_tvm PUBLIC
    tvm/nnvm/include)
target_include_directories(torch_tvm PUBLIC
    tvm/topi/include)
target_include_directories(torch_tvm PUBLIC
    tvm/src)
target_include_directories(torch_tvm PUBLIC
  $ENV{PYTORCH_DIR}/include
  ${PYBIND11_INCLUDE_DIR}
  )

