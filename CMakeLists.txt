cmake_minimum_required(VERSION 3.7)

file(GLOB TORCH_TVM_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

set(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")

message("Using pytorch dir $ENV{PYTORCH_DIR}")
link_directories($ENV{PYTORCH_DIR}/lib)
message("Using tvm dir $ENV{TVM_DIR}")
link_directories($ENV{TVM_DIR}/lib)
add_subdirectory(pybind11)

pybind11_add_module(torch_tvm SHARED ${TORCH_TVM_SRCS})
target_link_libraries(torch_tvm PUBLIC
  torch caffe2 c10 pybind11 tvm tvm_topi)

target_include_directories(torch_tvm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    $ENV{TVM_DIR}/include
    $ENV{TVM_DIR}/include/HalideIR
    $ENV{PYTORCH_DIR}/include
    ${PYBIND11_INCLUDE_DIR}
  )

