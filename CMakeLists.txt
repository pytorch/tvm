cmake_minimum_required(VERSION 3.7)

file(GLOB TORCH_TVM_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

set(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")

if ("${PYTORCH_DIR}" STREQUAL "")
  set(PYTORCH_DIR "$ENV{PYTORCH_DIR}")
endif()
if ("${TVM_DIR}" STREQUAL "")
  set(TVM_DIR "$ENV{TVM_DIR}")
endif()

message("-- Using PyTorch directory: ${PYTORCH_DIR}")
link_directories(${PYTORCH_DIR}/lib)
message("-- Using TVM directory: ${TVM_DIR}")
link_directories(${TVM_DIR}/lib)
add_subdirectory(pybind11)

pybind11_add_module(torch_tvm SHARED ${TORCH_TVM_SRCS})
target_link_libraries(torch_tvm PUBLIC
  torch caffe2 c10 pybind11 tvm tvm_topi)

target_include_directories(torch_tvm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${TVM_DIR}/include
    ${TVM_DIR}/include/HalideIR
    ${PYTORCH_DIR}/include
    ${PYBIND11_INCLUDE_DIR}
  )

