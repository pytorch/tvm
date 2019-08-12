cmake_minimum_required(VERSION 3.7)

file(GLOB TORCH_TVM_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/torch_tvm/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/torch_tvm/custom_tvm_ops/relay/*.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/torch_tvm/custom_tvm_ops/topi/*.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/torch_tvm/custom_tvm_ops/topi/contrib/*.cc
)

set(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")

# PYTORCH_DIR
IF(DEFINED ENV{PYTORCH_DIR})
  SET(PYTORCH_DIR $ENV{PYTORCH_DIR})
ENDIF()
message("Using pytorch dir ${PYTORCH_DIR}")

# TVM_DIR
IF(DEFINED ENV{TVM_DIR})
  SET(TVM_DIR $ENV{TVM_DIR})
ENDIF()
message("Using tvm dir ${TVM_DIR}")

link_directories(${PYTORCH_DIR}/lib)

add_subdirectory(pybind11)
add_subdirectory(${TVM_DIR})

pybind11_add_module(_torch_tvm SHARED ${TORCH_TVM_SRCS})
target_link_libraries(_torch_tvm PUBLIC
  torch pybind11 tvm tvm_topi)

target_include_directories(_torch_tvm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/torch_tvm
    ${TVM_DIR}/include
    ${TVM_DIR}/src
    ${TVM_DIR}/include/HalideIR
    ${TVM_DIR}/3rdparty/dmlc-core/include/
    ${TVM_DIR}/3rdparty/dlpack/include/
    ${PYTORCH_DIR}/include
    ${PYBIND11_INCLUDE_DIR}
)
