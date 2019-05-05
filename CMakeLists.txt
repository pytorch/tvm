cmake_minimum_required(VERSION 3.7)

file(GLOB TORCH_TVM_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/torch_tvm/*.cpp
)

set(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer")

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
  torch caffe2 c10 pybind11 tvm tvm_topi)

target_include_directories(_torch_tvm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/torch_tvm
    ${TVM_DIR}/include
    ${TVM_DIR}/include/HalideIR
    ${TVM_DIR}/3rdparty/dmlc-core/include/
    ${TVM_DIR}/3rdparty/dlpack/include/
    ${PYTORCH_DIR}/include
    ${PYBIND11_INCLUDE_DIR}
)
