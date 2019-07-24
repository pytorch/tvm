cmake_minimum_required(VERSION 3.7)

file(GLOB TOPI_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
)

set(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
get_filename_component(CUSTOM_TVM_OPS_DIR ../ ABSOLUTE)

# TVM_DIR
IF(DEFINED ENV{TVM_DIR})
  SET(TVM_DIR $ENV{TVM_DIR})
ENDIF()
message("Using tvm dir ${TVM_DIR}")

add_library(custom_tvm_ops_topi SHARED ${TOPI_SRCS})

target_include_directories(custom_tvm_ops_topi PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUSTOM_TVM_OPS_DIR}
    ${TVM_DIR}/include
    ${TVM_DIR}/topi/include
    ${TVM_DIR}/include/HalideIR
    ${TVM_DIR}/3rdparty/dmlc-core/include/
    ${TVM_DIR}/3rdparty/dlpack/include/
)
