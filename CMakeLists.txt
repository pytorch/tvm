cmake_minimum_required(VERSION 3.7)

set(TORCH_TVM_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

set(CMAKE_CXX_STANDARD 11)

link_directories($ENV{PYTORCH_DIR}/lib)
add_subdirectory(pybind11)

pybind11_add_module(torch_tvm SHARED main.cpp)
target_link_libraries(torch_tvm PUBLIC torch caffe2 c10 pybind11)

target_include_directories(torch_tvm PUBLIC
  $ENV{PYTORCH_DIR}/include
  ${PYBIND11_INCLUDE_DIR}
  )

