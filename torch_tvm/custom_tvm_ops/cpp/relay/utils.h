#pragma once

#include <cstdint>

namespace tvm {
namespace relay {

namespace helper {
int32_t get_pack_width(int32_t dim_size, int32_t pack_factor=16);
} // namespace helper

} // namespace relay
} // namespace tvm
