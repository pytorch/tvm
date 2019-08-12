#include <dmlc/logging.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <cmath>
#include <iostream>

namespace tvm {
namespace contrib {

using namespace runtime;

TVM_REGISTER_GLOBAL("tvm.contrib.find_minmax")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      DLTensor* input = args[0];
      DLTensor* data_min = args[1];
      DLTensor* data_max = args[2];
      // calculate the data_min and data_max
      CHECK(input->strides == nullptr) << "find_minmax does not support the dltensor with strides";
      auto data_ptr = static_cast<float *>(input->data);
      int m = input->shape[0];
      int n = input->shape[1];
      float d_min = data_ptr[0];
      float d_max = data_ptr[0];
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            d_min = std::min(data_ptr[i*n + j], d_min);
            d_max = std::max(data_ptr[i*n + j], d_max);
        }
      }
      auto out_ptr_min = static_cast<float *>(data_min->data);
      auto out_ptr_max = static_cast<float *>(data_max->data);
      *out_ptr_min =  d_min;
      *out_ptr_max =  d_max;
    });

TVM_REGISTER_GLOBAL("tvm.contrib.choose_quantize_params")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      DLTensor* data_min_ptr = args[0];
      DLTensor* data_max_ptr = args[1];
      DLTensor* zero_point_ptr = args[2];
      DLTensor* scale_ptr = args[3];
      int32_t qmin = args[4];
      int32_t qmax = args[5];

      float data_min = *(static_cast<float *>(data_min_ptr->data));
      float data_max = *(static_cast<float *>(data_max_ptr->data));
      // copy from fbgemm implementation
      double scale =
            (std::max(data_max, 0.f) - std::min(data_min, 0.f)) / ((double)qmax - qmin);
      if (scale == 0) {
          scale = 0.1;
      }
      data_min = std::min(data_min, 0.f);
      data_max = std::max(data_max, 0.f);
      double zero_point_from_min = qmin - data_min / scale;
      double zero_point_from_max = qmax - data_max / scale;
      double zero_point_from_min_error = std::fabs(qmin) + std::fabs(data_min / scale);
      double zero_point_from_max_error = std::fabs(qmax) + std::fabs(data_max / scale);
      double initial_zero_point =
          zero_point_from_min_error < zero_point_from_max_error
          ? zero_point_from_min
          : zero_point_from_max;

      int32_t nudged_zero_point = 0;
      if (initial_zero_point < qmin) {
       nudged_zero_point = qmin;
      } else if (initial_zero_point > qmax) {
       nudged_zero_point = qmax;
      } else {
       nudged_zero_point = std::nearbyint(initial_zero_point);
      }

      auto zero_point_data_ptr = static_cast<int32_t *>(zero_point_ptr->data);
      auto scale_data_ptr = static_cast<float *>(scale_ptr->data);
      *zero_point_data_ptr = nudged_zero_point;
      *scale_data_ptr = scale;
    });

} // namespace contrib
} // namespace tvm
