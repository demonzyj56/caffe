#include "caffe/util/csc_helpers.hpp"
#include "caffe/common.hpp"

template <typename Dtype>
__global__ void soft_thresholding_kernel(const int n, const Dtype thresh, Dtype *x) {
  CUDA_KERNEL_LOOP(index, n) {
    x[index] = x[index] > thresh ? x[index] - thresh :
      (x[index] < -thresh ? x[index] + thresh : 0.);
  }
}

template <>
void caffe_gpu_soft_thresholding<float>(const int n, const float thresh, float *x) {
  soft_thresholding_kernel<float><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
    n, thresh, x);
}

template <>
void caffe_gpu_soft_thresholding<double>(const int n, const double thresh, double *x) {
  soft_thresholding_kernel<double><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
    n, thresh, x);
}
