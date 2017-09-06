#include "caffe/util/conv_dict_wrapper.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
__global__ void set_shifted_kernel(const int vec_len, int n, int m, const Dtype *x,
    Dtype *y) {
    int nm = n * m;
    CUDA_KERNEL_LOOP(index, vec_len) {
        int shift_level = index / nm;
        int shift = index % nm;
        int c = shift % m;
        int r = (shift_level - shift / m + n) % n;
        y[index] = x[r * m + c];
    }
}

template <typename Dtype>
__global__ void index_shifted_kernel(const int vec_len, int n, int m, int N, int *y) {
    int nm = n * m;
    CUDA_KERNEL_LOOP(index, vec_len) {
        int shift_level = index / nm;
        int shift = index % nm;
        int c = shift % m;
        int r = shift / m;
        int column = ((r > shift_level ? r - n : r) + N) % N;
        y[index] = column * m + c;
    }
}

template <typename Dtype>
__global__ void index_inc_kernel(const int vec_len, const int inc, int *y) {
    CUDA_KERNEL_LOOP(index, vec_len) {
        y[index] = index * inc;
    }
}

// Standalone funtion to create a zero-based sparse CSR format convolutional dictionary
// from a dense local dictionary Dl.
// Dl has row n and column m, and d_values and d_columns have length n*m*N, d_ptrB have
// length N + 1.
template <typename Dtype>
void make_conv_dict_gpu(const int n, const int m, const Dtype *d_Dl, const int N,
    CSCParameter::Boundary boundary, Dtype *d_values, int *d_columns, int *d_ptrB) {
    CHECK_EQ(boundary, CSCParameter::CIRCULANT_BACK) 
        << "Only circulant back boundary is supported!";
    set_shifted_kernel<Dtype><<<CAFFE_GET_BLOCKS(N*m*n), CAFFE_CUDA_NUM_THREADS>>>(
        N*m*n, n, m, d_Dl, d_values);
    index_shifted_kernel<Dtype><<<CAFFE_GET_BLOCKS(N*m*n), CAFFE_CUDA_NUM_THREADS>>>(
        N*m*n, n, m, N, d_columns);
    index_inc_kernel<Dtype><<<CAFFE_GET_BLOCKS(N+1), CAFFE_CUDA_NUM_THREADS>>>(
        N+1, n*m, d_ptrB);
}

template void make_conv_dict_gpu<float>(const int n, const int m, const float *d_Dl, const int N,
    CSCParameter::Boundary boundary, float *d_values, int *d_columns, int *d_ptrB);
template void make_conv_dict_gpu<double>(const int n, const int m, const double *d_Dl, const int N,
    CSCParameter::Boundary boundary, double *d_values, int *d_columns, int *d_ptrB);

} // namespace caffe
