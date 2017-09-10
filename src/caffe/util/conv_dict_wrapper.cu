#include "caffe/util/conv_dict_wrapper.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
__global__ void set_shifted_kernel(const int vec_len, int n, int m, const Dtype *x,
    Dtype *y) {
    CUDA_KERNEL_LOOP(index, vec_len) {
        int conv_row = index / (n*m);
        int conv_col = index % (n*m);
        int block = conv_col / m;
        int col = conv_col % m;
        conv_row = conv_row > (n-1) ? (n-1) : conv_row;
        int row = (conv_row - block + n) % n;
        y[index] = x[row * m + col];
    }
}

template <typename Dtype>
__global__ void index_shifted_kernel(const int vec_len, int n, int m, int N, int *y) {
    CUDA_KERNEL_LOOP(index, vec_len) {
        int i = index / (n*m);
        int j_index = index % (n*m);
        int col = j_index % m;
        int j = j_index / m;
        int block = i >= n ? (i + j - n + 1) : (i >= j ? j : (N-n+j));
        y[index] = block * m + col;
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

// The creation of identity is in cu file because it requires a iota like kernel.
template <typename Dtype>
CSRWrapper<Dtype> &CSRWrapper<Dtype>::identity() {
    CHECK_EQ(row(), col());
    CHECK_EQ(row(), nnz());
    caffe_gpu_set(nnz(), Dtype(1), mutable_values());
    index_inc_kernel<Dtype><<<CAFFE_GET_BLOCKS(nnz()), CAFFE_CUDA_NUM_THREADS>>>(
        nnz(), 1, mutable_columns());
    index_inc_kernel<Dtype><<<CAFFE_GET_BLOCKS(row()+1), CAFFE_CUDA_NUM_THREADS>>>(
        row()+1, 1, mutable_ptrB());
    return *this;
}

// require instaniation
template CSRWrapper<float>  &CSRWrapper<float>::identity();
template CSRWrapper<double> &CSRWrapper<double>::identity();

template void make_conv_dict_gpu<float>(const int n, const int m, const float *d_Dl, const int N,
    CSCParameter::Boundary boundary, float *d_values, int *d_columns, int *d_ptrB);
template void make_conv_dict_gpu<double>(const int n, const int m, const double *d_Dl, const int N,
    CSCParameter::Boundary boundary, double *d_values, int *d_columns, int *d_ptrB);


} // namespace caffe
