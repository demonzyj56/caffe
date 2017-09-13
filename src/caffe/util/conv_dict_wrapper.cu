#include "caffe/util/conv_dict_wrapper.hpp"
#include "caffe/util/math_functions.hpp"
#include "thrust/execution_policy.h"
#include "thrust/count.h"
#include "thrust/copy.h"
#include "thrust/find.h"
#include "thrust/device_vector.h"
#include "thrust/device_ptr.h"
#include "thrust/functional.h"
#include "thrust/equal.h"

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

struct InverseIndex {
    explicit InverseIndex(int nnz, const int *d_inds, int not_found)
        : nnz_(nnz), not_found_(not_found),
        inds_(thrust::device_pointer_cast(d_inds)) {}
    __host__ __device__ int operator()(const int &x) const {
        thrust::device_ptr<const int> found = thrust::find(thrust::device, inds_, inds_+nnz_, x);
        return (found != inds_+nnz_ ? found-inds_ : not_found_);
    }
    int nnz_;
    int not_found_;
    thrust::device_ptr<const int> inds_;
};

struct NonNegative {
    __host__ __device__ bool operator()(int x) {
        return x >= 0;
    }
};

template <typename Dtype>
shared_ptr<CSRWrapper<Dtype> > CSRWrapper<Dtype>::clip_columns_gpu_(int nnz, const int *d_inds) {
    CHECK_GE(nnz_, nnz);
    thrust::device_ptr<const int> thrust_inds = thrust::device_pointer_cast(d_inds);
    CHECK(thrust::equal(thrust::device, thrust_inds, thrust_inds+nnz-1, thrust_inds+1, thrust::less<int>()));

    shared_ptr<CSRWrapper<Dtype> > clipped(new CSRWrapper<Dtype>(handle_, r_, nnz, -1));
    thrust::device_vector<int> stencil(nnz_, -1);
    thrust::device_ptr<Dtype> thrust_values = thrust::device_pointer_cast(mutable_values());
    thrust::device_ptr<int> thrust_columns = thrust::device_pointer_cast(mutable_columns());
    InverseIndex inverse_index(nnz, d_inds, -1);
    NonNegative non_negative;
    thrust::transform(thrust::device, thrust_columns, thrust_columns + nnz_, stencil.begin(), inverse_index);
    clipped->mutable_cpu_ptrB()[0] = 0;
    for (int i = 0; i < r_; ++i) {
        clipped->mutable_cpu_ptrB()[i+1] = clipped->mutable_cpu_ptrB()[i] +
            thrust::count_if(thrust::device, stencil.begin() + cpu_ptrB()[i],
                stencil.begin() + cpu_ptrE()[i], non_negative);
    }
    clipped->set_nnz(clipped->mutable_cpu_ptrB()[r_]);
    thrust::copy_if(thrust::device, thrust_values, thrust_values + nnz_, stencil.begin(),
        thrust::device_pointer_cast(clipped->mutable_values()), non_negative);
    thrust::copy_if(thrust::device, stencil.begin(), stencil.end(),
        thrust::device_pointer_cast(clipped->mutable_columns()), non_negative);
    return clipped;
}

// require instaniation
template CSRWrapper<float>  &CSRWrapper<float>::identity();
template CSRWrapper<double> &CSRWrapper<double>::identity();
template shared_ptr<CSRWrapper<float> > CSRWrapper<float>::clip_columns_gpu_(int nnz, const int *d_inds);
template shared_ptr<CSRWrapper<double> > CSRWrapper<double>::clip_columns_gpu_(int nnz, const int *d_inds);

template void make_conv_dict_gpu<float>(const int n, const int m, const float *d_Dl, const int N,
    CSCParameter::Boundary boundary, float *d_values, int *d_columns, int *d_ptrB);
template void make_conv_dict_gpu<double>(const int n, const int m, const double *d_Dl, const int N,
    CSCParameter::Boundary boundary, double *d_values, int *d_columns, int *d_ptrB);


} // namespace caffe
