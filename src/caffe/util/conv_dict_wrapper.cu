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
#include "thrust/remove.h"

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
    LOG(WARNING) << "This version is deprecated.";
    CHECK_EQ(boundary, CSCParameter::CIRCULANT_BACK) 
        << "Only circulant back boundary is supported!";
    set_shifted_kernel<Dtype><<<CAFFE_GET_BLOCKS(N*m*n), CAFFE_CUDA_NUM_THREADS>>>(
        N*m*n, n, m, d_Dl, d_values);
    index_shifted_kernel<<<CAFFE_GET_BLOCKS(N*m*n), CAFFE_CUDA_NUM_THREADS>>>(
        N*m*n, n, m, N, d_columns);
    index_inc_kernel<<<CAFFE_GET_BLOCKS(N+1), CAFFE_CUDA_NUM_THREADS>>>(
        N+1, n*m, d_ptrB);
    CUDA_POST_KERNEL_CHECK;
}

// note the dictionary is transposed
template <typename Dtype>
__global__ void copy_dict_kernel(int vec_len, int n, int m, const Dtype *Dl, Dtype *values) {
    CUDA_KERNEL_LOOP(index, vec_len) {
        int ind = index % (n*m);
        int row = ind % n;
        int col = ind / n;
        values[index] = Dl[row * m + col];
    }
}

// view from original D, or in CSC format
// Launch one kernel per patch of size kernel_h x kernel_w.
// vec_len is now height x width x m x channels.
// index = ((h * width + w) * m + mm) * channels + c
__global__ void patches_column_circulant_kernel(int vec_len, int m, int channels, int height,
        int width, int kernel_h, int kernel_w, int pad_h, int pad_w, int *columns) {
    CUDA_KERNEL_LOOP(index, vec_len) {
        int *local_columns = columns + index * kernel_h * kernel_w;
        int w_index = index / channels;
        int c = index % channels;
        int patch_index = w_index / m;
        /* int mm = w_index % m; */
        int h = patch_index / width;
        int w = patch_index % width;
        for (int kh = 0; kh < kernel_h; ++kh) {
            int h_offset = (h + kh - pad_h + height) % height;
            for (int kw = 0; kw < kernel_w; ++kw) {
                int w_offset = (w + kw - pad_w + width) % width;
                *local_columns++ = (c * height + h_offset) * width + w_offset;
            }
        }
    }
}

// Invalid columns are marked as -1.
__global__ void patches_column_padzeros_kernel(int vec_len, int m, int channels, int height,
        int width, int kernel_h, int kernel_w, int pad_h, int pad_w, int *columns) {
    CUDA_KERNEL_LOOP(index, vec_len) {
        int *local_columns = columns + index * kernel_h * kernel_w;
        int w_index = index / channels;
        int c = index % channels;
        int patch_index = w_index / m;
        int h = patch_index / width;
        int w = patch_index % width;
        for (int kh = 0; kh < kernel_h; ++kh) {
            int h_offset = h + kh - pad_h;
            for (int kw = 0; kw < kernel_w; ++kw) {
                int w_offset = w + kw - pad_w;
                *local_columns++ = (h_offset >= 0 && h_offset < height && w_offset >= 0
                    && w_offset < width) ? (c * height + h_offset) * width + w_offset : -1;
            }
        }
    }
}

template <typename Dtype>
void make_transposed_conv_dict_circulant_gpu(int n, int m, const Dtype *Dl, int channels, int height, int width,
        int kernel_h, int kernel_w, int pad_h, int pad_w, Dtype *values, int *columns, int *ptrB) {
    CHECK_EQ(n, channels * kernel_h * kernel_w);
    int nnz = n * m * height * width;
    int col = m * height * width;
    copy_dict_kernel<Dtype><<<CAFFE_GET_BLOCKS(nnz), CAFFE_CUDA_NUM_THREADS>>>(
        nnz, n, m, Dl, values);
    patches_column_circulant_kernel<<<CAFFE_GET_BLOCKS(col*channels), CAFFE_CUDA_NUM_THREADS>>>(
        col*channels, m, channels, height, width, kernel_h, kernel_w, pad_h, pad_w, columns);
    index_inc_kernel<<<CAFFE_GET_BLOCKS(col+1), CAFFE_CUDA_NUM_THREADS>>>(
        col+1, n, ptrB);
    CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void make_transposed_conv_dict_padzeros_gpu(int n, int m, const Dtype *Dl, int channels, int height, int width,
        int kernel_h, int kernel_w, int pad_h, int pad_w, Dtype *values, int *columns, int *ptrB) {
    CHECK_EQ(n, channels * kernel_h * kernel_w);
    int nnz = n * m * height * width;
    int col = m * height * width;
    copy_dict_kernel<Dtype><<<CAFFE_GET_BLOCKS(nnz), CAFFE_CUDA_NUM_THREADS>>>(
        nnz, n, m, Dl, values);
    patches_column_padzeros_kernel<<<CAFFE_GET_BLOCKS(col*channels), CAFFE_CUDA_NUM_THREADS>>>(
        col*channels, m, channels, height, width, kernel_h, kernel_w, pad_h, pad_w, columns);
    index_inc_kernel<<<CAFFE_GET_BLOCKS(col+1), CAFFE_CUDA_NUM_THREADS>>>(
        col+1, n, ptrB);
    CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void make_transposed_conv_dict_gpu(int n, int m, const Dtype *Dl, int channels, int height, int width,
        int kernel_h, int kernel_w, CSCParameter::Boundary boundary, Dtype *values, int *columns,
        int *ptrB) {
    CHECK_EQ(n, channels * kernel_h * kernel_w);
    switch(boundary) {
        case CSCParameter::CIRCULANT_BACK:
            make_transposed_conv_dict_circulant_gpu(n, m, Dl, channels, height, width, kernel_h, kernel_w, 
                0, 0, values, columns, ptrB);
            break;
        case CSCParameter::CIRCULANT_FRONT:
            make_transposed_conv_dict_circulant_gpu(n, m, Dl, channels, height, width, kernel_h, kernel_w, 
                kernel_h-1, kernel_w-1, values, columns, ptrB);
            break;
        case CSCParameter::PAD_BACK:
            make_transposed_conv_dict_padzeros_gpu(n, m, Dl, channels, height, width, kernel_h, kernel_w, 
                0, 0, values, columns, ptrB);
            break;
        case CSCParameter::PAD_FRONT:
            make_transposed_conv_dict_padzeros_gpu(n, m, Dl, channels, height, width, kernel_h, kernel_w, 
                kernel_h-1, kernel_w-1, values, columns, ptrB);
            break;
        case CSCParameter::PAD_BOTH:
            CHECK_EQ(kernel_h % 2, 1);
            CHECK_EQ(kernel_w % 2, 1);
            make_transposed_conv_dict_padzeros_gpu(n, m, Dl, channels, height, width, kernel_h, kernel_w,
                (kernel_h-1)/2, (kernel_w-1)/2, values, columns, ptrB);
            break;
        case CSCParameter::NOPAD:
            LOG(FATAL) << "Non padding boundary condition is not supported!";
        default:
            NOT_IMPLEMENTED;
    }
}

// The creation of identity is in cu file because it requires a iota like kernel.
template <typename Dtype>
CSRWrapper<Dtype> &CSRWrapper<Dtype>::identity() {
    CHECK_EQ(row(), col());
    CHECK_EQ(row(), nnz());
    caffe_gpu_set(nnz(), Dtype(1), mutable_values());
    index_inc_kernel<<<CAFFE_GET_BLOCKS(nnz()), CAFFE_CUDA_NUM_THREADS>>>(
        nnz(), 1, mutable_columns());
    index_inc_kernel<<<CAFFE_GET_BLOCKS(row()+1), CAFFE_CUDA_NUM_THREADS>>>(
        row()+1, 1, mutable_ptrB());
    CUDA_POST_KERNEL_CHECK;
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

struct Negative {
    __host__ __device__ bool operator()(int x) {
        return x < 0;
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

// Perform pruning on GPU side.
template <typename Dtype>
void CSRWrapper<Dtype>::prune() {
    thrust::device_ptr<Dtype> thrust_values = thrust::device_pointer_cast(mutable_values());
    thrust::device_ptr<int> thrust_columns = thrust::device_pointer_cast(mutable_columns());
    vector<int> valid_nnz(r_);
    for (int i = 0; i < r_; ++i) {
        valid_nnz[i] = thrust::count_if(thrust::device, thrust_columns + cpu_ptrB()[i],
            thrust_columns + cpu_ptrE()[i], NonNegative());
    }
    for (int i = 0; i < r_; ++i) {
        mutable_cpu_ptrB()[i+1] = cpu_ptrB()[i] + valid_nnz[i];
    }
    thrust::device_ptr<Dtype> values_end = thrust::remove_if(thrust_values, thrust_values + nnz_,
        thrust_columns, Negative());
    CHECK_EQ(cpu_ptrB()[r_], values_end - thrust_values);
    thrust::device_ptr<int> columns_end = thrust::remove_if(thrust_columns, thrust_columns + nnz_, Negative());
    CHECK_EQ(cpu_ptrB()[r_], columns_end - thrust_columns);
    nnz_ = cpu_ptrB()[r_];
}

// require instaniation
template CSRWrapper<float>  &CSRWrapper<float>::identity();
template CSRWrapper<double> &CSRWrapper<double>::identity();
template shared_ptr<CSRWrapper<float> > CSRWrapper<float>::clip_columns_gpu_(int nnz, const int *d_inds);
template shared_ptr<CSRWrapper<double> > CSRWrapper<double>::clip_columns_gpu_(int nnz, const int *d_inds);
template void CSRWrapper<float>::prune();
template void CSRWrapper<double>::prune();

template void make_conv_dict_gpu<float>(const int n, const int m, const float *d_Dl, const int N,
    CSCParameter::Boundary boundary, float *d_values, int *d_columns, int *d_ptrB);
template void make_conv_dict_gpu<double>(const int n, const int m, const double *d_Dl, const int N,
    CSCParameter::Boundary boundary, double *d_values, int *d_columns, int *d_ptrB);

template void make_transposed_conv_dict_gpu<float>(int n, int m, const float *Dl, int channels,
        int height, int width, int kernel_h, int kernel_w, CSCParameter::Boundary boundary,
        float *values, int *columns, int *ptrB);
template void make_transposed_conv_dict_gpu<double>(int n, int m, const double *Dl, int channels,
        int height, int width, int kernel_h, int kernel_w, CSCParameter::Boundary boundary,
        double *values, int *columns, int *ptrB);

} // namespace caffe
