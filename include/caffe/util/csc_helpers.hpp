#ifndef CAFFE_CSC_HELPERS_HPP_
#define CAFFE_CSC_HELPERS_HPP_ 
/*
 * Utils for csc layer.
 */
#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


// An implementation of LASSO on CPU using SPAMS toolbox.
// The matrix X, D, alpha are all two dimensional blobs,
// where the size of X is n-by-N, D is n-by-p, 
// alpha is p-by-N.
// L specify the maximum steps of the homotopy algorithm, which could be used as 
// a stopping criteria.
template <typename Dtype>
void lasso_cpu(const Blob<Dtype> *X, const Blob<Dtype> *D, Dtype lambda1, Dtype lambda2,
    int L, Blob<Dtype> *alpha);

// Utility function for transposing a matrix inplace or out of place.
// This routine is supported by OpenBLAS and MKL.
// Valid Dtype is 'float' and 'double'.
template <Dtype>
void caffe_cpu_imatcopy(const int rows, const int cols, Dtype *X);
template <Dtype>
void caffe_cpu_omatcopy(const int M, const int N, const Dtype *X, Dtype *Y);

// im2col/col2im dispatch function, supporting circulant boundary.
template <typename Dtype>
void im2col_csc_cpu(const Dtype *blob, const int nsamples, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const CSCParameter::Boundary boundary, Dtype *patches);
template <typename Dtype>
void im2col_csc_gpu(const Dtype *blob, const int nsamples, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const CSCParameter::Boundary boundary, Dtype *patches);
template <typename Dtype>
void col2im_csc_cpu(const Dtype *patches, const int nsamples, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const CSCParameter::Boundary boundary, Dtype *blob);
template <typename Dtype>
void col2im_csc_gpu(const Dtype *patches, const int nsamples, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const CSCParameter::Boundary boundary, Dtype *blob);

// Utility function for solving the following problem:
// (DtD + lambda2*I)beta = l.
// Here DtD is truncated such that only entries in `index` is not zero.
// l, index and beta all have length of nnz.
// WARNING: very naive implementation!
template <typename Dtype>
void csc_local_inverse_naive(const int m, const Dtype lambda2, const Dtype *DtD,
    const Dtype *l, const int *index, const int nnz, Dtype *beta);

} // namespace caffe

#endif // CAFFE_CSC_HELPERS_HPP_
