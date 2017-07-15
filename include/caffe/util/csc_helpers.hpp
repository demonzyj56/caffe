#ifndef CAFFE_CSC_HELPERS_HPP_
#define CAFFE_CSC_HELPERS_HPP_ 
/*
 * Utils for csc layer.
 */
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

// A simple data structure holding sparse matrix following CSC format.
// Notice that different from Blob<Dtype>, SpBlob<Dtype> is column major,
// which follows the Compressive Sensing Column (CSC) format.
// We don't put any data accessor here because this class for csc_layer
// mainly serves as a data storage for backward use.
template <typename Dtype>
class SpBlob {
 public:
  SpBlob()
      : values_(), rows_(), pB_(), nnz_(0), capacity_(0), nrow_(0), ncol_(0) {}
  explicit SpBlob(int nnz0, int nrow0, int ncol0);

  void Reshape(int nnz0, int nrow0, int ncol0);
  void CopyFrom(const SpBlob* other);
  void CopyFrom(const Dtype *values, const int *rows, const int *pB, const int *pE);
  void ToFull(Blob<Dtype> *full);

  const Dtype *values_data() const;
  const int *rows_data() const;
  const int *pB_data() const;
  const int *pE_data() const { return pB_data() + 1; }
  Dtype *mutable_values_data();
  int *mutable_rows_data();
  int *mutable_pB_data();
  int *mutable_pE_data() { return mutable_pB_data() + 1; }
  int nnz() const { return nnz_; }
  int nrow() const { return nrow_; }
  int ncol() const { return ncol_; }
  const Dtype at(int r, int c) const;
  Dtype *values_at(int c);
  int *rows_at(int c);
  int nnz_at(int c);
 private:
  shared_ptr<SyncedMemory> values_;
  shared_ptr<SyncedMemory> rows_;
  shared_ptr<SyncedMemory> pB_;
  int nnz_;
  int capacity_;
  int nrow_;
  int ncol_;
  DISABLE_COPY_AND_ASSIGN(SpBlob);
};


// An implementation of LASSO on CPU using SPAMS toolbox.
// The matrix X, D, alpha are all two dimensional blobs,
// where the size of X is n-by-N, D is n-by-p, 
// alpha is p-by-N.
// L specify the maximum steps of the homotopy algorithm, which could be used as 
// a stopping criteria.
// spalpha stores a sparse version of alpha.
template <typename Dtype>
void lasso_cpu(const Blob<Dtype> *X, const Blob<Dtype> *D, Dtype lambda1, Dtype lambda2,
    int L, Blob<Dtype> *alpha, SpBlob<Dtype> *spalpha);

// Utility function for transposing a matrix inplace or out of place.
// This routine is supported by OpenBLAS and MKL.
// Valid Dtype is 'float' and 'double'.
template <typename Dtype>
void caffe_cpu_imatcopy(const int rows, const int cols, Dtype *X);
template <typename Dtype>
void caffe_cpu_omatcopy(const int M, const int N, const Dtype *X, Dtype *Y);

// im2col/col2im dispatch function, supporting circulant boundary.
template <typename Dtype>
void im2col_cpu_circulant(const Dtype *blob, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, Dtype *patches);
template <typename DType>
void col2im_cpu_circulant(const DType *patches, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, DType *blob);
template <typename Dtype>
void im2col_csc_cpu(const Dtype *blob, const int nsamples, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const CSCParameter::Boundary boundary, Dtype *patches);
template <typename Dtype>
void col2im_csc_cpu(const Dtype *patches, const int nsamples, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const CSCParameter::Boundary boundary, Dtype *blob);

// Utility function for solving the following problem:
// (DtD + lambda2*I)beta = rhs.
// Here DtD is truncated such that only entries in `index` is not zero.
// index and beta all have length of nnz.
// rhs has a size of m.
// WARNING: very naive implementation!
template <typename Dtype>
void csc_local_inverse_naive(const int m, const Dtype lambda2, const Dtype *DtD,
    const Dtype *rhs, const int *index, const int nnz, Dtype *beta);

// Utility function for extracting patches.
// This essentially wraps around im2col so that I do not need to
// explicitly handle pointer directly.
template <typename Dtype>
inline void extract_patches(const Blob<Dtype> *blob, int kernel_h, int kernel_w,
    CSCParameter::Boundary boundary, Blob<Dtype> *patches) {
  CHECK_EQ(blob->shape(0)*blob->shape(1)*kernel_h*kernel_w, patches->shape(0));
  if (boundary == CSCParameter::NOPAD) {
    int patch_width = (blob->shape(2)-kernel_h+1) * (blob->shape(3)-kernel_w+1);
    CHECK_EQ(patches->shape(1), patch_width);
  } else {
    CHECK_EQ(patches->shape(1), blob->shape(2) * blob->shape(3));
  }
  im2col_csc_cpu(blob->cpu_data(), blob->shape(0), blob->shape(1), blob->shape(2),
    blob->shape(3), kernel_h, kernel_w, boundary, patches->mutable_cpu_data());
}
template void extract_patches<float>(const Blob<float> *blob, int kernel_h, int kernel_w,
    CSCParameter::Boundary boundary, Blob<float> *patches);
template void extract_patches<double>(const Blob<double> *blob, int kernel_h, int kernel_w,
    CSCParameter::Boundary boundary, Blob<double> *patches);

template <typename Dtype>
inline void aggregate_patches(const Blob<Dtype> *patches, int kernel_h, int kernel_w,
    CSCParameter::Boundary boundary, Blob<Dtype> *blob) {
  CHECK_EQ(blob->shape(0)*blob->shape(1)*kernel_h*kernel_w, patches->shape(0));
  if (boundary == CSCParameter::NOPAD) {
    int patch_width = (blob->shape(2)-kernel_h+1) * (blob->shape(3)-kernel_w+1);
    CHECK_EQ(patches->shape(1), patch_width);
  } else {
    CHECK_EQ(patches->shape(1), blob->shape(2) * blob->shape(3));
  }
  col2im_csc_cpu(patches->cpu_data(), blob->shape(0), blob->shape(1), blob->shape(2),
    blob->shape(3), kernel_h, kernel_w, boundary, blob->mutable_cpu_data());
}
template void aggregate_patches<float>(const Blob<float> *patches, int kernel_h, int kernel_w,
    CSCParameter::Boundary boundary, Blob<float> *blob);
template void aggregate_patches<double>(const Blob<double> *patches, int kernel_h, int kernel_w,
    CSCParameter::Boundary boundary, Blob<double> *blob);
 

} // namespace caffe

#endif // CAFFE_CSC_HELPERS_HPP_
