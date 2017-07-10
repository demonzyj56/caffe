#include <vector>
#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/mkl_alternate.hpp"
#include "caffe/util/csc_helpers.hpp"
#include "spams/cpp_library/cppspams.h"

namespace caffe {

template <typename Dtype>
void SpBlob<Dtype>::Reshape(int nnz0, int nrow0, int ncol0) {
  nnz_ = nnz0;
  nrow_ = nrow0;
  if (nnz0 > capacity_) {
    capacity_ = nnz0;
    values_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
    rows_.reset(new SyncedMemory(capacity_ * sizeof(int)));
  }
  if (ncol_ != ncol0) {
    ncol_ = ncol0;
    pB_.reset(new SyncedMemory((ncol_+1) * sizeof(int)));
  }
}

template <typename Dtype>
SpBlob<Dtype>::SpBlob(int nnz0, int nrow0, int ncol0)
  : capacity_(0) {
  Reshape(nnz0, nrow0, ncol0);
}

template <typename Dtype>
SpBlob<Dtype>::CopyFrom(const Dtype *values, const int *rows, const int *pB, const int *pE) {
  CHECK(values);
  CHECK(rows);
  CHECK(pB);
  CHECK(pE);
  CHECK(nnz_ <= capacity_);
  // Make sure capacity_ and SyncedMemory is the same
  if (values_->size() != capacity_ * sizeof(Dtype)) {
    values_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
  }
  if (rows_->size() != capacity_ * sizeof(int)) {
    rows_.reset(new SyncedMemory(capacity_ * sizeof(int)));
  }
  if (pB_->size() != (ncol_ + 1) * sizeof(int)) {
    pB_.reset(new SyncedMemory((ncol_+1) * sizeof(int)));
  }
  caffe_copy(nnz_, values, reinterpret_cast<Dtype *>(values_->mutable_cpu_data()));
  caffe_copy(nnz_, rows, reinterpret_cast<int *>(rows_->mutable_cpu_data()));
  caffe_copy(ncol_, pB, reinterpret_cast<int *>(pB_->mutable_cpu_data()));
  mutable_pB_data()[ncol_] = pE[ncol_-1];
}

template <typename Dtype>
void SpBlob<Dtype>::CopyFrom(const SpBlob *other) {
  Reshape(other->nnz(), other->nrow(), other->ncol());
  CopyFrom(other->values_data(), other->rows_data(), other->pB_data(), other->pE_data());
}

// Note that SpBlob is CSC format (column major) and Blob is row major.
template <typename Dtype>
void SpBlob<Dtype>::ToFull(Blob<Dtype> *full) {
  vector<int> fullshape(2);
  fullshape[0] = nrow_;
  fullshape[1] = ncol_;
  full->Reshape(fullshape);
  caffe_set(full->count(), Dtype(0), full.mutable_cpu_data());
  for (int c = 0; c < ncol_; ++i) {
    for (int row_ctr = pB_data()[c]; row_ctr != pE_data()[c]; ++row_ctr) {
      int r = mutable_rows_data()[row_ctr];
      Dtype val = mutable_values_data()[row_ctr];
      full->mutable_cpu_data()[r*ncol_+c] = val;
    }
  }
}

template <typename Dtype>
const Dtype *SpBlob<Dtype>::values_data() const {
  CHECK(values_);
  return reinterpret_cast<const Dtype *>(values_->cpu_data());
}

template <typename Dtype>
Dtype *SpBlob<Dtype>::mutable_values_data() {
  CHECK(values_);
  return reinterpret_cast<Dtype *>(values_->mutable_cpu_data());
}

template <typename Dtype>
const int *SpBlob<Dtype>::rows_data() const {
  CHECK(rows_);
  return reinterpret_cast<const int *>(rows_->cpu_data());
}

template <typename Dtype>
int *SpBlob<Dtype>::mutable_rows_data() {
  CHECK(rows_);
  return reinterpret_cast<int *>(rows_->mutable_cpu_data());
}

template <typename Dtype>
const int *SpBlob<Dtype>::pB_data() const {
  CHECK(pB_);
  return reinterpret_cast<const int *>(pB_->cpu_data());
}

template <typename Dtype>
int *SpBlob<Dtype>::mutable_pB_data() {
  CHECK(pB_);
  return reinterpret_cast<int *>(pB_->mutable_cpu_data());
}

// Note that caffe::Blob is row major (inrease last) while Matrix in spams is col major
// (increase first). We use a workaround for this problem:
template <typename Dtype>
void lasso_cpu(const Blob<Dtype> *X, const Blob<Dtype> *D, Dtype lambda1, Dtype lambda2,
      int L, Blob<Dtype> *alpha, SpBlob<Dtype> *spalpha) {
  CHECK_EQ(X->num_axes(), 2) << "Only 2D blob is allowed.";
  CHECK_EQ(D->num_axes(), 2) << "Only 2D blob is allowed.";
  CHECK_EQ(alpha->num_axes(), 2) << "Only 2D blob is allowed.";
  SyncedMemory X_trans_mem(X->count()*sizeof(Dtype));
  SyncedMemory D_trans_mem(D->count()*sizeof(Dtype));
  caffe_cpu_omatcopy(X->shape(0), X->shape(1), X->cpu_data(),
    reinterpret_cast<Dtype *>(X_trans_mem.mutable_cpu_data()));
  caffe_cpu_omatcopy(D->shape(0), D->shape(1), D->cpu_data(),
    reinterpret_cast<Dtype *>(D_trans_mem.mutable_cpu_data()));
  Matrix<Dtype> X_mat(reinterpret_cast<Dtype *>(X_trans_mem.mutable_cpu_data()),
    X->shape(0), X->shape(1));
  Matrix<Dtype> D_mat(reinterpret_cast<Dtype *>(D_trans_mem.mutable_cpu_data()),
    D->shape(0), D->shape(1));
  Matrix<Dtype> alpha_mat;
  try {
    SpMatrix<Dtype> *alpha_spmat = cppLasso(&X_mat, &D_mat, NULL, false, L, lambda1, lambda2);
  } catch (const char *err) {
    LOG(FATAL) << err;
  }
  alpha_spmat->toFullTrans(alpha_mat);
  spalpha->Reshape(alpha_spmat->nnz(), alpha_spmat->n(), alpha_spmat->m());
  spalpha->CopyFrom(alpha_spmat->v(), alpha_spmat->r(), alpha_spmat->pB(), alpha_spamt->pE());
  delete alpha_spmat;
  caffe_copy(alpha->count(), alpha_spmat.X(), alpha->mutable_cpu_data());
}

template <>
void caffe_cpu_imatcopy<float>(const int rows, const int cols, float *X) {
  int lda = cols;
  int ldb = rows;
#ifdef USE_MKL
  mkl_simatcopy('r', 't', rows, cols, float(1), X, lda, ldb);
#else
  cblas_simatcopy(CblasRowMajor, CblasTrans, rows, cols, float(1), X, lda, ldb);
#endif
}
template <>
void caffe_cpu_imatcopy<double>(const int rows, const int cols, double *X) {
  int lda = cols;
  int ldb = rows;
#ifdef USE_MKL
  mkl_dimatcopy('r', 't', rows, cols, double(1), X, lda, ldb);
#else
  cblas_dimatcopy(CblasRowMajor, CblasTrans, rows, cols, double(1), X, lda, ldb);
#endif
}
template <>
void caffe_cpu_omatcopy<float>(const int rows, const int cols, const float *X, float *Y) {
  int lda = cols;
  int ldb = rows;
#ifdef USE_MKL
  mkl_somatcopy('r', 't', rows, cols, float(1), X, lda, Y, ldb);
#else
  cblas_somatcopy(CblasRowMajor, CblasTrans, rows, cols, float(1), X, lda, Y, ldb);
#endif
}
template <>
void caffe_cpu_omatcopy<double>(const int rows, const int cols, const double *X, double *Y) {
  int lda = cols;
  int ldb = rows;
#ifdef USE_MKL
  mkl_somatcopy('r', 't', rows, cols, float(1), X, lda, Y, ldb);
#else
  cblas_somatcopy(CblasRowMajor, CblasTrans, rows, cols, float(1), X, lda, Y, ldb);
#endif
}

template <typename Dtype>
void im2col_cpu_circulant(const Dtype *blob, const int nsamples, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, Dtype *patches) {
	const int output_h = height;
	const int output_w = width;
	const int channel_size = height * width;
	// outer loop over rows of patches
	for (int n = nsamples; n; --n)
		for (int c = channels; c--; blob += channel_size) {
			for (int kernel_row = 0; kernel_row < kernel_h; ++kernel_row) {
				for (int kernel_col = 0; kernel_col < kernel_w; ++kernel_col) {
					// inner loop over cols of patches
					int input_row = (kernel_row - pad_h + height) % height;
					for (int output_rows = output_h; output_rows; output_rows--) {
						int input_col = (kernel_col - pad_w + width) % width;
						for (int output_cols = output_w; output_cols; output_cols--) {
							*(patches++) = blob[input_row * width + input_col];
							input_col = (input_col + 1) % width;
						}
						input_row = (input_row + 1) % height;
					}
				}
			}
		}
}

template <typename DType>
void col2im_cpu_circulant(const DType *patches, const int nsamples, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, DType *blob) {
	const int output_h = height;
	const int output_w = width;
	const int channel_size = height * width;
	for (int n = nsamples; n; --n)
		for (int c = channels; c--; blob += channel_size) {
			for (int kernel_row = 0; kernel_row < kernel_h; ++kernel_row) {
				for (int kernel_col = 0; kernel_col < kernel_w; ++kernel_col) {
					int input_row = (kernel_row - pad_h + height) % height;
					for (int output_rows = output_h; output_rows; output_rows--) {
						int input_col = (kernel_col - pad_w + width) % width;
						for (int output_cols = output_w; output_cols; output_cols--) {
							blob[input_row * width + input_col] += *(patches++);
							input_col = (input_col + 1) % width;
						}
						input_row = (input_row + 1) % height;
					}
				}
			}
		}
}


template <typename Dtype>
void im2col_csc_cpu(const Dtype *blob, const int nsamples, const int channels,
      const int height, const int width, const int kernel_h, const int kernel_w,
      CSCParameter::Boundary boundary, Dtype *patches) {
  int pad_h = 0;
  int pad_w = 0;
  int output_h = height;
  int output_w = width;
  switch(boundary) {
    case CSCParameter::NOPAD:
      output_h = height - kernel_h + 1;
      output_w = width - kernel_w + 1;
      break;
    case CSCParameter::PAD_FRONT:
    case CSCParameter::CIRCULANT_FRONT:
      pad_h = kernel_h - 1;
      pad_w = kernel_w - 1;
      break;
    case CSCParameter::PAD_BOTH:
      pad_h = (kernel_h - 1) / 2;
      pad_w = (kernel_w - 1) / 2;
      break;
    case CSCParameter::PAD_BACK:
    case CSCParameter::CIRCULANT_BACK:
      break;
    default:
      LOG(FATAL) << "Unknown CSC boundary type."
  }
  if (boundary == CSCParameter::CIRCULANT_BACK || boundary == CSCParameter::CIRCULANT_FRONT) {
    im2col_cpu_circulant(blob, nsamples, channels, height, width, kernel_h, kernel_w,
      pad_h, pad_w, patches);
  } else {
    im2col_cpu(blob, nsamples*channels, height, width, kernel_h, kernel_w,
      pad_h, pad_w, 1, 1, 1, 1, patches);
  }
}
template void im2col_csc_cpu<float>(const float *blob, const int nsamples, const int channels,
      const int height, const int width, const int kernel_h, const int kernel_w,
      CSCParameter::Boundary boundary, float *patches);
template void im2col_csc_cpu<double>(const double *blob, const int nsamples, const int channels,
      const int height, const int width, const int kernel_h, const int kernel_w,
      CSCParameter::Boundary boundary, double *patches);

template <typename Dtype>
void col2im_csc_cpu(const Dtype *patches, const int nsamples, const int channels,
      const int height, const int width, const int kernel_h, const int kernel_w,
      CSCParameter::Boundary boundary, Dtype *blob) {
  int pad_h = 0;
  int pad_w = 0;
  int output_h = height;
  int output_w = width;
  switch(boundary) {
    case CSCParameter::NOPAD:
      output_h = height - kernel_h + 1;
      output_w = width - kernel_w + 1;
      break;
    case CSCParameter::PAD_FRONT:
    case CSCParameter::CIRCULANT_FRONT:
      pad_h = kernel_h - 1;
      pad_w = kernel_w - 1;
      break;
    case CSCParameter::PAD_BOTH:
      pad_h = (kernel_h - 1) / 2;
      pad_w = (kernel_w - 1) / 2;
      break;
    case CSCParameter::PAD_BACK:
    case CSCParameter::CIRCULANT_BACK:
      break;
    default:
      LOG(FATAL) << "Unknown CSC boundary type."
  }
  if (boundary == CSCParameter::CIRCULANT_BACK || boundary == CSCParameter::CIRCULANT_FRONT) {
    col2im_cpu_circulant(patches, nsamples, channels, height, width, kernel_h, kernel_w,
      pad_h, pad_w, blob);
  } else {
    col2im_cpu(patches, nsamples*channels, height, width, kernel_h, kernel_w,
      pad_h, pad_w, 1, 1, 1, 1, blob);
  }
}
template void col2im_csc_cpu<float>(const float *patches, const int nsamples, const int channels,
      const int height, const int width, const int kernel_h, const int kernel_w,
      CSCParameter::Boundary boundary, float *blob);
template void col2im_csc_cpu<double>(const double *patches, const int nsamples, const int channels,
      const int height, const int width, const int kernel_h, const int kernel_w,
      CSCParameter::Boundary boundary, double *blob);

template <>
void csc_local_inverse_naive(const int m, const float lambda2, const float *DtD,
    const float *l, const int *index, const int nnz, float *beta) {
  if (nnz == 0) return;
  vector<float> DtD_shrunk(nnz*nnz);
  for (int i = 0; i < nnz; ++i) {
	for (in j = 0; j < nnz; ++j) {
	  DtD_shrunk[i*nnz + j] = DtD[index[i]*m + index[j]];
	  if (j == i) {
		DtD_shrunk[i*nnz + j] += lambda2;
	  }
	}
    // beta[i] = l[index[i]];
  }
  caffe_copy(nnz, l, beta);
  int info = LAPACKE_spotrf(LAPACK_ROW_MAJOR, 'L', nnz, DtD_shrunk.data(), nnz);
  CHECK_EQ(info, 0) << "Cholesky factorization is not successful!";
  info = LAPACKE_spotrs(LAPACK_ROW_MAJOR, 'L', nnz, nnz, DtD_shrunk.data(), beta, nnz, nnz);
  CHECK_EQ(info, 0) << "Solving linear system is not successful!";
}
template <>
void csc_local_inverse_naive(const int m, const double lambda2, const double *DtD,
    const double *l, const int *index, const int nnz, double *beta) {
  if (nnz == 0) return;
  vector<double> DtD_shrunk(nnz*nnz);
  for (int i = 0; i < nnz; ++i) {
	for (in j = 0; j < nnz; ++j) {
	  DtD_shrunk[i*nnz + j] = DtD[index[i]*m + index[j]];
	  if (j == i) {
		DtD_shrunk[i*nnz + j] += lambda2;
	  }
	}
    // beta[i] = l[index[i]];
  }
  caffe_copy(nnz, l, beta);
  int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', nnz, DtD_shrunk.data(), nnz);
  CHECK_EQ(info, 0) << "Cholesky factorization is not successful!";
  info = LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', nnz, nnz, DtD_shrunk.data(), beta, nnz, nnz);
  CHECK_EQ(info, 0) << "Solving linear system is not successful!";
}

} // namespace caffe
