#include <vector>
#include <cmath>  // for std::fabs
#include "gtest/gtest.h"
#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/ufiller.hpp"
#include "caffe/util/csc_helpers.hpp"
#include "caffe/test/test_caffe_main.hpp"

namespace caffe {
  
// Currently only cpu code is implemented.
template <typename Dtype>
class SpBlobTest : public MultiDeviceTest<CPUDevice<Dtype> > {
 protected:
  SpBlobTest()
      : spblob1_(new SpBlob<Dtype>(5, 4, 5)),
      spblob2_(new SpBlob<Dtype>()), blob_(new Blob<Dtype>()) {
  }
  virtual ~SpBlobTest() {
    delete spblob1_;
    delete spblob2_;
    delete blob_;
  }
  SpBlob<Dtype> *spblob1_;
  SpBlob<Dtype> *spblob2_;
  Blob<Dtype> *blob_;
};

TYPED_TEST_CASE(SpBlobTest, TestDtypes);

TYPED_TEST(SpBlobTest, TestInitialization) {
  EXPECT_TRUE(this->spblob1_);
  EXPECT_TRUE(this->spblob2_);
  EXPECT_TRUE(this->blob_);
  EXPECT_TRUE(this->spblob1_->values_data());
  EXPECT_TRUE(this->spblob1_->rows_data());
  EXPECT_TRUE(this->spblob1_->pB_data());
  EXPECT_TRUE(this->spblob1_->pE_data());
  EXTECT_EQ(this->spblob1_->nnz(), 5);
  EXTECT_EQ(this->spblob1_->nrow(), 4);
  EXTECT_EQ(this->spblob1_->ncol(), 5);
}

TYPED_TEST(SpBlobTest, TestReshape) {
  this->spblob1_->Reshape(10, 100, 100);
  EXPECT_EQ(this->spblob1_->nnz(), 10);
  EXPECT_EQ(this->spblob1_->nrow(), 100);
  EXPECT_EQ(this->spblob1_->ncol(), 100);
}

TYPED_TEST(SpBlobTest, TestDataCopy) {
  const Dtype values[] = {1., -2., -4., -1., 5., 8., 4., 2., -3., 6., 7., 4., -5.};
  const int rows[] = {0, 1, 3, 0, 1, 4, 2, 3, 0, 2, 3, 2, 4};
  const int pB[] = {0, 3, 6, 8, 11};
  const int pE[] = {3, 6, 8, 11, 13};
  this->spblob1_->Reshape(13, 5, 5);
  this->spblob1_->CopyFrom(values, rows, pB, pE);
  for (int i = 0; i < 13; ++i) {
    EXPECT_EQ(values[i], this->spblob1_->values_data()[i]);
    EXPECT_EQ(rows[i], this->spblob1_->rows_data()[i]);
  }
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(pB[i], this->spblob1_->pB_data()[i]);
    EXPECT_EQ(pE[i], this->spblob1_->pE_data()[i]);
  }
}
TYPED_TEST(SpBlobTest, TestBlobCopy) {
  const Dtype values[] = {1., -2., -4., -1., 5., 8., 4., 2., -3., 6., 7., 4., -5.};
  const int rows[] = {0, 1, 3, 0, 1, 4, 2, 3, 0, 2, 3, 2, 4};
  const int pB[] = {0, 3, 6, 8, 11};
  const int pE[] = {3, 6, 8, 11, 13};
  this->spblob2_->Reshape(13, 5, 5);
  this->spblob2_->CopyFrom(values, rows, pB, pE);
  this->spblob1_->CopyFrom(*this->spblob2_);
  for (int i = 0; i < 13; ++i) {
    EXPECT_EQ(values[i], this->spblob1_->values_data()[i]);
    EXPECT_EQ(rows[i], this->spblob1_->rows_data()[i]);
  }
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(pB[i], this->spblob1_->pB_data()[i]);
    EXPECT_EQ(pE[i], this->spblob1_->pE_data()[i]);
  }
}

TYPED_TEST(SpBlobTest, TestToFull) {
  const Dtype values[] = {1., -2., -4., -1., 5., 8., 4., 2., -3., 6., 7., 4., -5.};
  const int rows[] = {0, 1, 3, 0, 1, 4, 2, 3, 0, 2, 3, 2, 4};
  const int pB[] = {0, 3, 6, 8, 11};
  const int pE[] = {3, 6, 8, 11, 13};
  const Dtype full_vaules[] = {1., -1., 0., -3., 0,
                               -2., 5., 0., 0., 0.,
                               0., 0., 4., 6., 4.,
                               -4. 0., 2., 7., 0.,
                                0., 8., 0., 0., -5.};
  this->spblob2_->Reshape(13, 5, 5);
  this->spblob2_->CopyFrom(values, rows, pB, pE);
  this->spblob2->ToFull(this->blob_);
  EXPECT_EQ(this->blob_->count(), 25);
  EXPECT_EQ(this->blob_->shape(0), 5);
  EXPECT_EQ(this->blob_->shape(1), 5);
  vector<int> blob_shape = this->blob_->shape();
  EXPECT_EQ(blob_shape.size(), 2);
  for (int i = 0; i < this->blob_->count(); ++i) {
    EXPECT_EQ(this->blob_->cpu_data()[i], full_values[i]);
  }
}

TYPED_TEST(SpBlobTest, TestIndexAccess) {
  const Dtype values[] = {1., -2., -4., -1., 5., 8., 4., 2., -3., 6., 7., 4., -5.};
  const int rows[] = {0, 1, 3, 0, 1, 4, 2, 3, 0, 2, 3, 2, 4};
  const int pB[] = {0, 3, 6, 8, 11};
  const int pE[] = {3, 6, 8, 11, 13};
  const Dtype full_vaules[] = {1., -1., 0., -3., 0,
                               -2., 5., 0., 0., 0.,
                               0., 0., 4., 6., 4.,
                               -4. 0., 2., 7., 0.,
                                0., 8., 0., 0., -5.};
  this->spblob2_->Reshape(13, 5, 5);
  this->spblob2_->CopyFrom(values, rows, pB, pE);
  for (int i = 0; i < this->spblob2_->nrow(); ++i) {
    for (int j = 0; j < this->spblob2_->ncol(); ++j) {
      EXPECT_EQ(this->spblob2_->at(i, j), 
        full_values[i], this->spblob2_->ncol()*i + j);
    }
  }
}

template <typename Dtype>
class CPULassoTest : public MultiDeviceTest<CPUDevice<Dtype> > {
 protected:
  CPULassoTest() 
      : X_(new Blob<Dtype>()), D_(new Blob<Dtype>>()), lambda1_(0),
        lambda2_(0), L_(0), alpha_(new Blob<Dtype>()),
        spalpha_(new SpBlob<Dtype>()) {}
  virtual ~CPULassoTest() {
    delete X_;
    delete D_;
    delete alpha_;
    delete spalpha_;
  }

  // We solve a small probelm, where X \in R^{10x5}, D \in R^{10x160},
  // \alpha \in R^{160x5}.160x5.
  virtual void SetUp() {
    vector<int> X_shape(2);
    vector<int> D_shape(2);
    vector<int> alpha_shape(2);
    X_shape[0] = 10;
    X_shape[1] = 5;
    D_shape[0] = 10;
    D_shape[1] = 160;
    alpha_shape[0] = 160;
    alpha_shape[1] = 5;
    this->X_->Reshape(X_shape);
    this->D_->Reshape(D_shape);
    this->alpha_->Reshape(alpha_shape);
    FillerParameter filler_param;
    filler_param.set_mean(0.);
    filler_param.set_std(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.fill(this->X_);
    filler.fill(this->D_);
    this->lambda1_ = 0.1;
    this->lambds2_ = 0.1;
    this->L_ = 10;
    lasso_cpu(this->X_, this->D_, this->lambda1_, this->lambda2_, this->L_,
      this->alpha_, this->spalpha_);
  }

  Blob<Dtype>* const X_;
  Blob<Dtype>* const D_;
  Dtype lambda1_;
  Dtype lambda2_;
  int L_;
  Blob<Dtype> *alpha_;
  SpBlob<Dtype> *spalpha_;
};

TYPED_TEST_CASE(CPULassoTest, TestDtypes);

TYPED_TEST(CPULassoTest, TestNothing) {
}

TYPED_TEST(CPULassoTest, TestSparseReturn) {
  EXPECT(this->spalpha_->nnz() > 0);
  EXPECT_EQ(this->spalpha_->nrow(), this->alpha_->shape(0));
  EXPECT_EQ(this->spalpha_->ncol(), this->alpha_->shape(1));
}

TYPED_TEST(CPULassoTest, TestSparseAndDenseReturnCoincide) {
  for (int i = 0; i < this->alpha_->shape(0); ++i) {
    for (int j = 0; j < this->alpha_->shape(1); ++j) {
      EXPECT_EQ(this->spalha_->at(i, j),
        this->alpha_->cpu_data()[i * this->alpha_->shape(1) + j]);
    }
  }
}

template <typename Dtype>
class CPUMatCopyTest : public MultiDeviceTest<CPUDevice<Dtype> > {
 protected:
  CPUMatCopyTest()
    : vec_from_(20), vec_to_(20), rows_(4), cols_(5) {}

  virtual ~CPUMatCopyTest() {}
  virtual void SetUp() {
    for (size_t i = 0; i < vec_from_.size(); ++i) {
      vec_from_[i] = static_cast<Dtype>(i);
    }
  }
  vector<Dtype> vec_from_;
  vector<Dtype> vec_to_;
  int rows_;
  int cols_;
};

TYPED_TEST_CASE(CPUMatCopyTest, TestDtypes);

TYPED_TEST(CPUMatCopyTest, TestInPlaceCopy) {
  caffe_cpu_imatcopy(this->rows_, this->cols_, vec_from_.data());
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      EXPECT_EQ(vec_from_[j*rows_+i], i*cols+j);
    }
  }
}

TYPED_TEST(CPUMatCopyTest, TestOutOfPlaceCopy) {
  caffe_cpu_omatcopy(this->rows_, this->cols_, vec_from_.data(), vec_to_.data());
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      EXPECT_EQ(vec_to_[j*rows_+i], vec_from_[i*cols+j]);
    }
  }
}

template <Dtype>
class CSCLocalInverseNaiveTest : public MultiDeviceTest<CPUDevice<Dtype> > {
 protected:
  CSCLocalInverseNaiveTest()
      : m_(5), nnz_(3), lambda2_(0.5), DtD_(new Dtype[25]), rhs_(new Dtype[3])
        index_(new int[3]), beta_(new Dtype[3]) {}
  virtual ~CSCLocalInverseNaiveTest() {
    delete[] DtD_;
    delete[] rhs_;
    delete[] index_;
    delete[] beta_;
  }
  virtual void SetUp() {
    for (int i = 0; i < 25; ++i) DtD_[i] = 0.;
    for (int i = 2; i < 4; ++i) {
      for (int j = 0; j < 5; ++j) {
        DtD_[i * 5 + j] = 1.;
        DtD_[j * 5 + i] = 1.;
      }
    }
    this->index_[0] = 0;
    this->index_[1] = 1;
    this->index_[2] = 4;
    this->rhs_[0] = 1.;
    this->rhs_[1] = 2.;
    this->rhs_[2] = 3.;
    csc_local_inverse_naive(this->m_, this->lambda2_, this->DtD_, this->rhs_,
      this->index_, this->nnz_, this->beta_);
  }

  int m_;
  int nnz_;
  Dtype lambda2_;
  Dtype* const DtD_;
  Dtype* const rhs_;
  int *  const index_;
  Dtype* const beta_;
};

TYPED_TEST_CASE(CSCLocalInverseNaiveTest, TestDtypes);

TYPED_TEST(CSCLocalInverseNaiveTest, TestNothing) {
}

TYPED_TEST(CSCLocalInverseNaiveTest, TestInverseResults) {
  REQUIRE_LT(std::fabs(this->beta_[0] - 2.), 1e-3);
  REQUIRE_LT(std::fabs(this->beta_[1] - 4.), 1e-3);
  REQUIRE_LT(std::fabs(this->beta_[2] - 6.), 1e-3);
}

template <Dtype>
class Im2ColCPUCirculantTest : public MultiDeviceTest<CPUDevice<Dtype> > {
 protected:
  Im2ColCPUCirculantTest()
      : blob_(new Blob<Dtype>(2, 2, 3, 4)), patches_(new Blob<Dtype>()),
      nsamples_(2), channels_(2), height_(3), width_(4), kernel_h_(3),
      kernel_w_(3), blob_recon_(new Blob<Dtype>(2, 2, 3, 4)) {}

  virtual ~Im2ColCPUCirculantTest() {
    delete blob_;
    delete patches_;
    delete blob_recon_
  }

  virtual void SetUp() {
    vector<int> patch_shape(2);
    patch_shape(1) = patches_ * channels_ * kernel_h_ * kernel_w_;
    patch_shape(2) = height_ * width_;
    patches_->Reshape(patch_shape);
    for (int i = 0; i < blob_->count(); ++i) {
      blob_->mutable_cpu_data()[i] = static_cast<Dtype>(i+1);
    }
  }

  Blob<Dtype>* const blob_;
  Blob<Dtype>* const patches_;
  Blob<Dtype>* const blob_recon_;
  int nsamples_;
  int channels_;
  int height_;
  int width_;
  int kernel_h_;
  int kernel_w_;
};

TYPED_TEST_CASE(Im2ColCPUCirculantTest, TestDtypes);

TYPED_TEST(Im2ColCPUCirculantTest, TestCirculantFront) {
  const int patches_gt[] = {
    7,  8,  5,  6, 11, 12,  9, 10,  3,  4,  1,  2,
    8,  5,  6,  7, 12,  9, 10, 11,  4,  1,  2,  3,
    5,  6,  7,  8,  9, 10, 11, 12,  1,  2,  3,  4,
   11, 12,  9, 10,  3,  4,  1,  2,  7,  8,  5,  6,
   12,  9, 10, 11,  4,  1,  2,  3,  8,  5,  6,  7,
    9, 10, 11, 12,  1,  2,  3,  4,  5,  6,  7,  8,
    3,  4,  1,  2,  7,  8,  5,  6, 11, 12,  9, 10,
    4,  1,  2,  3,  8,  5,  6,  7, 12,  9, 10, 11,
    1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12
  };
  Dtype *patch_ptr = patches_->mutable_cpu_data();
  im2col_cpu_circulant(blob_->cpu_data(), nsamples_, channels_,
    height_, width_, kernel_h_, kernel_w, height_-1, width_-1,
    patch_ptr);
  int chunk_size = kernel_h_ * kernel_w_ * height_ * width_;
  for (int i = 0; i < nsmaples_*channels_; ++i) {
    for (int j = 0; j < chunk_size; ++j) {
      EXPECT_EQ(staic_cast<int>(patch_ptr[i*chunk_size+j]),
        patches_gt[j]+i*height_*wdith_);
    }
  }
}

TYPED_TEST(Im2ColCPUCirculantTest, TestCol2ImFront) {
  const int pad_h = height_ - 1;
  const int pad_w = width_ - 1;
  im2col_cpu_circulant(blob_->cpu_data(), nsamples_, channels_,
    height_, width_, kernel_h_, kernel_w_, pad_h, pad_w,
    patches_->mutable_cpu_data());
  col2im_cpu_circulant(patches_->cpu_data(), nsamples_, channels_,
    height_, width_, kernel_h_, kernel_w_, pad_h, pad_w,
    blob_recon_->mutable_cpu_data());
  for (int i = 0; i < blob_->count(); ++i) {
    EXPECT_EQ(blob_->cpu_data()[i]*kernel_h_*kernel_w_,
      blob_recon_->cpu_data()[i]);
  }
}

} // namespace caffe
