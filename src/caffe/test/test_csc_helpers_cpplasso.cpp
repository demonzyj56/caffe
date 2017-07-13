#include <vector>
#include "gtest/gtest.h"
#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/csc_helpers.hpp"
#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class CPULassoTest : public MultiDeviceTest<CPUDevice<Dtype> > {
 protected:
  CPULassoTest() 
      : X_(new Blob<Dtype>()), D_(new Blob<Dtype>()), lambda1_(0),
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
    filler.Fill(this->X_);
    filler.Fill(this->D_);
    this->lambda1_ = 0.1;
    this->lambda2_ = 0.1;
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
  EXPECT_GT(this->spalpha_->nnz(), 0);
  EXPECT_EQ(this->spalpha_->nrow(), this->alpha_->shape(0));
  EXPECT_EQ(this->spalpha_->ncol(), this->alpha_->shape(1));
}

TYPED_TEST(CPULassoTest, TestSparseAndDenseReturnCoincide) {
  for (int i = 0; i < this->alpha_->shape(0); ++i) {
    for (int j = 0; j < this->alpha_->shape(1); ++j) {
      EXPECT_EQ(this->spalpha_->at(i, j),
        this->alpha_->cpu_data()[i * this->alpha_->shape(1) + j]);
    }
  }
}

} // namespace caffe
