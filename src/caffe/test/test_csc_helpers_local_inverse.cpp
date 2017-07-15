#include <vector>
#include <cmath>  // for std::fabs
#include "gtest/gtest.h"
#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/csc_helpers.hpp"
#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class CSCLocalInverseNaiveTest : public MultiDeviceTest<CPUDevice<Dtype> > {
 protected:
  CSCLocalInverseNaiveTest()
      : m_(100), nnz_(3), lambda2_(0.5), DtD_(new Dtype[10000]), rhs_(new Dtype[100]),
        index_(new int[3]), beta_(new Dtype[3]) {}
  virtual ~CSCLocalInverseNaiveTest() {
    delete[] DtD_;
    delete[] rhs_;
    delete[] index_;
    delete[] beta_;
  }
  virtual void SetUp() {
    this->index_[0] = 0;
    this->index_[1] = 1;
    this->index_[2] = 4;
    // this->rhs_[0] = 1.;
    // this->rhs_[1] = 2.;
    // this->rhs_[2] = 3.;
    caffe_set(this->m_*this->m_, Dtype(1), this->DtD_);
    caffe_set(this->m_, Dtype(100), this->rhs_);
    for (int i = 0; i < this->nnz_; ++i) {
      for (int j = 0; j < this->nnz_; ++j) {
          this->DtD_[this->index_[i] * this->m_ + this->index_[j]] = Dtype(0);
      }
      this->rhs_[this->index_[i]] = Dtype(i+1);
    }
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

TYPED_TEST(CSCLocalInverseNaiveTest, TestSetUp) {
  EXPECT_TRUE(this->DtD_);
  EXPECT_TRUE(this->rhs_);
  EXPECT_TRUE(this->index_);
  EXPECT_TRUE(this->beta_);
}


TYPED_TEST(CSCLocalInverseNaiveTest, TestInverseResults) {
  csc_local_inverse_naive(this->m_, this->lambda2_, this->DtD_, this->rhs_,
    this->index_, this->nnz_, this->beta_);
  EXPECT_LT(std::fabs(this->beta_[0] - 2.), 1e-3);
  EXPECT_LT(std::fabs(this->beta_[1] - 4.), 1e-3);
  EXPECT_LT(std::fabs(this->beta_[2] - 6.), 1e-3);
}

} // namespace caffe
