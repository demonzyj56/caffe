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
  caffe_cpu_imatcopy(this->rows_, this->cols_, this->vec_from_.data());
  for (int i = 0; i < this->rows_; ++i) {
    for (int j = 0; j < this->cols_; ++j) {
      EXPECT_EQ(this->vec_from_[j*this->rows_+i], i*this->cols_+j);
    }
  }
}

TYPED_TEST(CPUMatCopyTest, TestOutOfPlaceCopy) {
  caffe_cpu_omatcopy(this->rows_, this->cols_, this->vec_from_.data(),
    this->vec_to_.data());
  for (int i = 0; i < this->rows_; ++i) {
    for (int j = 0; j < this->cols_; ++j) {
      EXPECT_EQ(this->vec_to_[j*this->rows_+i], this->vec_from_[i*this->cols_+j]);
    }
  }
}

} // namespace caffe
