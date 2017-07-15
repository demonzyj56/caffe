#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/csc_helpers.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "gtest/gtest.h"
#include <vector>

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
  EXPECT_EQ(this->spblob1_->nnz(), 5);
  EXPECT_EQ(this->spblob1_->nrow(), 4);
  EXPECT_EQ(this->spblob1_->ncol(), 5);
}

TYPED_TEST(SpBlobTest, TestReshape) {
  this->spblob1_->Reshape(10, 100, 100);
  EXPECT_EQ(this->spblob1_->nnz(), 10);
  EXPECT_EQ(this->spblob1_->nrow(), 100);
  EXPECT_EQ(this->spblob1_->ncol(), 100);
}

TYPED_TEST(SpBlobTest, TestDataCopy) {
  const TypeParam values[] = {1., -2., -4., -1., 5., 8., 4., 2., -3., 6., 7., 4., -5.};
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
  const TypeParam values[] = {1., -2., -4., -1., 5., 8., 4., 2., -3., 6., 7., 4., -5.};
  const int rows[] = {0, 1, 3, 0, 1, 4, 2, 3, 0, 2, 3, 2, 4};
  const int pB[] = {0, 3, 6, 8, 11};
  const int pE[] = {3, 6, 8, 11, 13};
  this->spblob2_->Reshape(13, 5, 5);
  this->spblob2_->CopyFrom(values, rows, pB, pE);
  this->spblob1_->CopyFrom(this->spblob2_);
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
  const TypeParam values[] = {1., -2., -4., -1., 5., 8., 4., 2., -3., 6., 7., 4., -5.};
  const int rows[] = {0, 1, 3, 0, 1, 4, 2, 3, 0, 2, 3, 2, 4};
  const int pB[] = {0, 3, 6, 8, 11};
  const int pE[] = {3, 6, 8, 11, 13};
  const TypeParam full_values[] = {1., -1., 0., -3., 0,
                               -2., 5., 0., 0., 0.,
                               0., 0., 4., 6., 4.,
                               -4., 0., 2., 7., 0.,
                                0., 8., 0., 0., -5.};
  this->spblob2_->Reshape(13, 5, 5);
  this->spblob2_->CopyFrom(values, rows, pB, pE);
  this->spblob2_->ToFull(this->blob_);
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
  const TypeParam values[] = {1., -2., -4., -1., 5., 8., 4., 2., -3., 6., 7., 4., -5.};
  const int rows[] = {0, 1, 3, 0, 1, 4, 2, 3, 0, 2, 3, 2, 4};
  const int pB[] = {0, 3, 6, 8, 11};
  const int pE[] = {3, 6, 8, 11, 13};
  const TypeParam full_values[] = {1., -1., 0., -3., 0,
                               -2., 5., 0., 0., 0.,
                               0., 0., 4., 6., 4.,
                               -4., 0., 2., 7., 0.,
                                0., 8., 0., 0., -5.};
  this->spblob2_->Reshape(13, 5, 5);
  this->spblob2_->CopyFrom(values, rows, pB, pE);
  for (int i = 0; i < this->spblob2_->nrow(); ++i) {
    for (int j = 0; j < this->spblob2_->ncol(); ++j) {
      EXPECT_EQ(this->spblob2_->at(i, j), 
        full_values[this->spblob2_->ncol()*i + j]);
    }
  }
}

TYPED_TEST(SpBlobTest, TestColumnAccess) {
  const TypeParam values[] = {1., -2., -4., -1., 5., 8., 4., 2., -3., 6., 7., 4., -5.};
  const int rows[] = {0, 1, 3, 0, 1, 4, 2, 3, 0, 2, 3, 2, 4};
  const int pB[] = {0, 3, 6, 8, 11};
  const int pE[] = {3, 6, 8, 11, 13};
  const TypeParam full_values[] = {1., -1., 0., -3., 0,
                               -2., 5., 0., 0., 0.,
                               0., 0., 4., 6., 4.,
                               -4., 0., 2., 7., 0.,
                                0., 8., 0., 0., -5.};
  this->spblob2_->Reshape(13, 5, 5);
  this->spblob2_->CopyFrom(values, rows, pB, pE);
  for (int i = 0; i < 5; ++i) {
    TypeParam *cols_data = this->spblob2_->values_at(i);
    int *cols_index = this->spblob2_->rows_at(i);
    int nrows = this->spblob2_->nnz_at(i);
    for (int j = 0; j < nrows; ++j) {
      TypeParam val = cols_data[j];
      int r = cols_index[j];
      EXPECT_EQ(val, full_values[r * 5 + i]);
    }
    
  }
}

} // namespace caffe
