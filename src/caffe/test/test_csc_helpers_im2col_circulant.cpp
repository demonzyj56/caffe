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
class Im2ColCPUCirculantTest : public MultiDeviceTest<CPUDevice<Dtype> > {
 protected:
  Im2ColCPUCirculantTest()
      : blob_(new Blob<Dtype>(2, 2, 3, 4)), patches_(new Blob<Dtype>()),
      nsamples_(2), channels_(2), height_(3), width_(4), kernel_h_(3),
      kernel_w_(3), blob_recon_(new Blob<Dtype>(2, 2, 3, 4)) {}

  virtual ~Im2ColCPUCirculantTest() {
    delete blob_;
    delete patches_;
    delete blob_recon_;
  }

  virtual void SetUp() {
    vector<int> patch_shape(2);
    patch_shape[0] = this->nsamples_ * this->channels_ * 
      this->kernel_h_ * this->kernel_w_;
    patch_shape[1] = this->height_ * this->width_;
    this->patches_->Reshape(patch_shape);
    for (int i = 0; i < this->blob_->count(); ++i) {
      this->blob_->mutable_cpu_data()[i] = static_cast<Dtype>(i+1);
    }
  }

  Blob<Dtype>* const blob_;
  Blob<Dtype>* const patches_;
  int nsamples_;
  int channels_;
  int height_;
  int width_;
  int kernel_h_;
  int kernel_w_;
  Blob<Dtype>* const blob_recon_;
};

TYPED_TEST_CASE(Im2ColCPUCirculantTest, TestDtypes);

TYPED_TEST(Im2ColCPUCirculantTest, TestSize) {
  EXPECT_TRUE(this->blob_);
  EXPECT_TRUE(this->patches_);
  EXPECT_TRUE(this->blob_recon_);
  EXPECT_EQ(this->blob_->shape(0), this->nsamples_);
  EXPECT_EQ(this->blob_->shape(1), this->channels_);
  EXPECT_EQ(this->blob_->shape(2), this->height_);
  EXPECT_EQ(this->blob_->shape(3), this->width_);
  EXPECT_EQ(this->patches_->shape(0),
    this->nsamples_*this->channels_*this->kernel_h_*this->kernel_w_);
  EXPECT_EQ(this->patches_->shape(1), this->height_*this->width_);
  EXPECT_EQ(this->blob_recon_->shape(0), this->nsamples_);
  EXPECT_EQ(this->blob_recon_->shape(1), this->channels_);
  EXPECT_EQ(this->blob_recon_->shape(2), this->height_);
  EXPECT_EQ(this->blob_recon_->shape(3), this->width_);
}

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
  TypeParam *patch_ptr = this->patches_->mutable_cpu_data();
  im2col_cpu_circulant(this->blob_->cpu_data(), this->nsamples_*this->channels_,
    this->height_, this->width_, this->kernel_h_,
    this->kernel_w_, this->kernel_h_-1, this->kernel_w_-1,
    patch_ptr);
  int chunk_size = this->kernel_h_ * this->kernel_w_ * this->height_ * this->width_;
  for (int i = 0; i < this->nsamples_*this->channels_; ++i) {
    for (int j = 0; j < chunk_size; ++j) {
      EXPECT_EQ(static_cast<int>(patch_ptr[i*chunk_size+j]),
        patches_gt[j]+i*this->height_*this->width_);
    }
  }
}

TYPED_TEST(Im2ColCPUCirculantTest, TestCol2ImFront) {
  const int pad_h = this->kernel_h_ - 1;
  const int pad_w = this->kernel_w_ - 1;
  im2col_cpu_circulant(this->blob_->cpu_data(), this->nsamples_*this->channels_,
    this->height_, this->width_, this->kernel_h_,
    this->kernel_w_, pad_h, pad_w,
    this->patches_->mutable_cpu_data());
  col2im_cpu_circulant(this->patches_->cpu_data(), this->nsamples_*this->channels_,
    this->height_, this->width_, this->kernel_h_, this->kernel_w_, pad_h, pad_w,
    this->blob_recon_->mutable_cpu_data());
  for (int i = 0; i < this->blob_->count(); ++i) {
    EXPECT_EQ(this->blob_->cpu_data()[i]*this->kernel_h_*this->kernel_w_,
      this->blob_recon_->cpu_data()[i]);
  }
}
} // namespace caffe
