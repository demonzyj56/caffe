#include <vector>
#include "gtest/gtest.h"
#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/csc_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"

namespace caffe {
template <typename TypeParam>
class CSCLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  CSCLayerTest()
      : blob_bottom_(new Blob<Dtype>(100, 3, 32, 32)), blob_top_(new Blob<Dtype>()) {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    // UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    this->blob_bottom_vec_.push_back(blob_bottom_);
    this->blob_top_vec_.push_back(blob_top_);
	lambda1_ = 1e-2;
	lambda2_ = 1.;
	admm_eta_ = 1.5;
	kernel_h_ = 6;
	kernel_w_ = 6;
	num_output_ = 1600;
	admm_max_iter_ = 1;
  }
  virtual ~CSCLayerTest() {
    for (size_t i = 0; i < blob_bottom_vec_.size(); ++i) {
      delete blob_bottom_vec_[i];
    }
    for (size_t i = 0; i < blob_top_vec_.size(); ++i) {
      delete blob_top_vec_[i];
    }
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  Dtype lambda1_;
  Dtype lambda2_;
  Dtype admm_eta_;
  int admm_max_iter_;
  int kernel_h_;
  int kernel_w_;
  int num_output_;
};

TYPED_TEST_CASE(CSCLayerTest, TestDtypesAndDevices);

TYPED_TEST(CSCLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CSCParameter * csc_param = layer_param.mutable_csc_param();
  csc_param->set_kernel_h(this->kernel_h_);
  csc_param->set_kernel_w(this->kernel_w_);
  csc_param->set_num_output(this->num_output_);
  shared_ptr<CSCLayer<Dtype> > layer(
    new CSCLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  vector<shared_ptr<Blob<Dtype> > > &blobs = layer->blobs();
  EXPECT_EQ(this->blob_top_->num(), 100);
  EXPECT_EQ(this->blob_top_->channels(), 1600);
  EXPECT_EQ(this->blob_top_->height(), 27);
  EXPECT_EQ(this->blob_top_->width(), 27);
  EXPECT_EQ(blobs.size(), 2);
  EXPECT_EQ(blobs[0]->shape().size(), 2);
  EXPECT_EQ(blobs[0]->shape(0), 108);
  EXPECT_EQ(blobs[0]->shape(1), 1600);
  EXPECT_EQ(blobs[1]->count(), 1);
}

TYPED_TEST(CSCLayerTest, TestForwardSanity) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CSCParameter * csc_param = layer_param.mutable_csc_param();
  csc_param->set_lambda1(this->lambda1_);
  csc_param->set_lambda2(this->lambda2_);
  csc_param->set_admm_eta(this->admm_eta_);
  csc_param->set_kernel_h(this->kernel_h_);
  csc_param->set_kernel_w(this->kernel_w_);
  csc_param->set_num_output(this->num_output_);
  csc_param->set_admm_max_iter(this->admm_max_iter_);
  csc_param->mutable_filler()->set_type("gaussian");
  csc_param->mutable_filler()->set_mean(0.);
  csc_param->mutable_filler()->set_std(1.);
  shared_ptr<CSCLayer<Dtype> > layer(
    new CSCLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
}

TYPED_TEST(CSCLayerTest, TestBackwardSanity) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CSCParameter * csc_param = layer_param.mutable_csc_param();
  csc_param->set_lambda1(this->lambda1_);
  csc_param->set_lambda2(this->lambda2_);
  csc_param->set_admm_eta(this->admm_eta_);
  csc_param->set_kernel_h(this->kernel_h_);
  csc_param->set_kernel_w(this->kernel_w_);
  csc_param->set_num_output(this->num_output_);
  csc_param->set_admm_max_iter(this->admm_max_iter_);
  csc_param->mutable_filler()->set_type("gaussian");
  csc_param->mutable_filler()->set_mean(0.);
  csc_param->mutable_filler()->set_std(1.);
  vector<bool> propagate_down(1);
  propagate_down[0] = true;
  shared_ptr<CSCLayer<Dtype> > layer(
    new CSCLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Backward(this->blob_top_vec_, propagate_down,
    this->blob_bottom_vec_);

}

} // namespace caffe
