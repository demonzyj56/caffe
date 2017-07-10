#include <vector>
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/csc_layer.hpp"
#include "caffe/util/csc_helpers.hpp"

namespace caffe {

template <typename Dtype>
void CSCLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CSCParameter csc_param = this->layer_param_.csc_param();
  lambda1_ = static_cast<Dtype>(csc_param.lambda1());
  lambda2_ = static_cast<Dtype>(csc_param.lambda2());
  kernel_h_ = static_cast<int>(csc_param.kernel_h());
  kernel_w_ = static_cast<int>(csc_param.kernel_w());
  num_output_ = static_cast<int>(csc_param.num_output());
  boundary_ = csc_param.boundary();
  admm_max_iter_ = static_cast<int>(csc_param.admm_max_iter());
  admm_max_rho_ = static_cast<Dtype>(csc_param.admm_max_rho());
  admm_eta_ = static_cast<Dtype>(csc_param.admm_eta());
  lasso_lars_L_ = static_cast<int>(csc_param.lasso_lars_L());
  channels_ = bottom[0].shape()[1];
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    // dictionary is stored here
    this->blobs_.resize(1);
    vector<int> dict_shape(2);
    dict_shape[0] = channels_ * kernel_h_ * kernel_w_;
    dict_shape[1] = num_output_;
    this->blob_[0].reset(new Blob<Dtype>(dict_shape));
    // fill the dictionary with initial value
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        csc_param.filler()));
    weight_filler->Fill(this->blobs_[0].get());
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  bottom_patch_shape_.resize(2);
  bottom_patch_shape_[0] = channels_ * kernel_h_ * kernel_w_;
  bottom_patch_shape_[1] = (boundary_ == CSCParameter::NOPAD) ?
      (bottom[0].shape(0)*(bottom[0].shape(2)-kernel_h_+1)*(bottom[0].shape(3)-kernel_w_+1)) : 
      (bottom[0].shape(0)*bottom[0].shape(2)*bottom[0].shape(3));
  top_patch_shape_.resize(2);
  top_patch_shape_[0] = num_output_;
  top_patch_shape_[1] = bottom_patch_shape_[1];
}

template <typename Dtype>
void CSCLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num_axes(), 4) << "Only 4-D bottom blob is suppoerted.";
  CHECK_EQ(bottom[0].shape(1), channels_) << "Channel size does not match.";
  vector<int> top_shape = bottom[0]->shape();
  top_shape[1] = num_output_;
  if (boundary_ == CSCParameter::NOPAD) {
      top_shape[2] -= kernel_h_ - 1;
      top_shape[3] -= kernel_w_ - 1;
  }
  top[0]->reshape(top_shape);
  // reshape buffer before forward, if changed
  bottom_patch_shape_[1] = (boundary_ == CSCParameter::NOPAD) ?
      (bottom[0].shape(0)*(bottom[0].shape(2)-kernel_h_+1)*(bottom[0].shape(3)-kernel_w_+1)) : 
      (bottom[0].shape(0)*bottom[0].shape(2)*bottom[0].shape(3));
  top_patch_shape_[1] = bottom_patch_shape_[1];
}

// Implement using ADMM.
template <typename Dtype>
void CSCLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const vector<int> &bottom_shape = bottom[0]->shape();
  int patch_size = bottom_patch_shape_[0];
  // initialize bottom_patch
  Blob<Dtype> bottom_patch(bottom_patch_shape_);
  Blob<Dtype> slice(bottom_patch_shape_);
  Blob<Dtype> dual_var(bottom_patch_shape_);
  Blob<Dtype> alpha(top_patch_shape_);
  Blob<Dtype> DLalpha_minus_u(bottom_patch_shape_);
  Blob<Dtype> bottom_recon(bottom_shape);
  im2col_csc_cpu(bottom[0]->cpu_data(), bottom_shape[0], channels_, bottom_shape[2],
    bottom_shape[2], bottom_shape[3], kernel_h_, kernel_w_, boundary_,
    bottom_patch.mutable_cpu_data());
  caffe_copy(bottom_patch.count(), bottom_patch.cpu_data(), slice.mutable_cpu_data());
  caffe_scal(slice.count(), 1./slice.shape(0), slice.mutable_cpu_data());
  caffe_set(dual_var.count(), Dtype(0), dual_var.mutable_cpu_data());
  Dtype rho = 1.;
  for (int t = 0; t < admm_max_iter_; ++t) {
    // local sparse pursuit
    caffe_axpy(slice.count(), Dtype(1), dual_var.cpu_data(), slice.mutable_cpu_data());
    lasso_cpu(&slice, &this->blob_[0], lambda1_/rho, lambda2_/rho, lasso_lars_L_, &alpha);
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, patch_size, alpha.shape(1),
      num_output_, Dtype(1), this->blob_[0].cpu_data(), alpha.cpu_data(), Dtype(0),
      DLalpha_minus_u.mutable_cpu_data());
    caffe_axpy(DLalpha_minus_u.count(), Dtype(-1), dual_var.cpu_data(),
      DLalpha_minus_u.mutable_cpu_data());
    // slice reconstruction
    caffe_cpu_scale(slice.count(), 1./rho, bottom_patch.cpu_data(), slice.mutable_cpu_data());
    caffe_axpy(slice.count(), Dtype(1), DLalpha_minus_u.cpu_data(), slice.mutable_cpu_data());
    // slice aggregation
    col2im_csc_cpu(slice.cpu_data(), bottom_shape[0], channels_,
      bottom_shape[2], bottom_shape[3], kernel_h_, kernel_w_, boundary_,
      bottom_recon.mutable_cpu_data());
    // slice update via local Laplacian, `dual_var` is only a temp variable
    im2col_csc_cpu(bottom_recon.cpu_data(), bottom_shape[0], channels_, bottom_shape[2],
      bottom_shape[3], kernel_h_, kernel_w_, boundary_, dual_var.mutable_cpu_data());
    caffe_axpy(slice.count(), -1./(rho+patch_size), dual_var.cpu_data(), slice.mutable_cpu_data());
    // dual variable update
    caffe_sub(dual_var.count(), slice.cpu_data(), DLalpha_minus_u.cpu_data(),
      dual_var.mutable_cpu_data());
    // multiplier update
    rho = (rho*admm_eta_ > admm_max_rho_) ? admm_max_rho_ : rho*admm_eta_;
  }
  // aggregate alpha to become output
  col2im_csc_cpu(alpha.cpu_data(), top[0]->shape(0), num_output_, top[0]->shape(2),
    top[0]->shape(3), kernel_h_, kernel_w_, boundary_, top[0]->mutable_cpu_data());
}

template <typename Dtype>
void CSCLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


} // namespace caffe
