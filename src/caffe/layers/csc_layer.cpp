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
  lasso_lars_L_ = static_cast<int>(csc_param.lasso_lars_l());
  channels_ = bottom[0]->shape(1);
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    // dictionary is stored here
    this->blobs_.resize(1);
    vector<int> dict_shape(2);
    dict_shape[0] = channels_ * kernel_h_ * kernel_w_;
    dict_shape[1] = num_output_;
    this->blobs_[0].reset(new Blob<Dtype>(dict_shape));
    // fill the dictionary with initial value
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        csc_param.filler()));
    weight_filler->Fill(this->blobs_[0].get());
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  bottom_patch_shape_.resize(2);
  bottom_patch_shape_[0] = channels_ * kernel_h_ * kernel_w_;
  bottom_patch_shape_[1] = (boundary_ == CSCParameter::NOPAD) ?
      (bottom[0]->shape(0)*(bottom[0]->shape(2)-kernel_h_+1)*(bottom[0]->shape(3)-kernel_w_+1)) : 
      (bottom[0]->shape(0)*bottom[0]->shape(2)*bottom[0]->shape(3));
  top_patch_shape_.resize(2);
  top_patch_shape_[0] = num_output_;
  top_patch_shape_[1] = bottom_patch_shape_[1];
}

template <typename Dtype>
void CSCLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num_axes(), 4) << "Only 4-D bottom blob is suppoerted.";
  CHECK_EQ(bottom[0]->shape(1), channels_) << "Channel size does not match.";
  vector<int> top_shape = bottom[0]->shape();
  top_shape[1] = num_output_;
  if (boundary_ == CSCParameter::NOPAD) {
      top_shape[2] -= kernel_h_ - 1;
      top_shape[3] -= kernel_w_ - 1;
  }
  top[0]->Reshape(top_shape);
  // reshape buffer before forward, if changed
  bottom_patch_shape_[1] = (boundary_ == CSCParameter::NOPAD) ?
      (bottom[0]->shape(0)*(bottom[0]->shape(2)-kernel_h_+1)*(bottom[0]->shape(3)-kernel_w_+1)) : 
      (bottom[0]->shape(0)*bottom[0]->shape(2)*bottom[0]->shape(3));
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
  im2col_csc_cpu(bottom[0]->cpu_data(), bottom_shape[0], channels_,
    bottom_shape[2], bottom_shape[3], kernel_h_, kernel_w_, boundary_,
    bottom_patch.mutable_cpu_data());
  caffe_copy(bottom_patch.count(), bottom_patch.cpu_data(), slice.mutable_cpu_data());
  caffe_scal(slice.count(), Dtype(1./slice.shape(0)), slice.mutable_cpu_data());
  caffe_set(dual_var.count(), Dtype(0), dual_var.mutable_cpu_data());
  Dtype rho = 1.;
  for (int t = 0; t < admm_max_iter_; ++t) {
    // local sparse pursuit
    caffe_axpy(slice.count(), Dtype(1), dual_var.cpu_data(), slice.mutable_cpu_data());
    lasso_cpu(&slice, this->blobs_[0].get(), lambda1_/rho, lambda2_/rho, lasso_lars_L_,
      &alpha, &this->spalpha_);
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, patch_size, alpha.shape(1),
      num_output_, Dtype(1), this->blobs_[0]->cpu_data(), alpha.cpu_data(), Dtype(0),
      DLalpha_minus_u.mutable_cpu_data());
    caffe_axpy(DLalpha_minus_u.count(), Dtype(-1), dual_var.cpu_data(),
      DLalpha_minus_u.mutable_cpu_data());
    // slice reconstruction
    caffe_cpu_scale(slice.count(), Dtype(1./rho), bottom_patch.cpu_data(), slice.mutable_cpu_data());
    caffe_axpy(slice.count(), Dtype(1), DLalpha_minus_u.cpu_data(), slice.mutable_cpu_data());
    // slice aggregation
    col2im_csc_cpu(slice.cpu_data(), bottom_shape[0], channels_,
      bottom_shape[2], bottom_shape[3], kernel_h_, kernel_w_, boundary_,
      bottom_recon.mutable_cpu_data());
    // slice update via local Laplacian, `dual_var` is only a temp variable
    im2col_csc_cpu(bottom_recon.cpu_data(), bottom_shape[0], channels_, bottom_shape[2],
      bottom_shape[3], kernel_h_, kernel_w_, boundary_, dual_var.mutable_cpu_data());
    caffe_axpy(slice.count(), Dtype(-1./(rho+patch_size)), dual_var.cpu_data(),
      slice.mutable_cpu_data());
    // dual variable update
    caffe_sub(dual_var.count(), slice.cpu_data(), DLalpha_minus_u.cpu_data(),
      dual_var.mutable_cpu_data());
    // multiplier update
    rho = (rho*admm_eta_ > admm_max_rho_) ? admm_max_rho_ : rho*admm_eta_;
  }
  // alpha is the output, simple do a copy
  caffe_copy(top[0]->count(), alpha.cpu_data(), top[0]->mutable_cpu_data());
}

template <typename Dtype>
void CSCLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
  vector<int> DtD_shape(2);
  DtD_shape[0] = num_output_;
  DtD_shape[1] = num_output_;
  Blob<Dtype> DtD(DtD_shape);
  caffe_cpu_gemm(CblasTrans, CblasNoTrans, num_output_, num_output_,
    bottom_patch_shape_[0], Dtype(1), this->blobs_[0]->cpu_data(),
    this->blobs_[0]->cpu_data(), Dtype(1), DtD.mutable_cpu_data());
  SpBlob<Dtype> spbeta;
  Blob<Dtype> beta(top_patch_shape_);
  spbeta.CopyFrom(&spalpha_);
  if (this->param_propagate_down_[0]) {
    // compute beta
    for (int i = 0; i < spalpha_.ncol(); ++i) {
      const Dtype *rhs = spalpha_.values_data() + spalpha_.pB_data()[i];
      const int *index = spalpha_.rows_data() + spalpha_.pB_data()[i];
      int nnz = spalpha_.pE_data()[i] - spalpha_.pB_data()[i];
      Dtype *lhs = spbeta.mutable_values_data() + spalpha_.pB_data()[i];
      csc_local_inverse_naive(num_output_, lambda2_, DtD.mutable_cpu_data(),
        rhs, index, nnz, lhs);
    }
    spbeta.ToFull(&beta);
    // compute derivative: first part
    Dtype *dict_diff = this->blobs_[0]->mutable_cpu_diff();
    caffe_set(this->blobs_[0]->count(), Dtype(0), dict_diff);
    Blob<Dtype> residual(bottom_patch_shape_);
    Blob<Dtype> bottom_recon(bottom[0]->shape());
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, bottom_patch_shape_[0],
      bottom_patch_shape_[1], num_output_, Dtype(1), this->blobs_[0]->cpu_data(),
      top[0]->cpu_data(), Dtype(1), residual.mutable_cpu_data());
    col2im_csc_cpu(residual.cpu_data(), bottom_recon.shape(0), bottom_recon.shape(1),
      bottom_recon.shape(2), bottom_recon.shape(3), kernel_h_, kernel_w_, boundary_,
      bottom_recon.mutable_cpu_data());
    caffe_sub(bottom_recon.count(), bottom[0]->cpu_data(), bottom_recon.cpu_data(),
      bottom_recon.mutable_cpu_data());
    im2col_csc_cpu(bottom_recon.cpu_data(), bottom_recon.shape(0), bottom_recon.shape(1),
      bottom_recon.shape(2), bottom_recon.shape(3), kernel_h_, kernel_w_, boundary_,
      residual.mutable_cpu_data());
    caffe_cpu_gemm(CblasNoTrans, CblasTrans, this->blobs_[0]->shape(0), this->blobs_[0]->shape(1),
      residual.shape(1), Dtype(1), residual.cpu_data(), beta.cpu_data(), Dtype(1), dict_diff);
    // compute derivative: second part
    Blob<Dtype> &Dlxbeta = residual;
    Blob<Dtype> &beta_recon = bottom_recon;
    Blob<Dtype> dict_buffer(this->blobs_[0]->shape());
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, bottom_patch_shape_[0],
      bottom_patch_shape_[1], num_output_, Dtype(1), this->blobs_[0]->cpu_data(),
      beta.cpu_data(), Dtype(1), Dlxbeta.mutable_cpu_data());
    col2im_csc_cpu(Dlxbeta.cpu_data(), beta_recon.shape(0), beta_recon.shape(1),
      beta_recon.shape(2), beta_recon.shape(3), kernel_h_, kernel_w_, boundary_,
      beta_recon.mutable_cpu_data());
    im2col_csc_cpu(beta_recon.cpu_data(), beta_recon.shape(0), beta_recon.shape(1),
      beta_recon.shape(2), beta_recon.shape(3), kernel_h_, kernel_w_, boundary_,
      Dlxbeta.mutable_cpu_data());
    caffe_cpu_gemm(CblasNoTrans, CblasTrans, dict_buffer.shape(0), dict_buffer.shape(1),
      Dlxbeta.shape(1), Dtype(1), Dlxbeta.cpu_data(), top[0]->cpu_data(), Dtype(1),
      dict_buffer.mutable_cpu_data());
    caffe_sub(this->blobs_[0]->count(), dict_diff, dict_buffer.cpu_data(), dict_diff);
    caffe_scal(this->blobs_[0]->count(), Dtype(1./bottom[0]->shape(0)), dict_diff);
  }
  if (propagate_down[0]) {
    NOT_IMPLEMENTED;
  }
}

#ifdef CPU_ONLY
STUB_GPU(CSCLayer);
#endif

INSTANTIATE_CLASS(CSCLayer);
REGISTER_LAYER_CLASS(CSC);

} // namespace caffe
