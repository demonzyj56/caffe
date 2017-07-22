#include <vector>
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/csc_layer.hpp"
#include "caffe/util/csc_helpers.hpp"
#include "caffe/util/benchmark.hpp"

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
  if (!this->blobs_.empty()) {
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
  this->alpha_ = shared_ptr<Blob<Dtype> > (new Blob<Dtype>(top_patch_shape_));
  this->spalpha_ = shared_ptr<SpBlob<Dtype> >(new SpBlob<Dtype>());
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
  bottom_patch_shape_.resize(2);
  bottom_patch_shape_[0] = channels_ * kernel_h_ * kernel_w_;
  bottom_patch_shape_[1] = (boundary_ == CSCParameter::NOPAD) ?
      (bottom[0]->shape(0)*(bottom[0]->shape(2)-kernel_h_+1)*(bottom[0]->shape(3)-kernel_w_+1)) : 
      (bottom[0]->shape(0)*bottom[0]->shape(2)*bottom[0]->shape(3));
  top_patch_shape_.resize(2);
  top_patch_shape_[0] = num_output_;
  top_patch_shape_[1] = bottom_patch_shape_[1];
  this->alpha_->Reshape(top_patch_shape_);
  caffe_set(this->alpha_->count(), Dtype(0), this->alpha_->mutable_cpu_data());
}
/*
// Implement using ADMM.
template <typename Dtype>
void CSCLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CPUTimer timer;
  timer.Start();
  const vector<int> &bottom_shape = bottom[0]->shape();
  int patch_size = channels_ * kernel_h_ * kernel_w_;
  // initialize bottom_patch
  Blob<Dtype> bottom_patch(bottom_patch_shape_);
  Blob<Dtype> slice(bottom_patch_shape_);
  Blob<Dtype> dual_var(bottom_patch_shape_);
  Blob<Dtype> DLalpha_minus_u(bottom_patch_shape_);
  Blob<Dtype> bottom_recon(bottom_shape);

  this->extract_patches_cpu_(bottom[0], &bottom_patch);
  caffe_copy(bottom_patch.count(), bottom_patch.cpu_data(), slice.mutable_cpu_data());
  caffe_scal(slice.count(), Dtype(1./patch_size), slice.mutable_cpu_data());
  caffe_set(dual_var.count(), Dtype(0), dual_var.mutable_cpu_data());
  Dtype rho = 1.;
  for (int t = 0; t < admm_max_iter_; ++t) {
    // local sparse pursuit
    caffe_axpy(slice.count(), Dtype(1), dual_var.cpu_data(), slice.mutable_cpu_data());
    lasso_cpu(&slice, this->blobs_[0].get(), lambda1_/rho, lambda2_/rho, lasso_lars_L_,
      this->alpha_.get(), this->spalpha_.get());
    this->gemm_Dlalpha_cpu_(this->alpha_.get(), &DLalpha_minus_u);
    caffe_axpy(DLalpha_minus_u.count(), Dtype(-1), dual_var.cpu_data(),
      DLalpha_minus_u.mutable_cpu_data());
    // slice reconstruction
    caffe_cpu_scale(slice.count(), Dtype(1./rho), bottom_patch.cpu_data(),
      slice.mutable_cpu_data());
    caffe_axpy(slice.count(), Dtype(1), DLalpha_minus_u.cpu_data(), slice.mutable_cpu_data());
    // slice aggregation
    this->aggregate_patches_cpu_(&slice, &bottom_recon);
    // slice update via local Laplacian, `dual_var` is only a temp variable
    this->extract_patches_cpu_(&bottom_recon, &dual_var);
    caffe_axpy(slice.count(), Dtype(-1./(rho+patch_size)), dual_var.cpu_data(),
      slice.mutable_cpu_data());
    // dual variable update
    caffe_sub(dual_var.count(), slice.cpu_data(), DLalpha_minus_u.cpu_data(),
      dual_var.mutable_cpu_data());
    // multiplier update
    rho = (rho*admm_eta_ > admm_max_rho_) ? admm_max_rho_ : rho*admm_eta_;
  }
  // reorder and copy to top blob
  int patch_width = top[0]->shape(2)*top[0]->shape(3);
  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int i = 0; i < top[0]->shape(0)*top[0]->shape(1); ++i) {
    int n = i / num_output_;
    int c = i % num_output_;
    int alpha_from = c*top_patch_shape_[1] + n*patch_width;
    int top_to = i * patch_width;
    caffe_copy(patch_width, this->alpha_->cpu_data() + alpha_from,
      top[0]->mutable_cpu_data() + top_to);
  }
  LOG(INFO) << "Forward time: " << timer.Seconds() << "s\n";
}
*/
template <typename Dtype>
void CSCLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const vector<int> &bottom_shape = bottom[0]->shape();
  Blob<Dtype> bottom_patch(bottom_patch_shape_);
  Blob<Dtype> alpha(top_patch_shape_);
  Blob<Dtype> grad(top_patch_shape_);
  Blob<Dtype> bottom_recon(bottom_shape);
  Blob<Dtype> alpha_diff(top_patch_shape_);
  Blob<Dtype> beta(top_patch_shape_);
  Dtype loss = bottom[0]->sumsq_data()/2. + this->alpha_->sumsq_data()*lambda2_/2.;
  Dtype eta = 1;
  Dtype t = 1;
  this->extract_patches_cpu_(bottom[0], &bottom_patch);
  caffe_set(alpha.count(), Dtype(0), alpha.mutable_cpu_data());
  caffe_set(beta.count(), Dtype(0), beta.mutable_cpu_data());
  caffe_cpu_gemm(CblasTrans, CblasNoTrans, this->blobs_[0]->shape(1),
    bottom_patch.shape(1), this->blobs_[0]->shape(0), Dtype(-1),
    this->blobs_[0]->cpu_data(), bottom_patch.cpu_data(), Dtype(0),
    grad.mutable_cpu_data());
  caffe_axpy(grad.count(), lambda2_, this->alpha_->cpu_data(), grad.mutable_cpu_data());
  for (int tt = 0; tt < admm_max_iter_; ++tt) {
    while (true) {
      CPUTimer total_timer;
      total_timer.Start();
      caffe_copy(alpha.count(), this->alpha_->cpu_data(), alpha.mutable_cpu_data());
      caffe_axpy(alpha.count(), Dtype(-1./eta), grad.cpu_data(),
        alpha.mutable_cpu_data());
      caffe_cpu_soft_thresholding(alpha.count(), Dtype(lambda1_/eta),
        alpha.mutable_cpu_data());
      // CPUTimer timer;
      // timer.Start();
      caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, this->blobs_[0]->shape(0),
        alpha.shape(1), this->blobs_[0]->shape(1), Dtype(1),
        this->blobs_[0]->cpu_data(), alpha.cpu_data(), Dtype(0),
        bottom_patch.mutable_cpu_data());
      // timer.Stop();
      this->aggregate_patches_cpu_(&bottom_patch, &bottom_recon);
      caffe_sub(bottom[0]->count(), bottom[0]->cpu_data(), bottom_recon.cpu_data(),
        bottom_recon.mutable_cpu_data());
      Dtype loss_new = bottom_recon.sumsq_data()/2. + alpha.sumsq_data()*lambda2_/2.;
      caffe_sub(alpha.count(), alpha.cpu_data(), this->alpha_->cpu_data(),
        alpha_diff.mutable_cpu_data());
      Dtype stop = loss + caffe_cpu_dot(alpha_diff.count(), grad.cpu_data(),
        alpha_diff.cpu_data()) + alpha_diff.sumsq_data()*eta/2. - loss_new;
      // std::cout
      //   // << "Aggregation time: " << timer.Seconds()
      //   << " Total time: " << total_timer.Seconds()
      //   << " Stop: " << stop << " eta: " << eta
      //   << " obj: " << loss + lambda1_*alpha.asum_data()
      //   << " sparsity: " <<
      //     1.-(Dtype)caffe_cpu_zero_norm(this->alpha_->count(), this->alpha_->cpu_data())/alpha_->count()
      //   << std::endl;
      if (stop >= 0) {
        Dtype t_new = (1 + std::sqrt(1+4*t*t)) / 2.;
        Dtype coeff = (t-1) / t_new;
        caffe_copy(alpha.count(), alpha.cpu_data(), this->alpha_->mutable_cpu_data());
        caffe_cpu_axpby(this->alpha_->count(), Dtype(-coeff), beta.cpu_data(),
          Dtype(1.+coeff), this->alpha_->mutable_cpu_data());
        caffe_copy(alpha.count(), alpha.cpu_data(), beta.mutable_cpu_data());
        t = t_new;
        this->gemm_Dlalpha_cpu_(this->alpha_.get(), &bottom_patch, true);
        this->aggregate_patches_cpu_(&bottom_patch, &bottom_recon);
        caffe_sub(bottom_recon.count(), bottom[0]->cpu_data(), bottom_recon.cpu_data(),
          bottom_recon.mutable_cpu_data());
        loss = bottom_recon.sumsq_data()/2. + this->alpha_->sumsq_data()*lambda2_/2.;
        this->extract_patches_cpu_(&bottom_recon, &bottom_patch);
        caffe_cpu_gemm(CblasTrans, CblasNoTrans, this->blobs_[0]->shape(1),
          bottom_patch.shape(1), this->blobs_[0]->shape(0), Dtype(-1),
          this->blobs_[0]->cpu_data(), bottom_patch.cpu_data(), Dtype(0),
          grad.mutable_cpu_data());
        caffe_axpy(grad.count(), lambda2_, this->alpha_->cpu_data(),
          grad.mutable_cpu_data());
        break; 
      }
      eta *= admm_eta_;
    }
  }
  int patch_width = top[0]->shape(2)*top[0]->shape(3);
  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int i = 0; i < top[0]->shape(0)*top[0]->shape(1); ++i) {
    int n = i / num_output_;
    int c = i % num_output_;
    int alpha_from = c*top_patch_shape_[1] + n*patch_width;
    int top_to = i * patch_width;
    caffe_copy(patch_width, beta.cpu_data() + alpha_from,
      top[0]->mutable_cpu_data() + top_to);
    // caffe_copy(patch_width, this->alpha_->cpu_data() + alpha_from,
    //   top[0]->mutable_cpu_data() + top_to);
  }
  admm_max_rho_ = eta;
  LOG(INFO) << "Nonzeros per column: "
    << (Dtype)caffe_cpu_zero_norm(beta.count(), beta.cpu_data())/beta.shape(1)
    << std::endl;
}

/*
 * Backward funciton computes the following things:
 * 1) Solve beta
 * 2) Compute first part of derivative
 * 3) Compute second part of derivative
 * */
/*
template <typename Dtype>
void CSCLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
  // CPUTimer timer;
  // timer.Start();
  // construct DtD
  vector<int> DtD_shape(2);
  DtD_shape[0] = num_output_;
  DtD_shape[1] = num_output_;
  Blob<Dtype> DtD(DtD_shape);
  caffe_cpu_gemm(CblasTrans, CblasNoTrans, num_output_, num_output_,
    bottom_patch_shape_[0], Dtype(1), this->blobs_[0]->cpu_data(),
    this->blobs_[0]->cpu_data(), Dtype(0), DtD.mutable_cpu_data());
  Blob<Dtype> residual(bottom_patch_shape_);
  Blob<Dtype> Dlbeta(bottom_patch_shape_);
  Blob<Dtype> beta(top_patch_shape_);
  Blob<Dtype> bottom_recon(bottom[0]->shape());
  Blob<Dtype> dict_buffer(this->blobs_[0]->shape());
  SpBlob<Dtype> &spbeta = *this->spalpha_;
  // solve rhs
  // use beta as buffer for top diff in patch view
  int patch_width = top[0]->shape(2) * top[0]->shape(3);
  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int i = 0; i < top[0]->shape(0)*top[0]->shape(1); ++i) {
    int n = i / num_output_;
    int c = i % num_output_;
    int offset_from = i * patch_width;
    int offset_to = c*top_patch_shape_[1] + n * patch_width;
    caffe_copy(patch_width, top[0]->cpu_data() + offset_from,
      beta.mutable_cpu_data() + offset_to);
  }
  // CPUTimer other_timer;
  // other_timer.Start();
  caffe_cpu_imatcopy(beta.shape(0), beta.shape(1), beta.mutable_cpu_data());
  // LOG(INFO) << "Others time: " << other_timer.Seconds() << "s\n";
  if (this->param_propagate_down_[0]) {
    // CPUTimer inverse_timer;
    // inverse_timer.Start();
    // solve beta
    for (int i  = 0; i < spbeta.ncol(); ++i) {
      Dtype *rhs = beta.mutable_cpu_data() + i * num_output_;
      int *index = spbeta.rows_at(i);
      int nnz = spbeta.nnz_at(i);
      Dtype *lhs = spbeta.values_at(i);
      csc_local_inverse_naive(num_output_, lambda2_, DtD.mutable_cpu_data(),
        rhs, index, nnz, lhs);
    }
    // LOG(INFO) << "Matrix inverse time: " << inverse_timer.Seconds() << "s\n";
    spbeta.ToFull(&beta);
    //first term
    this->gemm_Dlalpha_cpu_(this->alpha_.get(), &Dlbeta);
    this->aggregate_patches_cpu_(&Dlbeta, &bottom_recon);
    caffe_sub(bottom[0]->count(), bottom[0]->cpu_data(), bottom_recon.cpu_data(),
      bottom_recon.mutable_cpu_data());
    this->extract_patches_cpu_(&bottom_recon, &residual);
    caffe_cpu_gemm(CblasNoTrans, CblasTrans, residual.shape(0), beta.shape(0), 
      residual.shape(1), Dtype(1), residual.cpu_data(), beta.cpu_data(),
      Dtype(0), this->blobs_[0]->mutable_cpu_diff());
    //second term
    this->gemm_Dlalpha_cpu_(&beta, &Dlbeta);
    this->aggregate_patches_cpu_(&Dlbeta, &bottom_recon);
    this->extract_patches_cpu_(&bottom_recon, &residual);
    caffe_cpu_gemm(CblasNoTrans, CblasTrans, residual.shape(0), this->alpha_->shape(0),
      residual.shape(1), Dtype(1), residual.cpu_data(), this->alpha_->cpu_data(),
      Dtype(0), dict_buffer.mutable_cpu_data());
    caffe_axpy(this->blobs_[0]->count(), Dtype(-1), dict_buffer.cpu_data(),
      this->blobs_[0]->mutable_cpu_diff());
    caffe_scal(this->blobs_[0]->count(), Dtype(1./bottom[0]->shape(0)),
      this->blobs_[0]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    caffe_copy(bottom[0]->count(), bottom_recon.cpu_data(),
      bottom[0]->mutable_cpu_diff());
  }
  // LOG(INFO) << "Backward time: " << timer.Seconds() << "s\n";
}
*/

template <typename Dtype>
void CSCLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
  CPUTimer btimer;
  btimer.Start();
  Blob<Dtype> residual(bottom_patch_shape_);
  Blob<Dtype> Dlbeta(bottom_patch_shape_);
  Blob<Dtype> beta(top_patch_shape_);
  Blob<Dtype> bottom_recon(bottom[0]->shape());
  // ------------------------------------------------------------------------
  // use beta as buffer for top diff in patch view
  this->permute_num_channels_(top[0], &beta, true);
  // ------------------------------------------------------------------------
  if (this->param_propagate_down_[0]) {
  // ------------------------------------------------------------------------
    // solve beta
    // sparse_inverse(lambda2_, this->blobs_[0].get(), &beta);
    Dtype *beta_data = beta.mutable_cpu_data();
    Dtype *beta_diff = beta.mutable_cpu_diff();
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < beta.count(); ++i) {
      if (std::fabs(beta_data[i]) < 1e-6) {
        beta_diff[i] = Dtype(0);
      }
      // beta_diff[i] /= (lambda2_ + admm_max_rho_);
    }
  // ------------------------------------------------------------------------
  // ------------------------------------------------------------------------
    //first term
    this->gemm_Dlalpha_cpu_(&beta, &Dlbeta, true);
    this->aggregate_patches_cpu_(&Dlbeta, &bottom_recon);
    caffe_sub(bottom[0]->count(), bottom[0]->cpu_data(), bottom_recon.cpu_data(),
      bottom_recon.mutable_cpu_data());
    this->extract_patches_cpu_(&bottom_recon, &residual);
    caffe_cpu_gemm(CblasNoTrans, CblasTrans, residual.shape(0), beta.shape(0), 
      residual.shape(1), Dtype(1), residual.cpu_data(), beta.cpu_diff(),
      Dtype(0), this->blobs_[0]->mutable_cpu_diff());
  // ------------------------------------------------------------------------
  // ------------------------------------------------------------------------
    //second term
    this->gemm_Dlalpha_cpu_(&beta, &Dlbeta, false);
    this->aggregate_patches_cpu_(&Dlbeta, &bottom_recon);
    this->extract_patches_cpu_(&bottom_recon, &residual);
    caffe_cpu_gemm(CblasNoTrans, CblasTrans, residual.shape(0), beta.shape(0),
      residual.shape(1), Dtype(-1), residual.cpu_data(), beta.cpu_data(),
      Dtype(1), this->blobs_[0]->mutable_cpu_diff());
    caffe_scal(this->blobs_[0]->count(), Dtype(1./bottom[0]->shape(0)),
      this->blobs_[0]->mutable_cpu_diff());
  // ------------------------------------------------------------------------
  }
  if (propagate_down[0]) {
    caffe_copy(bottom[0]->count(), bottom_recon.cpu_data(),
      bottom[0]->mutable_cpu_diff());
  }
  // std::cout << "Backward elapsed time: " << btimer.Seconds() << std::endl;
}

// TODO(leoyolo): naive impl
template <typename Dtype>
void CSCLayer<Dtype>::extract_patches_cpu_(const Blob<Dtype> *blob, Blob<Dtype> *patches) {
  int patch_width = (boundary_ == CSCParameter::NOPAD) ?
    (blob->shape(2)-kernel_h_+1) * (blob->shape(3)-kernel_w_+1) :
    blob->shape(2) * blob->shape(3);
  CHECK_EQ(patches->shape(0), channels_ * kernel_h_ * kernel_w_);
  CHECK_EQ(patches->shape(1), blob->shape(0) * patch_width);
  Blob<Dtype> patches_buffer(patches->shape());
  im2col_csc_cpu(blob->cpu_data(), blob->shape(0), blob->shape(1), blob->shape(2),
    blob->shape(3), kernel_h_, kernel_w_, boundary_, patches_buffer.mutable_cpu_data());
  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int i = 0; i < blob->shape(0)*blob->shape(1); ++i) {
    int n = i / channels_;
    int c = i % channels_;
    int source_offset = i * patch_width;
    int target_offset = c * patches->shape(1) + n * patch_width;
    caffe_copy(patch_width, patches_buffer.cpu_data() + source_offset,
      patches->mutable_cpu_data() + target_offset);
  }
}

// TODO(leoyolo): naive impl
template <typename Dtype>
void CSCLayer<Dtype>::aggregate_patches_cpu_(const Blob<Dtype> *patches, Blob<Dtype> *blob) {
  int patch_width = (boundary_ == CSCParameter::NOPAD) ?
    (blob->shape(2)-kernel_h_+1) * (blob->shape(3)-kernel_w_+1) :
    blob->shape(2) * blob->shape(3);
  CHECK_EQ(patches->shape(0), channels_ * kernel_h_ * kernel_w_);
  CHECK_EQ(patches->shape(1), blob->shape(0) * patch_width);
  Blob<Dtype> patches_buffer(patches->shape());
  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int i = 0; i < blob->shape(0)*blob->shape(1); ++i) {
    int n = i / channels_;
    int c = i % channels_;
    int source_offset = c * patches->shape(1) + n * patch_width;
    int target_offset = i * patch_width;
    caffe_copy(patch_width, patches->cpu_data() + source_offset,
      patches_buffer.mutable_cpu_data() + target_offset);
  }
  col2im_csc_cpu(patches_buffer.cpu_data(), blob->shape(0), blob->shape(1), blob->shape(2),
    blob->shape(3), kernel_h_, kernel_w_, boundary_, blob->mutable_cpu_data());
}

// No matter computed using data or diff, the output is alway to Dlalpha->cpu_data().
template <typename Dtype>
void CSCLayer<Dtype>::gemm_Dlalpha_cpu_(const Blob<Dtype> *alpha, Blob<Dtype> *Dlalpha,
      bool prefer_data) {
  CHECK_EQ(alpha->shape(0), this->blobs_[0]->shape(1));
  CHECK_EQ(alpha->shape(1), Dlalpha->shape(1));
  CHECK_EQ(Dlalpha->shape(0), this->blobs_[0]->shape(0));
  const Dtype *ptr = prefer_data ? alpha->cpu_data() : alpha->cpu_diff();
  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, this->blobs_[0]->shape(0), alpha->shape(1),
    alpha->shape(0), Dtype(1), this->blobs_[0]->cpu_data(), ptr, Dtype(0),
    Dlalpha->mutable_cpu_data());
}

// It is actually to swap the num and channels dim in top.
template <typename Dtype>
void CSCLayer<Dtype>::permute_num_channels_(const Blob<Dtype> *top, Blob<Dtype> *patches,
      bool permute_diff) {
  CHECK_EQ(patches->shape(0), top->shape(1));
  CHECK_EQ(patches->shape(1), top->shape(0)*top->shape(2)*top->shape(3));
  int patch_width = top->shape(2) * top->shape(3);
  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int i = 0; i < top->shape(0)*top->shape(1); ++i) {
    int n = i / top->shape(1);
    int c = i % top->shape(1);
    int source_offset = i * patch_width;
    int target_offset = c * patches->shape(1) + n * patch_width;
    caffe_copy(patch_width, top->cpu_data() + source_offset,
      patches->mutable_cpu_data() + target_offset);
    if (permute_diff) {
      caffe_copy(patch_width, top->cpu_diff() + source_offset,
        patches->mutable_cpu_diff() + target_offset);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(CSCLayer);
#endif

INSTANTIATE_CLASS(CSCLayer);
REGISTER_LAYER_CLASS(CSC);

} // namespace caffe
