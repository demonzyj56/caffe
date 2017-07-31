#include <vector>
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/csc_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/im2patches.hpp"

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
    this->blobs_.resize(2);
    vector<int> dict_shape(2);
    dict_shape[0] = channels_ * kernel_h_ * kernel_w_;
    dict_shape[1] = num_output_;
    this->blobs_[0].reset(new Blob<Dtype>(dict_shape));
    this->blobs_[1].reset(new Blob<Dtype>(vector<int>(0)));
    // fill the dictionary with initial value
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        csc_param.filler()));
    weight_filler->Fill(this->blobs_[0].get());
    FillerParameter filler_param;
    filler_param.set_type("constant");
    filler_param.set_value(static_cast<Dtype>(csc_param.lambda1()));
    weight_filler.reset(GetFiller<Dtype>(filler_param));
    weight_filler->Fill(this->blobs_[1].get());
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
  // this->alpha_ = shared_ptr<Blob<Dtype> > (new Blob<Dtype>(top_patch_shape_));
  // initialize buffer
  this->alpha_ = shared_ptr<Blob<Dtype> > (new Blob<Dtype>());
  this->alpha_buffer_ = shared_ptr<Blob<Dtype> > (new Blob<Dtype>());
  this->beta_buffer_ = shared_ptr<Blob<Dtype> > (new Blob<Dtype>());
  this->grad_buffer_ = shared_ptr<Blob<Dtype> > (new Blob<Dtype>());
  this->alpha_diff_buffer_ = shared_ptr<Blob<Dtype> > (new Blob<Dtype>());
  this->bottom_patch_buffer_ = shared_ptr<Blob<Dtype> > (new Blob<Dtype>());
  this->bottom_recon_buffer_ = shared_ptr<Blob<Dtype> > (new Blob<Dtype>());
  CHECK_EQ(this->blobs_[1]->count(), 1);
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
  // this->alpha_->Reshape(top_patch_shape_);
  //caffe_set(this->alpha_->count(), Dtype(0), this->alpha_->mutable_cpu_data());
  admm_max_rho_ = static_cast<Dtype>(this->layer_param_.csc_param().admm_max_rho());
  // reshape buffer
  this->alpha_->Reshape(top_patch_shape_);
  this->alpha_buffer_->Reshape(top_patch_shape_);
  this->beta_buffer_->Reshape(top_patch_shape_);
  this->grad_buffer_->Reshape(top_patch_shape_);
  this->alpha_diff_buffer_->Reshape(top_patch_shape_);
  this->bottom_patch_buffer_->Reshape(bottom_patch_shape_);
  this->bottom_recon_buffer_->Reshape(bottom[0]->shape());
}

template <typename Dtype>
void CSCLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
 
  // const vector<int> &bottom_shape = bottom[0]->shape();
  // Blob<Dtype> bottom_patch(bottom_patch_shape_);
  // Blob<Dtype> alpha(top_patch_shape_);
  // Blob<Dtype> grad(top_patch_shape_);
  // Blob<Dtype> bottom_recon(bottom_shape);
  // Blob<Dtype> alpha_diff(top_patch_shape_);
  // Blob<Dtype> beta(top_patch_shape_);
  Blob<Dtype> &bottom_patch = *this->bottom_patch_buffer_;
  Blob<Dtype> &alpha = *this->alpha_buffer_;
  Blob<Dtype> &grad = *this->grad_buffer_;
  Blob<Dtype> &bottom_recon = *this->bottom_recon_buffer_;
  Blob<Dtype> &alpha_diff = *this->alpha_diff_buffer_;
  Blob<Dtype> &beta = *this->beta_buffer_;
  Dtype loss = bottom[0]->sumsq_data()/2.;
  Dtype eta = admm_max_rho_;
  Dtype t = 1;
  Dtype lambda1 = this->blobs_[1]->mutable_cpu_data()[0];
//  this->extract_patches_cpu_(bottom[0], &bottom_patch);
  this->im2patches_cpu_(bottom[0], &bottom_patch, false);
  caffe_set(alpha.count(), Dtype(0), alpha.mutable_cpu_data());
  caffe_set(this->alpha_->count(), Dtype(0), this->alpha_->mutable_cpu_data());
  caffe_set(beta.count(), Dtype(0), beta.mutable_cpu_data());
  caffe_cpu_gemm(CblasTrans, CblasNoTrans, this->blobs_[0]->shape(1),
    bottom_patch.shape(1), this->blobs_[0]->shape(0), Dtype(-1),
    this->blobs_[0]->cpu_data(), bottom_patch.cpu_data(), Dtype(0),
    grad.mutable_cpu_data());
  for (int tt = 0; tt < admm_max_iter_; ++tt) {
    while (true) {
      caffe_copy(alpha.count(), this->alpha_->cpu_data(), alpha.mutable_cpu_data());
      caffe_axpy(alpha.count(), Dtype(-1./eta), grad.cpu_data(),
        alpha.mutable_cpu_data());
      this->caffe_cpu_soft_thresholding_(alpha.count(), Dtype(lambda1/eta),
        alpha.mutable_cpu_data());
      this->gemm_Dlalpha_cpu_(&alpha, &bottom_patch, true);
//      this->aggregate_patches_cpu_(&bottom_patch, &bottom_recon);
	  this->patches2im_cpu_(&bottom_patch, &bottom_recon, false);
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
	  // LOG(INFO) << "Stop: " << stop;
      if (stop >= 0) {
        Dtype t_new = (1 + std::sqrt(1+4*t*t)) / 2.;
        Dtype coeff = (t-1) / t_new;
        caffe_copy(alpha.count(), alpha.cpu_data(), this->alpha_->mutable_cpu_data());
        caffe_cpu_axpby(this->alpha_->count(), Dtype(-coeff), beta.cpu_data(),
          Dtype(1.+coeff), this->alpha_->mutable_cpu_data());
        caffe_copy(alpha.count(), alpha.cpu_data(), beta.mutable_cpu_data());
        t = t_new;
        this->gemm_Dlalpha_cpu_(this->alpha_.get(), &bottom_patch, true);
//        this->aggregate_patches_cpu_(&bottom_patch, &bottom_recon);
		this->patches2im_cpu_(&bottom_patch, &bottom_recon, false);
        caffe_sub(bottom_recon.count(), bottom[0]->cpu_data(), bottom_recon.cpu_data(),
          bottom_recon.mutable_cpu_data());
        loss = bottom_recon.sumsq_data()/2. + this->alpha_->sumsq_data()*lambda2_/2.;
//        this->extract_patches_cpu_(&bottom_recon, &bottom_patch);
		this->im2patches_cpu_(&bottom_recon, &bottom_patch, false);
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
  // LOG(INFO) << "Nonzeros per column: "
  //     << (Dtype)this->caffe_zero_norm_(beta.count(), beta.cpu_data()) / beta.shape(1)
  //     << " eta: " << eta;
}

/*
 * Backward funciton computes the following things:
 * 1) Solve beta
 * 2) Compute first part of derivative
 * 3) Compute second part of derivative
 * */
template <typename Dtype>
void CSCLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
  CPUTimer btimer;
  btimer.Start();
  // Blob<Dtype> residual(bottom_patch_shape_);
  // Blob<Dtype> Dlbeta(bottom_patch_shape_);
  // Blob<Dtype> beta(top_patch_shape_);
  // Blob<Dtype> bottom_recon(bottom[0]->shape());
  Blob<Dtype> &residual = *bottom_patch_buffer_;
  Blob<Dtype> &beta = *beta_buffer_;
  Blob<Dtype> &bottom_recon = *bottom_recon_buffer_;
  // ------------------------------------------------------------------------
  // use beta as buffer for top diff in patch view
  this->permute_num_channels_cpu_(top[0], &beta, true);
  // ------------------------------------------------------------------------
  if (this->param_propagate_down_[0]) {
  // ------------------------------------------------------------------------
    // solve beta
    // sparse_inverse(lambda2_, this->blobs_[0].get(), &beta);
    Dtype *beta_data = beta.mutable_cpu_data();
    Dtype *beta_diff = beta.mutable_cpu_diff();
    for (int i = 0; i < beta.count(); ++i) {
      if (std::fabs(beta_data[i]) < 1e-6) {
        beta_diff[i] = Dtype(0);
      }
      // beta_diff[i] /= (lambda2_ + admm_max_rho_);
    }
    // LOG(INFO) << "Backward is scaled by " << lambda2_ + admm_max_rho_ << "\n";
  // ------------------------------------------------------------------------
  // ------------------------------------------------------------------------
    //first term
    // this->gemm_Dlalpha_cpu_(&beta, &Dlbeta, true);
    this->gemm_Dlalpha_cpu_(&beta, &residual, true);
//    this->aggregate_patches_cpu_(&Dlbeta, &bottom_recon);
	// this->patches2im_cpu_(&Dlbeta, &bottom_recon, false);
	this->patches2im_cpu_(&residual, &bottom_recon, false);
    caffe_sub(bottom[0]->count(), bottom[0]->cpu_data(), bottom_recon.cpu_data(),
      bottom_recon.mutable_cpu_data());
//    this->extract_patches_cpu_(&bottom_recon, &residual);
	this->im2patches_cpu_(&bottom_recon, &residual, false);
    caffe_cpu_gemm(CblasNoTrans, CblasTrans, residual.shape(0), beta.shape(0), 
      residual.shape(1), Dtype(1), residual.cpu_data(), beta.cpu_diff(),
      Dtype(0), this->blobs_[0]->mutable_cpu_diff());
  // ------------------------------------------------------------------------
  // ------------------------------------------------------------------------
    //second term
    // this->gemm_Dlalpha_cpu_(&beta, &Dlbeta, false);
    this->gemm_Dlalpha_cpu_(&beta, &residual, false);
//    this->aggregate_patches_cpu_(&Dlbeta, &bottom_recon);
	// this->patches2im_cpu_(&Dlbeta, &bottom_recon, false);
	this->patches2im_cpu_(&residual, &bottom_recon, false);
//    this->extract_patches_cpu_(&bottom_recon, &residual);
	this->im2patches_cpu_(&bottom_recon, &residual, false);
    caffe_cpu_gemm(CblasNoTrans, CblasTrans, residual.shape(0), beta.shape(0),
      residual.shape(1), Dtype(-1), residual.cpu_data(), beta.cpu_data(),
      Dtype(1), this->blobs_[0]->mutable_cpu_diff());
    caffe_scal(this->blobs_[0]->count(), Dtype(1./bottom[0]->shape(0)),
      this->blobs_[0]->mutable_cpu_diff());
  // ------------------------------------------------------------------------
  }
  if (this->param_propagate_down_[0]) {
    LOG(FATAL) << "Backward of lambda1 is not implemented";
  }
  if (propagate_down[0]) {
    caffe_copy(bottom[0]->count(), bottom_recon.cpu_data(),
      bottom[0]->mutable_cpu_diff());
  }
  // std::cout << "Backward elapsed time: " << btimer.Seconds() << std::endl;
//  admm_max_rho_ = Dtype(1);
}

template <typename Dtype>
void CSCLayer<Dtype>::im2patches_cpu_(const Blob<Dtype> *blob, Blob<Dtype> *patches,
	bool compute_diff) {
	CHECK_EQ(patches->shape(0), blob->shape(1) * kernel_h_ * kernel_w_);
	if (boundary_ == CSCParameter::NOPAD) {
		CHECK_EQ(patches->shape(1),
			blob->shape(0) * (blob->shape(2) - kernel_h_ + 1) * (blob->shape(3) - kernel_w_ + 1));
	} else {
		CHECK_EQ(patches->shape(1),
			blob->shape(0) * blob->shape(2) * blob->shape(3));
	}
	switch (boundary_) {
	case CSCParameter::CIRCULANT_FRONT:
		im2patches_circulant_cpu(blob->cpu_data(), blob->shape(0), blob->shape(1),
			blob->shape(2), blob->shape(3), kernel_h_, kernel_w_, kernel_h_-1, kernel_w_-1,
			patches->mutable_cpu_data());
		if (compute_diff) {
			im2patches_circulant_cpu(blob->cpu_diff(), blob->shape(0), blob->shape(1),
				blob->shape(2), blob->shape(3), kernel_h_, kernel_w_, kernel_h_ - 1, kernel_w_ - 1,
				patches->mutable_cpu_diff());
		}
		break;
	case CSCParameter::CIRCULANT_BACK:
		im2patches_circulant_cpu(blob->cpu_data(), blob->shape(0), blob->shape(1),
			blob->shape(2), blob->shape(3), kernel_h_, kernel_w_, 0, 0,
			patches->mutable_cpu_data());
		if (compute_diff) {
			im2patches_circulant_cpu(blob->cpu_diff(), blob->shape(0), blob->shape(1),
				blob->shape(2), blob->shape(3), kernel_h_, kernel_w_, 0, 0,
				patches->mutable_cpu_diff());
		}
		break;
	case CSCParameter::PAD_FRONT:
		im2patches_padzeros_cpu(blob->cpu_data(), blob->shape(0), blob->shape(1),
			blob->shape(2), blob->shape(3), blob->shape(2), blob->shape(3),
			kernel_h_, kernel_w_, kernel_h_ - 1, kernel_w_ - 1,
			patches->mutable_cpu_data());
		if (compute_diff) {
			im2patches_padzeros_cpu(blob->cpu_diff(), blob->shape(0), blob->shape(1),
				blob->shape(2), blob->shape(3), blob->shape(2), blob->shape(3),
				kernel_h_, kernel_w_, kernel_h_ - 1, kernel_w_ - 1,
				patches->mutable_cpu_diff());
		}
		break;
	case CSCParameter::PAD_BACK:
		im2patches_padzeros_cpu(blob->cpu_data(), blob->shape(0), blob->shape(1),
			blob->shape(2), blob->shape(3), blob->shape(2), blob->shape(3),
			kernel_h_, kernel_w_, 0, 0,
			patches->mutable_cpu_data());
		if (compute_diff) {
			im2patches_padzeros_cpu(blob->cpu_diff(), blob->shape(0), blob->shape(1),
				blob->shape(2), blob->shape(3), blob->shape(2), blob->shape(3),
				kernel_h_, kernel_w_, 0, 0,
				patches->mutable_cpu_diff());
		}
		break;
	case CSCParameter::PAD_BOTH:
		im2patches_padzeros_cpu(blob->cpu_data(), blob->shape(0), blob->shape(1),
			blob->shape(2), blob->shape(3), blob->shape(2), blob->shape(3),
			kernel_h_, kernel_w_, (kernel_h_ - 1)/2, (kernel_w_ - 1)/2,
			patches->mutable_cpu_data());
		if (compute_diff) {
			im2patches_padzeros_cpu(blob->cpu_diff(), blob->shape(0), blob->shape(1),
				blob->shape(2), blob->shape(3), blob->shape(2), blob->shape(3),
				kernel_h_, kernel_w_, (kernel_h_ - 1) / 2, (kernel_w_ - 1) / 2,
				patches->mutable_cpu_diff());
		}
		break;
	case CSCParameter::NOPAD:
		im2patches_padzeros_cpu(blob->cpu_data(), blob->shape(0), blob->shape(1),
			blob->shape(2), blob->shape(3), blob->shape(2) - kernel_h_ + 1, 
			blob->shape(3) - kernel_w_ + 1, kernel_h_, kernel_w_, 0, 0,
			patches->mutable_cpu_data());
		if (compute_diff) {
			im2patches_padzeros_cpu(blob->cpu_diff(), blob->shape(0), blob->shape(1),
				blob->shape(2), blob->shape(3), blob->shape(2) - kernel_h_ + 1,
				blob->shape(3) - kernel_w_ + 1, kernel_h_, kernel_w_, 0, 0,
				patches->mutable_cpu_diff());
		}
		break;
	default:
		LOG(FATAL) << "Unknown boundary type";
	}
}

template <typename Dtype>
void CSCLayer<Dtype>::patches2im_cpu_(const Blob<Dtype>* patches, Blob<Dtype>* blob, 
	bool compute_diff) {
	CHECK_EQ(patches->shape(0), blob->shape(1) * kernel_h_ * kernel_w_);
	if (boundary_ == CSCParameter::NOPAD) {
		CHECK_EQ(patches->shape(1),
			blob->shape(0) * (blob->shape(2) - kernel_h_ + 1) * (blob->shape(3) - kernel_w_ + 1));
	} else {
		CHECK_EQ(patches->shape(1),
			blob->shape(0) * blob->shape(2) * blob->shape(3));
	}
	switch (boundary_) {
	case CSCParameter::CIRCULANT_FRONT:
		patches2im_circulant_cpu(patches->cpu_data(), blob->shape(0), blob->shape(1),
			blob->shape(2), blob->shape(3), kernel_h_, kernel_w_, kernel_h_-1, kernel_w_-1,
			blob->mutable_cpu_data());
		if (compute_diff) {
			patches2im_circulant_cpu(patches->cpu_diff(), blob->shape(0), blob->shape(1),
				blob->shape(2), blob->shape(3), kernel_h_, kernel_w_, kernel_h_ - 1, kernel_w_ - 1,
				blob->mutable_cpu_diff());
		}
		break;
	case CSCParameter::CIRCULANT_BACK:
		patches2im_circulant_cpu(patches->cpu_data(), blob->shape(0), blob->shape(1),
			blob->shape(2), blob->shape(3), kernel_h_, kernel_w_, 0, 0,
			blob->mutable_cpu_data());
		if (compute_diff) {
			patches2im_circulant_cpu(patches->cpu_diff(), blob->shape(0), blob->shape(1),
				blob->shape(2), blob->shape(3), kernel_h_, kernel_w_, 0, 0,
				blob->mutable_cpu_diff());
		}
		break;
	case CSCParameter::PAD_FRONT:
		patches2im_padzeros_cpu(patches->cpu_data(), blob->shape(0), blob->shape(1),
			blob->shape(2), blob->shape(3), blob->shape(2), blob->shape(3),
			kernel_h_, kernel_w_, kernel_h_ - 1, kernel_w_ - 1,
			blob->mutable_cpu_data());
		if (compute_diff) {
			patches2im_padzeros_cpu(patches->cpu_diff(), blob->shape(0), blob->shape(1),
				blob->shape(2), blob->shape(3), blob->shape(2), blob->shape(3),
				kernel_h_, kernel_w_, kernel_h_ - 1, kernel_w_ - 1,
				blob->mutable_cpu_diff());
		}
		break;
	case CSCParameter::PAD_BACK:
		patches2im_padzeros_cpu(patches->cpu_data(), blob->shape(0), blob->shape(1),
			blob->shape(2), blob->shape(3), blob->shape(2), blob->shape(3),
			kernel_h_, kernel_w_, 0, 0,
			blob->mutable_cpu_data());
		if (compute_diff) {
			patches2im_padzeros_cpu(patches->cpu_diff(), blob->shape(0), blob->shape(1),
				blob->shape(2), blob->shape(3), blob->shape(2), blob->shape(3),
				kernel_h_, kernel_w_, 0, 0,
				blob->mutable_cpu_diff());
		}
		break;
	case CSCParameter::PAD_BOTH:
		patches2im_padzeros_cpu(patches->cpu_data(), blob->shape(0), blob->shape(1),
			blob->shape(2), blob->shape(3), blob->shape(2), blob->shape(3),
			kernel_h_, kernel_w_, (kernel_h_ - 1)/2, (kernel_w_ - 1)/2,
			blob->mutable_cpu_data());
		if (compute_diff) {
			patches2im_padzeros_cpu(patches->cpu_diff(), blob->shape(0), blob->shape(1),
				blob->shape(2), blob->shape(3), blob->shape(2), blob->shape(3),
				kernel_h_, kernel_w_, (kernel_h_ - 1) / 2, (kernel_w_ - 1) / 2,
				blob->mutable_cpu_diff());
		}
		break;
	case CSCParameter::NOPAD:
		patches2im_padzeros_cpu(patches->cpu_data(), blob->shape(0), blob->shape(1),
			blob->shape(2), blob->shape(3), blob->shape(2) - kernel_h_ + 1, 
			blob->shape(3) - kernel_w_ + 1, kernel_h_, kernel_w_, 0, 0,
			blob->mutable_cpu_data());
		if (compute_diff) {
			patches2im_padzeros_cpu(patches->cpu_diff(), blob->shape(0), blob->shape(1),
				blob->shape(2), blob->shape(3), blob->shape(2) - kernel_h_ + 1,
				blob->shape(3) - kernel_w_ + 1, kernel_h_, kernel_w_, 0, 0,
				blob->mutable_cpu_diff());
		}
		break;
	default:
		LOG(FATAL) << "Unknown boundary type";
	}
	
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
void CSCLayer<Dtype>::permute_num_channels_cpu_(const Blob<Dtype> *top, Blob<Dtype> *patches,
      bool permute_diff) {
  CHECK_EQ(patches->shape(0), top->shape(1));
  CHECK_EQ(patches->shape(1), top->shape(0)*top->shape(2)*top->shape(3));
  int patch_width = top->shape(2) * top->shape(3);
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

template <typename Dtype>
void CSCLayer<Dtype>::caffe_cpu_soft_thresholding_(const int n, Dtype thresh, Dtype* x) {
	for (int i = 0; i < n; ++i) {
		x[i] = x[i] > thresh ? x[i] - thresh :
			(x[i] < -thresh ? x[i] + thresh : Dtype(0));
	}
}


#ifdef CPU_ONLY
STUB_GPU(CSCLayer);
#endif

INSTANTIATE_CLASS(CSCLayer);
REGISTER_LAYER_CLASS(CSC);

} // namespace caffe
