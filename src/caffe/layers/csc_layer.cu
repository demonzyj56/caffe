#include <vector>

#include "caffe/layers/csc_layer.hpp"
#include "caffe/util/im2patches.hpp"

namespace caffe {

template <typename Dtype>
void CSCLayer<Dtype>::im2patches_gpu_(const Blob<Dtype> *blob, Blob<Dtype> *patches,
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
		im2patches_circulant_gpu(blob->gpu_data(), blob->shape(0), blob->shape(1),
			blob->shape(2), blob->shape(3), kernel_h_, kernel_w_, kernel_h_-1, kernel_w_-1,
			patches->mutable_gpu_data());
		if (compute_diff) {
			im2patches_circulant_gpu(blob->gpu_diff(), blob->shape(0), blob->shape(1),
				blob->shape(2), blob->shape(3), kernel_h_, kernel_w_, kernel_h_ - 1, kernel_w_ - 1,
				patches->mutable_gpu_diff());
		}
		break;
	case CSCParameter::CIRCULANT_BACK:
		im2patches_circulant_gpu(blob->gpu_data(), blob->shape(0), blob->shape(1),
			blob->shape(2), blob->shape(3), kernel_h_, kernel_w_, 0, 0,
			patches->mutable_gpu_data());
		if (compute_diff) {
			im2patches_circulant_gpu(blob->gpu_diff(), blob->shape(0), blob->shape(1),
				blob->shape(2), blob->shape(3), kernel_h_, kernel_w_, 0, 0,
				patches->mutable_gpu_diff());
		}
		break;
	case CSCParameter::PAD_FRONT:
		im2patches_padzeros_gpu(blob->gpu_data(), blob->shape(0), blob->shape(1),
			blob->shape(2), blob->shape(3), blob->shape(2), blob->shape(3),
			kernel_h_, kernel_w_, kernel_h_ - 1, kernel_w_ - 1,
			patches->mutable_gpu_data());
		if (compute_diff) {
			im2patches_padzeros_gpu(blob->gpu_diff(), blob->shape(0), blob->shape(1),
				blob->shape(2), blob->shape(3), blob->shape(2), blob->shape(3),
				kernel_h_, kernel_w_, kernel_h_ - 1, kernel_w_ - 1,
				patches->mutable_gpu_diff());
		}
		break;
	case CSCParameter::PAD_BACK:
		im2patches_padzeros_gpu(blob->gpu_data(), blob->shape(0), blob->shape(1),
			blob->shape(2), blob->shape(3), blob->shape(2), blob->shape(3),
			kernel_h_, kernel_w_, 0, 0,
			patches->mutable_gpu_data());
		if (compute_diff) {
			im2patches_padzeros_gpu(blob->gpu_diff(), blob->shape(0), blob->shape(1),
				blob->shape(2), blob->shape(3), blob->shape(2), blob->shape(3),
				kernel_h_, kernel_w_, 0, 0,
				patches->mutable_gpu_diff());
		}
		break;
	case CSCParameter::PAD_BOTH:
		im2patches_padzeros_gpu(blob->gpu_data(), blob->shape(0), blob->shape(1),
			blob->shape(2), blob->shape(3), blob->shape(2), blob->shape(3),
			kernel_h_, kernel_w_, (kernel_h_ - 1)/2, (kernel_w_ - 1)/2,
			patches->mutable_gpu_data());
		if (compute_diff) {
			im2patches_padzeros_gpu(blob->gpu_diff(), blob->shape(0), blob->shape(1),
				blob->shape(2), blob->shape(3), blob->shape(2), blob->shape(3),
				kernel_h_, kernel_w_, (kernel_h_ - 1) / 2, (kernel_w_ - 1) / 2,
				patches->mutable_gpu_diff());
		}
		break;
	case CSCParameter::NOPAD:
		im2patches_padzeros_gpu(blob->gpu_data(), blob->shape(0), blob->shape(1),
			blob->shape(2), blob->shape(3), blob->shape(2) - kernel_h_ + 1, 
			blob->shape(3) - kernel_w_ + 1, kernel_h_, kernel_w_, 0, 0,
			patches->mutable_gpu_data());
		if (compute_diff) {
			im2patches_padzeros_gpu(blob->gpu_diff(), blob->shape(0), blob->shape(1),
				blob->shape(2), blob->shape(3), blob->shape(2) - kernel_h_ + 1,
				blob->shape(3) - kernel_w_ + 1, kernel_h_, kernel_w_, 0, 0,
				patches->mutable_gpu_diff());
		}
		break;
	default:
		LOG(FATAL) << "Unknown boundary type";
	}
}

template <typename Dtype>
void CSCLayer<Dtype>::patches2im_gpu_(const Blob<Dtype>* patches, Blob<Dtype>* blob, 
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
		patches2im_circulant_gpu(patches->gpu_data(), blob->shape(0), blob->shape(1),
			blob->shape(2), blob->shape(3), kernel_h_, kernel_w_, kernel_h_-1, kernel_w_-1,
			blob->mutable_gpu_data());
		if (compute_diff) {
			patches2im_circulant_gpu(patches->gpu_diff(), blob->shape(0), blob->shape(1),
				blob->shape(2), blob->shape(3), kernel_h_, kernel_w_, kernel_h_ - 1, kernel_w_ - 1,
				blob->mutable_gpu_diff());
		}
		break;
	case CSCParameter::CIRCULANT_BACK:
		patches2im_circulant_gpu(patches->gpu_data(), blob->shape(0), blob->shape(1),
			blob->shape(2), blob->shape(3), kernel_h_, kernel_w_, 0, 0,
			blob->mutable_gpu_data());
		if (compute_diff) {
			patches2im_circulant_gpu(patches->gpu_diff(), blob->shape(0), blob->shape(1),
				blob->shape(2), blob->shape(3), kernel_h_, kernel_w_, 0, 0,
				blob->mutable_gpu_diff());
		}
		break;
	case CSCParameter::PAD_FRONT:
		patches2im_padzeros_gpu(patches->gpu_data(), blob->shape(0), blob->shape(1),
			blob->shape(2), blob->shape(3), blob->shape(2), blob->shape(3),
			kernel_h_, kernel_w_, kernel_h_ - 1, kernel_w_ - 1,
			blob->mutable_gpu_data());
		if (compute_diff) {
			patches2im_padzeros_gpu(patches->gpu_diff(), blob->shape(0), blob->shape(1),
				blob->shape(2), blob->shape(3), blob->shape(2), blob->shape(3),
				kernel_h_, kernel_w_, kernel_h_ - 1, kernel_w_ - 1,
				blob->mutable_gpu_diff());
		}
		break;
	case CSCParameter::PAD_BACK:
		patches2im_padzeros_gpu(patches->gpu_data(), blob->shape(0), blob->shape(1),
			blob->shape(2), blob->shape(3), blob->shape(2), blob->shape(3),
			kernel_h_, kernel_w_, 0, 0,
			blob->mutable_gpu_data());
		if (compute_diff) {
			patches2im_padzeros_gpu(patches->gpu_diff(), blob->shape(0), blob->shape(1),
				blob->shape(2), blob->shape(3), blob->shape(2), blob->shape(3),
				kernel_h_, kernel_w_, 0, 0,
				blob->mutable_gpu_diff());
		}
		break;
	case CSCParameter::PAD_BOTH:
		patches2im_padzeros_gpu(patches->gpu_data(), blob->shape(0), blob->shape(1),
			blob->shape(2), blob->shape(3), blob->shape(2), blob->shape(3),
			kernel_h_, kernel_w_, (kernel_h_ - 1)/2, (kernel_w_ - 1)/2,
			blob->mutable_gpu_data());
		if (compute_diff) {
			patches2im_padzeros_gpu(patches->gpu_diff(), blob->shape(0), blob->shape(1),
				blob->shape(2), blob->shape(3), blob->shape(2), blob->shape(3),
				kernel_h_, kernel_w_, (kernel_h_ - 1) / 2, (kernel_w_ - 1) / 2,
				blob->mutable_gpu_diff());
		}
		break;
	case CSCParameter::NOPAD:
		patches2im_padzeros_gpu(patches->gpu_data(), blob->shape(0), blob->shape(1),
			blob->shape(2), blob->shape(3), blob->shape(2) - kernel_h_ + 1, 
			blob->shape(3) - kernel_w_ + 1, kernel_h_, kernel_w_, 0, 0,
			blob->mutable_gpu_data());
		if (compute_diff) {
			patches2im_padzeros_gpu(patches->gpu_diff(), blob->shape(0), blob->shape(1),
				blob->shape(2), blob->shape(3), blob->shape(2) - kernel_h_ + 1,
				blob->shape(3) - kernel_w_ + 1, kernel_h_, kernel_w_, 0, 0,
				blob->mutable_gpu_diff());
		}
		break;
	default:
		LOG(FATAL) << "Unknown boundary type";
	}
	
}

template <typename Dtype>
void CSCLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const vector<int> &bottom_shape = bottom[0]->shape();
  Blob<Dtype> bottom_patch(bottom_patch_shape_);
  Blob<Dtype> alpha(top_patch_shape_);
  Blob<Dtype> grad(top_patch_shape_);
  Blob<Dtype> bottom_recon(bottom_shape);
  Blob<Dtype> alpha_diff(top_patch_shape_);
  Blob<Dtype> beta(top_patch_shape_);
  Dtype loss = bottom[0]->sumsq_data()/2.;
  Dtype eta = admm_max_rho_;
  Dtype t = 1;
  this->im2patches_gpu_(bottom[0], &bottom_patch, false);
//  caffe_gpu_set(alpha.count(), Dtype(0), alpha.mutable_gpu_data());
  caffe_gpu_set(beta.count(), Dtype(0), beta.mutable_gpu_data());
  caffe_gpu_gemm(CblasTrans, CblasNoTrans, this->blobs_[0]->shape(1),
    bottom_patch.shape(1), this->blobs_[0]->shape(0), Dtype(-1),
    this->blobs_[0]->gpu_data(), bottom_patch.gpu_data(), Dtype(0),
    grad.mutable_gpu_data());
  for (int tt = 0; tt < admm_max_iter_; ++tt) {
    while (true) {
      caffe_copy(alpha.count(), this->alpha_->gpu_data(), alpha.mutable_gpu_data());
      caffe_gpu_axpy(alpha.count(), Dtype(-1./eta), grad.gpu_data(),
        alpha.mutable_gpu_data());
      this->caffe_gpu_soft_thresholding_(alpha.count(), Dtype(lambda1_/eta),
        alpha.mutable_gpu_data());
      this->gemm_Dlalpha_gpu_(&alpha, &bottom_patch, true);
      this->patches2im_gpu_(&bottom_patch, &bottom_recon, false);
      caffe_gpu_sub(bottom[0]->count(), bottom[0]->gpu_data(), bottom_recon.gpu_data(),
        bottom_recon.mutable_gpu_data());
      Dtype loss_new = bottom_recon.sumsq_data()/2. + alpha.sumsq_data()*lambda2_/2.;
      caffe_gpu_sub(alpha.count(), alpha.gpu_data(), this->alpha_->gpu_data(),
        alpha_diff.mutable_gpu_data());
      Dtype dot_val = 0.;
      caffe_gpu_dot(alpha_diff.count(), grad.gpu_data(), alpha_diff.gpu_data(), &dot_val);
      Dtype stop = loss + dot_val  + alpha_diff.sumsq_data()*eta/2. - loss_new;
	  LOG(INFO) << "Stop: " << stop;
      if (stop >= 0) {
        Dtype t_new = (1 + std::sqrt(1+4*t*t)) / 2.;
        Dtype coeff = (t-1) / t_new;
        caffe_copy(alpha.count(), alpha.gpu_data(), this->alpha_->mutable_gpu_data());
        caffe_gpu_axpby(this->alpha_->count(), Dtype(-coeff), beta.gpu_data(),
          Dtype(1.+coeff), this->alpha_->mutable_gpu_data());
        caffe_copy(alpha.count(), alpha.gpu_data(), beta.mutable_gpu_data());
        t = t_new;
        this->gemm_Dlalpha_gpu_(this->alpha_.get(), &bottom_patch, true);
        this->patches2im_gpu_(&bottom_patch, &bottom_recon, false);
        caffe_gpu_sub(bottom_recon.count(), bottom[0]->gpu_data(), bottom_recon.gpu_data(),
          bottom_recon.mutable_gpu_data());
        loss = bottom_recon.sumsq_data()/2. + this->alpha_->sumsq_data()*lambda2_/2.;
        this->im2patches_gpu_(&bottom_recon, &bottom_patch, false);
        caffe_gpu_gemm(CblasTrans, CblasNoTrans, this->blobs_[0]->shape(1),
          bottom_patch.shape(1), this->blobs_[0]->shape(0), Dtype(-1),
          this->blobs_[0]->gpu_data(), bottom_patch.gpu_data(), Dtype(0),
          grad.mutable_gpu_data());
        caffe_gpu_axpy(grad.count(), lambda2_, this->alpha_->gpu_data(),
          grad.mutable_gpu_data());
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
    caffe_copy(patch_width, beta.gpu_data() + alpha_from,
      top[0]->mutable_gpu_data() + top_to);
  }
  admm_max_rho_ = eta;
  LOG(INFO) << "Nonzeros per column: "
    << (Dtype)this->caffe_zero_norm_(beta.count(), beta.cpu_data())/beta.shape(1)
    << " eta: " << eta;
}

template <typename Dtype>
__global__ void set_if_kernel(const int n, const Dtype *data, Dtype *diff) {
  CUDA_KERNEL_LOOP(index, n) {
    if (std::fabs(data[index]) < 1e-6) {
      diff[index] = Dtype(0);
    }
  }
}

template <typename Dtype>
void CSCLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
  Blob<Dtype> residual(bottom_patch_shape_);
  Blob<Dtype> Dlbeta(bottom_patch_shape_);
  Blob<Dtype> beta(top_patch_shape_);
  Blob<Dtype> bottom_recon(bottom[0]->shape());
  // ------------------------------------------------------------------------
  // use beta as buffer for top diff in patch view
  this->permute_num_channels_gpu_(top[0], &beta, true);
  // ------------------------------------------------------------------------
  if (this->param_propagate_down_[0]) {
  // ------------------------------------------------------------------------
    // solve beta
    Dtype *beta_data = beta.mutable_gpu_data();
    Dtype *beta_diff = beta.mutable_gpu_diff();
    set_if_kernel<Dtype><<<CAFFE_GET_BLOCKS(beta.count()), CAFFE_CUDA_NUM_THREADS>>>(
      beta.count(), beta_data, beta_diff);
	CUDA_POST_KERNEL_CHECK;
  // ------------------------------------------------------------------------
  // ------------------------------------------------------------------------
    //first term
    this->gemm_Dlalpha_gpu_(&beta, &Dlbeta, true);
    this->patches2im_gpu_(&Dlbeta, &bottom_recon, false);
    caffe_gpu_sub(bottom[0]->count(), bottom[0]->gpu_data(), bottom_recon.gpu_data(),
      bottom_recon.mutable_gpu_data());
    this->im2patches_gpu_(&bottom_recon, &residual, false);
    caffe_gpu_gemm(CblasNoTrans, CblasTrans, residual.shape(0), beta.shape(0), 
      residual.shape(1), Dtype(1), residual.gpu_data(), beta.gpu_diff(),
      Dtype(0), this->blobs_[0]->mutable_gpu_diff());
  // ------------------------------------------------------------------------
  // ------------------------------------------------------------------------
    //second term
    this->gemm_Dlalpha_gpu_(&beta, &Dlbeta, false);
    this->patches2im_gpu_(&Dlbeta, &bottom_recon, false);
    this->im2patches_gpu_(&bottom_recon, &residual, false);
    caffe_gpu_gemm(CblasNoTrans, CblasTrans, residual.shape(0), beta.shape(0),
      residual.shape(1), Dtype(-1), residual.gpu_data(), beta.gpu_data(),
      Dtype(1), this->blobs_[0]->mutable_gpu_diff());
    caffe_gpu_scal(this->blobs_[0]->count(), Dtype(1./bottom[0]->shape(0)),
      this->blobs_[0]->mutable_gpu_diff());
  // ------------------------------------------------------------------------
  }
  if (propagate_down[0]) {
    caffe_copy(bottom[0]->count(), bottom_recon.gpu_data(),
      bottom[0]->mutable_gpu_diff());
  }
}

template <typename Dtype>
void CSCLayer<Dtype>::gemm_Dlalpha_gpu_(const Blob<Dtype> *alpha, Blob<Dtype> *Dlalpha,
      bool prefer_data) {
  CHECK_EQ(alpha->shape(0), this->blobs_[0]->shape(1));
  CHECK_EQ(alpha->shape(1), Dlalpha->shape(1));
  CHECK_EQ(Dlalpha->shape(0), this->blobs_[0]->shape(0));
  const Dtype *ptr = prefer_data ? alpha->gpu_data() : alpha->gpu_diff();
  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, this->blobs_[0]->shape(0), alpha->shape(1),
    alpha->shape(0), Dtype(1), this->blobs_[0]->gpu_data(), ptr, Dtype(0),
    Dlalpha->mutable_gpu_data());
}


template <typename Dtype>
__global__ void soft_threshold_kernel(const int n, Dtype thresh, Dtype *x) {
	CUDA_KERNEL_LOOP(index, n) {
		x[index] = x[index] > thresh ? x[index] - thresh :
			(x[index] < -thresh ? x[index] + thresh : Dtype(0));
	}
}

template <typename Dtype>
void CSCLayer<Dtype>::caffe_gpu_soft_thresholding_(const int n, Dtype thresh, Dtype* x) {
	soft_threshold_kernel<Dtype><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
		n, thresh, x);
	CUDA_POST_KERNEL_CHECK;
}

// It is actually to swap the num and channels dim in top.
template <typename Dtype>
void CSCLayer<Dtype>::permute_num_channels_gpu_(const Blob<Dtype> *top, Blob<Dtype> *patches,
	bool permute_diff) {
	CHECK_EQ(patches->shape(0), top->shape(1));
	CHECK_EQ(patches->shape(1), top->shape(0)*top->shape(2)*top->shape(3));
	int patch_width = top->shape(2) * top->shape(3);
	for (int i = 0; i < top->shape(0)*top->shape(1); ++i) {
		int n = i / top->shape(1);
		int c = i % top->shape(1);
		int source_offset = i * patch_width;
		int target_offset = c * patches->shape(1) + n * patch_width;
		caffe_copy(patch_width, top->gpu_data() + source_offset,
			patches->mutable_gpu_data() + target_offset);
		if (permute_diff) {
			caffe_copy(patch_width, top->gpu_diff() + source_offset,
				patches->mutable_gpu_diff() + target_offset);
		}
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(CSCLayer);

} // namespace caffe
