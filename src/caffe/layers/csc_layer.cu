#include <vector>

#include "caffe/layers/csc_layer.hpp"


template <typename Dtype>
__global__ void im2patches_circulant_kernel(const int n, const Dtype *blob,
      const int channels, const int height, const int width, 
      const int kernel_h, const int kernel_w, const int patch_width,
      const int pad_h, const int pad_w, Dtype *patches) {
  // index is the coordinate of pixels on blob
  CAFFE_KERNEL_LOOP(index, n) {
    const int h_index = index / width;
    const int h_im = h_index % height;
    const int w_im = index % width;
    const int c_index = h_index / height;
    const int c_im = c_index % channels;
    const int n_im = c_index / channels;
    Dtype *local_patches = patches + 
      c_im * kernel_h * kernel_w * patch_width +
      n_im * height * width + width * h_im + w_im;
    const Dtype *local_blob = blob + (n_im * channels + c_im) * height * width;
    for (int i = 0; i < kernel_h; ++i) {
      int h_offset = (h_im - pad_h + height + i) % height;
      for (int j = 0; j < kernel_w; ++j) {
        int w_offset = (w_im - pad_w + width + j) % width;
        *local_patches = local_blob[h_offset * width + w_offset];
        local_patches += patch_width;
      }
    }
  }
}

template <typename Dtype>
__global__ void im2patches_padzeros_kernel(const int n, const Dtype *blob,
      const int channels, const int height, const int width, 
      const int kernel_h, const int kernel_w, const int patch_width,
      const int pad_h, const int pad_w, Dtype *patches) {
  // index is the coordinate of pixels on blob
  CAFFE_KERNEL_LOOP(index, n) {
    const int h_index = index / width;
    const int h_im = h_index % height;
    const int w_im = index % width;
    const int c_index = h_index / height;
    const int c_im = c_index % channels;
    const int n_im = c_index / channels;
    const int h_offset = h_im - pad_h;
    const int w_offset = w_im - pad_w;
    Dtype *local_patches = patches + 
      c_im * kernel_h * kernel_w * patch_width +
      n_im * height * width + width * h_im + w_im;
    const Dtype *local_blob = blob + (n_im * channels + c_im) * height * width;
    for (int i = 0; i < kernel_h; ++i) {
      int h = h_offset + i;
      for (int j = 0; j < kernel_w; ++j) {
        int w = w_offset + j;
        *local_patches = 
          (h >= 0 && w >= 0 && h < height && w < width) ?
          local_blob[h*width + w] : 0;
        loca_patches += patch_width;
      }
    }
  }
}

template <typename Dtype>
__global__ void im2patches_nopad_kernel(const int n , const Dtype *blob,
      const int channels, const int height, const int width,
      const int kernel_h, const int kernel_w, const int patch_h,
      const int patch_w, const int patch_width, Dtype *patches) {
  CAFFE_KERNEL_LOOP(index, n) {
    const int h_index = index / patch_w;
    const int h_im = h_index % patch_h;
    const int w_im = index % patch_w;
    const int c_index = h_index / patch_h;
    const int c_im = c_index % channels;
    const int n_im = c_index / channels;
    Dtype *local_patches = patches + 
      c_im * kernel_h * kernel_w * patch_width +
      n_im * patch_h * patch_w + patch_w * h_im + w_im;
    const Dtype *local_blob = blob + (n_im * channels + c_im) * height * width;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        *local_patches = local_blob[(h_im+i)*width + w_im+j];
        lcoal_patches += patch_width;
      }
    }
  }
}

template <typename Dtype>
__global__ void patches2im_circulant_kernel(const int n, const Dtype *patches
      const int channels, const int height, const int width, 
      const int kernel_h, const int kernel_w, const int patch_width,
      const int pad_h, const int pad_w, Dtype *blob) {
  // index is the coordinate of pixels on blob
  CAFFE_KERNEL_LOOP(index, n) {
    Dtype val = Dtype(0);
    const int h_index = index / width;
    const int h_im = h_index % height;
    const int w_im = index % width;
    const int c_index = h_index / height;
    const int c_im = c_index % channels;
    const int n_im = c_index / channels;
    Dtype *local_blob = blob + (n_im * channels + c_im) * height * width;
    const Dtype *local_patches = patches + c_im * kernel_h * kernel_w * patch_width +
      n_im * height * width;
    for (int i = 0; i < kernel_h; ++i) {
      int h_offset = (h_im + pad_h + height - i) % height;
      for (int j = 0; j < kernel_w; ++j) {
        int w_offset = (w_im + pad_w + width - j) % width;
        int local_patches_loc = (i*kernel_w+j) * patch_width + h_offset * width + w_offset;
        val += local_patches[local_patches_loc];
      }
    }
    local_blob[index] = val;
  }
}

template <typename Dtype>
__global__ void patches2im_padzeros_kernel(const int n, const Dtype *patches
      const int channels, const int height, const int width, 
      const int kernel_h, const int kernel_w, const int patch_width,
      const int pad_h, const int pad_w, Dtype *blob) {
  // index is the coordinate of pixels on blob
  CAFFE_KERNEL_LOOP(index, n) {
    Dtype val = Dtype(0);
    const int h_index = index / width;
    const int h_im = h_index % height;
    const int w_im = index % width;
    const int c_index = h_index / height;
    const int c_im = c_index % channels;
    const int n_im = c_index / channels;
    Dtype *local_blob = blob + (n_im * channels + c_im) * height * width;
    const Dtype *local_patches = patches + c_im * kernel_h * kernel_w * patch_width +
      n_im * height * width;
    for (int i = 0; i < kernel_h; ++i) {
      int h_offset = h_im + pad_h - i;
      for (int j = 0; j < kernel_w; ++j) {
        int w_offset = w_im + pad_w - j;
        if (h_offset >= 0 && w_offset >= 0 && h_offset < height && w_offset < width) {
          int local_patches_loc = (i*kernel_w+j) * patch_width + h_offset * width + w_offset;
          val += local_patches[local_patches_loc];
        }
      }
    }
    local_blob[index] = val;
  }
}

template <typename Dtype>
__global__ void patches2im_nopad_kernel(const int n , const Dtype *patches,
      const int channels, const int height, const int width,
      const int kernel_h, const int kernel_w, const int patch_h,
      const int patch_w, const int patch_width, Dtype *blob) {
  CAFFE_KERNEL_LOOP(index, n) {
    const int h_index = index / patch_w;
    const int h_im = h_index % patch_h;
    const int w_im = index % patch_w;
    const int c_index = h_index / patch_h;
    const int c_im = c_index % channels;
    const int n_im = c_index / channels;
    Dtype *local_blob = blob + (n_im * channels + c_im) * height * width;
    const Dtype *local_patches = patches + c_im *kernel_h * kernel_w * patch_width +
      n_im * patch_h * patch_w;
    for (int i = 0; i < kernel_h; ++i) {
      int h_offset = h_im - i;
      for (int j = 0; j < kernel_w; ++j) {
        int w_offset = w_im - j;
        if (h_offset >= 0 && w_offset >= 0 && h_offset < patch_h && w_offset < patch_w) {
          int local_patches_loc = (i*kernel_w+j) * patch_width + h_offset * patch_w + w_offset;
          val += local_patches[local_patches_loc];
        }
      }
    }
    local_blob[index] = val;
  }
}

template <typename Dtype>
void CSCLayer<Dtype>::im2patches_gpu_(const Blob<Dtype> *blob, Blob<Dtype> *patches, bool compute_diff) {
  int pad_h = 0;
  int pad_w = 0;
  int patch_width = patches->shape(1);
  int num_kernels = blob->count();
  switch (boundary_) {
  case CSCParameter::NOPAD:
    int patch_h = blob->shape(2)-kernel_h_+1;
    int patch_w = blob->shape(3)-kernel_w_+1;
    num_kernels = patch_h * patch_w * blob->shape(0) * blob->shape(1);
    im2patches_nopad_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                                     CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, blob->gpu_data(), blob->shape(1), blob->shape(2), blob->shape(3),
      kernel_h_, kernel_w_, patch_h, patch_w, patch_width, patches->mutable_gpu_data());
    if (compute_diff) {
      im2patches_nopad_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                                       CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, blob->gpu_diff(), blob->shape(1), blob->shape(2), blob->shape(3),
        kernel_h_, kernel_w_, patch_h, patch_w, patch_width, patches->mutable_gpu_diff());
    }
    break;
  case CSCParameter::PAD_FRONT:
    pad_h = kernel_h_ - 1;
    pad_w = kernel_w_ - 1;
    im2patches_padzeros_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                                        CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, blob->gpu_data(), blob->shape(1), blob->shape(2), blob->shape(3),
      kernel_h_, kernel_w_, pad_h, pad_w, patch_width, patches->mutable_gpu_data());
    if (compute_diff) {
      im2patches_padzeros_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                                          CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, blob->gpu_diff(), blob->shape(1), blob->shape(2), blob->shape(3),
        kernel_h_, kernel_w_, pad_h, pad_w, patch_width, patches->mutable_gpu_diff());
    }
    break;
  case CSCParameter::PAD_BACK:
    im2patches_padzeros_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                                        CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, blob->gpu_data(), blob->shape(1), blob->shape(2), blob->shape(3),
      kernel_h_, kernel_w_, pad_h, pad_w, patch_width, patches->mutable_gpu_data());
    if (compute_diff) {
      im2patches_padzeros_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                                          CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, blob->gpu_diff(), blob->shape(1), blob->shape(2), blob->shape(3),
        kernel_h_, kernel_w_, pad_h, pad_w, patch_width, patches->mutable_gpu_diff());
    }
    break;
  case CSCParameter::PAD_BOTH:
    pad_h = (kernel_h_ - 1) / 2;
    pad_w = (kernel_w_ - 1) / 2;
    im2patches_padzeros_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                                        CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, blob->gpu_data(), blob->shape(1), blob->shape(2), blob->shape(3),
      kernel_h_, kernel_w_, pad_h, pad_w, patch_width, patches->mutable_gpu_data());
    if (compute_diff) {
      im2patches_padzeros_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                                          CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, blob->gpu_diff(), blob->shape(1), blob->shape(2), blob->shape(3),
        kernel_h_, kernel_w_, pad_h, pad_w, patch_width, patches->mutable_gpu_diff());
    }
    break;
  case CSCParameter::CIRCULANT_FRONT:
    pad_h = kernel_h_ - 1;
    pad_w = kernel_w_ - 1;
    im2patches_circulant_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                                        CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, blob->gpu_data(), blob->shape(1), blob->shape(2), blob->shape(3),
      kernel_h_, kernel_w_, pad_h, pad_w, patch_width, patches->mutable_gpu_data());
    if (compute_diff) {
      im2patches_circulant_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                                          CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, blob->gpu_diff(), blob->shape(1), blob->shape(2), blob->shape(3),
        kernel_h_, kernel_w_, pad_h, pad_w, patch_width, patches->mutable_gpu_diff());
    }
    break;
  case CSCParameter::CIRCULANT_BACK:
    im2patches_circulant_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                                        CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, blob->gpu_data(), blob->shape(1), blob->shape(2), blob->shape(3),
      kernel_h_, kernel_w_, pad_h, pad_w, patch_width, patches->mutable_gpu_data());
    if (compute_diff) {
      im2patches_circulant_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                                          CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, blob->gpu_diff(), blob->shape(1), blob->shape(2), blob->shape(3),
        kernel_h_, kernel_w_, pad_h, pad_w, patch_width, patches->mutable_gpu_diff());
    }
    break;
  default:
    LOG(FATAL) << "Unknown boundary condition.";
  }
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void CSCLayer<Dtype>::patches2im_gpu_(const Blob<Dtype> *patches, Blob<Dtype> *blob, bool compute_diff) {
  int pad_h = 0;
  int pad_w = 0;
  int patch_width = patches->shape(1);
  int num_kernels = blob->count();
  switch (boundary_) {
  case CSCParameter::NOPAD:
    int patch_h = blob->shape(2)-kernel_h_+1;
    int patch_w = blob->shape(3)-kernel_w_+1;
    num_kernels = patch_h * patch_w * blob->shape(0) * blob->shape(1);
    patches2im_nopad_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                                     CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, patches->gpu_data(), blob->shape(1), blob->shape(2), blob->shape(3),
      kernel_h_, kernel_w_, patch_h, patch_w, patch_width, blob->mutable_gpu_data());
    if (compute_diff) {
      patches2im_nopad_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                                       CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, patches->gpu_diff(), blob->shape(1), blob->shape(2), blob->shape(3),
        kernel_h_, kernel_w_, patch_h, patch_w, patch_width, blob->mutable_gpu_diff());
    }
    break;
  case CSCParameter::PAD_FRONT:
    pad_h = kernel_h_ - 1;
    pad_w = kernel_w_ - 1;
    patches2im_padzeros_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                                        CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, patches->gpu_data(), blob->shape(1), blob->shape(2), blob->shape(3),
      kernel_h_, kernel_w_, pad_h, pad_w, patch_width, blob->mutable_gpu_data());
    if (compute_diff) {
      patches2im_padzeros_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                                          CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, patches->gpu_diff(), blob->shape(1), blob->shape(2), blob->shape(3),
        kernel_h_, kernel_w_, pad_h, pad_w, patch_width, blob->mutable_gpu_diff());
    }
    break;
  case CSCParameter::PAD_BACK:
    patches2im_padzeros_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                                        CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, patches->gpu_data(), blob->shape(1), blob->shape(2), blob->shape(3),
      kernel_h_, kernel_w_, pad_h, pad_w, patch_width, blob->mutable_gpu_data());
    if (compute_diff) {
      patches2im_padzeros_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                                          CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, patches->gpu_diff(), blob->shape(1), blob->shape(2), blob->shape(3),
        kernel_h_, kernel_w_, pad_h, pad_w, patch_width, blob->mutable_gpu_diff());
    }
    break;
  case CSCParameter::PAD_BOTH:
    pad_h = (kernel_h_ - 1) / 2;
    pad_w = (kernel_w_ - 1) / 2;
    patches2im_padzeros_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                                        CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, patches->gpu_data(), blob->shape(1), blob->shape(2), blob->shape(3),
      kernel_h_, kernel_w_, pad_h, pad_w, patch_width, blob->mutable_gpu_data());
    if (compute_diff) {
      patches2im_padzeros_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                                          CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, patches->gpu_diff(), blob->shape(1), blob->shape(2), blob->shape(3),
        kernel_h_, kernel_w_, pad_h, pad_w, patch_width, blob->mutable_gpu_diff());
    }
    break;
  case CSCParameter::CIRCULANT_FRONT:
    pad_h = kernel_h_ - 1;
    pad_w = kernel_w_ - 1;
    patches2im_circulant_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                                        CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, patches->gpu_data(), blob->shape(1), blob->shape(2), blob->shape(3),
      kernel_h_, kernel_w_, pad_h, pad_w, patch_width, blob->mutable_gpu_data());
    if (compute_diff) {
      patches2im_padzeros_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                                          CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, patches->gpu_diff(), blob->shape(1), blob->shape(2), blob->shape(3),
        kernel_h_, kernel_w_, pad_h, pad_w, patch_width, blob->mutable_gpu_diff());
    }
    break;
  case CSCParameter::CIRCULANT_BACK:
    patches2im_circulant_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                                        CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, patches->gpu_data(), blob->shape(1), blob->shape(2), blob->shape(3),
      kernel_h_, kernel_w_, pad_h, pad_w, patch_width, blob->mutable_gpu_data());
    if (compute_diff) {
      patches2im_circulant_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                                          CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, patches->gpu_diff(), blob->shape(1), blob->shape(2), blob->shape(3),
        kernel_h_, kernel_w_, pad_h, pad_w, patch_width, blob->mutable_gpu_diff());
    }
    break;
  default:
    LOG(FATAL) << "Unknown boundary condition.";
  }
  CUDA_POST_KERNEL_CHECK;
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
  caffe_gpu_set(alpha.count(), Dtype(0), alpha.mutable_gpu_data());
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
      caffe_gpu_soft_thresholding(alpha.count(), Dtype(lambda1_/eta),
        alpha.mutable_gpu_data());
      this->gemm_Dlalpha_gpu_(&alpha, &bottom_patch, true);
      this->patches2im_gpu_(&bottom_patch, &bottom_recon, false);
      caffe_gpu_sub(bottom[0]->count(), bottom[0]->gpu_data(), bottom_recon.gpu_data(),
        bottom_recon.mutable_gpu_data());
      Dtype loss_new = bottom_recon.sumsq_data()/2. + alpha.sumsq_data()*lambda2_/2.;
      caffe_gpu_sub(alpha.count(), alpha.gpu_data(), this->alpha_->gpu_data(),
        alpha_diff.mutable_gpu_data());
      Dtype stop = loss + caffe_gpu_dot(alpha_diff.count(), grad.gpu_data(),
        alpha_diff.gpu_data()) + alpha_diff.sumsq_data()*eta/2. - loss_new;
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
    << (Dtype)caffe_gpu_zero_norm(beta.count(), beta.gpu_data())/beta.shape(1)
    << " eta: " << eta;
}

template <typename Dtype>
__global__ void set_if_kernel(const int n, const Dtype *data, Dtype *diff) {
  CAFFE_KERNEL_LOOP(index, n) {
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
  this->permute_num_channels_(top[0], &beta, true);
  // ------------------------------------------------------------------------
  if (this->param_propagate_down_[0]) {
  // ------------------------------------------------------------------------
    // solve beta
    Dtype *beta_data = beta.mutable_gpu_data();
    Dtype *beta_diff = beta.mutable_gpu_diff();
    set_if_kernel<Dtype><<<CAFFE_GET_BLOCKS(beta.count()), CAFFE_CUDA_NUM_THREADS>>>(
      beta.count(), beta_data, beta_diff);
  // ------------------------------------------------------------------------
  // ------------------------------------------------------------------------
    //first term
    this->gemm_Dlalpha_gpu_(&beta, &Dlbeta, true);
    /* this->aggregate_patches_gpu_(&Dlbeta, &bottom_recon); */
    this->patches2im_gpu_(&Dlbeta, &bottom_recon);
    caffe_gpu_sub(bottom[0]->count(), bottom[0]->gpu_data(), bottom_recon.gpu_data(),
      bottom_recon.mutable_gpu_data());
    this->im2patches_gpu_(&bottom_recon, &residual);
    caffe_gpu_gemm(CblasNoTrans, CblasTrans, residual.shape(0), beta.shape(0), 
      residual.shape(1), Dtype(1), residual.gpu_data(), beta.gpu_diff(),
      Dtype(0), this->blobs_[0]->mutable_gpu_diff());
  // ------------------------------------------------------------------------
  // ------------------------------------------------------------------------
    //second term
    this->gemm_Dlalpha_gpu_(&beta, &Dlbeta, false);
    this->patches2im_gpu_(&Dlbeta, &bottom_recon);
    this->im2patches_gpu_(&bottom_recon, &residual);
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

INSTANTIATE_LAYER_GPU_FUNCS(CSCLayer);
