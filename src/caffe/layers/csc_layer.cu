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

