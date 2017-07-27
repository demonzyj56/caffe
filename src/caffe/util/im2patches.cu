#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/im2patches.hpp"

namespace caffe {

template <typename Dtype>
__global__ void im2patches_circulant_gpu_kernel(const int n, const Dtype *blob,
	const int channels, const int height, const int width,
	const int kernel_h, const int kernel_w, const int patch_width,
	const int pad_h, const int pad_w, Dtype *patches) {
	// index is the coordinate of pixels on blob
	CUDA_KERNEL_LOOP(index, n) {
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
__global__ void im2patches_padzeros_gpu_kernel(const int n, const Dtype *blob,
	const int channels, const int height, const int width,
	const int patch_h, const int patch_w,
	const int kernel_h, const int kernel_w, const int patch_width,
	const int pad_h, const int pad_w, Dtype *patches) {
	CUDA_KERNEL_LOOP(index, n) {
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
			int h = h_im - pad_h + i;
			for (int j = 0; j < kernel_w; ++j) {
				int w = w_im - pad_w + j;
				*local_patches = 
				(h >= 0 && w >= 0 && h < height && w < width) ?
					local_blob[h*width + w] : 0;
				local_patches += patch_width;
			}
		}
	}
}

template <typename Dtype>
__global__ void patches2im_circulant_gpu_kernel(const int n, const Dtype *patches,
	const int channels, const int height, const int width,
	const int kernel_h, const int kernel_w, const int patch_width,
	const int pad_h, const int pad_w, Dtype *blob) {
	// index is the coordinate of pixels on blob
	CUDA_KERNEL_LOOP(index, n) {
		Dtype val = Dtype(0);
		const int h_index = index / width;
		const int h_im = h_index % height;
		const int w_im = index % width;
		const int c_index = h_index / height;
		const int c_im = c_index % channels;
		const int n_im = c_index / channels;
		const Dtype *local_patches = patches + c_im * kernel_h * kernel_w * patch_width +
			n_im * height * width;
		for (int i = 0; i < kernel_h; ++i) {
			int h_offset = (h_im + pad_h + height - i) % height;
			for (int j = 0; j < kernel_w; ++j) {
				int w_offset = (w_im + pad_w + width - j) % width;
				int local_patches_loc = 
					(i*kernel_w + j) * patch_width + h_offset * width + w_offset;
				val += local_patches[local_patches_loc];
			}
		}
		blob[index] = val;
	}
}
template <typename Dtype>
__global__ void patches2im_padzeros_gpu_kernel(const int n, const Dtype *patches,
	const int channels, const int height, const int width,
	const int patch_h, const int patch_w,
	const int kernel_h, const int kernel_w, const int patch_width,
	const int pad_h, const int pad_w, Dtype *blob) {
	CUDA_KERNEL_LOOP(index, n) {
		Dtype val = Dtype(0);
		const int h_index = index / width;
		const int h_im = h_index % height;
		const int w_im = index % width;
		const int c_index = h_index / height;
		const int c_im = c_index % channels;
		const int n_im = c_index / channels;
		const Dtype *local_patches = patches + 
			c_im *kernel_h * kernel_w * patch_width +
			n_im * patch_h * patch_w;
		for (int i = 0; i < kernel_h; ++i) {
			int h_offset = h_im + pad_h - i;
			for (int j = 0; j < kernel_w; ++j) {
				int w_offset = w_im + pad_w - j;
				if (h_offset >= 0 && w_offset >= 0 && 
					h_offset < patch_h && w_offset < patch_w) {
					int local_patches_loc =
						(i*kernel_w + j) * patch_width + h_offset * patch_w + w_offset;
					val += local_patches[local_patches_loc];
				}
			}
		}
		blob[index] = val;
	}
}

template <typename Dtype>
void im2patches_circulant_gpu(const Dtype *blob, const int nsamples,
	const int channels, const int height, const int width,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, Dtype *patches) {
	const int patch_width = nsamples * height * width;
	const int num_kernels = nsamples * channels * height * width;
	im2patches_circulant_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
										     CAFFE_CUDA_NUM_THREADS >>>(
		num_kernels, blob, channels, height, width, kernel_h, kernel_w,
		patch_width, pad_h, pad_w, patches);
	CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void im2patches_padzeros_gpu(const Dtype *blob, const int nsamples,
	const int channels, const int height, const int width,
	const int patch_h, const int patch_w,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, Dtype *patches) {
	const int patch_width = nsamples * patch_h * patch_w;
	const int num_kernels = nsamples * channels * patch_h * patch_w;
	im2patches_padzeros_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
		CAFFE_CUDA_NUM_THREADS >>>(
		num_kernels, blob, channels, height, width, patch_h, patch_w,
		kernel_h, kernel_w, patch_width, pad_h, pad_w, patches);
	CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void patches2im_circulant_gpu(const Dtype *patches, const int nsamples,
	const int channels, const int height, const int width,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, Dtype *blob) {
	caffe_gpu_set(nsamples * channels * height * width, Dtype(0), blob);
	const int patch_width = nsamples * height * width;
	const int num_kernels = nsamples * channels * height * width;
	patches2im_circulant_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
		CAFFE_CUDA_NUM_THREADS >>>(
		num_kernels, patches, channels, height, width, kernel_h, kernel_w,
		patch_width, pad_h, pad_w, blob);
	CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void patches2im_padzeros_gpu(const Dtype *patches, const int nsamples,
	const int channels, const int height, const int width,
	const int patch_h, const int patch_w,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, Dtype *blob) {
	caffe_gpu_set(nsamples * channels * height * width, Dtype(0), blob);
	const int patch_width = nsamples * patch_h * patch_w;
	const int num_kernels = nsamples * channels * height * width;
	patches2im_padzeros_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
		CAFFE_CUDA_NUM_THREADS >>>(
		num_kernels, patches, channels, height, width, patch_h, patch_w,
		kernel_h, kernel_w, patch_width, pad_h, pad_w, blob);
	CUDA_POST_KERNEL_CHECK;
}



// instaniation

template void im2patches_circulant_gpu<float>(const float *blob, const int nsamples,
	const int channels, const int height, const int width,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, float *patches);
template void im2patches_circulant_gpu<double>(const double *blob, const int nsamples,
	const int channels, const int height, const int width,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, double *patches);

template void im2patches_padzeros_gpu<float>(const float *blob, const int nsamples,
	const int channels, const int height, const int width,
	const int patch_h, const int patch_w,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, float *patches);
template void im2patches_padzeros_gpu<double>(const double *blob, const int nsamples,
	const int channels, const int height, const int width,
	const int patch_h, const int patch_w,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, double *patches);

template void patches2im_circulant_gpu<float>(const float *patches, const int nsamples,
	const int channels, const int height, const int width,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, float *blob);
template void patches2im_circulant_gpu<double>(const double *patches, const int nsamples,
	const int channels, const int height, const int width,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, double *blob);

template void patches2im_padzeros_gpu<float>(const float *patches, const int nsamples,
	const int channels, const int height, const int width,
	const int patch_h, const int patch_w,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, float *blob);
template void patches2im_padzeros_gpu<double>(const double *patches, const int nsamples,
	const int channels, const int height, const int width,
	const int patch_h, const int patch_w,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, double *blob);

} // namespace caffe