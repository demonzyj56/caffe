#include "caffe/util/math_functions.hpp"
#include "caffe/util/im2patches.hpp"
namespace caffe {

template <typename Dtype>
void im2patches_circulant_cpu(const Dtype *blob, const int nsamples,
	const int channels, const int height, const int width,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, Dtype *patches) {
	const int patch_width = nsamples * height * width;
	for (int n = 0; n < nsamples; ++n) {
		for (int c = 0; c < channels; ++c) {
			const Dtype *local_blob = blob + (n * channels + c) * height * width;
			Dtype *local_patches = patches + c * patch_width * kernel_h * kernel_w + 
				n * height * width;
			for (int kh = 0; kh < kernel_h; ++kh) {
				for (int kw = 0; kw < kernel_w; ++kw) {
					for (int h = 0; h < height; ++h) {
						int h_offset = (h + kh - pad_h + height) % height;
						for (int w = 0; w < width; ++w) {
							int w_offset = (w + kw - pad_w + width) % width;
							local_patches[h * width + w] =
								local_blob[h_offset * width + w_offset];
						}
					}
					local_patches += patch_width;
				}
			}
		}
	}
}

template <typename Dtype>
void im2patches_padzeros_cpu(const Dtype *blob, const int nsamples,
	const int channels, const int height, const int width,
	const int patch_h, const int patch_w,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, Dtype *patches) {
	const int patch_width = nsamples * patch_h * patch_w;
	for (int n = 0; n < nsamples; ++n) {
		for (int c = 0; c < channels; ++c) {
			const Dtype *local_blob = blob + (n * channels + c) * height * width;
			Dtype *local_patches = patches + c * kernel_h * kernel_w * patch_width +
				n * patch_h * patch_w;
			for (int kh = 0; kh < kernel_h; ++kh) {
				for (int kw = 0; kw < kernel_w; ++kw) {
					for (int h = 0; h < patch_h; ++h) {
						int h_offset = h + kh - pad_h;
						for (int w = 0; w < patch_w; ++w) {
							int w_offset = w + kw - pad_w;
							local_patches[h * patch_w + w] =
								(h_offset >= 0 && h_offset < height &&
								w_offset >= 0 && w_offset < width) ?
								local_blob[h_offset * width + w_offset] : Dtype(0);
						}
					}
					local_patches += patch_width;
				}
			}
		}
	}
}

template <typename Dtype>
void patches2im_circulant_cpu(const Dtype *patches, const int nsamples,
	const int channels, const int height, const int width,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, Dtype *blob) {
	caffe_set(nsamples * channels * height * width, Dtype(0), blob);
	const int patch_width = nsamples * height * width;
	for (int n = 0; n < nsamples; ++n) {
		for (int c = 0; c < channels; ++c) {
			Dtype *local_blob = blob + (n * channels + c) * height * width;
			const Dtype *local_patches = patches + 
				c * patch_width * kernel_h * kernel_w +
				n * height * width;
			for (int kh = 0; kh < kernel_h; ++kh) {
				for (int kw = 0; kw < kernel_w; ++kw) {
					for (int h = 0; h < height; ++h) {
						int h_offset = (h + kh - pad_h + height) % height;
						for (int w = 0; w < width; ++w) {
							int w_offset = (w + kw - pad_w + width) % width;
							local_blob[h_offset * width + w_offset] +=
								local_patches[h * width + w];
						}
					}
					local_patches += patch_width;
				}
			}
		}
	}
}

template <typename Dtype>
void patches2im_padzeros_cpu(const Dtype *patches, const int nsamples,
	const int channels, const int height, const int width,
	const int patch_h, const int patch_w,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, Dtype *blob) {
	caffe_set(nsamples * channels * height * width, Dtype(0), blob);
	const int patch_width = nsamples * patch_h * patch_w;
	for (int n = 0; n < nsamples; ++n) {
		for (int c = 0; c < channels; ++c) {
			Dtype *local_blob = blob + (n * channels + c) * height * width;
			const Dtype *local_patches = patches + 
				c * kernel_h * kernel_w * patch_width +
				n * patch_h * patch_w;
			for (int kh = 0; kh < kernel_h; ++kh) {
				for (int kw = 0; kw < kernel_w; ++kw) {
					for (int h = 0; h < patch_h; ++h) {
						int h_offset = h + kh - pad_h;
						for (int w = 0; w < patch_w; ++w) {
							int w_offset = w + kw - pad_w;
							if (h_offset >= 0 && h_offset < height &&
								w_offset >= 0 && w_offset < width) {
								local_blob[h_offset * width + w_offset] +=
									local_patches[h * patch_w + w];
							}
						}
					}
					local_patches += patch_width;
				}
			}
		}
	}
}

// instaniation

template void im2patches_circulant_cpu<float>(const float *blob, const int nsamples,
	const int channels, const int height, const int width,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, float *patches);
template void im2patches_circulant_cpu<double>(const double *blob, const int nsamples,
	const int channels, const int height, const int width,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, double *patches);

template void im2patches_padzeros_cpu<float>(const float *blob, const int nsamples,
	const int channels, const int height, const int width,
	const int patch_h, const int patch_w,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, float *patches);
template void im2patches_padzeros_cpu<double>(const double *blob, const int nsamples,
	const int channels, const int height, const int width,
	const int patch_h, const int patch_w,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, double *patches);

template void patches2im_circulant_cpu<float>(const float *patches, const int nsamples,
	const int channels, const int height, const int width,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, float *blob);
template void patches2im_circulant_cpu<double>(const double *patches, const int nsamples,
	const int channels, const int height, const int width,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, double *blob);

template void patches2im_padzeros_cpu<float>(const float *patches, const int nsamples,
	const int channels, const int height, const int width,
	const int patch_h, const int patch_w,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, float *blob);
template void patches2im_padzeros_cpu<double>(const double *patches, const int nsamples,
	const int channels, const int height, const int width,
	const int patch_h, const int patch_w,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, double *blob);


} // namespace caffe