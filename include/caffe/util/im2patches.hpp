#ifndef _CAFFE_UTIL_IM2PATCHES_HPP_
#define _CAFFE_UTIL_IM2PATCHES_HPP_

namespace caffe {

template <typename Dtype>
void im2patches_circulant_cpu(const Dtype *blob, const int nsamples,
	const int channels, const int height, const int width,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, Dtype *patches);

template <typename Dtype>
void im2patches_padzeros_cpu(const Dtype *blob, const int nsamples,
	const int channels, const int height, const int width,
	const int patch_h, const int patch_w,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, Dtype *patches);

template <typename Dtype>
void patches2im_circulant_cpu(const Dtype *patches, const int nsamples,
	const int channels, const int height, const int width,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, Dtype *blob);

template <typename Dtype>
void patches2im_padzeros_cpu(const Dtype *patches, const int nsamples,
	const int channels, const int height, const int width,
	const int patch_h, const int patch_w,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, Dtype *blob);

template <typename Dtype>
void im2patches_circulant_gpu(const Dtype *blob, const int nsamples,
	const int channels, const int height, const int width,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, Dtype *patches);

template <typename Dtype>
void im2patches_padzeros_gpu(const Dtype *blob, const int nsamples,
	const int channels, const int height, const int width,
	const int patch_h, const int patch_w,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, Dtype *patches);

template <typename Dtype>
void patches2im_circulant_gpu(const Dtype *patches, const int nsamples,
	const int channels, const int height, const int width,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, Dtype *blob);

template <typename Dtype>
void patches2im_padzeros_gpu(const Dtype *patches, const int nsamples,
	const int channels, const int height, const int width,
	const int patch_h, const int patch_w,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, Dtype *blob);
} // namespace caffe




#endif // _CAFFE_UTIL_IM2PATCHES_HPP_