#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/im2patches.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class Im2patchesKernelTest : public GPUDeviceTest<Dtype> {
protected:
	Im2patchesKernelTest()
		: blob_bottom_(new Blob<Dtype>(10, 7, 5, 4)),
		blob_top_(new Blob<Dtype>()),
		blob_top_nopad_(new Blob<Dtype>()),
		blob_top_cpu_(new Blob<Dtype>()),
		blob_top_nopad_cpu_(new Blob<Dtype>()) {
		nsamples_ = blob_bottom_->num();
		channels_ = blob_bottom_->channels();
		height_ = blob_bottom_->height();
		width_ = blob_bottom_->width();
		kernel_h_ = 3;
		kernel_w_ = 3;
		patch_width_ = 200;
		patch_width_nopad_ = 60;
		patch_h_ = 5;
		patch_w_ = 4;
		patch_h_nopad_ = 3;
		patch_w_nopad_ = 2;
		vector<int> top_shape(2);
		vector<int> top_nopad_shape(2);
		top_shape[0] = channels_ * kernel_h_ * kernel_w_;
		top_shape[1] = patch_width_;
		top_nopad_shape[0] = top_shape[0];
		top_nopad_shape[1] = patch_width_nopad_;
		blob_top_->Reshape(top_shape);
		blob_top_cpu_->Reshape(top_shape);
		blob_top_nopad_->Reshape(top_nopad_shape);
		blob_top_nopad_cpu_->Reshape(top_nopad_shape);
		FillerParameter filler_param;
		GaussianFiller<Dtype> filler(filler_param);
		filler.Fill(this->blob_bottom_);
	}

	virtual ~Im2patchesKernelTest() {
		delete blob_bottom_;
		delete blob_top_;
		delete blob_top_nopad_;
		delete blob_top_cpu_;
		delete blob_top_nopad_cpu_;
	}

	Blob<Dtype>* const blob_bottom_;
	Blob<Dtype>* const blob_top_;
	Blob<Dtype>* const blob_top_nopad_;
	Blob<Dtype>* const blob_top_cpu_;
	Blob<Dtype>* const blob_top_nopad_cpu_;
	int height_;
	int width_;
	int channels_;
	int nsamples_;
	int kernel_h_;
	int kernel_w_;
	int patch_width_;
	int patch_width_nopad_;
	int patch_h_;
	int patch_w_;
	int patch_h_nopad_;
	int patch_w_nopad_;
};

TYPED_TEST_CASE(Im2patchesKernelTest, TestDtypes);

TYPED_TEST(Im2patchesKernelTest, TestCirculantFront) {
	const int pad_h = this->kernel_h_ - 1;
	const int pad_w = this->kernel_w_ - 1;
	const TypeParam *bottom_cpu_data = this->blob_bottom_->cpu_data();
	const TypeParam *bottom_data = this->blob_bottom_->gpu_data();
	TypeParam *top_data = this->blob_top_->mutable_gpu_data();
	TypeParam *cpu_data = this->blob_top_cpu_->mutable_cpu_data();
	// cpu version
	im2patches_circulant_cpu(bottom_cpu_data, this->nsamples_, this->channels_,
		this->height_, this->width_, this->kernel_h_, this->kernel_w_,
		pad_h, pad_w, cpu_data);
	im2patches_circulant_gpu(bottom_data, this->nsamples_, this->channels_,
		this->height_, this->width_, this->kernel_h_, this->kernel_w_,
		pad_h, pad_w, top_data);
	// The two version should coincide
	for (int i = 0; i < this->blob_top_->count(); ++i) {
		TypeParam cpuval = cpu_data[i];
		TypeParam gpuval = this->blob_top_->cpu_data()[i];
		EXPECT_EQ(cpuval, gpuval);
	}
}

TYPED_TEST(Im2patchesKernelTest, TestCirculantBack) {
	const int pad_h = 0;
	const int pad_w = 0;
	const TypeParam *bottom_cpu_data = this->blob_bottom_->cpu_data();
	const TypeParam *bottom_data = this->blob_bottom_->gpu_data();
	TypeParam *top_data = this->blob_top_->mutable_gpu_data();
	TypeParam *cpu_data = this->blob_top_cpu_->mutable_cpu_data();
	// cpu version
	im2patches_circulant_cpu(bottom_cpu_data, this->nsamples_, this->channels_,
		this->height_, this->width_, this->kernel_h_, this->kernel_w_,
		pad_h, pad_w, cpu_data);
	im2patches_circulant_gpu(bottom_data, this->nsamples_, this->channels_,
		this->height_, this->width_, this->kernel_h_, this->kernel_w_,
		pad_h, pad_w, top_data);
	// The two version should coincide
	for (int i = 0; i < this->blob_top_->count(); ++i) {
		TypeParam cpuval = cpu_data[i];
		TypeParam gpuval = this->blob_top_->cpu_data()[i];
		EXPECT_EQ(cpuval, gpuval);
	}
}

TYPED_TEST(Im2patchesKernelTest, TestZerosFront) {
	const int pad_h = this->kernel_h_ - 1;
	const int pad_w = this->kernel_w_ - 1;
	const int patch_h = this->height_;
	const int patch_w = this->width_;
	const TypeParam *bottom_cpu_data = this->blob_bottom_->cpu_data();
	const TypeParam *bottom_data = this->blob_bottom_->gpu_data();
	TypeParam *top_data = this->blob_top_->mutable_gpu_data();
	TypeParam *cpu_data = this->blob_top_cpu_->mutable_cpu_data();
	// cpu version
	im2patches_padzeros_cpu(bottom_cpu_data, this->nsamples_, this->channels_,
		this->height_, this->width_, patch_h, patch_w, this->kernel_h_,
		this->kernel_w_, pad_h, pad_w, cpu_data);
	// gpu version
	im2patches_padzeros_gpu(bottom_data, this->nsamples_, this->channels_,
		this->height_, this->width_, patch_h, patch_w, this->kernel_h_,
		this->kernel_w_, pad_h, pad_w, top_data);
	for (int i = 0; i < this->blob_top_->count(); ++i) {
		TypeParam cpuval = cpu_data[i];
		TypeParam gpuval = this->blob_top_->cpu_data()[i];
		EXPECT_EQ(cpuval, gpuval);
	}
}

TYPED_TEST(Im2patchesKernelTest, TestZerosBack) {
	const int pad_h = 0;
	const int pad_w = 0;
	const int patch_h = this->height_;
	const int patch_w = this->width_;
	const TypeParam *bottom_cpu_data = this->blob_bottom_->cpu_data();
	const TypeParam *bottom_data = this->blob_bottom_->gpu_data();
	TypeParam *top_data = this->blob_top_->mutable_gpu_data();
	TypeParam *cpu_data = this->blob_top_cpu_->mutable_cpu_data();
	// cpu version
	im2patches_padzeros_cpu(bottom_cpu_data, this->nsamples_, this->channels_,
		this->height_, this->width_, patch_h, patch_w, this->kernel_h_,
		this->kernel_w_, pad_h, pad_w, cpu_data);
	// gpu version
	im2patches_padzeros_gpu(bottom_data, this->nsamples_, this->channels_,
		this->height_, this->width_, patch_h, patch_w, this->kernel_h_,
		this->kernel_w_, pad_h, pad_w, top_data);
	for (int i = 0; i < this->blob_top_->count(); ++i) {
		TypeParam cpuval = cpu_data[i];
		TypeParam gpuval = this->blob_top_->cpu_data()[i];
		EXPECT_EQ(cpuval, gpuval);
	}
}

TYPED_TEST(Im2patchesKernelTest, TestZerosBoth) {
	const int pad_h = (this->kernel_h_ - 1) / 2;
	const int pad_w = (this->kernel_w_ - 1) / 2;
	const int patch_h = this->height_;
	const int patch_w = this->width_;
	const TypeParam *bottom_cpu_data = this->blob_bottom_->cpu_data();
	const TypeParam *bottom_data = this->blob_bottom_->gpu_data();
	TypeParam *top_data = this->blob_top_->mutable_gpu_data();
	TypeParam *cpu_data = this->blob_top_cpu_->mutable_cpu_data();
	// cpu version
	im2patches_padzeros_cpu(bottom_cpu_data, this->nsamples_, this->channels_,
		this->height_, this->width_, patch_h, patch_w, this->kernel_h_,
		this->kernel_w_, pad_h, pad_w, cpu_data);
	// gpu version
	im2patches_padzeros_gpu(bottom_data, this->nsamples_, this->channels_,
		this->height_, this->width_, patch_h, patch_w, this->kernel_h_,
		this->kernel_w_, pad_h, pad_w, top_data);
	for (int i = 0; i < this->blob_top_->count(); ++i) {
		TypeParam cpuval = cpu_data[i];
		TypeParam gpuval = this->blob_top_->cpu_data()[i];
		EXPECT_EQ(cpuval, gpuval);
	}
}

TYPED_TEST(Im2patchesKernelTest, TestZerosNopad) {
	const int pad_h = 0;
	const int pad_w = 0;
	const int patch_h = this->height_ - this->kernel_h_ + 1;
	const int patch_w = this->width_ - this->kernel_w_ + 1;
	const TypeParam *bottom_cpu_data = this->blob_bottom_->cpu_data();
	const TypeParam *bottom_data = this->blob_bottom_->gpu_data();
	TypeParam *top_data = this->blob_top_nopad_->mutable_gpu_data();
	TypeParam *cpu_data = this->blob_top_nopad_cpu_->mutable_cpu_data();
	// cpu version
	im2patches_padzeros_cpu(bottom_cpu_data, this->nsamples_, this->channels_,
		this->height_, this->width_, patch_h, patch_w, this->kernel_h_,
		this->kernel_w_, pad_h, pad_w, cpu_data);
	// gpu version
	im2patches_padzeros_gpu(bottom_data, this->nsamples_, this->channels_,
		this->height_, this->width_, patch_h, patch_w, this->kernel_h_,
		this->kernel_w_, pad_h, pad_w, top_data);
	for (int i = 0; i < this->blob_top_nopad_->count(); ++i) {
		TypeParam cpuval = cpu_data[i];
		TypeParam gpuval = this->blob_top_nopad_->cpu_data()[i];
		EXPECT_EQ(cpuval, gpuval);
	}
}

template <typename Dtype>
class Patches2imKernelTest : public GPUDeviceTest<Dtype> {
protected:
	Patches2imKernelTest()
		: patches_(new Blob<Dtype>()), patches_nopad_(new Blob<Dtype>()),
		blob_(new Blob<Dtype>()), blob_cpu_(new Blob<Dtype>()) {
		height_ = 5;
		width_ = 4;
		nsamples_ = 2;
		channels_ = 7;
		kernel_h_ = 3;
		kernel_w_ = 3;
		patch_h_ = height_;
		patch_w_ = width_;
		patch_h_nopad_ = height_ - kernel_h_ + 1;
		patch_w_nopad_ = width_ - kernel_w_ + 1;
		patch_width_ = nsamples_ * patch_h_ * patch_w_;
		patch_width_nopad_ = nsamples_ * patch_h_nopad_ * patch_w_nopad_;
		vector<int> patch_shape(2);
		vector<int> patch_nopad_shape(2);
		patch_shape[0] = channels_ * kernel_h_ * kernel_w_;
		patch_shape[1] = patch_width_;
		patch_nopad_shape[0] = channels_ * kernel_h_ * kernel_w_;
		patch_nopad_shape[1] = patch_width_nopad_;
		patches_->Reshape(patch_shape);
		patches_nopad_->Reshape(patch_nopad_shape);
		blob_->Reshape(nsamples_, channels_, height_, width_);
		blob_cpu_->Reshape(nsamples_, channels_, height_, width_);
		FillerParameter filler_param;
		GaussianFiller<Dtype> filler(filler_param);
		filler.Fill(patches_);
		filler.Fill(patches_nopad_);

	}
	virtual ~Patches2imKernelTest() {
		delete patches_;
		delete patches_nopad_;
		delete blob_;
		delete blob_cpu_;
	}

	Blob<Dtype>* const patches_;
	Blob<Dtype>* const patches_nopad_;
	Blob<Dtype>* const blob_;
	Blob<Dtype>* const blob_cpu_;
	int height_;
	int width_;
	int nsamples_;
	int channels_;
	int kernel_h_;
	int kernel_w_;
	int patch_width_;
	int patch_width_nopad_;
	int patch_h_;
	int patch_w_;
	int patch_h_nopad_;
	int patch_w_nopad_;
};

TYPED_TEST_CASE(Patches2imKernelTest, TestDtypes);

TYPED_TEST(Patches2imKernelTest, TestCirculantFront) {
	const int pad_h = this->kernel_h_ - 1;
	const int pad_w = this->kernel_w_ - 1;
	const TypeParam *patches_cpu_data = this->patches_->cpu_data();
	const TypeParam *patches_gpu_data = this->patches_->gpu_data();
	TypeParam *blob_cpu_data = this->blob_cpu_->mutable_cpu_data();
	TypeParam *blob_gpu_data = this->blob_->mutable_gpu_data();
	patches2im_circulant_cpu(patches_cpu_data, this->nsamples_, this->channels_,
		this->height_, this->width_, this->kernel_h_, this->kernel_w_,
		pad_h, pad_w, blob_cpu_data);
	patches2im_circulant_gpu(patches_gpu_data, this->nsamples_, this->channels_,
		this->height_, this->width_, this->kernel_h_, this->kernel_w_,
		pad_h, pad_w, blob_gpu_data);
	for (int i = 0; i < this->blob_->count(); ++i) {
		TypeParam cpuval = blob_cpu_data[i];
		TypeParam gpuval = this->blob_->cpu_data()[i];
		EXPECT_EQ(cpuval, gpuval);
	}
}

TYPED_TEST(Patches2imKernelTest, TestCirculantBack) {
	const int pad_h = 0;
	const int pad_w = 0;
	const TypeParam *patches_cpu_data = this->patches_->cpu_data();
	const TypeParam *patches_gpu_data = this->patches_->gpu_data();
	TypeParam *blob_cpu_data = this->blob_cpu_->mutable_cpu_data();
	TypeParam *blob_gpu_data = this->blob_->mutable_gpu_data();
	patches2im_circulant_cpu(patches_cpu_data, this->nsamples_, this->channels_,
		this->height_, this->width_, this->kernel_h_, this->kernel_w_,
		pad_h, pad_w, blob_cpu_data);
	patches2im_circulant_gpu(patches_gpu_data, this->nsamples_, this->channels_,
		this->height_, this->width_, this->kernel_h_, this->kernel_w_,
		pad_h, pad_w, blob_gpu_data);
	for (int i = 0; i < this->blob_->count(); ++i) {
		TypeParam cpuval = blob_cpu_data[i];
		TypeParam gpuval = this->blob_->cpu_data()[i];
		EXPECT_EQ(cpuval, gpuval);
	}
}

TYPED_TEST(Patches2imKernelTest, TestZerosFront) {
	const int pad_h = this->kernel_h_ - 1;
	const int pad_w = this->kernel_w_ - 1;
	const TypeParam *patches_cpu_data = this->patches_->cpu_data();
	const TypeParam *patches_gpu_data = this->patches_->gpu_data();
	TypeParam *blob_cpu_data = this->blob_cpu_->mutable_cpu_data();
	TypeParam *blob_gpu_data = this->blob_->mutable_gpu_data();
	patches2im_padzeros_cpu(patches_cpu_data, this->nsamples_, this->channels_,
		this->height_, this->width_, this->patch_h_, this->patch_w_,
		this->kernel_h_, this->kernel_w_, pad_h, pad_w, blob_cpu_data);
	patches2im_padzeros_gpu(patches_gpu_data, this->nsamples_, this->channels_,
		this->height_, this->width_, this->patch_h_, this->patch_w_,
		this->kernel_h_, this->kernel_w_, pad_h, pad_w, blob_gpu_data);
	for (int i = 0; i < this->blob_->count(); ++i) {
		TypeParam cpuval = blob_cpu_data[i];
		TypeParam gpuval = this->blob_->cpu_data()[i];
		EXPECT_EQ(cpuval, gpuval);
	}
}

TYPED_TEST(Patches2imKernelTest, TestZerosBack) {
	const int pad_h = 0;
	const int pad_w = 0;
	const TypeParam *patches_cpu_data = this->patches_->cpu_data();
	const TypeParam *patches_gpu_data = this->patches_->gpu_data();
	TypeParam *blob_cpu_data = this->blob_cpu_->mutable_cpu_data();
	TypeParam *blob_gpu_data = this->blob_->mutable_gpu_data();
	patches2im_padzeros_cpu(patches_cpu_data, this->nsamples_, this->channels_,
		this->height_, this->width_, this->patch_h_, this->patch_w_,
		this->kernel_h_, this->kernel_w_, pad_h, pad_w, blob_cpu_data);
	patches2im_padzeros_gpu(patches_gpu_data, this->nsamples_, this->channels_,
		this->height_, this->width_, this->patch_h_, this->patch_w_,
		this->kernel_h_, this->kernel_w_, pad_h, pad_w, blob_gpu_data);
	for (int i = 0; i < this->blob_->count(); ++i) {
		TypeParam cpuval = blob_cpu_data[i];
		TypeParam gpuval = this->blob_->cpu_data()[i];
		EXPECT_EQ(cpuval, gpuval);
	}
}

TYPED_TEST(Patches2imKernelTest, TestZerosBoth) {
	const int pad_h = (this->kernel_h_ - 1) / 2;
	const int pad_w = (this->kernel_w_ - 1) / 2;
	const TypeParam *patches_cpu_data = this->patches_->cpu_data();
	const TypeParam *patches_gpu_data = this->patches_->gpu_data();
	TypeParam *blob_cpu_data = this->blob_cpu_->mutable_cpu_data();
	TypeParam *blob_gpu_data = this->blob_->mutable_gpu_data();
	patches2im_padzeros_cpu(patches_cpu_data, this->nsamples_, this->channels_,
		this->height_, this->width_, this->patch_h_, this->patch_w_,
		this->kernel_h_, this->kernel_w_, pad_h, pad_w, blob_cpu_data);
	patches2im_padzeros_gpu(patches_gpu_data, this->nsamples_, this->channels_,
		this->height_, this->width_, this->patch_h_, this->patch_w_,
		this->kernel_h_, this->kernel_w_, pad_h, pad_w, blob_gpu_data);
	for (int i = 0; i < this->blob_->count(); ++i) {
		TypeParam cpuval = blob_cpu_data[i];
		TypeParam gpuval = this->blob_->cpu_data()[i];
		EXPECT_EQ(cpuval, gpuval);
	}
}

TYPED_TEST(Patches2imKernelTest, TestZerosNopad) {
	const int pad_h = 0;
	const int pad_w = 0;
	const TypeParam *patches_cpu_data = this->patches_nopad_->cpu_data();
	const TypeParam *patches_gpu_data = this->patches_nopad_->gpu_data();
	TypeParam *blob_cpu_data = this->blob_cpu_->mutable_cpu_data();
	TypeParam *blob_gpu_data = this->blob_->mutable_gpu_data();
	patches2im_padzeros_cpu(patches_cpu_data, this->nsamples_, this->channels_,
		this->height_, this->width_, this->patch_h_nopad_, this->patch_w_nopad_,
		this->kernel_h_, this->kernel_w_, pad_h, pad_w, blob_cpu_data);
	patches2im_padzeros_gpu(patches_gpu_data, this->nsamples_, this->channels_,
		this->height_, this->width_, this->patch_h_nopad_, this->patch_w_nopad_,
		this->kernel_h_, this->kernel_w_, pad_h, pad_w, blob_gpu_data);
	for (int i = 0; i < this->blob_->count(); ++i) {
		TypeParam cpuval = blob_cpu_data[i];
		TypeParam gpuval = this->blob_->cpu_data()[i];
		EXPECT_EQ(cpuval, gpuval);
	}
}

} // namespace caffe