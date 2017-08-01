#ifndef CAFFE_CSC_LAYER_HPP_
#define CAFFE_CSC_LAYER_HPP_
/*
 * Supervised convolutional sparse coding layer.
 * The dictionary is updated using gradients of alpha from the
 * previous layers.
 * Each forward step solves an elastic net problem:
 * alpha^*(x, D) = argmin_D 1/2\|x-D_l*alpha\|^2 + \lambda1\|alpha\|_1 + \
 *     \lambda2/2\|alpha\|^2.
 * Dl is the local concatenation of convolutional dictionary, which is
 * similar to a convolution kernel.
 */

#include <vector>


#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class CSCLayer : public Layer<Dtype> {
 public:
  explicit CSCLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CSC"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // utilities
  void im2patches_cpu_(const Blob<Dtype> *blob, Blob<Dtype> *patches, bool compute_diff);
  void patches2im_cpu_(const Blob<Dtype> *patches, Blob<Dtype> *blob, bool compute_diff);
  void gemm_Dlalpha_cpu_(const Blob<Dtype> *alpha, Blob<Dtype> *Dlalpha,
      bool prefer_data);
  void permute_num_channels_cpu_(const Blob<Dtype> *top, Blob<Dtype> *patches,
      bool permute_diff);
  void permute_num_channels_gpu_(const Blob<Dtype> *top, Blob<Dtype> *patches,
	  bool permute_diff);
  void caffe_gpu_soft_thresholding_(const int n, Dtype thresh, Dtype *x);
  void caffe_cpu_soft_thresholding_(const int n, Dtype thresh, Dtype *x);


  void im2patches_gpu_(const Blob<Dtype> *blob, Blob<Dtype> *patches, bool compute_diff);
  void patches2im_gpu_(const Blob<Dtype> *patches, Blob<Dtype> *blob, bool compute_diff);
  void gemm_Dlalpha_gpu_(const Blob<Dtype> *alpha, Blob<Dtype> *Dlalpha,
      bool prefer_data);

  Dtype get_lambda1_gpu_data_() const;
  Dtype get_lambda1_gpu_diff_() const;
  void set_lambda1_gpu_data_(Dtype l);
  void set_lambda1_gpu_diff_(Dtype l);

  inline int caffe_zero_norm_(const int n, const Dtype *x) {
	  int val = 0;
	  for (int i = 0; i < n; ++i) {
		  val += (x[i] > 1e-6 || x[i] < -1e-6);
	  }
	  return val;
  }

  // parameters
  Dtype lambda1_;
  Dtype lambda2_;
  int kernel_h_;
  int kernel_w_;
  int num_output_;
  CSCParameter::Boundary boundary_;
  // parameters for admm
  int admm_max_iter_;
  Dtype admm_max_rho_;
  Dtype admm_eta_;
  int lasso_lars_L_;  // integer determining termination of lasso-lars algo.
  // channels of the previous layer
  int channels_;

  vector<int> bottom_patch_shape_;
  vector<int> top_patch_shape_;
  shared_ptr<Blob<Dtype> > alpha_;

  // buffer for forward and backward
  shared_ptr<Blob<Dtype> > alpha_buffer_;
  shared_ptr<Blob<Dtype> > beta_buffer_;
  shared_ptr<Blob<Dtype> > grad_buffer_;
  shared_ptr<Blob<Dtype> > alpha_diff_buffer_;
  shared_ptr<Blob<Dtype> > bottom_patch_buffer_;
  shared_ptr<Blob<Dtype> > bottom_recon_buffer_;


};



}

#endif // CAFFE_CSC_LAYER_HPP_
