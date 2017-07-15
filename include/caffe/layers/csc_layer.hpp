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
#include "caffe/util/csc_helpers.hpp"

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
  void extract_patches_cpu_(const Blob<Dtype> *blob, Blob<Dtype> *patches);
  void aggregate_patches_cpu_(const Blob<Dtype> *patches, Blob<Dtype> *blob);
  void gemm_Dlalpha_cpu_(const Blob<Dtype> *alpha, Blob<Dtype> *Dlalpha);

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
  SpBlob<Dtype> spalpha_;

};



}

#endif // CAFFE_CSC_LAYER_HPP_
