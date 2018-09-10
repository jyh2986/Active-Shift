#ifndef CAFFE_ASHIFT_LAYER_HPP_
#define CAFFE_ASHIFT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

//#define TEST_ASHIFT_ENV

namespace caffe {

template <typename Dtype>
class ActiveShiftLayer: public Layer<Dtype> {
 public:
  explicit ActiveShiftLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  virtual inline const char* type() const { return "ActiveShift"; }
  
 protected:
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 private:
  //channel info
  int num_;
  int channels_;
  int num_output_;
  int channel_multiplier_; // out_channel/in_channel

  //spatial info
  int bottom_width_;
  int bottom_height_;
  int top_width_;
  int top_height_;

  //stride & pad
  int pad_w_;
  int pad_h_;
  int stride_w_;
  int stride_h_;

  //Shift initialization
  Dtype base_radius_;

  //Gradient control
  bool normalize_diff_;	//Normalization
  bool clip_gradient_;	//Clip Gradient
  int warming_up_;	//Warm-up

  //Regularization
  Dtype lattice_decay_;

  //for pos diff
  Blob<Dtype>* p_temp_buff_;
  Blob<Dtype> temp_multiplier_;
  Blob<Dtype> temp_buff_;	//temporary buffer for backprob

};


}  // namespace caffe

#endif  // CAFFE_ASHIFT_LAYER_HPP_
