#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/active_shift_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#define CH 4

namespace caffe {

template <typename TypeParam>
class ActiveShiftLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ActiveShiftLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, CH, 3, 4)),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~ActiveShiftLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};


template <typename Dtype>
void MyCheckGradientExhaustive(Layer<Dtype>* layer,
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top,
    int check_bottom = -1) {

  vector<bool> check_param;
  check_param.resize(3,false);
  check_param[0] = true;	//pos_x
  check_param[1] = true;	//pos_y
  check_param[2] = true;	//bottom

  GradientChecker<Dtype> checker(1e-2, 1e-3);
  layer->SetUp(bottom, top);
  CHECK_GT(top.size(), 0) << "Exhaustive mode requires at least one top blob.";
  // LOG(ERROR) << "Exhaustive Mode.";
  for (int i = 0; i < top.size(); ++i) {
    // LOG(ERROR) << "Exhaustive: blob " << i << " size " << top[i]->count();
    for (int j = 0; j < top[i]->count(); ++j) {
      // LOG(ERROR) << "Exhaustive: blob " << i << " data " << j;
      checker.CheckGradientSingle(layer, bottom, top, check_bottom, i, j, false, &check_param);
    }
  }
}

TYPED_TEST_CASE(ActiveShiftLayerTest, GPUDevice<float>);


TYPED_TEST(ActiveShiftLayerTest, TestGradientstride1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  ActiveShiftLayer<Dtype> layer(layer_param);

  MyCheckGradientExhaustive(&layer, this->blob_bottom_vec_,
	  this->blob_top_vec_);
}


TYPED_TEST(ActiveShiftLayerTest, TestGradient_stride2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ASLParameter* asl_param = layer_param.mutable_asl_param();

  asl_param->set_stride(2);
  //asl_param->set_pad(1);

  ActiveShiftLayer<Dtype> layer(layer_param);

  MyCheckGradientExhaustive(&layer, this->blob_bottom_vec_,
	  this->blob_top_vec_);
}

TYPED_TEST(ActiveShiftLayerTest, TestGradient_with_padding) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ASLParameter* asl_param = layer_param.mutable_asl_param();

  asl_param->set_pad_w(2);
  asl_param->set_pad_h(3);
  asl_param->set_stride(2);

  ActiveShiftLayer<Dtype> layer(layer_param);

  MyCheckGradientExhaustive(&layer, this->blob_bottom_vec_,
	  this->blob_top_vec_);
}


TYPED_TEST(ActiveShiftLayerTest, TestGradient_stride1_2x) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ASLParameter* asl_param = layer_param.mutable_asl_param();

  asl_param->set_multiplier(2);

  ActiveShiftLayer<Dtype> layer(layer_param);

  MyCheckGradientExhaustive(&layer, this->blob_bottom_vec_,
	  this->blob_top_vec_);
}


TYPED_TEST(ActiveShiftLayerTest, TestGradient_stride2_2x) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ASLParameter* asl_param = layer_param.mutable_asl_param();

  asl_param->set_stride(2);
  asl_param->set_multiplier(2);

  ActiveShiftLayer<Dtype> layer(layer_param);

  MyCheckGradientExhaustive(&layer, this->blob_bottom_vec_,
	  this->blob_top_vec_);
}

TYPED_TEST(ActiveShiftLayerTest, TestGradient_with_padding_2x) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ASLParameter* asl_param = layer_param.mutable_asl_param();

  asl_param->set_stride(2);
  asl_param->set_pad_w(2);
  asl_param->set_pad_h(3);
  asl_param->set_multiplier(2);

  ActiveShiftLayer<Dtype> layer(layer_param);

  MyCheckGradientExhaustive(&layer, this->blob_bottom_vec_,
	  this->blob_top_vec_);
}

}  // namespace caffe
