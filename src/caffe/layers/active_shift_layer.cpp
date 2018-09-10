#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/active_shift_layer.hpp"

namespace caffe {

template <typename Dtype>
void ActiveShiftLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	ConvolutionParameter conv_param = this->layer_param_.convolution_param();
	CHECK(!conv_param.has_num_output()) << "DEPRECATED!! use asl_param instead!!";

	ASLParameter asl_param = this->layer_param_.asl_param();

	//Stride
	if (asl_param.has_stride_h() || asl_param.has_stride_w()) {
		CHECK(!asl_param.has_stride()) << "Either stride or stride_h/w should be specified; not both.";

		stride_h_ = asl_param.stride_h();
		stride_w_ = asl_param.stride_w();
	} else {
		stride_h_ = asl_param.stride();
		stride_w_ = asl_param.stride();
	}

	//Pad
	if (asl_param.has_pad_h() || asl_param.has_pad_w()) {
		CHECK(!asl_param.has_pad()) << "Either pad or pad_h/w should be specified; not both.";

		pad_h_ = asl_param.pad_h();
		pad_w_ = asl_param.pad_w();
	} else {
		pad_h_ = asl_param.pad();
		pad_w_ = asl_param.pad();
	}

	//Channel settings
	channels_ = bottom[0]->shape(1);
	channel_multiplier_ = asl_param.multiplier();
	num_output_ = channels_ * channel_multiplier_;

	// Handle the parameters: weights and biases.
	// - blobs_[0] holds the filter x position
	// - blobs_[1] holds the filter y position
	this->blobs_.resize(2);
	this->blobs_[0].reset(new Blob<Dtype>(num_output_,1,1,1));
	this->blobs_[1].reset(new Blob<Dtype>(num_output_,1,1,1));

	//Filler
	base_radius_ = asl_param.base_radius();
	CHECK(base_radius_>=0) << "Base Radius should be greater or equal than 0";

	//gradient setup
	normalize_diff_ = asl_param.normalize();
	clip_gradient_ = asl_param.clip_gradient();
	warming_up_ = asl_param.warming_up();

	CHECK(!(normalize_diff_ && clip_gradient_!=0)) << "Normalize and clip_gradient should not be used at the same time.";

	//Setup Warming up
	if(warming_up_!=0)
	{
		Dtype* xposDiff = this->blobs_[0]->mutable_cpu_diff();
		Dtype* yposDiff = this->blobs_[1]->mutable_cpu_diff();

		caffe_set(this->blobs_[0]->count(), Dtype(0), xposDiff);
		caffe_set(this->blobs_[1]->count(), Dtype(0), yposDiff);
	}

	//Regularization
	lattice_decay_ = asl_param.lattice_decay();
	CHECK_GE(lattice_decay_, 0);

    //Bottom & top spatial size
    bottom_height_ = bottom[0]->height();
    bottom_width_ = bottom[0]->width();
    top_height_ = (bottom[0]->height() + 2 * pad_h_ - 1)/ stride_h_ + 1;
    top_width_ = (bottom[0]->width() + 2 * pad_w_ - 1)/ stride_w_ + 1;

	// Initialize Params
	Dtype *xpos_data = this->blobs_[0]->mutable_cpu_data();
	Dtype *ypos_data = this->blobs_[1]->mutable_cpu_data();

	if(base_radius_==0)
	{
		caffe_set(num_output_, Dtype(0.), xpos_data);
		caffe_set(num_output_, Dtype(0.), ypos_data);
	}
	else
	{
		FillerParameter filler_param;

#ifdef TEST_ASHIFT_ENV
		filler_param.set_min(0.3);
		filler_param.set_max(0.7);
#else
		filler_param.set_min(-base_radius_);
		filler_param.set_max(+base_radius_);
#endif

		UniformFiller<Dtype> filler(filler_param);
		filler.Fill(this->blobs_[0].get());
		filler.Fill(this->blobs_[1].get());
	}

	//
	vector<int> diff_multiplier_shape(1, channel_multiplier_*top_height_* top_width_);
	temp_multiplier_.Reshape(diff_multiplier_shape);
	caffe_set(temp_multiplier_.count(), Dtype(1), temp_multiplier_.mutable_cpu_data());

	//temp buff for position diff
	p_temp_buff_ = this->get_share_buffer();
	p_temp_buff_ = (p_temp_buff_ == NULL)?&temp_buff_:p_temp_buff_;
	p_temp_buff_->Reshape(2, num_output_, top_height_,top_width_);	// 2 for x,y pos_diff


	//reset backprob
	this->param_propagate_down_.resize(2);
	this->param_propagate_down_[0]=true;
	this->param_propagate_down_[1]=true;
}

template <typename Dtype>
void ActiveShiftLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	num_ = bottom[0]->shape(0);

	// Shape the tops.
	top[0]->Reshape(num_,num_output_,top_height_,top_width_);
}


inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

#define PSEUDO_FLOOR(X) ((int)((X)+32768)-32768)  //valid in |X|<32768

template <typename Dtype>
void ActiveShiftLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	const Dtype* xpos = this->blobs_[0]->cpu_data();
	const Dtype* ypos = this->blobs_[1]->cpu_data();
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	const int bottom_sp_dim = bottom_height_ * bottom_width_;

	int index = 0;

#if 1
	for (int n = 0; n < num_; ++n) {
		for (int c = 0; c < channels_; ++c) {
			const Dtype x = xpos[c];
			const Dtype y = ypos[c];

			const int x1 = PSEUDO_FLOOR(x);
			const int x2 = x1+1;
			const int y1 = PSEUDO_FLOOR(y);
			const int y2 = y1+1;

			const Dtype dx = x-x1;
			const Dtype dy = y-y1;

			const Dtype d11 = (1-dx)*(1-dy);
			const Dtype d21 = dx*(1-dy);
			const Dtype d12 = (1-dx)*dy;
			const Dtype d22 = dx*dy;

			const Dtype* data_im_ptr = bottom_data + n*channels_*bottom_sp_dim + c*bottom_sp_dim;

			for (int h = 0; h < top_height_; ++h) {
				for (int w = 0; w < top_width_; ++w) {
					const int h_offset = h * stride_h_ - pad_h_;
					const int w_offset = w * stride_w_ - pad_w_;


					const int w_im1 = w_offset + x1;
					const int w_im2 = w_offset + x2;
					const int h_im1 = h_offset + y1;
					const int h_im2 = h_offset + y2;

					const bool in_w1 = is_a_ge_zero_and_a_lt_b(w_im1, bottom_width_);
					const bool in_w2 = is_a_ge_zero_and_a_lt_b(w_im2, bottom_width_);
					const bool in_h1 = is_a_ge_zero_and_a_lt_b(h_im1, bottom_height_);
					const bool in_h2 = is_a_ge_zero_and_a_lt_b(h_im2, bottom_height_);

					//Dtype q11 = (h_im >= 0 && w_im >= 0 && h_im < bottom_height && w_im < bottom_width) ? data_im_ptr[h_im*bottom_width + w_im] : 0;
					Dtype q11 = in_w1 * in_h1 * data_im_ptr[h_im1*bottom_width_ + w_im1];

					//Dtype q21 = (h_im >= 0 && w_im >= 0 && h_im < bottom_height && w_im < bottom_width) ? data_im_ptr[h_im*bottom_width + w_im] : 0;
					Dtype q21 = in_w2 * in_h1 * data_im_ptr[h_im1*bottom_width_ + w_im2];

					//Dtype q12 = (h_im >= 0 && w_im >= 0 && h_im < bottom_height && w_im < bottom_width) ? data_im_ptr[h_im*bottom_width + w_im] : 0;
					Dtype q12 = in_w1 * in_h2 * data_im_ptr[h_im2*bottom_width_ + w_im1];

					//Dtype q22 = (h_im >= 0 && w_im >= 0 && h_im < bottom_height && w_im < bottom_width) ? data_im_ptr[h_im*bottom_width + w_im] : 0;
					Dtype q22 = in_w2 * in_h2 * data_im_ptr[h_im2*bottom_width_ + w_im2];


					Dtype val = q11*d11 + q21*d21 + q12*d12 + q22*d22;
					top_data[index] = val;

					index++;
				}
			}
		}
	}
#else

	/*if(stride_w == 1 && stride_h ==1)
	{
		caffe_set(top[0]->count(), Dtype(0.), top_data);

		for (int n = 0; n < num_; ++n) {
			for (int c = 0; c < channels; ++c) {
				const int x = (int)xpos[c];
				const int y = (int)ypos[c];

				const Dtype* data_im_ptr = bottom_data + n*channels*bottom_sp_dim + c*bottom_sp_dim;

				for (int h = 0; h < top_height; ++h) {
					Dtype* Y = &top_data[index];

					if(is_a_ge_zero_and_a_lt_b(h+y, bottom_height))
					{
						const Dtype* X = &data_im_ptr[(h+y)*bottom_width+std::max(0,x)];
						memcpy(Y, X, sizeof(Dtype) * (bottom_width - abs(x)));
					}

					index += top_width;
				}
			}
		}
	}
	else*/
	{
		for (int n = 0; n < num_; ++n) {
			for (int c = 0; c < channels_; ++c) {
				const int x = (int)xpos[c];
				const int y = (int)ypos[c];

				const Dtype* data_im_ptr = bottom_data + n*channels_*bottom_sp_dim + c*bottom_sp_dim;

				for (int h = 0; h < top_height; ++h) {
					for (int w = 0; w < top_width; ++w) {
						const int h_offset = h * stride_h_ - pad_h_;
						const int w_offset = w * stride_w_ - pad_w_;

						int h_im, w_im;

						h_im = h_offset + y;
						w_im = w_offset + x;
						//Dtype q11 = (h_im >= 0 && w_im >= 0 && h_im < bottom_height && w_im < bottom_width) ? data_im_ptr[h_im*bottom_width + w_im] : 0;
						Dtype val = (IS_IN_BOUND==1)?data_im_ptr[h_im*bottom_width_ + w_im]:0;

						top_data[index] = val;

						index++;
					}
				}
			}
		}
	}


#endif
}

template <typename Dtype>
void ActiveShiftLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	NOT_IMPLEMENTED;
}



#ifdef CPU_ONLY
STUB_GPU(ActiveShiftLayer);
#endif

INSTANTIATE_CLASS(ActiveShiftLayer);
REGISTER_LAYER_CLASS(ActiveShift);

}  // namespace caffe
