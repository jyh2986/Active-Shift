#include <vector>
#include <math.h>

#include "caffe/layers/active_shift_layer.hpp"

namespace caffe {


template <typename Dtype>
__global__ void Shift_Forward(const int count, const int num, const int out_channels, const int in_channels,
		const int top_height, const int top_width,
		const int bottom_height, const int bottom_width,
		const Dtype* xpos, const Dtype* ypos,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		const Dtype* bottom_data, Dtype* top) {
	CUDA_KERNEL_LOOP(index, count)
	{
		//n,c,i
		const int top_sp_dim = top_height * top_width;
		const int bottom_sp_dim = bottom_height * bottom_width;
		const int n = index/(out_channels * top_sp_dim);
		const int idx = index%(out_channels * top_sp_dim);
		const int c_out = idx/top_sp_dim;
		const int c_in = c_out/(out_channels/in_channels);
		const int sp_idx = idx%top_sp_dim;
		const int h = sp_idx/top_width;
		const int w = sp_idx%top_width;
		const Dtype* data_im_ptr = bottom_data + n*in_channels*bottom_sp_dim + c_in*bottom_sp_dim;

		const int h_offset = h * stride_h - pad_h;
		const int w_offset = w * stride_w - pad_w;

		Dtype val = 0;
		const Dtype x = xpos[c_in];
		const Dtype y = ypos[c_in];

		int h_im, w_im;
		int x1 = floorf(x);
		int x2 = x1+1;
		int y1 = floorf(y);
		int y2 = y1+1;

		h_im = h_offset + y1;
		w_im = w_offset + x1;
		Dtype q11 = (h_im >= 0 && w_im >= 0 && h_im < bottom_height && w_im < bottom_width) ? data_im_ptr[h_im*bottom_width + w_im] : 0;

		h_im = h_offset + y1;
		w_im = w_offset + x2;
		Dtype q21 = (h_im >= 0 && w_im >= 0 && h_im < bottom_height && w_im < bottom_width) ? data_im_ptr[h_im*bottom_width + w_im] : 0;

		h_im = h_offset + y2;
		w_im = w_offset + x1;
		Dtype q12 = (h_im >= 0 && w_im >= 0 && h_im < bottom_height && w_im < bottom_width) ? data_im_ptr[h_im*bottom_width + w_im] : 0;

		h_im = h_offset + y2;
		w_im = w_offset + x2;
		Dtype q22 = (h_im >= 0 && w_im >= 0 && h_im < bottom_height && w_im < bottom_width) ? data_im_ptr[h_im*bottom_width + w_im] : 0;

		Dtype dx = x-x1;
		Dtype dy = y-y1;

		val = q11*(1-dx)*(1-dy) + q21*dx*(1-dy) + q12*(1-dx)*dy + q22*dx*dy;
		top[index] = val;
	}
}


template <typename Dtype>
__global__ void Shift_Bottom_Backward_Stride1(const int count, const int out_channels, const int in_channels,
		const int top_height, const int top_width, //top
		const int bottom_height, const int bottom_width, //bottom
		const Dtype* xpos, const Dtype* ypos,
		const int pad_h, const int pad_w,
		const Dtype* top_diff, Dtype* bottom_diff) {
	CUDA_KERNEL_LOOP(index, count)
	{
		const int top_sp_dim = top_height * top_width;
		const int bottom_sp_dim = bottom_height * bottom_width;
		const int n = index/(in_channels * bottom_sp_dim);
		const int idx = index%(in_channels * bottom_sp_dim);
		const int c_in = idx/bottom_sp_dim;
		const int c_out = c_in*(out_channels/in_channels);
		const int sp_idx = idx%bottom_sp_dim;
		const int h_col = sp_idx/bottom_width;
		const int w_col = sp_idx%bottom_width;
		const Dtype* top_diff_ptr = top_diff + n*out_channels*top_sp_dim + c_out*top_sp_dim;

		const int h_offset = h_col + pad_h;
		const int w_offset = w_col + pad_w;


		bottom_diff[index] = 0;

		for (int c_off = 0; c_off<out_channels/in_channels; c_off++ )
		{
			Dtype val = 0;
			const Dtype x = -xpos[c_in];  //reverse position
			const Dtype y = -ypos[c_in];

			int h_im, w_im;


			int x1 = floorf(x);
			int x2 = x1+1;
			int y1 = floorf(y);
			int y2 = y1+1;

			//q11
			Dtype q11 = 0;

			h_im = (h_offset + y1);
			w_im = (w_offset + x1);
			q11 = (h_im >= 0 && w_im >= 0 && h_im < top_height && w_im < top_width) ? top_diff_ptr[h_im*top_width + w_im] : 0;

			//q21
			Dtype q21 = 0;

			h_im = (h_offset + y1);
			w_im = (w_offset + x2);
			q21 = (h_im >= 0 && w_im >= 0 && h_im < top_height && w_im < top_width) ? top_diff_ptr[h_im*top_width + w_im] : 0;

			//q12
			Dtype q12 = 0;

			h_im = (h_offset + y2);
			w_im = (w_offset + x1);
			q12 = (h_im >= 0 && w_im >= 0 && h_im < top_height && w_im < top_width) ? top_diff_ptr[h_im*top_width + w_im] : 0;

			//q22
			Dtype q22 = 0;

			h_im = (h_offset + y2);
			w_im = (w_offset + x2);
			q22 = (h_im >= 0 && w_im >= 0 && h_im < top_height && w_im < top_width) ? top_diff_ptr[h_im*top_width + w_im] : 0;

			Dtype dx = x-x1;
			Dtype dy = y-y1;

			val = q11*(1-dx)*(1-dy) + q21*dx*(1-dy) + q12*(1-dx)*dy + q22*dx*dy;
			bottom_diff[index] += val;
			top_diff_ptr+=top_sp_dim;
		}
	}
}


template <typename Dtype>
__global__ void Shift_Bottom_Backward(const int count, const int out_channels, const int in_channels,
		const int top_height, const int top_width, //top
		const int bottom_height, const int bottom_width, //bottom
		const Dtype* xpos, const Dtype* ypos,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		const Dtype* top_diff, Dtype* bottom_diff) {
	CUDA_KERNEL_LOOP(index, count)
	{
		const int top_sp_dim = top_height * top_width;
		const int bottom_sp_dim = bottom_height * bottom_width;
		const int n = index/(in_channels * bottom_sp_dim);
		const int idx = index%(in_channels * bottom_sp_dim);
		const int c_in = idx/bottom_sp_dim;
		const int c_out = c_in*(out_channels/in_channels);
		const int sp_idx = idx%bottom_sp_dim;
		const int h_col = sp_idx/bottom_width;
		const int w_col = sp_idx%bottom_width;
		const Dtype* top_diff_ptr = top_diff + n*out_channels*top_sp_dim + c_out*top_sp_dim;

		const int h_offset = h_col + pad_h;
		const int w_offset = w_col + pad_w;


		bottom_diff[index] = 0;

		for (int c_off = 0; c_off<out_channels/in_channels; c_off++ )
		{
			Dtype val = 0;
			const Dtype x = -xpos[c_in];  //reverse position
			const Dtype y = -ypos[c_in];

			int h_im, w_im;
			int x1 = floorf(x);
			int x2 = x1+1;
			int y1 = floorf(y);
			int y2 = y1+1;

			//q11
			Dtype q11 = 0;

			h_im = (h_offset + y1);
			w_im = (w_offset + x1);
			if(w_im%stride_w == 0 && h_im%stride_h ==0)
			{
				w_im=w_im/stride_w;
				h_im=h_im/stride_h;

				q11 = (h_im >= 0 && w_im >= 0 && h_im < top_height && w_im < top_width) ? top_diff_ptr[h_im*top_width + w_im] : 0;
			}

			//q21
			Dtype q21 = 0;

			h_im = (h_offset + y1);
			w_im = (w_offset + x2);
			if(w_im%stride_w == 0 && h_im%stride_h ==0)
			{
				w_im=w_im/stride_w;
				h_im=h_im/stride_h;

				q21 = (h_im >= 0 && w_im >= 0 && h_im < top_height && w_im < top_width) ? top_diff_ptr[h_im*top_width + w_im] : 0;
			}

			//q12
			Dtype q12 = 0;

			h_im = (h_offset + y2);
			w_im = (w_offset + x1);

			if(w_im%stride_w == 0 && h_im%stride_h ==0)
			{
				w_im=w_im/stride_w;
				h_im=h_im/stride_h;

				q12 = (h_im >= 0 && w_im >= 0 && h_im < top_height && w_im < top_width) ? top_diff_ptr[h_im*top_width + w_im] : 0;
			}

			//q22
			Dtype q22 = 0;

			h_im = (h_offset + y2);
			w_im = (w_offset + x2);

			if(w_im%stride_w == 0 && h_im%stride_h ==0)
			{
				w_im=w_im/stride_w;
				h_im=h_im/stride_h;

				q22 = (h_im >= 0 && w_im >= 0 && h_im < top_height && w_im < top_width) ? top_diff_ptr[h_im*top_width + w_im] : 0;
			}

			Dtype dx = x-x1;
			Dtype dy = y-y1;

			val = q11*(1-dx)*(1-dy) + q21*dx*(1-dy) + q12*(1-dx)*dy + q22*dx*dy;
			bottom_diff[index] += val;
			top_diff_ptr+=top_sp_dim;
		}
	}
}



template <typename Dtype>
__inline__ __device__ void myAtomicAdd(Dtype *buf, Dtype val);

template <>
__inline__ __device__ void myAtomicAdd<float>(float *buf, float val)
{
	atomicAdd(buf, val);
}

template <>
__inline__ __device__ void myAtomicAdd<double>(double *buf, double val)
{
	//Not Supported
}



template <typename Dtype>
__global__ void Shift_Position_Backward(const int count, const int num, const int out_channels, const int in_channels,
		const int top_height, const int top_width,
		const int bottom_height, const int bottom_width,
		const Dtype* xpos, const Dtype* ypos,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		const Dtype* bottom_data, const Dtype* top_diff, Dtype* pos_temp_buff_x, Dtype* pos_temp_buff_y) {
	CUDA_KERNEL_LOOP(index, count)
	{
		//n,c,i
		const int top_sp_dim = top_height * top_width;
		const int bottom_sp_dim = bottom_height * bottom_width;
		const int n = index/(out_channels * top_sp_dim);
		const int idx = index%(out_channels * top_sp_dim);
		const int c_mul = out_channels/in_channels;
		const int c_out = idx/top_sp_dim;
		const int c_in = c_out/c_mul;
		const int sp_idx = idx%top_sp_dim;
		const int h = sp_idx/top_width;
		const int w = sp_idx%top_width;
		const Dtype* data_im_ptr = bottom_data + n*in_channels*bottom_sp_dim + c_in*bottom_sp_dim;

		const int h_offset = h * stride_h - pad_h;
		const int w_offset = w * stride_w - pad_w;

		//output : 2*(C) x (1*H*W)
		const int kernel_offset = top_sp_dim;
		const int c_off = c_out % c_mul;

		Dtype* out_ptr_x = pos_temp_buff_x + (c_in*c_mul+c_off)*kernel_offset + (0*top_sp_dim + sp_idx); //(c_in*K,c_off,0,h,w)
		Dtype* out_ptr_y = pos_temp_buff_y + (c_in*c_mul+c_off)*kernel_offset + (0*top_sp_dim + sp_idx); //(c_in*K,c_off,0,h,w)



		Dtype val_x = 0, val_y = 0;

		const Dtype shiftX = xpos[c_in];
		const Dtype shiftY = ypos[c_in];


		const int ix1 = floorf(shiftX);
		const int ix2 = ix1+1;
		const int iy1 = floorf(shiftY);
		const int iy2 = iy1+1;
		const Dtype dx = shiftX-ix1;
		const Dtype dy = shiftY-iy1;

		const int h_im1 = h_offset + iy1;
		const int h_im2 = h_offset + iy2;

		const int w_im1 = w_offset + ix1;
		const int w_im2 = w_offset + ix2;

		const Dtype q11 = (h_im1 >= 0 && w_im1 >= 0 && h_im1 < bottom_height && w_im1 < bottom_width) ? data_im_ptr[h_im1*bottom_width + w_im1] : 0;
		const Dtype q21 = (h_im1 >= 0 && w_im2 >= 0 && h_im1 < bottom_height && w_im2 < bottom_width) ? data_im_ptr[h_im1*bottom_width + w_im2] : 0;
		const Dtype q12 = (h_im2 >= 0 && w_im1 >= 0 && h_im2 < bottom_height && w_im1 < bottom_width) ? data_im_ptr[h_im2*bottom_width + w_im1] : 0;
		const Dtype q22 = (h_im2 >= 0 && w_im2 >= 0 && h_im2 < bottom_height && w_im2 < bottom_width) ? data_im_ptr[h_im2*bottom_width + w_im2] : 0;

		val_x = (1-dy)*(q21-q11)+dy*(q22-q12);
		val_y = (1-dx)*(q12-q11)+dx*(q22-q21);

		//printf("(%d:%d), (%d,%d) : %f / %d:%f\n", c,f,n,sp_idx,val, diff_idx, val * top_diff[index] * scale);

		const Dtype diff_scale = top_diff[index];

		//reduce along batch dimension
		myAtomicAdd<Dtype>(out_ptr_x, val_x * diff_scale);
		myAtomicAdd<Dtype>(out_ptr_y, val_y * diff_scale);

		out_ptr_x += c_mul*kernel_offset;
		out_ptr_y += c_mul*kernel_offset;
	}
}

template <typename Dtype>
__global__ void applyShiftConstraint(const int n, const Dtype lattice_decay,
		const Dtype* xpos_data, const Dtype* ypos_data, Dtype* xpos_diff, Dtype* ypos_diff, const bool normalize, const float clip_gradient) {
  CUDA_KERNEL_LOOP(index, n) {
	  const Dtype xi = xpos_data[index];
	  const Dtype yi = ypos_data[index];
	  const Dtype ri = sqrt(xi*xi+yi*yi);

      //Lattice Regularization
	  if(lattice_decay>0)
	  {
		  Dtype p = lattice_decay;

		  const Dtype dx = xi - nearbyint(xi);
		  const Dtype dy = yi - nearbyint(yi);

		  xpos_diff[index] += dx*p;
		  ypos_diff[index] += dy*p;
	  }

	  if(normalize) //normalize
	  {
		  const Dtype dx = xpos_diff[index];
		  const Dtype dy = ypos_diff[index];
		  const Dtype dr = sqrt(dx*dx+dy*dy);

		  if(dr!=0)
		  {
			  xpos_diff[index] = dx/dr;
			  ypos_diff[index] = dy/dr;
		  }
	  }
	  else if(clip_gradient!=0)
	  {
		  const Dtype dx = xpos_diff[index];
		  const Dtype dy = ypos_diff[index];
		  const Dtype dr = sqrt(dx*dx+dy*dy);

		  if(dr>clip_gradient)
		  {
			  xpos_diff[index] = dx/dr*clip_gradient;
			  ypos_diff[index] = dy/dr*clip_gradient;
		  }
	  }

  }
}

template <typename Dtype>
void ActiveShiftLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* xpos = this->blobs_[0]->gpu_data();
  const Dtype* ypos = this->blobs_[1]->gpu_data();

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  int count = top[0]->count();
  Shift_Forward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
  <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
		  count, num_, num_output_, channels_,
		  top_height_, top_width_,
		  bottom_height_, bottom_width_,
		  xpos, ypos,
		  pad_h_, pad_w_,
		  stride_h_, stride_w_,
		  bottom_data, top_data);
}

template <typename Dtype>
void ActiveShiftLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* xpos = this->blobs_[0]->gpu_data();
  const Dtype* ypos = this->blobs_[1]->gpu_data();

  Dtype* diff_temp = p_temp_buff_->mutable_gpu_data();
  const Dtype *diff_muliplier = temp_multiplier_.gpu_data();

  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();


  //bottom diff
  int count = bottom[0]->count();

  if(stride_h_==1 && stride_w_==1)
  {
	  Shift_Bottom_Backward_Stride1<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
	  <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
			  count, num_output_, channels_,
			  top_height_, top_width_,
			  bottom_height_, bottom_width_,
			  xpos, ypos,
			  pad_h_, pad_w_,
			  top_diff, bottom_diff);
  }
  else
  {
	  Shift_Bottom_Backward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
	  <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
			  count, num_output_, channels_,
			  top_height_, top_width_,
			  bottom_height_, bottom_width_,
			  xpos, ypos,
			  pad_h_, pad_w_,
			  stride_h_, stride_w_,
			  top_diff, bottom_diff);
  }

  //Position diff
  Dtype* xposDiff = this->blobs_[0]->mutable_gpu_diff();
  Dtype* yposDiff = this->blobs_[1]->mutable_gpu_diff();

  if (this->param_propagate_down_[0] && this->param_propagate_down_[1] && Caffe::current_iter()>=warming_up_)
  {
	  count = top[0]->count();
	  int buf_offset = num_output_ * top_height_ * top_width_;	//(C*1) x (1*H*W)
	  caffe_gpu_set(2*buf_offset, Dtype(0), diff_temp);

	  Shift_Position_Backward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
	  <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
			  count, num_, num_output_, channels_,
			  top_height_, top_width_,
			  bottom_height_, bottom_width_,
			  xpos, ypos,
			  pad_h_, pad_w_,
			  stride_h_, stride_w_,
			  bottom_data, top_diff, diff_temp,  diff_temp + buf_offset);

	  caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_, channel_multiplier_* top_height_* top_width_, 1.,
			  diff_temp, diff_muliplier, 1., xposDiff);
	  caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_, channel_multiplier_* top_height_* top_width_, 1.,
			  diff_temp + buf_offset, diff_muliplier, 1., yposDiff);

	  // Position Regularization
	  count = num_output_;
	  applyShiftConstraint<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
	  <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
			  count , lattice_decay_,
			  xpos, ypos, xposDiff, yposDiff, normalize_diff_, clip_gradient_);
  }
  else
  {
	  caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), xposDiff);
	  caffe_gpu_set(this->blobs_[1]->count(), Dtype(0), yposDiff);
  }
}

template <>
void ActiveShiftLayer<double>::Backward_gpu(const vector<Blob<double>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<double>*>& bottom) {
	NOT_IMPLEMENTED;
}


INSTANTIATE_LAYER_GPU_FUNCS(ActiveShiftLayer);

}  // namespace caffe
