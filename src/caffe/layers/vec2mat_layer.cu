/* Copyright (c) 2015, Omid Hosseini
	Convert a vector output to a Matrix with size (mat_h_,mat_w_)
		(N, C, 1, 1) ----> (N, 1, mat_h_, mat_w_)
		C = mat_h_ * math_w_
*/ 

#include <vector>

#include "caffe/layers/vec2mat_layer.hpp"

namespace caffe {

template <typename Dtype>
void Vec2matLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  caffe_copy(top[0]->count(), bottom_data, top_data);
}

template <typename Dtype>
void Vec2matLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  if (propagate_down[0]) {
    caffe_gpu_set(bottom[0]->count(), static_cast<Dtype>(0), bottom_diff);
    caffe_copy(top[0]->count(), top_diff, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(Vec2matLayer);

}  // namespace caffe
