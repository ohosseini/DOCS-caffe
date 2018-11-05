/* Copyright (c) 2015, Omid Hosseini
	Convert a vector output to a 3D Matrix with size (mat_c_,mat_h_,mat_w_)
		(N, C, 1, 1) ----> (N, mat_c_, mat_h_, mat_w_)
		C = mat_c_ * mat_h_ * math_w_
*/ 
#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/layers/vec2mat_layer.hpp"

namespace caffe {

template <typename Dtype>
void Vec2matLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if (this->layer_param_.vec2mat_param().has_mat_c())
	mat_c_ = this->layer_param_.vec2mat_param().mat_c();
  else
	mat_c_ = 1;
  mat_h_ = this->layer_param_.vec2mat_param().mat_h();
  mat_w_ = this->layer_param_.vec2mat_param().mat_w();
  CHECK(mat_c_*mat_h_*mat_w_ == bottom[0]->channels()) << 
	"Error(Vec2mat): top and bottom must have the same number of elements.";
}

template <typename Dtype>
void Vec2matLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(), mat_c_, mat_h_, mat_w_);
}

template <typename Dtype>
void Vec2matLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  caffe_copy(top[0]->count(), bottom_data, top_data);
}

template <typename Dtype>
void Vec2matLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  if (propagate_down[0]) {
    caffe_set(bottom[0]->count(), static_cast<Dtype>(0), bottom_diff);
    caffe_copy(top[0]->count(), top_diff, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(Vec2matLayer);
#endif

INSTANTIATE_CLASS(Vec2matLayer);
REGISTER_LAYER_CLASS(Vec2mat);

}  // namespace caffe
