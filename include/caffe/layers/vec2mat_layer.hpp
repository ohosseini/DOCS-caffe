#ifndef CAFFE_VEC2MAT_LAYER_HPP_
#define CAFFE_VEC2MAT_LAYER_HPP_

#include "caffe/layer.hpp"

namespace caffe {

/* Copyright (c) 2015, Omid Hosseini
	Convert a vector output to a Matrix with size (mat_h_,mat_w_)
		(N, C, 1, 1) ----> (N, 1, mat_h_, mat_w_)
		C = mat_h_ * math_w_
*/ 
template <typename Dtype>
class Vec2matLayer : public Layer<Dtype> {
 public:
  explicit Vec2matLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Vec2mat"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
//  virtual inline DiagonalAffineMap<Dtype> coord_map() {
//    return DiagonalAffineMap<Dtype>::identity(2);
//  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int mat_h_, mat_w_, mat_c_;
};

}  // namespace caffe

#endif  // CAFFE_VEC2MAT_LAYER_HPP_
