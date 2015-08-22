#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
    // Set Weight4pos
  bool USEWEIGHT = this->layer_param_.sig_param().use_weight();
  LOG(INFO) << "USEWEIGHT:" << USEWEIGHT;
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) { 
  // The forward pass computes the sigmoid outputs.
  bool USEWEIGHT = this->layer_param_.sig_param().use_weight();
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  Dtype loss = 0;
  Dtype pos_loss = 0;
  Dtype neg_loss = 0;
  Dtype p_avg_loss = 0;
  Dtype n_avg_loss = 0;

  int pos_targetnum = 0;
  int neg_targetnum = 0;
  // To achieve the same order of magnitude without using weight
  // we mulitply it with channels firstly
  if (USEWEIGHT == true){
    for (int i = 0; i < count; ++i) {
      if (target[i] == 1){
        pos_targetnum++;
        pos_loss -= input_data[i] * (target[i] - (input_data[i] >= 0)) -
          log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
      }
      else{        
        neg_targetnum++;
        neg_loss -= input_data[i] * (target[i] - (input_data[i] >= 0)) -
          log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
      }
    }

    p_avg_loss = pos_targetnum>0? pos_loss / pos_targetnum : 0;
    n_avg_loss = neg_targetnum>0? neg_loss / neg_targetnum : 0;
    loss = 0.5 * (p_avg_loss + n_avg_loss) * channels;
  }
  else{
    for (int i = 0; i < count; ++i) {
    loss -= input_data[i] * (target[i] - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
    }
  }

  top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  bool USEWEIGHT = this->layer_param_.sig_param().use_weight();
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_sub(count, sigmoid_output_data, target, bottom_diff);
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    // TODO: how loss_weight evolve from prop loss
    // TODO: find a way to allocate the weight

    caffe_scal(count, loss_weight / num, bottom_diff);
    if (USEWEIGHT == true)
    {
      Dtype* weight_array = NULL;
      int pos_targetnum = 0;
      int neg_targetnum = 0;
      for (int idx = 0; idx < count; ++idx)
      {
        if (target[idx] == 1)
          pos_targetnum++;
        else
          neg_targetnum++;
      }
      for (int idx = 0; idx < count; ++idx)
      {
        if (target[idx] == 1)
          weight_array[idx] = 0.5*num / pos_targetnum ;
        else
          weight_array[idx] = 0.5*num / neg_targetnum ;
      }     
      caffe_mul(count,bottom_diff,weight_array,bottom_diff);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU_BACKWARD(SigmoidCrossEntropyLossLayer, Backward);
#endif

INSTANTIATE_CLASS(SigmoidCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(SigmoidCrossEntropyLoss);

}  // namespace caffe
