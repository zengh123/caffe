#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
    const Dtype* target_g = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_copy(count, sigmoid_output_data, bottom_diff);
    caffe_gpu_axpy(count, Dtype(-1), target_g, bottom_diff);
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_gpu_scal(count, loss_weight / num, bottom_diff);

    bool USEWEIGHT = this->layer_param_.sig_param().use_weight();
    LOG(INFO) << "USEWEIGHT" << USEWEIGHT;
    const Dtype* target = bottom[1]->cpu_data();

    if (USEWEIGHT == true)
    {
      Dtype weight_array[count];
      memset(weight_array,0,sizeof(weight_array));
      Dtype *d_weight_array;
      cudaMalloc((void **)&d_weight_array,sizeof(weight_array));
      int pos_targetnum = 0;
      int neg_targetnum = 0;

      for (int idx = 0; idx < count; ++idx)
      {
        if (target[idx] == 1)
        {
          pos_targetnum++;
        }
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


      cudaMemcpy(d_weight_array,weight_array,sizeof(weight_array),cudaMemcpyHostToDevice);
      caffe_gpu_mul(count,bottom_diff,d_weight_array,bottom_diff);

 
    }
  }
}

INSTANTIATE_LAYER_GPU_BACKWARD(SigmoidCrossEntropyLossLayer);


}  // namespace caffe
