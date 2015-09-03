#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

//TODO: these include remained discussed
#include <cmath>
#include <cmath>


#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
//TODO: overlook top K and ignore 
}

template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
    << "The data and label should have the same number of instances";
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels())
    << "The data and label should have the same number of channels";
  CHECK_EQ(bottom[0]->height(), bottom[1]->height())
    << "The data and label should have the same height";
  CHECK_EQ(bottom[0]->width(), bottom[1]->width())
    << "The data and label should have the same width";
  // Top will contain:
  // top[0] = Sensitivity or Recall (TP/P),
  // top[1] = Specificity (TN/N),
  // top[2] = Harmonic Mean of Sens and Spec, (2/(P/TP+N/TN))
  // top[3] = Precision (TP / (TP + FP))
  // top[4] = F1 Score (2 TP / (2 TP + FP + FN))
  vector<int> top_shape(4,1);  // Accuracy is a scalar; 0 axes.
  top_shape[1] = 5;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  //todo: why use dtype instead of int
  Dtype true_positive = 0;
  Dtype false_positive = 0;
  Dtype true_negative = 0;
  Dtype false_negative = 0;
  int count_pos = 0;
  int count_neg = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  // Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  int count = bottom[0]->count();
  for (int ind = 0; ind < count; ++ind) {
	  int label = static_cast<int>(bottom_label[ind]);
	  if (label == 1 )
	  {
	  	true_positive += (bottom_data[ind] >= 0);
	  	false_negative += (bottom_data[ind] < 0);
	  	count_pos ++;
	  }
	  if (label == 0)
	  {
	  	true_negative += (bottom_data[ind] < 0);
	  	false_positive += (bottom_data[ind] >= 0);
	  	count_neg ++;
	  }

  }
  Dtype sensitivity = (count_pos > 0)? (true_positive / count_pos) : 0;
  Dtype specificity = (count_neg > 0)? (true_negative / count_neg) : 0;
  Dtype harmmean = ((count_pos + count_neg) > 0)?
    2 / (count_pos / true_positive + count_neg / true_negative) : 0;
//  Dtype precission = (true_positive > 0)?
//   (true_positive / (true_positive + false_positive)) : 0;
  Dtype precission = (count > 0)?
    ( (true_positive + true_negative) / count) : 0;
  Dtype f1_score = (true_positive > 0)?
    2 * true_positive /
    (2 * true_positive + false_positive + false_negative) : 0;

  //LOG(INFO) << "Sensitivity: " << sensitivity;
  //LOG(INFO) << "Specificity: " << specificity;
  //LOG(INFO) << "Harmonic Mean of Sens and Spec: " << harmmean;
  //LOG(INFO) << "Precission: " << precission;
  //LOG(INFO) << "F1 Score: " << f1_score;

  top[0]->mutable_cpu_data()[0] = sensitivity;
  top[0]->mutable_cpu_data()[1] = specificity;
  top[0]->mutable_cpu_data()[2] = harmmean;
  top[0]->mutable_cpu_data()[3] = precission;
  top[0]->mutable_cpu_data()[4] = f1_score;
}

INSTANTIATE_CLASS(MultiLabelAccuracyLayer);
REGISTER_LAYER_CLASS(MultiLabelAccuracy);


}