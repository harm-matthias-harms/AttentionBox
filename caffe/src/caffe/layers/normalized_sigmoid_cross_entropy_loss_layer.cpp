#include <vector>
#include <cmath>
#include "caffe/layers/normalized_sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

using std::floor;
namespace caffe {

template <typename Dtype>
void NormalizedSigmoidCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void NormalizedSigmoidCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "NORMALIZED_SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count in first two bottoms.";
  // CHECK_EQ(bottom[0]->shape(0), bottom[2]->shape(0));
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void NormalizedSigmoidCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();

  const int width = bottom[0]->shape(2); const int height = bottom[0]->shape(3);
  Dtype loss = 0;
  for (int i = 0; i < count; ++i) {
    // loss -= input_data[i] * (target[i] - (input_data[i] >= 0)) -
    //     log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
    loss -= ( input_data[i] * (target[i] - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0))) );
  }
  top[0]->mutable_cpu_data()[0] = loss / ( width * height * num );
}

template <typename Dtype>
void NormalizedSigmoidCrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[2]) { LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs."; }
  if (propagate_down[1]) { LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs."; }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target     = bottom[1]->cpu_data();

    const int width = bottom[0]->shape(2);const int height = bottom[0]->shape(3);

    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_sub(count, sigmoid_output_data, target, bottom_diff);
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight / ( num * width * height ), bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU_BACKWARD(NormalizedSigmoidCrossEntropyLossLayer, Backward);
#endif

INSTANTIATE_CLASS(NormalizedSigmoidCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(NormalizedSigmoidCrossEntropyLoss);

}  // namespace caffe
