/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "caffe2/core/net.h"
#include "caffe2/core/tensor.h"

namespace carml {
using namespace caffe2;
template <typename TargetDev>
class Predictor {
 public:
  using TargetDevice = TargetDev;
  using TensorDevice = Tensor<TargetDevice>;
  using TensorDeviceVector = std::vector<TensorDevice*>;
  using TensorCPUVector = std::vector<TensorCPU>;
  // using TensorVector = std::vector<TensorCPU*>;
  // Runs the `init_net` once, then saves the `run_net` to be executed
  // in `::run`
  Predictor(NetDef& init_net, NetDef& run_net) {
    init_net.mutable_device_option()->set_device_type(CUDA);
    run_net.mutable_device_option()->set_device_type(CUDA);

    auto init_net_ = CreateNet(init_net, &ws_);
    run_net_ = CreateNet(run_net, &ws_);
    CAFFE_ENFORCE(init_net_->Run());

    for (auto ii = 0; ii < run_net.external_input_size(); ii++) {
      input_names_.emplace_back(run_net.external_input(ii));
    }
    for (auto ii = 0; ii < run_net.external_output_size(); ii++) {
      output_names_.emplace_back(run_net.external_output(ii));
    }
  }

  ~Predictor() {}

  // Executes `run_net` on the inputs.
  // The first `inputs.size()` inputs from run_net::external_inputs
  // are shared with the data in `inputs`.

  // Precondition:
  //   inputs.size() <= run_net_.external_inputs.size()

  // Postcondition:
  //   outputs->size() == run_net.external_inputs.size()

  // Returns true on success
  bool run(const TensorCPUVector& inputs, TensorCPUVector* outputs) {
    CAFFE_ENFORCE(inputs.size() <= input_names_.size());

    for (auto ii = 0; ii < inputs.size(); ii++) {
      shareInputTensor(&ws_, input_names_[ii], inputs[ii]);
    }

    if (!run_net_->Run()) {
      return false;
    }

    outputs->resize(output_names_.size());
    for (auto ii = 0; ii < outputs->size(); ii++) {
      (*outputs)[ii] = extractOutputTensor(&ws_, output_names_[ii]);
    }
    return true;
  }

  NetBase* net() const { return run_net_.get(); };

  Workspace* ws() { return &ws_; };

 private:
  std::unique_ptr<NetBase> run_net_;
  std::vector<string> input_names_;
  std::vector<string> output_names_;
  Workspace ws_;

  void enforceIsTensor(Workspace* ws, const std::string& name) {
    auto blob = ws->GetBlob(name);
    CAFFE_ENFORCE(blob, "Blob does not exist: ", name);
    CAFFE_ENFORCE(blob->template IsType<TensorDevice>(),
                  "Blob is not a CPU Tensor: ", name);
  }

  void shareInputTensor(Workspace* ws, const std::string& name,
                        TensorCPU& input) {
    enforceIsTensor(ws, name);
    auto* blob = ws->GetBlob(name);
    CAFFE_ENFORCE(blob, "Blob: ", name, " does not exist");
    auto* tensor = blob->template GetMutable<TensorDevice>();
    tensor->ResizeLike(input);
    tensor->ShareData(input);
  }

  TensorCPU extractOutputTensor(Workspace* ws, const std::string& name) {
    enforceIsTensor(ws, name);
    auto* blob = ws->GetBlob(name);
    CAFFE_ENFORCE(blob, "Blob: ", name, " does not exist");
    if (blob_.IsType<TensorCUDA>()) {
      return TensorCPU(blob->template Get<TensorDevice>());
    }
    return blob->Get<TensorCPU>();
  }
};
}
