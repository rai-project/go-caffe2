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

using namespace caffe2;

template <typename TargetDev>
class Predictor {
 public:
  using TargetDevice = TargetDev;
  using TensorDevice = Tensor<TargetDevice>;
  using TensorDeviceVector = std::vector<TensorDevice*>;
  // Runs the `init_net` once, then saves the `run_net` to be executed
  // in `::run`

Predictor(
    const NetDef& init_net,
    const NetDef& run_net,
    Workspace* parent = nullptr)
    : run_net_(run_net), ws_(parent) {
  CAFFE_ENFORCE(ws_.RunNetOnce(init_net));
  CAFFE_ENFORCE(ws_.CreateNet(run_net));
}

~Predictor() {}


  // Executes `run_net` on the inputs.
  // The first `inputs.size()` inputs from run_net::external_inputs
  // are shared with the data in `inputs`.

  // Precondition:
  //   inputs.size() <= run_net_.external_inputs.size()

  // Postcondition:
  //   outputs->size() == run_net.external_inputs.size()

bool run(const TensorDeviceVector& inputs, TensorDeviceVector* outputs) {
  CAFFE_ENFORCE(inputs.size() <= run_net_.external_input_size());
  for (auto i = 0; i < inputs.size(); ++i) {
    shareInputTensorDevice(&ws_, run_net_.external_input(i), inputs[i]);
  }

  if (!ws_.RunNet(run_net_.name())) {
    return false;
  }

  outputs->resize(run_net_.external_output_size());
  for (auto i = 0; i < outputs->size(); ++i) {
    (*outputs)[i] = extractOutputTensorDevice(&ws_, run_net_.external_output(i));
  }
  return true;
}

  const NetDef& def() const {
    return run_net_;
  };

  Workspace* ws() {
    return &ws_;
  };

 private:
  NetDef run_net_;
  Workspace ws_;

void enforceIsTensorDevice(Workspace* ws, const std::string& name) {
  auto blob = ws->GetBlob(name);
  CAFFE_ENFORCE(blob, "Blob does not exist: ", name);
  CAFFE_ENFORCE(
      blob->template IsType<TensorDevice>(), "Blob is not a CPU TensorDevice: ", name);
}

void shareInputTensorDevice(
    Workspace* ws,
    const std::string& name,
    TensorDevice* input) {
  enforceIsTensorDevice(ws, name);
  auto* blob = ws->GetBlob(name);
  CAFFE_ENFORCE(blob, "Blob: ", name, " does not exist");
  auto* tensor = blob->template GetMutable<TensorDevice>();
  tensor->ResizeLike(*input);
  tensor->ShareData(*input);
}

TensorDevice* extractOutputTensorDevice(Workspace* ws, const std::string& name) {
  enforceIsTensorDevice(ws, name);
  auto* blob = ws->GetBlob(name);
  CAFFE_ENFORCE(blob, "Blob: ", name, " does not exist");
  return blob->template GetMutable<TensorDevice>();
}
};

