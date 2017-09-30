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

namespace caffe2 {
class Predictor {
 public:
  using TensorVector = std::vector<TensorCPU*>;
  // Runs the `init_net` once, then saves the `run_net` to be executed
  // in `::run`
  Predictor(const NetDef& init_net, const NetDef& run_net,
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

  // Returns true on success
  bool run(const TensorVector& inputs, TensorVector* outputs) {
    CAFFE_ENFORCE(inputs.size() <= run_net_.external_input_size());
    for (auto i = 0; i < inputs.size(); ++i) {
      shareInputTensor(&ws_, run_net_.external_input(i), inputs[i]);
    }

    if (!ws_.RunNet(run_net_.name())) {
      return false;
    }

    outputs->resize(run_net_.external_output_size());
    for (auto i = 0; i < outputs->size(); ++i) {
      (*outputs)[i] = extractOutputTensor(&ws_, run_net_.external_output(i));
    }
    return true;
  }

  const NetDef& def() const { return run_net_; };

  Workspace* ws() { return &ws_; };

 private:
  NetDef run_net_;
  Workspace ws_;

  void enforceIsTensor(Workspace* ws, const std::string& name) {
    auto blob = ws->GetBlob(name);
    CAFFE_ENFORCE(blob, "Blob does not exist: ", name);
    CAFFE_ENFORCE(blob->template IsType<TensorCPU>(),
                  "Blob is not a CPU Tensor: ", name);
  }

  void shareInputTensor(Workspace* ws, const std::string& name,
                        TensorCPU* input) {
    enforceIsTensor(ws, name);
    auto* blob = ws->GetBlob(name);
    CAFFE_ENFORCE(blob, "Blob: ", name, " does not exist");
    auto* tensor = blob->template GetMutable<TensorCPU>();
    tensor->ResizeLike(*input);
    tensor->ShareData(*input);
  }

  TensorCPU* extractOutputTensor(Workspace* ws, const std::string& name) {
    enforceIsTensor(ws, name);
    auto* blob = ws->GetBlob(name);
    CAFFE_ENFORCE(blob, "Blob: ", name, " does not exist");
    return blob->template GetMutable<TensorCPU>();
  }
};
}
