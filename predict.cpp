#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// #include <caffe2/core/predictor.h>
#include "caffe2/core/init.h"
#include "caffe2/core/predictor.h"
#include "caffe2/utils/proto_utils.h"

#include "json.hpp"
#include "predict.hpp"

using namespace caffe2;
using std::string;
using json = nlohmann::json;

/* Pair (label, confidence) representing a prediction. */
using Prediction = std::pair<int, float>;

PredictorContext New(char* model_file, char* trained_file) {
  try {
    NetDef init_net, predict_net;
    CAFFE_ENFORCE(ReadProtoFromFile(model_file, &init_net));
    CAFFE_ENFORCE(ReadProtoFromFile(trained_file, &predict_net));
    const auto ctx = new Predictor(model_file, trained_file);
    return (void*)ctx;
  } catch (const std::invalid_argument& ex) {
    LOG(ERROR) << "exception: " << ex.what();
    errno = EINVAL;
    return nullptr;
  }
}

void Init() { ::google::InitGoogleLogging("go-caffe2"); }

const char* Predict(PredictorContext pred, float* imageData) {

  // std::vector<TIndex> dims({1, image.channels(), image.rows, image.cols});
  std::vector<TIndex> dims({1, 3, 227, 227});
  TensorCPU input(dims, imageData, NULL);
  Predictor::TensorVector inputVec({&input}), outputVec;
  auto predictor = (Predictor*)pred;
  predictor.run(inputVec, &outputVec);
  auto &output = *(outputVec[0]);

  std::vector<Prediction> predictions;
  for (int idx = 0; idx < output.size(); idx++) {
    predictions.emplace_back(std::make_pair(idx, output[idx]));
  }

  json preds = json::array();
  for (const auto prediction : predictions) {
    preds.push_back(
        {{"index", prediction.first}, {"probability", prediction.second}});
  }
  auto res = strdup(preds.dump().c_str());
  return res;
}

void Delete(PredictorContext pred) {
  auto predictor = (Predictor*)pred;
  delete predictor;
}

// void SetMode(int mode) { Caffe::set_mode((caffe::Caffe::Brew)mode); }
