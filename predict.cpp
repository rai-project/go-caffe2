#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <caffe2/core/init.h>
#include <caffe2/core/predictor.h>
#include <caffe2/utils/proto_utils.h>

#include "json.hpp"
#include "predict.hpp"

using namespace caffe2;
using std::string;
using json = nlohmann::json;

/* Pair (label, confidence) representing a prediction. */
using Prediction = std::pair<int, float>;

PredictorContext New(char *predict_net_file, char *init_net_file) {
  try {
    NetDef init_net, predict_net;
    CAFFE_ENFORCE(ReadProtoFromFile(init_net_file, &init_net));
    CAFFE_ENFORCE(ReadProtoFromFile(predict_net_file, &predict_net));
    const auto ctx = new Predictor(init_net, predict_net);
    std::cout << "..  " << predict_net.external_input_size() << "\n";
    return (void *)ctx;
  } catch (const std::invalid_argument &ex) {
    LOG(ERROR) << "exception: " << ex.what();
    errno = EINVAL;
    return nullptr;
  }
}

const char *Predict(PredictorContext pred, float *imageData, const int channels,
  const int width, const int height) {
  const auto image_size = channels * width * height;

  std::vector<float> data;
  data.reserve(image_size);
  std::copy(imageData, imageData + image_size, data.begin());
  std::vector<TIndex> dims({1, channels, width, height});

  TensorCPU input;
  input.Resize(dims);
  input.ShareExternalPointer(data.data());

  Predictor::TensorVector inputVec({&input}), outputVec{};
  auto predictor = (Predictor *)pred;
  predictor->run(inputVec, &outputVec);
  auto &output = *(outputVec[0]);
  const auto &probs = output.data<float>();

  std::vector<Prediction> predictions;
  for (int idx = 0; idx < output.size(); idx++) {
    predictions.emplace_back(std::make_pair(idx, probs[idx]));
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
  auto predictor = (Predictor *)pred;
  if (predictor) {
    delete predictor;
  }
}

void Init() {
  int dummy_argc = 1;
  const char *dummy_name = "go-caffe2";
  char **dummy_argv = const_cast<char **>(&dummy_name);
  GlobalInit(&dummy_argc, &dummy_argv);
}
