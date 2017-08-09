#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <caffe2/core/init.h>
#include <caffe2/core/predictor.h>
#include <caffe2/utils/proto_utils.h>

// #include "opencv2/highgui/highgui.hpp"
// #include "opencv2/imgproc/imgproc.hpp"

#include "json.hpp"
#include "predict.hpp"

using namespace caffe2;
using std::string;
using json = nlohmann::json;

/* Pair (label, confidence) representing a prediction. */
using Prediction = std::pair<int, float>;

PredictorContext New(char *init_net_file, char *predict_net_file) {
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

const char *Predict(PredictorContext pred, float *imageData) {
  // auto image = cv::imread(image_file);
  // image.convertTo(image, CV_32FC3, 1.0, -128);
  // vector<cv::Mat> channels(3);
  // cv::split(image, channels);
  // std::vector<float> data;
  // for (auto &c : channels) {
  //   data.insert(data.end(), (float *)c.datastart, (float *)c.dataend);
  // }
  // std::vector<TIndex> dims({1, image.channels(), image.rows, image.cols});

  const int channels = 3;
  const int width = 227;
  const int height = 227;
  const int size = channels * width * height;

  std::vector<float> data;
  data.reserve(size);
  std::copy(imageData, imageData + size, data.begin());
  std::vector<TIndex> dims({1, channels, width, height});

  std::cout << "size = " << size << std::endl;
  std::cout << "dims = " << dims << std::endl;

  // TensorCPU input(dims, data, NULL);
  TensorCPU input;
  input.Resize(dims);
  input.ShareExternalPointer(data.data());

  std::cout << "input dims = " << input.dims() << std::endl;

  std::cout << "inputVec declare  " << std::endl;
  Predictor::TensorVector inputVec({&input}), outputVec{};
  auto predictor = (Predictor *)pred;
  std::cout << "predictor->run(inputVec, &outputVec);" << std::endl;
  predictor->run(inputVec, &outputVec);
  std::cout << "auto &output = *(outputVec[0])" << std::endl;
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
