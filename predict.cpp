#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <caffe2/core/init.h>
#include <caffe2/core/predictor.h>

#include <caffe2/core/common.h>
#include <caffe2/core/observer.h>
#include <caffe2/core/operator.h>
#include <caffe2/utils/proto_utils.h>

#include "json.hpp"
#include "predict.hpp"
#include "timer.h"
#include "timer.impl.hpp"

using namespace caffe2;
using std::string;
using json = nlohmann::json;

/* Pair (label, confidence) representing a prediction. */
using Prediction = std::pair<int, float>;

struct PredictorObject {
  PredictorObject(caffe2::Predictor *const ctx) : ctx_(ctx){};

  caffe2::Predictor *const &context() { return ctx_; }

  caffe2::Predictor *const ctx_;
  profile *prof_{nullptr};
};

template <class T>
class TimeObserver final : public ObserverBase<T> {
 public:
  explicit TimeObserver<T>(T *subject, profile **prof)
      : ObserverBase<T>(subject), prof_(prof) {}
  ~TimeObserver() {}

 private:
  profile **prof_;
  profile_entry *entry_;
  bool Start() override;
  bool Stop() override;
};

template <>
bool TimeObserver<NetBase>::Start() {
  *this->prof_ = new profile("name", "metadata");
  for (auto *op : subject_->GetOperators()) {
    op->SetObserver(caffe2::make_unique<TimeObserver<OperatorBase>>(op, prof_));
  }
  const auto p = *this->prof_;
  p->start();
  return true;
}

template <>
bool TimeObserver<NetBase>::Stop() {
  const auto p = *this->prof_;
  p->end();
  return true;
}

template <>
bool TimeObserver<OperatorBase>::Start() {
  const auto op = this->subject();
  (void)op;
  this->entry_ = new profile_entry("entry_name", "entry_metadata");
  return true;
}

template <>
bool TimeObserver<OperatorBase>::Stop() {
  this->entry_->end();
  const auto p = *this->prof_;
  p->add(this->entry_);
  return true;
}

PredictorContext New(char *predict_net_file, char *init_net_file) {
  try {
    NetDef init_net, predict_net;
    CAFFE_ENFORCE(ReadProtoFromFile(init_net_file, &init_net));
    CAFFE_ENFORCE(ReadProtoFromFile(predict_net_file, &predict_net));
    const auto ctx = new Predictor(init_net, predict_net);
    auto p = new PredictorObject(ctx);
    return (PredictorContext)p;
  } catch (const std::invalid_argument &ex) {
    LOG(ERROR) << "exception: " << ex.what();
    errno = EINVAL;
    return nullptr;
  }
}

const char *Predict(PredictorContext pred0, float *imageData, const int batch,
                    const int channels, const int width, const int height) {
  auto obj = (PredictorObject *)pred0;

  const auto image_size = batch * channels * width * height;
  std::vector<float> data;
  data.reserve(image_size);
  std::copy(imageData, imageData + image_size, data.begin());
  std::vector<TIndex> dims({batch, channels, width, height});

  TensorCPU input;
  input.Resize(dims);
  input.ShareExternalPointer(data.data());

  Predictor::TensorVector inputVec{&input}, outputVec{};
  auto predictor = obj->context();

  auto ws = predictor->ws();
  auto net_def = predictor->def();
  auto net = ws->GetNet(net_def.name());
  if (obj->prof_ != nullptr) {
    unique_ptr<TimeObserver<NetBase>> net_ob =
        make_unique<TimeObserver<NetBase>>(net, &obj->prof_);
    net->SetObserver(std::move(net_ob));
  } else {
    net->RemoveObserver();
  }

  predictor->run(inputVec, &outputVec);
  auto &output = *(outputVec[0]);
  const auto len = output.size() / batch;
  const auto &probs = output.data<float>();

  std::vector<Prediction> predictions;
  predictions.reserve(output.size());
  for (int cnt = 0; cnt < batch; cnt++) {
    for (int idx = 0; idx < len; idx++) {
      predictions.emplace_back(std::make_pair(idx, probs[cnt * len + idx]));
    }
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
  auto predictor = (PredictorObject *)pred;
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

void StartProfiling(PredictorContext pred, const char *name,
                    const char *metadata) {
  auto predictor = (PredictorObject *)pred;
  if (name == nullptr) {
    name = "";
  }
  if (metadata == nullptr) {
    metadata = "";
  }
  predictor->prof_ = new profile(name, metadata);
}

void EndProfiling(PredictorContext pred) {
  auto predictor = (PredictorObject *)pred;
  if (predictor->prof_) {
    predictor->prof_->end();
  }
}

void DisableProfiling(PredictorContext pred) {
  auto predictor = (PredictorObject *)pred;
  if (predictor->prof_) {
    predictor->prof_->reset();
    delete predictor->prof_;
  }
  predictor->prof_ = nullptr;
}

char *ReadProfile(PredictorContext pred) {
  auto predictor = (PredictorObject *)pred;
  const auto s = predictor->prof_->read();
  const auto cstr = s.c_str();
  return strdup(cstr);
}
