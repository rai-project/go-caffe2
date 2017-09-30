#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <caffe2/core/init.h>

#include <caffe2/core/common.h>
#include <caffe2/core/observer.h>
#include <caffe2/core/operator.h>
#include <caffe2/utils/proto_utils.h>

#ifdef WITH_CUDA
#include <caffe2/core/context_gpu.h>
#endif

#include "json.hpp"
#include "predict.hpp"
#include "predictor.h"
#include "timer.h"
#include "timer.impl.hpp"

using namespace caffe2;
using std::string;
using json = nlohmann::json;

/* Pair (label, confidence) representing a prediction. */
using Prediction = std::pair<int, float>;

struct PredictorObject
{
  using Context = CUDAContext;
  PredictorObject(Predictor<Context> *const ctx) : ctx_(ctx){};

  Predictor<Context> *const &context() { return ctx_; }

  Predictor<Context> *const ctx_;
  bool profile_enabled_{false};
  std::string profile_name_{""}, profile_metadata_{""};
  profile *prof_{nullptr};
};

static void SetCUDA()
{
  DeviceOption option;
  option.set_device_type(CUDA);
  new CUDAContext(option);
}

static void delete_prof(profile **prof)
{
  if (prof == nullptr)
  {
    return;
  }
  if (*prof == nullptr)
  {
    return;
  }
  auto p = *prof;
  p->reset();
  delete p;
  *prof = nullptr;
}

template <class T>
class TimeObserver final : public ObserverBase<T>
{
public:
  explicit TimeObserver<T>(T *subject, profile **prof,
                           std::string profile_name = "",
                           std::string profile_metadata = "")
      : ObserverBase<T>(subject),
        prof_(prof),
        profile_name_(profile_name),
        profile_metadata_(profile_metadata) {}
  ~TimeObserver() {}

private:
  profile **prof_{nullptr};
  profile_entry *entry_{nullptr};
  std::string profile_name_{""}, profile_metadata_{""};
  bool Start() override;
  bool Stop() override;
};

template <>
bool TimeObserver<NetBase>::Start()
{
  const auto net = this->subject();
  auto net_name = net->Name();
  if (net_name.empty())
  {
    net_name = profile_name_;
  }
  *this->prof_ = new profile(net_name, profile_metadata_);
  for (auto *op : subject_->GetOperators())
  {
    op->SetObserver(caffe2::make_unique<TimeObserver<OperatorBase>>(op, prof_));
  }
  const auto p = *this->prof_;
  p->start();
  return true;
}

template <>
bool TimeObserver<NetBase>::Stop()
{
  const auto p = *this->prof_;
  p->end();
  return true;
}

template <>
bool TimeObserver<OperatorBase>::Start()
{
  const auto &op = this->subject();
  std::string name{""}, metadata{""};
  if (op->has_debug_def())
  {
    const auto &opdef = op->debug_def();
    name = opdef.type();
    metadata = opdef.name();
  }
  this->entry_ = new profile_entry(name, metadata);
  return true;
}

template <>
bool TimeObserver<OperatorBase>::Stop()
{
  this->entry_->end();
  const auto p = *this->prof_;
  p->add(this->entry_);
  return true;
}

PredictorContext New(char *predict_net_file, char *init_net_file)
{
  try
  {
    Workspace workspace;
    SetCUDA();
    NetDef init_net, predict_net;
    CAFFE_ENFORCE(ReadProtoFromFile(init_net_file, &init_net));
    CAFFE_ENFORCE(ReadProtoFromFile(predict_net_file, &predict_net));
    init_net.mutable_device_option()->set_device_type(CUDA);
    predict_net.mutable_device_option()->set_device_type(CUDA);
    const auto ctx = new Predictor<CUDAContext>(init_net, predict_net, &workspace);
    auto p = new PredictorObject(ctx);
    return (PredictorContext)p;
  }
  catch (const std::invalid_argument &ex)
  {
    LOG(ERROR) << "exception: " << ex.what();
    errno = EINVAL;
    return nullptr;
  }
  catch (const std::exception &e)
  {
    LOG(ERROR) << "exception: " << e.what();
    std::cout << "exception: " << e.what() << "\n";
    errno = EINVAL;
    return nullptr;
  }
}

const char *Predict(PredictorContext pred0, float *imageData, const int batch,
                    const int channels, const int width, const int height)
{
  auto obj = (PredictorObject *)pred0;

  const auto image_size = batch * channels * width * height;
  std::vector<float> data;
  data.reserve(image_size);
  std::copy(imageData, imageData + image_size, data.begin());
  std::vector<TIndex> dims({batch, channels, width, height});

  TensorCUDA input;
  input.Resize(dims);
  input.ShareExternalPointer(data.data());

  Predictor<CUDAContext>::TensorDeviceVector inputVec{&input}, outputVec{};
  auto predictor = obj->context();

  auto ws = predictor->ws();
  auto net_def = predictor->def();
  auto net = ws->GetNet(net_def.name());
  if (obj->profile_enabled_)
  {
    unique_ptr<TimeObserver<NetBase>> net_ob =
        make_unique<TimeObserver<NetBase>>(net, &obj->prof_, obj->profile_name_,
                                           obj->profile_metadata_);
    net->SetObserver(std::move(net_ob));
  }
  else
  {
    net->RemoveObserver();
  }

  net_def.mutable_device_option()->set_device_type(CUDA);
  predictor->run(inputVec, &outputVec);
  auto &output = *(outputVec[0]);
  const auto len = output.size() / batch;
  const auto &probs = output.data<float>();

  std::vector<Prediction> predictions;
  predictions.reserve(output.size());
  for (int cnt = 0; cnt < batch; cnt++)
  {
    for (int idx = 0; idx < len; idx++)
    {
      predictions.emplace_back(std::make_pair(idx, probs[cnt * len + idx]));
    }
  }

  json preds = json::array();
  for (const auto prediction : predictions)
  {
    preds.push_back(
        {{"index", prediction.first}, {"probability", prediction.second}});
  }
  auto res = strdup(preds.dump().c_str());
  return res;
}

void Delete(PredictorContext pred)
{
  auto predictor = (PredictorObject *)pred;
  if (predictor)
  {
    delete_prof(&predictor->prof_);
    delete predictor;
  }
}

void Init()
{
  int dummy_argc = 1;
  const char *dummy_name = "go-caffe2";
  char **dummy_argv = const_cast<char **>(&dummy_name);
  GlobalInit(&dummy_argc, &dummy_argv);

  SetCUDA();
}

void StartProfiling(PredictorContext pred, const char *name,
                    const char *metadata)
{
  auto predictor = (PredictorObject *)pred;
  if (name == nullptr)
  {
    name = "";
  }
  if (metadata == nullptr)
  {
    metadata = "";
  }
  predictor->profile_enabled_ = true;
  predictor->profile_name_ = std::string(name);
  predictor->profile_metadata_ = std::string(metadata);
}

void EndProfiling(PredictorContext pred)
{
  auto predictor = (PredictorObject *)pred;
  if (predictor->prof_)
  {
    predictor->prof_->end();
  }
}

void DisableProfiling(PredictorContext pred)
{
  auto predictor = (PredictorObject *)pred;
  delete_prof(&predictor->prof_);
  predictor->profile_name_ = std::string("");
  predictor->profile_metadata_ = std::string("");
}

char *ReadProfile(PredictorContext pred)
{
  auto predictor = (PredictorObject *)pred;
  const auto s = predictor->prof_->read();
  const auto cstr = s.c_str();
  return strdup(cstr);
}
