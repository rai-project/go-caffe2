#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "caffe2/proto/caffe2.pb.h"

#include <caffe2/core/common.h>
#include <caffe2/core/init.h>
#include <caffe2/utils/proto_utils.h>

#include <caffe2/core/observer.h>
#include <caffe2/core/operator.h>

#ifdef WITH_CUDA
#include <caffe2/core/context_gpu.h>
#endif  // WITH_CUDA

#include "carml_predictor.h"
#include "json.hpp"
#include "predict.hpp"
#include "timer.h"
#include "timer.impl.hpp"

using namespace caffe2;
using std::string;
using json = nlohmann::json;

/* Pair (label, confidence) representing a prediction. */
using Prediction = std::pair<int, float>;

template <typename Context>
struct PredictorObject {
  PredictorObject(carml::Predictor<Context> *ctx) : ctx_(ctx){};
  ~PredictorObject() {
    if (ctx_) {
      delete ctx_;
    }
  }
  carml::Predictor<Context> *const &context() { return ctx_; }

  carml::Predictor<Context> *ctx_;
  bool profile_enabled_{false};
  std::string profile_name_{""}, profile_metadata_{""};
  profile *prof_{nullptr};
};

static void delete_prof(profile **prof) {
  if (prof == nullptr) {
    return;
  }
  if (*prof == nullptr) {
    return;
  }
  auto p = *prof;
  p->reset();
  delete p;
  *prof = nullptr;
}

template <class T>
class TimeObserver final : public ObserverBase<T> {
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
  // change overriding return type to void
  // to make it a covariant
  // TODO: check if it breaks anything?
  void Start() override;
  void Stop() override;
};

template <>
void TimeObserver<NetBase>::Start() {
  const auto net = this->subject();
  auto net_name = net->Name();
  if (net_name.empty()) {
    net_name = profile_name_;
  }
  *this->prof_ = new profile(net_name, profile_metadata_);
  for (auto *op : subject_->GetOperators()) {
    op->AttachObserver(
        caffe2::make_unique<TimeObserver<OperatorBase>>(op, prof_));
  }
  const auto p = *this->prof_;
  p->start();
}

template <>
void TimeObserver<NetBase>::Stop() {
  const auto p = *this->prof_;
  p->end();
}

template <>
void TimeObserver<OperatorBase>::Start() {
  const auto &op = this->subject();
  std::string name{""}, metadata{""};
  if (op->has_debug_def()) {
    const auto &opdef = op->debug_def();
    name = opdef.type();
    metadata = opdef.name();
  }
  this->entry_ = new profile_entry(name, metadata);
}

template <>
void TimeObserver<OperatorBase>::Stop() {
  this->entry_->end();
  const auto p = *this->prof_;
  p->add(this->entry_);
}

template <typename Context>
static PredictorContext newImpl(char *predict_net_file, char *init_net_file) {
  try {
    NetDef init_net, predict_net;
    CAFFE_ENFORCE(ReadProtoFromFile(init_net_file, &init_net));
    CAFFE_ENFORCE(ReadProtoFromFile(predict_net_file, &predict_net));

    auto ctx = new carml::Predictor<Context>(init_net, predict_net);
    auto p = new PredictorObject<Context>(ctx);
    return (PredictorContext)p;
  } catch (const std::invalid_argument &ex) {
    LOG(ERROR) << "exception: " << ex.what();
    errno = EINVAL;
    return nullptr;
  } catch (...) {
    LOG(ERROR) << "exception: catch all"
               << "\n";
    return nullptr;
  }
}

PredictorContext NewCaffe2(char *predict_net_file, char *init_net_file,
                           DeviceKind device_kind) {
  if (device_kind == CPU_DEVICE_KIND) {
    return newImpl<CPUContext>(predict_net_file, init_net_file);
  }

#ifdef WITH_CUDA
  return newImpl<CUDAContext>(predict_net_file, init_net_file);
#else  // WITH_CUDA
  return NULL;
#endif
}

template <typename Context>
static const char *predictImpl(PredictorObject<Context> *obj, float *imageData,
                               const int batch, const int channels,
                               const int width, const int height) {
  const auto image_size = batch * channels * width * height;
  std::vector<float> data;
  data.reserve(image_size);
  std::copy(imageData, imageData + image_size, data.begin());
  std::vector<TIndex> dims({batch, channels, width, height});

  TensorCPU input;
  input.Resize(dims);
  input.ShareExternalPointer(data.data());

  auto predictor = obj->context();

  using input_vector_t = typename carml::Predictor<Context>::TensorCPUVector;
  using output_vector_t =
      typename carml::Predictor<Context>::TensorOutputVector;
  input_vector_t inputVec{&input};
  output_vector_t outputVec{};

  auto net = predictor->net();

  if (obj->profile_enabled_) {
    unique_ptr<TimeObserver<NetBase>> net_ob =
        make_unique<TimeObserver<NetBase>>(net, &obj->prof_, obj->profile_name_,
                                           obj->profile_metadata_);
    net->AttachObserver(std::move(net_ob));
  }

  predictor->run(inputVec, &outputVec);

  auto len = outputVec[0].size() / batch;
  auto probs = (float *)outputVec[0].raw_data();

  std::vector<Prediction> predictions;
  predictions.reserve(outputVec[0].size());
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

const char *PredictCaffe2(PredictorContext pred, float *imageData,
                          const int batch, const int channels, const int width,
                          const int height, DeviceKind device_kind) {
  if (device_kind == CPU_DEVICE_KIND) {
    return predictImpl<CPUContext>((PredictorObject<CPUContext> *)pred,
                                   imageData, batch, channels, width, height);
  }

#ifdef WITH_CUDA
  return predictImpl<CUDAContext>((PredictorObject<CUDAContext> *)pred,
                                  imageData, batch, channels, width, height);
#else  // WITH_CUDA
  return NULL;
#endif
}

template <typename Context>
static void deleteImpl(PredictorObject<Context> *predictor) {
  if (predictor) {
    delete_prof(&predictor->prof_);
    delete predictor;
  }
}

#ifdef WITH_CUDA
static CUDAContext *cuda_ctx = nullptr;
#endif  // WITH_CUDA

void DeleteCaffe2(PredictorContext pred, DeviceKind device_kind) {
  if (device_kind == CPU_DEVICE_KIND) {
    deleteImpl<CPUContext>((PredictorObject<CPUContext> *)pred);
    return;
  }
#ifdef WITH_CUDA
  if (cuda_ctx != nullptr) {
    delete cuda_ctx;
    cuda_ctx = nullptr;
  }
#endif  // WITH_CUDA

#ifdef WITH_CUDA
  deleteImpl<CUDAContext>((PredictorObject<CUDAContext> *)pred);
#endif
}

int InitCUDACaffe2() {
#ifdef WITH_CUDA
  static bool initialized_cuda = false;
  if (initialized_cuda) {
    return true;
  }
  initialized_cuda = true;
  DeviceOption option;
  option.set_device_type(CUDA);
  cuda_ctx = new CUDAContext(option);

  return true;
#else
  return false;
#endif
}

int InitCaffe2(DeviceKind device_kind) {
  static bool initialized_caffe = false;
  if (initialized_caffe) {
    return true;
  }
  initialized_caffe = true;
  int dummy_argc = 1;
  const char *dummy_name = "go-caffe2";
  char **dummy_argv = const_cast<char **>(&dummy_name);
  GlobalInit(&dummy_argc, &dummy_argv);
  return InitCUDACaffe2();
}

template <typename Context>
static void startProfilingImpl(PredictorObject<Context> *predictor,
                               const char *name, const char *metadata) {
  if (name == nullptr) {
    name = "";
  }
  if (metadata == nullptr) {
    metadata = "";
  }
  predictor->profile_enabled_ = true;
  predictor->profile_name_ = std::string(name);
  predictor->profile_metadata_ = std::string(metadata);
}
void StartProfilingCaffe2(PredictorContext pred, const char *name,
                          const char *metadata, DeviceKind device_kind) {
  if (device_kind == CPU_DEVICE_KIND) {
    return startProfilingImpl<CPUContext>((PredictorObject<CPUContext> *)pred,
                                          name, metadata);
  }

#ifdef WITH_CUDA
  startProfilingImpl<CUDAContext>((PredictorObject<CUDAContext> *)pred, name,
                                  metadata);
#endif
}

template <typename Context>
static void endProfilingImpl(PredictorObject<Context> *predictor) {
  if (predictor && predictor->prof_) {
    predictor->prof_->end();
  }
}

void EndProfilingCaffe2(PredictorContext pred, DeviceKind device_kind) {
  if (device_kind == CPU_DEVICE_KIND) {
    endProfilingImpl<CPUContext>((PredictorObject<CPUContext> *)pred);
    return;
  }

#ifdef WITH_CUDA
  endProfilingImpl<CUDAContext>((PredictorObject<CUDAContext> *)pred);
#endif
}

template <typename Context>
static void disableProfilingImpl(PredictorObject<Context> *predictor) {
  if (!predictor || !predictor->prof_) {
    return;
  }
  delete_prof(&predictor->prof_);
  predictor->profile_name_ = std::string("");
  predictor->profile_metadata_ = std::string("");
}

void DisableProfilingCaffe2(PredictorContext pred, DeviceKind device_kind) {
  if (device_kind == CPU_DEVICE_KIND) {
    disableProfilingImpl<CPUContext>((PredictorObject<CPUContext> *)pred);
    return;
  }

#ifdef WITH_CUDA
  disableProfilingImpl<CUDAContext>((PredictorObject<CUDAContext> *)pred);
#endif
}

template <typename Context>
static char *readProfileImpl(PredictorObject<Context> *predictor) {
  if (!predictor || !predictor->prof_) {
    return NULL;
  }
  const auto s = predictor->prof_->read();
  const auto cstr = s.c_str();
  return strdup(cstr);
}

char *ReadProfileCaffe2(PredictorContext pred, DeviceKind device_kind) {
  if (device_kind == CPU_DEVICE_KIND) {
    return readProfileImpl<CPUContext>((PredictorObject<CPUContext> *)pred);
  }

#ifdef WITH_CUDA
  return readProfileImpl<CUDAContext>((PredictorObject<CUDAContext> *)pred);
#else  // WITH_CUDA
  return NULL;
#endif
}
