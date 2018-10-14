#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <caffe2/core/common.h>
#include <caffe2/core/init.h>
#include <caffe2/utils/proto_utils>
#include "caffe2/proto/caffe2.pb.h"

#include <caffe2/core/observer.h>
#include <caffe2/core/operator.h>

#ifdef WITH_CUDA
#include <caffe2/core/context_gpu.h>
#endif  // WITH_CUDA

#include "carml_predictor.h"
#include "predictor.hpp"
#include "timer.h"
#include "timer.impl.hpp"

using namespace caffe2;
using std::string;

template <typename Context>
struct Predictor {
  Predictor(carml::Predictor<Context> *ctx) : ctx_(ctx){};

  ~Predictor() {
    if (ctx_) {
      delete ctx_;
    }
  }

  carml::Predictor<Context> *const &context() { return ctx_; }

  carml::Predictor<Context> *ctx_;
  bool profile_enabled_{false};
  std::string profile_name_{""}, profile_metadata_{""};
  profile *prof_{nullptr};
  int pred_len_;
  const float *result_{nullptr};
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

  void set_layer_sequence_index(int ii) { layer_sequence_index_ = ii; }

 private:
  profile **prof_{nullptr};
  profile_entry *entry_{nullptr};
  std::string profile_name_{""}, profile_metadata_{""};
  int layer_sequence_index_{0};  // this is not valid for net
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
  int current_layer_sequence_index = 1;
  for (auto *op : subject_->GetOperators()) {
    auto obs = caffe2::make_unique<TimeObserver<OperatorBase>>(op, prof_);
    obs->set_layer_sequence_index(current_layer_sequence_index);
    op->AttachObserver(std::move(obs));
    current_layer_sequence_index++;
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
  shapes_t shapes{};
  for (const auto shape : op->InputTensorShapes()) {
    std::vector<int> dims{};
    for (const auto s : shape.dims()) {
      dims.emplace_back(s);
    }
    shapes.emplace_back(dims);
  }
  this->entry_ =
      new profile_entry(layer_sequence_index_, name, metadata, shapes);
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
    auto p = new Predictor<Context>(ctx);
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
void *predictImpl(Predictor<Context> *obj, float *imageData, const int batch,
                  const int channels, const int width, const int height) {
  obj->result_ = nullptr;

  const auto image_size = batch * channels * width * height;
  std::vector<float> data;
  data.reserve(image_size);
  std::copy(imageData, imageData + image_size, data.begin());
  std::vector<TIndex> dims({batch, channels, width, height});

  TensorCPU input;
  input.Resize(dims);
  input.ShareExternalPointer(data.data());

  auto predictor = obj->context();

  using input_vector_t = typename carml::Predictor<Context>::TensorInputVector;
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

  obj->pred_len_ = outputVec[0].size() / batch;
  obj->result_ = (float *)outputVec[0].raw_data();
}

void PredictCaffe2(PredictorContext pred, float *imageData, const int batch,
                   const int channels, const int width, const int height,
                   DeviceKind device_kind) {
  if (device_kind == CPU_DEVICE_KIND) {
    predictImpl<CPUContext>((Predictor<CPUContext> *)pred, imageData, batch,
                            channels, width, height);
  }

#ifdef WITH_CUDA
  predictImpl<CUDAContext>((Predictor<CUDAContext> *)pred, imageData, batch,
                           channels, width, height);
#endif
}

template <typename Context>
const float *getPredictionsImpl(Predictor<Context> *predictor) {
  if (predictor == nullptr) {
    return nullptr;
  }

  return predictor->result_;
}

const float *GetPredictionsCaffe2(PredictorContext pred,
                                  DeviceKind device_kind) {
  if (device_kind == CPU_DEVICE_KIND) {
    return getPredictionsImpl<CPUContext>((Predictor<CPUContext> *)pred);
  }

#ifdef WITH_CUDA
  return getPredictionsImpl<CUDAContext>((Predictor<CPUContext> *)pred);
#endif
}

template <typename Context>
int getPredLenImpl(Predictor<Context> *predictor) {
  if (predictor == nullptr) {
    return -1;
  }

  return predictor->pred_len_;
}

int GetPredLenCaffe2(PredictorContext pred, DeviceKind device_kind) {
  if (device_kind == CPU_DEVICE_KIND) {
    return getPredLenImpl<CPUContext>((Predictor<CPUContext> *)pred);
  }

#ifdef WITH_CUDA
  return getPredLenImpl<CUDAContext>((Predictor<CPUContext> *)pred);
#endif
}

template <typename Context>
static void deleteImpl(Predictor<Context> *predictor) {
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
    deleteImpl<CPUContext>((Predictor<CPUContext> *)pred);
    return;
  }
#ifdef WITH_CUDA
  if (cuda_ctx != nullptr) {
    delete cuda_ctx;
    cuda_ctx = nullptr;
  }
#endif  // WITH_CUDA

#ifdef WITH_CUDA
  deleteImpl<CUDAContext>((Predictor<CUDAContext> *)pred);
#endif
}

template <typename Context>
static void startProfilingImpl(Predictor<Context> *predictor, const char *name,
                               const char *metadata) {
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
    return startProfilingImpl<CPUContext>((Predictor<CPUContext> *)pred, name,
                                          metadata);
  }

#ifdef WITH_CUDA
  startProfilingImpl<CUDAContext>((Predictor<CUDAContext> *)pred, name,
                                  metadata);
#endif
}

template <typename Context>
static void endProfilingImpl(Predictor<Context> *predictor) {
  if (predictor && predictor->prof_) {
    predictor->prof_->end();
  }
}

void EndProfilingCaffe2(PredictorContext pred, DeviceKind device_kind) {
  if (device_kind == CPU_DEVICE_KIND) {
    endProfilingImpl<CPUContext>((Predictor<CPUContext> *)pred);
    return;
  }

#ifdef WITH_CUDA
  endProfilingImpl<CUDAContext>((Predictor<CUDAContext> *)pred);
#endif
}

template <typename Context>
static void disableProfilingImpl(Predictor<Context> *predictor) {
  if (!predictor || !predictor->prof_) {
    return;
  }
  delete_prof(&predictor->prof_);
  predictor->profile_name_ = std::string("");
  predictor->profile_metadata_ = std::string("");
}

void DisableProfilingCaffe2(PredictorContext pred, DeviceKind device_kind) {
  if (device_kind == CPU_DEVICE_KIND) {
    disableProfilingImpl<CPUContext>((Predictor<CPUContext> *)pred);
    return;
  }

#ifdef WITH_CUDA
  disableProfilingImpl<CUDAContext>((Predictor<CUDAContext> *)pred);
#endif
}

template <typename Context>
static char *readProfileImpl(Predictor<Context> *predictor) {
  if (!predictor || !predictor->prof_) {
    return NULL;
  }
  const auto s = predictor->prof_->read();
  const auto cstr = s.c_str();
  return strdup(cstr);
}

char *ReadProfileCaffe2(PredictorContext pred, DeviceKind device_kind) {
  if (device_kind == CPU_DEVICE_KIND) {
    return readProfileImpl<CPUContext>((Predictor<CPUContext> *)pred);
  }

#ifdef WITH_CUDA
  return readProfileImpl<CUDAContext>((Predictor<CUDAContext> *)pred);
#else  // WITH_CUDA
  return NULL;
#endif
}
