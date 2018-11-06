#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <caffe2/core/common.h>
#include <caffe2/core/init.h>
#include <caffe2/core/net.h>
#include <caffe2/core/observer.h>
#include <caffe2/core/operator.h>
#include <caffe2/utils/proto_utils.h>

#include "caffe2/proto/caffe2.pb.h"

#include <caffe2/core/tensor.h>

#ifdef WITH_CUDA
#include <caffe2/core/context_gpu.h>
#endif  // WITH_CUDA

#include "predictor.hpp"
#include "timer.h"
#include "timer.impl.hpp"

#include "backward.hpp"

namespace backward {

backward::SignalHandling sh;

}  // namespace backward

#if 1
#define DEBUG_STMT std ::cout << __func__ << "  " << __LINE__ << "\n";
#else
#define DEBUG_STMT
#endif

using namespace caffe2;
using std::string;

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

class Predictor {
 public:
  Predictor(NetDef *init_net, NetDef *net_def, DeviceKind device_kind);
  void Predict(float *imageData, std::string input_type, const int batch,
               const int channels, const int width, const int height);

  DeviceKind device_kind_;
  std::shared_ptr<Workspace> ws_;
  NetBase *net_;
  std::vector<string> input_names_;
  std::vector<string> output_names_;
  int pred_len_;
  const float *result_{nullptr};
  bool profile_enabled_{false};
  profile *prof_{nullptr};
  std::string profile_name_{""}, profile_metadata_{""};
};

static std::string get_backend(std::string backend) {
  if (backend != "builtin") {
    string engine =
        backend == "nnpack"
            ? "NNPACK"
            : backend == "eigen"
                  ? "EIGEN"
                  : backend == "mkl"
                        ? "MKLDNN"
                        : backend == "cuda"
                              ? "CUDA"
                              : backend == "dnnlowp"
                                    ? "DNNLOWP"
                                    : backend == "dnnlowp_acc16"
                                          ? "DNNLOWP_ACC16"
                                          : backend == "default" ? "" : "NONE";
    return engine;
  }
  return backend;
}

static void set_operator_engine(NetDef *net, DeviceKind device_kind) {
  std::string backend = "";
  if (device_kind == CUDA_DEVICE_KIND) {
#ifdef WITH_CUDA
    backend = get_backend("cuda");
#else
    throw std::runtime_error("Not set WITH_CUDA = 1");
#endif  // WITH_CUDA
  } else {
    backend = get_backend("eigen");
  }

  for (int i = 0; i < net->op_size(); i++) {
    caffe2::OperatorDef *op_def = net->mutable_op(i);
    op_def->set_engine(backend);
  }
}

Predictor::Predictor(NetDef *init_net, NetDef *net_def,
                     DeviceKind device_kind) {
  ws_ = std::make_shared<Workspace>(new Workspace());
  DEBUG_STMT
  device_kind_ = device_kind;
  DEBUG_STMT
  ws_->RunNetOnce(*init_net);
  DEBUG_STMT
#if 0
  auto init_net_ = ws_.CreateNet(init_net);
      DEBUG_STMT
  if (!init_net_->Run()) {
	  throw std::runtime_error("cannot run init network");
  }
#endif

  DEBUG_STMT
  if (!net_def->has_name()) {
    net_def->set_name("go-caffe2");
  }
  net_ = ws_->CreateNet(*net_def);

  DEBUG_STMT
  for (auto ii = 0; ii < net_def->external_input_size(); ii++) {
    input_names_.emplace_back(net_def->external_input(ii));
  }
  DEBUG_STMT
  for (auto ii = 0; ii < net_def->external_output_size(); ii++) {
    output_names_.emplace_back(net_def->external_output(ii));
  }
  DEBUG_STMT
}

PredictorContext NewCaffe2(char *init_net_file, char *net_file,
                           DeviceKind device_kind) {
  try {
    NetDef init_net, net;
    if (!ReadProtoFromFile(init_net_file, &init_net)) {
      throw std::runtime_error("cannot read init net file");
    }
    set_operator_engine(&init_net, device_kind);
    if (!ReadProtoFromFile(net_file, &net)) {
      throw std::runtime_error("cannot read net file");
    }
    set_operator_engine(&net, device_kind);
    auto ctx = new Predictor(&init_net, &net, device_kind);
    return (PredictorContext)ctx;
  } catch (const std::invalid_argument &ex) {
    LOG(ERROR) << "exception: " << ex.what();
    errno = EINVAL;
    return nullptr;
  } catch (std::exception &ex) {
    LOG(ERROR) << "exception: catch all [ " << ex.what() << "]"
               << "\n";
    return nullptr;
  }
}

#ifdef WITH_CUDA
static CUDAContext *cuda_context = nullptr;
#endif  // WITH_CUDA

void InitCaffe2(DeviceKind device_kind) {
  static bool initialized_caffe = false;
  if (initialized_caffe) {
    return;
  }

  caffe2::ClearGlobalNetObservers();
  caffe2::ShowLogInfoToStderr();
  initialized_caffe = true;
  int dummy_argc = 1;
  const char *dummy_name = "go-caffe2";
  char **dummy_argv = const_cast<char **>(&dummy_name);
  GlobalInit(&dummy_argc, &dummy_argv);

  if (device_kind == CUDA_DEVICE_KIND) {
#ifdef WITH_CUDA
    static bool initialized_cuda = false;
    if (initialized_cuda) {
      return;
    }
    initialized_cuda = true;
    DeviceOption option;
    option.set_device_type(PROTO_CUDA);
    cuda_context = new CUDAContext(option);
    return;
#else
    throw std::runtime_error("Not set WITH_CUDA = 1");
#endif
  }
}

void Predictor::Predict(float *imageData, std::string input_type,
                        const int batch, const int channels, const int width,
                        const int height) {
  result_ = nullptr;
  DEBUG_STMT
  if (profile_enabled_) {
    unique_ptr<TimeObserver<NetBase>> net_ob =
        make_unique<TimeObserver<NetBase>>(net_, &prof_, profile_name_,
                                           profile_metadata_);
    net_->AttachObserver(std::move(net_ob));
  }
  DEBUG_STMT
  const auto data_size = batch * channels * width * height;
  std::vector<float> data;
  DEBUG_STMT
  data.reserve(data_size);
  DEBUG_STMT
  std::copy(imageData, imageData + data_size, data.begin());
  std::vector<int64_t> dims({batch, channels, width, height});

  DEBUG_STMT
  auto name = input_names_[0];
  auto *blob = ws_->GetBlob(name);
  if (blob == nullptr) {
    blob = ws_->CreateBlob(input_names_[0]);
  }
  if (device_kind_ == CUDA_DEVICE_KIND) {
#ifdef WITH_CUDA
    auto cpu_tensor = BlobGetMutableTensor(blob, caffe2::CPU);
    cpu_tensor->Resize(dims);
    cpu_tensor->ShareExternalPointer(data.data());

    auto tensor = blob->GetMutable<caffe2::TensorCUDA>();
    tensor->Resize(dims);
    tensor->CopyFrom(*cpu_tensor);

    if (input_type == "uint8_t") {
      tensor->mutable_data<uint8_t>();
    } else if (input_type == "float") {
      tensor->mutable_data<float>();
    } else {
      std::runtime_error("Unsupported input type");
    }
#else
    throw std::runtime_error("Not set WITH_CUDA = 1");
#endif  // WITH_CUDA
  } else {
    auto tensor = BlobGetMutableTensor(blob, caffe2::CPU);
    tensor->Resize(dims);
    tensor->ShareExternalPointer(data.data());

    if (input_type == "uint8_t") {
      tensor->mutable_data<uint8_t>();
    } else if (input_type == "float") {
      tensor->mutable_data<float>();
    } else {
      std::runtime_error("Unsupported input type");
    }
  }
  DEBUG_STMT
  if (!net_->Run()) {
    throw std::runtime_error("invalid run");
  }

  std::vector<TensorCPU> outputVec{};

  outputVec.resize(output_names_.size());
  for (auto ii = 0; ii < outputVec.size(); ii++) {
    auto name = output_names_[ii];
    auto *blob = ws_->GetBlob(name);
    if (blob == nullptr) {
      throw std::runtime_error("output blob does not exist");
    }
    TensorCPU output_tensor;

    if (device_kind_ == CUDA_DEVICE_KIND) {
#ifdef WITH_CUDA
      // Copy TensorGPU to TensorCPU
      // Note: copy constructor no longer permitted
      // should be using Clone() explicitly
      output_tensor.CopyFrom(blob->Get<TensorCUDA>().Clone());
#else
      throw std::runtime_error("Not set WITH_CUDA = 1");
#endif  // WITH_CUDA
    } else {
      output_tensor = (blob->Get<TensorCPU>()).Clone();
    }
    outputVec[ii].ResizeLike(output_tensor);
    outputVec[ii].ShareData(output_tensor);
  }

  pred_len_ = outputVec[0].size() / batch;
  result_ = (float *)outputVec[0].raw_data();
}

error_t PredictCaffe2(PredictorContext pred, float *imageData,
                      const char *input_type, const int batch,
                      const int channels, const int width, const int height) {
  try {
    auto predictor = (Predictor *)pred;
    if (predictor == nullptr) {
      std ::cout << __func__ << "  " << __LINE__ << " ... got a null pointer\n";
      return error_invalid_memory;
    }
    predictor->Predict(imageData, input_type, batch, channels, width, height);
    return success;
  } catch (std::exception &ex) {
    LOG(ERROR) << "exception: catch all [ " << ex.what() << "]"
               << "\n";
    return error_exception;
  }
}

const float *GetPredictionsCaffe2(PredictorContext pred) {
  try {
    auto predictor = (Predictor *)pred;
    if (predictor == nullptr) {
      return nullptr;
    }
    return predictor->result_;
  } catch (std::exception &ex) {
    LOG(ERROR) << "exception: catch all [ " << ex.what() << "]"
               << "\n";
    return nullptr;
  }
}

void DeleteCaffe2(PredictorContext pred) {
  try {
    auto predictor = (Predictor *)pred;
    if (predictor == nullptr) {
      return;
    }
    if (predictor->prof_) {
      predictor->prof_->reset();
      delete predictor->prof_;
      predictor->prof_ = nullptr;
    }
    delete predictor;

#ifdef WITH_CUDA
    if (cuda_context != nullptr) {
      delete cuda_context;
      cuda_context = nullptr;
    }
#endif  // WITH_CUDA
  } catch (std::exception &ex) {
    LOG(ERROR) << "exception: catch all [ " << ex.what() << "]"
               << "\n";
    return;
  }
}

void StartProfilingCaffe2(PredictorContext pred, const char *name,
                          const char *metadata) {
  try {
    if (name == nullptr) {
      name = "";
    }
    if (metadata == nullptr) {
      metadata = "";
    }
    if (pred == nullptr) {
      return;
    }
    DEBUG_STMT

    auto predictor = (Predictor *)pred;
    predictor->profile_enabled_ = true;
    predictor->profile_name_ = std::string(name);
    predictor->profile_metadata_ = std::string(metadata);
  } catch (std::exception &ex) {
    LOG(ERROR) << "exception: catch all [ " << ex.what() << "]"
               << "\n";
    return;
  }
}

void EndProfilingCaffe2(PredictorContext pred) {
  try {
    auto predictor = (Predictor *)pred;
    if (predictor == nullptr) {
      return;
    }
    if (predictor->prof_) {
      predictor->prof_->end();
    }
  } catch (std::exception &ex) {
    LOG(ERROR) << "exception: catch all [ " << ex.what() << "]"
               << "\n";
    return;
  }
}

void DisableProfilingCaffe2(PredictorContext pred) {
  try {
    auto predictor = (Predictor *)pred;
    if (predictor == nullptr) {
      return;
    }
    if (predictor->prof_) {
      predictor->prof_->reset();
      predictor->profile_name_ = std::string("");
      predictor->profile_metadata_ = std::string("");
    }
  } catch (std::exception &ex) {
    LOG(ERROR) << "exception: catch all [ " << ex.what() << "]"
               << "\n";
    return;
  }
}

char *ReadProfileCaffe2(PredictorContext pred) {
  try {
    auto predictor = (Predictor *)pred;
    if (predictor == nullptr) {
      return strdup("");
    }
    if (predictor->prof_ == nullptr) {
      return strdup("");
    }
    const auto s = predictor->prof_->read();
    const auto cstr = s.c_str();
    return strdup(cstr);
  } catch (std::exception &ex) {
    LOG(ERROR) << "exception: catch all [ " << ex.what() << "]"
               << "\n";
    return nullptr;
  }
}

int GetPredLenCaffe2(PredictorContext pred) {
  try {
    auto predictor = (Predictor *)pred;
    if (predictor == nullptr) {
      return 0;
    }
    return predictor->pred_len_;
  } catch (std::exception &ex) {
    LOG(ERROR) << "exception: catch all [ " << ex.what() << "]"
               << "\n";
    return 0;
  }
}
