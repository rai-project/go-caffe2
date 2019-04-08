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
#include <caffe2/core/types.h>
#include <caffe2/onnx/backend.h>
#include <caffe2/onnx/backend_rep.h>
#include <caffe2/utils/proto_utils.h>

#include <caffe2/proto/caffe2.pb.h>

#include <caffe2/core/tensor.h>

#ifdef WITH_CUDA
#include <caffe2/core/context_gpu.h>
#endif  // WITH_CUDA

#include "predictor.hpp"
#include "timer.h"
#include "timer.impl.hpp"

using namespace caffe2;
using namespace ONNX_NAMESPACE;
using std::string;

#ifdef WITH_CUDA
static CUDAContext *cuda_context = nullptr;
#endif  // WITH_CUDA

namespace mlmodelscope {

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
  void Predict();
  void SetInput(int64_t idx, Caffe2_DataType ty, void *data, int64_t *shape,
                int64_t ndims);
  void *GetOutput(int idx);

  DeviceKind device_kind_;

  Workspace *ws_{nullptr};
  NetBase *net_;
  std::vector<string> input_names_;
  std::vector<string> output_names_;
  int pred_len_;
  bool profile_enabled_{false};
  profile *prof_{nullptr};

  caffe2::onnx::Caffe2BackendRep *onnx_backend_;

  std::string profile_name_{""}, profile_metadata_{""};
};
}  // namespace mlmodelscope

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

static void set_operator_engine(NetDef *net, DeviceType device_type) {
  net->mutable_device_option()->set_device_type(TypeToProto(device_type));

  for (int i = 0; i < net->op_size(); i++) {
    caffe2::OperatorDef *op_def = net->mutable_op(i);
    op_def->mutable_device_option()->set_device_type(TypeToProto(device_type));
  }
}

static void set_operator_engine(NetDef *net, std::string backend,
                                DeviceType device_type) {
  set_operator_engine(net, device_type);

  for (int i = 0; i < net->op_size(); i++) {
    caffe2::OperatorDef *op_def = net->mutable_op(i);
    op_def->set_engine(backend);
  }
}

static void set_operator_engine(NetDef *net, DeviceKind device_kind) {
  if (device_kind == CUDA_DEVICE_KIND) {
#ifdef WITH_CUDA
    set_operator_engine(net, get_backend("cuda"), caffe2::CUDA);
    return;
#else
    throw std::runtime_error(
        "ERROR: go-caffe2 was complied with nogpu tag set");
#endif  // WITH_CUDA
  }
  set_operator_engine(net, get_backend("eigen"), caffe2::CPU);
}

mlmodelscope::Predictor::Predictor(NetDef *init_net, NetDef *pred_net_def,
                                   DeviceKind device_kind) {
  ws_ = new Workspace();
  device_kind_ = device_kind;
  ws_->RunNetOnce(*init_net);

  for (auto in : pred_net_def->external_input()) {
    auto *blob = ws_->GetBlob(in);
    if (!blob) {
      ws_->CreateBlob(in);
    }
    input_names_.emplace_back(in);
  }
  for (auto out : pred_net_def->external_output()) {
    auto *blob = ws_->GetBlob(out);
    if (!blob) {
      ws_->CreateBlob(out);
    }
    output_names_.emplace_back(out);
  }

  if (!pred_net_def->has_name()) {
    pred_net_def->set_name("go-caffe2");
  }
  net_ = ws_->CreateNet(*pred_net_def);
}

void mlmodelscope::Predictor::SetInput(int64_t idx, Caffe2_DataType ty,
                                       void *data, int64_t *cshape,
                                       int64_t ndims) {
  std::vector<int64_t> dims(cshape, cshape + ndims);
  size_t flattened_length =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());

  auto input_name = input_names_[idx];

  auto *blob = ws_->GetBlob(input_name);
  if (blob == nullptr) {
    blob = ws_->CreateBlob(input_name);
  }

  if (device_kind_ == CUDA_DEVICE_KIND) {
#ifdef WITH_CUDA
    Tensor cpu_tensor(dims, caffe2::CPU);
    cpu_tensor.ShareExternalPointer(data);
    auto tensor = BlobGetMutableTensor(blob, caffe2::CUDA);
    tensor->CopyFrom(cpu_tensor);
#else
    throw std::runtime_error(
        "ERROR: go-caffe2 was compiled with nogpu tag set");
#endif  // WITH_CUDA
  } else {
    auto tensor = BlobGetMutableTensor(blob, caffe2::CPU);
    tensor->Resize(dims);
    tensor->ShareExternalPointer(data);
  }
}

void mlmodelscope::Predictor::Predict(float *imageData, std::string input_type,
                                      const int batch_size, const int channels,
                                      const int width, const int height) {
  using mlmodelscope::TimeObserver;
  if (profile_enabled_) {
    auto net_ob = make_unique<TimeObserver<NetBase>>(
        net_, &prof_, profile_name_, profile_metadata_);
    net_->AttachObserver(std::move(net_ob));
  }

  if (!net_->Run()) {
    throw std::runtime_error("invalid run");
  }
}

TensorInfo mlmodelscope::Predictor::GetOutputInfo(int idx) {
  auto output_name = output_names_[idx];
  auto *output_blob = ws_->GetBlob(output_name);
  if (output_blob == nullptr) {
    throw std::runtime_error("output blob does not exist");
  }

  auto output_tensor = output_blob->Get<TensorCPU>();

  if (device_kind_ == CUDA_DEVICE_KIND) {
    output_tensor = output_blob->Get<caffe2::TensorCUDA>();
  }

  return TensorInfo{.size = output_tensor.numel(),
                    .nbytes = output_tensor.nbytes(),
                    dims = output_tensor.sizes().data(),
                    ndims = output_tensor.sizes().size()};
}

void *mlmodelscope::Predictor::GetOutput(int idx) {
  auto output_name = output_names_[idx];
  auto *output_blob = ws_->GetBlob(output_name);
  if (output_blob == nullptr) {
    throw std::runtime_error("output blob does not exist");
  }

  if (device_kind_ == CUDA_DEVICE_KIND) {
#ifdef WITH_CUDA
    auto output_tensor = output_blob->Get<caffe2::TensorCUDA>();
    void *result = (void *)malloc(output_tensor.nbytes());
    pred_len_ = output_tensor.size() / batch_size;
    cuda_context->CopyBytesToCPU(output_tensor.nbytes(),
                                 output_tensor.raw_data(), result);
    cuda_context->FinishDeviceComputation();
    return result;
#else
    throw std::runtime_error(
        "ERROR: go-caffe2 was compiled with nogpu tag set");
#endif  // WITH_CUDA
  }
  auto output_tensor = output_blob->Get<TensorCPU>();
  pred_len_ = output_tensor.size() / batch_size;
  void *result = (void *)malloc(output_tensor.nbytes());
  memcpy(result, output_tensor.raw_data(), output_tensor.nbytes());
  return result;
}

PredictorContext NewCaffe2(char *init_net_file, char *pred_net_file,
                           DeviceKind device_kind) {
  try {
    NetDef init_net, pred_net;
    if (!ReadProtoFromFile(init_net_file, &init_net)) {
      throw std::runtime_error("cannot read init net file");
    }
    set_operator_engine(&init_net, device_kind);
    if (!ReadProtoFromFile(pred_net_file, &pred_net)) {
      throw std::runtime_error("cannot read pred net file");
    }
    set_operator_engine(&pred_net, device_kind);
    auto ctx = new mlmodelscope::Predictor(&init_net, &pred_net, device_kind);
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

PredictorContext NewCaffe2FromOnnx(char *model_data, int64_t model_data_len,
                                   DeviceKind device_kind) {
  try {
    caffe2::onnx::Caffe2Backend onnx_instance;
    std::vector<caffe2::onnx::Caffe2Ops> extras;
    std::string content(model_data, model_data_len);
    auto onnx_backend = onnx_instance.Prepare(
        content, (device_kind == CUDA_DEVICE_KIND ? "CUDA" : "CPU"), extras);
    auto init_net = onnx_backend->init_net();
    auto pred_net = onnx_backend->pred_net();
    if (device_kind == CUDA_DEVICE_KIND) {
      set_operator_engine(&pred_net, get_backend("cuda"), caffe2::CUDA);
      set_operator_engine(&init_net, get_backend("cuda"), caffe2::CUDA);
    } else {
      set_operator_engine(&pred_net, caffe2::CPU);
      set_operator_engine(&init_net, caffe2::CPU);
    }
    auto ctx = new mlmodelscope::Predictor(&init_net, &pred_net, device_kind);
    ctx->onnx_backend_ = onnx_backend;
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

void InitCaffe2(DeviceKind device_kind) {
  static bool initialized_caffe = false;
  if (initialized_caffe) {
    return;
  }

  caffe2::ClearGlobalNetObservers();
  // caffe2::ShowLogInfoToStderr();
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
    throw std::runtime_error(
        "ERROR: go-caffe2 was compiled with nogpu tag set");
#endif
  }
}

error_t AddInputCaffe2(PredictorContext pred, int64_t idx, Caffe2_DataType ty,
                       void *data, int64_t *shape, int64_t ndims) {
  try {
    auto predictor = (mlmodelscope::Predictor *)pred;
    if (predictor == nullptr) {
      std ::cout << __func__ << "  " << __LINE__ << " ... got a null pointer\n";
      return error_invalid_memory;
    }
    predictor->SetInput(idx, ty, data, shape, ndims);
    return success;
  } catch (std::exception &ex) {
    LOG(ERROR) << "exception: catch all [ " << ex.what() << "]"
               << "\n";
    return error_exception;
  }
}
error_t PredictCaffe2(PredictorContext pred) {
  try {
    auto predictor = (mlmodelscope::Predictor *)pred;
    if (predictor == nullptr) {
      std ::cout << __func__ << "  " << __LINE__ << " ... got a null pointer\n";
      return error_invalid_memory;
    }
    predictor->Predict();
    return success;
  } catch (std::exception &ex) {
    LOG(ERROR) << "exception: catch all [ " << ex.what() << "]"
               << "\n";
    return error_exception;
  }
}

void *GetPredictionsCaffe2(PredictorContext pred, int idx) {
  try {
    auto predictor = (mlmodelscope::Predictor *)pred;
    if (predictor == nullptr) {
      return nullptr;
    }
    auto result = predictor->GetOutput(idx);
    if (result == nullptr) {
      throw std::runtime_error("expected a non-nil result");
    }
    return (void *)result;
  } catch (std::exception &ex) {
    LOG(ERROR) << "exception: catch all [ " << ex.what() << "]"
               << "\n";
    return nullptr;
  }
}

void DeleteCaffe2(PredictorContext pred) {
  try {
    auto predictor = (mlmodelscope::Predictor *)pred;
    if (predictor == nullptr) {
      return;
    }
    if (predictor->ws_ != nullptr) {
      delete predictor->ws_;
    }
    if (predictor->prof_) {
      predictor->prof_->reset();
      delete predictor->prof_;
      predictor->prof_ = nullptr;
    }
    delete predictor;

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
    auto predictor = (mlmodelscope::Predictor *)pred;
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
    auto predictor = (mlmodelscope::Predictor *)pred;
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
    auto predictor = (mlmodelscope::Predictor *)pred;
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
    auto predictor = (mlmodelscope::Predictor *)pred;
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
    auto predictor = (mlmodelscope::Predictor *)pred;
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
