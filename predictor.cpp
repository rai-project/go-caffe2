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
  Predictor(NetDef &init_net, NetDef &net, DeviceKind device_kind);
  void Predict(float *imageData, const int batch, const int channels,
               const int width, const int height);

  DeviceKind device_kind_;
  Workspace ws_;
  std::unique_ptr<NetBase> net_;
  std::vector<string> input_names_;
  std::vector<string> output_names_;
  int pred_len_;
  const float *result_{nullptr};
  bool profile_enabled_{false};
  profile *prof_{nullptr};
  std::string profile_name_{""}, profile_metadata_{""};
};

void set_net_engine(NetDef *net_def, DeviceType device_type,
                    const string &backend) {
  for (int i = 0; i < net_def->op_size(); i++) {
    caffe2::OperatorDef *op_def = net_def->mutable_op(i);
    op_def->set_engine(backend);
    std::cout<<"device type ="<< TypeToProto(device_type)<<"\n";
    op_def->mutable_device_option()->set_device_type(TypeToProto(device_type));
  }
}

Predictor::Predictor(NetDef &init_net, NetDef &net, DeviceKind device_kind) {
  if (device_kind == CUDA_DEVICE_KIND) {
#ifdef WITH_CUDA
      DEBUG_STMT
    set_net_engine(&init_net, DeviceType::CUDA, "CUDA");
      DEBUG_STMT
    set_net_engine(&net,(DeviceType::CUDA), "CUDA");
      DEBUG_STMT
#else
    CAFFE_THROW("Not set WITH_CUDA = 1");
#endif  // WITH_CUDA
  } else {
    set_net_engine(&init_net, DeviceType::CPU, "EIGEN");
    set_net_engine(&net, DeviceType::CPU, "EIGEN");
  }

  device_kind_ = device_kind;
DEBUG_STMT

  auto init_net_ = CreateNet(init_net, &ws_);
      DEBUG_STMT
  CAFFE_ENFORCE(init_net_->Run());

      DEBUG_STMT
  net_ = CreateNet(net, &ws_);

      DEBUG_STMT
  for (auto ii = 0; ii < net.external_input_size(); ii++) {
    input_names_.emplace_back(net.external_input(ii));
  }
      DEBUG_STMT
  for (auto ii = 0; ii < net.external_output_size(); ii++) {
    output_names_.emplace_back(net.external_output(ii));
  }
      DEBUG_STMT
}

PredictorContext NewCaffe2(char *init_net_file, char *net_file,
                           DeviceKind device_kind) {
  try {
    NetDef init_net, net;
    CAFFE_ENFORCE(ReadProtoFromFile(init_net_file, &init_net));
    CAFFE_ENFORCE(ReadProtoFromFile(net_file, &net));
    auto ctx = new Predictor(init_net, net, device_kind);
    return (PredictorContext)ctx;
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

#ifdef WITH_CUDA
static CUDAContext *cuda_context = nullptr;
#endif  // WITH_CUDA

void InitCaffe2(DeviceKind device_kind) {
  static bool initialized_caffe = false;
  if (initialized_caffe) {
    return;
  }
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
    CAFFE_THROW("Not set WITH_CUDA = 1");
#endif
  }
}

void Predictor::Predict(float *imageData, const int batch, const int channels,
                        const int width, const int height) {
  result_ = nullptr;

  if (profile_enabled_) {
    unique_ptr<TimeObserver<NetBase>> net_ob =
        make_unique<TimeObserver<NetBase>>(net_.get(), &prof_, profile_name_,
                                           profile_metadata_);
    net_->AttachObserver(std::move(net_ob));
  }

  const auto data_size = batch * channels * width * height;
  std::vector<float> data;
  data.reserve(data_size);
  std::copy(imageData, imageData + data_size, data.begin());
  std::vector<int64_t> dims({batch, channels, width, height});

  // currently supports one input tensor
  TensorCPU input_tensor;
  input_tensor.Resize(dims);
  input_tensor.ShareExternalPointer(data.data());

  std::vector<TensorCPU *> inputVec{&input_tensor};
  std::vector<TensorCPU> outputVec{};

  CAFFE_ENFORCE(inputVec.size() <= input_names_.size());

  for (auto ii = 0; ii < inputVec.size(); ii++) {
    auto name = input_names_[ii];
    auto *blob = ws_.GetBlob(name);
    CAFFE_ENFORCE(blob, "Blob: ", name, " does not exist");
    if (device_kind_ == CUDA_DEVICE_KIND) {
#ifdef WITH_CUDA
      auto *tensor = blob->GetMutable<TensorCUDA>();
      tensor->CopyFrom(*inputVec[ii]);
#else
      CAFFE_THROW("Not set WITH_CUDA = 1");
#endif  // WITH_CUDA
    } else {
      auto *tensor = blob->GetMutable<TensorCPU>();
      tensor->ResizeLike(*inputVec[ii]);
      tensor->ShareData(*inputVec[ii]);
    }
  }

  CAFFE_ENFORCE(net_->Run());

  outputVec.resize(output_names_.size());
  for (auto ii = 0; ii < outputVec.size(); ii++) {
    auto name = output_names_[ii];
    auto *blob = ws_.GetBlob(name);
    CAFFE_ENFORCE(blob, "Blob: ", name, " does not exist");
    TensorCPU output_tensor;

    if (device_kind_ == CUDA_DEVICE_KIND) {
#ifdef WITH_CUDA
      // Copy TensorGPU to TensorCPU
      // Note: copy constructor no longer permitted
      // should be using Clone() explicitly
      output_tensor.CopyFrom(blob->Get<TensorCUDA>().Clone());
#else
      CAFFE_THROW("Not set WITH_CUDA = 1");
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

void PredictCaffe2(PredictorContext pred, float *imageData, const int batch,
                   const int channels, const int width, const int height) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  predictor->Predict(imageData, batch, channels, width, height);
  return;
}

const float *GetPredictionsCaffe2(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return nullptr;
  }
  return predictor->result_;
}

void DeleteCaffe2(PredictorContext pred) {
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
}

void StartProfilingCaffe2(PredictorContext pred, const char *name,
                          const char *metadata) {
  if (name == nullptr) {
    name = "";
  }
  if (metadata == nullptr) {
    metadata = "";
  }
  DEBUG_STMT

  auto predictor = (Predictor *)pred;
  predictor->profile_enabled_ = true;
  predictor->profile_name_ = std::string(name);
  predictor->profile_metadata_ = std::string(metadata);
}

void EndProfilingCaffe2(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  if (predictor->prof_) {
    predictor->prof_->end();
  }
}

void DisableProfilingCaffe2(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  if (predictor->prof_) {
    predictor->prof_->reset();
    predictor->profile_name_ = std::string("");
    predictor->profile_metadata_ = std::string("");
  }
}

char *ReadProfileCaffe2(PredictorContext pred) {
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
}

int GetPredLenCaffe2(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return 0;
  }
  return predictor->pred_len_;
}
