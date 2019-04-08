#ifndef __PREDICTOR_HPP__
#define __PREDICTOR_HPP__

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include <stddef.h>
#include <stdint.h>

#include "timer.h"

typedef void *PredictorContext;

typedef enum {
  UNKNOWN_DEVICE_KIND = 1,
  CPU_DEVICE_KIND = 0,
  CUDA_DEVICE_KIND = 1
} Caffe2_DeviceKind;

typedef enum Caffe2_DataType {
  Caffe2_Unknown = 0,
  Caffe2_Byte = 1,
  Caffe2_Char = 2,
  Caffe2_Short = 3,
  Caffe2_Int = 4,
  Caffe2_Long = 5,
  Caffe2_Half = 6,
  Caffe2_Float = 7,
  Caffe2_Double = 8,

} Caffe2_DataType;

typedef struct Caffe2_TensorInfo {
  int64_t num_elems;
  size_t nbytes;
  int *dims;
  size_t ndims;
} Caffe2_TensorInfo;

PredictorContext NewCaffe2(char *init_net_file, char *net_file,
                           Caffe2_DeviceKind device);
PredictorContext NewCaffe2FromOnnx(char *onnx_data, int64_t onnx_data_len,
                                   Caffe2_DeviceKind device);

void InitCaffe2(Caffe2_DeviceKind device_kind);

error_t AddInputCaffe2(PredictorContext pred, int64_t idx, Caffe2_DataType ty,
                       void *data, int64_t *shape, int64_t ndims);

error_t PredictCaffe2(PredictorContext pred);

void *GetPredictionsCaffe2(PredictorContext pred, int idx);

void DeleteCaffe2(PredictorContext pred);

void StartProfilingCaffe2(PredictorContext pred, const char *name,
                          const char *metadata);

void EndProfilingCaffe2(PredictorContext pred);

void DisableProfilingCaffe2(PredictorContext pred);

char *ReadProfileCaffe2(PredictorContext pred);

int GetPredLenCaffe2(PredictorContext pred);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // __PREDICTOR_HPP__
