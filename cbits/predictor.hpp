#ifndef __PREDICTOR_HPP__
#define __PREDICTOR_HPP__

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include <stddef.h>

typedef void *PredictorContext;

typedef enum { CPU_DEVICE_KIND = 0, CUDA_DEVICE_KIND = 1 } DeviceKind;

PredictorContext NewCaffe2(char *init_net_file, char *net_file,
                           DeviceKind device);

void InitCaffe2(DeviceKind device_kind);

void PredictCaffe2(PredictorContext pred, float *imageData, const int batch,
                   const int channels, const int width, const int height);

const float *GetPredictionsCaffe2(PredictorContext pred);

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
