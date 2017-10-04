#ifndef __PREDICTOR_HPP__
#define __PREDICTOR_HPP__

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include <stddef.h>

typedef void *PredictorContext;

typedef enum { CPU_DEVICE_KIND = 0, CUDA_DEVICE_KIND = 1 } DeviceKind;

PredictorContext NewCaffe2(char *predict_net_file, char *init_net_file,
                     DeviceKind device);

const char *PredictCaffe2(PredictorContext pred, float *imageData, const int batch,
                    const int channels, const int width, const int height,
                    DeviceKind device);
void DeleteCaffe2(PredictorContext pred, DeviceKind device);

int InitCaffe2(DeviceKind device);

int InitCUDACaffe2();

void StartProfilingCaffe2(PredictorContext pred, const char *name,
                    const char *metadata, DeviceKind device);

void EndProfilingCaffe2(PredictorContext pred, DeviceKind device);

void DisableProfilingCaffe2(PredictorContext pred, DeviceKind device);

char *ReadProfileCaffe2(PredictorContext pred, DeviceKind device);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // __PREDICTOR_HPP__
