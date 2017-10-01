#ifndef __PREDICTOR_HPP__
#define __PREDICTOR_HPP__

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include <stddef.h>

typedef void *PredictorContext;

typedef enum { CPU_DEVICE_KIND = 0, CUDA_DEVICE_KIND = 1 } DeviceKind;

PredictorContext New(char *predict_net_file, char *init_net_file,
                     DeviceKind device);

const char *Predict(PredictorContext pred, float *imageData, const int batch,
                    const int channels, const int width, const int height,
                    DeviceKind device);
void Delete(PredictorContext pred, DeviceKind device);

int Init(DeviceKind device);

int InitCUDA();

void StartProfiling(PredictorContext pred, const char *name,
                    const char *metadata, DeviceKind device);

void EndProfiling(PredictorContext pred, DeviceKind device);

void DisableProfiling(PredictorContext pred, DeviceKind device);

char *ReadProfile(PredictorContext pred, DeviceKind device);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // __PREDICTOR_HPP__
