#ifndef __PREDICTOR_HPP__
#define __PREDICTOR_HPP__

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include <stddef.h>

typedef void *PredictorContext;

PredictorContext New(char *predict_net_file, char *init_net_file);

const char *Predict(PredictorContext pred, float *imageData, const int batch,
                    const int channels, const int width, const int height);
void Delete(PredictorContext pred);

void Init();

void StartProfiling(PredictorContext pred, const char *name,
                    const char *metadata);

void EndProfiling(PredictorContext pred);

void DisableProfiling(PredictorContext pred);

char *ReadProfile(PredictorContext pred);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // __PREDICTOR_HPP__
