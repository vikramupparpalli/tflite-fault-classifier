/*! 
 * @file
 * @brief Public API for the TFLite Fault Classifier Model
 *
 * This header provides the interface for integrating the TFLite Fault Classifier Model into an embedded application.
 *
 * Usage:
 *   1. Call TfLite_FaultClassifierModel_Init() once at startup.
 *   2. For each new set of sensor data, call TfLite_FaultClassifierModel_SetSensorData(), then TfLite_FaultClassifierModel_RunInference().
 *   3. After inference, use TfLite_FaultClassifierModel_GetClassScores(), TfLite_FaultClassifierModel_GetPredictedLabel(), or check the returned class index.
 *
 * Fault class mapping:
 *   0 = NO_FAULT
 *   1 = OVERCURRENT
 *   2 = OVERVOLTAGE
 *   3 = UNDERVOLTAGE
 *   4 = OVERTEMP
 *   5 = VFO_FAULT
 */

#ifndef TFLITE_FAULTCLASSIFIERMODEL_H
#define TFLITE_FAULTCLASSIFIERMODEL_H

#include "I_TfliteModel.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the TFLite Fault Classifier Model.
 *
 * Must be called once before using the model API.
 *
 * @return 0 on success, <0 on error
 */
int TfLite_FaultClassifierModel_Init(void);

/**
 * @brief Singleton instance of the model implementing I_TfliteModel interface.
 */
extern I_TfliteModel_t TfLite_FaultClassifierModel;

#ifdef __cplusplus
}
#endif

#endif // TFLITE_FAULTCLASSIFIERMODEL_H
