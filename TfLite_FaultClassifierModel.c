
/*! 
 * @file
 * @brief Implementation of the TFLite Fault Classifier Model (I_TfliteModel interface)
 *
 * This module provides a TinyML-based fault classifier for motor control systems, implementing the I_TfliteModel interface.
 *
 * Fault detection is performed by running inference on sensor data. The predicted fault class is stored in:
 *   TfLite_FaultClassifierModel_Module.instance._private.predictedClass
 *
 * - If predictedClass == 0, then "NO_FAULT" is detected.
 * - If predictedClass > 0, a specific fault has been detected (see TFLITE_FAULTCLASSIFIERMODEL_LABELS).
 *
 * Usage:
 *   1. Call TfLite_FaultClassifierModel_Init() once at startup.
 *   2. For each new set of sensor data, call SetSensorData(), then RunInference().
 *   3. After inference, check predictedClass or use GetPredictedLabel() to determine the detected fault.
 *
 * Copyright GE Appliances - Confidential - All rights reserved
 */

#include "I_TfliteModel.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/system_setup.h"
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "model_data.h"  // uint8_t fault_classifier_tflite[] = { ... };


#define TFLITE_FAULTCLASSIFIERMODEL_ARENA_SIZE 16384
#define TFLITE_FAULTCLASSIFIERMODEL_NUM_FEATURES 6
#define TFLITE_FAULTCLASSIFIERMODEL_NUM_CLASSES 6


#define TFLITE_FAULTCLASSIFIERMODEL_SCALER_MEAN_I_MAX           10.5f
#define TFLITE_FAULTCLASSIFIERMODEL_SCALER_MEAN_I_IMBALANCE      1.2f
#define TFLITE_FAULTCLASSIFIERMODEL_SCALER_MEAN_V_NORMALIZED     1.0f
#define TFLITE_FAULTCLASSIFIERMODEL_SCALER_MEAN_TEMP_NORMALIZED  0.42f
#define TFLITE_FAULTCLASSIFIERMODEL_SCALER_MEAN_VFO_FEEDBACK     1.0f
#define TFLITE_FAULTCLASSIFIERMODEL_SCALER_MEAN_I_RMS            10.3f

#define TFLITE_FAULTCLASSIFIERMODEL_SCALER_SCALE_I_MAX           2.8f
#define TFLITE_FAULTCLASSIFIERMODEL_SCALER_SCALE_I_IMBALANCE     0.85f
#define TFLITE_FAULTCLASSIFIERMODEL_SCALER_SCALE_V_NORMALIZED    0.18f
#define TFLITE_FAULTCLASSIFIERMODEL_SCALER_SCALE_TEMP_NORMALIZED 0.21f
#define TFLITE_FAULTCLASSIFIERMODEL_SCALER_SCALE_VFO_FEEDBACK    1.0f
#define TFLITE_FAULTCLASSIFIERMODEL_SCALER_SCALE_I_RMS           2.9f


static const char *TFLITE_FAULTCLASSIFIERMODEL_LABELS[TFLITE_FAULTCLASSIFIERMODEL_NUM_CLASSES] = {
    "NO_FAULT",
    "OVERCURRENT",
    "OVERVOLTAGE",
    "UNDERVOLTAGE",
    "OVERTEMP",
    "VFO_FAULT"
};



/**
 * @struct TfLite_FaultClassifierModel_t
 * @brief Internal state for the TFLite Fault Classifier Model.
 *
 * @var sensorData      Latest input sensor data (raw features)
 * @var features        Engineered features for inference
 * @var classScores     Output class scores from the model (int8)
 * @var predictedClass  Index of the predicted fault class (0 = NO_FAULT, >0 = fault)
 *
 * After calling RunInference(), predictedClass holds the detected fault index.
 */
typedef struct
{
    struct {
        float sensorData[TFLITE_FAULTCLASSIFIERMODEL_NUM_FEATURES];
        float features[TFLITE_FAULTCLASSIFIERMODEL_NUM_FEATURES];
        int8_t classScores[TFLITE_FAULTCLASSIFIERMODEL_NUM_CLASSES];
        uint8_t predictedClass;
    } _private;
} TfLite_FaultClassifierModel_t;

static struct {
    const tflite::Model *model;
    tflite::MicroInterpreter *interpreter;
    uint8_t tensorArena[TFLITE_FAULTCLASSIFIERMODEL_ARENA_SIZE];
    TfLiteTensor *inputTensor;
    TfLiteTensor *outputTensor;
    TfLite_FaultClassifierModel_t instance;
} TfLite_FaultClassifierModel_Module = {0};

/*! 
 * @brief Feature engineering for the classifier
 */
static void TfLite_FaultClassifierModel_EngineerFeatures(const float *sensorData, float *features)
{
    float Ia = sensorData[0];
    float Ib = sensorData[1];
    float Ic = sensorData[2];
    float Vdc = sensorData[3];
    float Temp = sensorData[4];
    float VFO_feedback = sensorData[5];

    float I_max = (Ia > Ib) ? Ia : Ib;
    I_max = (I_max > Ic) ? I_max : Ic;
    features[0] = I_max;

    float I_min = (Ia < Ib) ? Ia : Ib;
    I_min = (I_min < Ic) ? I_min : Ic;
    float I_imbalance = I_max - I_min;
    features[1] = I_imbalance;

    float V_normalized = Vdc / 340.0f;
    features[2] = V_normalized;

    float Temp_normalized = Temp / 125.0f;
    features[3] = Temp_normalized;

    features[4] = VFO_feedback;

    float I_sq_sum = (Ia*Ia + Ib*Ib + Ic*Ic) / 3.0f;
    float I_rms_estimate = sqrtf(I_sq_sum);
    features[5] = I_rms_estimate;
}

/*! 
 * @brief Feature scaling and quantization
 */
static void TfLite_FaultClassifierModel_ScaleAndQuantizeFeatures(const float *features, TfLiteTensor *inputTensor)
{
    float means[TFLITE_FAULTCLASSIFIERMODEL_NUM_FEATURES] = {
        TFLITE_FAULTCLASSIFIERMODEL_SCALER_MEAN_I_MAX,
        TFLITE_FAULTCLASSIFIERMODEL_SCALER_MEAN_I_IMBALANCE,
        TFLITE_FAULTCLASSIFIERMODEL_SCALER_MEAN_V_NORMALIZED,
        TFLITE_FAULTCLASSIFIERMODEL_SCALER_MEAN_TEMP_NORMALIZED,
        TFLITE_FAULTCLASSIFIERMODEL_SCALER_MEAN_VFO_FEEDBACK,
        TFLITE_FAULTCLASSIFIERMODEL_SCALER_MEAN_I_RMS
    };
    float scales[TFLITE_FAULTCLASSIFIERMODEL_NUM_FEATURES] = {
        TFLITE_FAULTCLASSIFIERMODEL_SCALER_SCALE_I_MAX,
        TFLITE_FAULTCLASSIFIERMODEL_SCALER_SCALE_I_IMBALANCE,
        TFLITE_FAULTCLASSIFIERMODEL_SCALER_SCALE_V_NORMALIZED,
        TFLITE_FAULTCLASSIFIERMODEL_SCALER_SCALE_TEMP_NORMALIZED,
        TFLITE_FAULTCLASSIFIERMODEL_SCALER_SCALE_VFO_FEEDBACK,
        TFLITE_FAULTCLASSIFIERMODEL_SCALER_SCALE_I_RMS
    };
    for (int i = 0; i < TFLITE_FAULTCLASSIFIERMODEL_NUM_FEATURES; i++)
    {
        float scaled = (features[i] - means[i]) / scales[i];
        int16_t q = (int16_t)roundf(scaled * 128.0f);
        inputTensor->data.int8[i] = (q < -128) ? -128 : ((q > 127) ? 127 : (int8_t)q);
    }
}


/*! 
 * @brief Set sensor data for inference
 */
static void TfLite_FaultClassifierModel_SetSensorData(I_TfliteModel_t *instance, const float *inputFeatures, uint32_t numFeatures)
{
    if (numFeatures != TFLITE_FAULTCLASSIFIERMODEL_NUM_FEATURES) return;
    memcpy(TfLite_FaultClassifierModel_Module.instance._private.sensorData, inputFeatures, sizeof(float) * TFLITE_FAULTCLASSIFIERMODEL_NUM_FEATURES);
}

/*! 
 * @brief Run inference and return predicted class
 *
 * After calling this function, the detected fault is available in:
 *   TfLite_FaultClassifierModel_Module.instance._private.predictedClass
 *
 * - 0: NO_FAULT
 * - 1: OVERCURRENT
 * - 2: OVERVOLTAGE
 * - 3: UNDERVOLTAGE
 * - 4: OVERTEMP
 * - 5: VFO_FAULT
 *
 * You can also use TfLite_FaultClassifierModel_GetPredictedLabel() for a string label.
 */
static uint8_t TfLite_FaultClassifierModel_RunInference(I_TfliteModel_t *instance)
{
    TfLite_FaultClassifierModel_EngineerFeatures(TfLite_FaultClassifierModel_Module.instance._private.sensorData, TfLite_FaultClassifierModel_Module.instance._private.features);
    TfLite_FaultClassifierModel_ScaleAndQuantizeFeatures(TfLite_FaultClassifierModel_Module.instance._private.features, TfLite_FaultClassifierModel_Module.inputTensor);
    if (TfLite_FaultClassifierModel_Module.interpreter->Invoke() != kTfLiteOk)
    {
        TfLite_FaultClassifierModel_Module.instance._private.predictedClass = 0;
        memset(TfLite_FaultClassifierModel_Module.instance._private.classScores, 0, sizeof(TfLite_FaultClassifierModel_Module.instance._private.classScores));
        return 0;
    }
    for (int i = 0; i < TFLITE_FAULTCLASSIFIERMODEL_NUM_CLASSES; i++)
    {
        TfLite_FaultClassifierModel_Module.instance._private.classScores[i] = TfLite_FaultClassifierModel_Module.outputTensor->data.int8[i];
    }
    int8_t maxScore = TfLite_FaultClassifierModel_Module.instance._private.classScores[0];
    TfLite_FaultClassifierModel_Module.instance._private.predictedClass = 0;
    for (int i = 1; i < TFLITE_FAULTCLASSIFIERMODEL_NUM_CLASSES; i++)
    {
        if (TfLite_FaultClassifierModel_Module.instance._private.classScores[i] > maxScore)
        {
            maxScore = TfLite_FaultClassifierModel_Module.instance._private.classScores[i];
            TfLite_FaultClassifierModel_Module.instance._private.predictedClass = i;
        }
    }
    return TfLite_FaultClassifierModel_Module.instance._private.predictedClass;
}

/*! 
 * @brief Get class scores (raw output)
 */
static void TfLite_FaultClassifierModel_GetClassScores(I_TfliteModel_t *instance, void *scores, uint32_t numScores)
{
    if (numScores > TFLITE_FAULTCLASSIFIERMODEL_NUM_CLASSES) numScores = TFLITE_FAULTCLASSIFIERMODEL_NUM_CLASSES;
    memcpy(scores, TfLite_FaultClassifierModel_Module.instance._private.classScores, numScores);
}

/*! 
 * @brief Get predicted class label string
 *
 * Returns a pointer to a string describing the detected fault.
 * Example: "NO_FAULT", "OVERCURRENT", etc.
 */
static const char *TfLite_FaultClassifierModel_GetPredictedLabel(I_TfliteModel_t *instance)
{
    if (TfLite_FaultClassifierModel_Module.instance._private.predictedClass < TFLITE_FAULTCLASSIFIERMODEL_NUM_CLASSES)
    {
        return TFLITE_FAULTCLASSIFIERMODEL_LABELS[TfLite_FaultClassifierModel_Module.instance._private.predictedClass];
    }
    return "UNKNOWN";
}

/*! 
 * @brief Reset model state/statistics
 */
static void TfLite_FaultClassifierModel_Reset(I_TfliteModel_t *instance)
{
    memset(&TfLite_FaultClassifierModel_Module.instance, 0, sizeof(TfLite_FaultClassifierModel_Module.instance));
}


static const I_TfliteModel_Api_t TfLite_FaultClassifierModel_Api = {
    .SetSensorData = TfLite_FaultClassifierModel_SetSensorData,
    .RunInference = TfLite_FaultClassifierModel_RunInference,
    .GetClassScores = TfLite_FaultClassifierModel_GetClassScores,
    .GetPredictedLabel = TfLite_FaultClassifierModel_GetPredictedLabel,
    .Reset = TfLite_FaultClassifierModel_Reset
};

I_TfliteModel_t TfLite_FaultClassifierModel = {
    .api = &TfLite_FaultClassifierModel_Api
};

/*! 
 * @brief Initialize the TFLite Fault Classifier Model
 */
int TfLite_FaultClassifierModel_Init(void)
{
    tflite::InitializeTarget();
    TfLite_FaultClassifierModel_Module.model = tflite::GetModel(fault_classifier_tflite);
    if (TfLite_FaultClassifierModel_Module.model->version() != TFLITE_SCHEMA_VERSION) return -1;
    static tflite::MicroMutableOpResolver<6> resolver;
    if (resolver.AddFullyConnected() != kTfLiteOk) return -2;
    if (resolver.AddRelu() != kTfLiteOk) return -3;
    if (resolver.AddSoftmax() != kTfLiteOk) return -4;
    static tflite::MicroInterpreter static_interpreter(
        TfLite_FaultClassifierModel_Module.model, resolver, TfLite_FaultClassifierModel_Module.tensorArena, TFLITE_FAULTCLASSIFIERMODEL_ARENA_SIZE
    );
    TfLite_FaultClassifierModel_Module.interpreter = &static_interpreter;
    if (TfLite_FaultClassifierModel_Module.interpreter->AllocateTensors() != kTfLiteOk) return -5;
    TfLite_FaultClassifierModel_Module.inputTensor = TfLite_FaultClassifierModel_Module.interpreter->input(0);
    TfLite_FaultClassifierModel_Module.outputTensor = TfLite_FaultClassifierModel_Module.interpreter->output(0);
    TfLite_FaultClassifierModel_Reset(&TfLite_FaultClassifierModel);
    return 0;
}
