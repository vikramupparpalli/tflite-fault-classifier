/*!
 * @file
 * @brief Generic TFLite Model interface for embedded inference
 */

#ifndef I_TFLITEMODEL_H
#define I_TFLITEMODEL_H

#include <stdint.h>
#include <stdbool.h>

struct I_TfliteModel_Api_t;

typedef struct
{
   const struct I_TfliteModel_Api_t *api;
} I_TfliteModel_t;

typedef struct I_TfliteModel_Api_t
{
   /*
    * Set raw sensor data for inference
    * @param instance
    * @param ... (model-specific input features)
    */
   void (*SetSensorData)(I_TfliteModel_t *instance, const float *input_features, uint32_t num_features);

   /*
    * Run inference and return predicted class index
    * @param instance
    * @returns predicted class index (model-specific)
    */
   uint8_t (*RunInference)(I_TfliteModel_t *instance);

   /*
    * Get class scores (raw output, int8 or float)
    * @param instance
    * @param scores Output buffer for class scores
    * @param num_scores Number of output classes
    */
   void (*GetClassScores)(I_TfliteModel_t *instance, void *scores, uint32_t num_scores);

   /*
    * Get predicted class label string
    * @param instance
    * @returns pointer to label string
    */
   const char* (*GetPredictedLabel)(I_TfliteModel_t *instance);

   /*
    * Reset model state/statistics
    * @param instance
    */
   void (*Reset)(I_TfliteModel_t *instance);

} I_TfliteModel_Api_t;

#define TfliteModel_SetSensorData(instance, input_features, num_features) \
   (instance)->api->SetSensorData((instance), (input_features), (num_features))

#define TfliteModel_RunInference(instance) \
   (instance)->api->RunInference(instance)

#define TfliteModel_GetClassScores(instance, scores, num_scores) \
   (instance)->api->GetClassScores((instance), (scores), (num_scores))

#define TfliteModel_GetPredictedLabel(instance) \
   (instance)->api->GetPredictedLabel(instance)

#define TfliteModel_Reset(instance) \
   (instance)->api->Reset(instance)

#endif
