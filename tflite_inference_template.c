/*
 * Motor Control Fault Classifier - TFLite Micro Inference
 * 
 * Integrates quantized TinyML model into 16 kHz interrupt loop
 * 
 * Requirements:
 *   - TensorFlow Lite for Microcontrollers (tfLite Micro)
 *   - fault_classifier.tflite model (binary blob in flash)
 *   - ~40 KB RAM for model weights and working buffers
 * 
 * Model info:
 *   - Input: 6 features (int8 quantized)
 *   - Output: 6 class probabilities (int8)
 *   - Latency: ~30-50 µs on STM32 Cortex-M4 @ 80 MHz
 *   - Model size: ~5-8 KB
 */

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/system_setup.h"
#include <string.h>
#include <math.h>

// ============================================================================
// CONFIGURATION
// ============================================================================

// Include the quantized model as a C array
// Generated from: xxd -i fault_classifier.tflite > model_data.h
#include "model_data.h"  // uint8_t fault_classifier_tflite[] = { ... };
                        // extern const int fault_classifier_tflite_len;

// Memory arena for TFLite interpreter (must be in SRAM, typically 8-16 KB)
#define ARENA_SIZE 16384  // 16 KB for model weights and activations

// Feature scaling parameters (from training)
// These values come from model_info.json exported during training
#define SCALER_MEAN_I_MAX           10.5f
#define SCALER_MEAN_I_IMBALANCE      1.2f
#define SCALER_MEAN_V_NORMALIZED     1.0f
#define SCALER_MEAN_TEMP_NORMALIZED  0.42f
#define SCALER_MEAN_VFO_FEEDBACK     1.0f
#define SCALER_MEAN_I_RMS            10.3f

#define SCALER_SCALE_I_MAX           2.8f
#define SCALER_SCALE_I_IMBALANCE     0.85f
#define SCALER_SCALE_V_NORMALIZED    0.18f
#define SCALER_SCALE_TEMP_NORMALIZED 0.21f
#define SCALER_SCALE_VFO_FEEDBACK    1.0f
#define SCALER_SCALE_I_RMS           2.9f

// Output class names
const char* FAULT_LABELS[] = {
    "NO_FAULT",
    "OVERCURRENT",
    "OVERVOLTAGE",
    "UNDERVOLTAGE",
    "OVERTEMP",
    "VFO_FAULT",
};

// ============================================================================
// GLOBAL STATE
// ============================================================================

// TFLite model and interpreter
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

// Memory arena
uint8_t tensor_arena[ARENA_SIZE];

// Input/output tensors
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

// Feature buffer (raw measurements)
struct {
    float Ia, Ib, Ic;           // Phase currents (A)
    float Vdc;                  // DC bus voltage (V)
    float Temp;                 // IPM temperature (°C)
    float VFO_feedback;         // Gate driver ON/OFF feedback (1=ON, 0=OFF)
} sensor_data;

// Engineered features for model input
float features[6];

// Model output
uint8_t predicted_class = 0;
int8_t class_scores[6];

// Inference statistics (optional, for diagnostics)
struct {
    uint32_t inference_count;
    uint32_t total_us;
    uint32_t max_us;
} inference_stats;

// ============================================================================
// INITIALIZATION
// ============================================================================

/**
 * Initialize TFLite Micro interpreter.
 * Call this once at startup, before 16 kHz loop begins.
 */
int fault_classifier_init(void) {
    tflite::InitializeTarget();
    
    // Load model from flash
    model = tflite::GetModel(fault_classifier_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        return -1;  // Model version mismatch
    }
    
    // Create resolver (must include all ops used in model)
    static tflite::MicroMutableOpResolver<6> resolver;
    if (resolver.AddFullyConnected() != kTfLiteOk) return -2;
    if (resolver.AddRelu() != kTfLiteOk) return -3;
    if (resolver.AddSoftmax() != kTfLiteOk) return -4;
    
    // Create interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, ARENA_SIZE
    );
    interpreter = &static_interpreter;
    
    // Allocate tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        return -5;
    }
    
    // Get input/output tensor pointers
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);
    
    // Verify tensor types
    if (input_tensor->type != kTfLiteInt8) {
        return -6;  // Expected int8 input
    }
    if (output_tensor->type != kTfLiteInt8) {
        return -7;  // Expected int8 output
    }
    
    // Initialize stats
    inference_stats.inference_count = 0;
    inference_stats.total_us = 0;
    inference_stats.max_us = 0;
    
    return 0;  // Success
}

// ============================================================================
// FEATURE ENGINEERING
// ============================================================================

/**
 * Extract 7 engineered features from raw sensor measurements.
 * Must match the feature engineering done during training.
 * 
 * Raw inputs (6 values):
 *   - Ia, Ib, Ic (phase currents in Amps)
 *   - Vdc (DC bus voltage in Volts)
 *   - Temp (IPM temperature in °C)
 *   - VFO_feedback (gate driver ON/OFF feedback)
 *
 * Engineered features (6 values):
 *   - I_max: Maximum phase current
 *   - I_imbalance: Spread between phase currents
 *   - V_normalized: DC voltage as fraction of 340V nominal
 *   - Temp_normalized: Temperature as fraction of 125°C max
 *   - VFO_feedback: Pass-through ON/OFF feedback
 *   - I_rms_estimate: RMS current of three phases
 */
void engineer_features(void) {
    float Ia = sensor_data.Ia;
    float Ib = sensor_data.Ib;
    float Ic = sensor_data.Ic;
    float Vdc = sensor_data.Vdc;
    float Temp = sensor_data.Temp;
    float VFO_feedback = sensor_data.VFO_feedback;

    // Feature 1: Max phase current
    float I_max = (Ia > Ib) ? Ia : Ib;
    I_max = (I_max > Ic) ? I_max : Ic;
    features[0] = I_max;

    // Feature 2: Phase current imbalance
    float I_min = (Ia < Ib) ? Ia : Ib;
    I_min = (I_min < Ic) ? I_min : Ic;
    float I_imbalance = I_max - I_min;
    features[1] = I_imbalance;

    // Feature 3: Voltage normalized (340V nominal)
    float V_normalized = Vdc / 340.0f;
    features[2] = V_normalized;

    // Feature 4: Temperature normalized (125°C max)
    float Temp_normalized = Temp / 125.0f;
    features[3] = Temp_normalized;

    // Feature 5: VFO_feedback (pass-through, 1=ON, 0=OFF)
    features[4] = VFO_feedback;

    // Feature 6: RMS current estimate
    float I_sq_sum = (Ia*Ia + Ib*Ib + Ic*Ic) / 3.0f;
    float I_rms_estimate = sqrtf(I_sq_sum);
    features[5] = I_rms_estimate;
}

// ============================================================================
// FEATURE SCALING & QUANTIZATION
// ============================================================================

/**
 * Normalize features (zero-mean, unit variance) and quantize to int8.
 * Must use the same scaler parameters from training (in model_info.json).
 */
void scale_and_quantize_features(void) {
    // Scaling parameters (from StandardScaler during training)
    float means[6] = {
        SCALER_MEAN_I_MAX,
        SCALER_MEAN_I_IMBALANCE,
        SCALER_MEAN_V_NORMALIZED,
        SCALER_MEAN_TEMP_NORMALIZED,
        SCALER_MEAN_VFO_FEEDBACK,
        SCALER_MEAN_I_RMS
    };

    float scales[6] = {
        SCALER_SCALE_I_MAX,
        SCALER_SCALE_I_IMBALANCE,
        SCALER_SCALE_V_NORMALIZED,
        SCALER_SCALE_TEMP_NORMALIZED,
        SCALER_SCALE_VFO_FEEDBACK,
        SCALER_SCALE_I_RMS
    };

    // Normalize: (x - mean) / scale
    float scaled[6];
    for (int i = 0; i < 6; i++) {
        scaled[i] = (features[i] - means[i]) / scales[i];
    }

    // Quantize to int8: scale to [-128, 127] range
    // (Inverse of dequantization: value = (quantized - zero_point) * scale)
    // For simplicity, assume zero_point=0, scale=1/128 (from training)
    for (int i = 0; i < 6; i++) {
        int16_t q = (int16_t)roundf(scaled[i] * 128.0f);
        // Clamp to int8 range
        input_tensor->data.int8[i] = (q < -128) ? -128 : ((q > 127) ? 127 : (int8_t)q);
    }
}

// ============================================================================
// INFERENCE
// ============================================================================

/**
 * Run inference on current sensor data.
 * Should be called periodically (every 100 ms or every N cycles).
 * 
 * Timing: ~30-50 µs on ARM Cortex-M4 @ 80 MHz
 * 
 * Returns: Predicted fault class index (0-6)
 */
uint8_t run_inference(void) {
    if (interpreter == nullptr) {
        return 0;  // No fault if model not initialized
    }
    
    // Record start time for latency measurement
    uint32_t start_us = get_microseconds();  // Implement based on your timer
    
    // 1. Engineer features from raw sensor data
    engineer_features();
    
    // 2. Scale and quantize
    scale_and_quantize_features();
    
    // 3. Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
        return 0;  // Error: return NO_FAULT (safe state)
    }
    
    // 4. Get output probabilities
    for (int i = 0; i < 6; i++) {
        class_scores[i] = output_tensor->data.int8[i];
    }

    // 5. Find max class (highest confidence)
    int8_t max_score = class_scores[0];
    predicted_class = 0;
    for (int i = 1; i < 6; i++) {
        if (class_scores[i] > max_score) {
            max_score = class_scores[i];
            predicted_class = i;
        }
    }
    
    // 6. Update statistics
    uint32_t elapsed_us = get_microseconds() - start_us;
    inference_stats.inference_count++;
    inference_stats.total_us += elapsed_us;
    if (elapsed_us > inference_stats.max_us) {
        inference_stats.max_us = elapsed_us;
    }
    
    return predicted_class;
}

// ============================================================================
// INTEGRATION WITH 16 kHz LOOP
// ============================================================================

/**
 * Global inference control
 */
struct {
    uint32_t sample_interval;  // Run inference every N samples (e.g., 1600 = every 100 ms)
    uint32_t sample_count;
    uint8_t last_prediction;
    uint32_t prediction_age_samples;
} inference_control = {
    .sample_interval = 1600,  // 100 ms at 16 kHz = 1600 samples
    .sample_count = 0,
    .last_prediction = 0,
    .prediction_age_samples = 0
};

/**
 * Call from within your 16 kHz interrupt handler.
 * 
 * Example:
 *   void __attribute__((interrupt)) tim1_handler(void) {
 *       // ... existing ADC reads, PWM updates, etc. ...
 *       fault_classifier_16khz_tick(Ia, Ib, Ic, Vdc, Temp, VFO_feedback);
 *       // ... rest of interrupt ...
 *   }
 */
void fault_classifier_16khz_tick(
    float Ia, float Ib, float Ic,
    float Vdc, float Temp,
    float VFO_feedback
) {
    // Update sensor data
    sensor_data.Ia = Ia;
    sensor_data.Ib = Ib;
    sensor_data.Ic = Ic;
    sensor_data.Vdc = Vdc;
    sensor_data.Temp = Temp;
    sensor_data.VFO_feedback = VFO_feedback;
    
    // Periodically run inference
    inference_control.sample_count++;
    inference_control.prediction_age_samples++;
    
    if (inference_control.sample_count >= inference_control.sample_interval) {
        inference_control.sample_count = 0;
        inference_control.last_prediction = run_inference();
        inference_control.prediction_age_samples = 0;
    }
}

/**
 * Get the latest fault prediction (non-blocking).
 */
uint8_t get_fault_prediction(void) {
    return inference_control.last_prediction;
}

/**
 * Get human-readable fault name.
 */
const char* get_fault_name(uint8_t fault_class) {
    if (fault_class < 6) {
        return FAULT_LABELS[fault_class];
    }
    return "UNKNOWN";
}

/**
 * Get prediction confidence (approximate, from int8 scores).
 * Returns 0-100 (percent).
 */
uint8_t get_prediction_confidence(void) {
    if (predicted_class >= 6) return 0;

    // Confidence = (max_score - mean_of_others) / scale
    int8_t max_score = class_scores[predicted_class];
    int32_t sum_others = 0;
    for (int i = 0; i < 6; i++) {
        if (i != predicted_class) {
            sum_others += class_scores[i];
        }
    }
    int8_t mean_other = sum_others / 5;
    
    // Rough confidence metric (0-100)
    int16_t confidence = 100 + (max_score - mean_other) / 2;
    if (confidence < 0) confidence = 0;
    if (confidence > 100) confidence = 100;
    
    return (uint8_t)confidence;
}

// ============================================================================
// DIAGNOSTICS & LOGGING
// ============================================================================

/**
 * Print model inference statistics (for debugging/optimization).
 */
void print_inference_stats(void) {
    if (inference_stats.inference_count == 0) return;
    
    uint32_t avg_us = inference_stats.total_us / inference_stats.inference_count;
    
    printf("Inference Stats:\n");
    printf("  Total inferences: %lu\n", inference_stats.inference_count);
    printf("  Avg latency: %lu µs\n", avg_us);
    printf("  Max latency: %lu µs\n", inference_stats.max_us);
    printf("  Budget: 62.5 µs (%.1f%% used)\n", 
           (float)avg_us / 62.5f * 100.0f);
}

/**
 * Dump current feature values (for debugging).
 */
void dump_features(void) {
    printf("Features: [");
    for (int i = 0; i < 6; i++) {
        printf("%.3f", features[i]);
        if (i < 5) printf(", ");
    }
    printf("]\n");
}

/**
 * Dump model output scores (for debugging).
 */
void dump_output_scores(void) {
    printf("Output scores: [");
    for (int i = 0; i < 6; i++) {
        printf("%d", class_scores[i]);
        if (i < 5) printf(", ");
    }
    printf("]\n");
    printf("Prediction: %s (confidence: %d%%)\n",
           get_fault_name(predicted_class),
           get_prediction_confidence());
}

// ============================================================================
// STUB: Replace with your timer implementation
// ============================================================================

/**
 * Get current time in microseconds.
 * Implement based on your microcontroller's timer.
 * 
 * Example for STM32:
 *   static inline uint32_t get_microseconds(void) {
 *       return TIM6->CNT;  // Assuming TIM6 counts in µs
 *   }
 */
uint32_t get_microseconds(void) {
    // TODO: Implement for your platform
    return 0;
}

// ============================================================================
// END OF INFERENCE CODE
// ============================================================================
