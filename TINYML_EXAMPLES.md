"""
TinyML Motor Fault Classifier - Practical Examples
===================================================

Copy & paste examples to get started immediately
"""

# ============================================================================
# EXAMPLE 1: Train Model in 5 Minutes
# ============================================================================
"""
File: train_example.py

Quick training example - generates model in ~5 minutes
"""

from tinyml_fault_classifier import (
    MotorFaultDataGenerator,
    engineer_features,
    build_tinyml_model,
    train_model,
    convert_to_tflite_quantized,
    save_tflite_model,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

print("Step 1: Generate synthetic fault data...")
generator = MotorFaultDataGenerator(seed=42)
X_raw, y = generator.generate_balanced_dataset()
print(f"  Generated {len(X_raw)} samples")

print("\nStep 2: Engineer features...")
X_features = engineer_features(X_raw)
print(f"  Shape: {X_features.shape}")

print("\nStep 3: Normalize...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

print("\nStep 4: Split data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("\nStep 5: Train Keras model...")
model, history = train_model(X_train, y_train, X_test, y_test, epochs=50)

print("\nStep 6: Evaluate...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"  Test Accuracy: {accuracy:.4f}")

print("\nStep 7: Convert to TFLite...")
tflite_model = convert_to_tflite_quantized(model, X_train)
print(f"  Model size: {len(tflite_model)} bytes")

print("\nStep 8: Save...")
save_tflite_model(tflite_model, 'fault_classifier.tflite')
model.save('fault_classifier_keras.h5')

print("\n✓ DONE! Your model is ready:")
print("  - fault_classifier.tflite (for microcontroller)")
print("  - fault_classifier_keras.h5 (for retraining)")


# ============================================================================
# EXAMPLE 2: Validate Model Accuracy
# ============================================================================
"""
File: validate_example.py

Validate quantized model on test data
"""

from validate_and_retrain import TFLiteValidator
from tinyml_fault_classifier import engineer_features

print("Loading model...")
validator = TFLiteValidator(
    'fault_classifier.tflite',
    'model_info.json'
)

print("\nValidating on test data...")
# Note: You'll need to create a test CSV with format:
# Ia,Ib,Ic,Vdc,Temp,VFO_feedback,R_winding,label

metrics = validator.validate_on_csv('test_data.csv', engineer_features)

print("\n" + "="*70)
print(f"Model Accuracy: {metrics['accuracy']:.4f}")
print(f"Low confidence predictions: {np.sum(metrics['low_confidence_mask'])}")
print("="*70)

# Print confusion matrix
print("\nConfusion Matrix:")
print(metrics['confusion_matrix'])

# Print per-class results
print("\nPer-Class Results:")
print(metrics['report'])


# ============================================================================
# EXAMPLE 3: Embedded C - Minimal Integration
# ============================================================================
"""
File: motor_control.c

Minimal example integrating classifier into 16 kHz interrupt
"""

#include "fault_classifier.h"

// Global sensor values (updated by ADC ISR)
volatile float Ia_reading = 0;
volatile float Ib_reading = 0;
volatile float Ic_reading = 0;
volatile float Vdc_reading = 0;
volatile float Temp_reading = 0;
volatile float VFO_feedback = 1; // 1 = ON, 0 = OFF
volatile float R_winding = 1.0;

// Startup: Initialize classifier
void init_motor_control(void) {
    int result = fault_classifier_init();
    if (result != 0) {
        printf("ERROR: Classifier init failed: %d\n", result);
        // Disable ML, use rule-based only
    }
    
    // Start 16 kHz timer
    start_16khz_timer();
}

// 16 kHz interrupt handler
void __attribute__((interrupt)) TIM1_UP_IRQHandler(void) {
    // Clear interrupt flag
    TIM1->SR &= ~TIM_SR_UIF;
    
    // ===== SENSOR READS =====
    float Ia = Ia_reading;
    float Ib = Ib_reading;
    float Ic = Ic_reading;
    float Vdc = Vdc_reading;
    float Temp = Temp_reading;
    
    // ===== FAULT CLASSIFICATION (every 100 ms) =====
    fault_classifier_16khz_tick(Ia, Ib, Ic, Vdc, Temp, VFO_feedback, R_winding);
    
    // ===== GET LATEST PREDICTION =====
    // (prediction is updated every 100 ms, not every cycle)
    uint8_t fault_class = get_fault_prediction();
    
    // ===== DECIDE ACTION =====
    if (fault_class == 0) {
        // NO_FAULT: Normal operation
        set_pwm_duty(torque_command);
    } else {
        // FAULT DETECTED: Take action
        uint8_t confidence = get_prediction_confidence();
        const char* fault_name = get_fault_name(fault_class);
        
        if (confidence >= 80) {
            // High confidence: Act immediately
            disable_pwm();
            log_fault(fault_class, Ia, Vdc, Temp);
        } else if (confidence >= 60) {
            // Medium confidence: Warn, reduce power
            reduce_torque(0.5);  // 50% power
            log_warning(fault_class, confidence);
        } else {
            // Low confidence: Ignore, use rules only
        }
    }
    
    // ===== REST OF ISR (your normal motor control) =====
    update_pwm_commutation();
    sample_temperature();
}

// Main loop
int main(void) {
    init_motor_control();
    
    while (1) {
        // Application code...
        
        // Every 1 second, log diagnostics
        static int log_counter = 0;
        if (log_counter++ >= 16000) {
            uint8_t fault = get_fault_prediction();
            printf("Fault: %s (confidence: %d%%)\n",
                   get_fault_name(fault),
                   get_prediction_confidence());
            log_counter = 0;
        }
    }
}


// ============================================================================
// EXAMPLE 4: Feature Engineering - Verify C Code Matches Python
// ============================================================================
"""
Python (training):
"""

def engineer_features_python(X_raw):
    Ia, Ib, Ic, Vdc, Temp, VFO_feedback, R_winding = X_raw[0]
    
    I_max = max(Ia, Ib, Ic)
    I_imbalance = max(Ia, Ib, Ic) - min(Ia, Ib, Ic)
    V_normalized = Vdc / 48.0
    Temp_normalized = Temp / 120.0
    VFO_feedback = 1 if VFO_feedback else 0
    R_normalized = R_winding - 1.0
    I_rms = (Ia**2 + Ib**2 + Ic**2) / 3.0) ** 0.5
    
    return [I_max, I_imbalance, V_normalized, Temp_normalized, 
            VFO_feedback, R_normalized, I_rms]


"""
C (embedded):
"""

void engineer_features(void) {
    float Ia = sensor_data.Ia;
    float Ib = sensor_data.Ib;
    float Ic = sensor_data.Ic;
    float Vdc = sensor_data.Vdc;
    float Temp = sensor_data.Temp;
    float VFO_feedback = sensor_data.VFO_feedback; // 1 = ON, 0 = OFF
    float R_winding = sensor_data.R_winding;
    
    // Feature 1: Max phase current
    float I_max = (Ia > Ib) ? Ia : Ib;
    I_max = (I_max > Ic) ? I_max : Ic;
    features[0] = I_max;
    
    // Feature 2: Phase current imbalance
    float I_min = (Ia < Ib) ? Ia : Ib;
    I_min = (I_min < Ic) ? I_min : Ic;
    float I_imbalance = I_max - I_min;
    features[1] = I_imbalance;
    
    // Feature 3: Voltage normalized (48V nominal)
    float V_normalized = Vdc / 48.0f;
    features[2] = V_normalized;
    
    // Feature 4: Temperature normalized (0-120°C)
    float Temp_normalized = Temp / 120.0f;
    features[3] = Temp_normalized;
    
    // Feature 5: VFO feedback (ON/OFF)
    float VFO_feedback_feature = (VFO_feedback != 0) ? 1.0f : 0.0f;
    features[4] = VFO_feedback_feature;
    
    // Feature 6: Resistance normalized (1.0 baseline)
    float R_normalized = R_winding - 1.0f;
    features[5] = R_normalized;
    
    // Feature 7: RMS current estimate
    float I_sq_sum = (Ia*Ia + Ib*Ib + Ic*Ic) / 3.0f;
    float I_rms_estimate = sqrtf(I_sq_sum);
    features[6] = I_rms_estimate;
}


# ============================================================================
# EXAMPLE 5: Real-World Data Collection & Retraining
# ============================================================================
"""
After 1 week of operation, you'll have real fault data.
Here's how to retrain the model:
"""

from validate_and_retrain import RealDataRetrainer

print("Step 1: Load real-world fault data collected from your system...")
retrainer = RealDataRetrainer(
    'fault_classifier_keras.h5',      # Original model
    'model_info.json'                  # Original metadata
)

X_train, X_test, y_train, y_test, scaler = retrainer.load_real_data_csv(
    'real_faults_week1.csv'            # CSV from your field deployment
)

print("\nStep 2: Retrain with real data (fine-tuning)...")
model, history = retrainer.retrain(
    X_train, X_test, y_train, y_test,
    epochs=30,
    batch_size=32
)

print("\nStep 3: Plot training history...")
retrainer.plot_training_history(history)  # Saves retraining_history.png

print("\nStep 4: Save improved model...")
tflite_path, info_path = retrainer.save_retrained_model(model, scaler, version='v2')

print("\n✓ Retrained model ready:")
print(f"  - {tflite_path} (new quantized model)")
print(f"  - {info_path} (updated scaling parameters)")
print("\nNext: Deploy to hardware by running:")
print(f"  xxd -i {tflite_path} > model_data_v2.h")


# ============================================================================
# EXAMPLE 6: Compare Model Versions
# ============================================================================
"""
Compare synthetic-trained vs real-data-trained models
"""

from validate_and_retrain import compare_models
from tinyml_fault_classifier import engineer_features

print("Comparing model versions...")
compare_models(
    'fault_classifier.tflite',      # v1: synthetic data trained
    'fault_classifier_v2.tflite',   # v2: fine-tuned on real data
    'model_info.json',
    'model_info_v2.json',
    'real_faults_test_set.csv',     # Real data you're validating on
    engineer_features
)

print("\nGenerated: model_comparison.png (confusion matrices)")
print("Check if accuracy improved!")


# ============================================================================
# EXAMPLE 7: Simulating Faults for Testing (Before Hardware)
# ============================================================================
"""
Generate test CSV to validate model on simulated faults
"""

import numpy as np
import pandas as pd
from tinyml_fault_classifier import MotorFaultDataGenerator

print("Generating test data with all fault types...")

generator = MotorFaultDataGenerator(seed=999)

# Generate each fault type
datasets = [
    (*generator.healthy_operation(n_samples=100), 'healthy'),
    (*generator.overcurrent_fault(n_samples=50), 'overcurrent'),
    (*generator.overvoltage_fault(n_samples=30), 'overvoltage'),
    (*generator.undervoltage_fault(n_samples=30), 'undervoltage'),
    (*generator.overtemp_fault(n_samples=40), 'overtemp'),
    (*generator.vfo_fault(n_samples=20), 'vfo'),
    (*generator.resistance_degrade_fault(n_samples=25), 'resistance'),
]

# Combine into one dataset
X_all = []
y_all = []
labels_all = []

for X, y, label_name in datasets:
    X_all.append(X)
    y_all.append(y)
    labels_all.extend([label_name] * len(y))

X_all = np.vstack(X_all)
y_all = np.hstack(y_all)

# Create DataFrame for CSV export
df = pd.DataFrame(X_all, columns=['Ia', 'Ib', 'Ic', 'Vdc', 'Temp', 'VFO_freq', 'R_winding'])
df['label'] = y_all
df['label_name'] = labels_all

df.to_csv('test_faults.csv', index=False)
print(f"Saved {len(df)} test samples to test_faults.csv")

# Validate model on this test data
print("\nNow validate your model:")
print("  python validate_and_retrain.py validate \\")
print("    --model fault_classifier.tflite \\")
print("    --model_info model_info.json \\")
print("    --data test_faults.csv")


# ============================================================================
# EXAMPLE 8: Profile Inference Latency on Hardware
# ============================================================================
"""
C code to measure actual inference time on your microcontroller
"""

#include "fault_classifier.h"
#include <stdint.h>

// Timer function (implement for your microcontroller)
// Example for STM32:
static inline uint32_t get_microseconds(void) {
    return DWT->CYCCNT / 80;  // For 80 MHz clock
}

void measure_inference_performance(void) {
    printf("\n=== INFERENCE PERFORMANCE MEASUREMENT ===\n");
    
    uint32_t inference_times[10];
    uint32_t total_time = 0;
    uint32_t max_time = 0;
    
    // Simulate 10 inferences
    for (int i = 0; i < 10; i++) {
        uint32_t start = get_microseconds();
        
        // Run inference with dummy data
        sensor_data.Ia = 10.5f;
        sensor_data.Ib = 10.2f;
        sensor_data.Ic = 10.1f;
        sensor_data.Vdc = 48.3f;
        sensor_data.Temp = 55.2f;
        sensor_data.VFO_freq = 16050;
        sensor_data.R_winding = 1.02f;
        
        run_inference();
        
        uint32_t elapsed = get_microseconds() - start;
        inference_times[i] = elapsed;
        total_time += elapsed;
        max_time = (elapsed > max_time) ? elapsed : max_time;
    }
    
    uint32_t avg_time = total_time / 10;
    
    printf("Average latency:  %lu µs\n", avg_time);
    printf("Max latency:      %lu µs\n", max_time);
    printf("Budget (62.5µs):  %s\n", max_time <= 62500 ? "✓ PASS" : "✗ FAIL");
    printf("CPU usage:        %.1f%%\n", (float)avg_time / 62.5f * 100.0f);
}


# ============================================================================
# EXAMPLE 9: Hybrid Fault Detection (ML + Rule-Based)
# ============================================================================
"""
Combining ML predictions with rule-based thresholds for robustness
"""

#include "fault_classifier.h"

struct {
    uint8_t rule_based_fault;
    uint8_t ml_prediction;
    uint8_t final_decision;
    uint8_t confidence;
} hybrid_state;

uint8_t hybrid_fault_detection(
    float Ia, float Ib, float Ic,
    float Vdc, float Temp
) {
    // 1. Rule-based checks (fast, always run)
    hybrid_state.rule_based_fault = 0;
    
    if (Ia > 18.0f || Ib > 18.0f || Ic > 18.0f) {
        hybrid_state.rule_based_fault = 1;  // OVERCURRENT
    } else if (Vdc > 55.0f) {
        hybrid_state.rule_based_fault = 2;  // OVERVOLTAGE
    } else if (Vdc < 40.0f) {
        hybrid_state.rule_based_fault = 3;  // UNDERVOLTAGE
    } else if (Temp > 100.0f) {
        hybrid_state.rule_based_fault = 4;  // OVERTEMP
    }
    
    // 2. ML prediction (slow, periodic)
    hybrid_state.ml_prediction = get_fault_prediction();
    hybrid_state.confidence = get_prediction_confidence();
    
    // 3. Voting logic
    if (hybrid_state.rule_based_fault != 0) {
        if (hybrid_state.ml_prediction == hybrid_state.rule_based_fault) {
            // Agreement: high confidence
            hybrid_state.final_decision = hybrid_state.rule_based_fault;
        } else if (hybrid_state.ml_prediction != 0) {
            // Disagreement: log and investigate
            log_diagnostic_conflict(
                hybrid_state.rule_based_fault,
                hybrid_state.ml_prediction
            );
            // Trust rule-based (faster, more reliable for hard thresholds)
            hybrid_state.final_decision = hybrid_state.rule_based_fault;
        } else {
            // Only rules triggered
            hybrid_state.final_decision = hybrid_state.rule_based_fault;
        }
    } else if (hybrid_state.ml_prediction != 0 && hybrid_state.confidence >= 80) {
        // ML detected something rule-based missed
        hybrid_state.final_decision = hybrid_state.ml_prediction;
    } else {
        // No fault
        hybrid_state.final_decision = 0;
    }
    
    return hybrid_state.final_decision;
}


# ============================================================================
# EXAMPLE 10: CSV Data Format for Training & Validation
# ============================================================================
"""
Your real-world fault data should match this format:
"""

# test_data.csv format:
# Ia,Ib,Ic,Vdc,Temp,VFO_freq,R_winding,label
# 9.8,10.1,9.9,48.2,50.5,16000,1.0,0
# 10.5,10.3,10.1,48.1,51.2,16020,1.01,0
# 12.2,13.5,11.8,48.0,52.1,16000,1.02,0
# 18.5,19.2,17.8,48.3,55.2,15950,1.03,0
# 22.1,23.5,21.3,48.5,62.1,16050,1.05,1
# ...
# 
# Label meanings:
#   0 = NO_FAULT
#   1 = OVERCURRENT
#   2 = OVERVOLTAGE
#   3 = UNDERVOLTAGE
#   4 = OVERTEMP
#   5 = VFO_FAULT
#   6 = RESISTANCE_DEGRADE

# Example Python code to create this from your embedded logs:
import pandas as pd

# Read raw logs from your system
logs = pd.read_csv('system_logs.csv')  # Your original data

# Manually verify ~50-100 samples and assign ground truth labels
# (This is the data collection / labeling phase)
ground_truth = pd.DataFrame({
    'timestamp': logs['timestamp'],
    'Ia': logs['current_a'],
    'Ib': logs['current_b'],
    'Ic': logs['current_c'],
    'Vdc': logs['bus_voltage'],
    'Temp': logs['ipm_temp'],
    'VFO_freq': logs['vfo_frequency'],
    'R_winding': logs['winding_resistance_estimate'],
    'label': logs['fault_annotation'],  # Human-assigned (0-6)
})

ground_truth.to_csv('real_fault_data_verified.csv', index=False)

print(f"Verified {len(ground_truth)} samples")
print(f"Class distribution: {ground_truth['label'].value_counts()}")
