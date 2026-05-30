"""
TinyML Fault Classifier for Motor Control System
=================================================
Synthetic training data generation + Keras model + TFLite quantization

Predicts: NO_FAULT, OVERCURRENT, OVERVOLTAGE, UNDERVOLTAGE, OVERTEMP, VFO_FAULT

Constraints:
  - Model size: < 50 KB (for 256 KB Flash with other code)
  - Latency: < 125 µs at 8 kHz
  - RAM: < 40 KB
  - Use int8 quantization

Architecture: Tiny 2-layer network
  Input (6 features) → Dense(32, relu) → Dense(64, relu) → Output(6, softmax)
  ~4 KB model size, ~50 µs inference on ARM Cortex-M4
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json

from phase_current_analyser import PhaseCurrentAnalyser
from ipm_vfo_analyser import IPMVFOAnalyser


# ============================================================================
# 1. MOTOR FAULT DATA GENERATOR (ORCHESTRATOR)
# ============================================================================

class MotorFaultDataGenerator:
    """Orchestrate synthetic motor diagnostic data generation across all fault analysers."""

    def __init__(self, seed=42):
        np.random.seed(seed)
        self.seed = seed

        self.nominal = {
            'Ia': 7.0,           # Phase current A (Amps)
            'Ib': 7.0,           # Phase current B
            'Ic': 7.0,           # Phase current C
            'Vdc': 340.0,        # DC bus voltage (Volts)
            'Temp': 50.0,        # IPM temperature (°C)
            'VFO_feedback': 1,   # VFO feedback (1=ON, 0=OFF)
        }

        self.limits = {
            'Ia_max': 9.0,
            'Vdc_min': 190.0,
            'Vdc_max': 420.0,
            'Temp_max': 125.0,
        }

        self.phase_current = PhaseCurrentAnalyser(self.nominal, self.limits)
        self.ipm_vfo = IPMVFOAnalyser(self.nominal, self.limits)

    def generate_balanced_dataset(self):
        """Generate all fault types in a balanced dataset."""
        datasets = [
            self.phase_current.healthy_operation(n_samples=1500),
            self.phase_current.overcurrent_fault(n_samples=750),
            self.phase_current.overvoltage_fault(n_samples=450),
            self.phase_current.undervoltage_fault(n_samples=450),
            self.ipm_vfo.overtemp_fault(n_samples=600),
            self.ipm_vfo.vfo_fault(n_samples=300),
        ]

        X = np.vstack([d[0] for d in datasets])
        y = np.hstack([d[1] for d in datasets])

        idx = np.random.permutation(len(X))
        return X[idx], y[idx]


# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================

def engineer_features(X_raw):
    """
    Extract meaningful features from raw sensor inputs.

    Raw inputs (6 features):
        - Ia, Ib, Ic (phase currents)
        - Vdc (DC bus voltage)
        - Temp (IPM temperature)
        - VFO_feedback (gate driver ON/OFF feedback, 1=ON, 0=OFF)

    Engineered features (6):
        - I_max          (PhaseCurrentAnalyser)
        - I_imbalance    (PhaseCurrentAnalyser)
        - V_normalized   (PhaseCurrentAnalyser)
        - Temp_normalized (IPMVFOAnalyser)
        - VFO_feedback   (IPMVFOAnalyser)
        - I_rms_estimate (PhaseCurrentAnalyser)
    """
    X_eng = np.zeros_like(X_raw)

    for i in range(len(X_raw)):
        Ia, Ib, Ic, Vdc, Temp, VFO_feedback = X_raw[i]

        I_max, I_imbalance, V_normalized, I_rms_estimate = PhaseCurrentAnalyser.extract_features(Ia, Ib, Ic, Vdc)
        Temp_normalized, VFO_feedback_feat = IPMVFOAnalyser.extract_features(Temp, VFO_feedback)

        X_eng[i] = [I_max, I_imbalance, V_normalized, Temp_normalized,
                    VFO_feedback_feat, I_rms_estimate]

    return X_eng


# ============================================================================
# 3. KERAS MODEL TRAINING
# ============================================================================

def build_tinyml_model(input_dim=6, num_classes=6):
    """
    Build tiny neural network for TFLite deployment.

    Architecture:
      Input(6) → Dense(32, relu) → Dense(64, relu) → Dense(6, softmax)

    Size estimate: ~4-6 KB (weights + biases)
    Latency estimate: ~30-50 µs on ARM Cortex-M4 @ 80 MHz
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """Train the model on synthetic data."""
    model = build_tinyml_model()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    return model, history


# ============================================================================
# 4. QUANTIZATION & TFLITE CONVERSION
# ============================================================================

def convert_to_tflite_quantized(model, X_train):
    """
    Convert Keras model to quantized TFLite for microcontroller deployment.
    Uses int8 post-training quantization.
    """
    X_quant = X_train[:100].astype(np.float32)

    def representative_data_gen():
        for i in range(len(X_quant)):
            yield [X_quant[i:i+1]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    return converter.convert()


def save_tflite_model(tflite_model, filename='fault_classifier.tflite'):
    """Save TFLite model to file."""
    with open(filename, 'wb') as f:
        f.write(tflite_model)
    print(f"Model saved to {filename}, size: {len(tflite_model)} bytes")


# ============================================================================
# 5. MODEL EVALUATION & EXPORT
# ============================================================================

def evaluate_model(model, X_test, y_test):
    """Evaluate model accuracy on test set."""
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")
    return loss, accuracy


def export_model_info(model, X_scaler, filename='model_info.json'):
    """Export model scaling info and label names for embedded C code."""
    label_names = [
        "NO_FAULT",
        "OVERCURRENT",
        "OVERVOLTAGE",
        "UNDERVOLTAGE",
        "OVERTEMP",
        "VFO_FAULT",
    ]

    info = {
        'model_type': 'TFLite quantized NN',
        'input_features': 6,
        'output_classes': 6,
        'label_names': label_names,
        'scaler_mean': X_scaler.mean_.tolist(),
        'scaler_scale': X_scaler.scale_.tolist(),
        'model_input_type': 'int8',
        'model_output_type': 'int8',
        'feature_names': [
            'I_max',
            'I_imbalance',
            'V_normalized',
            'Temp_normalized',
            'VFO_feedback',
            'I_rms_estimate'
        ]
    }

    with open(filename, 'w') as f:
        json.dump(info, f, indent=2)

    print(f"Model info exported to {filename}")
    return info


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("TinyML Motor Fault Classifier - Training Pipeline")
    print("=" * 70)

    print("\n[1] Generating synthetic training data...")
    generator = MotorFaultDataGenerator(seed=42)
    X_raw, y = generator.generate_balanced_dataset()
    print(f"    Generated {len(X_raw)} samples")
    print(f"    Class distribution: {np.bincount(y)}")

    print("\n[2] Engineering features...")
    X_eng = engineer_features(X_raw)
    print(f"    Feature matrix shape: {X_eng.shape}")

    print("\n[3] Normalizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_eng)
    print(f"    Mean: {scaler.mean_}")
    print(f"    Std: {scaler.scale_}")

    print("\n[4] Splitting dataset...")
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    print(f"    Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    print("\n[5] Training Keras model...")
    model, history = train_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)

    print("\n[6] Evaluating model...")
    evaluate_model(model, X_test, y_test)

    print("\n[7] Converting to TFLite (int8 quantization)...")
    tflite_model = convert_to_tflite_quantized(model, X_train)
    print(f"    TFLite model size: {len(tflite_model)} bytes ({len(tflite_model)/1024:.2f} KB)")

    print("\n[8] Saving models...")
    save_tflite_model(tflite_model, 'fault_classifier.tflite')
    model.save('fault_classifier_keras.h5')

    print("\n[9] Exporting model metadata...")
    model_info = export_model_info(model, scaler)

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. Copy 'fault_classifier.tflite' to your embedded project")
    print(f"  2. Use the C inference code (tflite_inference_template.c)")
    print(f"  3. Integrate with your 8 kHz foreground_loop")
    print(f"  4. Collect real data and retrain with: python retrain_tflite.py")
