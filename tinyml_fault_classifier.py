"""
TinyML Fault Classifier for Motor Control System
=================================================
Synthetic training data generation + Keras model + TFLite quantization

Predicts: NO_FAULT, OVERCURRENT, OVERVOLTAGE, UNDERVOLTAGE, OVERTEMP, VFO_FAULT, RESISTANCE_DEGRADE

Constraints:
  - Model size: < 50 KB (for 256 KB Flash with other code)
  - Latency: < 62.5 µs at 16 kHz (tight!)
  - RAM: < 40 KB
  - Use int8 quantization

Architecture: Tiny 2-layer network
  Input (7 features) → Dense(32, relu) → Dense(64, relu) → Output(7, softmax)
  ~4 KB model size, ~50 µs inference on ARM Cortex-M4
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json

# ============================================================================
# 1. SYNTHETIC DATA GENERATION
# ============================================================================

class MotorFaultDataGenerator:
    """Generate synthetic motor diagnostic data with realistic fault scenarios."""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.seed = seed
        
        # Nominal operating point (from your diagnostics framework)
        self.nominal = {
            'Ia': 7.0,           # Phase current A (Amps)
            'Ib': 7.0,           # Phase current B
            'Ic': 7.0,           # Phase current C
            'Vdc': 340.0,          # DC bus voltage (Volts)
            'Temp': 50.0,         # IPM temperature (°C)
            'VFO_feedback': 1,    # VFO feedback (1=ON, 0=OFF)
            'R_winding': 3.5,     # Normalized winding resistance (per-unit)
        }
        
        # Operating ranges
        self.limits = {
            'Ia_max': 9.0,
            'Vdc_min': 190.0,
            'Vdc_max': 420.0,
            'Temp_max': 125.0,
            'R_max_degrade': 6.0,  
        }
    
    def healthy_operation(self, n_samples=1000):
        """Normal operation: small noise around nominal."""
        data = []
        for _ in range(n_samples):
            Ia = self.nominal['Ia'] + np.random.normal(0, 0.5)  # ±0.5A noise
            Ib = self.nominal['Ib'] + np.random.normal(0, 0.5)
            Ic = self.nominal['Ic'] + np.random.normal(0, 0.5)
            
            Vdc = self.nominal['Vdc'] + np.random.normal(0, 0.8)  # ±0.8V noise
            Temp = self.nominal['Temp'] + np.random.normal(0, 2.0)  # ±2°C noise
            VFO_feedback = 1  # Gate driver ON in healthy
            R_winding = self.nominal['R_winding'] + np.random.normal(0, 0.02)
            
            # Clamp to reasonable ranges
            Ia = np.clip(Ia, 0, self.limits['Ia_max'])
            Vdc = np.clip(Vdc, self.limits['Vdc_min'], self.limits['Vdc_max'])
            Temp = np.clip(Temp, 0, self.limits['Temp_max'])
            
            data.append([Ia, Ib, Ic, Vdc, Temp, VFO_feedback, R_winding])
        return np.array(data), np.zeros(n_samples, dtype=int)  # Label 0: NO_FAULT
    
    def overcurrent_fault(self, n_samples=500):
        """Overcurrent: sustained high phase current."""
        data = []
        for _ in range(n_samples):
            # Ramp from nominal to fault
            t = np.random.uniform(0, 1)  # Progression through fault development
            Ia = self.nominal['Ia'] + t * (self.limits['Ia_max'] - self.nominal['Ia'])
            Ib = self.nominal['Ib'] + t * (self.limits['Ia_max'] - self.nominal['Ib'])
            Ic = self.nominal['Ic'] + t * (self.limits['Ia_max'] - self.nominal['Ic'])
            
            Vdc = self.nominal['Vdc'] + np.random.normal(0, 1.0)  # Normal voltage
            Temp = self.nominal['Temp'] + np.random.normal(0, 2.0)
            VFO_feedback = 1
            R_winding = self.nominal['R_winding'] + np.random.normal(0, 0.02)
            
            data.append([Ia, Ib, Ic, Vdc, Temp, VFO_feedback, R_winding])
        return np.array(data), np.ones(n_samples, dtype=int)  # Label 1: OVERCURRENT
    
    def overvoltage_fault(self, n_samples=300):
        """Overvoltage: DC bus voltage spiked."""
        data = []
        for _ in range(n_samples):
            t = np.random.uniform(0, 1)
            Ia = self.nominal['Ia'] + np.random.normal(0, 0.5)
            Ib = self.nominal['Ib'] + np.random.normal(0, 0.5)
            Ic = self.nominal['Ic'] + np.random.normal(0, 0.5)
            
            # Ramp voltage up to fault
            Vdc = self.nominal['Vdc'] + t * (self.limits['Vdc_max'] - self.nominal['Vdc'])
            Temp = self.nominal['Temp'] + np.random.normal(0, 2.0)
            VFO_feedback = 1
            R_winding = self.nominal['R_winding'] + np.random.normal(0, 0.02)
            
            data.append([Ia, Ib, Ic, Vdc, Temp, VFO_feedback, R_winding])
        return np.array(data), np.full(n_samples, 2, dtype=int)  # Label 2: OVERVOLTAGE
    
    def undervoltage_fault(self, n_samples=300):
        """Undervoltage: supply sag or loss."""
        data = []
        for _ in range(n_samples):
            t = np.random.uniform(0, 1)
            Ia = self.nominal['Ia'] + np.random.normal(0, 0.5)
            Ib = self.nominal['Ib'] + np.random.normal(0, 0.5)
            Ic = self.nominal['Ic'] + np.random.normal(0, 0.5)
            
            # Ramp voltage down to fault
            Vdc = self.nominal['Vdc'] - t * (self.nominal['Vdc'] - self.limits['Vdc_min'])
            Temp = self.nominal['Temp'] + np.random.normal(0, 2.0)
            VFO_feedback = 1
            R_winding = self.nominal['R_winding'] + np.random.normal(0, 0.02)
            
            data.append([Ia, Ib, Ic, Vdc, Temp, VFO_feedback, R_winding])
        return np.array(data), np.full(n_samples, 3, dtype=int)  # Label 3: UNDERVOLTAGE
    
    def overtemp_fault(self, n_samples=400):
        """Overtemperature: gradual or rapid thermal rise."""
        data = []
        for _ in range(n_samples):
            t = np.random.uniform(0, 1)
            Ia = self.nominal['Ia'] + np.random.normal(0, 0.5)
            Ib = self.nominal['Ib'] + np.random.normal(0, 0.5)
            Ic = self.nominal['Ic'] + np.random.normal(0, 0.5)
            
            Vdc = self.nominal['Vdc'] + np.random.normal(0, 1.0)
            # Ramp temperature up to fault
            Temp = self.nominal['Temp'] + t * (self.limits['Temp_max'] - self.nominal['Temp'])
            VFO_feedback = 1
            R_winding = self.nominal['R_winding'] + np.random.normal(0, 0.02)
            
            data.append([Ia, Ib, Ic, Vdc, Temp, VFO_feedback, R_winding])
        return np.array(data), np.full(n_samples, 4, dtype=int)  # Label 4: OVERTEMP
    
    def vfo_fault(self, n_samples=200):
        """VFO fault: Gate driver feedback OFF or toggling unexpectedly."""
        data = []
        for _ in range(n_samples):
            Ia = self.nominal['Ia'] + np.random.normal(0, 0.5)
            Ib = self.nominal['Ib'] + np.random.normal(0, 0.5)
            Ic = self.nominal['Ic'] + np.random.normal(0, 0.5)
            Vdc = self.nominal['Vdc'] + np.random.normal(0, 1.0)
            Temp = self.nominal['Temp'] + np.random.normal(0, 2.0)
            # VFO_feedback OFF (0) or toggling (simulate with random 0/1)
            VFO_feedback = 0 if np.random.rand() > 0.5 else 1
            R_winding = self.nominal['R_winding'] + np.random.normal(0, 0.02)
            data.append([Ia, Ib, Ic, Vdc, Temp, VFO_feedback, R_winding])
        return np.array(data), np.full(n_samples, 5, dtype=int)  # Label 5: VFO_FAULT
    
    def resistance_degrade_fault(self, n_samples=250):
        """Winding resistance degradation: increasing resistance over time."""
        data = []
        for _ in range(n_samples):
            t = np.random.uniform(0, 1)
            Ia = self.nominal['Ia'] + np.random.normal(0, 0.5)
            Ib = self.nominal['Ib'] + np.random.normal(0, 0.5)
            Ic = self.nominal['Ic'] + np.random.normal(0, 0.5)
            
            Vdc = self.nominal['Vdc'] + np.random.normal(0, 1.0)
            Temp = self.nominal['Temp'] + np.random.normal(0, 2.0)
            VFO_feedback = 1
            # Ramp resistance up to degradation threshold
            R_winding = self.nominal['R_winding'] + t * (self.limits['R_max_degrade'] - self.nominal['R_winding'])
            
            data.append([Ia, Ib, Ic, Vdc, Temp, VFO_feedback, R_winding])
        return np.array(data), np.full(n_samples, 6, dtype=int)  # Label 6: RESISTANCE_DEGRADE
    
    def generate_balanced_dataset(self):
        """Generate all fault types in balanced dataset."""
        datasets = [
            self.healthy_operation(n_samples=1500),
            self.overcurrent_fault(n_samples=750),
            self.overvoltage_fault(n_samples=450),
            self.undervoltage_fault(n_samples=450),
            self.overtemp_fault(n_samples=600),
            self.vfo_fault(n_samples=300),
            self.resistance_degrade_fault(n_samples=375),
        ]
        
        X = np.vstack([d[0] for d in datasets])
        y = np.hstack([d[1] for d in datasets])
        
        # Shuffle
        idx = np.random.permutation(len(X))
        return X[idx], y[idx]


# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================

def engineer_features(X_raw):
        """
        Extract meaningful features from raw sensor inputs.

        Raw inputs (7 features):
                - Ia, Ib, Ic (phase currents)
                - Vdc (DC bus voltage)
                - Temp (IPM temperature)
                - VFO_feedback (gate driver ON/OFF feedback, 1=ON, 0=OFF)
                - R_winding (estimated winding resistance)

        Engineered features (7):
            - I_max (maximum phase current)
            - I_imbalance (difference between max and min phase)
            - V_normalized (Vdc as fraction of nominal)
            - Temp_normalized (Temp as fraction of max)
            - VFO_feedback (pass-through, 1=ON, 0=OFF)
            - R_normalized (resistance increase from baseline)
            - I_rms_estimate (rough current RMS)
        """
    X_eng = np.zeros_like(X_raw)
    
    for i in range(len(X_raw)):
        Ia, Ib, Ic, Vdc, Temp, VFO_feedback, R_winding = X_raw[i]
        
        # Feature 1: Max phase current
        I_max = np.max([Ia, Ib, Ic])
        
        # Feature 2: Current imbalance (spread between phases)
        I_phases = np.array([Ia, Ib, Ic])
        I_imbalance = np.max(I_phases) - np.min(I_phases)
        
        # Feature 3: Voltage normalized (340V nominal)
        V_normalized = Vdc / 340.0
        
        # Feature 4: Temperature normalized (0-125°C range)
        Temp_normalized = Temp / 125.0
        
        # Feature 5: VFO_feedback (pass-through, 1=ON, 0=OFF)
        VFO_feedback_feat = VFO_feedback
        
        # Feature 6: Resistance normalized (increase from 1.0 per-unit baseline)
        R_normalized = R_winding - 1.0
        
        # Feature 7: Rough RMS current estimate
        I_rms_estimate = np.sqrt((Ia**2 + Ib**2 + Ic**2) / 3.0)
        
        X_eng[i] = [I_max, I_imbalance, V_normalized, Temp_normalized, 
                VFO_feedback_feat, R_normalized, I_rms_estimate]
    
    return X_eng


# ============================================================================
# 3. KERAS MODEL TRAINING
# ============================================================================

def build_tinyml_model(input_dim=7, num_classes=7):
    """
    Build tiny neural network for TFLite deployment.
    
    Architecture:
      Input(7) → Dense(32, relu) → Dense(64, relu) → Dense(7, softmax)
      
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
    # Representative dataset for quantization
    X_quant = X_train[:100].astype(np.float32)

    def representative_data_gen():
        for i in range(len(X_quant)):
            yield [X_quant[i:i+1]]

    # Convert with quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    return tflite_model


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
    """
    Export model scaling info and label names for embedded C code.
    """
    label_names = [
        "NO_FAULT",
        "OVERCURRENT",
        "OVERVOLTAGE",
        "UNDERVOLTAGE",
        "OVERTEMP",
        "VFO_FAULT",
        "RESISTANCE_DEGRADE"
    ]
    
    info = {
        'model_type': 'TFLite quantized NN',
        'input_features': 7,
        'output_classes': 7,
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
            'R_normalized',
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
    
    # 1. Generate synthetic data
    print("\n[1] Generating synthetic training data...")
    generator = MotorFaultDataGenerator(seed=42)
    X_raw, y = generator.generate_balanced_dataset()
    print(f"    Generated {len(X_raw)} samples")
    print(f"    Class distribution: {np.bincount(y)}")
    
    # 2. Feature engineering
    print("\n[2] Engineering features...")
    X_eng = engineer_features(X_raw)
    print(f"    Feature matrix shape: {X_eng.shape}")
    
    # 3. Normalize features
    print("\n[3] Normalizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_eng)
    print(f"    Mean: {scaler.mean_}")
    print(f"    Std: {scaler.scale_}")
    
    # 4. Train/val/test split
    print("\n[4] Splitting dataset...")
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    print(f"    Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # 5. Train Keras model
    print("\n[5] Training Keras model...")
    model, history = train_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)
    
    # 6. Evaluate
    print("\n[6] Evaluating model...")
    evaluate_model(model, X_test, y_test)
    
    # 7. Convert to TFLite with quantization
    print("\n[7] Converting to TFLite (int8 quantization)...")
    tflite_model = convert_to_tflite_quantized(model, X_train)
    print(f"    TFLite model size: {len(tflite_model)} bytes ({len(tflite_model)/1024:.2f} KB)")
    
    # 8. Save models
    print("\n[8] Saving models...")
    save_tflite_model(tflite_model, 'fault_classifier.tflite')
    model.save('fault_classifier_keras.h5')
    
    # 9. Export metadata for embedded code
    print("\n[9] Exporting model metadata...")
    model_info = export_model_info(model, scaler)
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. Copy 'fault_classifier.tflite' to your embedded project")
    print(f"  2. Use the C inference code (tflite_inference_template.c)")
    print(f"  3. Integrate with your 16 kHz interrupt loop")
    print(f"  4. Collect real data and retrain with: python retrain_tflite.py")
