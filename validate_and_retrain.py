pip install numpy"""
TinyML Model Validation & Retraining
=====================================

Utilities for:
1. Validating quantized TFLite model accuracy
2. Retraining with real collected fault data
3. Comparing model versions (synthetic vs real-trained)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import json
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# 1. TFLITE MODEL VALIDATION
# ============================================================================

class TFLiteValidator:
    """Validate quantized TFLite model against test data."""
    
    def __init__(self, model_path, model_info_path):
        """
        Load TFLite model and metadata.
        
        Args:
            model_path: Path to .tflite file
            model_info_path: Path to model_info.json
        """
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        with open(model_info_path, 'r') as f:
            self.info = json.load(f)
        
        self.scaler_mean = np.array(self.info['scaler_mean'])
        self.scaler_scale = np.array(self.info['scaler_scale'])
        self.labels = self.info['label_names']
        
    def predict(self, X_features):
        """
        Run inference on batch of engineered features.
        
        Args:
            X_features: (N, 7) array of engineered features
        
        Returns:
            predictions: (N,) array of class indices
            confidences: (N,) array of max softmax scores
        """
        predictions = []
        confidences = []
        
        for features in X_features:
            # Normalize
            normalized = (features - self.scaler_mean) / self.scaler_scale
            
            # Quantize to int8
            scaled = (normalized * 128).astype(np.int8)
            
            # Inference
            self.interpreter.set_tensor(self.input_details[0]['index'], 
                                       np.array([scaled], dtype=np.int8))
            self.interpreter.invoke()
            
            # Get output
            output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            
            # Dequantize (rough conversion from int8 to confidence)
            # int8 range [-128, 127] → softmax-like score [0, 255]
            output_float = (output.astype(np.float32) + 128) / 2.56  # Scale to ~0-100
            
            pred_class = np.argmax(output_float)
            confidence = np.max(output_float) / 100.0  # Normalize to 0-1
            
            predictions.append(pred_class)
            confidences.append(confidence)
        
        return np.array(predictions), np.array(confidences)
    
    def validate_on_csv(self, csv_path, feature_engineer_func):
        """
        Validate model on CSV data with ground truth labels.
        
        CSV format: Ia, Ib, Ic, Vdc, Temp, VFO_feedback, label
        
        Args:
            csv_path: Path to CSV file
            feature_engineer_func: Function to engineer features from raw inputs
        
        Returns:
            metrics: Dictionary with accuracy, confusion matrix, etc.
        """
        df = pd.read_csv(csv_path)
        
        # Extract raw features and labels
        X_raw = df[['Ia', 'Ib', 'Ic', 'Vdc', 'Temp', 'VFO_feedback']].values
        y_true = df['label'].values
        
        # Engineer features
        X_features = feature_engineer_func(X_raw)
        
        # Predict
        y_pred, confidences = self.predict(X_features)
        
        # Metrics
        accuracy = accuracy_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=self.labels)
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'report': report,
            'y_pred': y_pred,
            'y_true': y_true,
            'confidences': confidences,
            'low_confidence_mask': confidences < 0.7
        }
    
    def print_metrics(self, metrics):
        """Print validation metrics."""
        print(f"\n{'='*70}")
        print(f"TFLite Model Validation Results")
        print(f"{'='*70}\n")
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        
        print(f"\nClassification Report:")
        print(metrics['report'])
        
        low_conf_count = np.sum(metrics['low_confidence_mask'])
        print(f"\nLow confidence predictions (< 70%): {low_conf_count} / {len(metrics['y_true'])}")
        
        return metrics


# ============================================================================
# 2. RETRAINING WITH REAL DATA
# ============================================================================

class RealDataRetrainer:
    """Retrain model using real-world collected fault data."""
    
    def __init__(self, old_model_path, old_model_info_path):
        """Load original Keras model and metadata."""
        self.old_model = keras.models.load_model(old_model_path)
        
        with open(old_model_info_path, 'r') as f:
            self.old_info = json.load(f)
        
        self.old_scaler_mean = np.array(self.old_info['scaler_mean'])
        self.old_scaler_scale = np.array(self.old_info['scaler_scale'])
    
    def load_real_data_csv(self, csv_path, test_split=0.2):
        """
        Load real-world fault data from CSV.
        
        Format: Ia,Ib,Ic,Vdc,Temp,VFO_feedback,label

        Label mapping:
          0 = NO_FAULT
          1 = OVERCURRENT
          2 = OVERVOLTAGE
          3 = UNDERVOLTAGE
          4 = OVERTEMP
          5 = VFO_FAULT
        """
        df = pd.read_csv(csv_path)

        X_raw = df[['Ia', 'Ib', 'Ic', 'Vdc', 'Temp', 'VFO_feedback']].values
        y = df['label'].values
        
        # Engineer features (same as training)
        from tinyml_fault_classifier import engineer_features as eng_features
        X_features = eng_features(X_raw)
        
        # Fit new scaler on real data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_features)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_split, random_state=42, stratify=y
        )
        
        print(f"Loaded real data: {len(X_raw)} samples")
        print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
        print(f"  Class distribution: {np.bincount(y)}")
        
        return X_train, X_test, y_train, y_test, scaler
    
    def retrain(self, X_train, X_test, y_train, y_test, epochs=30, batch_size=32):
        """
        Retrain Keras model on real data.
        Uses transfer learning: start from old weights, fine-tune.
        """
        print(f"\n{'='*70}")
        print(f"Retraining with real data")
        print(f"{'='*70}\n")
        
        # Evaluate old model first
        old_loss, old_acc = self.old_model.evaluate(X_test, y_test, verbose=0)
        print(f"Old model test accuracy: {old_acc:.4f}")
        
        # Fine-tune: train with lower learning rate
        self.old_model.optimizer.learning_rate = 0.001  # Lower LR for fine-tuning
        
        history = self.old_model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Evaluate new model
        new_loss, new_acc = self.old_model.evaluate(X_test, y_test, verbose=0)
        print(f"\nNew model test accuracy: {new_acc:.4f}")
        print(f"Improvement: {(new_acc - old_acc)*100:.2f}%")
        
        return self.old_model, history
    
    def save_retrained_model(self, model, scaler, version='v2'):
        """Save retrained model, TFLite version, and metadata."""
        
        # Save Keras model
        model_path = f'fault_classifier_{version}_keras.h5'
        model.save(model_path)
        print(f"Saved Keras model: {model_path}")
        
        # Convert to TFLite with int8 quantization
        # Use real data as representative dataset
        X_sample = np.random.randn(100, 7).astype(np.float32)  # Placeholder
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        tflite_path = f'fault_classifier_{version}.tflite'
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"Saved TFLite model: {tflite_path} ({len(tflite_model)} bytes)")
        
        # Save metadata with new scaler
        info = {
            'model_type': 'TFLite quantized NN (retrained on real data)',
            'input_features': 6,
            'output_classes': 6,
            'version': version,
            'scaler_mean': scaler.mean_.tolist(),
            'scaler_scale': scaler.scale_.tolist(),
            'label_names': [
                'NO_FAULT',
                'OVERCURRENT',
                'OVERVOLTAGE',
                'UNDERVOLTAGE',
                'OVERTEMP',
                'VFO_FAULT',
            ],
            'feature_names': [
                'I_max',
                'I_imbalance',
                'V_normalized',
                'Temp_normalized',
                'VFO_feedback',
                'I_rms_estimate'
            ]
        }
        
        info_path = f'model_info_{version}.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"Saved model info: {info_path}")
        
        return tflite_path, info_path
    
    def plot_training_history(self, history):
        """Plot training curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        ax1.plot(history.history['loss'], label='Train')
        ax1.plot(history.history['val_loss'], label='Validation')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Model Loss')
        ax1.legend()
        ax1.grid()
        
        # Accuracy
        ax2.plot(history.history['accuracy'], label='Train')
        ax2.plot(history.history['val_accuracy'], label='Validation')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.legend()
        ax2.grid()
        
        plt.tight_layout()
        plt.savefig('retraining_history.png', dpi=150)
        print("Saved plot: retraining_history.png")
        plt.close()


# ============================================================================
# 3. COMPARISON TOOLS
# ============================================================================

def compare_models(model_v1_path, model_v2_path, model_info_v1, model_info_v2, 
                   test_data_csv, feature_engineer_func):
    """Compare two model versions side-by-side."""
    
    print(f"\n{'='*70}")
    print(f"Model Comparison: v1 (synthetic) vs v2 (real-trained)")
    print(f"{'='*70}\n")
    
    validator_v1 = TFLiteValidator(model_v1_path, model_info_v1)
    validator_v2 = TFLiteValidator(model_v2_path, model_info_v2)
    
    metrics_v1 = validator_v1.validate_on_csv(test_data_csv, feature_engineer_func)
    metrics_v2 = validator_v2.validate_on_csv(test_data_csv, feature_engineer_func)
    
    print(f"Model v1 Accuracy: {metrics_v1['accuracy']:.4f}")
    print(f"Model v2 Accuracy: {metrics_v2['accuracy']:.4f}")
    print(f"Improvement: {(metrics_v2['accuracy'] - metrics_v1['accuracy'])*100:.2f}%")
    
    # Confusion matrix heatmap comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.heatmap(metrics_v1['confusion_matrix'], annot=True, fmt='d', ax=ax1, cmap='Blues')
    ax1.set_title('v1 (Synthetic Data)')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    sns.heatmap(metrics_v2['confusion_matrix'], annot=True, fmt='d', ax=ax2, cmap='Blues')
    ax2.set_title('v2 (Real Data Trained)')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150)
    print("Saved plot: model_comparison.png")
    plt.close()


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Validate and retrain TinyML motor fault classifier'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate TFLite model')
    validate_parser.add_argument('--model', required=True, help='Path to .tflite model')
    validate_parser.add_argument('--model_info', required=True, help='Path to model_info.json')
    validate_parser.add_argument('--data', required=True, help='Path to test data CSV')
    
    # Retrain command
    retrain_parser = subparsers.add_parser('retrain', help='Retrain with real data')
    retrain_parser.add_argument('--old_model', required=True, help='Original Keras model (.h5)')
    retrain_parser.add_argument('--old_info', required=True, help='Original model_info.json')
    retrain_parser.add_argument('--data', required=True, help='Real-world fault data CSV')
    retrain_parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two model versions')
    compare_parser.add_argument('--model_v1', required=True, help='v1 TFLite model')
    compare_parser.add_argument('--model_v2', required=True, help='v2 TFLite model')
    compare_parser.add_argument('--info_v1', required=True, help='v1 model_info.json')
    compare_parser.add_argument('--info_v2', required=True, help='v2 model_info.json')
    compare_parser.add_argument('--data', required=True, help='Test data CSV')
    
    args = parser.parse_args()
    
    # Import feature engineering function
    from tinyml_fault_classifier import engineer_features
    
    if args.command == 'validate':
        validator = TFLiteValidator(args.model, args.model_info)
        metrics = validator.validate_on_csv(args.data, engineer_features)
        validator.print_metrics(metrics)
    
    elif args.command == 'retrain':
        retrainer = RealDataRetrainer(args.old_model, args.old_info)
        X_train, X_test, y_train, y_test, scaler = retrainer.load_real_data_csv(args.data)
        model, history = retrainer.retrain(X_train, X_test, y_train, y_test, epochs=args.epochs)
        retrainer.plot_training_history(history)
        retrainer.save_retrained_model(model, scaler, version='v2')
    
    elif args.command == 'compare':
        compare_models(args.model_v1, args.model_v2, args.info_v1, args.info_v2,
                      args.data, engineer_features)
    
    else:
        parser.print_help()
