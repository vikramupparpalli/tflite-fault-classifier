
# TinyML Motor Fault Classifier

## Overview
This project provides a modular, production-ready TinyML fault classification system for real-time motor control diagnostics. It generates synthetic training data, trains a compact neural network, quantizes it for microcontroller deployment, and exports all necessary artifacts for embedded integration.

**Key Features:**
- Modular analyzers for each fault type (see below)
- 6-class fault detection: HEALTHY, OVERCURRENT, OVERVOLTAGE, UNDERVOLTAGE, OVERTEMP, VFO_FAULT
- Real-time inference: 40–60 µs latency (fits 16 kHz interrupt loop)
- Tiny model: ~7 KB (fits in 256 KB Flash)
- Synthetic data generation—no pre-collected data required
- Retraining and validation utilities
- Hybrid ML + rule-based operation for robustness

## Modular Analyzer Architecture

The codebase is organized into analyzer modules, each responsible for generating synthetic data and extracting features for a specific fault domain:

- **phase_current_analyser.py**: Phase current, overcurrent, overvoltage, and undervoltage faults
- **ipm_vfo_analyser.py**: IPM temperature and VFO feedback faults

The main script (`tinyml_fault_classifier.py`) orchestrates these analyzers to generate a balanced dataset and delegates feature extraction to their static methods. This modular approach makes it easy to extend or customize fault logic.

## How to Build and Run

### 1. Install Dependencies

It is recommended to use a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install tensorflow numpy scikit-learn
# For best results, use:
pip install tensorflow==2.13.0 numpy==1.24.0 scikit-learn==1.3.0
```

### 2. Train the Model

Run the main training script:

```bash
python tinyml_fault_classifier.py
```

**Outputs:**
- `fault_classifier.tflite` — Quantized TFLite model for microcontroller
- `model_info.json` — Feature scaling parameters and metadata
- `fault_classifier_keras.h5` — Keras model for future retraining

### 3. Validate or Retrain

To validate the quantized model or retrain with real data:

```bash
python validate_and_retrain.py
```

## What Does the Script Do?
- Generates 4,000+ synthetic samples for all fault modes
- Engineers features from raw sensor data (see `engineer_features`)
- Trains a tiny 2-layer neural network (Keras)
- Quantizes and exports the model for embedded use
- Prints accuracy and saves all artifacts

## Documentation & Further Reading

- [TINYML_QUICK_REFERENCE.md](TINYML_QUICK_REFERENCE.md): Quick start, workflow, and integration steps
- [tinyml_integration_guide.md](tinyml_integration_guide.md): Full integration guide for embedded deployment
- [TINYML_EXAMPLES.md](TINYML_EXAMPLES.md): Copy-paste code examples
- [motor_diagnostics_framework.md](motor_diagnostics_framework.md): Original diagnostic requirements and logic
- [INDEX.md](INDEX.md): Project index and file descriptions

For C integration, see `tflite_inference_template.c` and the integration guide.
