# Integration Diagram

Below is a high-level diagram showing how the TFLite Fault Classifier Module can be integrated into a typical motor control application:

```mermaid
flowchart TD
	subgraph MCU [Microcontroller]
		A[Sensor Inputs\n(Ia, Ib, Ic, Vdc, Temp, VFO_feedback)] -->|Sampled at 8kHz| B[Feature Engineering\n(C code)]
		B --> C[TfLite Fault Classifier Model\n(TfLite_FaultClassifierModel)]
		C --> D{Fault Class Index\n(0=NO_FAULT, 1=OVERCURRENT, ...)}
		D -->|NO_FAULT| E[Normal Motor Control]
		D -->|FAULT| F[Fault Handling\n(Shutdown, Logging, etc)]
	end

	subgraph App [Application Layer]
		G[Initialization\n(TfLite_FaultClassifierModel_Init)]
		H[SetSensorData + RunInference\n(every control loop)]
		I[GetPredictedLabel / ClassScores]
	end

	G --> H
	H --> I
	H -.-> B
	I -.-> F
```

**Description:**
- Sensor data is sampled in the control loop (e.g., 8 kHz interrupt).
- Features are engineered in C to match the model's expectations.
- The feature vector is passed to the classifier module using the generic interface.
- The predicted fault class index is used to decide between normal operation and fault handling.
- The application can log, display, or act on the predicted label or class scores as needed.

# Fault Classifier Module Design

## Overview

This module implements a TinyML-based fault classifier for embedded motor control systems. It is designed to run on microcontrollers and uses a quantized TensorFlow Lite (TFLite) model to detect various motor faults in real time. The implementation is modular and generic, allowing the same interface to be used for any TinyML model that follows the same input/output conventions.

## C Module Description

The core implementation is provided in:

- **TfLite_FaultClassifierModel.c** — Implements the model logic and inference routines.
- **TfLite_FaultClassifierModel.h** — Public API for application integration.
- **I_TfliteModel.h** — Generic interface for any TFLite-based TinyML model.

### Key Features
- Runs a quantized TFLite model for fault classification.
- Feature engineering, normalization, and quantization are performed in C to match the Python training pipeline.
- Designed for real-time, low-latency inference in interrupt or control loop contexts.
- Fault class output is a simple integer index, with a string label available for diagnostics/logging.

## Generic Interface: `I_TfliteModel`

The interface defined in `I_TfliteModel.h` allows any TinyML model to be integrated in a uniform way. The API includes:

- `SetSensorData(instance, input_features, num_features)` — Provide new input data (features) to the model.
- `RunInference(instance)` — Run inference and return the predicted class index.
- `GetClassScores(instance, scores, num_scores)` — Retrieve raw output scores from the model.
- `GetPredictedLabel(instance)` — Get a human-readable string for the predicted class.
- `Reset(instance)` — Reset the model state/statistics.

This abstraction allows the application to interact with any TFLite-based model (fault classifier, anomaly detector, etc.) using the same function signatures.

## Fault Classifier Model API

The fault classifier module provides a singleton instance implementing the generic interface:

```c
#include "TfLite_FaultClassifierModel.h"

// Initialization (call once at startup)
TfLite_FaultClassifierModel_Init();

// Usage in control loop or interrupt:
float features[6] = {Ia, Ib, Ic, Vdc, Temp, VFO_feedback};
TfliteModel_SetSensorData(&TfLite_FaultClassifierModel, features, 6);
uint8_t fault_class = TfliteModel_RunInference(&TfLite_FaultClassifierModel);
const char* fault_label = TfliteModel_GetPredictedLabel(&TfLite_FaultClassifierModel);
```

### Fault Class Mapping

The output class index maps to fault types as follows:

| Index | Label         |
|-------|--------------|
| 0     | NO_FAULT     |
| 1     | OVERCURRENT  |
| 2     | OVERVOLTAGE  |
| 3     | UNDERVOLTAGE |
| 4     | OVERTEMP     |
| 5     | VFO_FAULT    |

## Extending for Other TinyML Models

Any TFLite model that accepts a feature vector and produces a class or score output can implement the `I_TfliteModel` interface. This enables:

- Plug-and-play replacement of models (e.g., swap fault classifier for anomaly detector)
- Uniform integration in application code
- Easier testing and benchmarking of different ML models

## Summary

This design enables robust, real-time ML inference on embedded systems with a clean, generic interface. The approach is scalable to other TinyML use cases beyond fault classification.
