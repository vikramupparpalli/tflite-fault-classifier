# TinyML, TensorFlow Lite, Keras, and scikit-learn in This Project

This repository is a full tinyML pipeline for motor fault detection. The Python side generates synthetic fault data, extracts features, trains a small Keras model, and exports a quantized TensorFlow Lite model. The embedded side loads that model, reproduces the same feature math, and runs inference inside an existing real-time control system.

This tutorial is written to help you understand each module and function in the repository, then integrate the result into an embedded target with enough ROM and RAM. The current embedded implementation in [TfLite_FaultClassifierModel.c](TfLite_FaultClassifierModel.c) uses a 16 KB tensor arena and a compact 6-feature, 6-class int8 model. That is a good fit for an MCU-class device such as RA6T3, provided you leave headroom for the TensorFlow Lite Micro library, your application code, stack, and RTOS or bare-metal runtime.

## What tinyML Means Here

tinyML means building machine learning systems that can run on very constrained hardware. In practice that means:

- Small model size
- Low RAM usage for inference
- Deterministic runtime behavior
- A preprocessing pipeline that can be reproduced exactly on the device

For this project, the important rule is not just “make the model small.” The more important rule is “make the whole pipeline small and reproducible,” including feature engineering, scaling, quantization, and inference.

## What This Project Detects

The classifier predicts six states:

- NO_FAULT
- OVERCURRENT
- OVERVOLTAGE
- UNDERVOLTAGE
- OVERTEMP
- VFO_FAULT

The model is trained from synthetic data, but the architecture and feature flow are intended for a real embedded control loop.

## Repository Map

The main Python training pipeline is [tinyml_fault_classifier.py](tinyml_fault_classifier.py).

The synthetic data generation and feature extraction logic is split across:

- [phase_current_analyser.py](phase_current_analyser.py)
- [ipm_vfo_analyser.py](ipm_vfo_analyser.py)

The retraining and validation path is [validate_and_retrain.py](validate_and_retrain.py).

The embedded-facing C interface is split across:

- [I_TfliteModel.h](I_TfliteModel.h)
- [TfLite_FaultClassifierModel.h](TfLite_FaultClassifierModel.h)
- [TfLite_FaultClassifierModel.c](TfLite_FaultClassifierModel.c)

The exported model metadata is in [model_info.json](model_info.json).

## End-to-End Data Flow

The full flow is:

1. Generate synthetic raw sensor data.
2. Convert raw values into engineered features.
3. Normalize those features using a standard scaler.
4. Train a compact Keras network.
5. Convert the trained network into quantized TFLite.
6. Export the model and scaler metadata.
7. Embed the model in firmware.
8. Recreate the same feature math on the MCU.
9. Run inference inside the control loop.
10. Apply a fault action in the existing embedded system.

## Python Side: Module by Module

### [tinyml_fault_classifier.py](tinyml_fault_classifier.py)

This is the orchestrator. It owns the full training pipeline and ties the analyzers, scaler, Keras model, and TFLite export together.

#### `MotorFaultDataGenerator`

This class combines the two analyzer modules and produces a balanced dataset.

##### `__init__(self, seed=42)`

Initializes the generator with nominal sensor values and hard limits.

It defines:

- nominal phase currents
- nominal DC bus voltage
- nominal temperature
- VFO feedback state
- safety limits for current, voltage, and temperature

It also instantiates:

- `PhaseCurrentAnalyser`
- `IPMVFOAnalyser`

Why it exists:

- keeps nominal values in one place
- creates a reproducible dataset
- separates fault logic by domain

##### `generate_balanced_dataset(self)`

Builds a dataset with class balance across the fault types.

It calls:

- `PhaseCurrentAnalyser.healthy_operation()`
- `PhaseCurrentAnalyser.overcurrent_fault()`
- `PhaseCurrentAnalyser.overvoltage_fault()`
- `PhaseCurrentAnalyser.undervoltage_fault()`
- `IPMVFOAnalyser.overtemp_fault()`
- `IPMVFOAnalyser.vfo_fault()`

It then stacks the data and labels, shuffles them, and returns `X_raw, y`.

Why it matters:

- the neural network gets examples for every class
- the class distribution is under control
- training is less biased than a heavily imbalanced dataset

#### `engineer_features(X_raw)`

This function transforms raw sensor readings into the features the model actually learns from.

Raw row layout:

- `Ia`
- `Ib`
- `Ic`
- `Vdc`
- `Temp`
- `VFO_feedback`

It uses:

- `PhaseCurrentAnalyser.extract_features(Ia, Ib, Ic, Vdc)`
- `IPMVFOAnalyser.extract_features(Temp, VFO_feedback)`

Output features:

- `I_max`
- `I_imbalance`
- `V_normalized`
- `Temp_normalized`
- `VFO_feedback`
- `I_rms_estimate`

This is one of the most important functions in the project because the embedded firmware must produce the same result.

#### `build_tinyml_model(input_dim=6, num_classes=6)`

Builds the Keras model.

Current architecture:

- Input: 6 features
- Dense(32, relu)
- Dense(64, relu)
- Dense(6, softmax)

The model is intentionally small enough to quantize and run on a microcontroller.

Why this architecture works for tinyML:

- dense layers are cheap and easy to deploy
- the model is small enough for embedded inference
- the input dimension is fixed and known at compile time

#### `train_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)`

Trains the Keras model on the normalized feature vectors.

Training details:

- optimizer: Adam
- loss: sparse categorical crossentropy
- metric: accuracy

Why this function matters:

- it learns the mapping from engineered features to fault class
- it produces the floating-point model before TFLite conversion

#### `convert_to_tflite_quantized(model, X_train)`

Converts the trained Keras model into quantized TFLite.

Key details:

- post-training quantization is enabled
- a representative dataset is provided
- the supported ops are restricted to int8 TFLite builtins
- the input and output types are int8

Why this matters:

- smaller runtime footprint
- faster inference on embedded hardware
- better fit for MCU deployment than a full TensorFlow model

#### `save_tflite_model(tflite_model, filename='fault_classifier.tflite')`

Writes the binary `.tflite` file to disk.

In embedded deployment this file becomes the source for a generated C array or binary asset.

#### `evaluate_model(model, X_test, y_test)`

Evaluates the Keras model on the held-out test set.

Why it matters:

- confirms the model generalizes beyond the training data
- gives you a quick quality check before deployment

#### `export_model_info(model, X_scaler, filename='model_info.json')`

Exports metadata needed by the MCU side.

It stores:

- model type
- input feature count
- output class count
- label names
- scaler mean
- scaler scale
- feature names
- model input/output types

This metadata is essential because the embedded side must reproduce normalization exactly.

### [phase_current_analyser.py](phase_current_analyser.py)

This module handles phase current and DC bus related synthetic data.

#### `PhaseCurrentAnalyser.__init__(self, nominal, limits)`

Stores nominal values and limits for currents, voltage, and temperature.

This keeps the synthetic data generation consistent across all phase-current related faults.

#### `healthy_operation(self, n_samples=1000)`

Generates nominal operating data with noise around the baseline.

Typical output characteristics:

- currents hover near the nominal phase current
- DC bus stays near nominal voltage
- temperature varies slightly
- VFO feedback remains asserted

Purpose:

- teaches the model what normal operation looks like
- reduces false positives during inference

#### `overcurrent_fault(self, n_samples=500)`

Generates samples where the phase currents ramp upward toward the current limit.

Purpose:

- teaches the model to identify sustained current stress

#### `overvoltage_fault(self, n_samples=300)`

Generates samples where the DC bus ramps upward toward the voltage limit.

Purpose:

- teaches the model to detect supply overshoot or inverter-related voltage faults

#### `undervoltage_fault(self, n_samples=300)`

Generates samples where the DC bus sags downward toward the undervoltage limit.

Purpose:

- teaches the model to detect brownout or supply dip behavior

#### `extract_features(Ia, Ib, Ic, Vdc)`

Produces the current and voltage features used by the model:

- `I_max`: the maximum phase current
- `I_imbalance`: spread between max and min phase current
- `V_normalized`: DC bus normalized to 340 V
- `I_rms_estimate`: rough RMS estimate across the three phases

This function is central to the embedded story because the firmware must match it exactly.

### [ipm_vfo_analyser.py](ipm_vfo_analyser.py)

This module handles IPM temperature and VFO feedback faults.

#### `IPMVFOAnalyser.__init__(self, nominal, limits)`

Stores nominal thermal and feedback values.

#### `overtemp_fault(self, n_samples=400)`

Generates rising temperature samples up to the thermal limit.

Purpose:

- trains the classifier to recognize thermal runaway or overheating behavior

#### `vfo_fault(self, n_samples=200)`

Generates samples where the gate-driver feedback is lost or toggles abnormally.

Purpose:

- trains the model to detect loss of VFO readiness

#### `extract_features(Temp, VFO_feedback)`

Produces the thermal features:

- `Temp_normalized`: temperature scaled by 125°C
- `VFO_feedback`: pass-through digital state

This module is intentionally simple because thermal and feedback signals are lower-dimensional than the current signals.

### Training and Export Sequence in `tinyml_fault_classifier.py`

The `if __name__ == '__main__'` block runs the whole pipeline in order:

1. Build the synthetic dataset.
2. Engineer features.
3. Normalize with `StandardScaler`.
4. Split into train, validation, and test sets.
5. Train the Keras model.
6. Evaluate accuracy.
7. Convert to TFLite.
8. Save `.tflite` and `.h5` artifacts.
9. Export the scaler and labels to JSON.

The important thing to understand is that the JSON file is not optional. It is the bridge between the training environment and the embedded runtime.

### [validate_and_retrain.py](validate_and_retrain.py)

This script is for the lifecycle after the first deployment.

It provides:

- TFLite validation against test CSV data
- retraining using real field data
- model comparison between versions

This is how tinyML usually matures in the field: start with synthetic data, deploy, collect real behavior, then update the model.

## Embedded Side: Module by Module

### [I_TfliteModel.h](I_TfliteModel.h)

This is the generic interface for embedded inference models.

It defines a small API with the following operations:

- `SetSensorData`
- `RunInference`
- `GetClassScores`
- `GetPredictedLabel`
- `Reset`

The point of this abstraction is to make the classifier pluggable. Your application code can hold a generic `I_TfliteModel_t` and invoke the model without caring about the concrete implementation.

#### `SetSensorData(instance, input_features, num_features)`

Stores the latest raw input values inside the model instance.

In this project those inputs are the 6 raw sensor values.

#### `RunInference(instance)`

Runs feature engineering, scaling, quantization, and the TFLite interpreter.

Returns the predicted class index.

#### `GetClassScores(instance, scores, num_scores)`

Copies the raw class scores out of the model state.

Use this if you want confidence analysis or logging.

#### `GetPredictedLabel(instance)`

Returns a string label for the last predicted class.

#### `Reset(instance)`

Clears internal state and prediction history.

### [TfLite_FaultClassifierModel.h](TfLite_FaultClassifierModel.h)

This is the public API for the specific fault classifier model.

#### `TfLite_FaultClassifierModel_Init(void)`

Must be called once at startup.

What it does:

- initializes the TensorFlow Lite Micro target
- loads the model from `model_data.h`
- checks schema compatibility
- sets up the interpreter
- allocates the tensor arena
- prepares the input and output tensors
- resets the model state

Return values:

- `0` on success
- negative values on initialization failure

This function is the first embedded call you should care about.

#### `TfLite_FaultClassifierModel`

This is the singleton instance implementing the `I_TfliteModel` interface.

The design is intentionally simple for embedded use: one model instance, one interpreter, one tensor arena.

### [TfLite_FaultClassifierModel.c](TfLite_FaultClassifierModel.c)

This is the concrete embedded implementation. It is the most important file for RA6T3 integration because it shows exactly what memory is needed and exactly how inference is run.

#### Model Configuration Constants

The file defines:

- `TFLITE_FAULTCLASSIFIERMODEL_ARENA_SIZE` = 16384
- `TFLITE_FAULTCLASSIFIERMODEL_NUM_FEATURES` = 6
- `TFLITE_FAULTCLASSIFIERMODEL_NUM_CLASSES` = 6

These numbers are the core embedded budget.

What they imply for RA6T3:

- You need at least 16 KB of contiguous RAM for the tensor arena.
- You need extra RAM for stack, buffers, and your application.
- Flash must hold the TensorFlow Lite Micro library, this model code, and the generated model blob.

The exact Flash cost depends on your compiler, link-time optimization, and which TFLM ops are pulled in, so always verify with the final map file.

#### Scaler Constants

The file stores the scaler mean and scale values as compile-time constants.

These must match [model_info.json](model_info.json) and the training pipeline.

Do not treat them as optional tuning values. They are part of the trained model contract.

#### `TfLite_FaultClassifierModel_EngineerFeatures(const float *sensorData, float *features)`

This function converts the six raw sensor values into the six engineered features.

It computes:

- `I_max`
- `I_imbalance`
- `V_normalized`
- `Temp_normalized`
- `VFO_feedback`
- `I_rms_estimate`

This is the embedded mirror of the Python feature engineering logic.

If this function differs from `engineer_features()` in Python, the classifier will be fed the wrong distribution.

#### `TfLite_FaultClassifierModel_ScaleAndQuantizeFeatures(const float *features, TfLiteTensor *inputTensor)`

This function applies normalization and int8 quantization.

For each feature:

1. subtract the stored mean
2. divide by the stored scale
3. multiply by 128
4. round to the nearest integer
5. clamp to the int8 range

This is what prepares the input tensor for the quantized TFLite model.

Why this function matters:

- it is the bridge between real-world engineering units and model input space
- it directly affects inference accuracy
- it must be fast and deterministic

#### `TfLite_FaultClassifierModel_SetSensorData(I_TfliteModel_t *instance, const float *inputFeatures, uint32_t numFeatures)`

Copies raw inputs into the model instance.

Behavior:

- ignores the call if the feature count is wrong
- stores the latest sensor values for the next inference call

This function is designed to be called from your control loop or a higher-level scheduler.

#### `TfLite_FaultClassifierModel_RunInference(I_TfliteModel_t *instance)`

This is the central runtime function.

It performs these steps:

1. engineer features from the raw sensor data
2. scale and quantize the features
3. invoke the TFLite Micro interpreter
4. copy output scores into internal storage
5. find the maximum score
6. store the winning class as the predicted label

If inference fails, it returns `0` and clears the score buffer.

Why it matters:

- this is the function that actually decides the fault class
- it is the function your firmware should call periodically

#### `TfLite_FaultClassifierModel_GetClassScores(I_TfliteModel_t *instance, void *scores, uint32_t numScores)`

Copies the raw int8 scores from the output tensor into your buffer.

Use this for:

- debugging
- telemetry
- confidence tracking
- offline analysis

#### `TfLite_FaultClassifierModel_GetPredictedLabel(I_TfliteModel_t *instance)`

Returns a string representation of the latest prediction.

This is useful for logging, diagnostics, and developer debug output.

#### `TfLite_FaultClassifierModel_Reset(I_TfliteModel_t *instance)`

Clears the internal state.

This is useful at boot or when you need to reinitialize the classifier state machine.

#### `TfLite_FaultClassifierModel_Init(void)`

The initialization function does the following:

- calls `tflite::InitializeTarget()`
- loads the model with `tflite::GetModel(fault_classifier_tflite)`
- checks the schema version
- creates a resolver with the required ops
- creates the static micro interpreter
- allocates tensors
- fetches the input and output tensors
- resets the model instance

The current op resolver adds only the ops needed by this model:

- FullyConnected
- Relu
- Softmax

This is good tinyML practice because it avoids pulling unnecessary operator code into Flash.

## How the Embedded Integration Should Work

The intended usage pattern is:

1. Initialize the model once at startup.
2. Feed each new set of sensor readings into `SetSensorData`.
3. Call `RunInference` on the schedule you choose.
4. Read the predicted class or class scores.
5. Trigger control actions if a fault is detected.

The existing interface is generic enough to drop into a larger embedded application without exposing TensorFlow Lite Micro details to the rest of your code.

## RA6T3 Integration Strategy

RA6T3 is a reasonable target for this design if your system has enough headroom for the TFLite Micro runtime and your application logic. The model itself is small, but the library and runtime overhead are what you must budget carefully.

### RAM Budgeting

The most obvious RAM consumer is the tensor arena:

- tensor arena: 16 KB

You should also reserve RAM for:

- stack
- control-loop state
- ADC and sensor buffers
- logging buffers
- RTOS objects if you use an RTOS

Practical advice:

- keep the arena in static memory
- avoid heap allocation in the inference path
- measure total runtime memory in the final firmware image

If you are tight on RAM, the first things to review are the arena size, logging buffers, and any large local arrays in your control code.

### ROM / Flash Budgeting

The Flash footprint includes:

- your application code
- TensorFlow Lite Micro library code
- the generated model blob in `model_data.h`
- the fault classifier wrapper code

The precise footprint depends on compiler options and linked ops, so the correct workflow is:

1. build with the target compiler
2. inspect the map file
3. reduce unused code paths
4. keep only the ops the model uses

The current model only needs fully connected layers, ReLU, and softmax, so the resolver stays small.

### Scheduling on a Real-Time System

Do not run model inference in a way that can break control-loop timing.

Good patterns:

- run the model in a low-priority periodic task
- update the sensor snapshot in the fast loop
- invoke inference every N cycles
- keep the fast loop deterministic

For an existing embedded system, the simplest approach is often:

- sample sensors in the fast loop
- store the latest values in a shared structure
- trigger classifier inference from a slower task or a throttled periodic callback

### Data Ownership and Thread Safety

Because the current implementation stores raw sensor data inside a singleton instance, you should be careful about concurrent access.

Recommendations:

- write sensor data from one context only
- read predictions from one context only
- protect shared access with a lock or critical section if multiple contexts are involved

If the firmware is bare metal and single-threaded, the design is simpler. If you are on an RTOS, protect the classifier interface like any other shared resource.

## Suggested Integration Pattern for an Existing Embedded System

### Step 1: Add the model files

Add these files to your firmware project:

- [I_TfliteModel.h](I_TfliteModel.h)
- [TfLite_FaultClassifierModel.h](TfLite_FaultClassifierModel.h)
- [TfLite_FaultClassifierModel.c](TfLite_FaultClassifierModel.c)
- generated `model_data.h`

### Step 2: Add TensorFlow Lite Micro

Integrate the TensorFlow Lite Micro source into your build system.

Keep only the required ops for this model:

- fully connected
- ReLU
- softmax

This reduces code size and keeps the Flash budget under control.

### Step 3: Ensure the model blob is available

The generated `model_data.h` should expose the TFLite flatbuffer as a C array.

The current implementation expects the symbol `fault_classifier_tflite`.

### Step 4: Initialize at startup

Call `TfLite_FaultClassifierModel_Init()` during system initialization.

If it returns an error, do not continue as if the model is available.

### Step 5: Feed sensor data

Pass the latest six raw sensor values into `SetSensorData`.

Input order expected by the current model:

1. Ia
2. Ib
3. Ic
4. Vdc
5. Temp
6. VFO_feedback

### Step 6: Run inference periodically

Call `RunInference` at a rate that fits your system timing.

You do not need to infer every control interrupt if the fault mode changes slowly. In many embedded systems, periodic inference every 10 ms, 50 ms, or 100 ms is enough, depending on the response requirement.

### Step 7: Use the result

Read:

- `GetPredictedLabel` for string output
- `GetClassScores` for scores
- the class index returned by `RunInference` for fast logic

Then map the prediction to your system action:

- no fault: continue normal operation
- warning: log or reduce power
- fault: stop PWM, latch fault, or transition to a safe state

## Memory Budget Guidance for RA6T3

The safest way to think about this design is in layers.

### Layer 1: Model runtime RAM

- 16 KB tensor arena

### Layer 2: Firmware RAM

- control state
- sensor samples
- queue or log buffers
- RTOS objects if used

### Layer 3: Application headroom

- stack margin
- interrupt stack margin
- debug output buffers

If you are validating fit on RA6T3, the key question is not “does the model fit?” It is “does the whole application fit with headroom under worst-case runtime conditions?”

### Layer 4: ROM / Flash

- TensorFlow Lite Micro code
- model wrapper code
- generated model blob
- your application code

## What to Verify on Hardware

Before deployment, verify:

- model initialization succeeds at boot
- inference is deterministic over repeated inputs
- scaling matches the Python metadata exactly
- the arena size is stable and not overflowing
- the fast loop timing remains within limits
- no memory corruption occurs when the classifier runs repeatedly

## Recommended Validation Checklist

- Train the model from the current Python pipeline.
- Export `fault_classifier.tflite` and `model_info.json`.
- Generate `model_data.h`.
- Port the C wrapper into the firmware project.
- Confirm the model initializes successfully on target.
- Feed known good and known bad inputs.
- Compare the embedded result to the Python result.
- Measure Flash and RAM usage from the final build.

## How the Pieces Relate

- scikit-learn standardizes the feature space.
- Keras learns the classifier.
- TensorFlow Lite compresses the model and makes it deployable.
- The analyzer modules define the synthetic training data and feature math.
- The C wrapper reproduces the same preprocessing and executes the quantized model.
- The embedded application decides how to react to the fault class.

## Reading Order for Learning the Project

If your goal is to understand the entire stack, read in this order:

1. [README.md](README.md)
2. [tinyml_fault_classifier.py](tinyml_fault_classifier.py)
3. [phase_current_analyser.py](phase_current_analyser.py)
4. [ipm_vfo_analyser.py](ipm_vfo_analyser.py)
5. [model_info.json](model_info.json)
6. [I_TfliteModel.h](I_TfliteModel.h)
7. [TfLite_FaultClassifierModel.h](TfLite_FaultClassifierModel.h)
8. [TfLite_FaultClassifierModel.c](TfLite_FaultClassifierModel.c)
9. [validate_and_retrain.py](validate_and_retrain.py)

## Short Summary

Keras trains the small classifier, scikit-learn prepares and normalizes the data, TensorFlow Lite compresses it for deployment, and the embedded C wrapper makes the model usable inside a real-time MCU application such as an RA6T3-based system.
