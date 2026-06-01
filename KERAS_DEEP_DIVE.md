# Keras Deep Dive for This Project

This document explains the Keras side of the motor fault classifier in detail, using the current implementation in [tinyml_fault_classifier.py](tinyml_fault_classifier.py) and the retraining flow in [validate_and_retrain.py](validate_and_retrain.py).

It focuses on how the model is built, why specific Keras choices were made, how training behaves, how fine-tuning works, and what matters for successful TensorFlow Lite deployment on embedded targets.

## Scope of the Keras Implementation

In this repository, Keras is responsible for:

- defining the neural network architecture
- compiling the training objective and optimizer
- fitting model weights using training and validation data
- evaluating test accuracy before export
- serializing a Keras model artifact for future retraining

Keras is not responsible for:

- raw data generation (handled by analyzer modules)
- feature engineering logic (handled in Python functions before model input)
- embedded runtime inference (handled by TensorFlow Lite Micro C code)

## Where to Read the Keras Code

Primary files:

- [tinyml_fault_classifier.py](tinyml_fault_classifier.py)
- [validate_and_retrain.py](validate_and_retrain.py)

Most important functions and methods:

- `build_tinyml_model`
- `train_model`
- `evaluate_model`
- `RealDataRetrainer.__init__`
- `RealDataRetrainer.retrain`
- `RealDataRetrainer.save_retrained_model`

## Data Contract Before Keras Sees Inputs

The Keras model receives feature vectors, not raw ADC values.

Input dimensionality is fixed at 6 features per sample:

- I_max
- I_imbalance
- V_normalized
- Temp_normalized
- VFO_feedback
- I_rms_estimate

Data shape entering Keras training is:

- `X_train`: `(N, 6)` float values after scaling
- `y_train`: `(N,)` integer class indices from 0 to 5

This class-index representation is the reason the model uses sparse categorical crossentropy.

## Model Architecture in Detail

The model is built by `build_tinyml_model(input_dim=6, num_classes=6)`.

Architecture:

1. Input layer with shape `(6,)`
2. Dense layer: 32 units, ReLU activation
3. Dense layer: 64 units, ReLU activation
4. Output layer: 6 units, softmax activation

### Parameter Count Breakdown

Total trainable parameters can be computed exactly:

1. Dense(6 -> 32): `6 * 32 + 32 = 224`
2. Dense(32 -> 64): `32 * 64 + 64 = 2112`
3. Dense(64 -> 6): `64 * 6 + 6 = 390`

Total:

- `224 + 2112 + 390 = 2726` trainable parameters

Why this matters:

- model stays small enough for tinyML
- dense-only topology is easy for TFLite conversion
- runtime op set remains minimal on embedded targets

## Compilation Choices and Their Meaning

The model is compiled with:

- optimizer: Adam
- loss: sparse categorical crossentropy
- metric: accuracy

### Why Adam

Adam is robust for small tabular feature sets and usually converges quickly without heavy hyperparameter tuning.

### Why Sparse Categorical Crossentropy

Labels are integer encoded (0..5), not one-hot encoded. Sparse categorical crossentropy is the correct loss for this label format.

### Why Accuracy as Metric

Accuracy is simple and useful for first-pass model checks, especially with near-balanced classes.

For production analysis, per-class precision and recall should also be tracked in validation scripts.

## Training Loop Behavior

The helper `train_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)`:

- instantiates a fresh model
- trains for a fixed number of epochs
- evaluates validation set each epoch
- returns both trained model and `History`

Important characteristics of the current setup:

- no callbacks are configured
- no early stopping is used
- no model checkpointing is used
- reproducibility depends mainly on NumPy seeding in data generation

This is intentionally simple and good for clarity, but may train longer than needed if validation has already plateaued.

## Dataset Split Strategy and Keras Impact

The main pipeline performs two stratified splits:

1. 70% train, 30% temporary
2. temporary split into 15% validation and 15% test

Impact on Keras training:

- training uses only the train split
- validation metrics are visible during fitting
- final unbiased performance estimate comes from the test split

Because split is stratified, class proportions remain stable across train, validation, and test partitions.

## Feature Scaling and Why Keras Depends on It

Keras receives standardized features from `StandardScaler`.

Standardization formula per feature:

`x_scaled = (x - mean) / std`

Practical effect:

- gradients are better conditioned
- convergence is faster and more stable
- one feature with larger numeric range does not dominate updates

Critical integration point:

- scaler mean and scale are exported to [model_info.json](model_info.json)
- embedded preprocessing must use the same values

## Evaluation Flow

`evaluate_model(model, X_test, y_test)` runs Keras evaluation on the holdout set.

It returns:

- test loss
- test accuracy

Interpretation guidance:

- high train and high test: good fit
- high train and lower validation/test: possible overfitting
- low train and low test: underfitting or feature limitations

## Keras Artifact Produced by Main Training

The main script saves:

- `fault_classifier_keras.h5`

Why keep this artifact:

- enables transfer/fine-tuning with real data later
- allows architecture and weights to be reloaded quickly
- provides a reproducible baseline before quantization

## Retraining and Fine-Tuning in Detail

Retraining logic is in `RealDataRetrainer` inside [validate_and_retrain.py](validate_and_retrain.py).

### `RealDataRetrainer.__init__(old_model_path, old_model_info_path)`

Loads the old Keras model using `keras.models.load_model`.

It also loads prior scaler metadata for reference.

### `load_real_data_csv(csv_path, test_split=0.2)`

Steps:

1. loads real CSV data
2. engineers features using the same function as original training
3. fits a new `StandardScaler` on real feature data
4. performs stratified train/test split

Important note:

- retraining uses real-data scaler values, which are then exported in versioned metadata

### `retrain(X_train, X_test, y_train, y_test, epochs=30, batch_size=32)`

Fine-tuning behavior:

1. evaluate baseline model on new test set
2. reduce learning rate to 0.001 for stable adaptation
3. continue fitting from existing weights
4. evaluate again and print improvement

This is transfer learning in a small tabular context.

### `save_retrained_model(model, scaler, version='v2')`

Saves:

- versioned Keras model (`fault_classifier_v2_keras.h5` style)
- versioned TFLite model
- versioned metadata JSON with new scaler parameters

Operationally important:

- class order must remain stable unless the entire stack is updated
- metadata and binary model must be deployed together as a pair

## Keras-to-TFLite Handoff: What Keras Must Guarantee

For clean conversion and embedded operation, the Keras model should remain:

- feed-forward and stateless
- composed of operators available in TFLite and TFLM
- trained against the same feature representation used in firmware

Current model satisfies this well by using only dense and softmax operations.

## Practical Tuning Knobs for This Specific Keras Pipeline

If you want to improve Keras training quality while keeping tinyML constraints, tune in this order:

1. epoch count and early stopping
2. batch size
3. width of hidden layers (for example 24/48 or 32/64)
4. dropout or L2 only if overfitting is observed

Recommended low-risk additions:

- EarlyStopping on validation loss
- ModelCheckpoint for best validation epoch
- ReduceLROnPlateau for smoother late-stage convergence

These can improve reliability without changing deployment architecture.

## Failure Modes and Debug Signals on Keras Side

Common symptoms and likely causes:

- unstable training loss: scaler mismatch, bad data ranges, or too high learning rate
- high training accuracy but low test accuracy: overfitting or synthetic-to-real gap
- class-specific failures: insufficient synthetic diversity for that fault mode
- poor retrain gains: domain mismatch or label noise in real CSV data

Useful debug actions:

- inspect per-class confusion matrix
- plot training and validation curves from `History`
- verify label distribution and class balance after split
- verify same feature engineering path is used in all scripts

## Reproducibility Notes

Current code sets NumPy seed in data generation. For tighter reproducibility across runs, also consider seeding TensorFlow and controlling deterministic ops where possible.

Reproducibility checklist:

- fixed random seeds
- pinned package versions
- archived model artifact and metadata
- saved train/validation/test split definition for experiments

## Keras Quality Gates Before Export

Before converting to TFLite and deploying to firmware, validate these gates:

1. test accuracy is stable across multiple runs
2. no class has unacceptable recall
3. validation curves do not show severe divergence
4. model size remains within tinyML budget after conversion
5. exported scaler metadata matches training data pipeline

## Embedded-Impact Notes from the Keras Perspective

Choices made in Keras directly affect embedded runtime:

- wider dense layers increase model size and memory pressure
- unsupported layers can block conversion or inflate runtime cost
- unstable training can force overly frequent model updates
- changing feature count breaks input contract with firmware

For RA6T3 deployment, keep architecture conservative unless there is measured need for additional complexity.

## Alternative Keras Approaches and How They Would Likely Turn Out

The current model is a compact dense MLP and is a strong baseline for this tabular 6-feature input. You can still try alternatives in Keras, but each option shifts tradeoffs between accuracy, robustness, Flash usage, RAM usage, and inference latency.

### Quick Outcome Matrix

| Approach | Accuracy Potential | Model Size / Latency Impact | TFLite / TFLM Risk | Typical Outcome in This Project |
|---|---|---|---|---|
| Smaller MLP (for example 16 -> 32) | Slightly lower or similar | Better (smaller and faster) | Very low | Good if memory or timing margin is tight |
| Wider MLP (for example 64 -> 128) | Can improve on complex boundaries | Worse (larger and slower) | Low | May overfit synthetic patterns unless real data is strong |
| Add Dropout | Better generalization in some cases | Minimal runtime impact when exported | Low | Useful if train/val gap appears |
| Add L2 regularization | Better robustness, smoother boundaries | Minimal runtime impact | Low | Often safer than heavy architecture change |
| Add BatchNormalization | Sometimes stabilizes training | Extra ops and parameters | Medium | Can help training, but may complicate tiny runtime budget |
| 1D CNN on feature vector | Usually little gain on 6 tabular features | Extra complexity | Medium | Rarely worth it for this feature shape |
| LSTM/GRU sequence model | Could help only with true temporal windows | Significantly larger and slower | Medium to high | Not recommended unless temporal history is explicitly modeled |
| Multi-head or multitask model | Better structure for shared tasks | Bigger model and output logic | Medium | Useful only if you redesign labels and downstream logic |
| Focal loss or class-weighted loss | Better minority-class recall | No size change | Low | Helpful when real data becomes imbalanced |

### 1) Smaller Dense Network

Example direction:

- Input -> Dense(16, relu) -> Dense(32, relu) -> Dense(6, softmax)

Likely outcome:

- Slight drop in top-line accuracy in some runs
- Better embedded behavior (smaller model, lower latency)
- Often enough when features are clean and well-engineered

When to choose it:

- RA6T3 memory headroom is tight
- timing margin is narrow
- baseline accuracy is already acceptable

### 2) Wider Dense Network

Example direction:

- Input -> Dense(64, relu) -> Dense(128, relu) -> Dense(6, softmax)

Likely outcome:

- Can fit synthetic data better
- Can improve test accuracy if the decision boundary is genuinely complex
- Raises overfitting risk, especially with synthetic-to-real domain gap
- Increases Flash and runtime cost

When to choose it:

- measured class confusion persists after data-quality improvements
- you can afford larger runtime footprint

### 3) Keep Architecture, Add Regularization

Low-risk options:

- L2 regularization on dense kernels
- light dropout between dense layers
- early stopping callback

Likely outcome:

- better validation stability
- lower overfitting risk
- usually no major embedded deployment penalty

This is often the best first alternative before changing layer widths.

### 4) Batch Normalization Variant

Possible architecture:

- Dense -> BatchNorm -> ReLU blocks

Likely outcome:

- smoother optimization in some training setups
- additional parameters and ops in export path
- more conversion/runtime complexity for tiny deployments

Recommendation:

- use only if training instability is proven and simpler fixes do not work

### 5) Temporal Keras Models (LSTM/GRU)

These become relevant only if you change the input contract from a single snapshot to a time window.

Required redesign:

- input shape becomes `(window_length, feature_count)`
- training data must contain aligned sequences
- embedded inference must maintain sequence buffers

Likely outcome:

- potentially better detection of evolving faults
- substantially higher memory and CPU requirements
- much more integration complexity in firmware

For current snapshot-style features, this is usually not worth the cost.

### 6) Loss-Level Alternatives Without Architecture Change

You can improve difficult classes while keeping model size stable by changing loss behavior.

Options:

- class-weighted sparse categorical crossentropy
- focal loss

Likely outcome:

- improved recall on rare or hard classes
- possible slight drop in overall accuracy
- no major model size change

This is highly practical once real field data introduces class imbalance.

### 7) Calibration-Focused Approach

Current deployment uses argmax over quantized outputs. If confidence behavior matters, you can add calibration analysis on the Keras side before conversion.

Options:

- temperature scaling on validation outputs
- reliability diagram checks

Likely outcome:

- better interpretation of confidence values
- safer threshold-based decision logic
- minimal architecture impact

### 8) Ensemble-Like Strategies

Train multiple small Keras models and aggregate votes in Python.

Likely outcome:

- sometimes more robust predictions
- deployment complexity increases significantly
- usually not suitable for MCU unless distilled back into one compact model

For RA6T3-style deployment, if ensemble helps offline, distill it into one student model before exporting.

## Recommended Experiment Order (Practical)

Use this order to get meaningful gains while preserving embedded constraints:

1. Keep current architecture, add EarlyStopping and best-checkpoint restore.
2. Add light L2 regularization and compare per-class recall.
3. Try a slightly smaller model and measure accuracy vs latency.
4. Try a slightly wider model only if needed.
5. Apply class-weighted or focal loss when real data is imbalanced.
6. Consider temporal models only after proving snapshot features are insufficient.

## Decision Rule for This Project

Given this repository and target context, the best default is:

- stay with dense MLP
- improve training procedure first
- modify model width only with measured evidence
- avoid sequence-heavy architectures unless requirements explicitly demand temporal memory

This gives the highest chance of improving field performance without breaking tinyML deployment constraints.

## Summary

In this project, Keras is the learning engine that maps engineered motor diagnostics features to six fault classes using a compact dense network. Its output is intentionally designed for smooth quantization and embedded deployment. The most important success factors are consistent feature scaling, stable training behavior, and strict artifact pairing between model binary and metadata.
