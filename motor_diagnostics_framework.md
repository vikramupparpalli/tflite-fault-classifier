# Motor Control Diagnostics Module - Organizational Framework

## System Parameters
- **Interrupt frequency**: 16 kHz (62.5 µs per sample)
- **Diagnostic scope**: 5 monitored conditions
- **Output**: Fault flags, warning states, and control actions

---

## 1. PHASE CURRENT MONITORING

### 1.1 Detection Strategy
- Sample three phase currents (Ia, Ib, Ic) at each interrupt
- Check against two thresholds: **sustainable** and **peak/transient**
- Apply filtering for noise rejection

### 1.2 Fault Conditions

| Fault Type | Detection Logic | Threshold | Persistence | Action |
|-----------|-----------------|-----------|-------------|--------|
| **Over-current (sustained)** | RMS or moving average of Ia, Ib, Ic | Peak_threshold (e.g. 1.5× nominal) | 1–5 ms (16–80 samples) | Reduce torque command; log |
| **Over-current (transient)** | Single sample max | Peak_peak_threshold (e.g. 2× nominal) | Immediate or after N samples | Soft current limit, no shutdown |
| **Phase imbalance** | Difference between max/min phase | Imbalance_threshold (e.g. 0.3A) | 10 ms+ | Flag abnormal, investigate |
| **Missing phase** | Detection of zero current in one phase | Current_min_threshold | 100 µs (few samples) | Fault (commutation hazard) |

### 1.3 Data Handling
- **Filtering**: Low-pass IIR or moving-average window (4–16 sample window = 0.25–1 ms)
- **Quantization**: 12–16 bit ADC; scale to engineering units (Amps)
- **Per-interrupt logic**: 
  ```
  On each 16 kHz interrupt:
    - Read raw ADC samples
    - Convert to Amps
    - Update moving-average register
    - Compare moving-avg against thresholds
    - Increment/reset persistence counters
    - Set fault flags when thresholds crossed
  ```

---

## 2. IPM TEMPERATURE MONITORING

### 2.1 Detection Strategy
- Single or multi-point temperature sensor (built into IPM)
- Two thresholds: **warning** and **fault/shutdown**
- Track rate of change for transient detection

### 2.2 Fault Conditions

| Fault Type | Detection Logic | Threshold | Delay | Action |
|-----------|-----------------|-----------|-------|--------|
| **Temperature warning** | T > T_warn | ~85°C (example) | 100 ms | Set warning flag; log event |
| **Over-temperature (fault)** | T > T_max | ~110°C (example) | 50 ms | Reduce PWM duty, prepare shutdown |
| **Thermal runaway** | dT/dt > rate_max | ~5°C/10ms | Immediate | Emergency shutdown |
| **Sensor malfunction** | T out of range OR no change over time | T_min–T_max range (e.g., -40 to +150°C) | 1 sec static | Fault flag; assume worst-case |

### 2.3 Data Handling
- **Filtering**: IIR first-order or moving-average (especially important for thermal slow dynamics)
- **Thermal time constant**: ~100–500 ms typical → use longer averaging window (100+ samples)
- **Per-interrupt logic**:
  ```
  On each 16 kHz interrupt (every 62.5 µs):
    - Read temperature ADC (may not update every cycle—sensor update rate ~1 kHz)
    - Update running average / IIR estimate
    - Calculate dT/dt from differentiation window
    - Compare against warn and fault thresholds
    - Set flags and log
  ```

---

## 3. DC BUS VOLTAGE MONITORING

### 3.1 Detection Strategy
- Monitor main supply voltage feeding the inverter
- Check for under-voltage and over-voltage conditions
- Detect brownout/sag events

### 3.2 Fault Conditions

| Fault Type | Detection Logic | Threshold | Persistence | Action |
|-----------|-----------------|-----------|-------------|--------|
| **Under-voltage** | Vdc < V_min | ~9V (for 12V system) or ~36V (48V) | 2–5 ms | Disable PWM, shutdown |
| **Over-voltage** | Vdc > V_max | ~15V (for 12V) or ~60V (48V) | 1–2 ms | Clamp PWM or fold-back |
| **Supply sag** | Vdc drops transient | V_min – δ | 1 sample (check fast) | Flag; optionally reduce torque |
| **Voltage ripple** | Peak-to-peak variation | Ripple_max (spec-dependent) | Continuous monitor | Log diagnostic; not usually fault |

### 3.3 Data Handling
- **Update rate**: DC bus voltage changes slowly; sample every cycle (62.5 µs)
- **Filtering**: Light filtering (4–8 sample moving-avg) to reject switching noise
- **Hysteresis**: For over/under-voltage, use hysteresis band (e.g., +0.5V margin on recovery)
- **Per-interrupt logic**:
  ```
  On each interrupt:
    - Read Vdc ADC
    - Apply light LP filter
    - Check against V_min and V_max (with hysteresis)
    - Compare rate of change for sag detection
    - Set fault or warning flag
  ```

---

## 4. IPM VFO (GATE DRIVER) FEEDBACK MONITORING

### 4.1 Detection Strategy
- Monitor digital ON/OFF feedback from intelligent power module (IPM) gate driver
- Detect loss of drive readiness (OFF = fault)

### 4.2 Fault Conditions

| Fault Type | Detection Logic | Persistence | Action |
|-----------|-----------------|-------------|--------|
| **VFO feedback loss** | VFO_feedback = 0 (gate driver OFF) | 1–10 ms | Disable output; fault |
| **IPM disable/fault pin** | Fault line from IPM pulled low | Immediate | Stop; hardware-level shutdown |

### 4.3 Data Handling
- **Signal type**: Digital input (GPIO)
- **Logic**: Sample VFO_feedback every interrupt (1 = ON, 0 = OFF)
- **Per-interrupt logic**:
  ```
  On each 16 kHz cycle:
    - Read VFO_feedback (digital input)
    - If VFO_feedback == 0 for >N cycles, set fault flag
    - If VFO_feedback == 1, system OK
  ```

---

## 5. MOTOR WINDING RESISTANCE (CHANGE DETECTION)

### 5.1 Detection Strategy
- Estimate or measure winding resistance R_winding over time
- Track changes relative to initial / baseline value
- Detect gradual degradation (aging, turn shorts) or sudden faults

### 5.2 Measurement Methods

**Option A: Periodic DC injection**
- Stop motor, apply low DC current pulse, measure voltage response
- Calculate R = V / I
- Interval: every 10–100 motor cycles or on-demand
- Pros: Direct measurement; Cons: Requires motor stop or careful injection

**Option B: Impedance estimation during dead-time**
- Use motor back-EMF and voltage/current data
- Estimate from (V_cmd - V_bemf) / I during active phases
- Runs continuously; no motor stop needed
- Pros: Online; Cons: Requires accurate back-EMF model

**Option C: Phase resistance monitoring**
- Measure voltage drop across each phase during current rise
- R = ΔV / ΔI during PWM transition
- High-frequency data available; requires filtering
- Pros: Frequent updates; Cons: Sensitive to noise, timing

### 5.3 Fault Conditions

| Fault Type | Detection Logic | Threshold | Persistence | Action |
|-----------|-----------------|-----------|-------------|--------|
| **Winding degradation** | R > R_baseline × 1.2–1.5 | 20–50% increase | 1+ hour of observation | Warning; log trend |
| **Turn-to-turn short** | R < R_baseline × 0.7–0.8 | 20–30% decrease | Multiple measurements | Fault; disable drive |
| **Open winding** | R → ∞; no current response | Resistance undefined | 1–10 ms | Immediate fault |
| **Temperature-dependent change** | dR/dT normal (~+0.4%/°C copper) | Expected slope | Continuous model | Filter thermal effects; compare to model |

### 5.4 Data Handling
- **Update rate**: Slow (every 100 ms to seconds, depending on method)
- **Baseline storage**: Store initial R at startup; reference for degradation
- **Filtering**: Use long time-constant averaging; resist single noisy measurement
- **Trending**: Maintain ring buffer of measurements (e.g., last 10 samples) to detect slope
- **Compensation**: Temperature-corrected baseline (R_baseline at ambient)
- **Per-measurement logic**:
  ```
  On each measurement cycle (e.g., every 100 ms):
    - Perform R measurement (DC injection, back-EMF fit, or phase resistance)
    - Temperature-correct: R_eff = R_measured / (1 + α·(T - T_ref))
    - Compare to R_baseline ± tolerance band
    - Update trend ring buffer
    - Check rate of change (dR/dt over last N samples)
    - Set degradation or fault flag if thresholds crossed
    - Log event with timestamp, value, and trend
  ```

---

## 6. STATE MACHINES & FAULT SEQUENCING

### 6.1 Example: Over-Current Fault State Machine
```
IDLE (no fault)
  ↓ [current > threshold && count < max_count]
CURRENT_WARNING (reduce torque slightly)
  ↓ [current > threshold && count >= max_count]
CURRENT_FAULT (halt PWM, log)
  ↓ [current < threshold && hysteresis_time passed]
IDLE (recovery after N seconds)
```

### 6.2 Example: Temperature Fault State Machine
```
NORMAL (T < T_warn)
  ↓ [T > T_warn && time > 100ms]
TEMP_WARNING (reduce power, notify)
  ↓ [T > T_max && time > 50ms]
TEMP_FAULT (shutdown, log)
  ↓ [T < T_max - 5°C && idle_time > 2min]
COOLING (monitor until T < T_warn)
  ↓ [manual_reset]
NORMAL
```

### 6.3 Hysteresis & De-Bouncing
- Use hysteresis band for each threshold (e.g., fault at V > V_max, recovery at V < V_max – 0.5V)
- Require N consecutive samples above/below threshold before state change
- Prevents chattering from noise

---

## 7. INTERRUPT LOOP PSEUDOCODE

```c
// 16 kHz timer interrupt (every 62.5 µs)
void interrupt_handler_16kHz(void) {
    
    // ===== Read ADC inputs =====
    uint16_t adc_Ia = read_adc(CH_IA);
    uint16_t adc_Ib = read_adc(CH_IB);
    uint16_t adc_Ic = read_adc(CH_IC);
    uint16_t adc_Vdc = read_adc(CH_VDC);
    uint16_t adc_Temp = read_adc(CH_TEMP);  // May update slower
    
    // Convert to engineering units
    float Ia = adc_Ia * ADC_SCALE_CURRENT;
    float Ib = adc_Ib * ADC_SCALE_CURRENT;
    float Ic = adc_Ic * ADC_SCALE_CURRENT;
    float Vdc = adc_Vdc * ADC_SCALE_VOLTAGE;
    float Temp = adc_Temp * ADC_SCALE_TEMP + TEMP_OFFSET;
    
    // ===== 1. Phase Current Check =====
    update_current_avg(Ia, Ib, Ic);  // IIR filter or moving average
    if (current_avg > CURRENT_THRESHOLD) {
        current_fault_count++;
        if (current_fault_count > CURRENT_FAULT_SAMPLES) {
            set_fault_flag(FAULT_OVERCURRENT);
            torque_limit = REDUCED_TORQUE;
        }
    } else {
        current_fault_count = 0;
    }
    
    // ===== 2. Temperature Check =====
    update_temp_avg(Temp);  // Slower update, but compare every cycle
    if (temp_avg > TEMP_WARNING_THRESHOLD) {
        temp_warning_count++;
        if (temp_warning_count > TEMP_WARNING_SAMPLES) {
            set_warning_flag(WARN_OVERTEMP);
        }
    }
    if (temp_avg > TEMP_FAULT_THRESHOLD) {
        temp_fault_count++;
        if (temp_fault_count > TEMP_FAULT_SAMPLES) {
            set_fault_flag(FAULT_OVERTEMP);
            disable_pwm();
        }
    }
    
    // ===== 3. Voltage Check =====
    if (Vdc < VBUS_MIN_THRESHOLD) {
        vbus_low_count++;
        if (vbus_low_count > VBUS_FAULT_SAMPLES) {
            set_fault_flag(FAULT_UNDERVOLT);
            disable_pwm();
        }
    } else if (Vdc > VBUS_MAX_THRESHOLD) {
        vbus_high_count++;
        if (vbus_high_count > VBUS_FAULT_SAMPLES) {
            set_fault_flag(FAULT_OVERVOLT);
            limit_duty();
        }
    } else {
        vbus_low_count = 0;
        vbus_high_count = 0;
    }
    
    // ===== 4. VFO Feedback Check =====
    if (VFO_feedback == 0) {
      vfo_off_count++;
      if (vfo_off_count > VFO_OFF_FAULT_SAMPLES) {
        set_fault_flag(FAULT_VFO_LOSS);
        disable_pwm();
      }
    } else {
      vfo_off_count = 0;
    }
    
    // ===== 5. Winding Resistance (Periodic) =====
    resistance_sample_count++;
    if (resistance_sample_count >= RESISTANCE_SAMPLE_INTERVAL) {
        resistance_sample_count = 0;
        float R_measured = measure_winding_resistance();  // Method-dependent
        update_resistance_trend(R_measured);
        check_resistance_degradation();
    }
    
    // ===== Output PWM / Control =====
    if (any_fault_flag_set()) {
        apply_fault_response();  // Ramp down, shutdown, etc.
    }
    
    // Log critical events
    if (fault_flag_changed) {
        log_fault_event(fault_code, current_values);
    }
}
```

---

## 8. DATA STRUCTURES & STORAGE

### 8.1 Filter State
```c
struct CurrentFilter {
    float avg_Ia, avg_Ib, avg_Ic;  // Running average
    uint32_t fault_count;
    uint32_t warning_count;
};

struct TempFilter {
    float avg;
    float prev;
    uint32_t warn_count;
    uint32_t fault_count;
};

struct VoltageMonitor {
    float avg_Vdc;
    uint32_t low_count, high_count;
};

struct ResistanceTrend {
    float baseline_R;
    float recent_R[10];  // Ring buffer of last 10 measurements
    float measured_at_temp[10];  // Temperature at each measurement
    uint32_t index;
    float rate_of_change;  // dR/dt
};
```

### 8.2 Fault Registers
```c
struct FaultStatus {
    uint8_t over_current    : 1;
    uint8_t over_temp       : 1;
    uint8_t under_volt      : 1;
    uint8_t over_volt       : 1;
    uint8_t vfo_loss        : 1;
    uint8_t winding_degrade : 1;
    uint8_t temp_warn       : 1;
    uint8_t reserved        : 1;
};

struct WarningStatus {
    uint8_t temp_warning    : 1;
    uint8_t resistance_trend: 1;
    uint8_t vfo_warning     : 1;
    uint8_t reserved        : 5;
};
```

---

## 9. TIMING & PRIORITIZATION

### Execution Priority (within 62.5 µs window)
1. **Critical (immediate check)**: VFO signal, voltage extremes → disable safety
2. **High (first half)**: Phase currents, temperature → set warnings
3. **Medium (every few cycles)**: Averaging updates, state machine transitions
4. **Low (periodic)**: Winding resistance, logging, trend analysis

### Execution Budget
- ADC reads & filtering: ~10 µs
- Threshold comparisons: ~5 µs
- State transitions & flags: ~5 µs
- Logging / buffer updates: ~5 µs
- PWM output & control: ~30 µs
- **Total safety margin**: ~10 µs (plenty of headroom for 62.5 µs cycle)

---

## 10. THRESHOLD SELECTION GUIDANCE

### Phase Current
- **Nominal**: 10 A (example)
- **Sustainable threshold**: 1.3–1.5× nominal → 13–15 A
- **Peak/transient threshold**: 1.8–2.0× nominal → 18–20 A
- **Persistence**: 1–5 ms (16–80 samples)

### Temperature
- **Warning**: ~85°C (leaves 25°C margin to max)
- **Fault/shutdown**: ~110°C (or per IPM datasheet)
- **Thermal runaway rate**: >5°C per 10 ms
- **Persistence**: 50–100 ms

### Voltage
- **Nominal (12V system)**: 12 V
- **Under-voltage**: <9.0 V (25% sag)
- **Over-voltage**: >15 V (25% overshoot)
- **Hysteresis**: ±0.5 V recovery band

### VFO
- **Expected frequency**: Per IPM datasheet (often ~10–50 kHz)
- **Tolerance band**: ±10–20%
- **Timeout**: 10–50 ms (loss of 100–500 edges)

### Winding Resistance
- **Baseline**: Measured at startup or during factory test
- **Degradation warning**: +20% from baseline
- **Fault threshold**: +50% from baseline, or >30% decrease (short)
- **Temperature coefficient**: ~+0.4%/°C for copper

---

## 11. TESTING & VALIDATION CHECKLIST

- [ ] Phase current detection: Verify at 1.1×, 1.5×, 2.0× nominal current
- [ ] Over-current persistence: Confirm N-sample counter prevents false positives
- [ ] Temperature ramp: Test with thermal chamber; verify warning and fault boundaries
- [ ] Voltage brown-out: Simulate supply sag; verify shutdown within 2 ms
- [ ] VFO timeout: Disable drive signal; verify fault flag within 10 ms
- [ ] Winding resistance: Baseline at cold and hot; verify degradation detection
- [ ] Hysteresis: Confirm no chatter near thresholds
- [ ] State recovery: Verify fault → idle transition and reset conditions
- [ ] Latency: Confirm all critical checks complete within 16 kHz cycle
- [ ] Logging: Verify events timestamped and non-blocking

---

## 12. REFERENCES & NOTES

- **IPM datasheet**: Confirm VFO spec, fault output polarity, thermal rating, current rating
- **Motor nameplate**: Nominal current, winding resistance (hot/cold), thermal time constant
- **Control system requirements**: Max torque, emergency stop sequence, recovery rules
- **EMC considerations**: Filter switching noise from current/voltage sense lines; shield VFO signal
- **Reliability target**: Specify MTBF for each fault; plan diagnostic logging for field analysis
