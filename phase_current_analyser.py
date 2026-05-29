"""
Phase Current Analyser
======================
Handles fault data generation and feature extraction for:
  - Phase currents (Ia, Ib, Ic)
  - DC Bus Voltage (V_dc)

Fault scenarios: NO_FAULT (0), OVERCURRENT (1), OVERVOLTAGE (2), UNDERVOLTAGE (3),
                 PERSISTENT_OVERCURRENT_IPM_STRESS (6)
"""

import numpy as np


class PhaseCurrentAnalyser:
    """Generates fault data and extracts features for phase current and DC bus voltage analysis."""

    def __init__(self, nominal: dict, limits: dict):
        self.nominal = nominal
        self.limits = limits

    def healthy_operation(self, n_samples=1000):
        """Normal operation: small noise around nominal for all channels."""
        data = []
        for _ in range(n_samples):
            Ia = np.clip(self.nominal['Ia'] + np.random.normal(0, 0.5), 0, self.limits['Ia_max'])
            Ib = np.clip(self.nominal['Ib'] + np.random.normal(0, 0.5), 0, self.limits['Ia_max'])
            Ic = np.clip(self.nominal['Ic'] + np.random.normal(0, 0.5), 0, self.limits['Ia_max'])
            Vdc = np.clip(self.nominal['Vdc'] + np.random.normal(0, 0.8), self.limits['Vdc_min'], self.limits['Vdc_max'])
            Temp = np.clip(self.nominal['Temp'] + np.random.normal(0, 2.0), 0, self.limits['Temp_max'])
            VFO_feedback = 1
            R_winding = self.nominal['R_winding'] + np.random.normal(0, 0.02)
            data.append([Ia, Ib, Ic, Vdc, Temp, VFO_feedback, R_winding])
        return np.array(data), np.zeros(n_samples, dtype=int)

    def overcurrent_fault(self, n_samples=500):
        """Overcurrent: phase currents ramp from nominal toward Ia_max."""
        data = []
        for _ in range(n_samples):
            t = np.random.uniform(0, 1)
            Ia = self.nominal['Ia'] + t * (self.limits['Ia_max'] - self.nominal['Ia'])
            Ib = self.nominal['Ib'] + t * (self.limits['Ia_max'] - self.nominal['Ib'])
            Ic = self.nominal['Ic'] + t * (self.limits['Ia_max'] - self.nominal['Ic'])
            Vdc = self.nominal['Vdc'] + np.random.normal(0, 1.0)
            Temp = self.nominal['Temp'] + np.random.normal(0, 2.0)
            VFO_feedback = 1
            R_winding = self.nominal['R_winding'] + np.random.normal(0, 0.02)
            data.append([Ia, Ib, Ic, Vdc, Temp, VFO_feedback, R_winding])
        return np.array(data), np.ones(n_samples, dtype=int)

    def overvoltage_fault(self, n_samples=300):
        """Overvoltage: DC bus ramps from nominal toward Vdc_max."""
        data = []
        for _ in range(n_samples):
            t = np.random.uniform(0, 1)
            Ia = self.nominal['Ia'] + np.random.normal(0, 0.5)
            Ib = self.nominal['Ib'] + np.random.normal(0, 0.5)
            Ic = self.nominal['Ic'] + np.random.normal(0, 0.5)
            Vdc = self.nominal['Vdc'] + t * (self.limits['Vdc_max'] - self.nominal['Vdc'])
            Temp = self.nominal['Temp'] + np.random.normal(0, 2.0)
            VFO_feedback = 1
            R_winding = self.nominal['R_winding'] + np.random.normal(0, 0.02)
            data.append([Ia, Ib, Ic, Vdc, Temp, VFO_feedback, R_winding])
        return np.array(data), np.full(n_samples, 2, dtype=int)

    def undervoltage_fault(self, n_samples=300):
        """Undervoltage: DC bus sags from nominal toward Vdc_min."""
        data = []
        for _ in range(n_samples):
            t = np.random.uniform(0, 1)
            Ia = self.nominal['Ia'] + np.random.normal(0, 0.5)
            Ib = self.nominal['Ib'] + np.random.normal(0, 0.5)
            Ic = self.nominal['Ic'] + np.random.normal(0, 0.5)
            Vdc = self.nominal['Vdc'] - t * (self.nominal['Vdc'] - self.limits['Vdc_min'])
            Temp = self.nominal['Temp'] + np.random.normal(0, 2.0)
            VFO_feedback = 1
            R_winding = self.nominal['R_winding'] + np.random.normal(0, 0.02)
            data.append([Ia, Ib, Ic, Vdc, Temp, VFO_feedback, R_winding])
        return np.array(data), np.full(n_samples, 3, dtype=int)

    def persistent_overcurrent_ipm_stress(self, n_samples=400):
        """
        Persistent overcurrent (>7 A) held long enough to drive abnormal IPM temperature rise.

        Phase currents are uniformly drawn above 7.0 A. Temperature climbs linearly
        over the window to model I²R heating of the IPM under sustained high current.
        Label 6.
        """
        CURRENT_THRESHOLD = 7.0
        data = []
        temp_rise = np.linspace(0, 15.0, n_samples)  # up to +15 °C over the window
        for i in range(n_samples):
            Ia = np.random.uniform(CURRENT_THRESHOLD, self.limits['Ia_max'])
            Ib = np.random.uniform(CURRENT_THRESHOLD, self.limits['Ia_max'])
            Ic = np.random.uniform(CURRENT_THRESHOLD, self.limits['Ia_max'])
            Vdc = self.nominal['Vdc'] + np.random.normal(0, 1.0)
            Temp = np.clip(
                self.nominal['Temp'] + temp_rise[i] + np.random.normal(0, 1.5),
                0, self.limits['Temp_max']
            )
            VFO_feedback = 1
            R_winding = self.nominal['R_winding'] + np.random.normal(0, 0.02)
            data.append([Ia, Ib, Ic, Vdc, Temp, VFO_feedback, R_winding])
        return np.array(data), np.full(n_samples, 6, dtype=int)

    @staticmethod
    def check_persistent_overcurrent_ipm_stress(
        current_window: np.ndarray,
        temp_window: np.ndarray,
        current_threshold: float = 7.0,
        persistence_ratio: float = 0.80,
        temp_delta_threshold: float = 5.0,
    ) -> dict:
        """
        Detect persistent overcurrent driving abnormal IPM temperature rise.

        Both conditions must hold simultaneously:
          1. At least `persistence_ratio` of samples have any phase current above
             `current_threshold` — i.e. the high current is not a transient spike.
          2. IPM temperature rose by at least `temp_delta_threshold` °C across the
             window — confirming thermal stress, not just a momentary current blip.

        Args:
            current_window:       shape (N, 3) — columns are [Ia, Ib, Ic] in amps
            temp_window:          shape (N,)   — IPM temperature readings in °C
            current_threshold:    per-phase current (A) that counts as overcurrent
            persistence_ratio:    fraction of samples required to exceed threshold
            temp_delta_threshold: minimum °C rise over the window to confirm stress

        Returns:
            dict:
                'detected'          — True if both conditions are met
                'persistence_ratio' — measured fraction of overcurrent samples
                'temp_delta'        — temperature rise (°C) across the window
                'fault_code'        — 6 if detected, else None
        """
        any_phase_over = np.any(current_window > current_threshold, axis=1)
        ratio = float(np.mean(any_phase_over))
        temp_delta = float(temp_window[-1] - temp_window[0]) if len(temp_window) > 1 else 0.0
        detected = ratio >= persistence_ratio and temp_delta >= temp_delta_threshold
        return {
            'detected': detected,
            'persistence_ratio': ratio,
            'temp_delta': temp_delta,
            'fault_code': 6 if detected else None,
        }

    @staticmethod
    def extract_features(Ia: float, Ib: float, Ic: float, Vdc: float):
        """
        Extract current and voltage features.

        Returns:
            I_max           — peak phase current
            I_imbalance     — spread between max and min phase
            V_normalized    — Vdc as fraction of 340V nominal
            I_rms_estimate  — rough three-phase RMS
        """
        I_phases = np.array([Ia, Ib, Ic])
        I_max = float(np.max(I_phases))
        I_imbalance = float(np.max(I_phases) - np.min(I_phases))
        V_normalized = Vdc / 340.0
        I_rms_estimate = float(np.sqrt((Ia**2 + Ib**2 + Ic**2) / 3.0))
        return I_max, I_imbalance, V_normalized, I_rms_estimate
