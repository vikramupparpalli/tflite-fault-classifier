"""
Phase Current Analyser
======================
Handles fault data generation and feature extraction for:
  - Phase currents (Ia, Ib, Ic)
  - DC Bus Voltage (V_dc)

Fault scenarios: NO_FAULT (0), OVERCURRENT (1), OVERVOLTAGE (2), UNDERVOLTAGE (3)
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
            data.append([Ia, Ib, Ic, Vdc, Temp, VFO_feedback])
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
            data.append([Ia, Ib, Ic, Vdc, Temp, VFO_feedback])
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
            data.append([Ia, Ib, Ic, Vdc, Temp, VFO_feedback])
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
            data.append([Ia, Ib, Ic, Vdc, Temp, VFO_feedback])
        return np.array(data), np.full(n_samples, 3, dtype=int)

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
