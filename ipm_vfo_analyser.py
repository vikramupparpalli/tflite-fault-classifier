"""
IPM Temperature & VFO Feedback Analyser
========================================
Handles fault data generation and feature extraction for:
  - IPM (Intelligent Power Module) temperature
  - VFO (Voltage Feedback Output / gate driver) feedback status

Fault scenarios: OVERTEMP (4), VFO_FAULT (5)
"""

import numpy as np


class IPMVFOAnalyser:
    """Generates fault data and extracts features for IPM temperature and VFO feedback analysis."""

    def __init__(self, nominal: dict, limits: dict):
        self.nominal = nominal
        self.limits = limits

    def overtemp_fault(self, n_samples=400):
        """Overtemperature: IPM temperature ramps from nominal toward Temp_max."""
        data = []
        for _ in range(n_samples):
            t = np.random.uniform(0, 1)
            Ia = self.nominal['Ia'] + np.random.normal(0, 0.5)
            Ib = self.nominal['Ib'] + np.random.normal(0, 0.5)
            Ic = self.nominal['Ic'] + np.random.normal(0, 0.5)
            Vdc = self.nominal['Vdc'] + np.random.normal(0, 1.0)
            Temp = self.nominal['Temp'] + t * (self.limits['Temp_max'] - self.nominal['Temp'])
            VFO_feedback = 1
            data.append([Ia, Ib, Ic, Vdc, Temp, VFO_feedback])
        return np.array(data), np.full(n_samples, 4, dtype=int)

    def vfo_fault(self, n_samples=200):
        """VFO fault: gate driver feedback is OFF or toggling unexpectedly."""
        data = []
        for _ in range(n_samples):
            Ia = self.nominal['Ia'] + np.random.normal(0, 0.5)
            Ib = self.nominal['Ib'] + np.random.normal(0, 0.5)
            Ic = self.nominal['Ic'] + np.random.normal(0, 0.5)
            Vdc = self.nominal['Vdc'] + np.random.normal(0, 1.0)
            Temp = self.nominal['Temp'] + np.random.normal(0, 2.0)
            VFO_feedback = 0 if np.random.rand() > 0.5 else 1
            data.append([Ia, Ib, Ic, Vdc, Temp, VFO_feedback])
        return np.array(data), np.full(n_samples, 5, dtype=int)

    @staticmethod
    def extract_features(Temp: float, VFO_feedback: float):
        """
        Extract IPM temperature and VFO feedback features.

        Returns:
            Temp_normalized  — temperature as fraction of 125°C max
            VFO_feedback     — gate driver status pass-through (1=ON, 0=OFF)
        """
        Temp_normalized = Temp / 125.0
        return Temp_normalized, float(VFO_feedback)
