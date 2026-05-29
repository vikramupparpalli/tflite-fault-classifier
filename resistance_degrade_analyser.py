"""
Resistance Degradation Analyser
================================
Handles fault data generation and feature extraction for:
  - Winding resistance degradation (R_winding increase over time)

Fault scenarios: RESISTANCE_DEGRADE (6)
"""

import numpy as np


class ResistanceDegradeAnalyser:
    """Generates fault data and extracts features for winding resistance degradation analysis."""

    def __init__(self, nominal: dict, limits: dict):
        self.nominal = nominal
        self.limits = limits

    def resistance_degrade_fault(self, n_samples=250):
        """Winding resistance degradation: R_winding ramps from nominal toward R_max_degrade."""
        data = []
        for _ in range(n_samples):
            t = np.random.uniform(0, 1)
            Ia = self.nominal['Ia'] + np.random.normal(0, 0.5)
            Ib = self.nominal['Ib'] + np.random.normal(0, 0.5)
            Ic = self.nominal['Ic'] + np.random.normal(0, 0.5)
            Vdc = self.nominal['Vdc'] + np.random.normal(0, 1.0)
            Temp = self.nominal['Temp'] + np.random.normal(0, 2.0)
            VFO_feedback = 1
            R_winding = self.nominal['R_winding'] + t * (self.limits['R_max_degrade'] - self.nominal['R_winding'])
            data.append([Ia, Ib, Ic, Vdc, Temp, VFO_feedback, R_winding])
        return np.array(data), np.full(n_samples, 6, dtype=int)

    @staticmethod
    def extract_features(R_winding: float):
        """
        Extract resistance degradation feature.

        Returns:
            R_normalized — resistance increase above 1.0 per-unit baseline
        """
        return R_winding - 1.0
