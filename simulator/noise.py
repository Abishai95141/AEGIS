"""
AEGIS 3.0 — Sensor Noise Models

Realistic CGM sensor degradation for robust validation:
- Gaussian measurement noise
- Signal dropout (random gaps)
- Sensor drift (slow bias accumulation)
- Calibration jumps (sudden offset changes)
"""

import numpy as np


class SensorNoiseModel:
    """
    Realistic CGM sensor noise model.

    Based on published CGM accuracy data: MARD ~10-12% for modern sensors,
    with known failure modes (dropouts, drift, compression artifacts).
    """

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        self.drift_bias = 0.0
        self.drift_rate = 0.0

    def apply_gaussian_noise(self, glucose_values, noise_std=10.0):
        """
        Add Gaussian measurement noise (±noise_std mg/dL).

        Typical CGM has ~10-15 mg/dL noise at 95% CI.
        """
        noise = self.rng.normal(0, noise_std, len(glucose_values))
        noisy = glucose_values + noise
        return np.clip(noisy, 20, 600)

    def apply_dropout(self, glucose_values, dropout_rate=0.03,
                      max_gap_steps=3):
        """
        Simulate random signal dropout.

        dropout_rate: probability of starting a dropout at each step
        max_gap_steps: maximum consecutive missing readings
        """
        result = glucose_values.copy()
        mask = np.ones(len(result), dtype=bool)
        i = 0
        while i < len(result):
            if self.rng.random() < dropout_rate:
                gap_len = self.rng.randint(1, max_gap_steps + 1)
                end = min(i + gap_len, len(result))
                mask[i:end] = False
                result[i:end] = np.nan
                i = end
            else:
                i += 1
        return result, mask

    def apply_drift(self, glucose_values, drift_rate_per_day=5.0):
        """
        Simulate slow sensor drift (bias accumulation over days).

        drift_rate_per_day: mg/dL bias accumulated per day
        Steps are assumed to be 5-min intervals (288 per day).
        """
        n = len(glucose_values)
        steps_per_day = 288
        drift_per_step = drift_rate_per_day / steps_per_day

        drift = np.zeros(n)
        bias = 0.0
        for i in range(n):
            bias += drift_per_step + self.rng.normal(0, drift_per_step * 0.1)
            drift[i] = bias
            # Occasional partial reset (simulates sensor recalibration)
            if self.rng.random() < 0.001:
                bias *= 0.3

        result = glucose_values + drift
        return np.clip(result, 20, 600), drift

    def apply_calibration_jump(self, glucose_values, jump_prob=0.002,
                                jump_magnitude=25.0):
        """
        Simulate sudden calibration jumps.

        These occur when the sensor recalibrates against a finger-prick
        reading, causing a sudden offset change.
        """
        result = glucose_values.copy()
        current_offset = 0.0
        for i in range(len(result)):
            if self.rng.random() < jump_prob:
                current_offset = self.rng.normal(0, jump_magnitude)
            result[i] += current_offset
        return np.clip(result, 20, 600)

    def apply_all(self, glucose_values, noise_std=10.0, dropout_rate=0.03,
                  drift_rate=5.0, jump_prob=0.002):
        """
        Apply all noise models simultaneously.

        Returns: (noisy_values, dropout_mask, drift_values)
        """
        # Order matters: drift first, then noise, then dropout
        drifted, drift_vals = self.apply_drift(glucose_values, drift_rate)
        noisy = self.apply_gaussian_noise(drifted, noise_std)
        noisy = self.apply_calibration_jump(noisy, jump_prob)
        result, mask = self.apply_dropout(noisy, dropout_rate)
        return result, mask, drift_vals
