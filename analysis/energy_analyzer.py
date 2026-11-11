
import numpy as np


class EnergyAnalyzer:

    def __init__(self, window_seconds=8.0, threshold=0.08):
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive.")

        if threshold < 0:
            raise ValueError("threshold cannot be negative.")

        self.window_seconds = window_seconds
        self.threshold = threshold

    def analyze(self, structure, timestamp):

        segments = structure["segments"]

        before = [
            segment["normalized_energy"]
            for segment in segments
            if timestamp - self.window_seconds <= segment["start"] < timestamp
        ]

        after = [
            segment["normalized_energy"]
            for segment in segments
            if timestamp <= segment["start"] < timestamp + self.window_seconds
        ]

        if not before or not after:
            return {
                "trend": "UNKNOWN",
                "energy_before": None,
                "energy_after": None,
                "change": None
            }

        energy_before = float(np.mean(before))
        energy_after = float(np.mean(after))

        change = energy_after - energy_before

        if change >= self.threshold:
            trend = "RISING"
        elif change <= -self.threshold:
            trend = "FALLING"
        else:
            trend = "STABLE"

        return {
            "trend": trend,
            "energy_before": round(energy_before, 3),
            "energy_after": round(energy_after, 3),
            "change": round(change, 3)
        }
