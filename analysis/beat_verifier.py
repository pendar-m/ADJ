
import numpy as np


class BeatVerifier:

    def __init__(self, tolerance=0.05):
        self.tolerance = tolerance

    def verify(
        self,
        beats_a,
        beats_b,
        bpm_a,
        bpm_b,
        cue_a,
        cue_b,
        tempo_ratio
    ):

        beats_a = np.asarray(beats_a, dtype=np.float64)
        beats_b = np.asarray(beats_b, dtype=np.float64)

        if len(beats_a) < 2 or len(beats_b) < 2:
            raise ValueError("Not enough beat data.")

        if bpm_a <= 0 or bpm_b <= 0 or tempo_ratio <= 0:
            raise ValueError("Invalid BPM or tempo ratio.")

        if not np.all(np.isfinite(beats_a)):
            raise ValueError("Invalid Track A beat data.")

        if not np.all(np.isfinite(beats_b)):
            raise ValueError("Invalid Track B beat data.")

        # Remove beats before the selected cues
        beats_a = beats_a[beats_a >= cue_a] - cue_a
        beats_b = beats_b[beats_b >= cue_b] - cue_b

        if len(beats_a) < 2 or len(beats_b) < 2:
            raise ValueError("Not enough beats after cue points.")

        # Apply tempo stretch to Track B beat timestamps
        beats_b = beats_b / tempo_ratio

        # Expected beat interval
        expected_interval = 60.0 / bpm_a

        # Measure actual intervals
        interval_a = np.median(np.diff(beats_a))
        interval_b = np.median(np.diff(beats_b))

        error_a = abs(interval_a - expected_interval)
        error_b = abs(interval_b - expected_interval)

        # Compare first beat positions
        phase_difference = abs(beats_a[0] - beats_b[0])

        # Normalize phase difference to one beat
        phase_difference %= expected_interval

        return {
            "expected_interval": expected_interval,
            "interval_a": interval_a,
            "interval_b": interval_b,
            "error_a": error_a,
            "error_b": error_b,
            "phase_difference": phase_difference,
            "tempo_a_valid": error_a <= self.tolerance,
            "tempo_b_valid": error_b <= self.tolerance,
            "phase_aligned": (
                phase_difference <= self.tolerance
                or expected_interval - phase_difference <= self.tolerance
            )
        }
