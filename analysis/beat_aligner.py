import numpy as np


class BeatAligner:

    def __init__(self, beats_to_skip=0, beats_per_bar=4):
        if beats_to_skip < 0:
            raise ValueError("beats_to_skip cannot be negative.")

        if beats_per_bar <= 0:
            raise ValueError("beats_per_bar must be greater than zero.")

        self.beats_to_skip = beats_to_skip
        self.beats_per_bar = beats_per_bar
    @staticmethod
    def _validate_beats(beat_times):
        beat_times = np.asarray(beat_times, dtype=np.float64)

        if beat_times.ndim != 1 or len(beat_times) == 0:
            raise ValueError("No valid beat times found.")

        if not np.all(np.isfinite(beat_times)):
            raise ValueError("Beat times contain invalid values.")

        if np.any(beat_times < 0):
            raise ValueError("Beat times cannot be negative.")

        if np.any(np.diff(beat_times) <= 0):
            raise ValueError("Beat times must be strictly increasing.")

        return beat_times

    def get_cue_time(self, beat_times):
        beat_times = self._validate_beats(beat_times)

        if self.beats_to_skip >= len(beat_times):
            raise ValueError("Not enough beats available.")

        return float(beat_times[self.beats_to_skip])

    def get_downbeat_aligned_cue(self, beat_times):
        beat_times = self._validate_beats(beat_times)

        target_index = (
            self.beats_to_skip // self.beats_per_bar
        ) * self.beats_per_bar

        if target_index >= len(beat_times):
            raise ValueError("Not enough beats for the requested bar.")

        return float(beat_times[target_index])

    def align(self, beat_times_a, beat_times_b):
        cue_a = self.get_downbeat_aligned_cue(beat_times_a)
        cue_b = self.get_downbeat_aligned_cue(beat_times_b)

        return {
            "cue_a": cue_a,
            "cue_b": cue_b,
            "offset_difference": round(cue_b - cue_a, 4)
        }

    @staticmethod
    def trim_audio(audio_data, sample_rate, cue_time):
        audio_data = np.asarray(audio_data)

        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive.")

        if not np.isfinite(cue_time) or cue_time < 0:
            raise ValueError("Cue time must be a valid non-negative value.")

        sample_index = int(round(cue_time * sample_rate))

        if audio_data.ndim == 1:
            if sample_index >= audio_data.shape[0]:
                raise ValueError("Cue time exceeds audio duration.")

            return np.ascontiguousarray(audio_data[sample_index:])

        if audio_data.ndim == 2:
            # Standardize on (channels, samples).
            if audio_data.shape[0] in (1, 2):
                if sample_index >= audio_data.shape[1]:
                    raise ValueError("Cue time exceeds audio duration.")

                return np.ascontiguousarray(
                    audio_data[:, sample_index:]
                )

            if audio_data.shape[1] in (1, 2):
                if sample_index >= audio_data.shape[0]:
                    raise ValueError("Cue time exceeds audio duration.")

                return np.ascontiguousarray(
                    audio_data[sample_index:, :]
                )

        raise ValueError("Unsupported audio shape.")