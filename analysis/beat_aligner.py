import numpy as np


class BeatAligner:

    def __init__(self, beats_to_skip=0):
        self.beats_to_skip = beats_to_skip

    def get_cue_time(self, beat_times):

        beat_times = np.asarray(beat_times, dtype=np.float64)

        if beat_times.ndim != 1 or len(beat_times) == 0:
            raise ValueError("No valid beat times found.")

        if self.beats_to_skip >= len(beat_times):
            raise ValueError("Not enough beats available.")

        return float(beat_times[self.beats_to_skip])

    def align(self, beat_times_a, beat_times_b):

        cue_a = self.get_cue_time(beat_times_a)
        cue_b = self.get_cue_time(beat_times_b)

        return {
            "cue_a": cue_a,
            "cue_b": cue_b,
            "offset_difference": round(cue_b - cue_a, 4)
        }

    @staticmethod
    def trim_audio(audio_data, sample_rate, cue_time):

        sample_index = int(round(cue_time * sample_rate))

        if audio_data.ndim == 1:
            return np.ascontiguousarray(audio_data[sample_index:])

        elif audio_data.ndim == 2:
            return np.ascontiguousarray(audio_data[:, sample_index:])

        raise ValueError("Unsupported audio shape.")