
import numpy as np


class PhraseDetector:

    def __init__(self, beats_per_phrase=16):
        if beats_per_phrase <= 0:
            raise ValueError("beats_per_phrase must be positive.")

        self.beats_per_phrase = beats_per_phrase

    def detect(self, beat_times, bpm, cue_time=0.0):

        beat_times = np.asarray(beat_times, dtype=np.float64)

        if bpm <= 0:
            raise ValueError("BPM must be positive.")

        if beat_times.ndim != 1 or len(beat_times) < 2:
            raise ValueError("Not enough beat data.")

        if not np.all(np.isfinite(beat_times)):
            raise ValueError("Beat times contain invalid values.")

        if np.any(np.diff(beat_times) <= 0):
            raise ValueError("Beat times must be strictly increasing.")

        # Keep beats after the selected cue
        beat_times = beat_times[beat_times >= cue_time]

        if len(beat_times) < self.beats_per_phrase:
            raise ValueError("Not enough beats for one phrase.")

        phrase_boundaries = []

        for index in range(
            self.beats_per_phrase,
            len(beat_times),
            self.beats_per_phrase
        ):
            phrase_boundaries.append(
                float(beat_times[index])
            )

        return {
            "beats_per_phrase": self.beats_per_phrase,
            "phrase_duration": round(
                self.beats_per_phrase * 60.0 / bpm, 3
            ),
            "boundaries": phrase_boundaries,
            "count": len(phrase_boundaries)
        }
