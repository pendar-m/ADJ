
import numpy as np


class SectionDetector:

    def __init__(
        self,
        energy_threshold=0.18,
        rhythm_threshold=0.15,
        min_section_duration=8.0
    ):
        self.energy_threshold = energy_threshold
        self.rhythm_threshold = rhythm_threshold
        self.min_section_duration = min_section_duration

    def detect(self, structure):

        segments = structure["segments"]

        if len(segments) < 2:
            raise ValueError("Not enough segments.")

        boundaries = []

        for i in range(1, len(segments)):

            previous = segments[i - 1]
            current = segments[i]

            energy_change = abs(
                current["normalized_energy"]
                - previous["normalized_energy"]
            )

            rhythm_change = abs(
                current["rhythmic_activity"]
                - previous["rhythmic_activity"]
            )

            if (
                energy_change >= self.energy_threshold
                or rhythm_change >= self.rhythm_threshold
            ):

                boundary_time = current["start"]

                if (
                    not boundaries
                    or boundary_time - boundaries[-1]
                    >= self.min_section_duration
                ):
                    boundaries.append(boundary_time)

        return {
            "boundaries": boundaries,
            "count": len(boundaries)
        }
