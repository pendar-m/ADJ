import numpy as np
from .energy_analyzer import EnergyAnalyzer


class TransitionCandidateSelector:

    def __init__(
        self,
        top_k=10,
        energy_weight=0.35,
        trend_weight=0.40,
        section_weight=0.25,
        trend_window=8.0,
        trend_threshold=0.08
    ):
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        weights = [
            energy_weight,
            trend_weight,
            section_weight
        ]

        if any(w < 0 for w in weights):
            raise ValueError("Weights cannot be negative.")

        if not np.isclose(sum(weights), 1.0):
            raise ValueError("Weights must add up to 1.0.")

        self.top_k = top_k
        self.energy_weight = energy_weight
        self.trend_weight = trend_weight
        self.section_weight = section_weight

        self.energy_analyzer = EnergyAnalyzer(
            window_seconds=trend_window,
            threshold=trend_threshold
        )

    @staticmethod
    def _get_energy(structure, timestamp):
        for segment in structure["segments"]:
            if segment["start"] <= timestamp < segment["end"]:
                return float(segment["normalized_energy"])

        return None

    @staticmethod
    def _get_section_distance(timestamp, boundaries):
        if not boundaries:
            return None

        return min(abs(timestamp - b) for b in boundaries)

    def _get_energy_trend(self, structure, timestamp):
        return self.energy_analyzer.analyze(
            structure,
            timestamp
        )

    @staticmethod
    def _trend_penalty(trend_a, trend_b):
        penalties = {
            ("FALLING", "RISING"): 0.00,
            ("STABLE", "RISING"): 0.10,
            ("FALLING", "STABLE"): 0.20,
            ("STABLE", "STABLE"): 0.25,
            ("RISING", "RISING"): 0.35,
            ("FALLING", "FALLING"): 0.65,
            ("RISING", "STABLE"): 0.70,
            ("STABLE", "FALLING"): 0.70,
            ("RISING", "FALLING"): 1.00
        }

        return penalties.get((trend_a, trend_b), 0.50)

    @staticmethod
    def _section_penalty(distance_a, distance_b):
        valid = [
            d for d in (distance_a, distance_b)
            if d is not None
        ]

        if not valid:
            return 0.50

        return min(float(np.mean(valid)) / 8.0, 1.0)

    def select(
        self,
        phrases_a,
        phrases_b,
        structure_a,
        structure_b,
        sections_a,
        sections_b,
        cue_a,
        cue_b,
        ratio_b
    ):
        if ratio_b <= 0:
            raise ValueError("ratio_b must be positive.")

        candidates = []

        for time_a in phrases_a["boundaries"]:
            source_time_a = float(time_a)

            if source_time_a < cue_a:
                continue

            energy_a = self._get_energy(
                structure_a,
                source_time_a
            )

            if energy_a is None:
                continue

            trend_a = self._get_energy_trend(
                structure_a,
                source_time_a
            )

            distance_a = self._get_section_distance(
                source_time_a,
                sections_a["boundaries"]
            )

            for time_b in phrases_b["boundaries"]:
                source_time_b = float(time_b)

                if source_time_b < cue_b:
                    continue

                energy_b = self._get_energy(
                    structure_b,
                    source_time_b
                )

                if energy_b is None:
                    continue

                trend_b = self._get_energy_trend(
                    structure_b,
                    source_time_b
                )

                distance_b = self._get_section_distance(
                    source_time_b,
                    sections_b["boundaries"]
                )

                energy_penalty = abs(energy_a - energy_b)

                trend_penalty = self._trend_penalty(
                    trend_a["trend"],
                    trend_b["trend"]
                )

                section_penalty = self._section_penalty(
                    distance_a,
                    distance_b
                )

                score = (
                    self.energy_weight * energy_penalty
                    + self.trend_weight * trend_penalty
                    + self.section_weight * section_penalty
                )

                candidates.append({
                    "source_time_a": source_time_a,
                    "source_time_b": source_time_b,
                    "processed_time_b": source_time_b / ratio_b,

                    "energy_a": round(energy_a, 3),
                    "energy_b": round(energy_b, 3),

                    "trend_a": trend_a["trend"],
                    "trend_b": trend_b["trend"],

                    "section_distance_a": distance_a,
                    "section_distance_b": distance_b,

                    "energy_penalty": round(
                        energy_penalty, 3
                    ),
                    "trend_penalty": round(
                        trend_penalty, 3
                    ),
                    "section_penalty": round(
                        section_penalty, 3
                    ),

                    "score": round(score, 4)
                })

        candidates.sort(key=lambda item: item["score"])

        return {
            "best_candidate": (
                candidates[0] if candidates else None
            ),
            "candidates": candidates[:self.top_k],
            "total_combinations": len(candidates)
        }