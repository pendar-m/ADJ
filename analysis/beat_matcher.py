
class BeatMatcher:

    def __init__(self, max_tempo_change=0.10):
        self.max_tempo_change = max_tempo_change

    def calculate_tempo_ratio(self, source_bpm, target_bpm):

        if source_bpm <= 0 or target_bpm <= 0:
            raise ValueError("BPM must be greater than zero.")

        ratio = target_bpm / source_bpm

        return ratio

    def calculate_tempo_change(self, source_bpm, target_bpm):

        ratio = self.calculate_tempo_ratio(
            source_bpm,
            target_bpm
        )

        tempo_change = (ratio - 1) * 100

        return round(tempo_change, 2)

    def can_match(self, source_bpm, target_bpm):

        ratio = self.calculate_tempo_ratio(
            source_bpm,
            target_bpm
        )

        return (
            1 - self.max_tempo_change
            <= ratio
            <= 1 + self.max_tempo_change
        )

    def analyze(self, source_bpm, target_bpm):

        ratio = self.calculate_tempo_ratio(
            source_bpm,
            target_bpm
        )

        tempo_change = self.calculate_tempo_change(
            source_bpm,
            target_bpm
        )

        return {
            "source_bpm": source_bpm,
            "target_bpm": target_bpm,
            "tempo_ratio": round(ratio, 4),
            "tempo_change_percent": tempo_change,
            "can_match": self.can_match(
                source_bpm,
                target_bpm
            )
        }
