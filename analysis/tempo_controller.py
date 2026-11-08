class TempoController:
    def __init__(self, start_bpm):
        if start_bpm <= 0:
            raise ValueError("BPM must be greater than zero.")

        self.original_bpm = float(start_bpm)
        self.current_bpm = float(start_bpm)
        self.target_bpm = float(start_bpm)

        self.start_bpm = float(start_bpm)
        self.duration = 0.0
        self.elapsed = 0.0
        self.is_recovering = False

    def set_tempo(self, bpm):
        if bpm <= 0:
            raise ValueError("BPM must be greater than zero.")

        self.current_bpm = float(bpm)
        self.target_bpm = float(bpm)
        self.start_bpm = float(bpm)

        self.is_recovering = False

    def recover(self, duration=10.0):
        if duration <= 0:
            raise ValueError("Duration must be greater than zero.")

        self.start_bpm = self.current_bpm
        self.target_bpm = self.original_bpm

        self.duration = float(duration)
        self.elapsed = 0.0
        self.is_recovering = True

    def update(self, delta_time):
        if not self.is_recovering:
            return self.current_bpm

        if delta_time <= 0:
            return self.current_bpm

        self.elapsed += delta_time

        progress = min(self.elapsed / self.duration, 1.0)

        # Smoothstep interpolation
        smooth_progress = (
            3 * progress ** 2
            - 2 * progress ** 3
        )

        self.current_bpm = (
            self.start_bpm
            + (self.target_bpm - self.start_bpm)
            * smooth_progress
        )

        if progress >= 1.0:
            self.current_bpm = self.target_bpm
            self.is_recovering = False

        return self.current_bpm

    def get_tempo_ratio(self):
        return self.current_bpm / self.original_bpm